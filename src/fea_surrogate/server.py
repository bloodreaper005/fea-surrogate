"""MCP server exposing the surrogate to a coding agent.

Two tools, deliberately. `surrogate_status` says whether a model is loaded and
what it claims about itself; `surrogate_screen_step` returns one
`SurrogateScreening/v1` document. Everything else belongs on the consumer side,
which already decides whether a document is admissible.

The model loads lazily so the server starts, and reports honestly, with no
weights present.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from .predict import Surrogate, screen_step
from .screening import ScreeningContractError


WEIGHTS_ENV = "FEA_SURROGATE_WEIGHTS"
CALIBRATION_ENV = "FEA_SURROGATE_CALIBRATION"


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                      allow_nan=False)


def _tool_call(call: Callable[[], Any]) -> str:
    try:
        return _json(call())
    except Exception as exc:
        raise ToolError(
            _json(
                {
                    "schema_version": "FeaSurrogateToolError/v1",
                    "status": "blocked",
                    "code": type(exc).__name__,
                    "message": str(exc),
                }
            )
        ) from None


def create_mcp(*, surrogate: Surrogate | None = None) -> FastMCP:
    mcp = FastMCP("fea-surrogate")
    loaded = surrogate
    lock = Lock()

    def paths() -> tuple[Path, Path]:
        weights = os.environ.get(WEIGHTS_ENV, "")
        calibration = os.environ.get(CALIBRATION_ENV, "")
        if not weights or not calibration:
            raise ScreeningContractError(
                f"set {WEIGHTS_ENV} and {CALIBRATION_ENV} to a trained model"
            )
        return Path(weights), Path(calibration)

    def get() -> Surrogate:
        nonlocal loaded
        if loaded is not None:
            return loaded
        with lock:
            if loaded is None:
                weights, calibration = paths()
                loaded = Surrogate.load(weights, calibration)
        return loaded

    @mcp.tool()
    def surrogate_status() -> str:
        """Report whether a trained surrogate is loaded and what it claims."""

        def build() -> dict[str, Any]:
            try:
                s = get()
            except Exception as exc:
                return {
                    "schema_version": "FeaSurrogateStatus/v1",
                    "status": "setup_required",
                    "message": str(exc),
                }
            return {
                "schema_version": "FeaSurrogateStatus/v1",
                "status": "ok",
                "model_sha256": s.weights_sha256,
                "coverage": s.calibration.coverage,
                "band_factor": round(math.exp(s.calibration.log_half_width), 4),
                "calibration_set_size": s.calibration.set_size,
                "domain": s.domain,
                "limitations": s.limitations,
                "gates_completion": False,
            }

        return _tool_call(build)

    @mcp.tool()
    def surrogate_screen_step(
        step_path: str,
        force_n: float,
        characteristic_length_mm: float = 0.0,
    ) -> str:
        """Screen a STEP file and return one SurrogateScreening/v1 document.

        A screening estimate, never evidence: it cannot complete, block or
        confirm a design. Pass `characteristic_length_mm` when the file's units
        are not millimetres; 0 uses the bounding-box diagonal as found.
        """
        return _tool_call(
            lambda: screen_step(
                get(),
                step_path,
                force_n=force_n,
                characteristic_length_mm=(
                    characteristic_length_mm if characteristic_length_mm > 0 else None
                ),
            )
        )

    return mcp


def main() -> None:
    create_mcp().run()


if __name__ == "__main__":
    main()
