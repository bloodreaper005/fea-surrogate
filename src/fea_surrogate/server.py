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
from .versions import FEATURE_COLUMNS, LAYOUTS, PROVENANCE
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
                "feature_version": s.version,
                "feature_columns": FEATURE_COLUMNS[s.version],
                "feature_layout": LAYOUTS[s.version],
                "provenance": PROVENANCE[s.version],
                "known_versions": PROVENANCE,
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
        family: str = "",
        fixed_json: str = "",
        loaded_json: str = "",
        load_direction_json: str = "",
        characteristic_length_mm: float = 0.0,
    ) -> str:
        """Screen a STEP file and return one SurrogateScreening/v1 document.

        A screening estimate, never evidence: it cannot complete, block or
        confirm a design. Pass `characteristic_length_mm` when the file's units
        are not millimetres; 0 uses the bounding-box diagonal as found.

        `family` selects the calibration band, and supplies the boundary
        conditions that family was trained under. Naming it earns the band that
        family measured; leaving it empty earns the widest band any family
        measured, because a part whose group is unknown has not been shown to
        be covered at the stated rate by anything narrower.

        `fixed_json` and `loaded_json` are geometric selectors and win over a
        family preset, for example {"axis": "x", "at": "min"},
        {"axis": "z", "between": [0.1, 0.9]} or {"radial": "inner"}.
        `load_direction_json` is a vector, for example [0, 0, -1].
        """
        return _tool_call(
            lambda: screen_step(
                get(),
                step_path,
                force_n=force_n,
                family=family or None,
                fixed=json.loads(fixed_json) if fixed_json else None,
                loaded=json.loads(loaded_json) if loaded_json else None,
                load_direction=(
                    json.loads(load_direction_json) if load_direction_json else None
                ),
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
