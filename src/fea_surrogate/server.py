"""MCP server exposing the surrogate to a coding agent.

Four tools. `surrogate_status` says whether a model is loaded and what it claims
about itself. `surrogate_inspect_cad` reads a CAD file and lists its faces, so a
load case can name them the way a pre-processor lets you click them.
`surrogate_screen` takes that load case and returns one `SurrogateScreening/v1`
document; `surrogate_screen_step` is the older, looser entry point, kept so
anything already wired to it keeps working.

Everything else belongs on the consumer side, which already decides whether a
document is admissible.

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

from .cad_intake import inspect, summarise
from .load_case import parse as parse_load_case, propose
from .neighbours import NeighbourIndex
from .predict import Surrogate, screen_step
from .versions import FEATURE_COLUMNS, LAYOUTS, PROVENANCE
from .screening import ScreeningContractError


WEIGHTS_ENV = "FEA_SURROGATE_WEIGHTS"
CALIBRATION_ENV = "FEA_SURROGATE_CALIBRATION"
#: Optional. Absent, screening still works and simply says nothing about what
#: the part resembles.
NEIGHBOURS_ENV = "FEA_SURROGATE_NEIGHBOURS"


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
    index: NeighbourIndex | None = None
    index_tried = False
    lock = Lock()

    def neighbours() -> NeighbourIndex | None:
        """The retrieval index, if one is configured. Never fatal."""
        nonlocal index, index_tried
        if index_tried:
            return index
        index_tried = True
        path = os.environ.get(NEIGHBOURS_ENV, "")
        if path:
            try:
                index = NeighbourIndex.load(path)
            except Exception:
                index = None
        return index

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
    def surrogate_inspect_cad(path: str) -> str:
        """List a CAD file's faces so a load case can name them.

        Reads BREP, STEP and IGES. Parasolid and native CAD formats need a
        licensed kernel OpenCASCADE does not have, and are refused by name.

        Returns the faces with area, centroid, normal and whether each is
        planar; a proposed load case; and what still has to be answered. The
        proposal is a starting point, not an answer -- holding the wrong face
        screens a different question and the result looks perfectly healthy.
        """

        def build() -> dict[str, Any]:
            solid = inspect(path)
            return {
                "schema_version": "FeaSurrogateCad/v1",
                "path": path,
                **solid.as_dict(),
                "summary": list(summarise(solid)),
                **propose(solid),
            }

        return _tool_call(build)

    @mcp.tool()
    def surrogate_screen(path: str, load_case_json: str, family: str = "") -> str:
        """Screen a part under a stated load case.

        `load_case_json` names held faces, the load and the material, in the
        shape `surrogate_inspect_cad` proposes. A load case with no held faces
        is refused rather than defaulted: an unconstrained body has rigid-body
        freedom and no stress solution.

        A screening estimate, never evidence. It cannot complete, block or
        confirm a design.
        """

        def build() -> dict[str, Any]:
            solid = inspect(path)
            case = parse_load_case(json.loads(load_case_json), solid)
            document = screen_step(
                get(),
                path,
                force_n=case.load.magnitude_n,
                family=family or None,
                fixed=dict(case.fixed),
                loaded=dict(case.loaded),
                load_direction=list(case.load.direction),
                poisson_ratio=case.poisson_ratio,
                neighbours=neighbours(),
            )
            document["assumptions"] = (
                list(document["assumptions"]) + case.assumptions()
            )
            return document

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
                neighbours=neighbours(),
            )
        )

    return mcp


def main() -> None:
    create_mcp().run()


if __name__ == "__main__":
    main()
