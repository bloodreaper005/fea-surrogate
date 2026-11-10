"""MCP server exposing the surrogate to a coding agent.

Six tools. `surrogate_status` says whether a model is loaded and what it claims
about itself. `surrogate_inspect_cad` reads a CAD file and lists its faces, so a
load case can name them the way a pre-processor lets you click them.
`surrogate_screen` takes that load case and returns one `SurrogateScreening/v1`
document; `surrogate_screen_step` is the older, looser entry point, kept so
anything already wired to it keeps working.

`surrogate_solve` is the other kind of thing entirely. It runs the solver the
training labels came from (4-node tetrahedra) on the part -- about 4 s end to
end on an 11k-node bracket, against about 1 s to screen it, most of which is
meshing -- and draws the field it computes. That is not a screening estimate
and is deliberately not dressed as one: its own schema, its own attestation,
its own stamp on the image, and nothing stored in the design session. Its job
is the coverage crosscheck; 4-node elements are stiff in bending, so it is not
the analysis to read deformation or design stress from.

`structural_solve` is that analysis: the terminal command's core, with 10-node
tetrahedra, several loads, pressures, gravity and a real material, drawn the way
Ansys Mechanical draws Equivalent Stress and Total Deformation.

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
from .features import mesh_step
from .render import render_faces
from .load_case import parse as parse_load_case, propose
from .render_solved import render_solved
from .solve_cad import compare_to_screening, estimate_seconds, solve_step
from .neighbours import NeighbourIndex
from .predict import Surrogate, screen_step
from .versions import FEATURE_COLUMNS, LAYOUTS, PROVENANCE
from .screening import ScreeningContractError, validate_document


WEIGHTS_ENV = "FEA_SURROGATE_WEIGHTS"
CALIBRATION_ENV = "FEA_SURROGATE_CALIBRATION"
#: Optional. Absent, screening still works and simply says nothing about what
#: the part resembles.
NEIGHBOURS_ENV = "FEA_SURROGATE_NEIGHBOURS"
#: Where solved contours and the crosscheck log are written.
OUTPUT_ENV = "FEA_SURROGATE_OUTPUT_DIR"


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


def _writable_dir(folder: Path) -> bool:
    """True if files can be created in `folder` (made if missing)."""
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    return os.access(folder, os.W_OK | os.X_OK)


def _output_root(path: str) -> Path:
    """Where a solve writes: FEA_SURROGATE_OUTPUT_DIR if set, else beside the
    STEP as before -- unless that folder is read-only (a checkout, a shared
    drive), then ./fea_results like the terminal command, then the temp folder.
    A read-only STEP folder used to fail the solve with PermissionError."""
    if os.environ.get(OUTPUT_ENV):
        return Path(os.environ[OUTPUT_ENV])
    import tempfile

    for candidate in (Path(path).resolve().parent, Path.cwd() / "fea_results",
                      Path(tempfile.gettempdir()) / "fea_results"):
        if _writable_dir(candidate):
            return candidate
    return Path(path).resolve().parent      # let the write say why it failed


def _log_crosscheck(out_dir: Path, path: str, crosscheck: dict[str, Any]) -> None:
    """Append one line of real-world coverage evidence.

    A synthetic sweep cannot produce this, however many samples it runs: these
    are parts someone chose to look at.
    """
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"part": str(path), **crosscheck}, sort_keys=True)
        with open(out_dir / "crosschecks.jsonl", "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass        # evidence is worth having, not worth failing a solve over


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
            out: dict[str, Any] = {
                "schema_version": "FeaSurrogateCad/v1",
                "path": path,
                **solid.as_dict(),
                "summary": list(summarise(solid)),
                **propose(solid),
            }
            # A numbered picture, because nobody holds a centroid in their head.
            # Meshing costs a second or two and is the difference between the
            # interview being answerable and being guessed at -- but a failure
            # here must not cost the caller the face table they asked for.
            try:
                directory = (
                    Path(os.environ[OUTPUT_ENV])
                    if os.environ.get(OUTPUT_ENV)
                    else Path(path).resolve().parent
                )
                image = render_faces(
                    solid,
                    mesh_step(path),
                    out_dir=directory,
                    stem=f"{Path(path).stem}_faces",
                )
                out["output_dir"] = str(directory)
                out["faces_image_relative_path"] = image.name
            except Exception as exc:
                out["faces_image_error"] = f"{type(exc).__name__}: {exc}"
            return out

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
            # build_document validated the document before the assumptions
            # were appended. Validate again, so what leaves the server is what
            # the consumer will accept rather than what it was a moment ago.
            validate_document(document)
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


    @mcp.tool()
    def surrogate_solve(
        path: str,
        load_case_json: str,
        family: str = "",
        allow_large: bool = False,
    ) -> str:
        """Run the 4-node label solver on the part: the screening crosscheck.

        This is not a screening estimate and not the design analysis. It runs
        `generate/solver.py`, the linear-elastic solver every training label
        came from, on this part under the stated load case, at the mesh the
        labels were computed on -- about 4 s end to end on an 11,000-node
        bracket, against about 1 s to screen it once the model is loaded --
        and draws the field it computes.

        4-node tetrahedra are stiff in bending: on the sample bracket's shelf
        this solve reads the tip deflection about 38% low and the bending
        stress about half of beam theory. So it is the right thing to compare a
        screening interval with (same solver, same mesh rule), and the wrong
        thing to read deformation or a design stress from: for those use
        `structural_solve` (10-node tetrahedra, Ansys-style results).

        The answer is a computed analysis of the idealisation -- linear
        elastic, small displacement, first-order tetrahedra, one uniform
        traction, no contact and no body loads -- not a strength
        certification. Use `critical_distance_mpa`, not the peak: an elastic
        peak at a re-entrant corner is singular and rises without bound as the
        mesh is refined, so it states the mesh rather than the part.

        Where a model is loaded, the same part is also screened and the result
        says whether the screening interval contained this solver's value.
        """

        def build() -> dict[str, Any]:
            out_dir = _output_root(path)
            solid = inspect(path)
            case = parse_load_case(json.loads(load_case_json), solid)
            solved = solve_step(path, case, allow_large=allow_large)
            render = render_solved(
                solved, out_dir=out_dir, stem=f"{Path(path).stem}_solved"
            )

            # Screen the same part, so every solve is a real coverage datapoint
            # -- measured on something someone cared about rather than on
            # geometry this project generated. Never fatal: a missing model must
            # not cost you the solve you just waited for.
            document = None
            try:
                document = screen_step(
                    get(),
                    path,
                    force_n=case.load.magnitude_n,
                    family=family or None,
                    fixed=dict(case.fixed),
                    loaded=dict(case.loaded),
                    load_direction=list(case.load.direction),
                    poisson_ratio=case.poisson_ratio,
                )
            except Exception:
                document = None

            crosscheck = compare_to_screening(solved, document)
            if crosscheck.get("available"):
                _log_crosscheck(out_dir, path, crosscheck)

            return {
                **solved.as_dict(),
                **render.as_dict(),
                # The image and value names are relative to this, the way the
                # contract's own `*_relative_path` fields are.
                "output_dir": str(out_dir),
                "assumptions": case.assumptions(),
                "crosscheck": crosscheck,
            }

        return _tool_call(build)

    @mcp.tool()
    def structural_solve(
        path: str,
        case_json: str,
        family: str = "",
        refinement: str = "medium",
        element_order: int = 2,
        allow_large: bool = False,
        crosscheck: bool = True,
    ) -> str:
        """Run a static structural analysis and draw it the way Ansys does.

        `case_json` is a `StaticStructuralCase/v1` document: fixed supports,
        any number of forces and pressures on named faces, a material preset
        or a custom Young's modulus and Poisson ratio, and optional gravity.
        The single-load shape the screening tools take is accepted too, as long
        as it names a Young's modulus or a preset -- deformation needs one, and
        it is never assumed.

        The solve uses 10-node quadratic tetrahedra (`element_order=1` gives
        the 4-node elements the surrogate's labels came from, which are stiff
        in bending). `refinement` is coarse, medium or fine; a `mesh` block in
        the document wins over both arguments.

        This is the same core function the `fea-structural` terminal command
        calls, so the two cannot answer the same case differently. The result
        is `FeaStaticStructural/v1`: averaged nodal von Mises stress with a
        singularity flag, total deformation, reactions against the applied
        load, and images of both fields. The surrogate is consulted only when
        the case is one it was trained on; its document is nested, untouched,
        under `screening`, never merged into the solved numbers. With
        `crosscheck` (the default here) a screened case is also solved with
        the 4-node label solver, to record whether the interval held; it adds
        about 4 s on the sample bracket.
        """

        def build() -> dict[str, Any]:
            import dataclasses
            from datetime import datetime

            from .structural_case import MeshSettings, parse_case
            from .structural_run import run_static_structural

            document = json.loads(case_json)
            case = parse_case(document)
            if not (isinstance(document, dict) and "mesh" in document):
                case = dataclasses.replace(
                    case,
                    mesh=MeshSettings(
                        element_order=int(element_order), refinement=refinement
                    ),
                )

            root = _output_root(path)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            out_dir = root / f"{Path(path).stem}-structural-{stamp}"
            suffix = 1
            while out_dir.exists():
                suffix += 1
                out_dir = root / f"{Path(path).stem}-structural-{stamp}-{suffix}"

            run = run_static_structural(
                path,
                case,
                out_dir=out_dir,
                family=family or None,
                allow_large=allow_large,
                crosscheck=bool(crosscheck),
            )
            return {**run.as_document(), "output_dir": str(out_dir)}

        return _tool_call(build)

    return mcp


def main() -> None:
    create_mcp().run()


if __name__ == "__main__":
    main()
