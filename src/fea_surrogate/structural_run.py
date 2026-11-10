"""One static structural run: geometry, mesh, screening gate, solve, pictures, files.

The terminal (`fea-structural`) and the MCP tool both call
`run_static_structural`, so the two paths cannot disagree about what was solved,
what was screened or what was written. Everything a person reads afterwards --
the results table, results.json, the images -- comes out of the `StructuralRun`
this returns.

Order, and why:

    inspect     refuse an assembly before spending a mesh on it
    mesh        (or reuse the one the interview showed, so face numbers match)
    case.json   written before anything can fail, so a failed run is replayable
    screening   only through `screenability.check`; the surrogate is direction
                and Poisson blind and not scale-invariant, so outside its
                trained load cases it must not speak at all
    solve       the quadratic-tetrahedron solver: this is the answer
    images      faces, setup, equivalent stress, total deformation, board
    files       results.vtu, results.json, summary.txt

A screening estimate and a solved result never share a number: the screening
document is kept untouched and nested, and its crosscheck compares it with the
P1 label solver it was trained against -- never with the P2 numbers.

Import-light by contract (stdlib + numpy): the mesher, solver, renderer and
model are imported inside the functions that use them.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .structural_case import FixedSupport, MeshSettings, StaticStructuralCase


SCHEMA = "FeaStaticStructural/v1"

WEIGHTS_ENV = "FEA_SURROGATE_WEIGHTS"
CALIBRATION_ENV = "FEA_SURROGATE_CALIBRATION"
NEIGHBOURS_ENV = "FEA_SURROGATE_NEIGHBOURS"

#: <repo>/models, used only when it exists (a source checkout with weights).
_REPO_MODELS = Path(__file__).resolve().parents[2] / "models"

#: Ansys Mechanical's default allowable change for convergence.
CONVERGENCE_ALLOWABLE_PCT = 20.0
#: Element-size factor of the convergence re-mesh.
CONVERGENCE_REFINE = 0.67

#: The analysis needs gmsh, scipy and matplotlib: the `solve` extra, no torch.
INSTALL_HINT = 'uv pip install -e ".[solve]"'
#: Only the surrogate (screening) needs torch: the `inference` extra.
SURROGATE_INSTALL_HINT = 'uv pip install -e ".[inference]"'

Progress = Callable[[str, str], None]


class StructuralRunError(RuntimeError):
    """The run could not be completed; the message says what to do next."""


class RunRefused(StructuralRunError):
    """The part or the request is outside what this analysis does
    (an assembly, a solve too large without permission)."""


# --------------------------------------------------------------------------
# Formatting shared with the CLI
# --------------------------------------------------------------------------


def fmt(value: float | None, sig: int = 4) -> str:
    """Human number: 200000, 2129, 1.25, 0.0003125 -- no '2.129e+03' in a table."""
    if value is None:
        return "-"
    value = float(value)
    if not math.isfinite(value):
        return str(value)
    if value == 0:
        return "0"
    magnitude = abs(value)
    if 1e-4 <= magnitude < 1e9:
        exponent = math.floor(math.log10(magnitude))
        decimals = max(0, sig - 1 - exponent)
        text = f"{value:.{decimals}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return "0" if text in ("-0", "") else text
    return f"{value:.{sig}g}"


def fmt_vec(vector: Sequence[float] | None, sig: int = 4) -> str:
    """Components below 1e-9 of the largest print as 0: a reaction of
    (4e-09, 2e-09, 1000) N is round-off, not a sideways force."""
    if vector is None:
        return "-"
    values = [float(c) for c in vector]
    largest = max((abs(c) for c in values if math.isfinite(c)), default=0.0)
    cleaned = [0.0 if math.isfinite(c) and abs(c) <= 1e-9 * largest else c for c in values]
    return "(" + ", ".join(fmt(c, sig) for c in cleaned) + ")"


def fmt_faces(tags: Sequence[int]) -> str:
    tags = [int(t) for t in tags]
    if not tags:
        return "interior"
    return ("face " if len(tags) == 1 else "faces ") + ", ".join(map(str, tags))


def axis_label(vector: Sequence[float], tol: float = 0.999) -> str:
    """'+Z' for a vector along an axis, '' otherwise."""
    for index, name in enumerate("XYZ"):
        c = float(vector[index])
        if abs(c) >= tol:
            return ("+" if c > 0 else "-") + name
    return ""


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ScreeningOutcome:
    """What the surrogate said, or why it said nothing.

    `document` is the SurrogateScreening/v1 document exactly as `screen_step`
    produced it -- nested, never merged into the solved results, so a reader
    can never take an interval for a computed value.
    """

    status: str                     # screened | not_screenable | unavailable | skipped | error
    reasons: tuple[str, ...]
    family: str | None
    document: Mapping[str, Any] | None = None
    crosscheck: Mapping[str, Any] | None = None
    seconds: float = 0.0
    #: Of `seconds`, the time spent importing torch and loading the weights:
    #: paid once per process, so it is reported apart from the screen itself.
    load_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "family": self.family,
            "document": self.document,
            "crosscheck": self.crosscheck,
        }


@dataclass(frozen=True)
class StructuralRun:
    path: Path
    out_dir: Path
    case: StaticStructuralCase
    mesh: Any                       # structural_mesh.AnalysisMesh
    solution: Any                   # structural.StaticSolution
    screening: ScreeningOutcome
    files: Mapping[str, Path]
    timings: Mapping[str, float]
    convergence: Mapping[str, Any] | None
    #: CAD facts from `cad_intake.inspect` (sha256, bbox, volume...).
    part: Mapping[str, Any] = field(default_factory=dict)
    #: Non-fatal problems (an image that failed to draw). Never hides a result.
    warnings: tuple[str, ...] = ()

    def yield_check(self) -> dict[str, Any] | None:
        """Yield / stress. Linear elastic only: a ratio, not a certification.

        Which stress:

        * the max, when it is not singular;
        * when it is, the max rises without bound under refinement and the max
          one wall thickness away leaves out the section's own stress (10-17 %
          low on a clamped cantilever, measured), so the stress extrapolated
          back to the flagged edge from 1-2 wall thicknesses is used
          (`max_stress_extrapolated`, the hot-spot method; within 2-4 % of beam
          theory at the root on medium meshes) -- or the max away from
          singularities when that is higher, because a design stress below a
          stress the part really carries elsewhere would be unsafe. The
          singular max is reported beside it as the conservative bound;
        * when the mesh is too coarse to extrapolate, the singular max itself
          (conservative and mesh-dependent), with the factor away from the
          singularity beside it as the optimistic end.

        4-node runs get no factor: they read bending stress low by a third to
        two thirds, and a factor computed from that is unsafe by as much.
        """
        material = self.case.material
        if material.tensile_yield_mpa is None:
            return None
        s = self.solution
        yield_mpa = float(material.tensile_yield_mpa)

        def factor(stress: float | None) -> float | None:
            return yield_mpa / float(stress) if stress is not None and float(stress) > 0 else None

        check: dict[str, Any] = {
            "tensile_yield_mpa": yield_mpa,
            "stress_used_mpa": None,
            "basis": "",
            "safety_factor": None,
            "safety_factor_at_singular_max": None,
            "safety_factor_away_from_singularities": None,
        }
        if int(self.mesh.order) == 1:
            check["basis"] = ("not given: 4-node tetrahedra under-read bending stress "
                              "(comparison only); rerun with --order 2")
            return check
        peak = float(s.max_stress.value)
        away = s.max_stress_away
        away_value = float(away.value) if away is not None else None
        extrapolated = getattr(s, "max_stress_extrapolated", None)
        if not s.max_stress.singular:
            used, basis = peak, "max equivalent stress (averaged nodal)"
        elif extrapolated is not None:
            used = float(extrapolated)
            basis = ("hot-spot stress: extrapolated to the flagged edge from 1-2 wall "
                     "thicknesses")
            if away_value is not None and away_value > used:
                used = away_value
                basis = "max away from singularities (above the hot-spot value at the edge)"
            check["safety_factor_at_singular_max"] = factor(peak)
        else:
            used = peak
            basis = ("singular max: the mesh is too coarse to extrapolate to the edge, so "
                     "this is conservative and mesh-dependent")
            check["safety_factor_away_from_singularities"] = factor(away_value)
        check.update(stress_used_mpa=used, basis=basis + ", linear elastic",
                     safety_factor=factor(used))
        return check

    def as_document(self) -> dict[str, Any]:
        """FeaStaticStructural/v1: JSON-safe, finite numbers only."""
        s, m = self.solution, self.mesh
        max_s, min_s, away = s.max_stress, s.min_stress, s.max_stress_away
        order = int(m.order)
        document = {
            "schema_version": SCHEMA,
            "attestation": "computed_analysis",
            "part": dict(self.part),
            "case": self.case.as_document(),
            "mesh": {
                "element": _element_name(order),
                "order": order,
                "nodes": int(m.node_count),
                "elements": int(m.element_count),
                "dof": int(m.dof_count),
                "element_size_mm": float(m.element_size_mm),
                "min_scaled_jacobian": float(m.min_scaled_jacobian),
                "valid_jacobians": bool(float(m.min_scaled_jacobian) > 0.0),
                "element_quality_min": _optional_float(getattr(m, "element_quality_min", None)),
                "element_quality_median": _optional_float(
                    getattr(m, "element_quality_median", None)),
                "straight_sided": bool(m.straight_sided),
            },
            "results": {
                "equivalent_stress": {
                    "units": "MPa",
                    "max": float(max_s.value),
                    "max_location_mm": list(max_s.location_mm),
                    "max_faces": list(max_s.faces),
                    "max_singular": bool(max_s.singular),
                    "max_singular_reason": max_s.reason,
                    "max_away_from_singularities": (
                        float(away.value) if away is not None else None
                    ),
                    "max_extrapolated_to_singularity": (
                        None if getattr(s, "max_stress_extrapolated", None) is None
                        else float(s.max_stress_extrapolated)
                    ),
                    "exclusion_radius_mm": float(s.exclusion_radius_mm),
                    "min": float(min_s.value),
                },
                "total_deformation": {
                    "units": "mm",
                    "max": float(s.max_deformation.value),
                    "max_location_mm": list(s.max_deformation.location_mm),
                    "max_faces": list(s.max_deformation.faces),
                    "min": float(s.min_deformation.value),
                    "auto_scale": float(s.deformation_scale_auto),
                },
                "reactions": [
                    {"label": r.label, "faces": list(r.faces), "force_n": list(r.force_n)}
                    for r in s.reactions
                ],
                "applied_total_n": list(s.applied_total_n),
                "reaction_total_n": list(s.reaction_total_n),
                "equilibrium_error": float(s.equilibrium_error),
                "yield_ratio": self.yield_check(),
            },
            "screening": self.screening.as_dict(),
            "convergence": dict(self.convergence) if self.convergence else None,
            "files": {name: _relative(path, self.out_dir) for name, path in self.files.items()},
            "timings_s": {k: round(float(v), 3) for k, v in self.timings.items()},
            "idealisation": [
                "linear elastic, small displacement",
                "10-node quadratic tetrahedra" if order == 2
                else "4-node linear tetrahedra (much too stiff in bending)",
                "fixed supports clamp all three translations",
                "no contact, no plasticity",
            ],
            "assumptions": self._assumptions(),
            "limitations": [
                "a computed analysis of this idealisation, not a strength certification",
                "stress at a flagged singularity (the edge of a fixed support, a "
                "sharp re-entrant edge) rises with mesh refinement; read the "
                "value away from it",
                "the yield check is linear elastic: no plasticity, fatigue, "
                "buckling or code safety factors",
            ],
            "warnings": list(self.warnings),
        }
        return _jsonable(document)

    def _assumptions(self) -> list[str]:
        out = list(self.case.assumptions())
        if self.part.get("healed"):
            out.append(
                "the file held no solid; its surfaces were sewn into one (a repair "
                "within a tolerance, not an exact read)"
            )
        if bool(self.mesh.straight_sided):
            out.append(
                "curved quadratic elements were invalid on this geometry, so "
                "straight-sided 10-node elements were used"
            )
        return out


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _element_name(order: int) -> str:
    return (
        "10-node quadratic tetrahedron (SOLID187-like)" if order == 2
        else "4-node linear tetrahedron"
    )


def _relative(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


def _jsonable(value: Any) -> Any:
    """numpy scalars/arrays to Python, tuples to lists, non-finite to None."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if hasattr(value, "tolist"):          # numpy array or scalar
        return _jsonable(value.tolist())
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return number if math.isfinite(number) else None


# --------------------------------------------------------------------------
# Building blocks the CLI also uses
# --------------------------------------------------------------------------


def default_out_dir(path: Path, stamp: str) -> Path:
    """./fea_results/<stem>-<stamp>/ -- never next to the STEP, which may be a
    read-only checkout or a shared folder."""
    return Path.cwd() / "fea_results" / f"{Path(path).stem}-{stamp}"


def find_surrogate_files(
    weights: str | Path | None = None,
    calibration: str | Path | None = None,
    neighbours: str | Path | None = None,
) -> tuple[Path | None, Path | None, Path | None]:
    """Flags win, then FEA_SURROGATE_WEIGHTS/_CALIBRATION/_NEIGHBOURS, then
    <repo>/models/{sweep_gnn.pt, calibration.json, neighbours.npz} if present.

    A flag or variable naming a missing file is returned as given, so the run
    can say that file is missing instead of silently falling back to another.
    """

    def pick(flag: Any, env: str, default_name: str) -> Path | None:
        if flag:
            return Path(flag).expanduser()
        if os.environ.get(env):
            return Path(os.environ[env]).expanduser()
        candidate = _REPO_MODELS / default_name
        return candidate if candidate.is_file() else None

    return (
        pick(weights, WEIGHTS_ENV, "sweep_gnn.pt"),
        pick(calibration, CALIBRATION_ENV, "calibration.json"),
        pick(neighbours, NEIGHBOURS_ENV, "neighbours.npz"),
    )


def read_calibration(path: Path | None) -> tuple[Mapping[str, Any] | None, str | None]:
    """(domain, calibration set_sha256) from calibration.json, without torch."""
    if path is None or not Path(path).is_file():
        return None, None
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    domain = raw.get("domain") if isinstance(raw.get("domain"), Mapping) else None
    calibration = raw.get("calibration") if isinstance(raw.get("calibration"), Mapping) else {}
    sha = calibration.get("set_sha256")
    return domain, (str(sha) if sha else None)


def screening_verdict(
    case: StaticStructuralCase,
    mesh,  # noqa: ANN001 - AnalysisMesh
    family: str | None,
    calibration: Path | None = None,
):  # -> screenability.ScreeningVerdict
    """The gate, with the domain and set hash read from the calibration in use."""
    from .screenability import check

    domain, sha = read_calibration(calibration)
    return check(case, mesh, family=family, domain=domain, calibration_set_sha256=sha)


def inspect_part(path: str | Path):  # -> cad_intake.Solid
    """Read the CAD and refuse an assembly before a mesh is spent on it."""
    from .cad_intake import inspect

    solid = inspect(path)
    if solid.solid_count > 1:
        raise RunRefused(
            f"{Path(path).name} holds {solid.solid_count} SEPARATE SOLIDS -- this is "
            "an assembly, not a part. Screening and solving both need one body at "
            "a time: export each part as its own STEP file and run it alone."
        )
    return solid


def mesh_part(path: str | Path, settings: MeshSettings):  # -> AnalysisMesh
    from .structural_mesh import mesh_for_analysis

    return mesh_for_analysis(
        Path(path),
        order=settings.element_order,
        refinement=settings.refinement,
        element_size_mm=settings.element_size_mm,
    )


def estimate_seconds(mesh) -> float | None:  # noqa: ANN001
    try:
        from .structural import estimate_solve_seconds

        return float(estimate_solve_seconds(int(mesh.dof_count)))
    except Exception:
        return None


def large_dof_threshold() -> int:
    try:
        from .structural import LARGE_DOF_THRESHOLD

        return int(LARGE_DOF_THRESHOLD)
    except Exception:
        return 250_000


def render_faces_image(mesh, out_path: Path) -> Path:  # noqa: ANN001
    from .render_structural import render_face_ids

    return Path(render_face_ids(mesh, Path(out_path)))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _part_facts(path: Path, solid, mesh) -> dict[str, Any]:  # noqa: ANN001
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "bbox_mm": {"min": list(solid.bbox_min), "max": list(solid.bbox_max)},
        "diagonal_mm": float(mesh.diagonal_mm),
        "volume_mm3": float(solid.volume),
        "face_count": len(mesh.faces),
        "healed": bool(solid.healed or getattr(mesh, "healed", False)),
    }


def _write_json(path: Path, document: Any) -> Path:
    path.write_text(
        json.dumps(_jsonable(document), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _log_crosscheck(out_dir: Path, path: Path, crosscheck: Mapping[str, Any]) -> Path | None:
    """One line of real-world coverage evidence per screened run (as the MCP
    server does): a synthetic sweep cannot produce parts someone cared about."""
    try:
        target = out_dir / "crosschecks.jsonl"
        line = json.dumps(_jsonable({"part": str(path), **crosscheck}), sort_keys=True)
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return target
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------
# Screening
# --------------------------------------------------------------------------


def _quiet_torch_import() -> None:
    """torch_geometric's import prints a torch.jit FutureWarning that is a
    deprecation notice, not a problem; it would read as an error mid-run."""
    warnings.filterwarnings("ignore", category=FutureWarning, message=r".*torch\.jit")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch")


#: Loaded surrogates by (weights, calibration) file identity, so a long-lived
#: caller (the MCP server) pays the torch import and weight load once.
_SURROGATES: dict[tuple[Any, ...], Any] = {}


def _file_key(path: Path) -> tuple[str, int, int]:
    stat = Path(path).stat()
    return (str(Path(path).resolve()), int(stat.st_mtime_ns), int(stat.st_size))


def _load_surrogate(weights: Path, calibration: Path):  # noqa: ANN202 - predict.Surrogate
    from .predict import Surrogate

    key = (_file_key(weights), _file_key(calibration))
    if key not in _SURROGATES:
        if len(_SURROGATES) >= 2:
            _SURROGATES.clear()
        _SURROGATES[key] = Surrogate.load(weights, calibration)
    return _SURROGATES[key]


#: Shown when the P1 crosscheck was not asked for.
CROSSCHECK_HINT = ("not run: --crosscheck solves this case with the 4-node solver the "
                   "training labels came from (about 4 s on the sample bracket)")


def _screen(
    path: Path,
    case: StaticStructuralCase,
    mesh,  # noqa: ANN001
    solid,  # noqa: ANN001
    out_dir: Path,
    *,
    family: str | None,
    weights: Path | None,
    calibration: Path | None,
    neighbours: Path | None,
    crosscheck: bool = False,
    report: Progress = lambda stage, message: None,
) -> tuple[ScreeningOutcome, Path | None]:
    started = time.perf_counter()
    load_seconds = 0.0

    def done(status: str, reasons: Sequence[str], **extra: Any) -> ScreeningOutcome:
        return ScreeningOutcome(
            status=status, reasons=tuple(reasons), family=family,
            seconds=time.perf_counter() - started, load_seconds=load_seconds, **extra,
        )

    verdict = screening_verdict(case, mesh, family, calibration)
    if not verdict.available:
        return done("not_screenable", verdict.reasons), None

    if weights is None or calibration is None:
        return done("unavailable", [
            "no trained surrogate found: pass --weights and --calibration, or set "
            f"{WEIGHTS_ENV} and {CALIBRATION_ENV}"
        ]), None
    for label, file in (("weights", weights), ("calibration", calibration)):
        if not Path(file).is_file():
            return done("unavailable", [f"surrogate {label} file not found: {file}"]), None

    import importlib.util

    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torch_geometric") is None:
        return done("unavailable", [
            "the surrogate needs torch and torch_geometric; install them with "
            f"{SURROGATE_INSTALL_HINT}"
        ]), None

    try:
        report("screen_detail", f"the {verdict.family!r} training case: loading the surrogate")
        with warnings.catch_warnings():
            _quiet_torch_import()
            loading = time.perf_counter()
            from .predict import screen_step

            surrogate = _load_surrogate(Path(weights), Path(calibration))
            load_seconds = time.perf_counter() - loading
            index = None
            if neighbours is not None and Path(neighbours).is_file():
                try:
                    from .neighbours import NeighbourIndex

                    index = NeighbourIndex.load(neighbours)
                except Exception:       # retrieval is context, never a blocker
                    index = None
            # The gate proved the case's faces ARE the family preset's on this
            # part, so the preset is passed (fixed/loaded None) rather than the
            # same faces as caller selectors: predict would otherwise add the
            # limitation "this one was supplied by the caller and differs from
            # it", which the gate has just shown to be false.
            report("screen_detail", "screening (meshes the part the way the sweep did)")
            document = screen_step(
                surrogate,
                path,
                force_n=float(verdict.force_n),
                family=verdict.family,
                fixed=None,
                loaded=None,
                load_direction=list(verdict.direction),
                poisson_ratio=case.material.poisson_ratio,
                neighbours=index,
            )
    except Exception as exc:
        return done("error", [f"{type(exc).__name__}: {exc}"]), None

    # The crosscheck runs the P1 label solver the surrogate's labels came from,
    # on its own sweep-rule mesh, so the critical-distance values compare like
    # with like. P2 numbers are never fed to compare_to_screening. It costs
    # more than the screen and the solve together on small parts, so it runs
    # only when asked for.
    reasons: list[str] = []
    check: dict[str, Any]
    log: Path | None = None
    if not crosscheck:
        return done("screened", reasons, document=document,
                    crosscheck={"available": False, "skipped": CROSSCHECK_HINT}), None
    try:
        report("screen_detail", "crosscheck: P1 label solver on the same single load")
        with warnings.catch_warnings():
            _quiet_torch_import()
            from .load_case import parse as parse_load_case
            from .solve_cad import compare_to_screening, solve_step

            legacy = parse_load_case(verdict.legacy_document, solid)
            solved = solve_step(path, legacy, allow_large=True)
        check = dict(compare_to_screening(solved, document))
        check["p1_critical_distance_mpa"] = float(solved.critical_distance_mpa)
        check["inside"] = check.get("inside")
        if check.get("available"):
            log = _log_crosscheck(out_dir, path, check)
    except Exception as exc:
        check = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
        reasons.append(f"crosscheck failed: {type(exc).__name__}: {exc}")
    return done("screened", reasons, document=document, crosscheck=check), log


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


def run_static_structural(
    path: str | Path,
    case: StaticStructuralCase,
    *,
    out_dir: str | Path,
    family: str | None = None,
    screen: bool = True,
    weights: str | Path | None = None,
    calibration: str | Path | None = None,
    neighbours: str | Path | None = None,
    render: bool = True,
    export_vtu: bool = True,
    converge: bool = False,
    allow_large: bool = False,
    crosscheck: bool = False,
    progress: Progress | None = None,
    mesh=None,  # noqa: ANN001 - AnalysisMesh to reuse (the one the face table showed)
) -> StructuralRun:
    """Solve `case` on the part at `path` and write every output to `out_dir`.

    `crosscheck` also solves a screened case with the 4-node label solver, to
    say whether the screening interval held (about 4 s on the sample bracket,
    more than screening and solving together, so it is opt-in).

    Raises CaseError for a case that does not fit the part (unknown face),
    RunRefused for an assembly or an over-large solve without `allow_large`,
    and the solver's own errors when the problem cannot be solved.
    """
    started = time.perf_counter()
    timings: dict[str, float] = {}
    report: Progress = progress or (lambda stage, message: None)
    path = Path(path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    notes: list[str] = []
    if case.mesh.element_order == 1:
        notes.append(
            "4-node tetrahedra are stiff in bending: deformation and bending stress "
            "read low (about half on thin walls); comparison only, no safety factor "
            "is given. Use element order 2 for results to act on."
        )

    def lap(name: str, since: float) -> None:
        timings[name] = time.perf_counter() - since

    t = time.perf_counter()
    report("inspect", f"Reading geometry: {path.name}")
    solid = inspect_part(path)
    lap("inspect", t)

    t = time.perf_counter()
    # A mesh made before the run (the CLI meshes before the interview, so the
    # face table and picture come from the mesh that is solved) still cost its
    # meshing time; report that, not the near-zero time it took to hand it over.
    earlier_meshing = 0.0
    if mesh is not None and int(mesh.order) == case.mesh.element_order:
        report("mesh", f"Reusing mesh: {int(mesh.node_count):,} nodes, {int(mesh.element_count):,} elements")
        earlier_meshing = float(getattr(mesh, "seconds", 0.0) or 0.0)
    else:
        report("mesh", f"Meshing ({case.mesh.describe()})")
        mesh = mesh_part(path, case.mesh)
    lap("mesh", t)
    timings["mesh"] += earlier_meshing
    if int(getattr(mesh, "solid_count", 1) or 1) > 1:
        raise RunRefused(
            f"{path.name} meshes into {mesh.solid_count} solids -- an assembly, not a "
            "part. Export each part as its own STEP file and run it alone."
        )
    case.check_faces(mesh.faces)

    files["case"] = _write_json(out / "case.json", case.as_document())

    threshold = large_dof_threshold()
    if int(mesh.dof_count) > threshold and not allow_large:
        seconds = estimate_seconds(mesh)
        raise RunRefused(
            f"this mesh has {int(mesh.dof_count):,} degrees of freedom, above "
            f"{threshold:,}; the solve would take about {fmt(seconds, 2)} s and "
            "several GB of memory. Allow it to go ahead (--allow-large), or use a "
            "coarser mesh (--mesh coarse or a larger --element-size)."
        )

    t = time.perf_counter()
    crosschecks: Path | None = None
    if not screen:
        screening = ScreeningOutcome("skipped", ("screening switched off (--no-screen)",), family)
    else:
        report("screen", "Screening gate (surrogate speaks only on its trained load cases)")
        w, c, n = find_surrogate_files(weights, calibration, neighbours)
        from .cad_intake import keep_sigint

        # The surrogate's mesher and the label solver (frozen: they define the
        # training data) start gmsh in a way that takes over Ctrl-C.
        with keep_sigint():
            screening, crosschecks = _screen(
                path, case, mesh, solid, out, family=family,
                weights=w, calibration=c, neighbours=n, crosscheck=crosscheck, report=report,
            )
    if crosschecks is not None:
        files["crosschecks"] = crosschecks
    lap("screen", t)
    if screening.load_seconds > 0.0:
        # Loading torch and the weights is paid once per process; the screen
        # itself is what a second part would cost.
        total_screen = timings.pop("screen")
        timings["load_model"] = float(screening.load_seconds)
        timings["screen"] = max(0.0, total_screen - float(screening.load_seconds))

    from .structural import solve_static

    t = time.perf_counter()
    seconds = estimate_seconds(mesh)
    report(
        "solve",
        f"Solving K u = F: {int(mesh.dof_count):,} DOF"
        + (f" (about {fmt(seconds, 2)} s)" if seconds else ""),
    )
    solution = solve_static(
        mesh, case, allow_large=allow_large,
        progress=lambda stage, message: report("solve_detail", message),
    )
    lap("solve", t)

    convergence = None
    if converge:
        t = time.perf_counter()
        report("converge", f"Convergence check: re-meshing at {CONVERGENCE_REFINE}x element size")
        convergence = _convergence(path, mesh, case, solution, allow_large)
        lap("converge", t)

    t = time.perf_counter()
    if render:
        report("render", "Drawing images")
        files.update(_render_all(path, out, mesh, case, solution, notes))
    lap("render", t)

    t = time.perf_counter()
    report("export", "Writing results files")
    if export_vtu:
        try:
            from .structural_export import write_vtu

            files["vtu"] = Path(write_vtu(solution, out / "results.vtu"))
        except Exception as exc:
            notes.append(f"results.vtu not written: {type(exc).__name__}: {exc}")
    files["results"] = out / "results.json"
    files["summary"] = out / "summary.txt"
    lap("files", t)
    timings["total"] = time.perf_counter() - started + earlier_meshing

    run = StructuralRun(
        path=path,
        out_dir=out,
        case=case,
        mesh=mesh,
        solution=solution,
        screening=screening,
        files=dict(files),
        timings=dict(timings),
        convergence=convergence,
        part=_part_facts(path, solid, mesh),
        warnings=tuple(notes),
    )
    _write_json(files["results"], run.as_document())
    files["summary"].write_text(format_summary(run) + "\n", encoding="utf-8")
    report("done", f"Done in {fmt(timings['total'], 3)} s")
    return run


def _render_all(path: Path, out: Path, mesh, case, solution, notes: list[str]) -> dict[str, Path]:  # noqa: ANN001
    """Each picture on its own: one failing must not cost the others, nor the
    solved numbers already in hand."""
    written: dict[str, Path] = {}
    try:
        from . import render_structural as rs
    except Exception as exc:
        notes.append(
            f"no images: {type(exc).__name__}: {exc} (install the solve extra: {INSTALL_HINT})"
        )
        return written

    # Two cameras, as a reader needs: the setup picture is framed on the
    # supports and loads, the result pictures on the hotspots. Drawing results
    # with the setup camera turned the Max away from the reader on a shelf
    # loaded from above, leaving its tag pointing at a blue region.
    try:
        view = rs.choose_view(mesh, case)
    except Exception:
        view = None
    try:
        result_view = rs.choose_result_view(solution)
    except Exception:
        result_view = None

    def attempt(name: str, filename: str, draw: Callable[[Path], Any]) -> None:
        target = out / filename
        try:
            written[name] = Path(draw(target))
        except Exception as exc:
            notes.append(f"{filename} not written: {type(exc).__name__}: {exc}")

    faces = out / "faces.png"
    if faces.is_file():
        written["faces_image"] = faces
    else:
        attempt("faces_image", "faces.png", lambda p: rs.render_face_ids(mesh, p))
    attempt("setup_image", "setup.png", lambda p: rs.render_setup(mesh, case, p, view=view))
    attempt("equivalent_stress_image", "equivalent_stress.png",
            lambda p: rs.render_result(solution, "equivalent_stress", p, view=result_view))
    attempt("total_deformation_image", "total_deformation.png",
            lambda p: rs.render_result(solution, "total_deformation", p, view=result_view))
    panels = [written[k] for k in ("setup_image", "equivalent_stress_image",
                                   "total_deformation_image") if k in written]
    if panels:
        attempt("board_image", "board.png", lambda p: rs.compose_board(
            panels, p, title=f"{path.name} - A: Static Structural"))
    return written


def _convergence(path: Path, mesh, case, solution, allow_large: bool) -> dict[str, Any]:  # noqa: ANN001
    """Re-mesh at 0.67x the element size and re-solve; Ansys' default allowable
    change is 20%. The stress compared is the max away from singularities when
    both meshes have one, since a singular max never converges."""
    from .structural import solve_static
    from .structural_mesh import mesh_for_analysis

    size = CONVERGENCE_REFINE * float(mesh.element_size_mm)
    fine = mesh_for_analysis(path, order=int(mesh.order), element_size_mm=size)
    threshold = large_dof_threshold()
    if int(fine.dof_count) > threshold and not allow_large:
        return {
            "status": "skipped",
            "reason": f"the refined mesh has {int(fine.dof_count):,} DOF, above "
                      f"{threshold:,}; rerun with --allow-large to include it",
        }
    refined = solve_static(fine, case, allow_large=allow_large)

    def change(a: float, b: float) -> float | None:
        return abs(b - a) / abs(b) * 100.0 if b else None

    if solution.max_stress_away is not None and refined.max_stress_away is not None:
        basis = "max away from singularities"
        s0, s1 = float(solution.max_stress_away.value), float(refined.max_stress_away.value)
    else:
        basis = "max equivalent stress"
        s0, s1 = float(solution.max_stress.value), float(refined.max_stress.value)
    d0, d1 = float(solution.max_deformation.value), float(refined.max_deformation.value)
    dc, sc = change(d0, d1), change(s0, s1)
    converged = dc is not None and sc is not None and max(dc, sc) < CONVERGENCE_ALLOWABLE_PCT
    return {
        "status": "converged" if converged else "not_converged",
        "verdict": (
            f"converged (<{CONVERGENCE_ALLOWABLE_PCT:g}%)" if converged else "not converged"
        ),
        "allowable_change_pct": CONVERGENCE_ALLOWABLE_PCT,
        "element_size_mm": [float(mesh.element_size_mm), float(fine.element_size_mm)],
        "nodes": [int(mesh.node_count), int(fine.node_count)],
        "max_deformation_mm": [d0, d1],
        "deformation_change_pct": dc,
        "stress_basis": basis,
        "stress_mpa": [s0, s1],
        "stress_change_pct": sc,
        "note": "results and images are from the first mesh",
    }


# --------------------------------------------------------------------------
# The results table
# --------------------------------------------------------------------------

_WIDTH = 92
_LABEL = 30
_VALUE = 16


def _row(label: str, value: str, note: str = "") -> list[str]:
    """label | value | note, the note wrapped under itself so the value column
    stays readable down the table."""
    import textwrap

    head = f"  {label:<{_LABEL}}"
    head += f"{value:<{_VALUE}}" if len(value) < _VALUE else value + "  "
    if not note:
        return [head.rstrip()]
    indent = len(head)
    if indent > _WIDTH - 24:
        return [head.rstrip()] + textwrap.wrap(
            note, width=_WIDTH, initial_indent=" " * (2 + _LABEL),
            subsequent_indent=" " * (2 + _LABEL),
        )
    wrapped = textwrap.wrap(note, width=_WIDTH - indent) or [""]
    return [head + wrapped[0]] + [" " * indent + chunk for chunk in wrapped[1:]]


def _wrap(text: str, indent: int = 4) -> list[str]:
    import textwrap

    return textwrap.wrap(
        text, width=_WIDTH, initial_indent=" " * (indent - 2) + "- ",
        subsequent_indent=" " * indent,
    ) or [""]


def format_screening(outcome: ScreeningOutcome) -> list[str]:
    """The screening block: always headed as an estimate, kept apart from the
    solved numbers, and the critical-distance value only in the crosscheck."""
    lines = ["SURROGATE SCREENING (estimate, not an analysis)"]
    status = outcome.status
    if status == "screened" and outcome.document:
        document = outcome.document
        prediction = (document.get("predictions") or [{}])[0]
        coverage = document.get("coverage")
        band = f"{outcome.family!r} training case" if outcome.family else "widest band"
        cover = f", {float(coverage):.0%} interval" if coverage is not None else ""
        lines += _row("Status", "screened", f"({band}{cover})")
        lines += _row(
            "Interval",
            f"{fmt(prediction.get('lower'), 4)} - {fmt(prediction.get('upper'), 4)} MPa",
            "critical-distance stress (a volume average, not the max)",
        )
        cc = outcome.crosscheck or {}
        if cc.get("available"):
            verdict = "inside" if cc.get("inside") else "OUTSIDE"
            lines += _row(
                "P1 label-solver check",
                f"{fmt(cc.get('p1_critical_distance_mpa'), 4)} MPa",
                f"{verdict} the interval (the solver the training labels came from)",
            )
        elif cc.get("error"):
            lines += _row("P1 label-solver check", "failed", str(cc["error"]))
        elif cc.get("skipped"):
            lines += _row("P1 label-solver check", "-", str(cc["skipped"]))
        for limitation in document.get("limitations") or []:
            text = str(limitation)
            if "extrapolat" in text or "not established" in text:
                lines += _wrap(text)
        # No ratio here: it varies with the part (0.32-0.37 of the surface max
        # on the sample bracket and the demo, where "about half" was printed),
        # and a solved number must not be dragged into the screening block.
        lines += _wrap(
            "the critical-distance value is an average a short distance below "
            "the surface, so in bending it reads well below the peak surface "
            "stress: it is not the design stress; use the solved results above"
        )
    elif status == "not_screenable":
        lines.append("  Not screenable; the solved results above are the answer:")
        for reason in outcome.reasons:
            lines += _wrap(reason)
    else:
        lines += _row("Status", status.replace("_", " "))
        for reason in outcome.reasons:
            lines += _wrap(reason)
    return lines


def format_summary(run: StructuralRun) -> str:
    """The terminal results table, also written to summary.txt."""
    s, m, case = run.solution, run.mesh, run.case
    letters = {
        tuple(item.faces): letter
        for letter, item in case.lettered() if isinstance(item, FixedSupport)
    }
    lines: list[str] = []
    lines.append(f"RESULTS  {run.path.name}  -  A: Static Structural")
    lines.append(
        f"Solved: linear elastic, {_element_name(int(m.order))}, {case.material.name}"
    )
    lines.append("=" * _WIDTH)
    if int(m.order) == 1:
        lines += _wrap(
            "WARNING: 4-node tetrahedra are stiff in bending -- deformation and "
            "bending stress read low (about half on thin walls). Comparison only: "
            "rerun with --order 2 for results to act on.", indent=0)

    max_s, away = s.max_stress, s.max_stress_away
    lines.append("EQUIVALENT (VON-MISES) STRESS   [MPa, averaged nodal]")
    lines += _row("Maximum", f"{fmt(max_s.value)} MPa",
                  f"at {fmt_vec(max_s.location_mm)} mm, {fmt_faces(max_s.faces)}")
    if max_s.singular:
        lines += _row("", "singular", f"{max_s.reason or 'at a flagged singularity'}; "
                      "it rises with mesh refinement")
    # The second number only earns a row when it says something the first does
    # not: when the max is singular, or something flagged sits near it. A row
    # repeating the max verbatim reads as if the two were different findings.
    if away is not None and (
        max_s.singular or float(away.value) < float(max_s.value) * (1.0 - 5e-3)
    ):
        lines += _row("Max away from singularities", f"{fmt(away.value)} MPa",
                      f"farther than {fmt(s.exclusion_radius_mm, 3)} mm from flagged "
                      f"features, at {fmt_vec(away.location_mm)} mm")
        extrapolated = getattr(s, "max_stress_extrapolated", None)
        if max_s.singular and extrapolated is not None:
            lines += _row("Hot-spot stress at the edge", f"{fmt(extrapolated)} MPa",
                          "extrapolated linearly from 1-2 wall thicknesses out; "
                          "the section's real stress without the singular spike")
    elif not max_s.singular:
        lines += _row("", "not singular",
                      "clear of support edges and sharp re-entrant corners")
    lines += _row("Minimum", f"{fmt(s.min_stress.value)} MPa")

    lines.append("TOTAL DEFORMATION   [mm]")
    lines += _row("Maximum", f"{fmt(s.max_deformation.value)} mm",
                  f"at {fmt_vec(s.max_deformation.location_mm)} mm, "
                  f"{fmt_faces(s.max_deformation.faces)}")
    lines += _row("Minimum", f"{fmt(s.min_deformation.value)} mm")
    scale = float(s.deformation_scale_auto)
    lines += _row("Image scale (auto)", f"{fmt(scale, 3)} x",
                  "deformation drawn exaggerated by this factor" if scale > 1.0 + 1e-9
                  else "deformation drawn at true size" if scale > 1.0 - 1e-9
                  else "deformation drawn reduced by this factor (it is large: "
                       "over 5% of the part's size)")

    lines.append("REACTIONS   [N]")
    for reaction in s.reactions:
        letter = letters.get(tuple(reaction.faces), "")
        magnitude = math.sqrt(sum(float(c) ** 2 for c in reaction.force_n))
        lines += _row(f"{letter} {reaction.label}".strip(), fmt_vec(reaction.force_n),
                      f"|R| {fmt(magnitude)} N on {fmt_faces(reaction.faces)}")
    lines += _row("Applied total", fmt_vec(s.applied_total_n))
    lines += _row("Reaction total", fmt_vec(s.reaction_total_n))
    lines += _row("Equilibrium error", f"{float(s.equilibrium_error) * 100:.2g} %")

    lines.append("YIELD CHECK   [linear elastic]")
    check = run.yield_check()
    if check is None:
        lines.append("  no tensile yield for this material, so no safety factor is given")
    else:
        lines += _row("Tensile yield", f"{fmt(check['tensile_yield_mpa'])} MPa")
        if check["stress_used_mpa"] is None:
            lines += _row("Safety factor", "-", check["basis"])
        else:
            lines += _row("Stress used", f"{fmt(check['stress_used_mpa'])} MPa", check["basis"])
            factor = check["safety_factor"]
            lines += _row(
                "Safety factor", fmt(factor, 3) if factor is not None else "-",
                "yield / stress; no plasticity, fatigue or code factors"
                + ("; below 1: the linear-elastic stress exceeds yield"
                   if factor is not None and factor < 1 else ""),
            )
            bound = check.get("safety_factor_at_singular_max")
            if bound is not None:
                lines += _row("At the singular max", fmt(bound, 3),
                              "conservative bound; it falls as the mesh is refined")
            optimistic = check.get("safety_factor_away_from_singularities")
            if optimistic is not None:
                lines += _row("Away from singularities", fmt(optimistic, 3),
                              "optimistic bound: leaves out the section at the flagged "
                              "edge; refine the mesh (medium or finer) for the hot-spot value")

    lines += format_screening(run.screening)

    lines.append("MESH & TIME")
    lines += _row("Mesh", f"{int(m.node_count):,} nodes, {int(m.element_count):,} elements, "
                  f"{int(m.dof_count):,} DOF")
    quality_min = getattr(m, "element_quality_min", None)
    quality_median = getattr(m, "element_quality_median", None)
    lines += _row("Element size", f"{fmt(m.element_size_mm, 3)} mm",
                  ("valid: no inverted elements" if float(m.min_scaled_jacobian) > 0
                   else "INVERTED elements present")
                  + ("; straight-sided" if bool(m.straight_sided) else ""))
    if quality_min is not None and quality_median is not None:
        lines += _row("Element quality", f"min {fmt(quality_min, 2)}",
                      f"median {fmt(quality_median, 2)} (Ansys metric: 1 = regular "
                      "tetrahedron, near 0 = sliver)")
    stages = ", ".join(
        f"{name.replace('_', ' ')} {float(value):.1f} s"
        for name, value in run.timings.items() if name != "total"
    )
    lines += _row("Time", f"{float(run.timings.get('total', 0.0)):.1f} s", stages)
    if run.convergence:
        conv = run.convergence
        if conv.get("status") == "skipped":
            lines += _row("Convergence", "skipped", str(conv.get("reason", "")))
        else:
            lines += _row(
                "Convergence", str(conv.get("verdict")),
                f"deformation {fmt(conv.get('deformation_change_pct'), 2)} %, "
                f"{conv.get('stress_basis')} {fmt(conv.get('stress_change_pct'), 2)} % "
                f"at {fmt(conv['element_size_mm'][1], 3)} mm",
            )
    for note in run.warnings:
        lines += _wrap("warning: " + note)

    lines.append(f"FILES   {run.out_dir}")
    for name, file in run.files.items():
        lines.append(f"  {_relative(file, run.out_dir):<28}{_FILE_NOTES.get(name, '')}")
    return "\n".join(lines)


_FILE_NOTES = {
    "case": "the case as solved (rerun with --case)",
    "faces_image": "numbered faces",
    "setup_image": "supports and loads",
    "equivalent_stress_image": "equivalent (von-Mises) stress",
    "total_deformation_image": "total deformation",
    "board_image": "setup + both results side by side",
    "vtu": "ParaView: displacement and stress fields",
    "results": "every number here, machine-readable",
    "summary": "this table",
    "crosschecks": "screening vs P1 label solve log",
}


__all__ = [
    "CALIBRATION_ENV",
    "CONVERGENCE_ALLOWABLE_PCT",
    "NEIGHBOURS_ENV",
    "SCHEMA",
    "WEIGHTS_ENV",
    "RunRefused",
    "ScreeningOutcome",
    "StructuralRun",
    "StructuralRunError",
    "axis_label",
    "default_out_dir",
    "estimate_seconds",
    "find_surrogate_files",
    "fmt",
    "fmt_faces",
    "fmt_vec",
    "format_screening",
    "format_summary",
    "inspect_part",
    "large_dof_threshold",
    "mesh_part",
    "read_calibration",
    "render_faces_image",
    "run_static_structural",
    "screening_verdict",
]
