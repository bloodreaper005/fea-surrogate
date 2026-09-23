"""Solve an imported CAD part, under the same load case the screening used.

Screening is triage: a fast estimate that aims an expensive check at the right
candidate. This is that expensive check, and it turns out to be far less
expensive than the framing suggested -- measured on an 11,229-node bracket, the
surrogate answers in 60 ms with one scalar interval and the solver answers in
3.5 s with the true field at every node.

So a contour plot needs no trained nodal head. `generate/solver.py` already
produces exactly the field a contour wants; it is what generated every training
label. What was missing was a way to point it at an imported STEP rather than at
a part the sweep had just built.

Two things this module refuses to do differently from the screening path:

* boundary conditions come from `face_selectors`, the same rule the sweep used,
  so the solve and the screen are posed on the same problem. Two paths that
  disagree about the load case is the defect this project has already paid for
  twice.
* the load case comes from `load_case.parse`, so the same interview answers
  drive both.

What comes back is the true answer **to this idealisation** -- linear elastic,
small displacement, P1 tetrahedra, one uniform traction, no contact and no body
loads. That is a much stronger claim than a screening estimate and it is still
not a strength certification.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


#: Above this, warn before spending the time. Measured on one bracket, solve
#: time runs 0.16 to 0.31 s per thousand nodes across a fourteen-fold range --
#: mildly superlinear, comfortable below 20k, noticeable near 50k.
NODE_WARNING_THRESHOLD = 20_000

#: Seconds per thousand nodes, for the estimate shown before a long solve. Taken
#: from the upper end of that measured range, so the warning errs slow.
SECONDS_PER_THOUSAND_NODES = 0.31


class SolveCadError(RuntimeError):
    """The part could not be posed as a solvable problem."""


@dataclass(frozen=True)
class SolvedPart:
    """A solved field, with the two scalars and which one to believe."""

    path: Path
    vertices: np.ndarray
    surface_faces: np.ndarray
    von_mises: np.ndarray            # (N,) nodal, MPa -- the contour
    peak_von_mises: float            # element peak; singular at a re-entrant corner
    critical_distance_mpa: float     # volume-averaged; the converged number
    element_von_mises: np.ndarray
    node_count: int
    element_count: int
    seconds: float
    force_n: float
    length_mm: float
    poisson_ratio: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "FeaSolve/v1",
            "attestation": "computed_analysis",
            "path": str(self.path),
            "critical_distance_mpa": self.critical_distance_mpa,
            "peak_von_mises_mpa": self.peak_von_mises,
            "use": (
                "critical_distance_mpa. The element peak is singular at a "
                "re-entrant corner and rises without bound under mesh "
                "refinement, so it states the mesh rather than the part."
            ),
            "nodes": self.node_count,
            "elements": self.element_count,
            "seconds": round(self.seconds, 3),
            "force_n": self.force_n,
            "characteristic_length_mm": self.length_mm,
            "poisson_ratio": self.poisson_ratio,
            "idealisation": [
                "linear elastic, small displacement",
                "first-order tetrahedra",
                "one uniform traction on the loaded faces",
                "no contact, no body loads, no thermal strain",
            ],
            "limitations": [
                "the true answer to this idealisation, not a strength "
                "certification and not a manufacturing release",
                "the element peak is mesh-dependent; the critical-distance "
                "value is the converged one",
            ],
        }


def estimate_seconds(node_count: int) -> float:
    return node_count / 1000.0 * SECONDS_PER_THOUSAND_NODES


def solve_step(
    path: str | Path,
    case,  # noqa: ANN001 - load_case.LoadCase
    *,
    allow_large: bool = False,
) -> SolvedPart:
    """Mesh an imported part, pose the stated load case on it, and solve.

    `case` is a `load_case.LoadCase`, the same object the screening path uses.
    """
    from .face_selectors import FaceSelectionError, face_mask, select_faces
    from .features import mesh_step
    from .generate.family import Sample, Tagging
    from .generate.harness import (
        MeshedSample,
        _signed_distance_to_box,
        _vertex_normals,
    )
    from .generate.solver import SolveError, solve

    resolved = Path(path)
    mesh = mesh_step(resolved)
    node_count = len(mesh.vertices)

    if node_count > NODE_WARNING_THRESHOLD and not allow_large:
        raise SolveCadError(
            f"this part meshes to {node_count} nodes, above the "
            f"{NODE_WARNING_THRESHOLD} at which a solve stops being "
            f"interactive -- roughly {estimate_seconds(node_count):.0f} s. "
            "Pass allow_large to go ahead, or screen it first: screening is "
            "60 ms and tells you whether this one is worth the wait."
        )

    try:
        fixed = face_mask(select_faces(case.fixed, mesh, "fixed"), mesh)
        loaded = face_mask(select_faces(case.loaded, mesh, "loaded"), mesh)
    except FaceSelectionError as exc:
        raise SolveCadError(str(exc)) from None

    vertices = np.asarray(mesh.vertices, dtype=float)
    triangles = np.ascontiguousarray(np.asarray(mesh.faces).T)
    direction = np.asarray(case.load.direction, dtype=float)

    features = np.concatenate(
        [
            vertices,
            _signed_distance_to_box(vertices),
            _vertex_normals(vertices, triangles),
            fixed.reshape(-1, 1).astype(float),
            loaded.reshape(-1, 1).astype(float),
            np.tile(direction, (node_count, 1)) * loaded.reshape(-1, 1),
            np.full((node_count, 1), case.poisson_ratio),
        ],
        axis=1,
    )

    # `Tagging` wants face tags, which mean nothing here -- the masks are already
    # resolved. It is only carried so the solver can read the load direction.
    sample = Sample(
        resolved.stem,
        {},
        Tagging((1,), (2,), tuple(direction)),
        case.poisson_ratio,
        case.load.magnitude_n,
    )
    meshed = MeshedSample(
        sample,
        vertices,
        np.asarray(mesh.tetrahedra),
        triangles,
        features,
        fixed,
        loaded,
    )

    started = time.time()
    try:
        result = solve(meshed)
    except SolveError as exc:
        raise SolveCadError(str(exc)) from None
    seconds = time.time() - started

    return SolvedPart(
        path=resolved,
        vertices=vertices,
        surface_faces=triangles,
        von_mises=np.asarray(result.von_mises, dtype=float),
        peak_von_mises=float(result.peak_von_mises),
        critical_distance_mpa=float(result.critical_distance_stress()),
        element_von_mises=np.asarray(result.element_von_mises, dtype=float),
        node_count=node_count,
        element_count=len(meshed.tetrahedra),
        seconds=seconds,
        force_n=case.load.magnitude_n,
        length_mm=meshed.characteristic_length_mm,
        poisson_ratio=case.poisson_ratio,
    )


def compare_to_screening(
    solved: SolvedPart, document: dict[str, Any] | None
) -> dict[str, Any]:
    """Did the screening interval contain the solved answer?

    Every solve is a free deployed-coverage datapoint, measured on a part someone
    actually cared about rather than on geometry this project generated. A
    synthetic sweep cannot produce that, however many samples it runs.
    """
    if not document:
        return {"available": False}
    predictions = document.get("predictions") or []
    if not predictions:
        return {"available": False}

    prediction = predictions[0]
    lower = float(prediction["lower"])
    upper = float(prediction["upper"])
    truth = solved.critical_distance_mpa
    return {
        "available": True,
        "quantity": prediction.get("quantity"),
        "screening_lower": lower,
        "screening_upper": upper,
        "solved": truth,
        "inside": bool(lower <= truth <= upper),
        "ratio_to_band_centre": (
            truth / float(np.sqrt(max(lower, 1e-12) * upper))
            if upper > 0
            else None
        ),
        "coverage_claimed": document.get("coverage"),
    }


__all__ = [
    "NODE_WARNING_THRESHOLD",
    "SECONDS_PER_THOUSAND_NODES",
    "SolveCadError",
    "SolvedPart",
    "compare_to_screening",
    "estimate_seconds",
    "solve_step",
]
