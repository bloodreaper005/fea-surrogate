"""Solving an imported part, and keeping it visibly apart from an estimate.

Screening is triage that aims an expensive check at the right candidate. This is
that check, and it is cheaper than the framing suggested -- 60 ms to screen an
11,000-node bracket against 3.5 s to solve it outright.

The two failures worth catching are a load case corrupted on its way through the
adapter, so the solve answers a different question from the screen, and a solved
render that cannot be told apart from a predicted one.
"""

from __future__ import annotations

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh", reason="meshing requires the inference extra")
pytest.importorskip("scipy", reason="solving requires the inference extra")
pytest.importorskip("torch", reason="mesh_step imports torch_geometric")

from fea_surrogate.cad_intake import inspect  # noqa: E402
from fea_surrogate.face_selectors import face_mask, select_faces  # noqa: E402
from fea_surrogate.features import mesh_step  # noqa: E402
from fea_surrogate.load_case import parse  # noqa: E402
from fea_surrogate.solve_cad import (  # noqa: E402
    NODE_WARNING_THRESHOLD,
    SolveCadError,
    compare_to_screening,
    estimate_seconds,
    solve_step,
)


def _bar(tmp_path, length=200.0, width=40.0, thickness=10.0):
    """A plain box, so the answer has a closed form to be checked against."""
    path = tmp_path / "bar.step"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("bar")
        gmsh.model.occ.addBox(0.0, 0.0, 0.0, length, width, thickness)
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


def _cantilever_case(path, force=1000.0):
    """Held at x-min, pushed down at x-max: the classic closed-form case."""
    solid = inspect(path)
    held = [f.tag for f in solid.faces if f.normal[0] < -0.9]
    pushed = [f.tag for f in solid.faces if f.normal[0] > 0.9]
    return solid, parse(
        {
            "fixed": {"faces": held},
            "load": {"faces": pushed, "magnitude_n": force,
                     "direction": [0.0, 0.0, -1.0]},
            "material": {"poisson_ratio": 0.3},
        },
        solid,
    )


def test_a_solved_cantilever_reproduces_the_closed_form(tmp_path) -> None:
    """sigma = 6FL/(Wt^2).

    The adapter's whole job is to hand the solver the load case it was given.
    If it corrupts anything on the way -- a mask, a direction, a magnitude --
    this is where it shows, because the right answer is known independently.
    """
    length, width, thickness, force = 200.0, 40.0, 10.0, 1000.0
    path = _bar(tmp_path, length, width, thickness)
    _, case = _cantilever_case(path, force)

    solved = solve_step(path, case)

    exact = 6 * force * length / (width * thickness**2)
    assert solved.peak_von_mises == pytest.approx(exact, rel=0.2)
    assert solved.critical_distance_mpa > 0.0
    # An average over a neighbourhood cannot exceed the maximum in it.
    assert solved.critical_distance_mpa <= solved.peak_von_mises


def test_the_solve_uses_the_same_masks_the_screening_would(tmp_path) -> None:
    """Two paths disagreeing about the load case is the defect already paid for
    twice here -- once in the selectors, once in the spring geometry."""
    path = _bar(tmp_path)
    _, case = _cantilever_case(path)
    mesh = mesh_step(path)

    expected_fixed = face_mask(select_faces(case.fixed, mesh, "fixed"), mesh)
    solved = solve_step(path, case)

    assert solved.node_count == len(mesh.vertices)
    assert int(expected_fixed.sum()) > 0
    assert solved.von_mises.shape == (solved.node_count,)


def test_stress_is_linear_in_the_applied_force(tmp_path) -> None:
    path = _bar(tmp_path)
    _, weak = _cantilever_case(path, force=1000.0)
    _, strong = _cantilever_case(path, force=3000.0)

    a = solve_step(path, weak).critical_distance_mpa
    b = solve_step(path, strong).critical_distance_mpa

    assert b == pytest.approx(3.0 * a, rel=0.02)


def test_a_part_too_large_to_be_interactive_warns_rather_than_stalling(
    tmp_path, monkeypatch
) -> None:
    """A minute of silence is indistinguishable from a hang."""
    path = _bar(tmp_path)
    _, case = _cantilever_case(path)
    monkeypatch.setattr("fea_surrogate.solve_cad.NODE_WARNING_THRESHOLD", 10)

    with pytest.raises(SolveCadError, match="allow_large"):
        solve_step(path, case)

    # And the caller can insist.
    assert solve_step(path, case, allow_large=True).node_count > 10


def test_the_warning_estimates_how_long_it_would_take() -> None:
    assert estimate_seconds(20_000) == pytest.approx(6.2, rel=0.01)
    assert estimate_seconds(NODE_WARNING_THRESHOLD) > 0.0


def test_the_summary_says_which_number_to_use(tmp_path) -> None:
    """The peak is singular at a re-entrant corner and states the mesh."""
    path = _bar(tmp_path)
    _, case = _cantilever_case(path)

    record = solve_step(path, case).as_dict()

    assert record["attestation"] == "computed_analysis"
    assert record["schema_version"] != "SurrogateScreening/v1"
    assert "critical_distance_mpa" in record["use"]
    assert any("not a strength certification" in x for x in record["limitations"])


def test_a_solve_crosschecks_the_screening_interval(tmp_path) -> None:
    """Every solve is a coverage datapoint on a part someone cared about."""
    path = _bar(tmp_path)
    _, case = _cantilever_case(path)
    solved = solve_step(path, case)

    document = {
        "coverage": 0.9,
        "predictions": [{
            "quantity": "von_mises_critical_distance",
            "lower": solved.critical_distance_mpa * 0.8,
            "upper": solved.critical_distance_mpa * 1.25,
            "units": "MPa",
        }],
    }
    inside = compare_to_screening(solved, document)

    assert inside["available"] and inside["inside"]
    assert inside["solved"] == pytest.approx(solved.critical_distance_mpa)

    document["predictions"][0]["upper"] = solved.critical_distance_mpa * 0.5
    document["predictions"][0]["lower"] = solved.critical_distance_mpa * 0.1
    assert not compare_to_screening(solved, document)["inside"]


def test_no_screening_document_is_not_a_failure(tmp_path) -> None:
    """A missing model must not cost the solve you already waited for."""
    path = _bar(tmp_path)
    _, case = _cantilever_case(path)

    assert compare_to_screening(solve_step(path, case), None) == {"available": False}


def test_a_load_case_naming_an_absent_face_is_refused(tmp_path) -> None:
    path = _bar(tmp_path)
    solid = inspect(path)
    case = parse(
        {
            "fixed": {"faces": [9999]},
            "load": {"faces": [1], "magnitude_n": 100.0},
        },
        solid,
    )

    with pytest.raises(SolveCadError, match="no face"):
        solve_step(path, case)
