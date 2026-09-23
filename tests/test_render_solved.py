"""A solved field must look solved, and a predicted one must not.

`render.py` washes out a predicted field, stamps ESTIMATE across it and heads it
NOT AN ANALYSIS, because a crisp contour borrows authority its data may not have.
A real solve has that authority, within its idealisation, and dressing it as an
estimate would be its own kind of lie.

The two renders therefore have to be distinguishable at a glance and provably
so -- which is what these check, by reading the text actually burned into the
output rather than trusting the code that wrote it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip("matplotlib", reason="rendering requires the render extra")

from fea_surrogate.render import (  # noqa: E402
    SCREENING_HEADER,
    SCREENING_WATERMARK,
    render_field,
)
from fea_surrogate.render_solved import SOLVED_HEADER, render_solved  # noqa: E402
from fea_surrogate.screening import ScreeningContractError  # noqa: E402


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class _Solved:
    """Stands in for `solve_cad.SolvedPart`, so these run without a solver."""

    vertices: np.ndarray
    surface_faces: np.ndarray
    von_mises: np.ndarray
    element_von_mises: np.ndarray
    peak_von_mises: float
    critical_distance_mpa: float
    node_count: int
    element_count: int
    seconds: float


def _tetrahedron(values=(10.0, 40.0, 90.0, 160.0)) -> _Solved:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    field = np.array(values, dtype=float)
    return _Solved(
        vertices=vertices,
        surface_faces=faces,
        von_mises=field,
        element_von_mises=field[:1],
        peak_von_mises=float(field.max()),
        critical_distance_mpa=60.0,
        node_count=4,
        element_count=1,
        seconds=0.12,
    )


def _svg_text(path) -> str:
    return path.read_text(encoding="utf-8")


def test_a_solved_render_says_solved_and_never_says_estimate(tmp_path) -> None:
    """The one check that stops the two kinds of plot being confused."""
    out = render_solved(_tetrahedron(), out_dir=tmp_path, fmt="svg", stem="s")
    text = _svg_text(out.image)

    assert "SOLVED" in text
    assert "linear elastic" in text
    assert SCREENING_WATERMARK not in text
    assert "NOT AN ANALYSIS" not in text


def test_a_screening_render_still_carries_both_warnings(tmp_path) -> None:
    """Guard the guard: if the screening stamp ever quietly disappeared, the
    test above would pass for the wrong reason."""
    values = np.array([10.0, 20.0, 30.0, 40.0])
    part = _tetrahedron()
    render_field(
        out_dir=tmp_path, quantity="von_mises", units="MPa",
        vertices=part.vertices, faces=part.surface_faces,
        values=values, lower=values * 0.8, upper=values * 1.2,
        coverage=0.9, model_name="m", model_version="1",
        stem="e", fmt="svg",
    )
    text = _svg_text(tmp_path / "e.svg")

    assert "NOT AN ANALYSIS" in text
    assert SCREENING_WATERMARK in text
    assert "SOLVED" not in text


def test_the_two_headers_cannot_be_mistaken_for_each_other() -> None:
    assert SCREENING_WATERMARK not in SOLVED_HEADER
    assert "SOLVED" not in SCREENING_HEADER


def test_the_stamp_names_the_converged_number_and_the_singular_one(tmp_path) -> None:
    """A reader has to know the peak states the mesh rather than the part."""
    out = render_solved(_tetrahedron(), out_dir=tmp_path, fmt="svg", stem="s")
    text = _svg_text(out.image)

    assert "singularity" in text
    assert "refinement" in text


def test_a_solved_render_writes_a_real_png_and_its_digest(tmp_path) -> None:
    out = render_solved(_tetrahedron(), out_dir=tmp_path)

    head = out.image.read_bytes()[:16]
    assert head.startswith(PNG_MAGIC)
    assert head[12:16] == b"IHDR"
    assert len(out.image_sha256) == 64
    assert len(out.mesh_sha256) == 64


def test_the_field_is_saved_alongside_the_picture(tmp_path) -> None:
    """The image is for looking at; the values are for computing with."""
    out = render_solved(_tetrahedron(), out_dir=tmp_path)

    np.testing.assert_allclose(np.load(out.values), [10.0, 40.0, 90.0, 160.0])


def test_a_field_that_does_not_match_the_mesh_is_refused(tmp_path) -> None:
    part = _tetrahedron()
    broken = _Solved(
        part.vertices, part.surface_faces, np.array([1.0, 2.0]),
        part.element_von_mises, 2.0, 1.0, 4, 1, 0.1,
    )

    with pytest.raises(ScreeningContractError, match="one value per vertex"):
        render_solved(broken, out_dir=tmp_path)


def test_a_non_finite_field_is_refused(tmp_path) -> None:
    with pytest.raises(ScreeningContractError, match="not finite"):
        render_solved(_tetrahedron((1.0, np.inf, 3.0, 4.0)), out_dir=tmp_path)
