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


def test_a_banded_contour_uses_discrete_levels(tmp_path) -> None:
    """Nine bands is what Ansys and Abaqus show, and the reason is readability:
    a continuous ramp says where the stress is high, bands let a reader put a
    number to a region."""
    from fea_surrogate.render import CONTOUR_LEVELS, banded_colours

    values = np.linspace(0.0, 100.0, 500)
    colours, norm, _ = banded_colours(values, 0.0, 100.0, "cividis")

    assert len({tuple(c) for c in colours}) == CONTOUR_LEVELS
    assert len(norm.boundaries) == CONTOUR_LEVELS + 1
    assert norm.boundaries[0] == 0.0 and norm.boundaries[-1] == 100.0


def test_banding_is_on_by_default_and_can_be_turned_off(tmp_path) -> None:
    banded = render_solved(_tetrahedron(), out_dir=tmp_path, stem="b")
    smooth = render_solved(_tetrahedron(), out_dir=tmp_path, stem="s", levels=0)

    assert banded.image.read_bytes() != smooth.image.read_bytes()


def test_the_colourbar_is_ticked_on_the_band_boundaries(tmp_path) -> None:
    """A band with no number either side of it is a colour to interpolate by eye."""
    out = render_solved(
        _tetrahedron((0.0, 30.0, 60.0, 90.0)),
        out_dir=tmp_path, fmt="svg", stem="s", levels=3,
    )
    text = out.image.read_text(encoding="utf-8")

    # Boundaries at 0, 30, 60, 90 for three bands over that range.
    for boundary in ("30", "60"):
        assert boundary in text


def test_a_degenerate_range_does_not_crash_the_banding() -> None:
    """A part under no stress at all is flat, and flat must still plot."""
    from fea_surrogate.render import banded_colours

    colours, norm, _ = banded_colours(np.zeros(10), 0.0, 0.0, "cividis")

    assert len(colours) == 10
    assert norm.boundaries[-1] > norm.boundaries[0]


def _spiky(n: int = 200, hot: int = 4) -> _Solved:
    """A realistic field: most of the part quiet, a few nodes at a singularity.

    Needs enough nodes for a percentile to mean anything. With four values the
    95th percentile lands between the top two and the clip correctly declines to
    fire, which is what the first version of this test got wrong.
    """
    rng = np.random.default_rng(0)
    vertices = rng.random((n, 3)) * 10.0
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    # The quiet region has to vary. Perfectly uniform, its 95th percentile
    # lands exactly on the minimum, and clipping there would collapse the range
    # to nothing -- which is what the guard in `render_solved` refuses to do.
    field = np.linspace(8.0, 20.0, n)
    field[:hot] = 900.0
    return _Solved(
        vertices=vertices, surface_faces=faces, von_mises=field,
        element_von_mises=field[:1], peak_von_mises=float(field.max()),
        critical_distance_mpa=15.0, node_count=n, element_count=1, seconds=0.1,
    )


def test_the_scale_is_clipped_so_a_singularity_does_not_eat_the_bands(tmp_path) -> None:
    """An elastic peak sits at a singularity and can be several times the stress
    anywhere else. Scaled to the maximum, one band covers the singular corner
    and the whole rest of the part crushes into the bottom band -- measured on
    the L-bracket, max 1903 MPa against a 95th percentile of 1223.
    """
    text = render_solved(
        _spiky(), out_dir=tmp_path, fmt="svg", stem="c"
    ).image.read_text(encoding="utf-8")

    # Stated on the plot, so the top band is never mistaken for the maximum.
    assert "clipped at p95" in text


def test_a_field_with_no_singularity_is_not_clipped(tmp_path) -> None:
    """Clipping a part that does not need it discards range for nothing."""
    flat = _spiky(hot=0)

    text = render_solved(
        flat, out_dir=tmp_path, fmt="svg", stem="n"
    ).image.read_text(encoding="utf-8")

    assert "clipped" not in text


def test_clipping_can_be_turned_off(tmp_path) -> None:
    text = render_solved(
        _spiky(), out_dir=tmp_path, fmt="svg", stem="f", clip_percentile=100.0
    ).image.read_text(encoding="utf-8")

    assert "clipped" not in text
