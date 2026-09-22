"""Retrieval: the check the domain gate cannot make.

A part can satisfy every declared bound and still be unlike anything in three
thousand samples. The gate has no way to see that, because the bounds say nothing
about shape. These tests are mostly about the descriptor being *invariant* to the
things it should be invariant to -- if it moves when a part is merely rescaled or
rotated, the distances it produces mean nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from fea_surrogate.neighbours import (
    DESCRIPTOR_NAMES,
    NeighbourIndex,
    NeighbourIndexError,
    describe,
)


def _cube_mesh(scale: float = 1.0, rotation: np.ndarray | None = None, seed: int = 0):
    """A crude tetrahedralised box with a held end and a loaded end."""
    rng = np.random.default_rng(seed)
    points = rng.random((60, 3)) * np.array([4.0, 2.0, 1.0]) * scale
    if rotation is not None:
        points = points @ rotation.T

    # Delaunay is the cheapest way to get a valid tetrahedral connectivity.
    scipy_spatial = pytest.importorskip("scipy.spatial", reason="needs scipy")
    tets = scipy_spatial.Delaunay(points).simplices

    axis = points[:, 0] if rotation is None else (points @ rotation)[:, 0]
    fixed = axis < np.quantile(axis, 0.15)
    loaded = axis > np.quantile(axis, 0.85)

    direction = np.array([0.0, 0.0, -1.0])
    if rotation is not None:
        direction = rotation @ direction

    n = len(points)
    x = np.concatenate(
        [
            points,
            np.zeros((n, 1)),                       # sdf, unused by describe
            np.zeros((n, 3)),                       # normals, unused
            fixed.reshape(-1, 1).astype(float),
            loaded.reshape(-1, 1).astype(float),
            np.tile(direction, (n, 1)) * loaded.reshape(-1, 1),
            np.full((n, 1), 0.31),
        ],
        axis=1,
    )
    return x, tets


class _Sample:
    def __init__(self, family, x, tets, tau, force_n=1000.0, length_mm=10.0):
        self.family, self.x, self.tetrahedra = family, x, tets
        self.tau, self.force_n, self.length_mm = tau, force_n, length_mm


def test_the_descriptor_has_the_length_it_declares() -> None:
    x, tets = _cube_mesh()

    assert describe(x, tets).shape == (len(DESCRIPTOR_NAMES),)


def test_the_descriptor_does_not_move_when_the_part_is_rescaled() -> None:
    """Every entry is a ratio, a fraction or a direction, so scale must vanish.

    If it did not, a part modelled in metres would be judged unlike the identical
    part modelled in millimetres.
    """
    small = describe(*_cube_mesh(scale=1.0))
    large = describe(*_cube_mesh(scale=25.0))

    np.testing.assert_allclose(small, large, rtol=1e-6, atol=1e-6)


def test_the_descriptor_survives_a_rotation() -> None:
    """The same part, arrived in a different orientation, is the same part."""
    angle = 0.7
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    upright = describe(*_cube_mesh())
    turned = describe(*_cube_mesh(rotation=rotation))

    np.testing.assert_allclose(upright, turned, rtol=1e-4, atol=1e-4)


def test_a_wrong_width_feature_matrix_is_refused() -> None:
    x, tets = _cube_mesh()

    with pytest.raises(NeighbourIndexError, match="13-column"):
        describe(x[:, :12], tets)


def test_a_part_in_the_index_retrieves_itself_at_zero_distance() -> None:
    samples = [
        _Sample("box", *_cube_mesh(seed=i), tau=100.0 + i) for i in range(12)
    ]
    index = NeighbourIndex.build(samples)

    found = index.query(describe(samples[5].x, samples[5].tetrahedra), k=3)

    assert found.nearest_distance == pytest.approx(0.0, abs=1e-9)
    assert found.neighbours[0].index == 5
    assert not found.extrapolating


def test_neighbours_carry_the_true_solved_stress() -> None:
    """The point of retrieval: a number the reader can argue with."""
    samples = [
        _Sample("box", *_cube_mesh(seed=i), tau=2.0, force_n=500.0, length_mm=10.0)
        for i in range(12)
    ]
    index = NeighbourIndex.build(samples)

    found = index.query(describe(samples[0].x, samples[0].tetrahedra), k=2)

    # tau * F / L^2 = 2 * 500 / 100
    assert found.neighbours[0].stress_mpa == pytest.approx(10.0)


def test_a_far_away_part_is_flagged_as_extrapolating_but_not_refused() -> None:
    """Being unusual is not being wrong; a hard gate would refuse every new design."""
    samples = [
        _Sample("box", *_cube_mesh(seed=i), tau=100.0) for i in range(12)
    ]
    index = NeighbourIndex.build(samples)

    far = describe(*_cube_mesh(seed=99))
    far = far + 50.0 * index.std       # far outside the training spread

    found = index.query(far, k=3)

    assert found.extrapolating
    assert found.neighbours            # still returns them; never a refusal
    assert any("extrapolating" in note for note in found.notes())


def test_the_notes_rescale_each_neighbour_to_the_query() -> None:
    """A neighbour's own stress is not comparable: different force, different size.

    Five parts near one bracket spanned 24 to 4767 MPa against a prediction of
    418 to 794, which told the reader nothing. tau is what transfers, so it is
    converted at the query's scale.
    """
    samples = [_Sample("box", *_cube_mesh(seed=i), tau=4.0) for i in range(12)]
    index = NeighbourIndex.build(samples)
    found = index.query(describe(*_cube_mesh(seed=3)), k=2)

    # factor = F / L^2; tau 4.0 at factor 250 is 1000 MPa under this load.
    notes = found.notes(factor=250.0)

    assert "1000 MPa" in notes[0]
    assert "rescaled to this part's load" in notes[0]


def test_without_a_query_scale_the_notes_report_dimensionless_tau() -> None:
    """Better an honest tau than a stress from some other part's load."""
    samples = [_Sample("box", *_cube_mesh(seed=i), tau=4.0) for i in range(12)]
    index = NeighbourIndex.build(samples)

    notes = index.query(describe(*_cube_mesh(seed=3)), k=2).notes()

    assert "tau 4" in notes[0]
    assert "MPa" not in notes[0]


def test_an_index_round_trips_through_disk(tmp_path) -> None:
    samples = [_Sample("box", *_cube_mesh(seed=i), tau=100.0 + i) for i in range(12)]
    built = NeighbourIndex.build(samples)
    built.save(tmp_path / "n.npz")

    loaded = NeighbourIndex.load(tmp_path / "n.npz")
    query = describe(*_cube_mesh(seed=4))

    assert loaded.threshold == pytest.approx(built.threshold)
    np.testing.assert_allclose(
        [n.distance for n in loaded.query(query).neighbours],
        [n.distance for n in built.query(query).neighbours],
    )


def test_an_index_for_a_different_descriptor_is_refused(tmp_path) -> None:
    """Silently comparing against a stale layout would give plausible distances."""
    samples = [_Sample("box", *_cube_mesh(seed=i), tau=100.0) for i in range(12)]
    NeighbourIndex.build(samples).save(tmp_path / "n.npz")

    with np.load(tmp_path / "n.npz", allow_pickle=False) as data:
        stored = {k: data[k] for k in data.files}
    stored["names"] = np.array(["something", "else"])
    np.savez_compressed(tmp_path / "n.npz", **stored)

    with pytest.raises(NeighbourIndexError, match="rebuild"):
        NeighbourIndex.load(tmp_path / "n.npz")


def test_a_missing_index_says_so(tmp_path) -> None:
    with pytest.raises(NeighbourIndexError, match="no neighbour index"):
        NeighbourIndex.load(tmp_path / "absent.npz")
