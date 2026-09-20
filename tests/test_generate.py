from __future__ import annotations

import numpy as np
import pytest

from fea_surrogate.generate.families.bracket import Bracket
from fea_surrogate.generate.family import FEATURE_COLUMNS, Tagging

gmsh = pytest.importorskip("gmsh", reason="meshing requires the inference extra")

from fea_surrogate.generate.harness import draw  # noqa: E402


def _drawn(seed: int = 0):
    return draw(Bracket(), np.random.default_rng(seed))


def test_a_drawn_sample_carries_the_full_feature_schema() -> None:
    sample = _drawn()

    assert sample.x.shape[0] == sample.vertices.shape[0]
    assert sample.x.shape[1] == FEATURE_COLUMNS
    assert np.isfinite(sample.x).all()


def test_the_load_location_is_in_the_features() -> None:
    """The omission that makes SFEM unable to answer the question we ask."""
    sample = _drawn()

    assert sample.loaded_mask.any()
    assert sample.fixed_mask.any()
    # direction columns are populated on loaded nodes and zero elsewhere
    direction = sample.x[:, 9:12]
    assert np.any(direction[sample.loaded_mask] != 0.0)
    assert np.all(direction[~sample.loaded_mask] == 0.0)


def test_poisson_ratio_is_swept_and_youngs_modulus_is_not_a_feature() -> None:
    """E drops out of a force-driven linear elastic stress field; nu does not."""
    a, b = _drawn(1), _drawn(2)

    assert a.sample.poisson_ratio != b.sample.poisson_ratio
    assert 0.27 <= a.sample.poisson_ratio <= 0.42
    assert np.allclose(a.x[:, 12], a.sample.poisson_ratio)


def test_the_mesh_is_a_solid_not_a_shell() -> None:
    sample = _drawn()

    assert sample.tetrahedra.shape[1] == 4
    assert len(sample.tetrahedra) > len(sample.surface_faces) // 4


def test_the_closed_form_anchor_scales_as_the_physics_says() -> None:
    """sigma = 6FL / (W t^2): linear in force, inverse square in thickness."""
    params = {
        "length_mm": 100.0,
        "width_mm": 40.0,
        "thickness_mm": 10.0,
        "height_mm": 80.0,
    }
    base = Bracket.closed_form_peak_mpa(params, 1000.0)

    assert Bracket.closed_form_peak_mpa(params, 2000.0) == pytest.approx(2 * base)
    thicker = dict(params, thickness_mm=20.0)
    assert Bracket.closed_form_peak_mpa(thicker, 1000.0) == pytest.approx(base / 4)


def test_a_family_must_hold_and_push_on_something() -> None:
    with pytest.raises(ValueError, match="fix at least one face"):
        Tagging(fixed_faces=(), loaded_faces=(1,), load_direction=(0, 0, -1))
    with pytest.raises(ValueError, match="load at least one face"):
        Tagging(fixed_faces=(1,), loaded_faces=(), load_direction=(0, 0, -1))
    with pytest.raises(ValueError, match="nonzero"):
        Tagging(fixed_faces=(1,), loaded_faces=(2,), load_direction=(0, 0, 0))


def test_drawing_is_reproducible_from_a_seed() -> None:
    assert _drawn(7).sample.params == _drawn(7).sample.params
