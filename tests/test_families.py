from __future__ import annotations

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh", reason="meshing requires the inference extra")
pytest.importorskip("scipy", reason="solving requires the inference extra")

from fea_surrogate.generate.families import ALL_FAMILIES  # noqa: E402
from fea_surrogate.generate.family import FEATURE_COLUMNS  # noqa: E402
from fea_surrogate.generate.harness import draw  # noqa: E402
from fea_surrogate.generate.solver import solve  # noqa: E402


@pytest.mark.parametrize("family", ALL_FAMILIES, ids=lambda f: f.name)
def test_every_family_draws_meshes_and_solves(family) -> None:
    meshed = draw(family(), np.random.default_rng(3))
    result = solve(meshed)

    assert meshed.x.shape[1] == FEATURE_COLUMNS
    assert meshed.fixed_mask.any() and meshed.loaded_mask.any()
    assert len(meshed.tetrahedra) > 0
    assert np.isfinite(result.peak_von_mises)
    assert result.peak_von_mises > 0.0
    assert np.isfinite(result.dimensionless_peak())


@pytest.mark.parametrize("family", ALL_FAMILIES, ids=lambda f: f.name)
def test_every_family_is_reproducible(family) -> None:
    a = draw(family(), np.random.default_rng(11)).sample
    b = draw(family(), np.random.default_rng(11)).sample

    assert a.params == b.params
    assert a.poisson_ratio == b.poisson_ratio


def test_orphan_nodes_do_not_make_the_solve_singular() -> None:
    """A swept helix leaves nodes in no tetrahedron; their DOFs have no
    stiffness and would return NaN rather than fail."""
    from fea_surrogate.generate.families import Spring

    meshed = draw(Spring(), np.random.default_rng(3))
    used = np.zeros(len(meshed.vertices), dtype=bool)
    used[np.unique(meshed.tetrahedra)] = True

    assert (~used).any(), "expected this mesh to contain orphan nodes"
    assert np.isfinite(solve(meshed).peak_von_mises)
