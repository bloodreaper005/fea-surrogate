from __future__ import annotations

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh", reason="meshing requires the inference extra")
pytest.importorskip("scipy", reason="solving requires the inference extra")

from fea_surrogate.generate.family import Sample, Tagging  # noqa: E402
from fea_surrogate.generate.harness import (  # noqa: E402
    MeshedSample,
    _signed_distance_to_box,
    _vertex_normals,
    mesh_size_for,
)
from fea_surrogate.generate.solver import SolveError, solve  # noqa: E402


def _cantilever(length=200.0, width=40.0, thickness=10.0, force=1000.0, nu=0.3):
    """A plain box fixed at one end, loaded transversely at the other."""
    extent = np.array([length, width, thickness])
    size = mesh_size_for(extent)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("cantilever")
        gmsh.model.occ.addBox(0.0, 0.0, 0.0, length, width, thickness)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size / 4.0)
        gmsh.model.mesh.generate(3)

        tags, coords, _ = gmsh.model.mesh.getNodes()
        vertices = coords.reshape(-1, 3)
        index = {int(t): i for i, t in enumerate(tags)}
        tt, _, tn = gmsh.model.mesh.getElements(dim=3)
        tets = np.concatenate(
            [
                np.array([index[int(x)] for x in nn]).reshape(-1, 4)
                for e, nn in zip(tt, tn)
                if int(e) == 4
            ]
        )
        rt, _, rn = gmsh.model.mesh.getElements(dim=2)
        tris = np.concatenate(
            [
                np.array([index[int(x)] for x in nn]).reshape(-1, 3)
                for e, nn in zip(rt, rn)
                if int(e) == 2
            ]
        )
    finally:
        gmsh.finalize()

    fixed = vertices[:, 0] < 1e-6
    loaded = vertices[:, 0] > length - 1e-6
    n = len(vertices)
    x = np.concatenate(
        [
            vertices,
            _signed_distance_to_box(vertices),
            _vertex_normals(vertices, tris),
            fixed.reshape(-1, 1).astype(float),
            loaded.reshape(-1, 1).astype(float),
            np.tile([0.0, 0.0, -1.0], (n, 1)) * loaded.reshape(-1, 1),
            np.full((n, 1), nu),
        ],
        axis=1,
    )
    sample = Sample("cantilever", {}, Tagging((1,), (2,), (0.0, 0.0, -1.0)), nu, force)
    return MeshedSample(sample, vertices, tets, tris, x, fixed, loaded)


def test_the_solver_reproduces_the_closed_form_cantilever() -> None:
    """sigma = 6FL/(Wt^2). If the labels disagree with this, the crosscheck
    that validates the surrogate is meaningless."""
    length, width, thickness, force = 200.0, 40.0, 10.0, 1000.0
    exact = 6 * force * length / (width * thickness**2)

    result = solve(_cantilever(length, width, thickness, force))

    assert result.peak_von_mises == pytest.approx(exact, rel=0.15)


def test_the_peak_comes_from_elements_not_the_averaged_field() -> None:
    """Nodal averaging smooths away the gradient that creates the peak."""
    result = solve(_cantilever())

    assert result.peak_von_mises > result.von_mises.max()
    assert len(result.element_von_mises) == len(result.mesh.tetrahedra)


def test_stress_is_linear_in_the_applied_force() -> None:
    a = solve(_cantilever(force=1000.0))
    b = solve(_cantilever(force=3000.0))

    assert b.peak_von_mises == pytest.approx(3.0 * a.peak_von_mises, rel=0.02)


def test_the_dimensionless_target_is_invariant_to_force() -> None:
    """tau = sigma L^2 / F is what makes the model transfer across scale."""
    a = solve(_cantilever(force=500.0))
    b = solve(_cantilever(force=25_000.0))

    assert a.dimensionless_peak() == pytest.approx(b.dimensionless_peak(), rel=0.02)


def test_an_unconstrained_body_is_refused() -> None:
    mesh = _cantilever()
    free = MeshedSample(
        mesh.sample,
        mesh.vertices,
        mesh.tetrahedra,
        mesh.surface_faces,
        mesh.x,
        np.zeros(len(mesh.vertices), dtype=bool),
        mesh.loaded_mask,
    )
    with pytest.raises(SolveError, match="rigid-body freedom"):
        solve(free)


def test_an_inadmissible_poisson_ratio_is_refused() -> None:
    mesh = _cantilever(nu=0.3)
    bad = MeshedSample(
        Sample("cantilever", {}, mesh.sample.tagging, 0.5, 1000.0),
        mesh.vertices,
        mesh.tetrahedra,
        mesh.surface_faces,
        mesh.x,
        mesh.fixed_mask,
        mesh.loaded_mask,
    )
    with pytest.raises(SolveError, match="Poisson"):
        solve(bad)
