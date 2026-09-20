"""Linear elastic solve on the tetrahedral mesh a family produced.

Constant-strain tetrahedra (P1 Lagrange), which is the element order SFEM
itself used. Implemented directly rather than through FEniCSx: the element is
four pages of algebra, the dependency is a conda-or-docker affair on macOS, and
a solver we can run in the test suite is worth more here than one we cannot.

Young's modulus is fixed at 1. That is not a simplification but a consequence:
for a force-driven problem with zero-displacement constraints, `u = K^-1 F`
scales as `1/E` while `sigma = D B u` scales as `E`, so the stress field is
independent of E exactly. Poisson's ratio does not cancel, which is why it is
swept and why it is a feature.

The peak of a constant-strain field is mesh-dependent, and refining the mesh
raises it without bound at a re-entrant corner or a point load. Peak stress
from this solver is a consistent training signal, not a certified value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .harness import MeshedSample


YOUNGS_MODULUS = 1.0   # stress is independent of E; see the module docstring


class SolveError(RuntimeError):
    """The system could not be assembled or solved."""


@dataclass(frozen=True)
class SolvedSample:
    """A meshed sample with its labels filled in.

    Two stress quantities, because they are used for different things.
    `von_mises` is averaged onto nodes and is the smooth field a contour plot
    wants. `peak_von_mises` comes from the unaveraged element stresses, because
    averaging smooths away precisely the gradient that creates the peak: on a
    cantilever it converges to 0.61, 0.75, 0.84 of the analytical answer as the
    mesh refines, where the element peak reaches 0.75, 0.96, 0.98. The scalar
    the surrogate is trained and calibrated on must be the one that agrees with
    the closed form, or the crosscheck that validates it is meaningless.
    """

    mesh: MeshedSample
    displacement: np.ndarray        # (N, 3)
    von_mises: np.ndarray           # (N,) nodal, averaged — the field
    element_von_mises: np.ndarray   # (M,) per element — the peak

    @property
    def peak_von_mises(self) -> float:
        return float(self.element_von_mises.max())

    def dimensionless_peak(self) -> float:
        """tau = sigma * L^2 / F, the target the surrogate is trained on."""
        length = self.mesh.characteristic_length_mm
        return self.peak_von_mises * length**2 / self.mesh.sample.force_n


def _elasticity_matrix(poisson: float) -> np.ndarray:
    nu = float(poisson)
    if not -1.0 < nu < 0.5:
        raise SolveError(f"Poisson's ratio {nu} is outside the admissible range")
    factor = YOUNGS_MODULUS / ((1.0 + nu) * (1.0 - 2.0 * nu))
    d = np.zeros((6, 6))
    d[:3, :3] = factor * nu
    d[0, 0] = d[1, 1] = d[2, 2] = factor * (1.0 - nu)
    d[3, 3] = d[4, 4] = d[5, 5] = factor * (1.0 - 2.0 * nu) / 2.0
    return d


def _shape_gradients(corners: np.ndarray) -> tuple[np.ndarray, float]:
    """Gradients of the four P1 shape functions, and the element volume."""
    matrix = np.vstack([np.ones(4), corners.T])       # 4 x 4
    volume = np.linalg.det(matrix) / 6.0
    if abs(volume) < 1e-18:
        raise SolveError("degenerate tetrahedron")
    inverse = np.linalg.inv(matrix)
    return inverse[:, 1:], abs(volume)                # (4, 3) gradients


def _strain_displacement(gradients: np.ndarray) -> np.ndarray:
    b = np.zeros((6, 12))
    for i in range(4):
        gx, gy, gz = gradients[i]
        c = 3 * i
        b[0, c] = gx
        b[1, c + 1] = gy
        b[2, c + 2] = gz
        b[3, c] = gy
        b[3, c + 1] = gx
        b[4, c + 1] = gz
        b[4, c + 2] = gy
        b[5, c] = gz
        b[5, c + 2] = gx
    return b


def _consistent_face_loads(
    vertices: np.ndarray,
    faces: np.ndarray,
    loaded: np.ndarray,
    direction: np.ndarray,
    force_n: float,
) -> np.ndarray:
    """Spread the total force over the loaded faces as a uniform traction.

    Consistent nodal loads for a P1 triangle are one third of the face load at
    each corner, so the resultant is exactly `force_n` regardless of how the
    loaded region happens to be meshed.
    """
    on_face = loaded[faces].all(axis=1)
    selected = faces[on_face]
    if selected.size == 0:
        raise SolveError("no meshed face lies entirely in the loaded region")

    a, b, c = (vertices[selected[:, i]] for i in range(3))
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = float(areas.sum())
    if total <= 0.0:
        raise SolveError("loaded region has no area")

    loads = np.zeros_like(vertices)
    share = (force_n / total) * areas / 3.0          # per node of each face
    for i in range(3):
        np.add.at(loads, selected[:, i], share[:, None] * direction[None, :])
    return loads


def solve(sample: MeshedSample) -> SolvedSample:
    """Solve the sample and return per-node displacement and von Mises stress."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    vertices = sample.vertices
    tets = sample.tetrahedra
    n = vertices.shape[0]
    d_matrix = _elasticity_matrix(sample.sample.poisson_ratio)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    gradients_cache: list[tuple[np.ndarray, float]] = []

    for element in tets:
        gradients, volume = _shape_gradients(vertices[element])
        gradients_cache.append((gradients, volume))
        b = _strain_displacement(gradients)
        ke = volume * (b.T @ d_matrix @ b)
        dofs = np.concatenate([[3 * i, 3 * i + 1, 3 * i + 2] for i in element])
        rows.append(np.repeat(dofs, 12))
        cols.append(np.tile(dofs, 12))
        data.append(ke.reshape(-1))

    stiffness = coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(3 * n, 3 * n),
    ).tocsr()

    direction = np.array(sample.sample.tagging.load_direction, dtype=float)
    direction /= np.linalg.norm(direction)
    loads = _consistent_face_loads(
        vertices,
        sample.surface_faces,
        sample.loaded_mask,
        direction,
        sample.sample.force_n,
    )

    constrained = np.flatnonzero(sample.fixed_mask)
    if constrained.size == 0:
        # An unconstrained body has a singular stiffness matrix: six rigid-body
        # modes with no restoring force. Caught here rather than left to the
        # linear solver, which would return noise rather than fail.
        raise SolveError("no node is fixed; the body has rigid-body freedom")
    free = np.ones(3 * n, dtype=bool)
    free[np.stack([3 * constrained, 3 * constrained + 1,
                   3 * constrained + 2], axis=1).reshape(-1)] = False

    # gmsh returns nodes that belong to no tetrahedron — 153 of 1379 on a
    # swept helix. They carry no stiffness, so leaving their degrees of freedom
    # in the system makes it exactly singular and the solve returns NaN rather
    # than failing. They are in no element, so excluding them changes no stress.
    referenced = np.zeros(n, dtype=bool)
    referenced[np.unique(tets)] = True
    free &= np.repeat(referenced, 3)

    if not free.any():
        raise SolveError("every degree of freedom is constrained")

    reduced = stiffness[free][:, free]
    solution = np.zeros(3 * n)
    solution[free] = spsolve(reduced.tocsc(), loads.reshape(-1)[free])
    displacement = solution.reshape(-1, 3)

    # Element stresses are constant over each tetrahedron. Both the raw values
    # and their nodal average are kept: see SolvedSample for why the peak must
    # not be taken from the averaged field.
    element_stress = np.zeros((len(tets), 6))
    accumulated = np.zeros((n, 6))
    counts = np.zeros(n)
    for i, (element, (gradients, _)) in enumerate(zip(tets, gradients_cache)):
        b = _strain_displacement(gradients)
        stress = d_matrix @ (b @ displacement[element].reshape(-1))
        element_stress[i] = stress
        accumulated[element] += stress
        counts[element] += 1.0
    nodal = accumulated / np.maximum(counts, 1.0)[:, None]

    def _von_mises(components: np.ndarray) -> np.ndarray:
        sx, sy, sz, txy, tyz, tzx = components.T
        return np.sqrt(
            0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
            + 3.0 * (txy**2 + tyz**2 + tzx**2)
        )

    return SolvedSample(
        mesh=sample,
        displacement=displacement,
        von_mises=_von_mises(nodal),
        element_von_mises=_von_mises(element_stress),
    )


__all__ = ["SolveError", "SolvedSample", "YOUNGS_MODULUS", "solve"]
