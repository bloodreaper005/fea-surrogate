"""Linear static structural solve on 10-node tetrahedra, Ansys-Mechanical style.

This is the analysis a person reads numbers off -- Equivalent (von-Mises)
Stress and Total Deformation -- and it is deliberately a different solver from
`generate/solver.py`, which made the surrogate's training labels and must never
change. That one is P1 (constant-strain), E = 1, one uniform traction; measured
against beam theory it is 16-54% low in bending at the surrogate's mesh, which
is fine for a consistent training signal and wrong for a deformation plot.

What this one does, and why:

    element      10-node quadratic tetrahedra (SOLID187-like), 4-point Gauss rule.
                 Within 1% of beam theory with one element through a wall. 4-node
                 tetrahedra are kept for comparison only (order 1).
    material     the real E and nu. Stress does not depend on E for these loads,
                 deformation goes as 1/E, and a defaulted E would be silently
                 wrong by the ratio of the moduli.
    loads        several at once, summed into one load vector (superposition):
                 a Force spread as a uniform traction over its faces' area, a
                 Pressure normal to every point of its faces (positive pushes in,
                 the Ansys convention), and gravity as a consistent body load.
    supports     fixed (all three translations) on every node of the faces.
    results      stress at each element's own nodes, averaged per node with
                 equal weights, THEN reduced to von Mises (MAPDL AVPRIN,0 -- the
                 Mechanical default), and |u|.
    honesty      the averaged nodal maximum is the headline, as in Ansys; it is
                 flagged when it lies within one wall thickness of a
                 fixed-support edge or a sharp re-entrant edge, where linear
                 elasticity predicts infinite stress and the number grows with
                 every refinement. The maximum farther than that same distance
                 is reported beside it, and the stress extrapolated back to the
                 edge from 1-2 wall thicknesses (the hot-spot method).

Reactions are R = K u - F at the constrained degrees of freedom and must
balance the applied loads; the solve is refused rather than reported when the
system is singular, the answer is not finite or the residual is not small.

Heavy imports (scipy) live inside functions: importing this module costs numpy.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .structural_case import (
    FixedSupport,
    ForceLoad,
    Gravity,
    PressureLoad,
    StaticStructuralCase,
)
from .structural_mesh import (
    TET10_EDGES,
    TRI_QUADRATURE_POINTS,
    TRI_QUADRATURE_WEIGHTS,
    AnalysisMesh,
    surface_jacobians,
    tri_shape_functions,
)


class StructuralSolveError(RuntimeError):
    """The analysis could not be solved honestly; the message says what to change."""


#: Above this many degrees of freedom a direct solve takes most of a minute
#: and several GB (188k DOF: 19 s to factor at 2.6 GB on an M4, so 250k is
#: about 40 s and 4 GB). Callers must pass allow_large; the CLI asks first.
LARGE_DOF_THRESHOLD = 250_000

#: ||K u - F|| / ||F|| above this means the factorisation did not solve the
#: system, whatever the factorisation returned.
RESIDUAL_LIMIT = 1e-6

#: Edges between CAD faces meeting at more than this angle are "sharp".
SHARP_EDGE_DEG = 30.0

#: A sharp edge is re-entrant (concave) when the neighbouring triangle's centre
#: lies outside the first triangle's plane by more than this times the edge
#: length; the margin keeps rounding on a flat joint from counting.
CONCAVE_TOLERANCE = 1e-3

#: MAPDL /DSCALE "auto": the largest displacement is drawn at 5% of the largest
#: model extent.
AUTO_SCALE_FRACTION = 0.05

#: Elements per assembly chunk. A 10-node element's stiffness is 900 doubles,
#: its strain matrices another 720; 20k elements keep the temporaries near
#: 300 MB however large the mesh.
CHUNK_ELEMENTS = {4: 50_000, 10: 20_000}

# Gauss rules on the reference tetrahedron (volume 1/6). The 4-point rule is
# exact for the degree-2 integrand B^T D B of a straight-sided 10-node element
# and is SOLID187's own rule.
_GA, _GB = 0.5854101966249685, 0.1381966011250105
TET_GAUSS4_POINTS = np.array([[_GB, _GB, _GB], [_GA, _GB, _GB], [_GB, _GA, _GB], [_GB, _GB, _GA]])
TET_GAUSS4_WEIGHTS = np.full(4, 1.0 / 24.0)
TET_GAUSS1_POINTS = np.array([[0.25, 0.25, 0.25]])
TET_GAUSS1_WEIGHTS = np.array([1.0 / 6.0])

#: dL_i/d(xi, eta, zeta) for L0 = 1 - xi - eta - zeta, L1 = xi, L2 = eta, L3 = zeta.
_DL = np.array([[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
_CORNERS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
#: Natural coordinates of the 10 nodes, in gmsh order.
TET10_NODE_POINTS = np.vstack(
    [_CORNERS] + [0.5 * (_CORNERS[i] + _CORNERS[j]) for i, j in TET10_EDGES]
)

Progress = Callable[[str, str], None]


def estimate_solve_seconds(dof: int) -> float:
    """Rough wall time for assembly plus a direct solve of this many DOF.

    Fitted to `_direct_solve` on a 10-core M4 (the bracket sample): 33k
    equations factor in 0.6 s and 178k in 19 s -- close to quadratic in DOF
    -- plus about 15 us per DOF of assembly and recovery. Timings vary by a
    factor of ~2 with machine load; 4-node meshes solve faster than this says.
    """
    dof = max(0, int(dof))
    return 1.5e-5 * dof + 6.0e-10 * dof**2


def _direct_solve(matrix: Any, rhs: np.ndarray) -> np.ndarray:
    """Solve the reduced stiffness system K u = F (K symmetric positive definite).

    SuperLU in symmetric mode: a minimum-degree ordering of K + K^T and
    diagonal pivots, which is what an SPD matrix allows. On the bracket's fine
    mesh (178k equations) that factors in 19 s and 2.6 GB, against 61 s and
    4.2 GB for spsolve's default COLAMD with partial pivoting -- the same
    answer to 1e-10. A singular K raises RuntimeError ("Factor is exactly
    singular"); a near-singular one is caught by the residual check after.
    """
    from scipy.sparse.linalg import splu

    factor = splu(matrix, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                  options={"SymmetricMode": True})
    return factor.solve(np.asarray(rhs, dtype=float))


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AppliedLoad:
    """One load as it was actually applied: the sum of its nodal forces."""

    label: str
    kind: str                                   # "force" | "pressure" | "gravity"
    faces: tuple[int, ...]
    area_mm2: float | None                      # isoparametric area of the faces; None for gravity
    resultant_n: tuple[float, float, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "kind": self.kind,
            "faces": list(self.faces),
            "area_mm2": self.area_mm2,
            "resultant_n": list(self.resultant_n),
        }


@dataclass(frozen=True)
class SupportReaction:
    """Ansys 'Force Reaction' of one support. A node shared by two supports
    counts for the first, so the reactions still add up to the total."""

    label: str
    faces: tuple[int, ...]
    force_n: tuple[float, float, float]

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "faces": list(self.faces), "force_n": list(self.force_n)}


@dataclass(frozen=True)
class Hotspot:
    """A probed extreme: value, node and where it is."""

    value: float
    node: int
    location_mm: tuple[float, float, float]
    faces: tuple[int, ...]                      # CAD faces carrying the node; () if interior
    singular: bool = False
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "node": self.node,
            "location_mm": list(self.location_mm),
            "faces": list(self.faces),
            "singular": self.singular,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class StaticSolution:
    """A solved static structural analysis: fields at every node, and probes."""

    mesh: AnalysisMesh
    case: StaticStructuralCase
    displacement_mm: np.ndarray                 # (N,3), real E
    stress_mpa: np.ndarray                      # (N,6) [xx,yy,zz,xy,yz,zx], nodal average
    von_mises_mpa: np.ndarray                   # (N,) from the averaged components
    total_deformation_mm: np.ndarray            # (N,) |u|
    applied: tuple[AppliedLoad, ...]
    reactions: tuple[SupportReaction, ...]
    applied_total_n: tuple[float, float, float]
    reaction_total_n: tuple[float, float, float]
    equilibrium_error: float                    # |sum F + sum R| / gross applied load
    max_stress: Hotspot
    min_stress: Hotspot
    #: Max over nodes farther than exclusion_radius_mm from every flagged node;
    #: None when nothing is flagged (or no node is that far from one).
    max_stress_away: Hotspot | None
    exclusion_radius_mm: float                  # = mesh.thinnest
    max_deformation: Hotspot
    min_deformation: Hotspot
    deformation_scale_auto: float
    residual: float                             # ||K u - F|| / ||F|| on the free DOFs
    dof_count: int                              # equations solved (free DOFs)
    timings: Mapping[str, float]
    #: Nodes on a fixed-support edge or a sharp re-entrant edge (sorted).
    singular_nodes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    #: When the max is singular: the stress near it extrapolated linearly back
    #: to the flagged edge from 1-2 x exclusion_radius_mm (the hot-spot
    #: method). None when the max is not singular or there are too few points.
    max_stress_extrapolated: float | None = None

    @property
    def element_order(self) -> int:
        return self.mesh.order

    def summary(self) -> dict[str, Any]:
        """JSON-safe numbers only (plain floats, ints, strings, lists)."""
        def vec(v: Sequence[float]) -> list[float]:
            return [float(c) for c in v]

        away = self.max_stress_away
        return {
            "element_order": int(self.mesh.order),
            "nodes": int(self.mesh.node_count),
            "elements": int(self.mesh.element_count),
            "dof": int(self.dof_count),
            "equivalent_stress": {
                "units": "MPa",
                "max": float(self.max_stress.value),
                "max_location_mm": vec(self.max_stress.location_mm),
                "max_faces": list(self.max_stress.faces),
                "max_singular": bool(self.max_stress.singular),
                "max_singular_reason": self.max_stress.reason,
                "max_away_from_singularities": None if away is None else float(away.value),
                "max_away_location_mm": None if away is None else vec(away.location_mm),
                "max_extrapolated_to_singularity": (
                    None if self.max_stress_extrapolated is None
                    else float(self.max_stress_extrapolated)
                ),
                "exclusion_radius_mm": float(self.exclusion_radius_mm),
                "min": float(self.min_stress.value),
                "min_location_mm": vec(self.min_stress.location_mm),
            },
            "total_deformation": {
                "units": "mm",
                "max": float(self.max_deformation.value),
                "max_location_mm": vec(self.max_deformation.location_mm),
                "max_faces": list(self.max_deformation.faces),
                "min": float(self.min_deformation.value),
                "auto_scale": float(self.deformation_scale_auto),
            },
            "applied": [a.as_dict() for a in self.applied],
            "reactions": [r.as_dict() for r in self.reactions],
            "applied_total_n": vec(self.applied_total_n),
            "reaction_total_n": vec(self.reaction_total_n),
            "equilibrium_error": float(self.equilibrium_error),
            "residual": float(self.residual),
            "singular_node_count": int(self.singular_nodes.size),
            "timings_s": {k: float(v) for k, v in self.timings.items()},
        }


# --------------------------------------------------------------------------
# Element math (ported from the validated prototypes)
# --------------------------------------------------------------------------


def elasticity_matrix(youngs_modulus_mpa: float, poisson_ratio: float) -> np.ndarray:
    """Isotropic D in Voigt order xx, yy, zz, xy, yz, zx with engineering shear --
    the same layout as `generate/solver.py`, with the real E."""
    e = float(youngs_modulus_mpa)
    nu = float(poisson_ratio)
    factor = e / ((1.0 + nu) * (1.0 - 2.0 * nu))
    d = np.zeros((6, 6))
    d[:3, :3] = factor * nu
    d[0, 0] = d[1, 1] = d[2, 2] = factor * (1.0 - nu)
    d[3, 3] = d[4, 4] = d[5, 5] = factor * (1.0 - 2.0 * nu) / 2.0
    return d


def von_mises(stress: np.ndarray) -> np.ndarray:
    """Equivalent stress from (...,6) components [xx,yy,zz,xy,yz,zx]."""
    sx, sy, sz, txy, tyz, tzx = np.moveaxis(np.asarray(stress, dtype=float), -1, 0)
    return np.sqrt(
        0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
        + 3.0 * (txy**2 + tyz**2 + tzx**2)
    )


def tet_shape_functions(nodes_per_element: int, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """N (P,n) and dN/d(xi,eta,zeta) (P,n,3) at natural points (P,3).

    10 nodes: corners L_i(2L_i - 1), mid-edges 4 L_i L_j on TET10_EDGES.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    lam = np.column_stack([1.0 - points.sum(axis=1), points])          # (P,4)
    count = len(points)
    if nodes_per_element == 4:
        return lam, np.broadcast_to(_DL, (count, 4, 3)).copy()
    if nodes_per_element != 10:
        raise ValueError(f"tetrahedra have 4 or 10 nodes, not {nodes_per_element}")
    n = np.zeros((count, 10))
    dn = np.zeros((count, 10, 3))
    for i in range(4):
        n[:, i] = lam[:, i] * (2.0 * lam[:, i] - 1.0)
        dn[:, i, :] = (4.0 * lam[:, i] - 1.0)[:, None] * _DL[i]
    for k, (i, j) in enumerate(TET10_EDGES):
        n[:, 4 + k] = 4.0 * lam[:, i] * lam[:, j]
        dn[:, 4 + k, :] = 4.0 * (lam[:, i, None] * _DL[j] + lam[:, j, None] * _DL[i])
    return n, dn


def _gradients(nodes: np.ndarray, elements: np.ndarray,
               dn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Physical shape gradients G (m,P,n,3) and det J (m,P) at the given points."""
    xe = nodes[elements]                                       # (m,n,3)
    jac = np.einsum("pni,mnj->mpij", dn, xe)                   # J_ij = dx_j / dxi_i
    det = np.linalg.det(jac)
    with np.errstate(all="ignore"):
        inv = np.linalg.inv(jac)
    return np.einsum("mpij,pnj->mpni", inv, dn), det


def _strain_matrix(g: np.ndarray) -> np.ndarray:
    """B (...,6,3n) from physical gradients (...,n,3), engineering shear."""
    count = g.shape[-2]
    b = np.zeros(g.shape[:-2] + (6, 3 * count))
    gx, gy, gz = g[..., 0], g[..., 1], g[..., 2]
    b[..., 0, 0::3] = gx
    b[..., 1, 1::3] = gy
    b[..., 2, 2::3] = gz
    b[..., 3, 0::3] = gy
    b[..., 3, 1::3] = gx
    b[..., 4, 1::3] = gz
    b[..., 4, 2::3] = gy
    b[..., 5, 0::3] = gz
    b[..., 5, 2::3] = gx
    return b


def _strain(g: np.ndarray, ue: np.ndarray) -> np.ndarray:
    """Engineering strain (m,P,6) from gradients (m,P,n,3) and nodal u (m,n,3).

    Identical to B @ u_e, without materialising B (which for stress recovery at
    10 nodes would be 1,800 doubles per element).
    """
    h = np.einsum("mpni,mnj->mpij", g, ue)                     # h_ij = du_j / dx_i
    return np.stack(
        [h[..., 0, 0], h[..., 1, 1], h[..., 2, 2],
         h[..., 1, 0] + h[..., 0, 1], h[..., 2, 1] + h[..., 1, 2], h[..., 2, 0] + h[..., 0, 2]],
        axis=-1,
    )


def _scatter(target: np.ndarray, index: np.ndarray, values: np.ndarray) -> None:
    """target[index] += values, summing repeats (bincount: much faster than add.at)."""
    index = np.asarray(index).ravel()
    values = np.asarray(values).reshape(index.size, -1)
    n = target.shape[0]
    flat = target.reshape(n, -1)
    for c in range(flat.shape[1]):
        flat[:, c] += np.bincount(index, weights=values[:, c], minlength=n)


def _element_label(order: int) -> str:
    return "10-node" if order == 2 else "4-node"


def _assemble(
    nodes: np.ndarray, elements: np.ndarray, d: np.ndarray
) -> tuple[Any, np.ndarray, float]:
    """Global stiffness (CSR, N/mm), consistent nodal volume shares (N,), volume.

    The volume shares are integral(N_k dV) per node, so a uniform body load b
    becomes F_k = share_k * b; for a straight 10-node element that is -V/20 at
    each corner and V/5 at each mid-edge, exactly.
    """
    from scipy.sparse import coo_matrix

    count = elements.shape[1]
    points, weights = (
        (TET_GAUSS4_POINTS, TET_GAUSS4_WEIGHTS) if count == 10
        else (TET_GAUSS1_POINTS, TET_GAUSS1_WEIGHTS)
    )
    shape, dn = tet_shape_functions(count, points)
    size = 3 * len(nodes)
    ndof = 3 * count
    local = np.arange(3)
    stiffness = None
    shares = np.zeros(len(nodes))
    volume = 0.0
    chunk = CHUNK_ELEMENTS[count]
    for start in range(0, len(elements), chunk):
        block = elements[start:start + chunk]
        g, det = _gradients(nodes, block, dn)
        bad = ~(det > 0.0)
        if bad.any():
            worst = int(np.flatnonzero(bad.any(axis=1))[0]) + start
            raise StructuralSolveError(
                f"{int(bad.any(axis=1).sum())} {_element_label(1 if count == 4 else 2)} "
                f"elements have a non-positive Jacobian (first: element {worst}), so "
                "their stiffness would be meaningless. Re-mesh with another "
                "refinement or element size; mesh_for_analysis normally prevents this."
            )
        wdet = det * weights                                    # (m,P)
        b = _strain_matrix(g)                                   # (m,P,6,ndof)
        ke = np.einsum("mp,mpki,kl,mplj->mij", wdet, b, d, b, optimize=True)
        dofs = (3 * block[:, :, None] + local).reshape(len(block), ndof)
        rows = np.repeat(dofs, ndof, axis=1).ravel()
        cols = np.tile(dofs, (1, ndof)).ravel()
        part = coo_matrix((ke.ravel(), (rows, cols)), shape=(size, size)).tocsr()
        stiffness = part if stiffness is None else stiffness + part
        _scatter(shares, block, np.einsum("mp,pn->mn", wdet, shape))
        volume += float(wdet.sum())
    return stiffness, shares, volume


def _surface_load(
    nodes: np.ndarray, triangles: np.ndarray, *,
    total_force: np.ndarray | None = None, pressure: float | None = None,
) -> tuple[np.ndarray, float]:
    """Consistent nodal forces (N,3) on outward triangles, and their area.

    total_force: spread as the uniform traction total/A, integrated as
    N_k * t * |J|. A comes from the same quadrature, so the nodal forces sum
    to `total_force` exactly (the shape functions partition unity); on a flat
    6-node triangle the corners get 0 and each mid-edge A/3.
    pressure: -p * outward normal at every quadrature point, which on a curved
    6-node triangle follows the curvature; the 6-point rule is exact for it.
    """
    n, _ = tri_shape_functions(triangles.shape[1], TRI_QUADRATURE_POINTS)
    jac = surface_jacobians(nodes, triangles)                   # (T,q,3) = n dA
    w = TRI_QUADRATURE_WEIGHTS
    da = np.linalg.norm(jac, axis=2)                            # (T,q)
    area = float(np.einsum("q,tq->", w, da))
    forces = np.zeros_like(nodes)
    if total_force is not None and area > 0.0:
        traction = np.asarray(total_force, dtype=float) / area
        per_node = np.einsum("q,tq,qn->tn", w, da, n)           # integral(N_k dA)
        _scatter(forces, triangles, per_node[..., None] * traction[None, None, :])
    if pressure is not None:
        per_node = -float(pressure) * np.einsum("q,qn,tqj->tnj", w, n, jac)
        _scatter(forces, triangles, per_node)
    return forces, area


def _recover_stress(nodes: np.ndarray, elements: np.ndarray, d: np.ndarray,
                    u: np.ndarray) -> np.ndarray:
    """Nodal stress (N,6): each element's stress AT its own nodes, then the
    unweighted mean over the elements sharing a node.

    For a straight 10-node element the strain is linear, so evaluating at the
    corners equals Ansys' linear extrapolation from the Gauss points; a 4-node
    element's stress is constant and the mean is the training solver's.
    """
    count = elements.shape[1]
    points = TET10_NODE_POINTS if count == 10 else TET_GAUSS1_POINTS
    _, dn = tet_shape_functions(count, points)
    total = np.zeros((len(nodes), 6))
    chunk = CHUNK_ELEMENTS[count]
    for start in range(0, len(elements), chunk):
        block = elements[start:start + chunk]
        g, det = _gradients(nodes, block, dn)
        if not np.all(det > 0.0):
            raise StructuralSolveError(
                "an element's Jacobian vanishes at one of its nodes, so its stress "
                "there is undefined. Re-mesh with another refinement or element size."
            )
        sigma = _strain(g, u[block]) @ d.T                      # (m,P,6)
        if count == 4:
            sigma = np.repeat(sigma, 4, axis=1)
        _scatter(total, block, sigma)
    counts = np.bincount(elements.ravel(), minlength=len(nodes)).astype(float)
    return total / np.maximum(counts, 1.0)[:, None]


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def _check_inputs(mesh: AnalysisMesh, case: StaticStructuralCase) -> None:
    if mesh.order not in (1, 2) or mesh.elements.shape[1] != (10 if mesh.order == 2 else 4):
        raise StructuralSolveError(
            f"the mesh claims order {mesh.order} but its elements have "
            f"{mesh.elements.shape[1]} nodes; rebuild it with mesh_for_analysis"
        )
    if case.mesh.element_order != mesh.order:
        raise StructuralSolveError(
            f"the case asks for element order {case.mesh.element_order} but the mesh "
            f"is order {mesh.order}; re-mesh with mesh_for_analysis(path, "
            f"order={case.mesh.element_order}) so the results say what was solved"
        )
    material = case.material
    e, nu = material.youngs_modulus_mpa, material.poisson_ratio
    if not (isinstance(e, (int, float)) and math.isfinite(e) and e > 0):
        raise StructuralSolveError(
            f"Young's modulus must be a positive number of MPa, not {e!r}; e.g. "
            "200000 for steel"
        )
    if not (isinstance(nu, (int, float)) and -1.0 < nu < 0.5):
        raise StructuralSolveError(
            f"Poisson's ratio must lie between -1 and 0.5 (exclusive), not {nu!r}; "
            "metals are about 0.3"
        )
    if case.gravity is not None and not (material.density_kg_m3 or 0) > 0:
        raise StructuralSolveError(
            "gravity needs the material's density (kg/m^3); give density_kg_m3 or "
            "disable gravity"
        )
    if not case.supports:
        raise StructuralSolveError(
            "no fixed support: without one the part is free to move as a rigid body "
            "and has no static solution. Fix at least one face."
        )
    if not case.loads and case.gravity is None:
        raise StructuralSolveError("the case has no load; add a force, a pressure or gravity")
    known = set(mesh.face_nodes)
    for letter, item in case.lettered():
        if isinstance(item, Gravity):
            continue
        unknown = sorted(set(item.faces) - known)
        if unknown:
            raise StructuralSolveError(
                f"{letter} ({item.label}) names face(s) {', '.join(map(str, unknown))}, "
                f"which this mesh does not have; its faces are "
                f"{', '.join(map(str, sorted(known)))}. Use the face table's numbers."
            )


def _refuse_floating_bodies(elements: np.ndarray, node_count: int, fixed: np.ndarray) -> None:
    """Refuse when a connected body carries no fixed node.

    Same reasoning as `generate/solver._refuse_unconstrained_bodies`: each free
    body brings six rigid-body modes, the system is singular and anything a
    solver returns from it is noise (an 18-body assembly once 'solved' to 1031
    MPa that way).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    count = elements.shape[1]
    first = np.repeat(elements[:, :1], count - 1, axis=1).ravel()
    rest = elements[:, 1:].ravel()
    graph = coo_matrix((np.ones(first.size), (first, rest)), shape=(node_count, node_count))
    _, labels = connected_components(graph, directed=False)
    used = np.zeros(node_count, dtype=bool)
    used[elements.ravel()] = True
    bodies = np.unique(labels[used])
    held = np.unique(labels[used & fixed])
    loose = np.setdiff1d(bodies, held)
    if loose.size == 0:
        return
    floating = int(np.isin(labels[used], loose).sum())
    raise StructuralSolveError(
        f"this mesh holds {bodies.size} disconnected bodies and {loose.size} of them "
        f"are not held anywhere, which is {6 * loose.size} rigid-body modes and "
        f"{floating} of {int(used.sum())} nodes with no restoring force. The linear "
        "system is singular and any answer from it would be noise. Solve the bodies "
        "one at a time, or fix a face on every one of them."
    )


# --------------------------------------------------------------------------
# Singularities
# --------------------------------------------------------------------------


def _faces_phrase(tags: Sequence[int]) -> str:
    return ("face " if len(tags) == 1 else "faces ") + ", ".join(map(str, tags))


def _singular_features(mesh: AnalysisMesh, case: StaticStructuralCase) -> dict[int, str]:
    """Nodes where linear elasticity predicts unbounded stress, with a reason each.

    (a) the edge of a fixed support: a node on a fixed face AND on a face that
        is not fixed -- the clamp corner every cantilever has.
    (b) a sharp re-entrant CAD edge: two boundary triangles of different faces
        meeting at more than 30 degrees with the second on the outward side of
        the first (an inside corner; an outside corner is not singular).
    """
    flagged: dict[int, str] = {}
    lettered = [(letter, s) for letter, s in case.lettered() if isinstance(s, FixedSupport)]
    fixed_faces = set(case.fixed_faces())
    on_free_face = np.zeros(mesh.node_count, dtype=bool)
    for tag, members in mesh.face_nodes.items():
        if tag not in fixed_faces:
            on_free_face[members] = True
    for letter, support in lettered:
        reason = f"edge of {support.label} {letter} on {_faces_phrase(support.faces)}"
        for tag in support.faces:
            for node in mesh.face_nodes[tag][on_free_face[mesh.face_nodes[tag]]].tolist():
                flagged.setdefault(int(node), reason)

    tris = mesh.boundary
    if len(tris) == 0:
        return flagged
    corner_pairs = [(0, 1), (1, 2), (0, 2)]
    pairs = np.concatenate([tris[:, list(p)] for p in corner_pairs], axis=0)
    owner = np.tile(np.arange(len(tris)), 3)
    side = np.repeat(np.arange(3), len(tris))
    lo, hi = np.minimum(pairs[:, 0], pairs[:, 1]), np.maximum(pairs[:, 0], pairs[:, 1])
    key = lo * np.int64(mesh.node_count) + hi
    order = np.argsort(key, kind="stable")
    _, starts, counts = np.unique(key[order], return_index=True, return_counts=True)
    # Only edges shared by exactly two triangles: a manifold edge of the solid.
    starts = starts[counts == 2]
    if starts.size == 0:
        return flagged
    first = order[starts]
    t1, t2 = owner[first], owner[order[starts + 1]]
    different = mesh.boundary_face[t1] != mesh.boundary_face[t2]
    first, t1, t2 = first[different], t1[different], t2[different]
    if first.size == 0:
        return flagged

    x = mesh.nodes
    def unit_normal(t: np.ndarray) -> np.ndarray:
        a, b, c = (x[tris[t, i]] for i in range(3))
        n = np.cross(b - a, c - a)
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)

    n1, n2 = unit_normal(t1), unit_normal(t2)
    sharp = np.einsum("ij,ij->i", n1, n2) < math.cos(math.radians(SHARP_EDGE_DEG))
    a_node, b_node = lo[first], hi[first]
    length = np.linalg.norm(x[b_node] - x[a_node], axis=1)
    centre2 = x[tris[t2, :3]].mean(axis=1)
    outside = np.einsum("ij,ij->i", centre2 - x[a_node], n1) > CONCAVE_TOLERANCE * length
    reentrant = np.flatnonzero(sharp & outside)
    for i in reentrant.tolist():
        fa, fb = sorted((int(mesh.boundary_face[t1[i]]), int(mesh.boundary_face[t2[i]])))
        reason = f"re-entrant edge between faces {fa} and {fb}"
        edge_nodes = [int(a_node[i]), int(b_node[i])]
        if tris.shape[1] == 6:
            edge_nodes.append(int(tris[t1[i], 3 + side[first[i]]]))
        for node in edge_nodes:
            flagged.setdefault(node, reason)
    return flagged


def _hotspot(mesh: AnalysisMesh, values: np.ndarray, node: int, **flags: Any) -> Hotspot:
    return Hotspot(
        value=float(values[node]),
        node=int(node),
        location_mm=tuple(float(c) for c in mesh.nodes[node]),
        faces=mesh.faces_at(node),
        **flags,
    )


# --------------------------------------------------------------------------
# Solve
# --------------------------------------------------------------------------


def solve_static(
    mesh: AnalysisMesh,
    case: StaticStructuralCase,
    *,
    allow_large: bool = False,
    progress: Progress | None = None,
) -> StaticSolution:
    """Solve the case on the mesh: one assembly, one load vector, one direct solve.

    Raises StructuralSolveError (with what to change) when the mesh is too
    large without `allow_large`, a body floats free, the system is singular,
    or the answer is not finite or does not satisfy K u = F.
    """
    from scipy.sparse.linalg import MatrixRankWarning
    from scipy.spatial import cKDTree

    def say(stage: str, message: str) -> None:
        if progress is not None:
            progress(stage, message)

    started = time.perf_counter()
    timings: dict[str, float] = {}
    _check_inputs(mesh, case)
    if mesh.dof_count > LARGE_DOF_THRESHOLD and not allow_large:
        raise StructuralSolveError(
            f"this mesh has {mesh.dof_count:,} degrees of freedom, above the "
            f"{LARGE_DOF_THRESHOLD:,} where a direct solve takes minutes and several "
            f"GB of memory (estimate {estimate_solve_seconds(mesh.dof_count):.0f} s). "
            "Pass allow_large to go ahead (CLI: --allow-large), or use a coarser "
            "refinement or a larger element size."
        )

    nodes = np.asarray(mesh.nodes, dtype=float)
    elements = np.asarray(mesh.elements, dtype=np.int64)
    n = len(nodes)
    material = case.material
    d = elasticity_matrix(material.youngs_modulus_mpa, material.poisson_ratio)

    # ---- supports ----------------------------------------------------------
    fixed = np.zeros(n, dtype=bool)
    support_nodes = []
    for support in case.supports:
        members = np.unique(np.concatenate([mesh.face_nodes[t] for t in support.faces]))
        support_nodes.append(members)
        fixed[members] = True
    used = np.zeros(n, dtype=bool)
    used[elements.ravel()] = True
    if not (fixed & used).any():
        raise StructuralSolveError(
            "the fixed faces carry no mesh nodes, so nothing is held; check the "
            "support's face numbers against the face table"
        )
    _refuse_floating_bodies(elements, n, fixed)

    # ---- stiffness -----------------------------------------------------------
    clock = time.perf_counter()
    say("assemble", f"assembling {len(elements):,} {_element_label(mesh.order)} elements "
                    f"({mesh.dof_count:,} DOF)")
    stiffness, shares, volume = _assemble(nodes, elements, d)
    timings["assemble"] = time.perf_counter() - clock

    # ---- loads ---------------------------------------------------------------
    clock = time.perf_counter()
    say("loads", f"applying {len(case.loads) + (case.gravity is not None)} load(s)")
    forces = np.zeros((n, 3))
    applied: list[AppliedLoad] = []
    gross = 0.0
    for letter, item in case.lettered():
        if isinstance(item, FixedSupport):
            continue
        if isinstance(item, Gravity):
            density = float(material.density_kg_m3)                    # kg/m^3
            body = density * 1e-9 * np.asarray(item.acceleration_mps2)  # N/mm^3
            part = shares[:, None] * body[None, :]
            applied.append(AppliedLoad(item.label, "gravity", (), None,
                                       tuple(float(c) for c in part.sum(axis=0))))
            gross += float(np.linalg.norm(body)) * volume
            forces += part
            continue
        triangles = mesh.face_triangles(item.faces)
        if len(triangles) == 0:
            raise StructuralSolveError(
                f"{letter} ({item.label}) is on {_faces_phrase(item.faces)}, which has no "
                "surface mesh to carry it; pick another face"
            )
        if isinstance(item, ForceLoad):
            part, area = _surface_load(nodes, triangles, total_force=np.asarray(item.vector_n))
            if not area > 0.0:
                raise StructuralSolveError(
                    f"{letter} ({item.label}): {_faces_phrase(item.faces)} has no area"
                )
            gross += float(item.magnitude_n)
            kind = "force"
        elif isinstance(item, PressureLoad):
            part, area = _surface_load(nodes, triangles, pressure=item.pressure_mpa)
            gross += abs(float(item.pressure_mpa)) * area
            kind = "pressure"
        else:  # pragma: no cover - the case validates its own load types
            raise StructuralSolveError(f"unsupported load {type(item).__name__}")
        applied.append(AppliedLoad(item.label, kind, tuple(item.faces), float(area),
                                   tuple(float(c) for c in part.sum(axis=0))))
        forces += part
    timings["loads"] = time.perf_counter() - clock

    # ---- solve ---------------------------------------------------------------
    clock = time.perf_counter()
    free_nodes = used & ~fixed
    free = np.flatnonzero(np.repeat(free_nodes, 3))
    if free.size == 0:
        raise StructuralSolveError(
            "every node is fixed, so nothing can deform; fix fewer faces"
        )
    rhs = forces.reshape(-1)[free]
    if not np.any(rhs):
        raise StructuralSolveError(
            "every loaded node is also fixed, so the loads go straight into the "
            "supports and nothing deforms. Load a face that is free to move."
        )
    say("solve", f"solving {free.size:,} equations "
                 f"(estimate {estimate_solve_seconds(free.size):.0f} s)")
    reduced = stiffness[free][:, free].tocsc()
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        try:
            solution = _direct_solve(reduced, rhs)
        except (MatrixRankWarning, RuntimeError) as exc:
            if isinstance(exc, RuntimeError) and "singular" not in str(exc).lower():
                raise StructuralSolveError(
                    f"the sparse factorisation failed ({exc}); re-mesh with another "
                    "refinement or element size"
                ) from None
            raise StructuralSolveError(
                "the stiffness matrix is singular: some part can still move as a "
                "rigid body. Fix a face on every body, or check that the fixed faces "
                "hold the part (a single line or point of support is not enough)."
            ) from None
    solution = np.asarray(solution, dtype=float).reshape(-1)
    if not np.all(np.isfinite(solution)):
        raise StructuralSolveError(
            "the solve returned non-finite displacements: the system is singular or "
            "badly conditioned. Check the supports hold every body, and that the "
            "material values are physical."
        )
    rhs_norm = float(np.linalg.norm(rhs))
    residual = float(np.linalg.norm(reduced @ solution - rhs) / rhs_norm)
    if not residual <= RESIDUAL_LIMIT:
        raise StructuralSolveError(
            f"the solution does not satisfy K u = F (relative residual {residual:.2e} "
            f"> {RESIDUAL_LIMIT:g}); the system is too badly conditioned to trust. "
            "Re-mesh with another refinement, and check the material values."
        )
    u_flat = np.zeros(3 * n)
    u_flat[free] = solution
    displacement = u_flat.reshape(n, 3)

    reaction = (stiffness @ u_flat - forces.reshape(-1)).reshape(n, 3)
    taken = np.zeros(n, dtype=bool)
    reactions = []
    for support, members in zip(case.supports, support_nodes):
        mine = members[~taken[members]]
        taken[mine] = True
        reactions.append(SupportReaction(support.label, tuple(support.faces),
                                         tuple(float(c) for c in reaction[mine].sum(axis=0))))
    applied_total = forces.sum(axis=0)
    reaction_total = reaction[fixed].sum(axis=0)
    equilibrium = float(np.linalg.norm(applied_total + reaction_total) / max(gross, 1e-30))
    timings["solve"] = time.perf_counter() - clock

    # ---- results -------------------------------------------------------------
    clock = time.perf_counter()
    say("recover", "recovering nodal stress")
    stress = _recover_stress(nodes, elements, d, displacement)
    vm = von_mises(stress)
    total = np.linalg.norm(displacement, axis=1)
    candidates = np.flatnonzero(used)

    max_node = int(candidates[np.argmax(vm[candidates])])
    min_node = int(candidates[np.argmin(vm[candidates])])
    flagged = _singular_features(mesh, case)
    singular_nodes = np.array(sorted(flagged), dtype=np.int64)
    # One radius answers both questions -- "is the max at a flagged feature?"
    # and "which nodes are away from one?" -- so no point can be both. (The
    # flag once used the element size: on a coarse mesh a point 4.9 mm from a
    # re-entrant edge was called singular AND "farther than 3.53 mm" from it.)
    radius = float(mesh.thinnest)
    max_flags: dict[str, Any] = {}
    away = None
    extrapolated = None
    if singular_nodes.size:
        tree = cKDTree(nodes[singular_nodes])
        distance, nearest = tree.query(nodes[max_node])
        if distance <= radius:
            reason = flagged[int(singular_nodes[nearest])]
            if distance > 0.05 * radius:
                reason += f", {_short(distance)} mm away"
            max_flags = {"singular": True, "reason": reason}
        far, _ = tree.query(nodes[candidates])
        outside = candidates[far > radius]
        if outside.size:
            away = _hotspot(mesh, vm, int(outside[np.argmax(vm[outside])]))
        # The hot-spot line needs nodes spread through 1-2 wall thicknesses; an
        # element larger than that leaves a handful of unrelated nodes in the
        # band (a 20 mm mesh "extrapolated" 66.8 MPa beside a 312 MPa max).
        if max_flags and float(mesh.element_size_mm) <= HOT_SPOT_MAX_ELEMENT_RATIO * radius:
            extrapolated = _hot_spot_stress(
                nodes[candidates], vm[candidates], far,
                centre=nodes[max_node], radius=radius,
                ceiling=float(vm[max_node]),
            )
    max_u = int(candidates[np.argmax(total[candidates])])
    min_u = int(candidates[np.argmin(total[candidates])])
    peak_u = float(total[max_u])
    largest_extent = float(np.max(nodes[candidates].max(axis=0) - nodes[candidates].min(axis=0)))
    scale = AUTO_SCALE_FRACTION * largest_extent / peak_u if peak_u > 0.0 else 1.0
    timings["recover"] = time.perf_counter() - clock
    timings["total"] = time.perf_counter() - started

    return StaticSolution(
        mesh=mesh,
        case=case,
        displacement_mm=displacement,
        stress_mpa=stress,
        von_mises_mpa=vm,
        total_deformation_mm=total,
        applied=tuple(applied),
        reactions=tuple(reactions),
        applied_total_n=tuple(float(c) for c in applied_total),
        reaction_total_n=tuple(float(c) for c in reaction_total),
        equilibrium_error=equilibrium,
        max_stress=_hotspot(mesh, vm, max_node, **max_flags),
        min_stress=_hotspot(mesh, vm, min_node),
        max_stress_away=away,
        exclusion_radius_mm=float(mesh.thinnest),
        max_deformation=_hotspot(mesh, total, max_u),
        min_deformation=_hotspot(mesh, total, min_u),
        deformation_scale_auto=float(scale),
        residual=residual,
        dof_count=int(free.size),
        timings=timings,
        singular_nodes=singular_nodes,
        max_stress_extrapolated=extrapolated,
    )


#: Distances from the flagged edge, as multiples of the wall thickness, over
#: which the hot-spot extrapolation reads the stress. It starts one thickness
#: out, past the support's three-dimensional zone (St-Venant: about one
#: thickness), where the section carries the beam's own stress gradient.
#: Measured on clamped cantilevers against beam theory: within 3 % on medium
#: and fine meshes, where 0.4-1.6 t swung from +1.4 % to -4.6 %.
HOT_SPOT_RANGE = (1.0, 2.0)
HOT_SPOT_BINS = 6
#: The extrapolation is attempted only when the element size is at most this
#: many wall thicknesses ("medium" and finer on most parts; not "coarse",
#: which is 1.6), so the band holds a line of nodes rather than a few strays.
HOT_SPOT_MAX_ELEMENT_RATIO = 1.25


def _short(value: float) -> str:
    """2 significant figures, no exponent for the distances printed here."""
    return f"{float(value):.2g}" if float(value) >= 0.01 else "0"


def _hot_spot_stress(
    points: np.ndarray,
    values: np.ndarray,
    distance: np.ndarray,
    *,
    centre: np.ndarray,
    radius: float,
    ceiling: float,
) -> float | None:
    """The stress at a singular edge, read the way a fatigue engineer reads a
    weld toe: extrapolate the surface stress linearly back to the edge.

    A singular max rises without bound under refinement, and the max one wall
    thickness away leaves out the section's real stress -- on a clamped
    cantilever that is the root bending stress, 10-17 % above the value one
    thickness out. Neither is a number to compare with yield. Fitting a line to
    the highest stress at each distance from the edge, between one and two
    wall thicknesses and within three thicknesses of the max, and reading it at
    the edge recovers the root stress (a straight line in bending is
    extrapolated exactly) without the singular spike.

    Returns None when fewer than three distance bins hold a point; never more
    than the singular max itself, never less than the stress at the near end.
    """
    if radius <= 0.0:
        return None
    near = np.linalg.norm(points - centre, axis=1) <= 3.0 * radius
    lo, hi = HOT_SPOT_RANGE
    edges = np.linspace(lo * radius, hi * radius, HOT_SPOT_BINS + 1)
    xs: list[float] = []
    ys: list[float] = []
    for a, b in zip(edges[:-1], edges[1:]):
        inside = near & (distance >= a) & (distance < b)
        if inside.any():
            k = int(np.argmax(np.where(inside, values, -np.inf)))
            xs.append(float(distance[k]))
            ys.append(float(values[k]))
    if len(xs) < 3:
        return None
    slope, intercept = np.polyfit(np.asarray(xs), np.asarray(ys), 1)
    floor = ys[0]
    return float(min(max(intercept, floor), ceiling))


__all__ = [
    "AUTO_SCALE_FRACTION",
    "LARGE_DOF_THRESHOLD",
    "RESIDUAL_LIMIT",
    "AppliedLoad",
    "Hotspot",
    "StaticSolution",
    "StructuralSolveError",
    "SupportReaction",
    "elasticity_matrix",
    "estimate_solve_seconds",
    "solve_static",
    "tet_shape_functions",
    "von_mises",
]
