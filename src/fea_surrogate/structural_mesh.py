"""Mesh a CAD part for a static structural analysis: 10-node tetrahedra by default.

This is NOT the surrogate's mesh. `features.mesh_step` is what the model was fitted
on -- linear tetrahedra, float32 coordinates, about four elements through the
thinnest wall, no per-face triangles -- and it must stay exactly as it is or the
calibration no longer describes the model. An analysis needs different things:

    quadratic elements     Linear (constant-strain) tetrahedra are far too stiff
                           in bending. Measured on the shipped bracket at the
                           surrogate's mesh: tip deflection -38.5% and shelf
                           bending stress -54% against beam theory. 10-node
                           tetrahedra (what Ansys Mechanical uses by default,
                           SOLID187) are within 1% with ONE element through the
                           wall, so the analysis mesh can be coarser and still
                           far more accurate.
    per-face triangles     A pressure acts normal to each point of the face it
                           is applied to, and a force is spread over that face's
                           own area. Rebuilding loaded triangles from a node mask
                           (the training solver's rule) leaks load onto a narrow
                           neighbouring face: 124 of 132 triangles of an
                           unselected 0.15 mm fillet were picked up in one test.
    outward winding        A pressure's sign depends on it. gmsh's winding was
                           outward on every triangle measured, but that is an
                           observation, not a contract, so every triangle is
                           checked against the tetrahedron that owns it.
    CAD edges              The renderer draws the part's real edges, and the
                           singularity flags need to know where the faces meet.

Face numbers are gmsh OCC surface tags from `cad_intake.import_shapes`, the same
import `cad_intake.inspect` uses, so the numbers in the face table are the
numbers a load case names. Field names (`vertices`, `face_nodes`,
`face_centroids`, `thinnest`) match `features.ScaledMesh`, so
`face_selectors.select_faces` and `face_mask` work on an AnalysisMesh unchanged.

Heavy imports (gmsh) live inside functions: importing this module costs numpy.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


class AnalysisMeshError(RuntimeError):
    """The part could not be meshed into a valid analysis mesh; the message says what to try."""


#: refinement -> (multiple of the thinnest dimension, smallest size as a fraction
#: of the CAD diagonal, largest size as a fraction of the CAD diagonal).
#:
#: "medium" puts about one 10-node element through the thinnest wall, which was
#: measured within ~1% of beam theory in bending (cantilever: tip -0.56%,
#: mid-span stress -0.20% at one element through; bracket shelf: -1.0%). The
#: bounds keep slender or stubby parts from producing absurd meshes. These are
#: NOT the surrogate's rule (thinnest/4 clipped to diag/80..diag/8) and must not
#: be confused with it.
REFINEMENTS: dict[str, tuple[float, float, float]] = {
    "coarse": (1.6, 1.0 / 40.0, 1.0 / 6.0),
    "medium": (1.0, 1.0 / 60.0, 1.0 / 8.0),
    "fine": (0.5, 1.0 / 100.0, 1.0 / 10.0),
}

#: gmsh refines toward curved surfaces until each element spans at most 2*pi/N
#: of curvature -- the equivalent of Ansys "Capture Curvature". 12 gives a hole
#: at least 12 quadratic edges around, which resolves its stress concentration
#: (plate-with-hole Kt within ~3% at that density, measured).
CURVATURE_ELEMENTS_PER_2PI = 12

#: Smallest element gmsh may use, as a fraction of the target size. The same
#: ratio the surrogate's mesher uses; it bounds curvature refinement.
MESH_SIZE_MIN_FRACTION = 0.25

#: A face is planar when no triangle normal deviates from the face's mean
#: normal by more than this. Decided from the mesh because OpenCASCADE calls
#: every face of a B-spline export "curved", flat ones included (7 of 21 on
#: component8.step).
PLANAR_TOLERANCE_DEG = 0.5

#: A non-planar face is "cylinder-like" when all its normals are perpendicular
#: to one axis within this angle. Display only: the face table says "cyl".
CYLINDER_TOLERANCE_DEG = 3.0

#: Tetrahedra per (element size)^3 of volume that gmsh produces, measured (~6
#: on a plate with a hole). Used only to refuse a size that would mesh for
#: minutes into something no direct solver could factor.
_ELEMENTS_PER_SIZE_CUBED = 6.0
MAX_ESTIMATED_ELEMENTS = 3_000_000

#: gmsh element types.
_TET4, _TET10, _TRI3, _TRI6, _LINE2, _LINE3 = 4, 11, 2, 9, 1, 8

#: gmsh "Tetrahedron 10" mid-edge node order (verified against node coordinates
#: on both sample parts): node 4+k sits on corner edge TET10_EDGES[k]. VTK puts
#: (1,3) before (2,3), so an exporter must swap nodes 8 and 9.
TET10_EDGES: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (0, 2), (0, 3), (2, 3), (1, 3))
#: gmsh "Triangle 6": node 3+k sits on corner edge TRI6_EDGES[k].
TRI6_EDGES: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (0, 2))

# Dunavant's 6-point, degree-4 rule on the reference triangle (area 1/2). A
# 6-node triangle's area vector x_s cross x_t is quadratic, so a pressure (and
# its resultant) integrates exactly; shape function times area vector is degree
# 4, which this rule is exact for too.
_QA, _QB = 0.445948490915965, 0.091576213509771
TRI_QUADRATURE_POINTS = np.array(
    [[_QA, _QA], [1.0 - 2.0 * _QA, _QA], [_QA, 1.0 - 2.0 * _QA],
     [_QB, _QB], [1.0 - 2.0 * _QB, _QB], [_QB, 1.0 - 2.0 * _QB]]
)
TRI_QUADRATURE_WEIGHTS = 0.5 * np.array(
    [0.223381589678011] * 3 + [0.109951743655322] * 3
)


# --------------------------------------------------------------------------
# Triangle shape functions and surface integrals (shared with the solver)
# --------------------------------------------------------------------------


def tri_shape_functions(nodes_per_triangle: int, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shape functions N (q,n) and their (s,t) derivatives (q,n,2) at `points` (q,2).

    3 nodes: linear. 6 nodes: quadratic in gmsh "Triangle 6" order, with
    L0 = 1-s-t, L1 = s, L2 = t; corners L(2L-1), mid-edges 4 Li Lj.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    s, t = points[:, 0], points[:, 1]
    l0, l1, l2 = 1.0 - s - t, s, t
    zeros, ones = np.zeros_like(s), np.ones_like(s)
    if nodes_per_triangle == 3:
        n = np.stack([l0, l1, l2], axis=1)
        ds = np.stack([-ones, ones, zeros], axis=1)
        dt = np.stack([-ones, zeros, ones], axis=1)
    elif nodes_per_triangle == 6:
        n = np.stack(
            [l0 * (2 * l0 - 1), l1 * (2 * l1 - 1), l2 * (2 * l2 - 1),
             4 * l0 * l1, 4 * l1 * l2, 4 * l0 * l2],
            axis=1,
        )
        ds = np.stack(
            [-(4 * l0 - 1), 4 * l1 - 1, zeros, 4 * (l0 - l1), 4 * l2, -4 * l2], axis=1
        )
        dt = np.stack(
            [-(4 * l0 - 1), zeros, 4 * l2 - 1, -4 * l1, 4 * l1, 4 * (l0 - l2)], axis=1
        )
    else:
        raise ValueError(f"triangles have 3 or 6 nodes, not {nodes_per_triangle}")
    return n, np.stack([ds, dt], axis=2)


def surface_jacobians(nodes: np.ndarray, triangles: np.ndarray,
                      points: np.ndarray = TRI_QUADRATURE_POINTS) -> np.ndarray:
    """x_s cross x_t at each point of each triangle, (T,q,3).

    Its length is the area Jacobian and its direction the normal, so with
    outward-wound triangles it is the outward area vector density: a pressure's
    nodal loads are -p * sum_q w_q N_k(q) * this, with no normalising needed.
    """
    _, dn = tri_shape_functions(triangles.shape[1], points)
    xt = nodes[triangles]                                   # (T,n,3)
    d_s = np.einsum("qn,tnj->tqj", dn[:, :, 0], xt)
    d_t = np.einsum("qn,tnj->tqj", dn[:, :, 1], xt)
    return np.cross(d_s, d_t)


def triangle_areas(nodes: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Isoparametric area of each triangle (T,): exact for flat ones."""
    jac = surface_jacobians(nodes, triangles)
    return np.einsum("q,tq->t", TRI_QUADRATURE_WEIGHTS, np.linalg.norm(jac, axis=2))


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FaceSummary:
    """A CAD face as the mesh sees it.

    `planar` and `normal` come from the mesh, not from OpenCASCADE: OCC calls
    every B-spline face curved, and its single sampled normal can come from a
    parameter point that is not even on a trimmed face. `area` and `centroid`
    are OpenCASCADE's (exact, and the same numbers `cad_intake.inspect` prints);
    the bounding box is the mesh's, because OCC's overshoots curved faces.
    """

    tag: int
    area: float
    centroid: tuple[float, float, float]
    normal: tuple[float, float, float]      # area-weighted outward mean; (0,0,0) if it cancels
    planar: bool
    kind: str                               # "plane" | "cylinder-like" | "curved"
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "area": self.area,
            "centroid": list(self.centroid),
            "normal": list(self.normal),
            "planar": self.planar,
            "kind": self.kind,
            "bbox_min": list(self.bbox_min),
            "bbox_max": list(self.bbox_max),
        }


@dataclass(frozen=True)
class AnalysisMesh:
    """A solid's analysis mesh, in millimetres, with everything a load needs.

    Node indices are positions in `nodes`. Elements and triangles use gmsh
    node order (see TET10_EDGES / TRI6_EDGES); boundary triangles are wound
    outward, verified against the tetrahedron that owns each one.
    """

    path: Path
    order: int                                  # 1 or 2
    nodes: np.ndarray                           # (N,3) float64 mm
    elements: np.ndarray                        # (M,4) or (M,10) int64
    boundary: np.ndarray                        # (F,3) or (F,6) int64, outward
    boundary_face: np.ndarray                   # (F,) int64 CAD face tag per triangle
    face_nodes: Mapping[int, np.ndarray]        # tag -> sorted node indices, mid-edge nodes included
    face_centroids: Mapping[int, tuple[float, float, float]]
    faces: tuple[FaceSummary, ...]              # largest first
    edges: np.ndarray                           # (E,2) int64 segments along CAD edges
    corner_mask: np.ndarray                     # (N,) bool, True for tetrahedron vertices
    element_size_mm: float
    thinnest: float                             # min(min CAD-bbox extent, 2V/A), as features.mesh_step
    diagonal_mm: float                          # norm of the mesh node extent
    min_scaled_jacobian: float
    straight_sided: bool                        # True if curved mid-edge nodes had to be straightened
    healed: bool
    solid_count: int
    seconds: float
    #: OpenCASCADE volume and surface area of the solid(s): what `thinnest`
    #: and the element size were computed from.
    volume_mm3: float | None = None
    area_mm2: float | None = None
    #: Shape quality (Ansys "Element Quality": 1 = regular tetrahedron, -> 0 a
    #: sliver), min and median over the elements' corners. `min_scaled_jacobian`
    #: is not a shape measure: gmsh's minSJ is 1 for every straight-sided tet,
    #: slivers included, and only says no element is inverted.
    element_quality_min: float | None = None
    element_quality_median: float | None = None

    # ---- ScaledMesh-compatible names ---------------------------------------

    @property
    def vertices(self) -> np.ndarray:
        """Alias of `nodes`, so face_selectors work on this mesh unchanged."""
        return self.nodes

    @property
    def node_count(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def element_count(self) -> int:
        return int(self.elements.shape[0])

    @property
    def dof_count(self) -> int:
        """Three translations per node: what sizes the direct solve."""
        return 3 * self.node_count

    @property
    def face_tags(self) -> tuple[int, ...]:
        return tuple(sorted(self.face_nodes))

    @property
    def extent(self) -> np.ndarray:
        return self.nodes.max(axis=0) - self.nodes.min(axis=0)

    # ---- queries -------------------------------------------------------------

    def face_triangles(self, tags: Iterable[int]) -> np.ndarray:
        """Rows of `boundary` that belong to the given CAD faces (outward)."""
        wanted = np.array(sorted({int(t) for t in tags}), dtype=np.int64)
        return self.boundary[np.isin(self.boundary_face, wanted)]

    def face(self, tag: int) -> FaceSummary:
        for summary in self.faces:
            if summary.tag == int(tag):
                return summary
        raise AnalysisMeshError(
            f"this part has no face {tag}; its faces are {_ranges(self.face_tags)}. "
            "Use a number from the face table."
        )

    def faces_at(self, node: int) -> tuple[int, ...]:
        """CAD faces whose surface carries this node; () for an interior node."""
        node = int(node)
        found = []
        for tag in self.face_tags:
            members = self.face_nodes[tag]
            i = int(np.searchsorted(members, node))
            if i < members.size and int(members[i]) == node:
                found.append(tag)
        return tuple(found)


def _ranges(tags: Iterable[int]) -> str:
    """1, 2, 3, 5 -> '1-3, 5', so a refusal on a 400-face part stays one line."""
    ordered = sorted(set(int(t) for t in tags))
    if not ordered:
        return "none"
    parts: list[str] = []
    start = prev = ordered[0]
    for tag in ordered[1:] + [None]:  # type: ignore[list-item]
        if tag is not None and tag == prev + 1:
            prev = tag
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        if tag is not None:
            start = prev = tag
    return ", ".join(parts)


# --------------------------------------------------------------------------
# Sizing
# --------------------------------------------------------------------------


def _thinnest(extent: np.ndarray, volume: float | None, area: float | None) -> float:
    """min(smallest positive bbox extent, 2V/A) -- exactly `features.mesh_step`.

    The box cannot see an internal thin wall; 2V/A can (a plate of thickness t
    has V/A close to t/2). The smaller wins, so the rule may only refine.
    """
    extent = np.asarray(extent, dtype=float)
    positive = extent[extent > 0.0]
    if positive.size == 0 or not np.all(np.isfinite(extent)):
        raise AnalysisMeshError("the geometry has no finite extent; it cannot be meshed")
    thinnest = float(positive.min())
    if volume is not None and area is not None and area > 0.0:
        thinnest = min(thinnest, 2.0 * float(volume) / float(area))
    return thinnest


def element_size_for(
    extent: Any,
    volume: float | None,
    area: float | None,
    refinement: str = "medium",
) -> float:
    """Target element size in mm for a part with this CAD extent, volume and area.

    medium = clip(thinnest, diag/60, diag/8); coarse = clip(1.6 thinnest,
    diag/40, diag/6); fine = clip(thinnest/2, diag/100, diag/10). For 10-node
    tetrahedra one element through a wall is within ~1% in bending (measured;
    reports/mechanics_validation.json), so "medium" is already an answer and
    "fine" is a convergence check.
    """
    if refinement not in REFINEMENTS:
        raise AnalysisMeshError(
            f"refinement must be one of {', '.join(REFINEMENTS)}, not {refinement!r}; "
            "or give an element size in mm"
        )
    extent = np.asarray(extent, dtype=float)
    diagonal = float(np.linalg.norm(extent))
    thinnest = _thinnest(extent, volume, area)
    factor, low, high = REFINEMENTS[refinement]
    return float(np.clip(factor * thinnest, diagonal * low, diagonal * high))


def _check_order(order: Any) -> int:
    if isinstance(order, bool) or order not in (1, 2):
        raise AnalysisMeshError(
            f"element order must be 2 (10-node tetrahedra, default) or 1 (4-node, "
            f"comparison only), not {order!r}"
        )
    return int(order)


def _check_size(element_size_mm: Any) -> float:
    if isinstance(element_size_mm, bool):
        raise AnalysisMeshError("element_size_mm must be a number of millimetres, e.g. 2.5")
    try:
        size = float(element_size_mm)
    except (TypeError, ValueError):
        raise AnalysisMeshError(
            f"element_size_mm must be a number of millimetres, e.g. 2.5, not {element_size_mm!r}"
        ) from None
    if not math.isfinite(size) or size <= 0.0:
        raise AnalysisMeshError(
            f"element_size_mm must be a positive number of millimetres, e.g. 2.5, not {element_size_mm!r}"
        )
    return size


# --------------------------------------------------------------------------
# gmsh (only ever called inside an initialised session)
# --------------------------------------------------------------------------


def _generate(size: float, order: int, *, straight: bool) -> None:
    import gmsh

    gmsh.option.setNumber("Mesh.MeshSizeMax", size)
    gmsh.option.setNumber("Mesh.MeshSizeMin", size * MESH_SIZE_MIN_FRACTION)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", CURVATURE_ELEMENTS_PER_2PI)
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 1 if straight else 0)
    try:
        gmsh.model.mesh.generate(3)
        if order == 2:
            gmsh.model.mesh.setOrder(2)
    except Exception as exc:  # gmsh raises a bare Exception with its log line
        raise AnalysisMeshError(
            f"gmsh could not mesh this part at element size {size:.4g} mm ({exc}). "
            "Try another refinement or element size; if every size fails, the CAD "
            "is likely to have gaps or self-intersections that need repair."
        ) from exc


def _min_scaled_jacobian() -> float:
    """Smallest gmsh 'minSJ' over all volume elements (<= 0 means inverted).

    gmsh bounds the Jacobian over the whole element (not just at sample points),
    so minSJ > 0 guarantees detJ > 0 at every Gauss point the solver uses.
    """
    import gmsh

    _, element_tags, _ = gmsh.model.mesh.getElements(3)
    values = [
        np.asarray(gmsh.model.mesh.getElementQualities(tags, "minSJ"))
        for tags in element_tags
        if len(tags)
    ]
    if not values:
        return float("nan")
    return float(min(v.min() for v in values))


#: Ansys' normalising constant for tetrahedra: a regular one scores exactly 1.
_TET_QUALITY_C = 124.70765802


def element_quality(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """Ansys Mechanical's "Element Quality" of each tetrahedron, from its corners:
    C * V / sqrt(sum of squared edge lengths)^3, 1 for a regular tetrahedron and
    near 0 for a sliver. Curved mid-edge nodes do not enter it, as in Ansys."""
    corners = np.asarray(nodes, dtype=float)[np.asarray(elements)[:, :4]]
    a, b, c, d = (corners[:, i] for i in range(4))
    volume = np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))) / 6.0
    squares = sum(np.einsum("ij,ij->i", q - p, q - p)
                  for p, q in ((a, b), (a, c), (a, d), (b, c), (b, d), (c, d)))
    return _TET_QUALITY_C * volume / np.maximum(squares, 1e-300) ** 1.5


def _extract(order: int) -> dict[str, Any]:
    """Copy everything out of gmsh before the session is finalised."""
    import gmsh

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    if node_tags.size == 0:
        raise AnalysisMeshError("meshing produced no nodes")
    node_tags = node_tags.astype(np.int64)
    nodes = np.asarray(coords, dtype=float).reshape(-1, 3)
    lookup = np.full(int(node_tags.max()) + 1, -1, dtype=np.int64)
    lookup[node_tags] = np.arange(node_tags.size, dtype=np.int64)

    tet_type, tri_type, per_tet, per_tri = (
        (_TET10, _TRI6, 10, 6) if order == 2 else (_TET4, _TRI3, 4, 3)
    )

    types, _, element_nodes = gmsh.model.mesh.getElements(3)
    blocks = []
    for etype, enodes in zip(types, element_nodes):
        if int(etype) != tet_type:
            name = gmsh.model.mesh.getElementProperties(int(etype))[0]
            raise AnalysisMeshError(
                f"the volume mesh contains {name} elements; only tetrahedra are supported"
            )
        blocks.append(lookup[enodes.astype(np.int64)].reshape(-1, per_tet))
    if not blocks:
        raise AnalysisMeshError(
            "meshing produced no tetrahedra; the solid may be too thin for the "
            "element size -- try a finer refinement or a smaller element size"
        )
    elements = np.concatenate(blocks, axis=0)

    face_triangles: dict[int, np.ndarray] = {}
    face_nodes: dict[int, np.ndarray] = {}
    face_centroids: dict[int, tuple[float, float, float]] = {}
    face_areas: dict[int, float] = {}
    face_types: dict[int, str] = {}
    for _, tag in gmsh.model.getEntities(2):
        tag = int(tag)
        etypes, _, enodes = gmsh.model.mesh.getElements(2, tag)
        tris = [
            lookup[n.astype(np.int64)].reshape(-1, per_tri)
            for t, n in zip(etypes, enodes)
            if int(t) == tri_type
        ]
        if any(int(t) != tri_type for t in etypes):
            raise AnalysisMeshError(
                f"face {tag} was meshed with non-triangular elements; only triangles are supported"
            )
        if not tris or sum(len(t) for t in tris) == 0:
            continue    # a degenerate face with no surface mesh cannot carry a load
        face_triangles[tag] = np.concatenate(tris, axis=0)
        ntags, _, _ = gmsh.model.mesh.getNodes(2, tag, includeBoundary=True)
        face_nodes[tag] = np.unique(lookup[ntags.astype(np.int64)])
        face_centroids[tag] = tuple(float(c) for c in gmsh.model.occ.getCenterOfMass(2, tag))
        face_areas[tag] = float(gmsh.model.occ.getMass(2, tag))
        try:
            face_types[tag] = str(gmsh.model.getType(2, tag))
        except Exception:       # pragma: no cover - older gmsh builds
            face_types[tag] = ""

    segments = []
    for _, curve in gmsh.model.getEntities(1):
        up, _ = gmsh.model.getAdjacencies(1, curve)
        # A seam (the closing line of a periodic face such as a full cylinder)
        # borders one face on both sides. It is a parametrisation artefact, not
        # an edge of the part, and drawing it would mislead.
        if len({int(s) for s in up}) < 2:
            continue
        etypes, _, enodes = gmsh.model.mesh.getElements(1, curve)
        for t, n in zip(etypes, enodes):
            if int(t) == _LINE2:
                segments.append(lookup[n.astype(np.int64)].reshape(-1, 2))
            elif int(t) == _LINE3:
                line = lookup[n.astype(np.int64)].reshape(-1, 3)   # end, end, middle
                segments.append(line[:, [0, 2]])
                segments.append(line[:, [2, 1]])
    edges = (
        np.concatenate(segments, axis=0) if segments else np.zeros((0, 2), dtype=np.int64)
    )

    return {
        "nodes": nodes,
        "elements": elements,
        "face_triangles": face_triangles,
        "face_nodes": face_nodes,
        "face_centroids": face_centroids,
        "face_areas": face_areas,
        "face_types": face_types,
        "edges": edges,
    }


# --------------------------------------------------------------------------
# Post-processing (numpy only)
# --------------------------------------------------------------------------


def _triple_keys(rows: np.ndarray, n: int) -> np.ndarray:
    """One int64 key per sorted node triple."""
    rows = np.sort(np.asarray(rows, dtype=np.int64), axis=1)
    if n < 2**20:
        return (rows[:, 0] * n + rows[:, 1]) * n + rows[:, 2]
    # Past ~1M nodes the packed key could overflow int64; rank rows instead.
    _, inverse = np.unique(rows, axis=0, return_inverse=True)
    return inverse.reshape(-1).astype(np.int64)


def _flip(triangles: np.ndarray) -> np.ndarray:
    """Reverse winding: swap corners 1 and 2 (and, for 6 nodes, re-seat the mid-edges)."""
    if triangles.shape[1] == 3:
        return triangles[:, [0, 2, 1]]
    return triangles[:, [0, 2, 1, 5, 4, 3]]


def _orient_outward(nodes: np.ndarray, elements: np.ndarray,
                    boundary: np.ndarray) -> np.ndarray:
    """Wind every boundary triangle so its right-hand normal points out of the solid.

    Each triangle is matched to the tetrahedron face it lies on; if the
    tetrahedron's fourth vertex is on the normal's side, the triangle is wound
    inward and is flipped. A triangle that matches no tetrahedron face means the
    surface and volume meshes disagree, which no load could be trusted on.
    """
    if boundary.size == 0:
        return boundary
    corners = elements[:, :4]
    local = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])   # face k omits vertex k
    tet_faces = corners[:, local].reshape(-1, 3)
    n = int(nodes.shape[0])
    if n < 2**20:
        tet_keys = _triple_keys(tet_faces, n)
        tri_keys = _triple_keys(boundary[:, :3], n)
    else:
        both = _triple_keys(np.vstack([tet_faces, boundary[:, :3]]), n)
        tet_keys, tri_keys = both[: len(tet_faces)], both[len(tet_faces):]

    order = np.argsort(tet_keys, kind="stable")
    sorted_keys = tet_keys[order]
    left = np.searchsorted(sorted_keys, tri_keys, side="left")
    right = np.searchsorted(sorted_keys, tri_keys, side="right")
    owners = right - left
    if np.any(owners == 0):
        missing = int(np.count_nonzero(owners == 0))
        raise AnalysisMeshError(
            f"{missing} surface triangles are not faces of any tetrahedron: the "
            "surface and volume meshes disagree. Try another refinement or element "
            "size; if it persists, repair the CAD (gaps or overlapping faces)."
        )
    # owners == 2 is an internal face (two tetrahedra share it); it has no
    # outside, so its winding is left as gmsh wrote it.
    single = owners == 1
    face_index = order[np.minimum(left, len(order) - 1)]
    tet = face_index // 4
    opposite = corners[tet, face_index % 4]
    a, b, c = (nodes[boundary[:, i]] for i in range(3))
    normal = np.cross(b - a, c - a)
    inward = single & (np.einsum("ij,ij->i", normal, nodes[opposite] - a) > 0.0)
    oriented = boundary.copy()
    if inward.any():
        oriented[inward] = _flip(boundary[inward])
    return oriented


def _face_summaries(
    nodes: np.ndarray,
    boundary: np.ndarray,
    boundary_face: np.ndarray,
    face_nodes: Mapping[int, np.ndarray],
    face_centroids: Mapping[int, tuple[float, float, float]],
    face_areas: Mapping[int, float],
    face_types: Mapping[int, str] | None = None,
) -> tuple[FaceSummary, ...]:
    """Mesh-based area-weighted outward normal, planarity and kind for each face.

    Planarity is decided from the mesh alone. "cylinder-like" trusts
    OpenCASCADE when it says the surface is a cylinder, and otherwise asks the
    mesh whether every normal is perpendicular to one axis -- using the curved
    triangles' own normals on a 10-node mesh (a bore reads 0.01-0.07 degrees off
    its axis, measured) and flat facets on a 4-node one (up to 7.5 degrees, so
    a B-spline bore meshed with 4-node elements reads "curved").
    """
    face_types = face_types or {}
    jac = surface_jacobians(nodes, boundary)                     # (F,q,3)
    area_vectors = np.einsum("q,fqj->fj", TRI_QUADRATURE_WEIGHTS, jac)
    areas = np.einsum("q,fq->f", TRI_QUADRATURE_WEIGHTS, np.linalg.norm(jac, axis=2))
    a, b, c = (nodes[boundary[:, i]] for i in range(3))
    flat = np.cross(b - a, c - a)
    flat /= np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-300)
    if boundary.shape[1] == 6:
        # The curved triangle's own normal at its centre: much closer to the
        # surface's normal than the flat corner facet's, for classifying a bore.
        centre = surface_jacobians(nodes, boundary, np.array([[1.0 / 3.0, 1.0 / 3.0]]))[:, 0]
        centre /= np.maximum(np.linalg.norm(centre, axis=1, keepdims=True), 1e-300)
    else:
        centre = flat

    cos_planar = math.cos(math.radians(PLANAR_TOLERANCE_DEG))
    sin_cylinder = math.sin(math.radians(CYLINDER_TOLERANCE_DEG))
    order = np.argsort(boundary_face, kind="stable")
    tags, starts = np.unique(boundary_face[order], return_index=True)
    stops = list(starts[1:]) + [len(order)]

    summaries = []
    for tag, start, stop in zip(tags.tolist(), starts.tolist(), stops):
        rows = order[start:stop]
        total = float(areas[rows].sum())
        vector = area_vectors[rows].sum(axis=0)
        length = float(np.linalg.norm(vector))
        if total > 0.0 and length > 1e-6 * total:
            mean = vector / length
            planar = bool(np.min(flat[rows] @ mean) >= cos_planar)
        else:
            # A closed face (a full bore) has no net normal; saying (0,0,0) is
            # honest where a normalised rounding error would not be.
            mean = np.zeros(3)
            planar = False
        if planar:
            kind = "plane"
        elif face_types.get(tag, "").lower() == "cylinder":
            kind = "cylinder-like"
        else:
            weights = areas[rows] / max(total, 1e-300)
            normals = centre[rows]
            scatter = np.einsum("t,ti,tj->ij", weights, normals, normals)
            _, vectors = np.linalg.eigh(scatter)
            axis = vectors[:, 0]
            kind = "cylinder-like" if float(np.max(np.abs(normals @ axis))) <= sin_cylinder else "curved"
        points = nodes[face_nodes[tag]]
        cad_area = face_areas.get(tag, float("nan"))
        summaries.append(
            FaceSummary(
                tag=int(tag),
                area=float(cad_area) if math.isfinite(cad_area) and cad_area > 0 else total,
                centroid=tuple(float(v) for v in face_centroids[tag]),
                normal=tuple(float(v) for v in mean),
                planar=planar,
                kind=kind,
                bbox_min=tuple(float(v) for v in points.min(axis=0)),
                bbox_max=tuple(float(v) for v in points.max(axis=0)),
            )
        )
    # Largest first, the same order `cad_intake.inspect` lists faces in.
    return tuple(sorted(summaries, key=lambda f: (-f.area, f.tag)))


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def mesh_for_analysis(
    path: str | Path,
    *,
    order: int = 2,
    refinement: str = "medium",
    element_size_mm: float | None = None,
) -> AnalysisMesh:
    """Mesh a STEP/IGES/BREP part with 10-node (or 4-node) tetrahedra for solving.

    Order 2 curves mid-edge nodes onto the CAD surfaces (a hole stays round).
    That can invert an element on thin curved features (2 of 31,643 on
    component8.step), so the scaled Jacobian is checked: if any element is
    inverted gmsh's high-order optimiser runs, and if that fails the mesh is
    regenerated with straight-sided elements (`straight_sided=True`). A mesh
    with a non-positive Jacobian is never returned.

    Raises `cad_intake.CadFormatError` for an unreadable file and
    `AnalysisMeshError` for a file with no solid or a mesh that cannot be made.
    """
    from .cad_intake import check_readable, import_shapes, start_gmsh

    order = _check_order(order)
    if refinement not in REFINEMENTS:
        raise AnalysisMeshError(
            f"refinement must be one of {', '.join(REFINEMENTS)}, not {refinement!r}; "
            "or give an element size in mm"
        )
    requested_size = None if element_size_mm is None else _check_size(element_size_mm)
    resolved = check_readable(path)

    import gmsh

    started = time.perf_counter()
    start_gmsh()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("analysis")
        # The intake's import, so face numbers are the face table's numbers.
        healed = import_shapes(resolved)
        volumes = gmsh.model.getEntities(3)
        if not volumes:
            raise AnalysisMeshError(
                "the file contains no solid, and sewing its surfaces into one did "
                "not close them. A surface or wireframe model has no volume to carry "
                "stress; export the part as a solid."
            )
        corners = np.array([gmsh.model.getBoundingBox(3, tag) for _, tag in volumes])
        extent = corners[:, 3:].max(axis=0) - corners[:, :3].min(axis=0)
        volume = float(sum(gmsh.model.occ.getMass(3, tag) for _, tag in volumes))
        area = float(sum(gmsh.model.occ.getMass(2, tag) for _, tag in gmsh.model.getEntities(2)))
        thinnest = _thinnest(extent, volume, area)
        size = (
            requested_size if requested_size is not None
            else element_size_for(extent, volume, area, refinement)
        )
        estimate = _ELEMENTS_PER_SIZE_CUBED * volume / size**3
        if estimate > MAX_ESTIMATED_ELEMENTS:
            raise AnalysisMeshError(
                f"an element size of {size:.4g} mm would give roughly "
                f"{estimate:,.0f} elements on this {volume:,.0f} mm³ part, far more "
                "than a direct solver can factor. Use a larger element size or a "
                "coarser refinement."
            )

        _generate(size, order, straight=False)
        min_sj = _min_scaled_jacobian()
        straight = False
        if order == 2 and not min_sj > 0.0:
            try:
                gmsh.model.mesh.optimize("HighOrder")
            except Exception:   # the optimiser failing is the case the fallback is for
                pass
            min_sj = _min_scaled_jacobian()
            if not min_sj > 0.0:
                gmsh.model.mesh.clear()
                _generate(size, order, straight=True)
                straight = True
                min_sj = _min_scaled_jacobian()
        if not min_sj > 0.0:
            raise AnalysisMeshError(
                f"the mesh has inverted elements (minimum scaled Jacobian "
                f"{min_sj:.3g}) even with straight-sided elements. Try another "
                "refinement or element size, or repair the CAD."
            )
        raw = _extract(order)
    finally:
        gmsh.finalize()

    nodes = raw["nodes"]
    elements = raw["elements"]
    face_triangles = raw["face_triangles"]
    if not face_triangles:
        raise AnalysisMeshError("meshing produced no surface triangles")
    boundary_face = np.concatenate(
        [np.full(len(t), tag, dtype=np.int64) for tag, t in face_triangles.items()]
    )
    boundary = _orient_outward(
        nodes, elements, np.concatenate(list(face_triangles.values()), axis=0)
    )
    corner_mask = np.zeros(len(nodes), dtype=bool)
    corner_mask[elements[:, :4].ravel()] = True
    quality = element_quality(nodes, elements)
    faces = _face_summaries(
        nodes, boundary, boundary_face, raw["face_nodes"], raw["face_centroids"],
        raw["face_areas"], raw["face_types"],
    )
    extent_mesh = nodes.max(axis=0) - nodes.min(axis=0)

    return AnalysisMesh(
        path=resolved,
        order=order,
        nodes=nodes,
        elements=elements,
        boundary=boundary,
        boundary_face=boundary_face,
        face_nodes=raw["face_nodes"],
        face_centroids=raw["face_centroids"],
        faces=faces,
        edges=raw["edges"],
        corner_mask=corner_mask,
        element_size_mm=float(size),
        thinnest=float(thinnest),
        diagonal_mm=float(np.linalg.norm(extent_mesh)),
        min_scaled_jacobian=float(min_sj),
        straight_sided=straight,
        healed=bool(healed),
        solid_count=len(volumes),
        seconds=time.perf_counter() - started,
        volume_mm3=volume,
        area_mm2=area,
        element_quality_min=float(quality.min()),
        element_quality_median=float(np.median(quality)),
    )


__all__ = [
    "AnalysisMesh",
    "AnalysisMeshError",
    "CURVATURE_ELEMENTS_PER_2PI",
    "FaceSummary",
    "MAX_ESTIMATED_ELEMENTS",
    "PLANAR_TOLERANCE_DEG",
    "REFINEMENTS",
    "TET10_EDGES",
    "TRI6_EDGES",
    "TRI_QUADRATURE_POINTS",
    "TRI_QUADRATURE_WEIGHTS",
    "element_quality",
    "element_size_for",
    "mesh_for_analysis",
    "surface_jacobians",
    "tri_shape_functions",
    "triangle_areas",
]
