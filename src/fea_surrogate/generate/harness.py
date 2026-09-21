"""Draw a family, mesh it, and assemble the node features.

Shared by every family, so a new family contributes geometry and tagging and
inherits everything else. Solving is a separate step: this produces the inputs
and the mesh, and a solver fills in the labels.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .family import FEATURE_COLUMNS, Family, Sample, Tagging, sample_force, sample_poisson


# Elements through the thinnest dimension. A fraction of the diagonal is the
# wrong rule for a slender part: a 200x40x10 beam meshed at 6% of its diagonal
# gets one element through the thickness and reports a third of its true
# bending stress. Bounds keep the node count finite at extreme aspect ratios.
ELEMENTS_THROUGH_THICKNESS = 4
MESH_SIZE_MIN_FRACTION = 1 / 80   # of the diagonal
MESH_SIZE_MAX_FRACTION = 1 / 8


def mesh_size_for(
    extent: np.ndarray,
    volume: float | None = None,
    area: float | None = None,
) -> float:
    """Characteristic element size for a part with this extent.

    The bounding box alone is not enough. An L-bracket's box is 180x60x90 and
    its plate is 12 mm thick, so sizing from the box gave h = 15 and less than
    one element through the plate; measured across the families, brackets got
    0.3 to 1.2 elements through their thin feature and welded T joints 0.5 to
    0.7. Those are the two families whose thin feature is internal, and welded_t
    is by a wide margin the hardest family to predict.

    `2 * volume / area` recovers a thin dimension the box cannot see, because a
    plate of thickness t has V/A close to t/2. It is used alongside the box
    minimum, never instead of it, and the smaller of the two wins: the rule may
    only refine.
    """
    diagonal = float(np.linalg.norm(extent))
    thinnest = float(np.min(extent[extent > 0.0])) if np.any(extent > 0.0) else diagonal
    if volume and area and area > 0.0:
        thinnest = min(thinnest, 2.0 * float(volume) / float(area))
    return float(
        np.clip(
            thinnest / ELEMENTS_THROUGH_THICKNESS,
            diagonal * MESH_SIZE_MIN_FRACTION,
            diagonal * MESH_SIZE_MAX_FRACTION,
        )
    )


@dataclass(frozen=True)
class MeshedSample:
    """A drawn sample, meshed, with model inputs assembled.

    `x` is the feature matrix; `labels` stays empty until a solver runs. The
    separation is deliberate: meshing is cheap and reproducible, solving is
    neither.
    """

    sample: Sample
    vertices: np.ndarray        # (N, 3) mm
    tetrahedra: np.ndarray      # (M, 4)
    surface_faces: np.ndarray   # (F, 3)
    x: np.ndarray               # (N, FEATURE_COLUMNS)
    fixed_mask: np.ndarray      # (N,) bool
    loaded_mask: np.ndarray     # (N,) bool

    @property
    def characteristic_length_mm(self) -> float:
        extent = self.vertices.max(axis=0) - self.vertices.min(axis=0)
        return float(np.linalg.norm(extent))


@contextmanager
def _gmsh_model(name: str) -> Iterator[None]:
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(name)
        yield
    finally:
        gmsh.finalize()


def _nodes_of_faces(face_tags: tuple[int, ...], index: dict[int, int]) -> np.ndarray:
    import gmsh

    found: set[int] = set()
    for tag in face_tags:
        tags, _, _ = gmsh.model.mesh.getNodes(2, tag, includeBoundary=True)
        found.update(index[int(t)] for t in tags)
    return np.array(sorted(found), dtype=np.int64)


def _signed_distance_to_box(pos: np.ndarray) -> np.ndarray:
    """Distance to the part's own bounding box, normalised by its diagonal.

    SFEM used a fixed [0,3]^3 box, which forces every incoming part to be
    rescaled into it. Normalising by the part's own diagonal instead makes the
    feature scale-free, which is the same reason the target is dimensionless.
    """
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    diagonal = float(np.linalg.norm(hi - lo)) or 1.0
    gaps = np.maximum(np.maximum(lo - pos, 0.0), np.maximum(pos - hi, 0.0))
    outside = np.linalg.norm(gaps, axis=1)
    inner = np.minimum(pos - lo, hi - pos).min(axis=1)
    signed = np.where(outside > 0.0, outside, -inner)
    return (signed / diagonal).reshape(-1, 1)


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices)
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    face_normal = np.cross(b - a, c - a)
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normal)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.where(lengths == 0.0, 1.0, lengths)


def draw(family: Family, rng: np.random.Generator) -> MeshedSample:
    """Draw one sample from `family` and mesh it."""
    import gmsh

    params = family.sample_params(rng)
    poisson = sample_poisson(rng)
    force = sample_force(rng)

    with _gmsh_model(family.name):
        family.build(params)
        gmsh.model.occ.synchronize()
        tagging: Tagging = family.tag(params)

        # Measure the solid, not the model: boolean construction leaves stray
        # lower-dimensional entities behind, and a cut pulley reads 16% wider
        # than it is if those are included.
        volumes = gmsh.model.getEntities(3)
        if not volumes:
            raise ValueError(f"family {family.name} produced no solid")
        corners = np.array(
            [gmsh.model.getBoundingBox(3, tag) for _, tag in volumes]
        )
        extent = corners[:, 3:].max(axis=0) - corners[:, :3].min(axis=0)
        volume = sum(gmsh.model.occ.getMass(3, tag) for _, tag in volumes)
        area = sum(
            gmsh.model.occ.getMass(2, tag) for _, tag in gmsh.model.getEntities(2)
        )
        size = mesh_size_for(extent, volume, area)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size / 4.0)
        gmsh.model.mesh.generate(3)

        tags, coords, _ = gmsh.model.mesh.getNodes()
        vertices = coords.reshape(-1, 3)
        index = {int(t): i for i, t in enumerate(tags)}

        tet_types, _, tet_nodes = gmsh.model.mesh.getElements(dim=3)
        tets = np.concatenate(
            [
                np.array([index[int(n)] for n in nodes], dtype=np.int64).reshape(-1, 4)
                for etype, nodes in zip(tet_types, tet_nodes)
                if int(etype) == 4
            ]
        )
        tri_types, _, tri_nodes = gmsh.model.mesh.getElements(dim=2)
        tris = np.concatenate(
            [
                np.array([index[int(n)] for n in nodes], dtype=np.int64).reshape(-1, 3)
                for etype, nodes in zip(tri_types, tri_nodes)
                if int(etype) == 2
            ]
        )

        fixed_nodes = _nodes_of_faces(tagging.fixed_faces, index)
        loaded_nodes = _nodes_of_faces(tagging.loaded_faces, index)

    n = vertices.shape[0]
    fixed_mask = np.zeros(n, dtype=bool)
    loaded_mask = np.zeros(n, dtype=bool)
    fixed_mask[fixed_nodes] = True
    loaded_mask[loaded_nodes] = True

    direction = np.array(tagging.load_direction, dtype=float)
    direction = direction / np.linalg.norm(direction)

    x = np.concatenate(
        [
            vertices,
            _signed_distance_to_box(vertices),
            _vertex_normals(vertices, tris),
            fixed_mask.reshape(-1, 1).astype(float),
            loaded_mask.reshape(-1, 1).astype(float),
            np.tile(direction, (n, 1)) * loaded_mask.reshape(-1, 1),
            np.full((n, 1), poisson),
        ],
        axis=1,
    )
    if x.shape[1] != FEATURE_COLUMNS:
        raise AssertionError(f"expected {FEATURE_COLUMNS} columns, built {x.shape[1]}")

    return MeshedSample(
        sample=Sample(
            family=family.name,
            params=params,
            tagging=tagging,
            poisson_ratio=poisson,
            force_n=force,
        ),
        vertices=vertices,
        tetrahedra=tets,
        surface_faces=tris,
        x=x,
        fixed_mask=fixed_mask,
        loaded_mask=loaded_mask,
    )


__all__ = ["MeshedSample", "draw", "mesh_size_for"]
