"""Turn geometry into the node features a given model version was trained on.

Two versions exist and both are kept. A model is only meaningful alongside the
feature construction it was fitted with, so the version is selected by what the
calibration record declares rather than by whichever happens to be newest.

    sfem/v1   12 columns, trained on cmudrc/SFEM
    sweep/v2  13 columns, trained on the fea-surrogate sweep

The difference is not cosmetic. `sfem/v1` encodes the load's *magnitude class*
and never where it lands, which is why a model fitted on it could not do better
than predict the mean over load placements and reached a conformal band of
x/div 19.75. `sweep/v2` carries `is_loaded` and `load_dir`, and reached x/div
1.91 on the same architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import Compose, FaceToEdge, GenerateMeshNormals

from .versions import (
    DEFAULT_VERSION,
    FEATURE_COLUMNS,
    SFEM_V1,
    SWEEP_V2,
    FeatureVersionError,
    require_version,
)


# sfem/v1 only: the training SDF was computed against this fixed box, so an
# incoming part has to be placed in it or the feature means nothing.
SDF_BBOX_MIN = (0.0, 0.0, 0.0)
SDF_BBOX_MAX = (3.0, 3.0, 3.0)

# sfem/v1 only: the load reached that model as a 3-way one-hot.
LOAD_CLASSES_N = (200.0, 2000.0, 20000.0)

MESH_SIZE_MAX = 0.1


class MeshingError(RuntimeError):
    """The geometry could not be meshed into the form the model expects."""


@dataclass(frozen=True)
class ScaledMesh:
    vertices: np.ndarray            # (N, 3)
    faces: np.ndarray               # (3, F) surface triangles
    scale: float
    original_diagonal: float
    tetrahedra: np.ndarray | None = None   # (M, 4); sweep/v2 only
    # CAD face tag -> the mesh nodes on that face. Captured while gmsh is still
    # alive, because a boundary condition has to be selected the way the sweep
    # selected it: by face, not by node coordinate. Selecting by coordinate is
    # what put deployed coverage at 0.415 against a calibrated 0.90.
    face_nodes: Mapping[int, np.ndarray] | None = None
    face_centroids: Mapping[int, tuple[float, float, float]] | None = None
    # min(smallest bbox extent, 2V/A). The bounding box cannot see a thin
    # internal feature and 2V/A can, which is why the tags express their
    # tolerances against it rather than against an overall dimension.
    thinnest: float | None = None


def mesh_step(path: str | Path, *, version: str = DEFAULT_VERSION) -> ScaledMesh:
    """Mesh a STEP file.

    `sfem/v1` rescales into the fixed training box; `sweep/v2` keeps the part's
    own millimetres, because that is what its sweep recorded.
    """
    require_version(version)
    try:
        import gmsh
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise MeshingError("gmsh is required to mesh STEP geometry") from exc

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("screening")
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()

        volumes = gmsh.model.getEntities(3)
        if not volumes:
            raise MeshingError("the STEP file contains no solid")
        corners = np.array([gmsh.model.getBoundingBox(3, t) for _, t in volumes])
        extent = corners[:, 3:].max(axis=0) - corners[:, :3].min(axis=0)
        diagonal = float(np.linalg.norm(extent))
        if not np.isfinite(diagonal) or diagonal <= 0.0:
            raise MeshingError("geometry has no finite extent")

        thinnest = float(np.min(extent[extent > 0.0]))
        if version == SFEM_V1:
            span = float(max(SDF_BBOX_MAX) - min(SDF_BBOX_MIN))
            scale = span / float(extent.max())
            size = MESH_SIZE_MAX / scale
        else:
            scale = 1.0
            # Same rule the sweep uses: the bounding box cannot see a thin
            # internal feature, and 2V/A can.
            volume = sum(gmsh.model.occ.getMass(3, tag) for _, tag in volumes)
            area = sum(
                gmsh.model.occ.getMass(2, tag)
                for _, tag in gmsh.model.getEntities(2)
            )
            if area > 0.0:
                thinnest = min(thinnest, 2.0 * volume / area)
            size = float(np.clip(thinnest / 4.0, diagonal / 80.0, diagonal / 8.0))

        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size / 4.0)
        gmsh.model.mesh.generate(2 if version == SFEM_V1 else 3)

        tags, coords, _ = gmsh.model.mesh.getNodes()
        if tags.size == 0:
            raise MeshingError("meshing produced no nodes")
        vertices = coords.reshape(-1, 3)
        index = {int(t): i for i, t in enumerate(tags)}

        types, _, node_tags = gmsh.model.mesh.getElements(dim=2)
        triangles = [
            np.array([index[int(n)] for n in nodes], dtype=np.int64).reshape(-1, 3)
            for etype, nodes in zip(types, node_tags)
            if int(etype) == 2
        ]
        if not triangles:
            raise MeshingError("meshing produced no triangles")
        # Transposing leaves a non-contiguous view, and torch_geometric's
        # transforms call .view() on it, which fails on that stride.
        faces = np.ascontiguousarray(np.concatenate(triangles, axis=0).T)

        face_nodes: dict[int, np.ndarray] = {}
        face_centroids: dict[int, tuple[float, float, float]] = {}
        for _, tag in gmsh.model.getEntities(2):
            node_tags, _, _ = gmsh.model.mesh.getNodes(2, int(tag),
                                                       includeBoundary=True)
            if node_tags.size == 0:
                continue
            face_nodes[int(tag)] = np.array(
                sorted({index[int(t)] for t in node_tags}), dtype=np.int64
            )
            centre = gmsh.model.occ.getCenterOfMass(2, int(tag))
            face_centroids[int(tag)] = tuple(float(c) for c in centre)

        tetrahedra = None
        if version != SFEM_V1:
            tet_types, _, tet_nodes = gmsh.model.mesh.getElements(dim=3)
            blocks = [
                np.array([index[int(n)] for n in nodes], dtype=np.int64).reshape(-1, 4)
                for etype, nodes in zip(tet_types, tet_nodes)
                if int(etype) == 4
            ]
            if not blocks:
                raise MeshingError("meshing produced no tetrahedra")
            tetrahedra = np.concatenate(blocks, axis=0)
    finally:
        gmsh.finalize()

    if version == SFEM_V1:
        lowered = vertices - vertices.min(axis=0)
        span = float(max(SDF_BBOX_MAX) - min(SDF_BBOX_MIN))
        vertices = lowered * (span / float(lowered.max()))

    return ScaledMesh(
        vertices=vertices.astype(np.float32),
        faces=faces,
        scale=scale,
        original_diagonal=diagonal,
        tetrahedra=tetrahedra,
        face_nodes=face_nodes,
        face_centroids=face_centroids,
        thinnest=thinnest,
    )


def _sdf_fixed_box(pos: torch.Tensor) -> torch.Tensor:
    """sfem/v1: signed distance to the fixed [0,3]^3 training box."""
    lo = torch.tensor(SDF_BBOX_MIN, dtype=pos.dtype)
    hi = torch.tensor(SDF_BBOX_MAX, dtype=pos.dtype)
    zero = torch.zeros((), dtype=pos.dtype)
    gaps = torch.stack(
        [
            torch.maximum(lo[i] - pos[:, i], torch.maximum(zero, pos[:, i] - hi[i]))
            for i in range(3)
        ],
        dim=1,
    )
    distance = gaps.pow(2).sum(dim=1).sqrt()
    inside = ((pos >= lo) & (pos <= hi)).all(dim=1)
    return torch.where(inside, -distance, distance).unsqueeze(1)


def _sdf_own_box(pos: torch.Tensor) -> torch.Tensor:
    """sweep/v2: signed distance to the part's own box, over its diagonal."""
    lo, hi = pos.min(dim=0).values, pos.max(dim=0).values
    diagonal = float(torch.linalg.norm(hi - lo)) or 1.0
    inner = torch.minimum(pos - lo, hi - pos).min(dim=1).values
    return (-inner / diagonal).unsqueeze(1)


_TET_PAIRS = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], dtype=np.int64)


def _tet_edges(tetrahedra: np.ndarray, node_count: int) -> np.ndarray:
    """Undirected edges of a tetrahedral mesh, deduplicated. (2, E)."""
    pairs = np.sort(tetrahedra[:, _TET_PAIRS].reshape(-1, 2), axis=1)
    key = np.unique(pairs[:, 0].astype(np.int64) * node_count + pairs[:, 1])
    lo, hi = key // node_count, key % node_count
    return np.stack([np.concatenate([lo, hi]), np.concatenate([hi, lo])])


def nearest_load_class(force_n: float) -> tuple[int, float]:
    """sfem/v1 only. Index of the closest trained load class."""
    i = int(np.argmin([abs(np.log(force_n) - np.log(c)) for c in LOAD_CLASSES_N]))
    return i, LOAD_CLASSES_N[i]


def build_graph(
    mesh: ScaledMesh,
    *,
    version: str = DEFAULT_VERSION,
    fixed_vertices: np.ndarray | None = None,
    loaded_vertices: np.ndarray | None = None,
    load_direction: tuple[float, float, float] = (0.0, 0.0, -1.0),
    poisson_ratio: float = 0.3,
    force_n: float = 1000.0,
) -> Data:
    """Assemble the node features for `version`."""
    require_version(version)

    pos = torch.from_numpy(mesh.vertices).float()
    data = Data(pos=pos, face=torch.from_numpy(mesh.faces).long())
    data = Compose([FaceToEdge(remove_faces=False), GenerateMeshNormals()])(data)
    n = pos.shape[0]

    if version != SFEM_V1:
        # Edges must come from the tetrahedra, as the sweep builds them.
        # FaceToEdge connects only surface triangles, which leaves every
        # interior node isolated: 43% of a bracket's nodes had degree zero,
        # message passing could not reach them, and the prediction was out by
        # a factor of twenty.
        if mesh.tetrahedra is None:
            raise FeatureVersionError(f"{version} needs a tetrahedral mesh")
        data.edge_index = torch.from_numpy(_tet_edges(mesh.tetrahedra, n)).long()

    if fixed_vertices is None:
        raise ValueError(
            "build_graph needs a fixed-vertex mask; guessing one screens a "
            "load case the caller did not ask for"
        )
    fixed = torch.from_numpy(np.asarray(fixed_vertices, dtype=bool))

    if version == SFEM_V1:
        data.sdf = _sdf_fixed_box(data.pos)
        fixed_free = torch.zeros((n, 2))
        fixed_free[fixed, 0] = 1.0
        fixed_free[~fixed, 1] = 1.0
        load = torch.zeros((n, 3))
        load[:, nearest_load_class(force_n)[0]] = 1.0
        data.x = torch.cat([data.pos, data.sdf, data.norm, fixed_free, load], dim=1)
    else:
        data.sdf = _sdf_own_box(data.pos)
        if loaded_vertices is None:
            raise ValueError("build_graph needs a loaded-vertex mask")
        loaded = torch.from_numpy(np.asarray(loaded_vertices, dtype=bool))
        direction = np.array(load_direction, dtype=float)
        direction = direction / (np.linalg.norm(direction) or 1.0)
        data.x = torch.cat(
            [
                data.pos,
                data.sdf,
                data.norm,
                fixed.reshape(-1, 1).float(),
                loaded.reshape(-1, 1).float(),
                torch.tensor(direction, dtype=torch.float).repeat(n, 1)
                * loaded.reshape(-1, 1).float(),
                torch.full((n, 1), float(poisson_ratio)),
            ],
            dim=1,
        )
        data.loaded_mask = loaded

    expected = FEATURE_COLUMNS[version]
    if data.x.shape[1] != expected:
        raise FeatureVersionError(
            f"{version} expects {expected} columns, built {data.x.shape[1]}"
        )
    data.fixed_mask = fixed
    return data


__all__ = [
    "DEFAULT_VERSION",
    "FEATURE_COLUMNS",
    "FeatureVersionError",
    "LOAD_CLASSES_N",
    "MeshingError",
    "SDF_BBOX_MAX",
    "SDF_BBOX_MIN",
    "SFEM_V1",
    "SWEEP_V2",
    "ScaledMesh",
    "build_graph",
    "mesh_step",
    "nearest_load_class",
]
