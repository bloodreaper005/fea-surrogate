"""Turn a STEP file into the exact node features the model was trained on.

Every choice here is dictated by how SFEM was preprocessed, not by what would
be sensible in isolation. Three of them are approximations that the domain
declaration has to carry, and they are named where they happen.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import Compose, FaceToEdge, GenerateMeshNormals


# The training SDF was computed against this box and nothing else, so an
# incoming part must be placed in the same one or the feature is meaningless.
SDF_BBOX_MIN = (0.0, 0.0, 0.0)
SDF_BBOX_MAX = (3.0, 3.0, 3.0)

# The load reaches the model as a 3-way one-hot, so an arbitrary force is
# snapped to the nearest of these. Magnitude itself is handled exactly, by the
# dimensionless target; what is approximated is only which regime the model is
# told it is in.
LOAD_CLASSES_N = (200.0, 2000.0, 20000.0)

MESH_SIZE_MAX = 0.1  # matches lc in the authors' mesh_generation.py


class MeshingError(RuntimeError):
    """The geometry could not be meshed into the form the model expects."""


@dataclass(frozen=True)
class ScaledMesh:
    """A surface mesh placed in the training box, with the scale that got it there."""

    vertices: np.ndarray        # (N, 3) inside SDF_BBOX
    faces: np.ndarray           # (3, F)
    scale: float                # multiply original coordinates by this
    original_diagonal: float    # bounding-box diagonal in the file's own units


def mesh_step(path: str | Path) -> ScaledMesh:
    """Mesh a STEP file and rescale it into the training bounding box."""
    try:
        import gmsh
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise MeshingError("gmsh is required to mesh STEP geometry") from exc

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("screening")
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()

        # Size the mesh relative to the part, then rescale, so the node count is
        # comparable to training regardless of the file's units.
        box = gmsh.model.getBoundingBox(-1, -1)
        extent = np.array(box[3:]) - np.array(box[:3])
        diagonal = float(np.linalg.norm(extent))
        if not np.isfinite(diagonal) or diagonal <= 0.0:
            raise MeshingError("geometry has no finite extent")

        span = float(max(SDF_BBOX_MAX) - min(SDF_BBOX_MIN))
        scale = span / float(extent.max())
        gmsh.option.setNumber("Mesh.MeshSizeMax", MESH_SIZE_MAX / scale)
        gmsh.option.setNumber("Mesh.MeshSizeMin", MESH_SIZE_MAX / scale / 10.0)
        gmsh.model.mesh.generate(2)

        tags, coords, _ = gmsh.model.mesh.getNodes()
        if tags.size == 0:
            raise MeshingError("meshing produced no nodes")
        vertices = coords.reshape(-1, 3)
        index = {int(t): i for i, t in enumerate(tags)}

        types, _, node_tags = gmsh.model.mesh.getElements(dim=2)
        triangles: list[np.ndarray] = []
        for etype, nodes in zip(types, node_tags):
            if int(etype) != 2:  # 3-node triangle
                continue
            tri = np.array([index[int(n)] for n in nodes], dtype=np.int64)
            triangles.append(tri.reshape(-1, 3))
        if not triangles:
            raise MeshingError("meshing produced no triangles")
        faces = np.concatenate(triangles, axis=0).T
    finally:
        gmsh.finalize()

    lowered = vertices - vertices.min(axis=0)
    scaled = lowered * (span / float(lowered.max()))
    return ScaledMesh(
        vertices=scaled.astype(np.float32),
        faces=faces,
        scale=span / float(extent.max()),
        original_diagonal=diagonal,
    )


def _sdf(pos: torch.Tensor) -> torch.Tensor:
    """Signed distance to the training box, reproducing the authors' ComputeSDF."""
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


def nearest_load_class(force_n: float) -> tuple[int, float]:
    """Index of the closest trained load class, and that class's magnitude."""
    diffs = [abs(np.log(force_n) - np.log(c)) for c in LOAD_CLASSES_N]
    i = int(np.argmin(diffs))
    return i, LOAD_CLASSES_N[i]


def build_graph(
    mesh: ScaledMesh,
    *,
    fixed_vertices: np.ndarray | None,
    force_n: float,
) -> Data:
    """Assemble the 12-column node features the model expects.

    `fixed_vertices` is a boolean mask over the mesh vertices. SFEM took the
    largest surface group as fixed and everything else free, so passing None
    reproduces that heuristic rather than guessing something cleverer.
    """
    pos = torch.from_numpy(mesh.vertices).float()
    face = torch.from_numpy(mesh.faces).long()

    data = Data(pos=pos, face=face)
    data = Compose([FaceToEdge(remove_faces=False), GenerateMeshNormals()])(data)
    data.sdf = _sdf(data.pos)

    n = pos.shape[0]
    fixed_free = torch.zeros((n, 2), dtype=torch.float)
    if fixed_vertices is None:
        # Lowest-z face as a stand-in for "the surface it sits on".
        z = pos[:, 2]
        mask = z <= (z.min() + 0.02 * (z.max() - z.min() + 1e-9))
    else:
        mask = torch.from_numpy(np.asarray(fixed_vertices, dtype=bool))
    fixed_free[mask, 0] = 1.0
    fixed_free[~mask, 1] = 1.0

    load_index, _ = nearest_load_class(force_n)
    load = torch.zeros((n, 3), dtype=torch.float)
    load[:, load_index] = 1.0

    data.x = torch.cat([data.pos, data.sdf, data.norm, fixed_free, load], dim=1)
    return data


__all__ = [
    "LOAD_CLASSES_N",
    "MESH_SIZE_MAX",
    "MeshingError",
    "ScaledMesh",
    "SDF_BBOX_MAX",
    "SDF_BBOX_MIN",
    "build_graph",
    "mesh_step",
    "nearest_load_class",
]
