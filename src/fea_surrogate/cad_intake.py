"""Read a CAD file and describe its faces, so a load case can name them.

Boundary conditions have to be specified against something. The sweep that
trained the model specifies them against *faces* -- `family.tag()` returns face
tags, and every node of a tagged face is held or loaded. The deployed path
specified them against *node coordinates* instead, through selectors like
`{"axis": "z", "at": "min"}`, and the two do not agree:

    spring   the preset held 20 of 6144 nodes, a tangent sliver of the helix,
             where training held every face of the bottom end coil
    flange   the preset held 26.5% of nodes in a slab through the whole plate,
             where training held only the hole walls

Measured across 600 fresh parts, that gap put deployed coverage at 0.415 against
a calibrated 0.90, with spring and flange at 0.00. The model was fine; it was
being asked a different question.

This module is the fix. It enumerates the faces of an imported solid with enough
description -- area, centroid, normal, planarity -- that a person or a preset can
point at one by name, the way a pre-processor lets you click it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


class CadFormatError(ValueError):
    """The file is not in a format OpenCASCADE can read."""


class CadIntakeError(RuntimeError):
    """The file was readable but contained nothing screenable."""


#: What gmsh's `occ.importShapes` accepts, from its own API documentation:
#: "Import BREP, STEP or IGES shapes". Nothing else goes through OpenCASCADE.
READABLE_SUFFIXES = frozenset({".step", ".stp", ".iges", ".igs", ".brep", ".brp"})

#: Formats people reasonably expect to work, with the reason they do not. A
#: Parasolid file needs a licensed Siemens kernel; OpenCASCADE cannot read one at
#: any price, so failing three calls later inside the mesher helps nobody.
_UNREADABLE = {
    ".x_t": "Parasolid text",
    ".x_b": "Parasolid binary",
    ".xmt_txt": "Parasolid text",
    ".xmt_bin": "Parasolid binary",
    ".sldprt": "SolidWorks part",
    ".ipt": "Inventor part",
    ".catpart": "CATIA part",
    ".prt": "NX or Creo part",
    ".3dm": "Rhino",
    ".sat": "ACIS",
}


@dataclass(frozen=True)
class Face:
    """One face of the solid, described well enough to be chosen deliberately."""

    tag: int
    area: float
    centroid: tuple[float, float, float]
    normal: tuple[float, float, float]
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    planar: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "area": self.area,
            "centroid": list(self.centroid),
            "normal": list(self.normal),
            "bbox_min": list(self.bbox_min),
            "bbox_max": list(self.bbox_max),
            "planar": self.planar,
        }


@dataclass(frozen=True)
class Solid:
    """What a CAD file turned out to contain."""

    path: Path
    faces: tuple[Face, ...]
    volume: float
    area: float
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    #: How many separate solids the file holds. An assembly is not a part: the
    #: surrogate has only ever seen single connected bodies, and a solve on
    #: several of them is singular unless every one is held.
    solid_count: int = 1
    #: True when the file held no solid and its surfaces had to be sewn into
    #: one. A repair, not a read: it closes gaps within a tolerance.
    healed: bool = False

    @property
    def extent(self) -> np.ndarray:
        return np.array(self.bbox_max) - np.array(self.bbox_min)

    @property
    def diagonal(self) -> float:
        return float(np.linalg.norm(self.extent))

    def face(self, tag: int) -> Face:
        for candidate in self.faces:
            if candidate.tag == tag:
                return candidate
        known = ", ".join(str(f.tag) for f in self.faces)
        raise CadIntakeError(f"no face {tag}; this solid has {known}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "faces": [f.as_dict() for f in self.faces],
            "volume": self.volume,
            "area": self.area,
            "bbox_min": list(self.bbox_min),
            "bbox_max": list(self.bbox_max),
            "diagonal": self.diagonal,
            "solid_count": self.solid_count,
            "healed": self.healed,
        }


def check_readable(path: str | Path) -> Path:
    """Refuse an unreadable format by name, before anything is loaded."""
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix in _UNREADABLE:
        raise CadFormatError(
            f"{_UNREADABLE[suffix]} ({suffix}) needs a licensed kernel that "
            "OpenCASCADE does not have. Export the part as STEP (.step) from "
            "your CAD system and screen that instead."
        )
    if suffix not in READABLE_SUFFIXES:
        readable = ", ".join(sorted(READABLE_SUFFIXES))
        raise CadFormatError(
            f"cannot read {suffix or 'a file with no extension'}; "
            f"OpenCASCADE reads {readable}"
        )
    if not resolved.is_file():
        raise CadFormatError(f"no such file: {resolved}")
    return resolved


def import_shapes(path: str | Path) -> bool:
    """Import into the current gmsh model, sewing surfaces only if we must.

    IGES is the reason this exists. It is a surface format in practice -- a box
    written to IGES and read back is six trimmed faces and no solid at all, so
    it imports cleanly and can never be screened. OpenCASCADE will sew those
    faces into a shell and close it into a solid, and on the same box that
    recovers the volume exactly.

    Tried plainly first and healed only on failure, because sewing is a repair:
    it closes gaps within a tolerance, and a file that already has a solid
    should not be quietly rebuilt. Returns whether healing was needed, so the
    caller can say so rather than let a repaired part pass for an exact one.
    """
    import gmsh

    gmsh.model.occ.importShapes(str(path))
    gmsh.model.occ.synchronize()
    if gmsh.model.getEntities(3):
        return False

    gmsh.model.occ.removeAllDuplicates()
    for _, tag in gmsh.model.getEntities(2):
        gmsh.model.occ.remove([(2, tag)], recursive=True)
    gmsh.option.setNumber("Geometry.OCCSewFaces", 1)
    gmsh.option.setNumber("Geometry.OCCMakeSolids", 1)
    gmsh.model.occ.importShapes(str(path))
    gmsh.model.occ.synchronize()
    return bool(gmsh.model.getEntities(3))



def _face_normal(tag: int) -> tuple[float, float, float]:
    """An outward normal, sampled at the middle of the face's parameter range.

    A curved face has no single normal, so this is the normal *somewhere* on it
    and is only meaningful for choosing among faces, never for computing
    anything. `planar` says which kind of face it came from.
    """
    import gmsh

    # Returns two vectors, [u_min, v_min] and [u_max, v_max] -- not four scalars.
    lower, upper = gmsh.model.getParametrizationBounds(2, tag)
    middle = [(lo + hi) / 2.0 for lo, hi in zip(lower, upper)]
    normal = gmsh.model.getNormal(tag, middle)
    vector = np.array(normal[:3], dtype=float)
    length = float(np.linalg.norm(vector))
    if length == 0.0:
        return (0.0, 0.0, 0.0)
    return tuple(float(v) for v in vector / length)


def _is_planar(tag: int) -> bool:
    import gmsh

    try:
        return gmsh.model.getType(2, tag).lower() == "plane"
    except Exception:       # pragma: no cover - older gmsh builds
        return False


def inspect(path: str | Path) -> Solid:
    """Import a CAD file and describe every face of its solid.

    Deliberately does not mesh. Meshing a part only to find out it has no solid,
    or to count its faces, is minutes of work for something the CAD kernel
    already knows.
    """
    import gmsh

    resolved = check_readable(path)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("intake")
        healed = import_shapes(resolved)

        volumes = gmsh.model.getEntities(3)
        if not volumes:
            raise CadIntakeError(
                "the file contains no solid, and sewing its surfaces into one "
                "did not close them. A surface or wireframe model cannot be "
                "screened; it has no volume to carry stress."
            )

        corners = np.array([gmsh.model.getBoundingBox(3, tag) for _, tag in volumes])
        bbox_min = corners[:, :3].min(axis=0)
        bbox_max = corners[:, 3:].max(axis=0)

        faces = []
        for _, tag in gmsh.model.getEntities(2):
            area = float(gmsh.model.occ.getMass(2, tag))
            centre = gmsh.model.occ.getCenterOfMass(2, tag)
            box = gmsh.model.getBoundingBox(2, tag)
            faces.append(
                Face(
                    tag=int(tag),
                    area=area,
                    centroid=tuple(float(c) for c in centre),
                    normal=_face_normal(tag),
                    bbox_min=tuple(float(v) for v in box[:3]),
                    bbox_max=tuple(float(v) for v in box[3:]),
                    planar=_is_planar(tag),
                )
            )
        if not faces:
            raise CadIntakeError("the solid has no faces")

        return Solid(
            path=resolved,
            # Largest first: the face a person means is usually a big one, and a
            # listing that opens with slivers is unreadable.
            faces=tuple(sorted(faces, key=lambda f: -f.area)),
            solid_count=len(volumes),
            healed=healed,
            volume=float(sum(gmsh.model.occ.getMass(3, tag) for _, tag in volumes)),
            area=float(sum(f.area for f in faces)),
            bbox_min=tuple(float(v) for v in bbox_min),
            bbox_max=tuple(float(v) for v in bbox_max),
        )
    finally:
        gmsh.finalize()


def nodes_of_faces(face_tags: tuple[int, ...], index: dict[int, int]) -> np.ndarray:
    """Every mesh node belonging to the given faces, as a boolean mask source.

    Mirrors `generate/harness.py:_nodes_of_faces`, which is how the training
    sweep turned a `Tagging` into a mask. Held here as well so the deployed path
    and the training path select nodes by the same rule rather than by two rules
    that happen to agree on some families.
    """
    import gmsh

    found: set[int] = set()
    for tag in face_tags:
        tags, _, _ = gmsh.model.mesh.getNodes(2, int(tag), includeBoundary=True)
        found.update(index[int(t)] for t in tags)
    return np.array(sorted(found), dtype=np.int64)


def summarise(solid: Solid, limit: int = 20) -> Iterator[str]:
    """One readable line per face, for an agent to show a person."""
    extent = solid.extent
    yield (
        f"{len(solid.faces)} faces, volume {solid.volume:.4g}, "
        f"bounding box {extent[0]:.4g} x {extent[1]:.4g} x {extent[2]:.4g}"
    )
    if solid.healed:
        yield (
            "  NOTE: this file held no solid; its surfaces were sewn into one. "
            "That is a repair within a tolerance, not an exact read."
        )
    if solid.solid_count > 1:
        yield (
            f"  {solid.solid_count} SEPARATE SOLIDS -- this is an assembly, not "
            "a part. Screening and solving both need one body at a time."
        )
    for face in solid.faces[:limit]:
        shape = "planar" if face.planar else "curved"
        cx, cy, cz = face.centroid
        nx, ny, nz = face.normal
        yield (
            f"  face {face.tag:>3}  {shape:>6}  area {face.area:>10.4g}  "
            f"centre ({cx:.4g}, {cy:.4g}, {cz:.4g})  "
            f"normal ({nx:+.2f}, {ny:+.2f}, {nz:+.2f})"
        )
    if len(solid.faces) > limit:
        yield f"  ... and {len(solid.faces) - limit} smaller faces"


__all__ = [
    "READABLE_SUFFIXES",
    "CadFormatError",
    "CadIntakeError",
    "Face",
    "Solid",
    "check_readable",
    "inspect",
    "nodes_of_faces",
    "summarise",
]
