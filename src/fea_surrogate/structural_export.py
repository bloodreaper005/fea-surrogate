"""Export a static structural solution for ParaView: one self-contained .vtu.

The PNGs are for reading; this file is for checking. A reviewer who wants to
probe a node, slice the part or plot stress along a path opens the VTU in
ParaView and gets the same nodal fields the pictures were drawn from.

VTK XML UnstructuredGrid, binary arrays base64-encoded inline (no appended
section, no compression), so the file is one piece of text that needs nothing
but numpy and the standard library to write -- no `vtk` or `meshio` dependency.

Element mapping: 10-node tetrahedra are VTK_QUADRATIC_TETRA (24). gmsh and VTK
agree on the corners and the first four midside nodes but swap the last two:
gmsh orders them 8:(2,3) 9:(1,3), VTK wants (1,3) then (2,3), hence the
reorder [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]. Getting this wrong does not fail
loudly -- ParaView draws a subtly twisted element -- so it is pinned by a test.
4-node tetrahedra are VTK_TETRA (10) with no reorder.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr

import numpy as np


VTK_TETRA = 10
VTK_QUADRATIC_TETRA = 24

#: gmsh tet10 node order -> VTK quadratic tetra node order.
GMSH_TO_VTK_TET10: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 9, 8)

#: Stress components as the solver stores them, which is also VTK's symmetric
#: tensor order, so ParaView recognises the 6-component array as a tensor.
STRESS_COMPONENTS: tuple[str, ...] = ("XX", "YY", "ZZ", "XY", "YZ", "XZ")


class StructuralExportError(ValueError):
    """The solution cannot be written as asked; the message says why."""


def _array(name: str, data: np.ndarray, vtk_type: str, components: int = 1,
           component_names: tuple[str, ...] = ()) -> str:
    """One <DataArray> with its bytes as base64(UInt64 byte count + raw data)."""
    raw = np.ascontiguousarray(data).tobytes()
    payload = base64.b64encode(np.uint64(len(raw)).tobytes() + raw).decode("ascii")
    attrs = [f'type="{vtk_type}"', f"Name={quoteattr(name)}"]
    if components != 1:
        attrs.append(f'NumberOfComponents="{components}"')
    for i, comp in enumerate(component_names):
        attrs.append(f"ComponentName{i}={quoteattr(comp)}")
    attrs.append('format="binary"')
    return f"        <DataArray {' '.join(attrs)}>\n          {payload}\n        </DataArray>\n"


def _point_field(solution: Any, attr: str, shape: tuple[int, ...], role: str) -> np.ndarray:
    value = getattr(solution, attr, None)
    if value is None:
        raise StructuralExportError(f"the solution has no {attr} ({role}) to export")
    array = np.asarray(value, dtype="<f8")
    if array.shape != shape:
        raise StructuralExportError(
            f"{role} has shape {array.shape} but the mesh needs {shape}; the solution and "
            "its mesh do not belong together"
        )
    if not np.all(np.isfinite(array)):
        raise StructuralExportError(
            f"{role} is not finite; a failed solve must not be exported as a result"
        )
    return array


def write_vtu(solution: Any, path: str | Path) -> Path:
    """Write nodes, elements and the nodal results to a ParaView-readable .vtu.

    Point data: "Displacement" (mm, 3 components), "Total Deformation" (mm),
    "Equivalent Stress" (MPa, von Mises from nodal-averaged components) and
    "Stress" (MPa, XX YY ZZ XY YZ XZ). Units are millimetres, newtons and MPa,
    the same as everywhere else in the analysis, and are named in a comment.
    """
    mesh = solution.mesh
    nodes = np.asarray(mesh.nodes, dtype="<f8")
    if nodes.ndim != 2 or nodes.shape[1] != 3 or len(nodes) == 0:
        raise StructuralExportError(f"mesh nodes must be (N,3), not {nodes.shape}")
    elements = np.asarray(mesh.elements, dtype=np.int64)
    if elements.ndim != 2 or elements.shape[1] not in (4, 10) or len(elements) == 0:
        raise StructuralExportError(
            f"elements must be (M,4) or (M,10) tetrahedra, not {elements.shape}"
        )
    if elements.min() < 0 or elements.max() >= len(nodes):
        raise StructuralExportError("element connectivity refers to nodes the mesh does not have")
    n, m = len(nodes), len(elements)
    if elements.shape[1] == 10:
        connectivity = elements[:, list(GMSH_TO_VTK_TET10)]
        cell_type = VTK_QUADRATIC_TETRA
    else:
        connectivity = elements
        cell_type = VTK_TETRA
    per_cell = connectivity.shape[1]

    displacement = _point_field(solution, "displacement_mm", (n, 3), "displacement")
    if getattr(solution, "total_deformation_mm", None) is not None:
        deformation = _point_field(solution, "total_deformation_mm", (n,), "total deformation")
    else:
        deformation = np.linalg.norm(displacement, axis=1)
    von_mises = _point_field(solution, "von_mises_mpa", (n,), "equivalent stress")
    stress = _point_field(solution, "stress_mpa", (n, 6), "stress components")

    point_data = "".join([
        _array("Displacement", displacement, "Float64", 3, ("X", "Y", "Z")),
        _array("Total Deformation", deformation, "Float64"),
        _array("Equivalent Stress", von_mises, "Float64"),
        _array("Stress", stress, "Float64", 6, STRESS_COMPONENTS),
    ])
    document = (
        '<?xml version="1.0"?>\n'
        "<!-- fea-surrogate static structural result. Units: mm (length, displacement), "
        "MPa (stress). Element: "
        + ("10-node quadratic tetrahedra" if cell_type == VTK_QUADRATIC_TETRA
           else "4-node linear tetrahedra")
        + ". Stress is the unweighted nodal average of element-node values. -->\n"
        '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" '
        'header_type="UInt64">\n'
        "  <UnstructuredGrid>\n"
        f'    <Piece NumberOfPoints="{n}" NumberOfCells="{m}">\n'
        '      <PointData Scalars="Equivalent Stress" Vectors="Displacement">\n'
        + point_data
        + "      </PointData>\n"
        "      <Points>\n"
        + _array("Points", nodes, "Float64", 3)
        + "      </Points>\n"
        "      <Cells>\n"
        + _array("connectivity", connectivity.astype("<i8"), "Int64")
        + _array("offsets", (np.arange(1, m + 1, dtype="<i8") * per_cell), "Int64")
        + _array("types", np.full(m, cell_type, dtype=np.uint8), "UInt8")
        + "      </Cells>\n"
        "    </Piece>\n"
        "  </UnstructuredGrid>\n"
        "</VTKFile>\n"
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding="ascii")
    return out


__all__ = [
    "GMSH_TO_VTK_TET10",
    "STRESS_COMPONENTS",
    "StructuralExportError",
    "VTK_QUADRATIC_TETRA",
    "VTK_TETRA",
    "write_vtu",
]
