"""The VTU a reviewer opens in ParaView holds exactly what the pictures showed.

Read back with the standard library only (xml.etree + base64), the way any VTK
reader would see it: counts, cell types, the gmsh -> VTK node reorder of the
10-node tetrahedron, and every array value round-trips bit for bit. The reorder
is checked geometrically -- VTK's midside node 8 must sit between corners 1 and
3 -- because a wrong order still parses and only shows up as a twisted element.

Light: numpy only, no mesher, no VTK.
"""

from __future__ import annotations

import base64
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from fea_surrogate.structural_export import (
    GMSH_TO_VTK_TET10,
    STRESS_COMPONENTS,
    VTK_QUADRATIC_TETRA,
    VTK_TETRA,
    StructuralExportError,
    write_vtu,
)

#: gmsh tet10 midside nodes: 4:(0,1) 5:(1,2) 6:(0,2) 7:(0,3) 8:(2,3) 9:(1,3).
GMSH_EDGES = ((0, 1), (1, 2), (0, 2), (0, 3), (2, 3), (1, 3))
#: VTK quadratic tetra midside nodes: (0,1) (1,2) (0,2) (0,3) (1,3) (2,3).
VTK_EDGES = ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))

_NUMPY = {"Float64": "<f8", "Int64": "<i8", "UInt8": "u1"}


@dataclass(frozen=True)
class _Mesh:
    nodes: np.ndarray
    elements: np.ndarray


@dataclass(frozen=True)
class _Solution:
    mesh: _Mesh
    displacement_mm: np.ndarray
    total_deformation_mm: Any
    von_mises_mpa: np.ndarray
    stress_mpa: np.ndarray


def _two_tet10() -> _Mesh:
    """Two quadratic tets sharing a face, in gmsh node order, midsides exact."""
    corners = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10], [10, 10, 10]], dtype=float)
    tets = [(0, 1, 2, 3), (1, 2, 3, 4)]
    nodes = [*corners]
    index: dict[tuple[int, int], int] = {}
    elements = []
    for tet in tets:
        row = list(tet)
        for a, b in GMSH_EDGES:
            key = tuple(sorted((tet[a], tet[b])))
            if key not in index:
                index[key] = len(nodes)
                nodes.append(0.5 * (corners[key[0]] + corners[key[1]]))
            row.append(index[key])
        elements.append(row)
    return _Mesh(np.array(nodes), np.array(elements, dtype=np.int64))


def _solution(mesh: _Mesh, seed: int = 0) -> _Solution:
    rng = np.random.default_rng(seed)
    n = len(mesh.nodes)
    u = rng.normal(size=(n, 3)) * 1e-3
    stress = rng.normal(size=(n, 6)) * 50.0
    vm = np.abs(rng.normal(size=n)) * 100.0
    return _Solution(mesh, u, np.linalg.norm(u, axis=1), vm, stress)


def _read(path) -> dict[str, Any]:
    """Every DataArray decoded: base64 of (UInt64 byte count + raw bytes)."""
    root = ET.parse(path).getroot()
    assert root.tag == "VTKFile"
    assert root.get("type") == "UnstructuredGrid"
    assert root.get("byte_order") == "LittleEndian"
    assert root.get("header_type") == "UInt64"
    piece = root.find("UnstructuredGrid/Piece")
    out: dict[str, Any] = {
        "points": int(piece.get("NumberOfPoints")),
        "cells": int(piece.get("NumberOfCells")),
        "point_data": piece.find("PointData"),
    }
    for array in root.iter("DataArray"):
        assert array.get("format") == "binary"
        blob = base64.b64decode(array.text.strip())
        size = int(np.frombuffer(blob[:8], dtype="<u8")[0])
        assert size == len(blob) - 8
        values = np.frombuffer(blob[8:], dtype=_NUMPY[array.get("type")])
        components = int(array.get("NumberOfComponents", "1"))
        out[array.get("Name")] = values.reshape(-1, components) if components > 1 else values
        out[array.get("Name") + "#attrs"] = dict(array.attrib)
    return out


def test_a_tet10_solution_round_trips_through_the_file(tmp_path) -> None:
    mesh = _two_tet10()
    solution = _solution(mesh)
    data = _read(write_vtu(solution, tmp_path / "out" / "results.vtu"))

    assert data["points"] == len(mesh.nodes) and data["cells"] == 2
    assert np.array_equal(data["Points"], mesh.nodes)
    assert np.array_equal(data["Displacement"], solution.displacement_mm)
    assert np.array_equal(data["Total Deformation"], solution.total_deformation_mm)
    assert np.array_equal(data["Equivalent Stress"], solution.von_mises_mpa)
    assert np.array_equal(data["Stress"], solution.stress_mpa)
    assert data["types"].tolist() == [VTK_QUADRATIC_TETRA] * 2
    assert data["offsets"].tolist() == [10, 20]
    assert data["connectivity"].reshape(-1, 10).tolist() == \
        mesh.elements[:, list(GMSH_TO_VTK_TET10)].tolist()


def test_the_reorder_puts_each_vtk_midside_node_on_its_own_edge(tmp_path) -> None:
    mesh = _two_tet10()
    data = _read(write_vtu(_solution(mesh), tmp_path / "r.vtu"))
    cells = data["connectivity"].reshape(-1, 10)
    points = data["Points"]

    assert GMSH_TO_VTK_TET10 == (0, 1, 2, 3, 4, 5, 6, 7, 9, 8)
    for cell in cells:
        for slot, (a, b) in enumerate(VTK_EDGES, start=4):
            assert np.allclose(points[cell[slot]], 0.5 * (points[cell[a]] + points[cell[b]])), slot


def test_vtk_volume_is_positive_for_gmsh_oriented_tets(tmp_path) -> None:
    """VTK wants corners 0,1,2 counter-clockwise seen from 3 -- gmsh's own rule."""
    mesh = _two_tet10()
    corners = mesh.nodes[mesh.elements[:, :4]]
    volume = np.einsum("ij,ij->i", np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
                       corners[:, 3] - corners[:, 0])
    assert np.all(volume > 0)


def test_point_data_is_named_for_paraview(tmp_path) -> None:
    data = _read(write_vtu(_solution(_two_tet10()), tmp_path / "r.vtu"))
    names = [a.get("Name") for a in data["point_data"].iter("DataArray")]

    assert names == ["Displacement", "Total Deformation", "Equivalent Stress", "Stress"]
    assert data["point_data"].get("Scalars") == "Equivalent Stress"
    assert data["point_data"].get("Vectors") == "Displacement"
    attrs = data["Stress#attrs"]
    assert attrs["NumberOfComponents"] == "6"
    assert [attrs[f"ComponentName{i}"] for i in range(6)] == list(STRESS_COMPONENTS)


def test_a_first_order_mesh_is_written_as_plain_tetrahedra(tmp_path) -> None:
    quadratic = _two_tet10()
    linear = _Mesh(quadratic.nodes[:5], quadratic.elements[:, :4])
    data = _read(write_vtu(_solution(linear), tmp_path / "p1.vtu"))

    assert data["types"].tolist() == [VTK_TETRA] * 2
    assert data["offsets"].tolist() == [4, 8]
    assert data["connectivity"].reshape(-1, 4).tolist() == linear.elements.tolist()


def test_total_deformation_is_derived_when_absent(tmp_path) -> None:
    mesh = _two_tet10()
    full = _solution(mesh)
    bare = _Solution(mesh, full.displacement_mm, None, full.von_mises_mpa, full.stress_mpa)
    data = _read(write_vtu(bare, tmp_path / "r.vtu"))
    assert np.allclose(data["Total Deformation"], np.linalg.norm(full.displacement_mm, axis=1))


def test_the_file_names_its_units(tmp_path) -> None:
    text = write_vtu(_solution(_two_tet10()), tmp_path / "r.vtu").read_text()
    assert "Units: mm" in text and "MPa" in text


def test_a_non_finite_result_is_refused(tmp_path) -> None:
    mesh = _two_tet10()
    good = _solution(mesh)
    vm = good.von_mises_mpa.copy()
    vm[0] = np.inf
    with pytest.raises(StructuralExportError, match="not finite"):
        write_vtu(_Solution(mesh, good.displacement_mm, good.total_deformation_mm, vm,
                            good.stress_mpa), tmp_path / "x.vtu")
    assert not (tmp_path / "x.vtu").exists()


def test_fields_that_do_not_match_the_mesh_are_refused(tmp_path) -> None:
    mesh = _two_tet10()
    good = _solution(mesh)
    with pytest.raises(StructuralExportError, match="do not belong together"):
        write_vtu(_Solution(mesh, good.displacement_mm[:-1], None, good.von_mises_mpa,
                            good.stress_mpa), tmp_path / "x.vtu")
    with pytest.raises(StructuralExportError, match="tetrahedra"):
        write_vtu(_solution(_Mesh(mesh.nodes, mesh.elements[:, :8])), tmp_path / "x.vtu")
    with pytest.raises(StructuralExportError, match="does not have"):
        write_vtu(_solution(_Mesh(mesh.nodes, mesh.elements + 100)), tmp_path / "x.vtu")
