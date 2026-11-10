"""The analysis mesh: the numbers a load names must be the faces it lands on.

A static structural answer is only as good as the mesh's bookkeeping. A face
number that means something different here than in the face table, a mid-edge
node in the wrong slot, a triangle wound inward (a pressure that pulls instead
of pushes), or a mesh quietly coarsened past one element through a wall would
each produce a healthy-looking contour plot of the wrong question. Every one of
those is checked here against something computed independently.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh", reason="meshing requires the inference extra")

from fea_surrogate import structural_mesh as sm  # noqa: E402
from fea_surrogate.cad_intake import CadFormatError, inspect  # noqa: E402
from fea_surrogate.face_selectors import face_mask, select_faces  # noqa: E402
from fea_surrogate.structural_mesh import (  # noqa: E402
    REFINEMENTS,
    TET10_EDGES,
    TRI6_EDGES,
    AnalysisMeshError,
    element_size_for,
    mesh_for_analysis,
    triangle_areas,
)

BOX = (100.0, 20.0, 10.0)
HOLE_CENTRE = (50.0, 20.0)
HOLE_RADIUS = 6.0


def _write(path: Path, build) -> Path:  # noqa: ANN001
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(path.stem)
        build(gmsh.model.occ)
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


def _box(path: Path, dx: float = BOX[0], dy: float = BOX[1], dz: float = BOX[2]) -> Path:
    return _write(path, lambda occ: occ.addBox(0.0, 0.0, 0.0, dx, dy, dz))


def _plate_with_hole(path: Path) -> Path:
    def build(occ):  # noqa: ANN001, ANN202
        plate = occ.addBox(0.0, 0.0, 0.0, 100.0, 40.0, 10.0)
        bore = occ.addCylinder(*HOLE_CENTRE, 0.0, 0.0, 0.0, 10.0, HOLE_RADIUS)
        occ.cut([(3, plate)], [(3, bore)])
    return _write(path, build)


@pytest.fixture(scope="module")
def box_path(tmp_path_factory) -> Path:
    return _box(tmp_path_factory.mktemp("box") / "box.step")


@pytest.fixture(scope="module")
def box(box_path) -> sm.AnalysisMesh:
    return mesh_for_analysis(box_path, refinement="coarse")


@pytest.fixture(scope="module")
def plate_path(tmp_path_factory) -> Path:
    return _plate_with_hole(tmp_path_factory.mktemp("plate") / "plate.step")


@pytest.fixture(scope="module")
def plate(plate_path) -> sm.AnalysisMesh:
    return mesh_for_analysis(plate_path, refinement="coarse")


def _bore_tag(mesh: sm.AnalysisMesh) -> int:
    (tag,) = [f.tag for f in mesh.faces if not f.planar]
    return tag


# --------------------------------------------------------------------------
# Node ordering
# --------------------------------------------------------------------------


def test_tet10_mid_edge_nodes_sit_at_their_edge_midpoints(box) -> None:
    """gmsh "Tetrahedron 10": node 4+k on corner edge TET10_EDGES[k]. On a box
    every edge is straight, so a misread order shows as a node far off its edge."""
    x = box.nodes
    assert box.order == 2 and box.elements.shape[1] == 10
    for k, (i, j) in enumerate(TET10_EDGES):
        middle = 0.5 * (x[box.elements[:, i]] + x[box.elements[:, j]])
        assert np.abs(x[box.elements[:, 4 + k]] - middle).max() < 1e-9


def test_tri6_mid_edge_nodes_follow_gmsh_order(box) -> None:
    x = box.nodes
    assert box.boundary.shape[1] == 6
    for k, (i, j) in enumerate(TRI6_EDGES):
        middle = 0.5 * (x[box.boundary[:, i]] + x[box.boundary[:, j]])
        assert np.abs(x[box.boundary[:, 3 + k]] - middle).max() < 1e-9


def test_a_first_order_mesh_has_four_node_elements(box_path) -> None:
    mesh = mesh_for_analysis(box_path, order=1, refinement="coarse")

    assert mesh.order == 1
    assert mesh.elements.shape[1] == 4 and mesh.boundary.shape[1] == 3
    assert mesh.corner_mask.all()
    assert mesh.dof_count == 3 * mesh.node_count


def test_corner_mask_marks_exactly_the_tetrahedron_vertices(box) -> None:
    corners = np.zeros(box.node_count, dtype=bool)
    corners[box.elements[:, :4].ravel()] = True
    assert np.array_equal(box.corner_mask, corners)
    # Mid-edge nodes are not corners: a 10-node mesh has several times more.
    assert box.node_count > 3 * int(corners.sum())


# --------------------------------------------------------------------------
# Faces
# --------------------------------------------------------------------------


def test_face_numbers_and_centroids_are_the_face_tables(box_path, box) -> None:
    """The face table comes from `cad_intake.inspect`; a load names those numbers."""
    solid = inspect(box_path)

    assert box.face_tags == tuple(sorted(f.tag for f in solid.faces))
    for face in solid.faces:
        assert box.face_centroids[face.tag] == pytest.approx(face.centroid, abs=1e-12)
        assert box.face(face.tag).area == pytest.approx(face.area, rel=1e-12)
    # Same order as the table: largest first.
    assert [f.tag for f in box.faces] == [f.tag for f in solid.faces]


def test_each_faces_triangles_cover_exactly_its_area(box_path, box) -> None:
    """Per-entity triangles, not a node-mask reconstruction that can leak load
    onto a neighbouring face."""
    solid = inspect(box_path)
    areas = triangle_areas(box.nodes, box.boundary)

    for face in solid.faces:
        mine = box.boundary_face == face.tag
        assert areas[mine].sum() == pytest.approx(face.area, rel=1e-9)
        assert np.array_equal(box.face_triangles([face.tag]), box.boundary[mine])
    assert areas.sum() == pytest.approx(sum(f.area for f in solid.faces), rel=1e-9)


def test_face_nodes_are_the_nodes_of_the_faces_triangles(box) -> None:
    for tag in box.face_tags:
        assert np.array_equal(box.face_nodes[tag], np.unique(box.face_triangles([tag])))


def test_boundary_triangles_point_out_of_the_solid(box) -> None:
    """Checked two independent ways: against each box face's outward axis, and
    against the fourth vertex of the tetrahedron that owns the triangle."""
    x = box.nodes
    a, b, c = (x[box.boundary[:, i]] for i in range(3))
    normals = np.cross(b - a, c - a)
    for face in box.faces:
        mine = box.boundary_face == face.tag
        assert (normals[mine] @ np.asarray(face.normal) > 0).all()

    opposite = {}
    for tet in box.elements[:, :4].tolist():
        for k in range(4):
            opposite[tuple(sorted(tet[:k] + tet[k + 1:]))] = tet[k]
    fourth = np.array([opposite[tuple(sorted(t))] for t in box.boundary[:, :3].tolist()])
    assert (np.einsum("ij,ij->i", normals, x[fourth] - a) < 0).all()


def test_inward_triangles_are_flipped_back_out(box) -> None:
    """gmsh's winding was outward on every part measured, which is an
    observation rather than a contract -- so the mesher checks it."""
    inverted = box.boundary.copy()
    inverted[::2] = sm._flip(inverted[::2])
    assert not np.array_equal(inverted, box.boundary)

    repaired = sm._orient_outward(box.nodes, box.elements, inverted)

    assert np.array_equal(repaired, box.boundary)


def test_box_faces_are_planar_with_axis_normals(box) -> None:
    lo, hi = np.zeros(3), np.array(BOX)
    for face in box.faces:
        assert face.planar and face.kind == "plane"
        axis = int(np.argmax(np.abs(face.normal)))
        assert abs(face.normal[axis]) == pytest.approx(1.0, abs=1e-12)
        # The normal points away from the box: at the min side it is negative.
        at_min = face.centroid[axis] == pytest.approx(lo[axis])
        assert (face.normal[axis] < 0) == at_min
        assert face.bbox_min[axis] == pytest.approx(face.bbox_max[axis])
        assert face.bbox_max[axis] == pytest.approx(lo[axis] if at_min else hi[axis])


def test_a_bore_is_cylinder_like_with_no_single_normal(plate_path, plate) -> None:
    """A closed bore's normals cancel; a normalised rounding error would be a
    confident wrong answer, so the normal is reported as zero."""
    bore = plate.face(_bore_tag(plate))
    solid = inspect(plate_path)

    assert not bore.planar
    assert bore.kind == "cylinder-like"
    assert bore.normal == (0.0, 0.0, 0.0)
    assert bore.area == pytest.approx(2 * math.pi * HOLE_RADIUS * 10.0, rel=1e-9)
    assert bore.area == pytest.approx(solid.face(bore.tag).area, rel=1e-12)
    others = [f for f in plate.faces if f.tag != bore.tag]
    assert all(f.planar and f.kind == "plane" for f in others)


def test_curved_mid_edge_nodes_lie_on_the_bore(plate) -> None:
    """Order 2 snaps mid-edge nodes onto the CAD surface, so a hole stays round."""
    nodes = plate.nodes[plate.face_nodes[_bore_tag(plate)]]
    radius = np.linalg.norm(nodes[:, :2] - np.array(HOLE_CENTRE), axis=1)
    assert np.abs(radius - HOLE_RADIUS).max() < 1e-6 * HOLE_RADIUS
    assert not plate.straight_sided


def test_face_lookup_names_the_valid_faces(box) -> None:
    with pytest.raises(AnalysisMeshError, match="faces are 1-6"):
        box.face(99)


def test_faces_at_a_node_lists_every_face_through_it(box) -> None:
    corner = int(np.argmin(np.linalg.norm(box.nodes, axis=1)))       # the (0,0,0) vertex
    assert len(box.faces_at(corner)) == 3
    interior = np.setdiff1d(np.arange(box.node_count), np.unique(box.boundary))
    assert interior.size == 0 or box.faces_at(int(interior[0])) == ()


def test_face_summaries_serialise(box) -> None:
    text = json.dumps([f.as_dict() for f in box.faces])
    assert '"kind": "plane"' in text


# --------------------------------------------------------------------------
# Edges
# --------------------------------------------------------------------------


def test_cad_edges_are_the_boxs_twelve_edges(box) -> None:
    x = box.nodes
    hi = np.array(BOX)
    segments = x[box.edges]                                           # (E,2,3)
    on_extreme = (np.isclose(segments, 0.0, atol=1e-9) | np.isclose(segments, hi, atol=1e-9))
    # Every segment end lies on a box edge: at least two coordinates at an extreme.
    assert (on_extreme.sum(axis=2) >= 2).all()
    length = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1).sum()
    assert length == pytest.approx(4 * sum(BOX), rel=1e-12)


def test_a_bores_seam_is_not_drawn_as_an_edge(plate) -> None:
    """OpenCASCADE closes a cylinder with a seam line; it is not an edge of the part."""
    x = plate.nodes[plate.edges]
    radius = np.linalg.norm(x[..., :2] - np.array(HOLE_CENTRE), axis=2)
    along_bore = (np.abs(radius - HOLE_RADIUS) < 1e-6).all(axis=1) & (
        np.abs(x[:, 0, 2] - x[:, 1, 2]) > 1e-9
    )
    assert not along_bore.any()
    # ...while both circular edges of the bore are there.
    circular = (np.abs(radius - HOLE_RADIUS) < 1e-6).all(axis=1)
    assert np.unique(np.round(x[circular][..., 2], 9)).tolist() == [0.0, 10.0]


# --------------------------------------------------------------------------
# Quality, sizing
# --------------------------------------------------------------------------


def test_no_element_is_inverted(box, plate) -> None:
    for mesh in (box, plate):
        assert mesh.min_scaled_jacobian > 0.0
        corners = mesh.nodes[mesh.elements[:, :4]]
        volume = np.einsum(
            "ij,ij->i", np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
            corners[:, 3] - corners[:, 0],
        )
        assert (volume > 0).all()


def test_thinnest_follows_the_surrogate_meshs_rule(box_path, box) -> None:
    """min(smallest CAD-box extent, 2V/A), computed here from the box's own
    dimensions rather than from anything the mesher reports."""
    dx, dy, dz = BOX
    volume = dx * dy * dz
    area = 2.0 * (dx * dy + dy * dz + dx * dz)
    expected = min(min(BOX), 2.0 * volume / area)

    assert box.thinnest == pytest.approx(expected, rel=1e-12)
    assert box.volume_mm3 == pytest.approx(volume, rel=1e-12)
    assert box.diagonal_mm == pytest.approx(math.sqrt(dx * dx + dy * dy + dz * dz), rel=1e-12)


def test_thinnest_agrees_with_features_mesh_step(box_path, box) -> None:
    pytest.importorskip("torch", reason="features imports torch_geometric")
    from fea_surrogate.features import mesh_step

    assert box.thinnest == pytest.approx(mesh_step(box_path).thinnest, rel=1e-12)


def test_the_surrogates_mesh_is_untouched_by_an_analysis_mesh(box_path) -> None:
    """gmsh options are process-global; the curvature sizing set here must not
    leak into the mesh the surrogate was calibrated on."""
    pytest.importorskip("torch", reason="features imports torch_geometric")
    from fea_surrogate.features import mesh_step

    before = mesh_step(box_path)
    mesh_for_analysis(box_path, refinement="fine")
    after = mesh_step(box_path)

    assert np.array_equal(before.vertices, after.vertices)
    assert np.array_equal(before.tetrahedra, after.tetrahedra)


def test_refinement_sizes_are_monotonic() -> None:
    for extent, volume, area in (
        (np.array([100.0, 10.0, 10.0]), 1e4, 4200.0),
        (np.array([109.4, 67.2, 31.7]), 4.2e4, 2.4e4),      # a bracket-like plate
        (np.array([20.0, 20.0, 20.0]), 8e3, 2400.0),        # a cube: the bounds win
    ):
        coarse, medium, fine = (element_size_for(extent, volume, area, r)
                                for r in ("coarse", "medium", "fine"))
        assert coarse > medium > fine > 0.0


def test_medium_puts_one_element_through_the_thinnest_wall() -> None:
    extent = np.array([100.0, 10.0, 10.0])
    volume, area = 1e4, 4200.0
    thinnest = 2 * volume / area
    diagonal = float(np.linalg.norm(extent))

    assert element_size_for(extent, volume, area, "medium") == pytest.approx(thinnest)
    assert element_size_for(extent, volume, area, "coarse") == pytest.approx(1.6 * thinnest)
    assert element_size_for(extent, volume, area, "fine") == pytest.approx(
        max(thinnest / 2, diagonal / 100)
    )
    assert set(REFINEMENTS) == {"coarse", "medium", "fine"}


def test_finer_refinement_gives_more_nodes(box_path, box) -> None:
    medium = mesh_for_analysis(box_path, refinement="medium")
    assert medium.element_size_mm < box.element_size_mm
    assert medium.node_count > box.node_count


def test_an_element_size_overrides_the_refinement(box_path) -> None:
    mesh = mesh_for_analysis(box_path, refinement="fine", element_size_mm=12.0)
    assert mesh.element_size_mm == 12.0


def test_straight_sided_fallback_when_curved_elements_invert(plate_path, monkeypatch) -> None:
    """If curving mid-edge nodes onto the CAD inverts an element and gmsh's
    optimiser cannot fix it, the mesh is rebuilt straight-sided -- and says so."""
    real = sm._min_scaled_jacobian
    calls = {"n": 0}

    def inverted_twice() -> float:
        calls["n"] += 1
        return -0.5 if calls["n"] <= 2 else real()

    monkeypatch.setattr(sm, "_min_scaled_jacobian", inverted_twice)
    mesh = mesh_for_analysis(plate_path, refinement="coarse")

    assert calls["n"] == 3
    assert mesh.straight_sided and mesh.min_scaled_jacobian > 0.0
    x = mesh.nodes
    for k, (i, j) in enumerate(TET10_EDGES):
        middle = 0.5 * (x[mesh.elements[:, i]] + x[mesh.elements[:, j]])
        assert np.abs(x[mesh.elements[:, 4 + k]] - middle).max() < 1e-9


def test_a_mesh_that_stays_inverted_is_refused(plate_path, monkeypatch) -> None:
    monkeypatch.setattr(sm, "_min_scaled_jacobian", lambda: -0.5)
    with pytest.raises(AnalysisMeshError, match="inverted"):
        mesh_for_analysis(plate_path, refinement="coarse")


# --------------------------------------------------------------------------
# Interop and refusals
# --------------------------------------------------------------------------


def test_face_selectors_work_on_an_analysis_mesh(box) -> None:
    """Same field names as ScaledMesh, so the family presets resolve unchanged."""
    (held,) = select_faces({"centroid": {"axis": "x", "at": "min"}}, box, "fixed")
    assert box.face(held).normal[0] == pytest.approx(-1.0)
    mask = face_mask((held,), box)
    assert mask.sum() == box.face_nodes[held].size
    assert np.allclose(box.vertices[mask, 0], 0.0)


def test_several_solids_are_counted(tmp_path) -> None:
    def build(occ):  # noqa: ANN001, ANN202
        occ.addBox(0.0, 0.0, 0.0, 30.0, 10.0, 10.0)
        occ.addBox(50.0, 0.0, 0.0, 30.0, 10.0, 10.0)

    mesh = mesh_for_analysis(_write(tmp_path / "pair.step", build), refinement="coarse")
    assert mesh.solid_count == 2


def test_a_surface_with_no_solid_is_refused(tmp_path) -> None:
    sheet = _write(tmp_path / "sheet.step", lambda occ: occ.addRectangle(0, 0, 0, 10, 10))
    with pytest.raises(AnalysisMeshError, match="no solid"):
        mesh_for_analysis(sheet)


def test_an_unreadable_format_is_refused_before_meshing(tmp_path) -> None:
    part = tmp_path / "part.sldprt"
    part.write_bytes(b"not a step file")
    with pytest.raises(CadFormatError, match="STEP"):
        mesh_for_analysis(part)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"order": 3}, "element order"),
        ({"order": True}, "element order"),
        ({"refinement": "ultra"}, "refinement"),
        ({"element_size_mm": -1.0}, "positive"),
        ({"element_size_mm": float("nan")}, "positive"),
        ({"element_size_mm": "big"}, "millimetres"),
    ],
)
def test_bad_mesh_settings_are_refused(box_path, kwargs, match) -> None:
    with pytest.raises(AnalysisMeshError, match=match):
        mesh_for_analysis(box_path, **kwargs)


def test_a_runaway_element_size_is_refused_before_meshing(box_path) -> None:
    """0.01 mm on a 100 mm part is minutes of meshing into a system no direct
    solver could factor; say so instead of hanging."""
    with pytest.raises(AnalysisMeshError, match="larger element size"):
        mesh_for_analysis(box_path, element_size_mm=0.01)


def test_element_quality_is_the_ansys_shape_measure(box) -> None:
    """R2: gmsh's minSJ is 1 for every straight-sided tetrahedron, slivers too,
    so it cannot be the quality line; Ansys' Element Quality can."""
    a = 1.0
    regular = np.array([[0, 0, 0], [a, 0, 0], [a / 2, a * math.sqrt(3) / 2, 0],
                        [a / 2, a * math.sqrt(3) / 6, a * math.sqrt(2.0 / 3.0)]])
    sliver = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1e-3]])
    nodes = np.vstack([regular, sliver])
    quality = sm.element_quality(nodes, np.array([[0, 1, 2, 3], [4, 5, 6, 7]]))
    assert quality[0] == pytest.approx(1.0, abs=1e-6)
    assert quality[1] < 0.01

    assert 0.0 < box.element_quality_min <= box.element_quality_median <= 1.0
    assert box.min_scaled_jacobian > 0.0
