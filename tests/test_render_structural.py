"""The Ansys-style pictures say what they show, and only that.

A solved stress plot is the most persuasive image in the project, so these read
the text actually burned into the output (SVG text survives as comments) rather
than trusting the code that wrote it: the solved stamp, the result type, the
unit, Max/Min and the deformation scale must be there; the screening words must
not. The geometry helpers underneath (exact isobands, the z-buffer that hides
edges and decides which labels are visible, the tri6 split) are checked
directly, because a subtly wrong band or a label on a hidden face would still
produce a convincing picture.

The mesh and fields here are stand-ins with the AnalysisMesh / StaticSolution
field names (a gmsh OCC box meshed to second order, an analytic cantilever-like
field), so these run without the solver. One test renders from a real solve
when the solver module is present.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh", reason="meshing requires the inference extra")
pytest.importorskip("matplotlib", reason="rendering requires the render extra")

from fea_surrogate import render as screening_render  # noqa: E402
from fea_surrogate.render_structural import (  # noqa: E402
    ANSYS_LEGEND_COLORS,
    DEFAULT_VIEW,
    StructuralRenderError,
    _basis,
    _DepthMap,
    _fit,
    _isobands,
    _surface,
    choose_view,
    compose_board,
    render_face_ids,
    render_result,
    render_setup,
    structural_header,
)
from fea_surrogate.structural_case import (  # noqa: E402
    FixedSupport,
    ForceLoad,
    Gravity,
    PressureLoad,
    StaticStructuralCase,
    material_from_key,
)


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
LENGTH, WIDTH, HEIGHT = 100.0, 12.0, 10.0


# --------------------------------------------------------------------------
# Stand-ins with the spec's field names
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Face:
    tag: int
    area: float
    centroid: tuple[float, float, float]
    normal: tuple[float, float, float]
    planar: bool = True
    kind: str = "plane"


@dataclass(frozen=True)
class _Mesh:
    path: Path
    order: int
    nodes: np.ndarray
    elements: np.ndarray
    boundary: np.ndarray
    boundary_face: np.ndarray
    face_nodes: Mapping[int, np.ndarray]
    faces: tuple[_Face, ...]
    edges: np.ndarray
    element_size_mm: float
    thinnest: float
    diagonal_mm: float

    @property
    def vertices(self) -> np.ndarray:
        return self.nodes

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def element_count(self) -> int:
        return len(self.elements)


@dataclass(frozen=True)
class _Hotspot:
    value: float
    node: int
    location_mm: tuple[float, float, float]
    faces: tuple[int, ...] = ()
    singular: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class _Solution:
    mesh: _Mesh
    case: Any
    displacement_mm: np.ndarray
    stress_mpa: np.ndarray
    von_mises_mpa: np.ndarray
    total_deformation_mm: np.ndarray
    max_stress: _Hotspot
    min_stress: _Hotspot
    max_stress_away: _Hotspot | None
    exclusion_radius_mm: float
    max_deformation: _Hotspot
    min_deformation: _Hotspot
    deformation_scale_auto: float
    extra: dict = field(default_factory=dict)


def _box_step(directory: Path, length=LENGTH, width=WIDTH, height=HEIGHT) -> Path:
    path = directory / "beam.step"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("beam")
        gmsh.model.occ.addBox(0.0, 0.0, 0.0, length, width, height)
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


def _mesh(path: Path, size: float = 5.0, order: int = 2) -> _Mesh:
    """Second-order gmsh mesh with the AnalysisMesh fields the renderer reads."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("m")
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size / 4)
        gmsh.model.mesh.generate(3)
        if order == 2:
            gmsh.model.mesh.setOrder(2)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        index = np.zeros(int(tags.max()) + 1, dtype=np.int64)
        index[tags.astype(np.int64)] = np.arange(len(tags))
        nodes = coords.reshape(-1, 3)
        tet, tri, line = (11, 9, 8) if order == 2 else (4, 2, 1)
        per_tet, per_tri, per_line = (10, 6, 3) if order == 2 else (4, 3, 2)
        types, _, blocks = gmsh.model.mesh.getElements(3)
        elements = np.concatenate([index[np.asarray(b, dtype=np.int64)].reshape(-1, per_tet)
                                   for t, b in zip(types, blocks) if t == tet])
        boundary, owner, face_nodes, faces = [], [], {}, []
        for _, ftag in gmsh.model.getEntities(2):
            types, _, blocks = gmsh.model.mesh.getElements(2, ftag)
            for t, b in zip(types, blocks):
                if t == tri:
                    rows = index[np.asarray(b, dtype=np.int64)].reshape(-1, per_tri)
                    boundary.append(rows)
                    owner.append(np.full(len(rows), ftag))
            node_tags, _, _ = gmsh.model.mesh.getNodes(2, ftag, includeBoundary=True)
            face_nodes[int(ftag)] = np.unique(index[node_tags.astype(np.int64)])
        edges = []
        for _, etag in gmsh.model.getEntities(1):
            types, _, blocks = gmsh.model.mesh.getElements(1, etag)
            for t, b in zip(types, blocks):
                if t == line:
                    seg = index[np.asarray(b, dtype=np.int64)].reshape(-1, per_line)
                    edges.append(np.concatenate([seg[:, [0, 2]], seg[:, [2, 1]]])
                                 if order == 2 else seg)
    finally:
        gmsh.finalize()
    boundary = np.concatenate(boundary)
    owner = np.concatenate(owner).astype(np.int64)
    # Outward winding, as AnalysisMesh guarantees: a box's outward side points
    # away from its centre.
    centre = nodes.mean(axis=0)
    normal = np.cross(nodes[boundary[:, 1]] - nodes[boundary[:, 0]],
                      nodes[boundary[:, 2]] - nodes[boundary[:, 0]])
    inward = np.einsum("ij,ij->i", normal, nodes[boundary[:, :3]].mean(axis=1) - centre) < 0
    flip = [0, 2, 1, 5, 4, 3] if order == 2 else [0, 2, 1]
    boundary[inward] = boundary[inward][:, flip]
    for ftag in sorted(face_nodes):
        rows = boundary[owner == ftag]
        cross = np.cross(nodes[rows[:, 1]] - nodes[rows[:, 0]], nodes[rows[:, 2]] - nodes[rows[:, 0]])
        n = cross.sum(axis=0)
        faces.append(_Face(int(ftag), float(0.5 * np.linalg.norm(cross, axis=1).sum()),
                           tuple(nodes[np.unique(rows)].mean(axis=0)), tuple(n / np.linalg.norm(n))))
    extent = nodes.max(axis=0) - nodes.min(axis=0)
    return _Mesh(path=path, order=order, nodes=nodes, elements=elements, boundary=boundary,
                 boundary_face=owner, face_nodes=face_nodes,
                 faces=tuple(sorted(faces, key=lambda f: -f.area)),
                 edges=np.concatenate(edges), element_size_mm=size,
                 thinnest=float(extent.min()), diagonal_mm=float(np.linalg.norm(extent)))


def _face_with_normal(mesh: _Mesh, axis: int, sign: float) -> int:
    return next(f.tag for f in mesh.faces if f.normal[axis] * sign > 0.9)


def _case(mesh: _Mesh, *, pressure: bool = False, gravity: bool = False) -> StaticStructuralCase:
    held = _face_with_normal(mesh, 0, -1.0)
    tip = _face_with_normal(mesh, 0, +1.0)
    top = _face_with_normal(mesh, 2, +1.0)
    loads: list[Any] = [ForceLoad((tip,), (0.0, 0.0, -500.0))]
    if pressure:
        loads.append(PressureLoad((top,), 1.5))
    return StaticStructuralCase(
        supports=(FixedSupport((held,)),), loads=tuple(loads),
        material=material_from_key("steel"), gravity=Gravity() if gravity else None,
    )


def _solution(mesh: _Mesh, case: Any, *, singular: bool = True) -> _Solution:
    """Cantilever-like analytic field: deflection ~ x^2(3L - x), bending stress
    linear through the depth and falling towards the tip."""
    p = mesh.nodes
    x = p[:, 0] / LENGTH
    zc = HEIGHT / 2
    u = np.zeros_like(p)
    u[:, 2] = -2.0 * x ** 2 * (3.0 - x) / 2.0
    u[:, 0] = -6.0 * x * (1.0 - x / 2.0) * (p[:, 2] - zc) / LENGTH
    stress = np.zeros((len(p), 6))
    stress[:, 0] = 300.0 * (1.0 - x) * (p[:, 2] - zc) / zc
    stress[:, 3] = 4.0 * np.sin(3.0 * x)
    vm = np.sqrt(stress[:, 0] ** 2 + 3.0 * stress[:, 3] ** 2)
    td = np.linalg.norm(u, axis=1)

    def spot(values: np.ndarray, node: int, **kw: Any) -> _Hotspot:
        return _Hotspot(float(values[node]), node, tuple(p[node]), **kw)

    away = np.where(x > 0.1, vm, 0.0)
    imax = int(np.argmax(vm))
    return _Solution(
        mesh=mesh, case=case, displacement_mm=u, stress_mpa=stress, von_mises_mpa=vm,
        total_deformation_mm=td,
        max_stress=spot(vm, imax, singular=singular,
                        reason="edge of A Fixed Support (face 1)" if singular else None),
        min_stress=spot(vm, int(np.argmin(vm))),
        max_stress_away=spot(vm, int(np.argmax(away))) if singular else None,
        exclusion_radius_mm=mesh.thinnest,
        max_deformation=spot(td, int(np.argmax(td))),
        min_deformation=spot(td, int(np.argmin(td))),
        deformation_scale_auto=0.05 * LENGTH / float(td.max()),
    )


@pytest.fixture(scope="module")
def beam(tmp_path_factory) -> _Mesh:
    return _mesh(_box_step(tmp_path_factory.mktemp("beam")))


@pytest.fixture(scope="module")
def solved(beam) -> _Solution:
    return _solution(beam, _case(beam))


def _svg(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Result pictures
# --------------------------------------------------------------------------


def test_equivalent_stress_says_what_it_is_and_that_it_is_solved(solved, tmp_path) -> None:
    text = _svg(render_result(solved, "equivalent_stress", tmp_path / "s.svg"))

    for needed in ("A: Static Structural", "Equivalent Stress",
                   "Type: Equivalent (von-Mises) Stress", "Unit: MPa", "Time: 1 s",
                   "Deformation scale", "Auto Scale", " Max", " Min",
                   structural_header(2)):
        assert needed in text, needed
    # The screening render's words never appear on a solved picture.
    assert "ESTIMATE" not in text
    assert "NOT AN ANALYSIS" not in text
    assert screening_render.SCREENING_WATERMARK not in text


def test_total_deformation_says_what_it_is(solved, tmp_path) -> None:
    text = _svg(render_result(solved, "total_deformation", tmp_path / "d.svg"))

    for needed in ("A: Static Structural", "Total Deformation", "Type: Total Deformation",
                   "Unit: mm", "Deformation scale", "Max", "Min", structural_header(2)):
        assert needed in text, needed
    assert "ESTIMATE" not in text
    assert "NOT AN ANALYSIS" not in text
    assert "singular" not in text


def test_png_is_written_at_the_requested_path(solved, tmp_path) -> None:
    out = render_result(solved, "equivalent_stress", tmp_path / "deep" / "stress.png")

    assert out == tmp_path / "deep" / "stress.png"
    assert out.read_bytes()[:8] == PNG_MAGIC


def test_a_flagged_max_is_called_singular_and_the_value_away_from_it_is_given(solved, tmp_path) -> None:
    text = _svg(render_result(solved, "equivalent_stress", tmp_path / "s.svg"))

    assert "Max (singular)" in text
    assert "Max is at or next to a singularity" in text
    assert "rises with mesh refinement" in text
    assert "edge of A Fixed Support (face 1)" in text
    away = solved.max_stress_away.value
    assert f"away from it (beyond {solved.exclusion_radius_mm:g} mm): {away:.4g}" in text


def test_an_unflagged_max_carries_no_singularity_sentence(beam, tmp_path) -> None:
    clean = _solution(beam, _case(beam), singular=False)
    text = _svg(render_result(clean, "equivalent_stress", tmp_path / "s.svg"))

    assert "singular" not in text
    assert "Max" in text


def test_the_legend_runs_from_the_field_min_to_its_max(solved, tmp_path) -> None:
    text = _svg(render_result(solved, "total_deformation", tmp_path / "d.svg"))

    top = float(solved.total_deformation_mm.max())
    assert f"{top:.5g} Max" in text
    assert "0 Min" in text


def test_scale_options_are_named_in_the_title_block(solved, tmp_path) -> None:
    true_scale = _svg(render_result(solved, "total_deformation", tmp_path / "t.svg", scale="true"))
    assert "Deformation scale: 1 (True Scale)" in true_scale
    user = _svg(render_result(solved, "total_deformation", tmp_path / "u.svg", scale=3.5))
    assert "Deformation scale: 3.5 (User Defined)" in user
    auto = _svg(render_result(solved, "total_deformation", tmp_path / "a.svg"))
    assert f"Deformation scale: {solved.deformation_scale_auto:.3g} (Auto Scale)" in auto


@pytest.mark.parametrize("scale", ["big", -1.0, float("nan"), True])
def test_a_bad_scale_is_refused(solved, tmp_path, scale) -> None:
    with pytest.raises(StructuralRenderError, match="scale"):
        render_result(solved, "total_deformation", tmp_path / "x.png", scale=scale)


def test_an_unknown_quantity_is_refused_with_the_choices(solved, tmp_path) -> None:
    with pytest.raises(StructuralRenderError, match="equivalent_stress, total_deformation"):
        render_result(solved, "von_mises", tmp_path / "x.png")


def test_a_non_finite_field_is_refused_not_drawn(solved, tmp_path) -> None:
    bad = solved.von_mises_mpa.copy()
    bad[3] = np.nan
    broken = _Solution(**{**solved.__dict__, "von_mises_mpa": bad})
    with pytest.raises(StructuralRenderError, match="not finite"):
        render_result(broken, "equivalent_stress", tmp_path / "x.png")


def test_a_uniform_field_still_draws(beam, tmp_path) -> None:
    flat = _solution(beam, _case(beam))
    flat = _Solution(**{**flat.__dict__, "von_mises_mpa": np.full(len(beam.nodes), 7.0)})
    out = render_result(flat, "equivalent_stress", tmp_path / "f.png")
    assert out.read_bytes()[:8] == PNG_MAGIC


def test_a_first_order_mesh_is_labelled_as_four_node(tmp_path) -> None:
    mesh = _mesh(_box_step(tmp_path), size=8.0, order=1)
    text = _svg(render_result(_solution(mesh, _case(mesh)), "equivalent_stress", tmp_path / "p1.svg"))
    assert structural_header(1) in text
    assert "4-node linear tetrahedra" in text


def test_the_screening_palette_is_still_not_a_rainbow() -> None:
    """The solved renderer brought a rainbow into the project; it must stay here."""
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    assert screening_render.FIELD_COLORMAP not in {
        "jet", "rainbow", "turbo", "gist_rainbow", "hsv", "nipy_spectral"}
    screening = {to_hex(c) for c in colormaps[screening_render.FIELD_COLORMAP](np.linspace(0, 1, 9))}
    assert not screening & set(ANSYS_LEGEND_COLORS)
    assert len(set(ANSYS_LEGEND_COLORS)) == 9
    assert ANSYS_LEGEND_COLORS[0] == "#0000ff" and ANSYS_LEGEND_COLORS[-1] == "#ff0000"


def test_structural_header_names_the_element() -> None:
    assert structural_header(2) == "SOLVED — linear elastic, 10-node tetrahedra"
    # The 4-node stamp carries its caveat, so a screenshot cannot lose it.
    assert structural_header(1) == (
        "SOLVED — linear elastic, 4-node tetrahedra (stiff in bending: comparison only)"
    )
    with pytest.raises(StructuralRenderError):
        structural_header(3)


# --------------------------------------------------------------------------
# Setup and face pictures
# --------------------------------------------------------------------------


def test_the_setup_picture_letters_every_item_and_shows_no_result(beam, tmp_path) -> None:
    case = _case(beam, pressure=True, gravity=True)
    text = _svg(render_setup(beam, case, tmp_path / "setup.svg"))

    assert "Fixed Support" in text
    assert "Force: 500 N along −Z" in text
    assert "Pressure: 1.5 MPa" in text
    assert "Standard Earth Gravity: 9.8066 m/s²" in text
    for letter in "ABCD":
        assert f"<!-- {letter} -->" in text or f"<!-- {letter} (behind) -->" in text
    assert "inputs, not results" in text
    assert "SOLVED" not in text
    assert "ESTIMATE" not in text


def test_the_setup_picture_refuses_a_face_the_mesh_lacks(beam, tmp_path) -> None:
    case = StaticStructuralCase(supports=(FixedSupport((99,)),),
                                loads=(ForceLoad((1,), (0, 0, -1)),),
                                material=material_from_key("steel"))
    with pytest.raises(StructuralRenderError, match="99"):
        render_setup(beam, case, tmp_path / "x.png")


def test_the_face_picture_numbers_every_face_once(beam, tmp_path) -> None:
    text = _svg(render_face_ids(beam, tmp_path / "faces.svg"))

    for face in beam.faces:
        assert text.count(f"<!-- {face.tag} -->") == 1, face.tag
    assert f"{len(beam.faces)} faces · numbers match the face table" in text
    # A box is fully visible from two opposite iso views.
    assert "not visible" not in text


def test_compose_board_lays_pictures_out_and_refuses_what_it_cannot_read(solved, beam, tmp_path) -> None:
    a = render_result(solved, "equivalent_stress", tmp_path / "a.png", dpi=60)
    b = render_result(solved, "total_deformation", tmp_path / "b.png", dpi=60)
    c = render_face_ids(beam, tmp_path / "c.png", dpi=60)
    board = compose_board([a, b, c], tmp_path / "board.png", title="beam — Static Structural", dpi=60)
    assert board.read_bytes()[:8] == PNG_MAGIC

    with pytest.raises(StructuralRenderError, match="at least one"):
        compose_board([], tmp_path / "x.png")
    with pytest.raises(StructuralRenderError, match="missing"):
        compose_board([tmp_path / "nope.png"], tmp_path / "x.png")
    svg = render_result(solved, "total_deformation", tmp_path / "d.svg")
    with pytest.raises(StructuralRenderError, match="PNG"):
        compose_board([svg], tmp_path / "x.png")


# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------


def test_choose_view_keeps_the_default_without_a_reason_to_move(beam) -> None:
    assert choose_view(beam) == DEFAULT_VIEW
    # Tip face (+X) and top are visible from the default view already.
    assert choose_view(beam, _case(beam)) == DEFAULT_VIEW


def test_choose_view_turns_to_show_a_load_the_default_hides(beam) -> None:
    held = _face_with_normal(beam, 0, +1.0)
    hidden = _face_with_normal(beam, 1, +1.0)    # +Y face: behind, from (24, -58)
    case = StaticStructuralCase(supports=(FixedSupport((held,)),),
                                loads=(ForceLoad((hidden,), (0.0, -100.0, 0.0)),),
                                material=material_from_key("steel"))
    eye = _basis(*DEFAULT_VIEW)[2]
    assert eye @ np.array([0.0, 1.0, 0.0]) < 0
    view = choose_view(beam, case)
    assert view != DEFAULT_VIEW
    assert _basis(*view)[2] @ np.array([0.0, 1.0, 0.0]) > 0
    assert view[0] > 0, "prefers looking from above"


# --------------------------------------------------------------------------
# Geometry underneath
# --------------------------------------------------------------------------


def test_isobands_partition_a_triangle_exactly() -> None:
    """Field s = x on a right triangle: each band's area is known in closed form."""
    P = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    tris = np.array([[0, 1, 2]])
    s = P[:, 0].copy()
    bounds = np.linspace(0.0, 3.0, 4)
    polys, bands, parents = _isobands(P, tris, s, bounds)

    def area(poly: np.ndarray) -> float:
        x, y = poly[:, 0], poly[:, 1]
        return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    by_band = {k: sum(area(p) for p, b in zip(polys, bands) if b == k) for k in range(3)}
    # Area of the triangle where x in [a, b] is (3-a)^2/2 - (3-b)^2/2.
    for k in range(3):
        a, b = bounds[k], bounds[k + 1]
        assert by_band[k] == pytest.approx(((3 - a) ** 2 - (3 - b) ** 2) / 2, rel=1e-12)
    for poly, band in zip(polys, bands):
        assert poly[:, 0].min() >= bounds[band] - 1e-12
        assert poly[:, 0].max() <= bounds[band + 1] + 1e-12
    assert set(parents.tolist()) == {0}


def test_isobands_leave_a_one_band_triangle_whole() -> None:
    P = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    polys, bands, _ = _isobands(P, np.array([[0, 1, 2]]), np.array([0.12, 0.15, 0.2]),
                                np.linspace(0.0, 1.0, 10))
    assert len(polys) == 1 and bands.tolist() == [1]
    assert np.allclose(polys[0][:3], P)


def test_tri6_splits_into_four_triangles_with_the_parent_winding(beam) -> None:
    tris, parent = _surface(beam)
    assert len(tris) == 4 * len(beam.boundary)
    n_sub = np.cross(beam.nodes[tris[:, 1]] - beam.nodes[tris[:, 0]],
                     beam.nodes[tris[:, 2]] - beam.nodes[tris[:, 0]])
    b = beam.boundary
    n_parent = np.cross(beam.nodes[b[:, 1]] - beam.nodes[b[:, 0]],
                        beam.nodes[b[:, 2]] - beam.nodes[b[:, 0]])
    assert np.all(np.einsum("ij,ij->i", n_sub, n_parent[parent]) > 0)


def test_the_depth_map_hides_what_is_behind_and_shows_what_is_in_front() -> None:
    # Two squares facing the viewer (+z), one 5 mm behind the other but offset.
    V = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0],
                  [5, 5, -5], [15, 5, -5], [15, 15, -5], [5, 15, -5]], dtype=float)
    tris = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    depth = _DepthMap(V, tris, resolution=200, count=True)
    behind = np.array([[7.0, 7.0, -5.0]])       # back square, under the front one
    exposed = np.array([[13.0, 13.0, -5.0]])    # back square, nothing in front
    front = np.array([[2.0, 2.0, 0.0]])
    tol = 2 * depth.pix
    assert not depth.visible(behind, tol)[0]
    assert depth.visible(exposed, tol)[0]
    assert depth.visible(front, tol)[0]
    # Visible area: the front square fully, the back one minus the overlap.
    area = depth.tri_pixels * depth.pix ** 2
    assert area[:2].sum() == pytest.approx(100.0, rel=0.03)
    assert area[2:].sum() == pytest.approx(75.0, rel=0.03)


def test_fit_keeps_a_long_part_inside_its_box() -> None:
    """The clipping bug of the old renderer: long parts lost their ends."""
    pts = np.array([[-400.0, -3.0], [400.0, 3.0], [0.0, 10.0]])
    frame = _fit(pts, (100.0, 50.0, 900.0, 650.0))
    px = frame.px(pts)
    assert px[:, 0].min() >= 100 and px[:, 0].max() <= 900
    assert px[:, 1].min() >= 50 and px[:, 1].max() <= 650
    assert math.isclose(frame.k * 800.0, 800.0 * (1 - 2 * 0.03), rel_tol=1e-9)


# --------------------------------------------------------------------------
# From a real solve, when the solver is present
# --------------------------------------------------------------------------


def test_renders_from_a_real_solve(tmp_path) -> None:
    structural = pytest.importorskip("fea_surrogate.structural")
    structural_mesh = pytest.importorskip("fea_surrogate.structural_mesh")
    pytest.importorskip("scipy")
    from fea_surrogate.structural_export import write_vtu

    step = _box_step(tmp_path)
    mesh = structural_mesh.mesh_for_analysis(step, order=2, refinement="coarse")
    case = _case(mesh)
    solution = structural.solve_static(mesh, case)
    for quantity in ("equivalent_stress", "total_deformation"):
        out = render_result(solution, quantity, tmp_path / f"{quantity}.svg")
        text = _svg(out)
        assert structural_header(2) in text and "Max" in text and "Min" in text
        assert "ESTIMATE" not in text
    assert render_setup(mesh, case, tmp_path / "setup.png").read_bytes()[:8] == PNG_MAGIC
    assert render_face_ids(mesh, tmp_path / "faces.png").read_bytes()[:8] == PNG_MAGIC
    assert write_vtu(solution, tmp_path / "r.vtu").stat().st_size > 0
