"""Reading a CAD file, and refusing the ones that cannot be read.

The interview has to be able to name faces before it can ask which of them is
held. Everything here is about producing that vocabulary, and about failing
early and legibly when the file is one OpenCASCADE cannot open at all.
"""

from __future__ import annotations

import pytest

gmsh = pytest.importorskip("gmsh", reason="meshing requires the inference extra")

from fea_surrogate.cad_intake import (  # noqa: E402
    CadFormatError,
    CadIntakeError,
    check_readable,
    inspect,
    summarise,
)


def _box(tmp_path, dx=40.0, dy=20.0, dz=10.0):
    path = tmp_path / "box.step"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("box")
        gmsh.model.occ.addBox(0.0, 0.0, 0.0, dx, dy, dz)
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


@pytest.mark.parametrize(
    "suffix,name",
    [(".x_t", "Parasolid"), (".x_b", "Parasolid"), (".sldprt", "SolidWorks"),
     (".catpart", "CATIA"), (".sat", "ACIS")],
)
def test_a_format_needing_a_licensed_kernel_is_refused_by_name(suffix, name) -> None:
    """Failing three calls deep inside the mesher helps nobody."""
    with pytest.raises(CadFormatError, match=name):
        check_readable("part" + suffix)


def test_the_refusal_says_how_to_get_unstuck() -> None:
    with pytest.raises(CadFormatError, match=r"STEP"):
        check_readable("part.x_t")


def test_an_unknown_extension_lists_what_can_be_read() -> None:
    with pytest.raises(CadFormatError, match=r"\.step"):
        check_readable("drawing.dwg")


def test_a_readable_extension_that_does_not_exist_is_refused(tmp_path) -> None:
    with pytest.raises(CadFormatError, match="no such file"):
        check_readable(tmp_path / "absent.step")


def test_a_box_enumerates_six_planar_faces_with_the_right_areas(tmp_path) -> None:
    solid = inspect(_box(tmp_path, 40.0, 20.0, 10.0))

    assert len(solid.faces) == 6
    assert all(face.planar for face in solid.faces)
    assert solid.volume == pytest.approx(8000.0)
    # 2*(40*20 + 40*10 + 20*10) = 2800
    assert solid.area == pytest.approx(2800.0)
    assert sorted(round(f.area) for f in solid.faces) == [200, 200, 400, 400, 800, 800]


def test_faces_are_listed_largest_first(tmp_path) -> None:
    """A listing that opens with slivers is unreadable."""
    areas = [face.area for face in inspect(_box(tmp_path)).faces]

    assert areas == sorted(areas, reverse=True)


def test_every_face_can_be_found_by_its_tag(tmp_path) -> None:
    solid = inspect(_box(tmp_path))

    for face in solid.faces:
        assert solid.face(face.tag) is face


def test_an_unknown_tag_says_which_tags_exist(tmp_path) -> None:
    solid = inspect(_box(tmp_path))

    with pytest.raises(CadIntakeError, match="this solid has"):
        solid.face(9999)


def test_normals_point_along_the_axes_of_a_box(tmp_path) -> None:
    solid = inspect(_box(tmp_path))
    normals = {tuple(round(c) for c in face.normal) for face in solid.faces}

    assert normals == {
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)
    }


def test_the_summary_names_every_face_and_the_bounding_box(tmp_path) -> None:
    lines = list(summarise(inspect(_box(tmp_path, 40.0, 20.0, 10.0))))

    assert "6 faces" in lines[0]
    assert len(lines) == 7
    assert all("face" in line for line in lines[1:])


def test_a_surface_model_with_no_solid_is_refused(tmp_path) -> None:
    """A surface has no volume to carry stress."""
    path = tmp_path / "sheet.step"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("sheet")
        gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, 10.0, 10.0)
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
    finally:
        gmsh.finalize()

    with pytest.raises(CadIntakeError, match="no solid"):
        inspect(path)


def _write(tmp_path, suffix, dx=200.0, dy=40.0, dz=10.0):
    path = tmp_path / f"bar.{suffix}"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("bar")
        gmsh.model.occ.addBox(0.0, 0.0, 0.0, dx, dy, dz)
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


@pytest.mark.parametrize("suffix", ["step", "brep", "iges"])
def test_every_format_we_claim_to_read_yields_the_same_solid(suffix, tmp_path) -> None:
    """Declaring a format readable is a promise that it produces a screenable part.

    IGES is why this is parametrised. It is a surface format in practice: a box
    written to IGES and read back is six trimmed faces and no solid at all, so
    it imported cleanly and could never be screened -- readable and useless.
    """
    solid = inspect(_write(tmp_path, suffix))

    assert solid.solid_count == 1
    assert len(solid.faces) == 6
    assert solid.volume == pytest.approx(80_000.0, rel=0.01)


def test_iges_is_healed_and_says_so(tmp_path) -> None:
    """Sewing closes gaps within a tolerance. That is a repair, not a read, and
    a repaired part must not pass for an exact one."""
    iges = inspect(_write(tmp_path, "iges"))
    step = inspect(_write(tmp_path, "step"))

    assert iges.healed is True
    assert step.healed is False
    assert any("sewn into one" in line for line in summarise(iges))
    assert not any("sewn into one" in line for line in summarise(step))


def test_a_file_that_already_has_a_solid_is_not_rebuilt(tmp_path) -> None:
    """Sewing a part that does not need it would quietly alter geometry."""
    from fea_surrogate.cad_intake import import_shapes

    # Written before the session opens: `_write` finalizes gmsh when it is done,
    # which would tear down the session this test is about to use.
    step = _write(tmp_path, "step")

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("m")
        healed = import_shapes(step)
        assert healed is False
        assert len(gmsh.model.getEntities(3)) == 1
    finally:
        gmsh.finalize()


def test_a_surface_that_cannot_be_closed_is_still_refused(tmp_path) -> None:
    """Healing must not turn a refusal into a wrong answer."""
    path = tmp_path / "sheet.iges"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("sheet")
        gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, 10.0, 10.0)
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
    finally:
        gmsh.finalize()

    with pytest.raises(CadIntakeError, match="did not close them"):
        inspect(path)
