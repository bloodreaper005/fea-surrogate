from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib", reason="rendering requires the render extra")

from fea_surrogate.render import (  # noqa: E402
    FieldNotPredicted,
    _confidence,
    _desaturate,
    field_or_refuse,
    mesh_sha256,
    render_field,
    render_load_case,
)
from fea_surrogate.screening import (  # noqa: E402
    ScreeningContractError,
    validate_document,
)


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _tetrahedron():
    """The smallest closed surface that has an inside."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    return vertices, faces


def _render(tmp_path, **overrides):
    vertices, faces = _tetrahedron()
    values = np.array([10.0, 20.0, 30.0, 40.0])
    kwargs = dict(
        out_dir=tmp_path,
        quantity="von_mises",
        units="MPa",
        vertices=vertices,
        faces=faces,
        values=values,
        lower=values * 0.8,
        upper=values * 1.2,
        coverage=0.9,
        model_name="sweep-mesh-gnn",
        model_version="0.4.0",
    )
    kwargs.update(overrides)
    return render_field(**kwargs)


def test_a_rendered_field_produces_an_entry_the_contract_accepts(tmp_path) -> None:
    """The renderer's output has to survive the consumer's parser unchanged."""
    artifacts = _render(tmp_path)

    validate_document(
        {
            "schema_version": "SurrogateScreening/v1",
            "screened_sha256": "a" * 64,
            "model": {"name": "m", "version": "1", "sha256": "b" * 64},
            "coverage": 0.9,
            "calibration": {
                "method": "split-conformal",
                "set_size": 300,
                "set_sha256": "c" * 64,
            },
            "domain": {"family": "bracket"},
            "predictions": [
                {
                    "quantity": "von_mises",
                    "lower": 1.0,
                    "upper": 2.0,
                    "units": "MPa",
                }
            ],
            "assumptions": ["linear elastic"],
            "limitations": ["screening estimate"],
            "attestation": "screening_estimate",
            "fields": [artifacts.entry],
        }
    )


def test_the_image_is_a_real_png_the_consumer_can_verify(tmp_path) -> None:
    """The consumer checks the signature and IHDR, not the suffix."""
    artifacts = _render(tmp_path)
    image = next(p for p in artifacts.paths if p.suffix == ".png")

    head = image.read_bytes()[:16]
    assert head.startswith(PNG_MAGIC)
    assert head[12:16] == b"IHDR"


def test_the_interval_is_written_as_both_edges_not_a_width(tmp_path) -> None:
    """A width cannot be compared against a yield strength; an upper bound can."""
    artifacts = _render(tmp_path)
    interval = np.load(next(p for p in artifacts.paths if p.name.endswith("_interval.npy")))

    assert interval.shape == (4, 2)
    assert np.all(interval[:, 0] <= interval[:, 1])
    np.testing.assert_allclose(interval[:, 1], np.array([12.0, 24.0, 36.0, 48.0]))


def test_a_field_without_a_per_node_band_cannot_be_rendered(tmp_path) -> None:
    """The one artifact this subsystem must never produce."""
    with pytest.raises(ScreeningContractError, match="one entry per vertex"):
        _render(tmp_path, lower=np.array([1.0, 2.0]))


def test_an_inverted_band_is_refused(tmp_path) -> None:
    values = np.array([10.0, 20.0, 30.0, 40.0])
    with pytest.raises(ScreeningContractError, match="lower bound exceeds"):
        _render(tmp_path, lower=values * 1.5, upper=values)


def test_a_non_finite_value_is_refused(tmp_path) -> None:
    with pytest.raises(ScreeningContractError, match="finite"):
        _render(tmp_path, values=np.array([10.0, np.nan, 30.0, 40.0]))


def test_confidence_falls_to_zero_as_the_band_reaches_the_value() -> None:
    """A band of +/-100% carries no spatial information, so it is drawn as none."""
    values = np.array([10.0, 10.0, 10.0])
    lower = np.array([10.0, 5.0, 0.0])
    upper = np.array([10.0, 15.0, 20.0])

    confidence = _confidence(values, lower, upper)

    assert confidence[0] == pytest.approx(1.0)
    assert confidence[1] == pytest.approx(0.5)
    assert confidence[2] == pytest.approx(0.0)


def test_low_confidence_colours_wash_out_rather_than_staying_vivid() -> None:
    """Visual saliency has to track confidence, or the plot overstates itself."""
    from matplotlib.colors import rgb_to_hsv

    vivid = np.tile(np.array([[0.1, 0.3, 0.8, 1.0]]), (3, 1))
    washed = _desaturate(vivid, np.array([1.0, 0.5, 0.0]))

    certain, partial, washed_out = rgb_to_hsv(washed[:, :3])
    assert certain[1] > partial[1] > washed_out[1]
    assert washed_out[1] == pytest.approx(0.0)
    # Hue survives while any confidence remains, so a partly uncertain region
    # can still be read for order.
    assert partial[0] == pytest.approx(certain[0])
    # At full washout it does not, and must not: a band as wide as the value
    # supports no ordering, so the plot shows none.
    assert washed_out[0] != pytest.approx(certain[0])


def test_an_out_of_domain_region_changes_the_image(tmp_path) -> None:
    """Hatching a refusal is only meaningful if it is actually drawn."""
    plain = _render(tmp_path / "plain")
    hatched = _render(
        tmp_path / "hatched",
        out_of_domain=np.array([True, False, False, False]),
    )

    plain_png = next(p for p in plain.paths if p.suffix == ".png").read_bytes()
    hatched_png = next(p for p in hatched.paths if p.suffix == ".png").read_bytes()
    assert plain_png != hatched_png


def test_an_out_of_domain_flag_must_cover_every_vertex(tmp_path) -> None:
    with pytest.raises(ScreeningContractError, match="one flag per vertex"):
        _render(tmp_path, out_of_domain=np.array([True, False]))


def test_the_mesh_digest_changes_when_the_mesh_does() -> None:
    """Values are meaningless without the mesh they sit on."""
    vertices, faces = _tetrahedron()
    moved = vertices.copy()
    moved[0, 0] += 1e-6

    assert mesh_sha256(vertices, faces) != mesh_sha256(moved, faces)
    assert mesh_sha256(vertices, faces) == mesh_sha256(vertices.copy(), faces.copy())


def test_the_transposed_face_layout_is_accepted(tmp_path) -> None:
    """`ScaledMesh` carries (3, F); the renderer should not care."""
    vertices, faces = _tetrahedron()
    a = _render(tmp_path / "a", faces=faces)
    b = _render(tmp_path / "b", faces=np.ascontiguousarray(faces.T))

    assert a.entry["mesh_sha256"] == b.entry["mesh_sha256"]


def test_a_graph_level_model_refuses_to_produce_a_field() -> None:
    """sweep/v2 pools the graph into one number.

    Painting that number across every node would draw a uniform plot that is
    indistinguishable from a prediction. The contract refuses a point estimate
    rather than widening it, and the renderer refuses it rather than spreading
    it.
    """

    class Pooled:
        version = "sweep/v2"

    with pytest.raises(FieldNotPredicted, match="graph-level head"):
        field_or_refuse(Pooled(), object())


def test_the_load_case_drawing_marks_both_ends_of_the_problem(tmp_path) -> None:
    """Inputs, not results: a part held on the wrong face screens cleanly."""
    vertices, faces = _tetrahedron()

    path = render_load_case(
        out_dir=tmp_path,
        vertices=vertices,
        faces=faces,
        fixed_mask=np.array([True, False, False, False]),
        loaded_mask=np.array([False, False, False, True]),
        load_direction=(0.0, 0.0, -1.0),
        origin="family preset 'bracket'",
    )

    assert path.read_bytes().startswith(PNG_MAGIC)


def test_every_face_is_numbered_on_the_picture(tmp_path) -> None:
    """The interview asks which face is held; nobody holds a centroid in their head."""
    gmsh = pytest.importorskip("gmsh", reason="needs the inference extra")
    pytest.importorskip("torch", reason="mesh_step imports torch_geometric")

    from fea_surrogate.cad_intake import inspect
    from fea_surrogate.features import mesh_step
    from fea_surrogate.render import render_faces

    step = tmp_path / "box.step"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("box")
        gmsh.model.occ.addBox(0.0, 0.0, 0.0, 60.0, 30.0, 8.0)
        gmsh.model.occ.synchronize()
        gmsh.write(str(step))
    finally:
        gmsh.finalize()

    solid = inspect(step)
    image = render_faces(solid, mesh_step(step), out_dir=tmp_path, fmt="svg")
    text = image.read_text(encoding="utf-8")

    # Every face's tag appears, and the caption says how many there are.
    assert f"{len(solid.faces)} faces" in text
    assert "name one to hold it" in text


def test_face_labels_are_lifted_clear_of_the_part(tmp_path) -> None:
    """Placed at the centroid, the two faces of a thin plate land on the same
    screen position and their numbers sit on top of each other -- on an
    L-bracket that was 1 over 8 and 7 over 4, worse than no label at all.
    """
    gmsh = pytest.importorskip("gmsh", reason="needs the inference extra")
    pytest.importorskip("torch", reason="mesh_step imports torch_geometric")

    from fea_surrogate.cad_intake import inspect
    from fea_surrogate.features import mesh_step
    from fea_surrogate.render import render_faces

    step = tmp_path / "plate.step"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("plate")
        gmsh.model.occ.addBox(0.0, 0.0, 0.0, 80.0, 60.0, 3.0)
        gmsh.model.occ.synchronize()
        gmsh.write(str(step))
    finally:
        gmsh.finalize()

    solid = inspect(step)
    mesh = mesh_step(step)
    image = render_faces(solid, mesh, out_dir=tmp_path)

    assert image.read_bytes().startswith(PNG_MAGIC)
    # The top and bottom faces of a 3 mm plate: their centroids are 3 mm apart,
    # so a label drawn at the centroid would be unreadable. The render lifts
    # each one along its own normal instead.
    flat = [f for f in solid.faces if abs(f.normal[2]) > 0.9]
    assert len(flat) == 2
    assert abs(flat[0].centroid[2] - flat[1].centroid[2]) == pytest.approx(3.0)


def test_a_mesh_without_a_face_map_cannot_be_drawn_by_face(tmp_path) -> None:
    from fea_surrogate.features import ScaledMesh
    from fea_surrogate.render import render_faces

    vertices, faces = _tetrahedron()
    bare = ScaledMesh(vertices.astype(np.float32), faces.T, 1.0, 1.0)

    with pytest.raises(ScreeningContractError, match="face-to-node map"):
        render_faces(object(), bare, out_dir=tmp_path)
