"""The deployed path must select the faces the training tags select.

This is the test whose absence let a broken deployed path ship. Measured across
600 fresh parts, screening through geometric node selectors gave 0.415 coverage
against a calibrated 0.90, with spring and flange at 0.00 -- not because the
model was wrong, but because it was being asked a different question. A spring
was held at 20 nodes on its helical surface where training held the whole end
coil; a flange was held across 26.5% of its nodes where training held only the
faces whose centroid lies inside the plate.

Nothing compared the two selections, so nothing noticed.
"""

from __future__ import annotations

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh", reason="meshing requires the inference extra")
pytest.importorskip("scipy", reason="the families require the inference extra")
pytest.importorskip("torch", reason="mesh_step imports torch_geometric")

from fea_surrogate.cad_intake import inspect  # noqa: E402
from fea_surrogate.face_selectors import (  # noqa: E402
    FAMILY_FACE_PRESETS,
    FaceSelectionError,
    face_mask,
    select_faces,
)
from fea_surrogate.features import mesh_step  # noqa: E402
from fea_surrogate.generate.families import ALL_FAMILIES  # noqa: E402
from fea_surrogate.generate.harness import _gmsh_model, draw  # noqa: E402


FAMILIES = {family.name: family for family in ALL_FAMILIES}


def _described(tags, centroids, areas) -> set:
    """A face by geometry, not by tag.

    Face *numbers* are not comparable across the two paths: one builds the solid
    from parameters, the other imports a STEP, and the kernel renumbers. Centroid
    and area are properties of the shape and survive the round trip.
    """
    return {(tuple(np.round(centroids[t], 3)), round(areas[t], 3)) for t in tags}


def _training_and_deployed(name: str, tmp_path, seed: int = 11):
    """What `family.tag()` picks, and what the deployed preset picks."""
    family = FAMILIES[name]()
    params = dict(draw(family, np.random.default_rng(seed)).sample.params)

    step = tmp_path / f"{name}.step"
    with _gmsh_model(name):
        family.build(params)
        gmsh.model.occ.synchronize()
        tagging = family.tag(params)
        faces = [int(t) for _, t in gmsh.model.getEntities(2)]
        centroids = {
            t: np.array(gmsh.model.occ.getCenterOfMass(2, t)) for t in faces
        }
        areas = {t: float(gmsh.model.occ.getMass(2, t)) for t in faces}
        training = {
            "fixed": _described(tagging.fixed_faces, centroids, areas),
            "loaded": _described(tagging.loaded_faces, centroids, areas),
        }
        # Counts as well as identities. Comparing sets alone collapses two
        # coincident faces into one entry, which is exactly how a duplicated
        # spring end face survived this check while doubling the held nodes.
        counts = {
            "fixed": len(tagging.fixed_faces),
            "loaded": len(tagging.loaded_faces),
        }
        gmsh.write(str(step))

    solid = inspect(step)
    imported_centroids = {f.tag: np.array(f.centroid) for f in solid.faces}
    imported_areas = {f.tag: f.area for f in solid.faces}

    mesh = mesh_step(step)
    deployed, deployed_counts = {}, {}
    for role in ("fixed", "loaded"):
        tags = select_faces(FAMILY_FACE_PRESETS[name][role], mesh, role)
        deployed[role] = _described(tags, imported_centroids, imported_areas)
        deployed_counts[role] = len(tags)
    return training, deployed, mesh, counts, deployed_counts


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_the_deployed_preset_selects_the_faces_training_tagged(name, tmp_path) -> None:
    training, deployed, _, counts, deployed_counts = _training_and_deployed(
        name, tmp_path
    )

    assert deployed["fixed"] == training["fixed"], (
        f"{name}: held faces differ from what the sweep trained on"
    )
    assert deployed["loaded"] == training["loaded"], (
        f"{name}: loaded faces differ from what the sweep trained on"
    )
    assert deployed_counts == counts, (
        f"{name}: the two paths agree on which faces but not how many, which "
        "means the built solid carries a face the exported one does not"
    )


def test_the_swept_solid_has_no_duplicated_faces(tmp_path) -> None:
    """`addPipe` does not consume its profile, and gmsh meshes what is left.

    One sample carried 24 nodes on the swept spring's bottom cap and 25 more on
    the orphaned profile 0.044 mm away, sharing none -- so the bottom end had
    twice the nodes of the top, and a STEP export of the same part had neither.
    """
    for name in ("spring", "pulley", "bracket"):
        _, _, mesh, counts, deployed_counts = _training_and_deployed(name, tmp_path)
        assert deployed_counts == counts, f"{name} has an orphaned face"


def test_a_spring_is_held_at_its_end_face_not_a_sliver_of_the_helix(tmp_path) -> None:
    """The failure that cost the most, named directly.

    The end-coil cross section is a small fraction of a spring's nodes -- that is
    correct and expected. What matters is that the held nodes are a *face's*
    nodes, so they lie in a plane and span it, rather than the handful of nodes
    that happen to sit at the lowest z on a curved surface.
    """
    *_, mesh, _, _ = _training_and_deployed("spring", tmp_path)
    tags = select_faces(FAMILY_FACE_PRESETS["spring"]["fixed"], mesh, "fixed")
    vertices = np.asarray(mesh.vertices, dtype=float)
    held = vertices[face_mask(tags, mesh)]

    assert len(tags) == 1
    # Thin in z relative to the whole spring: a cross section, not a slab.
    assert float(np.ptp(held[:, 2])) < 0.25 * float(np.ptp(vertices[:, 2]))
    # And a real face rather than a tangent line: it has extent in both of the
    # other axes. The old node selector failed exactly here.
    assert float(np.ptp(held[:, 0])) > 0.0
    assert float(np.ptp(held[:, 1])) > 0.0


def test_an_unmatched_selector_is_an_error_not_an_empty_mask(tmp_path) -> None:
    """An empty selection would screen an unconstrained body without saying so."""
    *_, mesh, _, _ = _training_and_deployed("bracket", tmp_path)

    with pytest.raises(FaceSelectionError, match="no face"):
        select_faces({"faces": [9999]}, mesh, "fixed")


def test_a_node_coordinate_selector_is_refused_by_the_face_selector(tmp_path) -> None:
    """`{"axis": "x", "at": "min"}` is the shape of the selector that caused this.

    It is still valid as a node selector, but it must not be silently accepted
    where a face selector is expected.
    """
    *_, mesh, _, _ = _training_and_deployed("bracket", tmp_path)

    with pytest.raises(FaceSelectionError, match="faces', 'centroid' or 'radial'"):
        select_faces({"axis": "x", "at": "min"}, mesh, "fixed")


def test_the_mesh_carries_what_face_selection_needs(tmp_path) -> None:
    *_, mesh, _, _ = _training_and_deployed("bracket", tmp_path)

    assert mesh.face_nodes
    assert mesh.face_centroids
    assert set(mesh.face_nodes) == set(mesh.face_centroids)
    assert mesh.thinnest and mesh.thinnest > 0.0


def test_the_characteristic_length_comes_from_the_mesh_not_the_cad_box(
    tmp_path,
) -> None:
    """The sweep nondimensionalised by the mesh bounding box.

    `MeshedSample.characteristic_length_mm` is the norm of the mesh vertex
    extent, and the model is fitted on tau = sigma * L^2 / F with that L. If
    screening derives L from the CAD box instead, the prediction is wrong by
    (L_mesh / L_cad)^2.

    OpenCASCADE's box for a curved face bounds its control polygon and
    overshoots the surface. Every family here is planar enough that the two
    agree to seven figures -- measured on a bored hex prism they differ by
    10.2%, which is why the sweep could never have shown this and why the
    deployed-coverage gate cannot either.
    """
    _, _, mesh, _, _ = _training_and_deployed("bracket", tmp_path)
    vertices = np.asarray(mesh.vertices, dtype=float)
    from_mesh = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))

    assert mesh.original_diagonal == pytest.approx(from_mesh, rel=1e-5)


def test_a_curved_part_is_where_the_two_boxes_disagree(tmp_path) -> None:
    """Guard the guard: on a planar part the check above passes either way."""
    import gmsh as _gmsh

    path = tmp_path / "ball.step"
    _gmsh.initialize()
    try:
        _gmsh.option.setNumber("General.Terminal", 0)
        _gmsh.model.add("ball")
        _gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 20.0)
        _gmsh.model.occ.synchronize()
        cad = _gmsh.model.getBoundingBox(3, 1)
        cad_diagonal = float(
            np.linalg.norm(np.array(cad[3:]) - np.array(cad[:3]))
        )
        _gmsh.write(str(path))
    finally:
        _gmsh.finalize()

    mesh = mesh_step(path)

    # A tessellated sphere sits inside its analytic box, so the mesh extent is
    # the smaller of the two -- and it is the one the model was fitted on.
    assert mesh.original_diagonal < cad_diagonal
