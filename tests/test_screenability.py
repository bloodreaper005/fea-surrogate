"""The gate that decides whether the surrogate may speak about a case at all.

The surrogate is blind to load direction and Poisson ratio and is not scale
invariant (measured: a bracket screened 165-313 MPa whether it was pushed along
-z, where the solve gave 208.6 MPa, or along +x, where it gave 3.9). So the gate
is strict on purpose, and every rule needs a test that shows it closing.

Light: the mesh here is a stand-in carrying exactly what the gate reads (face
summaries, centroids, node lists, size), so this runs without gmsh.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from fea_surrogate import screenability
from fea_surrogate.face_selectors import FAMILY_FACE_PRESETS
from fea_surrogate.load_case import parse as parse_load_case
from fea_surrogate.screenability import (
    FAMILY_LENGTH_RANGES_MM,
    RANGES_MEASURED_ON_SET,
    check,
    matching_families,
    mesh_volume_mm3,
)
from fea_surrogate.structural_case import (
    MATERIAL_PRESETS,
    FixedSupport,
    ForceLoad,
    Gravity,
    PressureLoad,
    StaticStructuralCase,
    custom_material,
)

STEEL = MATERIAL_PRESETS["structural_steel"]


def box_mesh(length=100.0, width=20.0, height=10.0, *, solid_count=1, curved=()):
    """A box as the gate sees an AnalysisMesh: faces 1 x-min, 2 x-max, 3 y-min,
    4 y-max, 5 z-min, 6 z-max."""
    L, W, H = length, width, height
    corners = np.array([[x, y, z] for x in (0, L) for y in (0, W) for z in (0, H)], float)

    def nodes(axis, value):
        return np.flatnonzero(np.isclose(corners[:, axis], value))

    spec = [
        (1, (0, W / 2, H / 2), (-1, 0, 0), W * H, 0, 0.0),
        (2, (L, W / 2, H / 2), (1, 0, 0), W * H, 0, L),
        (3, (L / 2, 0, H / 2), (0, -1, 0), L * H, 1, 0.0),
        (4, (L / 2, W, H / 2), (0, 1, 0), L * H, 1, W),
        (5, (L / 2, W / 2, 0), (0, 0, -1), L * W, 2, 0.0),
        (6, (L / 2, W / 2, H), (0, 0, 1), L * W, 2, H),
    ]
    faces, centroids, face_nodes = [], {}, {}
    for tag, centre, normal, area, axis, value in spec:
        idx = nodes(axis, value)
        pts = corners[idx]
        faces.append(SimpleNamespace(
            tag=tag, area=float(area), centroid=centre, normal=normal,
            planar=tag not in curved, kind="plane" if tag not in curved else "curved",
            bbox_min=tuple(pts.min(axis=0)), bbox_max=tuple(pts.max(axis=0)),
        ))
        centroids[tag] = centre
        face_nodes[tag] = idx
    faces.sort(key=lambda f: -f.area)
    return SimpleNamespace(
        faces=tuple(faces), face_centroids=centroids, face_nodes=face_nodes,
        vertices=corners, nodes=corners, thinnest=min(L, W, H),
        diagonal_mm=float(math.sqrt(L * L + W * W + H * H)), solid_count=solid_count,
        volume_mm3=L * W * H,
    )


def bracket_case(vector=(0.0, 0.0, -500.0), *, material=STEEL, gravity=None, extra_loads=()):
    return StaticStructuralCase(
        supports=(FixedSupport(faces=(1,)),),
        loads=(ForceLoad(faces=(2,), vector_n=vector), *extra_loads),
        material=material,
        gravity=gravity,
    )


def joined(verdict) -> str:
    return " | ".join(verdict.reasons)


# --------------------------------------------------------------------------


def test_the_bracket_training_case_is_screenable() -> None:
    mesh = box_mesh()
    verdict = check(bracket_case(), mesh, family="bracket")

    assert verdict.available, verdict.reasons
    assert verdict.reasons == ()
    assert verdict.force_n == pytest.approx(500.0)
    assert verdict.direction == pytest.approx((0.0, 0.0, -1.0))
    assert verdict.fixed_faces == (1,) and verdict.loaded_faces == (2,)
    assert verdict.matching_families == ("bracket",)
    # the legacy document is exactly what load_case.parse (and screening) reads
    legacy = parse_load_case(verdict.legacy_document)
    assert legacy.load.magnitude_n == pytest.approx(500.0)
    assert legacy.load.direction == pytest.approx((0.0, 0.0, -1.0))
    assert legacy.fixed == {"faces": [1]}
    assert legacy.poisson_ratio == pytest.approx(0.3)


def test_either_sign_of_the_trained_axis_passes_because_von_mises_ignores_sign() -> None:
    assert check(bracket_case((0, 0, 500.0)), box_mesh(), family="bracket").available


def test_a_slightly_skewed_axis_passes_and_a_real_change_of_direction_does_not() -> None:
    tilt = math.radians(8)
    skewed = (0.0, 500 * math.sin(tilt), -500 * math.cos(tilt))
    assert check(bracket_case(skewed), box_mesh(), family="bracket").available

    sideways = check(bracket_case((500.0, 0, 0)), box_mesh(), family="bracket")
    assert not sideways.available
    assert sideways.legacy_document is None
    assert "direction" in joined(sideways) and "+X" in joined(sideways)


def test_several_loads_are_not_screenable() -> None:
    case = bracket_case(extra_loads=(ForceLoad(faces=(6,), vector_n=(0, 0, -100.0)),))
    verdict = check(case, box_mesh(), family="bracket")
    assert not verdict.available
    assert "2 loads" in joined(verdict)
    assert matching_families(case, box_mesh()) == ()


def test_gravity_is_allowed_only_while_negligible() -> None:
    mesh = box_mesh()   # 20000 mm^3 of steel weighs about 1.54 N
    small = check(bracket_case(gravity=Gravity()), mesh, family="bracket")
    assert small.available, small.reasons
    assert small.legacy_document["gravity"] == {"enabled": True, "g": pytest.approx(9.80665)}

    heavy = check(bracket_case((0, 0, -10.0), gravity=Gravity()), mesh, family="bracket")
    assert not heavy.available
    assert "self-weight" in joined(heavy)


def test_gravity_alone_is_not_screenable() -> None:
    case = StaticStructuralCase(supports=(FixedSupport(faces=(1,)),), loads=(),
                                material=STEEL, gravity=Gravity())
    verdict = check(case, box_mesh(), family="bracket")
    assert not verdict.available and "only load" in joined(verdict)


def test_a_planar_pressure_becomes_one_force_along_the_inward_normal() -> None:
    # flange preset: held at faces whose centroid lies inside the thickness (the
    # four sides), loaded on top, pushed along -z: a pressure on the top face.
    case = StaticStructuralCase(
        supports=(FixedSupport(faces=(1, 2, 3, 4)),),
        loads=(PressureLoad(faces=(6,), pressure_mpa=1.5),),
        material=STEEL,
    )
    verdict = check(case, box_mesh(), family="flange")
    assert verdict.available, verdict.reasons
    assert verdict.force_n == pytest.approx(1.5 * 100 * 20)
    assert verdict.direction == pytest.approx((0.0, 0.0, -1.0))
    assert matching_families(case, box_mesh()) == ("flange",)


def test_suction_points_the_equivalent_force_outward() -> None:
    case = StaticStructuralCase(
        supports=(FixedSupport(faces=(1, 2, 3, 4)),),
        loads=(PressureLoad(faces=(6,), pressure_mpa=-0.5),),
        material=STEEL,
    )
    verdict = check(case, box_mesh(), family="flange")
    assert verdict.available
    assert verdict.direction == pytest.approx((0.0, 0.0, 1.0))
    assert verdict.legacy_document["load"]["magnitude_n"] == pytest.approx(1000.0)


def test_a_pressure_on_a_curved_face_is_not_one_traction() -> None:
    case = StaticStructuralCase(
        supports=(FixedSupport(faces=(1, 2, 3, 4)),),
        loads=(PressureLoad(faces=(6,), pressure_mpa=1.5),),
        material=STEEL,
    )
    verdict = check(case, box_mesh(curved=(6,)), family="flange")
    assert not verdict.available
    assert "curved" in joined(verdict)


def test_no_family_is_refused_and_the_matching_one_is_suggested() -> None:
    verdict = check(bracket_case(), box_mesh(), family=None)
    assert not verdict.available
    assert "no surrogate family" in joined(verdict)
    assert "--family bracket" in joined(verdict)


def test_spring_is_refused_and_agrees_with_predict() -> None:
    verdict = check(bracket_case(), box_mesh(), family="spring")
    assert not verdict.available
    assert "spring" in joined(verdict) and "refused" in joined(verdict)


def test_the_local_untransferable_list_matches_predict() -> None:
    pytest.importorskip("torch", reason="predict imports torch")
    pytest.importorskip("torch_geometric", reason="predict imports torch_geometric")
    from fea_surrogate.predict import UNTRANSFERABLE_FAMILIES

    assert set(UNTRANSFERABLE_FAMILIES) == set(screenability.UNTRANSFERABLE_FAMILIES)


def test_an_unknown_family_lists_the_known_ones() -> None:
    verdict = check(bracket_case(), box_mesh(), family="gearbox")
    assert not verdict.available
    assert "unknown family" in joined(verdict) and "bracket" in joined(verdict)


def test_faces_other_than_the_presets_are_refused() -> None:
    case = StaticStructuralCase(
        supports=(FixedSupport(faces=(5,)),),
        loads=(ForceLoad(faces=(2,), vector_n=(0, 0, -500.0)),),
        material=STEEL,
    )
    verdict = check(case, box_mesh(), family="bracket")
    assert not verdict.available
    assert "held faces [5] differ" in joined(verdict)
    assert "[1]" in joined(verdict)

    other_load = StaticStructuralCase(
        supports=(FixedSupport(faces=(1,)),),
        loads=(ForceLoad(faces=(6,), vector_n=(0, 0, -500.0)),),
        material=STEEL,
    )
    verdict = check(other_load, box_mesh(), family="bracket")
    assert "loaded faces [6] differ" in joined(verdict)


def test_poisson_outside_the_trained_range_is_refused() -> None:
    rubbery = custom_material(3000.0, 0.45)
    verdict = check(bracket_case(material=rubbery), box_mesh(), family="bracket")
    assert not verdict.available
    assert "Poisson" in joined(verdict) and "0.27" in joined(verdict)

    wide = {"poisson_ratio": {"min": 0.2, "max": 0.49}}
    assert check(bracket_case(material=rubbery), box_mesh(), family="bracket", domain=wide).available


def test_size_outside_the_family_training_range_is_refused() -> None:
    big = box_mesh(length=400.0)
    verdict = check(bracket_case(), big, family="bracket")
    assert not verdict.available
    assert "size" in joined(verdict) and "58-367" in joined(verdict)

    tiny = box_mesh(length=30.0, width=10.0, height=5.0)
    assert not check(bracket_case(), tiny, family="bracket").available


def test_ranges_measured_on_another_calibration_set_are_not_trusted() -> None:
    same = check(bracket_case(), box_mesh(), family="bracket",
                 calibration_set_sha256=RANGES_MEASURED_ON_SET)
    assert same.available

    other = check(bracket_case(), box_mesh(), family="bracket", calibration_set_sha256="f" * 64)
    assert not other.available
    assert "different calibration set" in joined(other)


def test_an_assembly_is_refused() -> None:
    verdict = check(bracket_case(), box_mesh(solid_count=2), family="bracket")
    assert not verdict.available and "2 solids" in joined(verdict)


def test_every_failing_rule_is_reported_not_just_the_first() -> None:
    case = StaticStructuralCase(
        supports=(FixedSupport(faces=(5,)),),
        loads=(ForceLoad(faces=(2,), vector_n=(500.0, 0, 0)),),
        material=custom_material(3000.0, 0.45),
    )
    verdict = check(case, box_mesh(length=400.0, solid_count=2), family="bracket")
    text = joined(verdict)
    for fragment in ("2 solids", "held faces", "direction", "size", "Poisson"):
        assert fragment in text
    assert len(verdict.reasons) >= 5


def test_ranges_cover_every_family_preset() -> None:
    assert set(FAMILY_LENGTH_RANGES_MM) == set(FAMILY_FACE_PRESETS)


def test_mesh_volume_comes_from_the_tetrahedra_when_not_given() -> None:
    nodes = np.array([[0, 0, 0], [6, 0, 0], [0, 6, 0], [0, 0, 6]], float)
    mesh = SimpleNamespace(nodes=nodes, elements=np.array([[0, 1, 2, 3]]))
    assert mesh_volume_mm3(mesh) == pytest.approx(36.0)
    assert mesh_volume_mm3(SimpleNamespace()) is None


def test_the_verdict_serialises() -> None:
    document = check(bracket_case(), box_mesh(), family="bracket").as_dict()
    assert document["available"] is True
    assert document["direction"] == pytest.approx([0.0, 0.0, -1.0])
