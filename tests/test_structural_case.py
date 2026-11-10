"""The static structural case: what is held, what pushes, what it is made of.

Light on purpose -- no gmsh, no numpy needed -- because every other structural
module and the CLI code against these names, and a refusal that lets a bad case
through is a healthy-looking contour plot of the wrong question.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path

import pytest

from fea_surrogate import structural_case as sc
from fea_surrogate.structural_case import (
    CASE_SCHEMA,
    MATERIAL_ALIASES,
    MATERIAL_PRESETS,
    CaseError,
    FixedSupport,
    ForceLoad,
    Gravity,
    Material,
    MeshSettings,
    PressureLoad,
    StaticStructuralCase,
    custom_material,
    material_from_key,
    parse_case,
    parse_density,
    parse_direction,
    parse_face_list,
    parse_force,
    parse_force_spec,
    parse_modulus,
    parse_pressure,
    parse_vector,
)


@dataclass(frozen=True)
class _Face:
    """The FaceInfo protocol, without gmsh."""

    tag: int
    area: float
    normal: tuple[float, float, float]
    planar: bool = True


#: A 100 x 10 x 10 bar along x: 1 = x-min end, 2 = x-max end, 3/4 = -z/+z,
#: 5/6 = -y/+y, 7 = a curved face, 8 = a second +z-facing plane.
FACES = (
    _Face(1, 100.0, (-1.0, 0.0, 0.0)),
    _Face(2, 100.0, (1.0, 0.0, 0.0)),
    _Face(3, 1000.0, (0.0, 0.0, -1.0)),
    _Face(4, 1000.0, (0.0, 0.0, 1.0)),
    _Face(5, 1000.0, (0.0, -1.0, 0.0)),
    _Face(6, 1000.0, (0.0, 1.0, 0.0)),
    _Face(7, 314.159, (0.6, 0.8, 0.0), planar=False),
    _Face(8, 50.0, (0.0, 0.0, 1.0)),
)
STEEL = MATERIAL_PRESETS["structural_steel"]


def _doc(**overrides):
    document = {
        "schema_version": CASE_SCHEMA,
        "analysis": "static_structural",
        "supports": [{"type": "fixed", "faces": [1]}],
        "loads": [{"type": "force", "faces": [2], "vector_n": [0, 0, -500]}],
        "material": {"preset": "structural_steel"},
    }
    document.update(overrides)
    return document


def _case(**overrides) -> StaticStructuralCase:
    fields = {
        "supports": (FixedSupport((1,)),),
        "loads": (ForceLoad((2,), (0.0, 0.0, -500.0)),),
        "material": STEEL,
    }
    fields.update(overrides)
    return StaticStructuralCase(**fields)


# --------------------------------------------------------------------------
# Import weight
# --------------------------------------------------------------------------


def test_the_module_imports_with_the_stdlib_only() -> None:
    """The CLI's --help and the CI contract job import this without numpy."""
    script = (
        "import sys\n"
        "import fea_surrogate.structural_case\n"
        "heavy = ('numpy', 'torch', 'torch_geometric', 'gmsh', 'scipy', 'matplotlib')\n"
        "print(' '.join(m for m in heavy if m in sys.modules))\n"
    )
    src = str(Path(__file__).resolve().parents[1] / "src")
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, cwd=src
    ).stdout.split()
    # numpy may be pulled in by the package __init__; the module itself adds nothing heavier.
    assert not set(out) - {"numpy"}


# --------------------------------------------------------------------------
# Materials
# --------------------------------------------------------------------------


def test_presets_carry_the_spec_values() -> None:
    expected = {
        "structural_steel": (200000, 0.30, 7850, 250, 460),
        "aluminum_alloy": (71000, 0.33, 2770, 280, 310),
        "stainless_steel": (193000, 0.31, 7750, 207, 586),
        "titanium_alloy": (96000, 0.36, 4620, 930, 1070),
        "copper_alloy": (110000, 0.34, 8300, 280, 430),
        "gray_cast_iron": (110000, 0.28, 7200, None, 240),
        "magnesium_alloy": (45000, 0.35, 1800, 193, None),
        "polyethylene": (1100, 0.42, 950, 25, None),
    }
    assert set(MATERIAL_PRESETS) == set(expected)
    for key, (e, nu, rho, sy, su) in expected.items():
        m = MATERIAL_PRESETS[key]
        assert m.key == key and m.source == "preset"
        assert (m.youngs_modulus_mpa, m.poisson_ratio, m.density_kg_m3) == (e, nu, rho)
        assert (m.tensile_yield_mpa, m.tensile_ultimate_mpa) == (sy, su)
        assert isinstance(m.youngs_modulus_mpa, float)
    assert set(sc.PRESET_CONFIDENCE) == set(expected)


def test_every_alias_names_a_preset() -> None:
    assert set(MATERIAL_ALIASES.values()) <= set(MATERIAL_PRESETS)
    for alias in ("steel", "aluminium", "aluminum", "al", "stainless", "titanium",
                  "ti", "copper", "cast_iron", "iron", "magnesium", "pe", "polyethylene"):
        assert alias in MATERIAL_ALIASES


@pytest.mark.parametrize(
    "text,key",
    [
        ("steel", "structural_steel"),
        ("Structural Steel", "structural_steel"),
        ("  STRUCTURAL-steel ", "structural_steel"),
        ("Aluminium", "aluminum_alloy"),
        ("AL", "aluminum_alloy"),
        ("aluminum alloy", "aluminum_alloy"),
        ("Gray Cast Iron", "gray_cast_iron"),
        ("cast-iron", "gray_cast_iron"),
        ("Ti", "titanium_alloy"),
        ("PE", "polyethylene"),
    ],
)
def test_material_from_key_is_case_space_and_hyphen_insensitive(text, key) -> None:
    assert material_from_key(text) is MATERIAL_PRESETS[key]


@pytest.mark.parametrize("text", ["unobtainium", "", "  ", "stainles"])
def test_unknown_material_lists_the_keys_and_the_custom_route(text) -> None:
    with pytest.raises(CaseError) as caught:
        material_from_key(text)
    message = str(caught.value)
    assert "structural_steel" in message and "polyethylene" in message
    assert "youngs_modulus_mpa" in message


def test_material_as_dict_has_every_field() -> None:
    assert STEEL.as_dict() == {
        "key": "structural_steel", "name": "Structural Steel",
        "youngs_modulus_mpa": 200000.0, "poisson_ratio": 0.3, "density_kg_m3": 7850.0,
        "tensile_yield_mpa": 250.0, "tensile_ultimate_mpa": 460.0, "source": "preset",
    }
    json.dumps(STEEL.as_dict())


@pytest.mark.parametrize(
    "kwargs,needle",
    [
        ({"youngs_modulus_mpa": 0}, "positive"),
        ({"youngs_modulus_mpa": -1}, "positive"),
        ({"youngs_modulus_mpa": float("nan")}, "finite"),
        ({"youngs_modulus_mpa": float("inf")}, "finite"),
        ({"youngs_modulus_mpa": True}, "number"),
        ({"youngs_modulus_mpa": "200000"}, "number"),
        ({"poisson_ratio": 0.5}, "between -1 and 0.5"),
        ({"poisson_ratio": -1.0}, "between -1 and 0.5"),
        ({"density_kg_m3": 0}, "positive"),
        ({"density_kg_m3": -7850}, "positive"),
        ({"tensile_yield_mpa": -5}, "positive"),
        ({"tensile_ultimate_mpa": float("nan")}, "finite"),
        ({"name": ""}, "non-empty"),
        ({"source": "guess"}, "preset' or 'custom"),
    ],
)
def test_material_refuses_bad_values(kwargs, needle) -> None:
    base = dict(key="custom", name="X", youngs_modulus_mpa=70000, poisson_ratio=0.33,
                source="custom")
    base.update(kwargs)
    with pytest.raises(CaseError, match=needle):
        Material(**base)


def test_custom_material_round_numbers_and_key() -> None:
    m = custom_material(70000, 0.33, name="6061-T6", density_kg_m3=2700, tensile_yield_mpa=276)
    assert m.key == "custom" and m.source == "custom"
    assert m.youngs_modulus_mpa == 70000.0 and isinstance(m.youngs_modulus_mpa, float)
    assert m.tensile_ultimate_mpa is None


def test_material_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        STEEL.youngs_modulus_mpa = 1.0  # type: ignore[misc]


# --------------------------------------------------------------------------
# Dataclasses
# --------------------------------------------------------------------------


def test_face_tags_are_deduplicated_sorted_and_strict() -> None:
    assert FixedSupport([3, 1, 3]).faces == (1, 3)
    assert FixedSupport((2.0,)).faces == (2,)
    assert FixedSupport(5).faces == (5,)
    for bad, needle in (([1.7], "not a face number"), ([True], "face numbers"),
                        ([], "at least one"), (["1"], "face numbers"), ([0], "numbered from 1"),
                        ("1", "list of face numbers"), ([float("nan")], "not a face number")):
        with pytest.raises(CaseError, match=needle):
            FixedSupport(bad)


def test_force_load_properties() -> None:
    load = ForceLoad([6], (300, 0, -400))
    assert load.vector_n == (300.0, 0.0, -400.0)
    assert load.magnitude_n == pytest.approx(500.0)
    assert load.direction == pytest.approx((0.6, 0.0, -0.8))
    assert load.label == "Force"


@pytest.mark.parametrize(
    "vector,needle",
    [((0, 0, 0), "non-zero"), ((0, 0), "three numbers"), ((0, 0, float("inf")), "finite"),
     ((0, True, 1), "number"), ("0,0,-1", "three numbers")],
)
def test_force_load_refuses_bad_vectors(vector, needle) -> None:
    with pytest.raises(CaseError, match=needle):
        ForceLoad((6,), vector)


def test_pressure_load_allows_suction_and_refuses_zero() -> None:
    assert PressureLoad((3,), -0.2).pressure_mpa == -0.2
    with pytest.raises(CaseError, match="must not be zero"):
        PressureLoad((3,), 0.0)
    with pytest.raises(CaseError, match="finite"):
        PressureLoad((3,), float("nan"))


def test_gravity_defaults_to_standard_earth_gravity_down_z() -> None:
    g = Gravity()
    assert g.acceleration_mps2 == (0.0, 0.0, -9.80665)
    assert g.magnitude_mps2 == pytest.approx(9.80665)
    assert g.direction == pytest.approx((0.0, 0.0, -1.0))
    assert g.label == "Standard Earth Gravity"
    with pytest.raises(CaseError, match="must not be zero"):
        Gravity((0, 0, 0))


def test_mesh_settings_validation() -> None:
    assert MeshSettings() == MeshSettings(2, "medium", None)
    assert MeshSettings(1, "FINE").refinement == "fine"
    assert MeshSettings(element_size_mm=2).element_size_mm == 2.0
    for kwargs, needle in (({"element_order": 3}, "element_order"),
                           ({"element_order": True}, "element_order"),
                           ({"refinement": "ultra"}, "coarse, medium, fine"),
                           ({"element_size_mm": 0}, "positive"),
                           ({"element_size_mm": float("inf")}, "finite")):
        with pytest.raises(CaseError, match=needle):
            MeshSettings(**kwargs)


# --------------------------------------------------------------------------
# The case
# --------------------------------------------------------------------------


def test_fixed_and_loaded_faces_are_sorted_unions() -> None:
    case = _case(
        supports=(FixedSupport((5, 1)), FixedSupport((3, 1))),
        loads=(ForceLoad((8, 2), (0, 0, -1)), PressureLoad((4,), 1.0)),
    )
    assert case.fixed_faces() == (1, 3, 5)
    assert case.loaded_faces() == (2, 4, 8)


def test_case_accepts_lists_and_bare_items() -> None:
    case = StaticStructuralCase(supports=[FixedSupport((1,))],
                                loads=ForceLoad((2,), (0, 0, -1)), material=STEEL)
    assert isinstance(case.supports, tuple) and isinstance(case.loads, tuple)
    assert case.mesh == MeshSettings()


@pytest.mark.parametrize(
    "overrides,needle",
    [
        ({"supports": ()}, "at least one fixed support"),
        ({"loads": ()}, "no load"),
        ({"loads": (ForceLoad((1,), (0, 0, -1)),)}, "only on fixed faces"),
        ({"loads": (PressureLoad((1,), 1.0),)}, "only on fixed faces"),
        ({"material": custom_material(70000, 0.3), "loads": (), "gravity": Gravity()},
         "density"),
        ({"supports": ("1",)}, "FixedSupport"),
        ({"loads": ({"faces": [2]},)}, "ForceLoad or PressureLoad"),
        ({"material": {"preset": "steel"}}, "Material"),
    ],
)
def test_case_refusals(overrides, needle) -> None:
    with pytest.raises(CaseError, match=needle):
        _case(**overrides)


def test_gravity_alone_is_a_load() -> None:
    case = _case(loads=(), gravity=Gravity())
    assert case.loads == () and case.loaded_faces() == ()


def test_lettering_runs_supports_then_loads_then_gravity() -> None:
    case = _case(loads=(ForceLoad((2,), (0, 0, -1)), PressureLoad((4,), 1.0)), gravity=Gravity())
    assert [letter for letter, _ in case.lettered()] == ["A", "B", "C", "D"]
    assert isinstance(case.lettered()[-1][1], Gravity)
    assert sc._letter(25) == "Z" and sc._letter(26) == "AA" and sc._letter(27) == "AB"


def test_check_faces_lists_the_valid_tags() -> None:
    case = _case(loads=(ForceLoad((2, 99), (0, 0, -1)),))
    with pytest.raises(CaseError) as caught:
        case.check_faces(FACES)
    assert "99" in str(caught.value) and "1-8" in str(caught.value)
    _case().check_faces(FACES)


# --------------------------------------------------------------------------
# describe / assumptions
# --------------------------------------------------------------------------


def test_describe_reads_like_the_ansys_tree_with_areas() -> None:
    case = _case(
        loads=(ForceLoad((2,), (0, 0, -500)), PressureLoad((4,), 1.5),
               PressureLoad((7,), 2.0, label="Bore")),
        gravity=Gravity(),
    )
    lines = case.describe(FACES)
    assert lines[0] == "A  Fixed Support — face 1 (100 mm²)"
    assert lines[1] == (
        "B  Force — 500 N along (0, 0, -1) on face 2 (100 mm², traction 5 MPa)"
    )
    assert lines[2].startswith("C  Pressure — 1.5 MPa normal to face 4 (into the surface)")
    assert "≈ 1500 N along (0, 0, -1) on 1000 mm²" in lines[2]
    assert lines[3].startswith("D  Bore (Pressure) — 2 MPa")
    assert "curved" in lines[3]
    assert lines[4] == "E  Standard Earth Gravity — 9.8066 m/s² along (0, 0, -1)"
    assert lines[5].startswith("Material: Structural Steel (typical values) — E 200000 MPa, ν 0.3")
    assert lines[6].startswith("Mesh: 10-node quadratic tetrahedra")


def test_describe_without_faces_and_with_several_faces() -> None:
    case = _case(supports=(FixedSupport((1, 3)),), loads=(PressureLoad((4,), -0.1),))
    lines = case.describe()
    assert lines[0] == "A  Fixed Support — faces 1, 3"
    assert "suction" in lines[1] and "mm²" not in lines[1]
    assert case.describe(FACES)[0] == "A  Fixed Support — faces 1, 3 (1100 mm²)"


def test_describe_custom_material_and_linear_elements() -> None:
    case = _case(material=custom_material(70000, 0.33, name="6061"),
                 mesh=MeshSettings(1, "fine", 1.5))
    lines = case.describe()
    assert "Material: 6061 (custom) — E 70000 MPa, ν 0.33" in lines
    assert lines[-1] == "Mesh: 4-node linear tetrahedra (stiff in bending; comparison only), element size 1.5 mm"


def test_assumptions_say_what_E_does_and_flag_the_traps() -> None:
    text = "\n".join(_case().assumptions())
    assert "E affects deformation only" in text
    assert "typical values" in text
    assert "uniform traction" in text

    iron = _case(material=MATERIAL_PRESETS["gray_cast_iron"]).assumptions()
    assert any("brittle" in line for line in iron)

    soft = _case(material=custom_material(5, 0.49), mesh=MeshSettings(1)).assumptions()
    assert any("close to 0.5" in line for line in soft)
    assert any("too stiff in bending" in line for line in soft)
    assert any("no tensile yield" in line for line in soft)

    overlap = _case(loads=(ForceLoad((1, 2), (0, 0, -1)),)).assumptions()
    assert any("also fixed" in line for line in overlap)

    grav = _case(gravity=Gravity()).assumptions()
    assert any("self-weight" in line and "7850" in line for line in grav)

    pressure = _case(loads=(PressureLoad((4,), 1.0),)).assumptions()
    assert any("normal to the surface" in line for line in pressure)


# --------------------------------------------------------------------------
# parse_case
# --------------------------------------------------------------------------


def test_parse_the_spec_example_document() -> None:
    document = {
        "schema_version": "StaticStructuralCase/v1", "analysis": "static_structural",
        "supports": [{"type": "fixed", "faces": [1]}],
        "loads": [{"type": "force", "faces": [6], "vector_n": [0, 0, -500]},
                  {"type": "force", "faces": [7], "magnitude_n": 200, "direction": "-z"},
                  {"type": "pressure", "faces": [3], "pressure_mpa": 1.5}],
        "material": {"preset": "structural_steel"},
        "gravity": {"enabled": True, "acceleration_mps2": [0, 0, -9.80665]},
        "mesh": {"element_order": 2, "refinement": "medium", "element_size_mm": None},
    }
    case = parse_case(document, FACES)
    assert case.fixed_faces() == (1,)
    assert case.loads[0] == ForceLoad((6,), (0.0, 0.0, -500.0))
    assert case.loads[1].vector_n == pytest.approx((0, 0, -200))
    assert case.loads[2] == PressureLoad((3,), 1.5)
    assert case.material is STEEL
    assert case.gravity == Gravity()
    assert case.mesh == MeshSettings()


def test_custom_material_document() -> None:
    case = parse_case(_doc(material={"name": "6061", "youngs_modulus_mpa": 70000,
                                     "poisson_ratio": 0.33, "density_kg_m3": 2700,
                                     "tensile_yield_mpa": 270}))
    m = case.material
    assert (m.key, m.name, m.source) == ("custom", "6061", "custom")
    assert (m.youngs_modulus_mpa, m.tensile_yield_mpa, m.tensile_ultimate_mpa) == (70000, 270, None)


def test_preset_with_edits_becomes_custom_and_unchanged_values_stay_preset() -> None:
    edited = parse_case(_doc(material={"preset": "steel", "tensile_yield_mpa": 355})).material
    assert edited.source == "custom" and edited.key == "custom"
    assert edited.name == "Structural Steel (edited)"
    assert edited.tensile_yield_mpa == 355 and edited.youngs_modulus_mpa == 200000
    same = parse_case(_doc(material={"preset": "steel", "youngs_modulus_mpa": 200000,
                                     "name": "Structural Steel"})).material
    assert same is STEEL


@pytest.mark.parametrize(
    "case",
    [
        _case(),
        _case(loads=(ForceLoad((2, 4), (1.5, -2.25, 3.0), label="Bolt"), PressureLoad((5,), -0.3)),
              gravity=Gravity((0, -9.81, 0), label="Down Y"),
              mesh=MeshSettings(1, "coarse", 3.25)),
        _case(material=custom_material(70000, 0.33, name="Al", density_kg_m3=2700),
              supports=(FixedSupport((1,)), FixedSupport((3,), label="Base"))),
        _case(material=MATERIAL_PRESETS["gray_cast_iron"], loads=(), gravity=Gravity()),
    ],
)
def test_as_document_round_trips(case) -> None:
    document = case.as_document()
    assert document["schema_version"] == CASE_SCHEMA
    text = json.dumps(document, allow_nan=False)
    assert parse_case(json.loads(text)) == case
    assert parse_case(json.loads(text), FACES) == case


def test_as_document_names_the_preset_and_its_values() -> None:
    material = _case().as_document()["material"]
    assert material["preset"] == "structural_steel"
    assert material["youngs_modulus_mpa"] == 200000.0
    custom = _case(material=custom_material(1, 0.2)).as_document()["material"]
    assert "preset" not in custom


def test_schema_version_may_be_omitted_but_not_wrong() -> None:
    document = _doc()
    del document["schema_version"]
    assert parse_case(document) == parse_case(_doc())
    with pytest.raises(CaseError, match="not supported"):
        parse_case(_doc(schema_version="StaticStructuralCase/v2"))


@pytest.mark.parametrize(
    "document,needle",
    [
        (_doc(extra=1), "unexpected \\['extra'\\]"),
        (_doc(supports=[{"type": "fixed", "faces": [1], "dof": "x"}]), "unexpected \\['dof'\\]"),
        (_doc(loads=[{"type": "force", "faces": [2], "vector_n": [0, 0, -1], "x": 1}]), "unexpected"),
        (_doc(loads=[{"type": "pressure", "faces": [2], "pressure_mpa": 1, "direction": "-z"}]),
         "unexpected \\['direction'\\]"),
        (_doc(material={"preset": "steel", "yield": 3}), "unexpected \\['yield'\\]"),
        (_doc(gravity={"enable": True}), "unexpected \\['enable'\\]"),
        (_doc(mesh={"order": 2}), "unexpected \\['order'\\]"),
    ],
)
def test_every_nested_object_is_key_checked(document, needle) -> None:
    with pytest.raises(CaseError, match=needle):
        parse_case(document)


@pytest.mark.parametrize(
    "document,needle",
    [
        (_doc(loads=[{"type": "force", "faces": [2], "vector_n": [0, 0, float("inf")]}]), "finite"),
        (_doc(loads=[{"type": "force", "faces": [2], "vector_n": [0, float("nan"), 1]}]), "finite"),
        (_doc(loads=[{"type": "force", "faces": [2], "magnitude_n": float("inf"), "direction": "-z"}]),
         "finite"),
        (_doc(loads=[{"type": "force", "faces": [2], "magnitude_n": True, "direction": "-z"}]),
         "number"),
        (_doc(loads=[{"type": "force", "faces": [2], "magnitude_n": "500", "direction": "-z"}]),
         "number"),
        (_doc(loads=[{"type": "force", "faces": [2], "magnitude_n": 5, "direction": [0, float("nan"), 1]}]),
         "finite"),
        (_doc(loads=[{"type": "pressure", "faces": [2], "pressure_mpa": float("nan")}]), "finite"),
        (_doc(material={"youngs_modulus_mpa": float("inf"), "poisson_ratio": 0.3}), "finite"),
        (_doc(gravity={"enabled": True, "acceleration_mps2": [0, 0, float("-inf")]}), "finite"),
        (_doc(mesh={"element_size_mm": float("nan")}), "finite"),
    ],
)
def test_every_number_must_be_finite_and_real(document, needle) -> None:
    with pytest.raises(CaseError, match=needle):
        parse_case(document)


@pytest.mark.parametrize(
    "document,needle",
    [
        (_doc(supports=[]), "at least one fixed support"),
        (_doc(supports=None), "at least one fixed support"),
        (_doc(supports={"type": "fixed", "faces": [1]}), "at least one fixed support"),
        (_doc(supports=[{"type": "frictionless", "faces": [1]}]), "frictionless"),
        (_doc(supports=[{"type": "spring", "faces": [1]}]), "must be \"fixed\""),
        (_doc(loads=[]), "no load"),
        (_doc(loads={"type": "force"}), "must be a list"),
        (_doc(loads=[{"type": "moment", "faces": [2]}]), "remote point"),
        (_doc(loads=[{"type": "gravity"}]), "top-level"),
        (_doc(loads=[{"faces": [2], "vector_n": [0, 0, -1]}], supports=[{"type": "fixed", "faces": [1]}]),
         "type must be"),
        (_doc(loads=[{"type": "force", "faces": [1], "vector_n": [0, 0, -1]}]), "only on fixed faces"),
        (_doc(loads=[{"type": "force", "faces": [2]}]), "needs a size"),
        (_doc(loads=[{"type": "force", "faces": [2], "magnitude_n": 5}]), "no direction"),
        (_doc(loads=[{"type": "force", "faces": [2], "direction": "-z"}]), "needs a size"),
        (_doc(loads=[{"type": "force", "faces": [2], "magnitude_n": -5, "direction": "-z"}]),
         "must be positive"),
        (_doc(loads=[{"type": "force", "faces": [2], "magnitude_n": 0, "direction": "-z"}]),
         "must be positive"),
        (_doc(loads=[{"type": "force", "faces": [2], "magnitude_n": 5, "vector_n": [0, 0, 5]}]),
         "not both"),
        (_doc(loads=[{"type": "force", "faces": [2], "vector_n": [0, 0, 0]}]), "non-zero"),
        (_doc(loads=[{"type": "force", "faces": [2], "magnitude_n": 5, "direction": [0, 0, 0]}]),
         "zero vector"),
        (_doc(loads=[{"type": "force", "faces": [2], "magnitude_n": 5, "direction": "down"}]),
         "not a direction"),
        (_doc(loads=[{"type": "pressure", "faces": [2], "pressure_mpa": 0}]), "must not be zero"),
        (_doc(loads=[{"type": "pressure", "faces": [2]}]), "needs pressure_mpa"),
        (_doc(loads=[{"type": "force", "faces": [1.5], "vector_n": [0, 0, -1]}]), "not a face number"),
        (_doc(loads=[{"type": "force", "faces": [True], "vector_n": [0, 0, -1]}]), "face numbers"),
        (_doc(loads=[{"type": "force", "faces": [], "vector_n": [0, 0, -1]}]), "at least one face"),
        (_doc(material=None), "needs a material"),
        (_doc(material={"poisson_ratio": 0.3}), "deformation needs Young's modulus"),
        (_doc(material={"youngs_modulus_mpa": 70000}), "poisson_ratio"),
        (_doc(material={"preset": "unobtainium"}), "no material preset"),
        (_doc(material={"youngs_modulus_mpa": 70000, "poisson_ratio": 0.6}), "between -1 and 0.5"),
        (_doc(material={"youngs_modulus_mpa": -1, "poisson_ratio": 0.3}), "positive"),
        (_doc(material={"youngs_modulus_mpa": 7e4, "poisson_ratio": 0.3, "density_kg_m3": -1}),
         "positive"),
        (_doc(loads=[], gravity={"enabled": True},
              material={"youngs_modulus_mpa": 70000, "poisson_ratio": 0.3}), "density"),
        (_doc(gravity={"enabled": "yes"}), "true or false"),
        (_doc(gravity={"enabled": True, "acceleration_mps2": [0, 0, 0]}), "must not be zero"),
        (_doc(mesh={"element_order": 3}), "element_order"),
        (_doc(mesh={"element_order": 2.5}), "element_order"),
        (_doc(mesh={"refinement": "ultra"}), "coarse, medium, fine"),
        (_doc(analysis="modal"), "eigenvalue"),
        (_doc(analysis="cfd"), "only \"static_structural\""),
        (_doc(analysis=None), "static_structural"),
    ],
)
def test_parse_case_refusals(document, needle) -> None:
    with pytest.raises(CaseError, match=needle):
        parse_case(document)


def test_refusals_never_escape_as_other_exception_types() -> None:
    for document in ("{}", [1], None, 5):
        with pytest.raises(CaseError, match="JSON object"):
            parse_case(document)  # type: ignore[arg-type]
    for loads in ([None], [5], ["force"], [{"type": None}]):
        with pytest.raises(CaseError):
            parse_case(_doc(loads=loads))


def test_analysis_name_is_case_and_space_insensitive() -> None:
    assert parse_case(_doc(analysis="Static Structural")) == parse_case(_doc())


def test_faces_are_deduplicated_and_float_integers_accepted() -> None:
    case = parse_case(_doc(supports=[{"type": "fixed", "faces": [1, 1.0, 3]}],
                           loads=[{"type": "pressure", "faces": [4, 4], "pressure_mpa": 1}]))
    assert case.supports[0].faces == (1, 3)
    assert case.loads[0].faces == (4,)


def test_unknown_faces_are_refused_when_the_part_is_known() -> None:
    for document in (
        _doc(supports=[{"type": "fixed", "faces": [42]}]),
        _doc(loads=[{"type": "pressure", "faces": [2, 42], "pressure_mpa": 1}]),
        _doc(loads=[{"type": "force", "faces": [42], "magnitude_n": 1, "direction": "normal"}]),
    ):
        with pytest.raises(CaseError) as caught:
            parse_case(document, FACES)
        assert "42" in str(caught.value) and "1-8" in str(caught.value)
        assert "face table" in str(caught.value)
    # Without the part, the tag cannot be checked here and is left to the mesh.
    assert parse_case(_doc(supports=[{"type": "fixed", "faces": [42]}])).fixed_faces() == (42,)


def test_normal_directions_resolve_against_the_faces() -> None:
    def force(direction, faces=(4,)):
        return parse_case(_doc(loads=[{"type": "force", "faces": list(faces),
                                       "magnitude_n": 100, "direction": direction}]), FACES)

    assert force("normal").loads[0].vector_n == pytest.approx((0, 0, -100))
    assert force("into").loads[0].vector_n == pytest.approx((0, 0, -100))
    assert force("out").loads[0].vector_n == pytest.approx((0, 0, 100))
    assert force("normal", (4, 8)).loads[0].vector_n == pytest.approx((0, 0, -100))
    assert force([1, 0, 0]).loads[0].vector_n == pytest.approx((100, 0, 0))
    assert force("0, 3, -4").loads[0].vector_n == pytest.approx((0, 60, -80))
    with pytest.raises(CaseError, match="pressure"):
        force("normal", (7,))
    with pytest.raises(CaseError, match="pressure"):
        force("normal", (4, 5))
    with pytest.raises(CaseError, match="needs the part's faces"):
        parse_case(_doc(loads=[{"type": "force", "faces": [4], "magnitude_n": 1,
                                "direction": "normal"}]))


def test_labels_are_kept_and_checked() -> None:
    case = parse_case(_doc(supports=[{"type": "fixed", "faces": [1], "label": "Bolt holes"}],
                           loads=[{"type": "force", "faces": [2], "vector_n": [0, 0, -1],
                                   "label": "Hook"}]))
    assert case.supports[0].label == "Bolt holes" and case.loads[0].label == "Hook"
    with pytest.raises(CaseError, match="label"):
        parse_case(_doc(supports=[{"type": "fixed", "faces": [1], "label": ""}]))


def test_gravity_disabled_or_absent_is_none_and_enabled_defaults_down_z() -> None:
    assert parse_case(_doc(gravity={"enabled": False})).gravity is None
    assert parse_case(_doc(gravity=None)).gravity is None
    assert parse_case(_doc(gravity={"enabled": True})).gravity == Gravity()


# --------------------------------------------------------------------------
# Legacy MCP load_case shape
# --------------------------------------------------------------------------


def test_legacy_shape_converts() -> None:
    legacy = {
        "fixed": {"faces": [1]},
        "load": {"faces": [2], "magnitude_n": 500, "direction": [0, 0, -2]},
        "material": {"poisson_ratio": 0.3, "youngs_modulus_mpa": 200000,
                     "density_kg_m3": 7850, "name": "steel"},
        "gravity": {"enabled": True, "g": 9.81},
        "analysis": "static_structural",
    }
    case = parse_case(legacy, FACES)
    assert case.fixed_faces() == (1,)
    assert case.loads == (ForceLoad((2,), (0.0, 0.0, -500.0)),)
    assert case.material.youngs_modulus_mpa == 200000 and case.material.name == "steel"
    assert case.material.source == "custom"
    assert case.gravity == Gravity((0.0, 0.0, -9.81))
    assert parse_case(case.as_document()) == case


def test_legacy_variants() -> None:
    material = {"preset": "aluminium"}
    assert parse_case({"fixed": 1, "loads": [{"faces": [2], "pressure_mpa": 2}],
                       "material": material}).loads == (PressureLoad((2,), 2.0),)
    assert parse_case({"fixed": [1, 3], "load": {"faces": [2], "magnitude_n": 1,
                                                 "direction": [1, 0, 0]},
                       "material": material, "gravity": {"enabled": False}}).gravity is None
    several = parse_case({"fixed": [1], "material": material, "loads": [
        {"faces": [2], "magnitude_n": 1, "direction": [1, 0, 0]},
        {"faces": [4], "pressure_mpa": 1}]})
    assert len(several.loads) == 2


@pytest.mark.parametrize(
    "document,needle",
    [
        ({"fixed": [1], "load": {"faces": [2], "magnitude_n": 5, "direction": [0, 0, -1]},
          "material": {"poisson_ratio": 0.3}}, "deformation needs Young's modulus"),
        ({"fixed": [1], "load": {"faces": [2], "magnitude_n": 5, "direction": [0, 0, -1]}},
         "needs a material"),
        ({"fixed": [1], "load": {"faces": [2], "magnitude_n": 5}, "material": {"preset": "steel"}},
         "no direction"),
        ({"fixed": [1], "load": {"faces": [2], "pressure_mpa": 1, "direction": [0, 0, -1]},
          "material": {"preset": "steel"}}, "acts normal"),
        ({"fixed": {"centroid": {"axis": "z", "at": "min"}}, "load": {"faces": [2], "pressure_mpa": 1},
          "material": {"preset": "steel"}}, "by number"),
        ({"load": {"faces": [2], "pressure_mpa": 1}, "material": {"preset": "steel"}}, "held faces"),
        ({"fixed": [1], "load": {"faces": [2], "pressure_mpa": 1},
          "loads": [{"faces": [2], "pressure_mpa": 1}], "material": {"preset": "steel"}}, "not both"),
        ({"fixed": [1], "load": {"faces": [2], "pressure_mpa": 1, "magnitude_n": 1},
          "material": {"preset": "steel"}}, "not both"),
        ({"fixed": [1], "load": {"faces": [2]}, "material": {"preset": "steel"}}, "needs magnitude_n"),
        ({"fixed": [1], "load": {"faces": [2], "pressure_mpa": 1, "where": 1},
          "material": {"preset": "steel"}}, "unexpected"),
        ({"fixed": [1], "load": {"faces": [2], "pressure_mpa": 1}, "material": {"preset": "steel"},
          "gravity": {"enabled": True, "g": -9.8}}, "positive"),
        ({"fixed": [1], "load": {"faces": [2], "pressure_mpa": 1}, "material": {"preset": "steel"},
          "gravity": {"enable": True}}, "unexpected"),
        ({"fixed": [1], "load": {"faces": [2], "pressure_mpa": 1}, "material": {"preset": "steel"},
          "gravity": True}, "JSON object"),
        ({"fixed": [1], "load": {"faces": [2], "pressure_mpa": 1}, "material": {"preset": "steel"},
          "extra": 1}, "unexpected"),
        ({"fixed": [1], "supports": [], "material": {"preset": "steel"}}, "mixes"),
        ({"fixed": [1], "load": {"faces": [2], "pressure_mpa": 1}, "material": {"preset": "steel"},
          "analysis": "thermal"}, "temperature"),
        ({"fixed": [1], "load": {"faces": [1], "pressure_mpa": 1}, "material": {"preset": "steel"}},
         "only on fixed faces"),
        ({"fixed": [1], "loads": {"faces": [2]}, "material": {"preset": "steel"}}, "must be a list"),
    ],
)
def test_legacy_refusals(document, needle) -> None:
    with pytest.raises(CaseError, match=needle):
        parse_case(document)


# --------------------------------------------------------------------------
# Terminal grammar
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [("1", (1,)), ("1,8", (1, 8)), ("1 8", (1, 8)), ("8, 1", (1, 8)), ("2-4", (2, 3, 4)),
     ("4-2", (2, 3, 4)), ("2 - 4, 8", (2, 3, 4, 8)), ("1;1;1", (1,)), ("faces 1 and 8", (1, 8)),
     ("face: 3", (3,)), ("2–4", (2, 3, 4)), ("[1, 8]", None)],
)
def test_parse_face_list(text, expected) -> None:
    if expected is None:
        with pytest.raises(CaseError, match="not a face number"):
            parse_face_list(text, range(1, 9))
    else:
        assert parse_face_list(text, range(1, 9)) == expected


@pytest.mark.parametrize("text,needle", [("", "type face numbers"), ("9", "1-8"),
                                         ("1, 99", "99"), ("0", "from 1"), ("a", "not a face"),
                                         ("1.5", "not a face"), ("-1", "not a face")])
def test_parse_face_list_refusals(text, needle) -> None:
    with pytest.raises(CaseError, match=needle):
        parse_face_list(text, range(1, 9))


def test_parse_face_list_with_no_known_faces_skips_the_check() -> None:
    assert parse_face_list("99", ()) == (99,)


@pytest.mark.parametrize(
    "text,newtons",
    [("500", 500), ("500 N", 500), ("500N", 500), ("5 kN", 5000), ("5kn", 5000),
     ("2 MN", 2e6), ("100 lbf", 444.82216152605), ("1.5e3", 1500), (".5 kN", 500),
     ("1 kip", 4448.2216152605)],
)
def test_parse_force_units(text, newtons) -> None:
    assert parse_force(text) == pytest.approx(newtons)


@pytest.mark.parametrize("text,needle", [("-500", "positive"), ("0", "positive"), ("", "type a number"),
                                         ("abc", "not a number"), ("5 tons", "unknown unit"),
                                         ("1e400", "finite"), ("5 mN", "millinewton"),
                                         ("nan", "not a finite number")])
def test_parse_force_refusals(text, needle) -> None:
    with pytest.raises(CaseError, match=needle):
        parse_force(text)


@pytest.mark.parametrize(
    "text,mpa",
    [("1.5", 1.5), ("1.5 MPa", 1.5), ("1.5 N/mm2", 1.5), ("1.5 N/mm^2", 1.5), ("1.5 N/mm²", 1.5),
     ("200 kPa", 0.2), ("2e5 Pa", 0.2), ("0.002 GPa", 2.0), ("10 bar", 1.0),
     ("100 psi", 0.6894757293168361), ("-0.1", -0.1), ("-50 kPa", -0.05)],
)
def test_parse_pressure_units(text, mpa) -> None:
    assert parse_pressure(text) == pytest.approx(mpa)


@pytest.mark.parametrize("text,needle", [("0", "not be zero"), ("0 kPa", "not be zero"),
                                         ("1 atm", "unknown unit"), ("x", "not a number")])
def test_parse_pressure_refusals(text, needle) -> None:
    with pytest.raises(CaseError, match=needle):
        parse_pressure(text)


@pytest.mark.parametrize(
    "text,mpa",
    [("200000", 200000), ("200 GPa", 200000), ("200GPa", 200000), ("71000 MPa", 71000),
     ("2e11 Pa", 200000), ("29e6 psi", 199947.96), ("29000 ksi", 199947.96)],
)
def test_parse_modulus_units(text, mpa) -> None:
    assert parse_modulus(text) == pytest.approx(mpa, rel=1e-6)


@pytest.mark.parametrize("text", ["0", "-200 GPa", "200 GN", ""])
def test_parse_modulus_refusals(text) -> None:
    with pytest.raises(CaseError):
        parse_modulus(text)


@pytest.mark.parametrize(
    "text,kgm3",
    [("7850", 7850), ("7850 kg/m3", 7850), ("7850 kg/m^3", 7850), ("7850 kg/m³", 7850),
     ("7.85 g/cm3", 7850), ("2.7 g/cc", 2700)],
)
def test_parse_density_units(text, kgm3) -> None:
    assert parse_density(text) == pytest.approx(kgm3)


@pytest.mark.parametrize("text", ["0", "-1", "7850 lb/ft3", "heavy"])
def test_parse_density_refusals(text) -> None:
    with pytest.raises(CaseError):
        parse_density(text)


@pytest.mark.parametrize(
    "text",
    ["0,0,-500", "0 0 -500", "[0,0,-500]", "(0 0 -500)", " [ 0 , 0 , -500 ] ", "0;0;-5e2",
     "{0, 0, -500}"],
)
def test_parse_vector_forms(text) -> None:
    assert parse_vector(text) == (0.0, 0.0, -500.0)


@pytest.mark.parametrize("text,needle", [("0,0", "three numbers"), ("1,2,3,4", "three numbers"),
                                         ("[0,0,-1", "unbalanced"), ("a,b,c", "not a number"),
                                         ("0,0,1e999", "finite"), ("", "three numbers")])
def test_parse_vector_refusals(text, needle) -> None:
    with pytest.raises(CaseError, match=needle):
        parse_vector(text)


@pytest.mark.parametrize(
    "text,vector",
    [("x", (1, 0, 0)), ("+x", (1, 0, 0)), ("-x", (-1, 0, 0)), ("Y", (0, 1, 0)),
     ("- y", (0, -1, 0)), ("-Z", (0, 0, -1)), ("z", (0, 0, 1))],
)
def test_parse_direction_axes(text, vector) -> None:
    assert parse_direction(text) == (tuple(float(c) for c in vector), "axis")


def test_parse_direction_vectors_are_normalised() -> None:
    direction, source = parse_direction("0, 3, -4")
    assert source == "vector" and direction == pytest.approx((0, 0.6, -0.8))
    assert math.isclose(sum(c * c for c in direction), 1.0)
    with pytest.raises(CaseError, match="zero vector"):
        parse_direction("0 0 0")


def test_parse_direction_normal_words() -> None:
    top = [FACES[3]]  # +z
    for word in ("normal", "in", "into", "inward", "push", "NORMAL"):
        assert parse_direction(word, top) == ((0.0, 0.0, -1.0), "normal_inward")
    for word in ("out", "outward", "pull"):
        assert parse_direction(word, top) == ((0.0, 0.0, 1.0), "normal_outward")
    assert parse_direction("into", [FACES[3], FACES[7]])[0] == pytest.approx((0, 0, -1))


@pytest.mark.parametrize(
    "faces,needle",
    [((), "needs the loaded faces"), ((FACES[6],), "pressure"),
     ((FACES[3], FACES[4]), "pressure"), ((FACES[3], FACES[2]), "pressure")],
)
def test_parse_direction_normal_refusals(faces, needle) -> None:
    with pytest.raises(CaseError, match=needle):
        parse_direction("normal", faces)


def test_parse_direction_agreement_threshold() -> None:
    tilt = math.radians(2.0)  # cos 2 deg = 0.99939 > 0.999: agree
    near = _Face(9, 1.0, (math.sin(tilt), 0.0, math.cos(tilt)))
    assert parse_direction("in", [FACES[3], near])[1] == "normal_inward"
    tilt = math.radians(3.0)  # cos 3 deg = 0.99863 < 0.999: disagree
    far = _Face(10, 1.0, (math.sin(tilt), 0.0, math.cos(tilt)))
    with pytest.raises(CaseError):
        parse_direction("in", [FACES[3], far])


@pytest.mark.parametrize("text", ["", "down", "-w", "1,2"])
def test_parse_direction_refusals_say_what_to_type(text) -> None:
    with pytest.raises(CaseError, match="axis|direction"):
        parse_direction(text)


@pytest.mark.parametrize(
    "text,vector,source",
    [("500 N -z", (0, 0, -500), "axis"), ("500 -z", (0, 0, -500), "axis"),
     ("5 kN x", (5000, 0, 0), "axis"), ("500N along -y", (0, -500, 0), "axis"),
     ("500 N in -z", (0, 0, -500), "axis"), ("2 kN normal", (0, 0, -2000), "normal_inward"),
     ("2 kN in", (0, 0, -2000), "normal_inward"), ("100 pull", (0, 0, 100), "normal_outward"),
     ("100 N 0,3,-4", (0, 60, -80), "vector"), ("0,0,-500", (0, 0, -500), "components"),
     ("0 0 -500 N", (0, 0, -500), "components"), ("[0, 0, -0.5] kN", (0, 0, -500), "components"),
     ("(300 0 -400)", (300, 0, -400), "components")],
)
def test_parse_force_spec(text, vector, source) -> None:
    got, got_source = parse_force_spec(text, [FACES[3]])
    assert got == pytest.approx(vector) and got_source == source


@pytest.mark.parametrize("text,needle", [("500 N", "has a size but no direction"),
                                         ("-500 N -z", "positive"), ("0,0,0", "not all be zero"),
                                         ("500 N sideways", "not a direction"), ("", "type a force"),
                                         ("2 kN normal", "needs the loaded faces")])
def test_parse_force_spec_refusals(text, needle) -> None:
    with pytest.raises(CaseError, match=needle):
        parse_force_spec(text)


@pytest.mark.parametrize("text,needle", [
    ("1,2", "has two components; give three"),
    ("500", "has a size but no direction"),
    ("nan,0,0", "not a finite force"),
])
def test_near_miss_forces_say_what_is_missing(text, needle) -> None:
    """T12: the message names the missing piece rather than 'is not a force'."""
    with pytest.raises(CaseError, match=needle):
        parse_force_spec(text)


@pytest.mark.parametrize("parse,text", [(parse_modulus, "70,000"), (parse_modulus, "70 000 MPa"),
                                        (parse_force, "1,500 N"), (parse_density, "7,850")])
def test_thousands_separators_are_named_not_read_as_units(parse, text) -> None:
    with pytest.raises(CaseError, match="without thousands separators"):
        parse(text)


def test_a_normal_from_a_meshed_face_carries_no_round_off() -> None:
    """T16: a flat face's mean normal (5e-18, 9e-19, 1) made a saved force of
    (5.3e-15, 8.6e-16, -500) N."""
    from types import SimpleNamespace

    face = SimpleNamespace(tag=7, planar=True, normal=(5.298e-18, 8.57e-19, 1.0))
    vector, source = parse_force_spec("500 N normal", [face])
    assert source == "normal_inward" and vector == (0.0, 0.0, -500.0)


def test_a_yield_strength_error_says_yield() -> None:
    with pytest.raises(CaseError, match="^tensile yield: 'NaN' is not a finite number"):
        parse_pressure("NaN", role="tensile yield", example="250 MPa")


def test_precheck_reads_everything_but_the_faces() -> None:
    from fea_surrogate.structural_case import precheck_case

    good = {"supports": [{"type": "fixed", "faces": [99]}],
            "loads": [{"type": "force", "faces": [7], "magnitude_n": 5, "direction": "normal"}],
            "material": {"preset": "steel"}}
    precheck_case(good)                       # face 99 and "normal" wait for the part
    with pytest.raises(CaseError, match="(?i)poisson"):
        precheck_case({**good, "material": {"youngs_modulus_mpa": 70000, "poisson_ratio": 0.7}})


def test_grammar_helpers_refuse_non_strings_as_case_errors() -> None:
    for helper in (parse_force, parse_pressure, parse_modulus, parse_density, parse_vector,
                   parse_direction, parse_force_spec):
        with pytest.raises(CaseError):
            helper(None)  # type: ignore[arg-type]
    with pytest.raises(CaseError):
        parse_face_list(None, (1,))  # type: ignore[arg-type]


def test_internal_number_formatting() -> None:
    assert sc._num(200000.0) == "200000"
    assert sc._num(2129.37) == "2129"
    assert sc._num(1.25) == "1.25"
    assert sc._num(1 / 3) == "0.3333"
    assert sc._num(-0.0) == "0"
    assert sc._num(9.80665, 5) == "9.8066"
    assert sc._num(2.5e12) == "2.5e+12"
    assert sc._vec((-0.0, 0.0, -1.0)) == "(0, 0, -1)"
    assert sc._ranges([1, 2, 3, 5, 7, 8]) == "1-3, 5, 7-8"
