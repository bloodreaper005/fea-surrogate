"""What a static structural analysis is asked: supports, loads, material, mesh.

This is the setup tree of an Ansys Mechanical "Static Structural" system turned
into frozen data: Fixed Supports, Forces, Pressures, Standard Earth Gravity, an
Engineering-Data-style material and the mesh controls. The terminal interview,
the `--case` JSON file and the MCP tool all end up here, so a case means the
same thing whichever way it arrived.

It is a separate model from `load_case.LoadCase` on purpose. That one is the
surrogate's screening gate and can only hold what the sweep trained on: one
uniform traction, no body load, E unused. A real analysis needs several loads,
pressures that act normal to curved faces, gravity that is applied rather than
merely checked, and a Young's modulus that matters because deformation is
reported. Widening `LoadCase` would have silently widened what the surrogate is
allowed to screen; the screenability gate derives a legacy single-load document
from this case instead, and only when the case is inside the trained domain.

Two rules run through everything here:

    refuse, do not guess   No default Young's modulus, no default load
                           direction, no float face tags rounded to a face. A
                           wrong guess produces a healthy-looking contour plot
                           of a question nobody asked.
    say what to type       Every refusal names the field and shows an input
                           that would have been accepted.

Import-light by contract (stdlib only): the CLI's `--help`, the case checker and
the CI contract job import this module without numpy's heavier neighbours, gmsh
or matplotlib. `tests/test_boundaries.py` enforces it.
"""

from __future__ import annotations

import math
import numbers
import re
from dataclasses import dataclass, field
from typing import Any, Collection, Iterable, Mapping, Protocol, Sequence, Union


class CaseError(ValueError):
    """The analysis case is incomplete or inconsistent; message says how to fix it."""


CASE_SCHEMA = "StaticStructuralCase/v1"
STATIC_STRUCTURAL = "static_structural"

#: m/s^2. ISO 80000-3 standard gravity, which is also Ansys' "Standard Earth
#: Gravity" magnitude.
STANDARD_GRAVITY = 9.80665

#: The mesh refinement names `structural_mesh.REFINEMENTS` maps to sizes. Kept
#: here as names only so a case can be validated without the mesher.
REFINEMENT_LEVELS = ("coarse", "medium", "fine")
ELEMENT_ORDERS = (1, 2)

#: Two face normals "agree" above this cosine (about 2.6 degrees). Used to decide
#: whether "normal" is one direction for a set of faces.
NORMALS_AGREE_COS = 0.999


class FaceInfo(Protocol):
    """What this module reads from a face: `cad_intake.Face` and
    `structural_mesh.FaceSummary` both satisfy it."""

    tag: int
    area: float
    normal: tuple[float, float, float]
    planar: bool


# --------------------------------------------------------------------------
# Numbers and face tags
# --------------------------------------------------------------------------


def _real(value: Any, role: str, example: str) -> float:
    """A finite real number, refusing bools, strings and NaN/inf.

    `True` is an `int` in Python and JSON `true` would otherwise arrive as 1 N.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise CaseError(
            f"{role} must be a number, not {value!r}; e.g. {example}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise CaseError(f"{role} must be a finite number, not {value!r}; e.g. {example}")
    return number


def _vector3(value: Any, role: str, example: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Iterable):
        raise CaseError(f"{role} must be three numbers, e.g. {example}")
    items = list(value)
    if len(items) != 3:
        raise CaseError(
            f"{role} must be three numbers (x, y, z), got {len(items)}; e.g. {example}"
        )
    x, y, z = (_real(v, role, example) for v in items)
    return (x, y, z)


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(float(c) * float(c) for c in v))


def _unit(v: Sequence[float], role: str, example: str) -> tuple[float, float, float]:
    length = _norm(v)
    if not length > 1e-12 or not math.isfinite(length):
        raise CaseError(f"{role} must not be the zero vector; e.g. {example}")
    return (float(v[0]) / length, float(v[1]) / length, float(v[2]) / length)


def _tag(value: Any, role: str) -> int:
    """One face tag. 1.0 is face 1; 1.7 is a typo, not face 1."""
    if isinstance(value, bool):
        raise CaseError(f"{role} must be face numbers like [1, 8], not {value!r}")
    if isinstance(value, numbers.Integral):
        tag = int(value)
    elif isinstance(value, numbers.Real):
        number = float(value)
        if not math.isfinite(number) or number != math.floor(number):
            raise CaseError(
                f"{role}: {value!r} is not a face number; face numbers are whole "
                "numbers from the face table, e.g. [1, 8]"
            )
        tag = int(number)
    else:
        raise CaseError(f"{role} must be face numbers like [1, 8], not {value!r}")
    if tag < 1:
        raise CaseError(
            f"{role}: {tag} is not a face number; faces are numbered from 1 in the "
            "face table"
        )
    return tag


def _face_tags(value: Any, role: str) -> tuple[int, ...]:
    """Sorted, de-duplicated face tags.

    De-duplication matters: a face listed twice must not count its area twice
    when a force is spread as a traction (the legacy parser doubled it).
    """
    if isinstance(value, Mapping) and set(value) == {"faces"}:
        value = value["faces"]
    if isinstance(value, (bool, numbers.Real)):
        value = [value]
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Iterable):
        raise CaseError(f"{role} must be a list of face numbers, e.g. [1, 8]")
    tags = sorted({_tag(v, role) for v in value})
    if not tags:
        raise CaseError(f"{role} must name at least one face, e.g. [1]")
    return tuple(tags)


def _ranges(tags: Iterable[int]) -> str:
    """1, 2, 3, 5 -> '1-3, 5': a refusal listing 400 faces must stay one line."""
    ordered = sorted(set(tags))
    if not ordered:
        return "none"
    parts: list[str] = []
    start = prev = ordered[0]
    for tag in ordered[1:] + [None]:  # type: ignore[list-item]
        if tag is not None and tag == prev + 1:
            prev = tag
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        if tag is not None:
            start = prev = tag
    return ", ".join(parts)


def _check_known(tags: Iterable[int], known: Collection[int], role: str) -> None:
    unknown = sorted(set(tags) - set(known))
    if unknown:
        noun = "face" if len(unknown) == 1 else "faces"
        raise CaseError(
            f"{role} names {noun} {', '.join(map(str, unknown))}, which this part "
            f"does not have; its faces are {_ranges(known)}. Use the numbers from "
            "the face table (`fea-structural inspect PART.step` prints it)."
        )


def _num(value: float, sig: int = 4) -> str:
    """Human number: 200000, 2129, 1.25, 0.3333 -- no '2.129e+03' in a summary."""
    value = float(value)
    if value == 0 or not math.isfinite(value):
        return "0" if value == 0 else str(value)
    magnitude = abs(value)
    if 1e-4 <= magnitude < 1e9:
        exponent = math.floor(math.log10(magnitude))
        decimals = max(0, sig - 1 - exponent)
        text = f"{value:.{decimals}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return "0" if text in ("-0", "") else text
    return f"{value:.{sig}g}"


def _vec(v: Sequence[float], sig: int = 4) -> str:
    """A vector for people: round-off a billion times smaller than its largest
    component (a mesh normal's -2.3e-18) prints as 0, not as noise."""
    values = [float(c) for c in v]
    scale = max((abs(c) for c in values if math.isfinite(c)), default=0.0)
    cleaned = [0.0 if scale and abs(c) < 1e-9 * scale else c for c in values]
    return "(" + ", ".join(_num(c, sig) for c in cleaned) + ")"


def _faces_phrase(tags: Sequence[int]) -> str:
    return ("face " if len(tags) == 1 else "faces ") + ", ".join(map(str, tags))


def _check_keys(value: Any, allowed: Collection[str], role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CaseError(f"{role} must be a JSON object with keys {sorted(allowed)}")
    unknown = sorted(str(k) for k in set(value) - set(allowed))
    if unknown:
        raise CaseError(
            f"unexpected {unknown} in {role}; it accepts {sorted(allowed)}"
        )
    return value


def _label(value: Any, default: str, role: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise CaseError(f"{role}.label must be a non-empty string, e.g. \"{default}\"")
    return value.strip()


# --------------------------------------------------------------------------
# Material
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Material:
    """Linear isotropic elastic material, Engineering-Data style.

    E drives deformation only: with fixed supports and force, pressure or
    gravity loads, stress does not depend on E. It is still required, because
    Total Deformation is a headline result and a defaulted E would make it
    silently wrong by the ratio of the moduli.
    """

    key: str
    name: str
    youngs_modulus_mpa: float
    poisson_ratio: float
    density_kg_m3: float | None = None
    tensile_yield_mpa: float | None = None
    tensile_ultimate_mpa: float | None = None
    source: str = "preset"

    def __post_init__(self) -> None:
        for attr in ("key", "name"):
            if not isinstance(getattr(self, attr), str) or not getattr(self, attr).strip():
                raise CaseError(f"material {attr} must be a non-empty string")
        if self.source not in ("preset", "custom"):
            raise CaseError(
                f"material source must be 'preset' or 'custom', not {self.source!r}"
            )
        e = _real(self.youngs_modulus_mpa, "Young's modulus (MPa)", "200000 for steel")
        if not e > 0:
            raise CaseError(
                f"Young's modulus must be positive, not {_num(e)} MPa; e.g. 200000 "
                "(MPa) for steel or 71000 for aluminium"
            )
        nu = _real(self.poisson_ratio, "Poisson's ratio", "0.3")
        if not -1.0 < nu < 0.5:
            raise CaseError(
                f"Poisson's ratio must lie between -1 and 0.5 (exclusive), not "
                f"{_num(nu)}; metals are about 0.3, rubber-like solids approach 0.5"
            )
        object.__setattr__(self, "youngs_modulus_mpa", e)
        object.__setattr__(self, "poisson_ratio", nu)
        for attr, role, example in (
            ("density_kg_m3", "density (kg/m^3)", "7850 for steel"),
            ("tensile_yield_mpa", "tensile yield strength (MPa)", "250"),
            ("tensile_ultimate_mpa", "tensile ultimate strength (MPa)", "460"),
        ):
            raw = getattr(self, attr)
            if raw is None:
                continue
            value = _real(raw, role, example)
            if not value > 0:
                raise CaseError(f"{role} must be positive when given; e.g. {example}")
            object.__setattr__(self, attr, value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "youngs_modulus_mpa": self.youngs_modulus_mpa,
            "poisson_ratio": self.poisson_ratio,
            "density_kg_m3": self.density_kg_m3,
            "tensile_yield_mpa": self.tensile_yield_mpa,
            "tensile_ultimate_mpa": self.tensile_ultimate_mpa,
            "source": self.source,
        }


def _preset(key, name, e, nu, rho, yield_mpa, uts_mpa) -> Material:  # noqa: ANN001
    return Material(key, name, e, nu, rho, yield_mpa, uts_mpa, "preset")


#: Typical room-temperature values, in the spirit of Ansys Engineering Data's
#: General Materials. They are "typical values", not a specification: a design
#: decision needs the supplier's data sheet. Confidence per preset is in
#: PRESET_CONFIDENCE.
MATERIAL_PRESETS: dict[str, Material] = {
    m.key: m
    for m in (
        _preset("structural_steel", "Structural Steel", 200000.0, 0.30, 7850.0, 250.0, 460.0),
        _preset("aluminum_alloy", "Aluminum Alloy", 71000.0, 0.33, 2770.0, 280.0, 310.0),
        _preset("stainless_steel", "Stainless Steel", 193000.0, 0.31, 7750.0, 207.0, 586.0),
        _preset("titanium_alloy", "Titanium Alloy", 96000.0, 0.36, 4620.0, 930.0, 1070.0),
        _preset("copper_alloy", "Copper Alloy", 110000.0, 0.34, 8300.0, 280.0, 430.0),
        _preset("gray_cast_iron", "Gray Cast Iron", 110000.0, 0.28, 7200.0, None, 240.0),
        _preset("magnesium_alloy", "Magnesium Alloy", 45000.0, 0.35, 1800.0, 193.0, None),
        _preset("polyethylene", "Polyethylene", 1100.0, 0.42, 950.0, 25.0, None),
    )
}

#: How sure we are that each preset matches the Ansys Engineering Data default:
#: "high" = checked against it; "medium"/"low" = typical handbook values.
PRESET_CONFIDENCE: dict[str, str] = {
    "structural_steel": "high",
    "aluminum_alloy": "high",
    "stainless_steel": "medium",
    "titanium_alloy": "medium",
    "copper_alloy": "medium",
    "gray_cast_iron": "medium",
    "magnesium_alloy": "medium",
    "polyethylene": "low",
}

MATERIAL_ALIASES: dict[str, str] = {
    "steel": "structural_steel",
    "structural_steel": "structural_steel",
    "mild_steel": "structural_steel",
    "aluminium": "aluminum_alloy",
    "aluminum": "aluminum_alloy",
    "aluminium_alloy": "aluminum_alloy",
    "al": "aluminum_alloy",
    "stainless": "stainless_steel",
    "ss": "stainless_steel",
    "titanium": "titanium_alloy",
    "ti": "titanium_alloy",
    "copper": "copper_alloy",
    "cu": "copper_alloy",
    "cast_iron": "gray_cast_iron",
    "grey_cast_iron": "gray_cast_iron",
    "gray_iron": "gray_cast_iron",
    "grey_iron": "gray_cast_iron",
    "iron": "gray_cast_iron",
    "magnesium": "magnesium_alloy",
    "mg": "magnesium_alloy",
    "pe": "polyethylene",
    "polyethylene": "polyethylene",
}


def _material_token(text: str) -> str:
    return re.sub(r"[\s\-_]+", "_", text.strip().lower()).strip("_")


def material_from_key(text: str) -> Material:
    """A preset by key, alias or display name: 'Structural Steel', 'steel', 'al'.

    Case, spaces and hyphens do not matter. Anything else is refused with the
    list of keys rather than matched fuzzily -- 'stainles' guessed as stainless
    is harmless, 'Ti' guessed as tin would not be.
    """
    if not isinstance(text, str) or not text.strip():
        raise CaseError(_unknown_material_message(text))
    token = _material_token(text)
    if token in MATERIAL_PRESETS:
        return MATERIAL_PRESETS[token]
    if token in MATERIAL_ALIASES:
        return MATERIAL_PRESETS[MATERIAL_ALIASES[token]]
    for material in MATERIAL_PRESETS.values():
        if _material_token(material.name) == token:
            return material
    raise CaseError(_unknown_material_message(text))


def _unknown_material_message(text: Any) -> str:
    return (
        f"no material preset {text!r}; presets are {', '.join(MATERIAL_PRESETS)} "
        "(also: steel, aluminium, stainless, titanium, copper, cast_iron, "
        "magnesium, pe). For anything else give a custom material: Young's "
        "modulus in MPa and Poisson's ratio, e.g. youngs_modulus_mpa 70000, "
        "poisson_ratio 0.33"
    )


def custom_material(
    youngs_modulus_mpa: float,
    poisson_ratio: float,
    *,
    name: str = "Custom material",
    density_kg_m3: float | None = None,
    tensile_yield_mpa: float | None = None,
    tensile_ultimate_mpa: float | None = None,
) -> Material:
    """A user-defined material. Key 'custom' so it round-trips through a case file."""
    return Material(
        key="custom",
        name=name,
        youngs_modulus_mpa=youngs_modulus_mpa,
        poisson_ratio=poisson_ratio,
        density_kg_m3=density_kg_m3,
        tensile_yield_mpa=tensile_yield_mpa,
        tensile_ultimate_mpa=tensile_ultimate_mpa,
        source="custom",
    )


# --------------------------------------------------------------------------
# Supports, loads, gravity, mesh
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedSupport:
    """Ansys 'Fixed Support': all three translations zero on every node of the faces."""

    faces: tuple[int, ...]
    label: str = "Fixed Support"

    def __post_init__(self) -> None:
        object.__setattr__(self, "faces", _face_tags(self.faces, f"{self.label} faces"))
        object.__setattr__(self, "label", _label(self.label, "Fixed Support", "support"))


@dataclass(frozen=True)
class ForceLoad:
    """Ansys 'Force' on faces: a total force in N spread as a uniform traction
    over the faces' total area. The vector carries the sign, so the magnitude is
    always positive."""

    faces: tuple[int, ...]
    vector_n: tuple[float, float, float]
    label: str = "Force"

    def __post_init__(self) -> None:
        object.__setattr__(self, "faces", _face_tags(self.faces, f"{self.label} faces"))
        vector = _vector3(self.vector_n, f"{self.label} vector_n (N)", "[0, 0, -500]")
        if not _norm(vector) > 0:
            raise CaseError(
                f"{self.label} must be a non-zero force; give components in N, "
                "e.g. [0, 0, -500], or a magnitude and a direction, e.g. 500 N -z"
            )
        object.__setattr__(self, "vector_n", vector)
        object.__setattr__(self, "label", _label(self.label, "Force", "load"))

    @property
    def magnitude_n(self) -> float:
        return _norm(self.vector_n)

    @property
    def direction(self) -> tuple[float, float, float]:
        m = self.magnitude_n
        return (self.vector_n[0] / m, self.vector_n[1] / m, self.vector_n[2] / m)


@dataclass(frozen=True)
class PressureLoad:
    """Ansys 'Pressure': acts normal to the surface at every point. Positive
    pushes INTO the surface; negative is suction. On a curved face (a bore) the
    net force is far smaller than p*A, which is why this is not a force."""

    faces: tuple[int, ...]
    pressure_mpa: float
    label: str = "Pressure"

    def __post_init__(self) -> None:
        object.__setattr__(self, "faces", _face_tags(self.faces, f"{self.label} faces"))
        p = _real(self.pressure_mpa, f"{self.label} pressure (MPa)", "1.5")
        if p == 0:
            raise CaseError(
                f"{self.label} must not be zero; give MPa, positive pushes into "
                "the surface (e.g. 1.5), negative pulls (suction, e.g. -0.1)"
            )
        object.__setattr__(self, "pressure_mpa", p)
        object.__setattr__(self, "label", _label(self.label, "Pressure", "load"))


@dataclass(frozen=True)
class Gravity:
    """Self-weight: body force rho*g over the whole volume. Needs the density."""

    acceleration_mps2: tuple[float, float, float] = (0.0, 0.0, -STANDARD_GRAVITY)
    label: str = "Standard Earth Gravity"

    def __post_init__(self) -> None:
        vector = _vector3(
            self.acceleration_mps2, "gravity acceleration_mps2", "[0, 0, -9.80665]"
        )
        if not _norm(vector) > 0:
            raise CaseError(
                "gravity acceleration must not be zero; e.g. [0, 0, -9.80665] for "
                "standard gravity along -Z, or disable gravity"
            )
        object.__setattr__(self, "acceleration_mps2", vector)
        object.__setattr__(self, "label", _label(self.label, "Standard Earth Gravity", "gravity"))

    @property
    def magnitude_mps2(self) -> float:
        return _norm(self.acceleration_mps2)

    @property
    def direction(self) -> tuple[float, float, float]:
        m = self.magnitude_mps2
        a = self.acceleration_mps2
        return (a[0] / m, a[1] / m, a[2] / m)


@dataclass(frozen=True)
class MeshSettings:
    """element_order 2 = 10-node tets (default, SOLID187-like); 1 = 4-node tets,
    which are far too stiff in bending and exist for comparison only.
    element_size_mm, when given, overrides the refinement level."""

    element_order: int = 2
    refinement: str = "medium"
    element_size_mm: float | None = None

    def __post_init__(self) -> None:
        order = self.element_order
        if isinstance(order, bool) or not isinstance(order, numbers.Real) or order not in ELEMENT_ORDERS:
            raise CaseError(
                f"mesh element_order must be 2 (10-node tetrahedra, default) or 1 "
                f"(4-node, comparison only), not {order!r}"
            )
        object.__setattr__(self, "element_order", int(order))
        refinement = self.refinement.strip().lower() if isinstance(self.refinement, str) else None
        if refinement not in REFINEMENT_LEVELS:
            raise CaseError(
                f"mesh refinement must be one of {', '.join(REFINEMENT_LEVELS)}, not "
                f"{self.refinement!r}; or give element_size_mm"
            )
        object.__setattr__(self, "refinement", refinement)
        if self.element_size_mm is not None:
            size = _real(self.element_size_mm, "mesh element_size_mm", "2.5")
            if not size > 0:
                raise CaseError(
                    "mesh element_size_mm must be positive, e.g. 2.5, or null to use "
                    "the refinement level"
                )
            object.__setattr__(self, "element_size_mm", size)

    def describe(self) -> str:
        element = (
            "10-node quadratic tetrahedra (SOLID187-like)"
            if self.element_order == 2
            else "4-node linear tetrahedra (stiff in bending; comparison only)"
        )
        size = (
            f"element size {_num(self.element_size_mm)} mm"
            if self.element_size_mm is not None
            else f"{self.refinement} refinement"
        )
        return f"{element}, {size}"


Load = Union[ForceLoad, PressureLoad]


def _letter(index: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA: Ansys letters the environment tree this way."""
    letters = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(ord("A") + rem) + letters
    return letters


@dataclass(frozen=True)
class StaticStructuralCase:
    """A complete, solvable static structural setup.

    Validated on construction, so the CLI building one from prompts and a JSON
    file parsed by `parse_case` are held to the same rules.
    """

    supports: tuple[FixedSupport, ...]
    loads: tuple[Load, ...]
    material: Material
    gravity: Gravity | None = None
    mesh: MeshSettings = field(default_factory=MeshSettings)

    def __post_init__(self) -> None:
        # A single support or load passed bare is accepted as a one-item tuple.
        supports = (
            (self.supports,) if isinstance(self.supports, FixedSupport)
            else tuple(self.supports) if isinstance(self.supports, Iterable) else ()
        )
        loads = (
            (self.loads,) if isinstance(self.loads, (ForceLoad, PressureLoad))
            else tuple(self.loads) if isinstance(self.loads, Iterable) else ()
        )
        if not supports:
            raise CaseError(
                "the case needs at least one fixed support: without one the part "
                "is free to move as a rigid body and has no static solution. Give "
                "the faces that are held, e.g. supports [{\"type\": \"fixed\", "
                "\"faces\": [1]}] or --fixed 1"
            )
        for support in supports:
            if not isinstance(support, FixedSupport):
                raise CaseError(
                    f"supports must be FixedSupport, not {type(support).__name__}; "
                    "only fixed supports are available"
                )
        for load in loads:
            if not isinstance(load, (ForceLoad, PressureLoad)):
                raise CaseError(
                    f"loads must be ForceLoad or PressureLoad, not {type(load).__name__}"
                )
        if not isinstance(self.material, Material):
            raise CaseError("the case needs a Material; use material_from_key or custom_material")
        if self.gravity is not None and not isinstance(self.gravity, Gravity):
            raise CaseError("gravity must be a Gravity or None")
        if not isinstance(self.mesh, MeshSettings):
            raise CaseError("mesh must be MeshSettings")
        object.__setattr__(self, "supports", supports)
        object.__setattr__(self, "loads", loads)

        if not loads and self.gravity is None:
            raise CaseError(
                "the case has no load: add a force or a pressure (e.g. --force "
                "6:0,0,-500 or {\"type\": \"force\", \"faces\": [6], \"vector_n\": "
                "[0, 0, -500]}), or enable gravity"
            )
        if self.gravity is not None and self.material.density_kg_m3 is None:
            raise CaseError(
                "gravity needs the material's density: give density_kg_m3 (e.g. "
                "7850 for steel) or pick a preset material, or disable gravity"
            )
        fixed = set(self.fixed_faces())
        for letter, item in self.lettered():
            if isinstance(item, (ForceLoad, PressureLoad)) and set(item.faces) <= fixed:
                raise CaseError(
                    f"load {letter} ({item.label} on {_faces_phrase(item.faces)}) acts "
                    "only on fixed faces, so it would go straight into the support "
                    "and stress nothing. Load a face that is free to move, or fix a "
                    "different face."
                )

    # ---- queries -----------------------------------------------------------

    def fixed_faces(self) -> tuple[int, ...]:
        return tuple(sorted({t for s in self.supports for t in s.faces}))

    def loaded_faces(self) -> tuple[int, ...]:
        return tuple(sorted({t for load in self.loads for t in load.faces}))

    def lettered(self) -> list[tuple[str, FixedSupport | ForceLoad | PressureLoad | Gravity]]:
        """Supports, then loads, then gravity, lettered A, B, C... One source
        for the letters so the confirmation box and the setup picture agree."""
        items: list[Any] = [*self.supports, *self.loads]
        if self.gravity is not None:
            items.append(self.gravity)
        return [(_letter(i), item) for i, item in enumerate(items)]

    def check_faces(self, faces: Sequence[FaceInfo]) -> None:
        """Refuse any face tag the part does not have, listing the valid ones."""
        known = {int(f.tag) for f in faces}
        for letter, item in self.lettered():
            if isinstance(item, Gravity):
                continue
            _check_known(item.faces, known, f"{letter} ({item.label})")

    # ---- documents -----------------------------------------------------------

    def as_document(self) -> dict[str, Any]:
        """StaticStructuralCase/v1, round-trippable through `parse_case`."""
        loads: list[dict[str, Any]] = []
        for load in self.loads:
            if isinstance(load, ForceLoad):
                loads.append({
                    "type": "force", "label": load.label, "faces": list(load.faces),
                    "vector_n": list(load.vector_n),
                })
            else:
                loads.append({
                    "type": "pressure", "label": load.label, "faces": list(load.faces),
                    "pressure_mpa": load.pressure_mpa,
                })
        m = self.material
        material: dict[str, Any] = {}
        if MATERIAL_PRESETS.get(m.key) == m:
            material["preset"] = m.key
        material.update({
            "name": m.name,
            "youngs_modulus_mpa": m.youngs_modulus_mpa,
            "poisson_ratio": m.poisson_ratio,
            "density_kg_m3": m.density_kg_m3,
            "tensile_yield_mpa": m.tensile_yield_mpa,
            "tensile_ultimate_mpa": m.tensile_ultimate_mpa,
        })
        gravity: dict[str, Any] = {"enabled": self.gravity is not None}
        if self.gravity is not None:
            gravity["acceleration_mps2"] = list(self.gravity.acceleration_mps2)
            gravity["label"] = self.gravity.label
        return {
            "schema_version": CASE_SCHEMA,
            "analysis": STATIC_STRUCTURAL,
            "supports": [
                {"type": "fixed", "label": s.label, "faces": list(s.faces)}
                for s in self.supports
            ],
            "loads": loads,
            "material": material,
            "gravity": gravity,
            "mesh": {
                "element_order": self.mesh.element_order,
                "refinement": self.mesh.refinement,
                "element_size_mm": self.mesh.element_size_mm,
            },
        }

    def assumptions(self) -> list[str]:
        """What a reader must know so as not to mistake the idealisation for the part."""
        m = self.material
        out = [
            "E affects deformation only: with fixed supports and force, pressure or "
            "gravity loads, stress does not depend on E, and deformation scales as 1/E",
        ]
        if m.source == "preset":
            confidence = PRESET_CONFIDENCE.get(m.key, "unknown")
            out.append(
                f"material '{m.name}' uses typical values (confidence {confidence}); "
                "use the supplier's data sheet for a design decision"
            )
        else:
            out.append(f"material '{m.name}' values were given by the user")
        if m.tensile_yield_mpa is None:
            if m.tensile_ultimate_mpa is not None:
                out.append(
                    f"'{m.name}' has no tensile yield (brittle): von Mises stress is "
                    "not its failure measure, so no yield safety factor is given; "
                    "compare maximum principal stress with the ultimate strength"
                )
            else:
                out.append("no tensile yield given, so no yield safety factor is reported")
        if m.poisson_ratio > 0.45:
            out.append(
                f"Poisson's ratio {_num(m.poisson_ratio)} is close to 0.5 (nearly "
                "incompressible); 4-node tetrahedra lock there, so keep element order 2"
            )
        if self.mesh.element_order == 1:
            out.append(
                "4-node tetrahedra are much too stiff in bending: deformation and "
                "stress are under-predicted; use element order 2 for results"
            )
        fixed = set(self.fixed_faces())
        for letter, item in self.lettered():
            if isinstance(item, ForceLoad):
                out.append(
                    f"{letter} {item.label}: {_num(item.magnitude_n)} N spread as a "
                    f"uniform traction over the total area of {_faces_phrase(item.faces)}"
                )
            elif isinstance(item, PressureLoad):
                sense = "pushes into" if item.pressure_mpa > 0 else "pulls out of (suction)"
                out.append(
                    f"{letter} {item.label}: {_num(item.pressure_mpa)} MPa acts normal to "
                    f"the surface at every point and {sense} {_faces_phrase(item.faces)}"
                )
            elif isinstance(item, Gravity):
                out.append(
                    f"{letter} {item.label}: {_num(item.magnitude_mps2, 5)} m/s² along "
                    f"{_vec(item.direction)} applied as self-weight from density "
                    f"{_num(m.density_kg_m3 or 0)} kg/m³; check that this direction "
                    "is 'down' for the way the part was modelled"
                )
            if isinstance(item, (ForceLoad, PressureLoad)):
                shared = sorted(set(item.faces) & fixed)
                if shared:
                    out.append(
                        f"{letter} {item.label}: {_faces_phrase(shared)} also fixed; the "
                        "share of the load on them goes straight into the support"
                    )
        return out

    def describe(self, faces: Sequence[FaceInfo] | None = None) -> list[str]:
        """Lines for the confirmation box, lettered like the Ansys environment tree.

        With `faces` the lines carry areas and the traction or equivalent force,
        which is what lets a person spot a load on the wrong face before solving.
        """
        by_tag = {int(f.tag): f for f in faces} if faces else {}

        def area(tags: Sequence[int]) -> float | None:
            if not by_tag or any(t not in by_tag for t in tags):
                return None
            return float(sum(float(by_tag[t].area) for t in tags))

        def title(item: Any, kind: str) -> str:
            return item.label if item.label == kind else f"{item.label} ({kind})"

        lines: list[str] = []
        for letter, item in self.lettered():
            if isinstance(item, FixedSupport):
                a = area(item.faces)
                where = _faces_phrase(item.faces) + (f" ({_num(a)} mm²)" if a else "")
                lines.append(f"{letter}  {title(item, 'Fixed Support')} — {where}")
            elif isinstance(item, ForceLoad):
                a = area(item.faces)
                text = (
                    f"{letter}  {title(item, 'Force')} — {_num(item.magnitude_n)} N along "
                    f"{_vec(item.direction)} on {_faces_phrase(item.faces)}"
                )
                if a:
                    text += f" ({_num(a)} mm², traction {_num(item.magnitude_n / a)} MPa)"
                lines.append(text)
            elif isinstance(item, PressureLoad):
                sense = "into the surface" if item.pressure_mpa > 0 else "suction, out of the surface"
                text = (
                    f"{letter}  {title(item, 'Pressure')} — {_num(item.pressure_mpa)} MPa "
                    f"normal to {_faces_phrase(item.faces)} ({sense})"
                )
                a = area(item.faces)
                if a:
                    normal = _common_normal([by_tag[t] for t in item.faces])
                    if normal is not None:
                        sign = 1.0 if item.pressure_mpa > 0 else -1.0
                        resultant = tuple(-sign * c for c in normal)
                        text += (
                            f" ≈ {_num(abs(item.pressure_mpa) * a)} N along "
                            f"{_vec(resultant)} on {_num(a)} mm²"
                        )
                    else:
                        text += f" on {_num(a)} mm² (curved or non-parallel faces)"
                lines.append(text)
            elif isinstance(item, Gravity):
                lines.append(
                    f"{letter}  {item.label} — {_num(item.magnitude_mps2, 5)} m/s² along "
                    f"{_vec(item.direction)}"
                )
        m = self.material
        props = [f"E {_num(m.youngs_modulus_mpa)} MPa", f"ν {_num(m.poisson_ratio)}"]
        if m.density_kg_m3 is not None:
            props.append(f"ρ {_num(m.density_kg_m3)} kg/m³")
        if m.tensile_yield_mpa is not None:
            props.append(f"yield {_num(m.tensile_yield_mpa)} MPa")
        if m.tensile_ultimate_mpa is not None:
            props.append(f"ultimate {_num(m.tensile_ultimate_mpa)} MPa")
        origin = " (typical values)" if m.source == "preset" else " (custom)"
        lines.append(f"Material: {m.name}{origin} — {', '.join(props)}")
        lines.append(f"Mesh: {self.mesh.describe()}")
        return lines


# --------------------------------------------------------------------------
# Terminal grammar
# --------------------------------------------------------------------------

_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_QUANTITY = re.compile(rf"^\s*({_NUMBER})\s*(.*?)\s*$")

_FORCE_UNITS = {"n": 1.0, "kn": 1e3, "lbf": 4.4482216152605, "kip": 4448.2216152605,
                "kips": 4448.2216152605}
_PRESSURE_UNITS = {"mpa": 1.0, "n/mm2": 1.0, "kpa": 1e-3, "pa": 1e-6, "gpa": 1e3,
                   "bar": 0.1, "psi": 0.006894757293168361, "ksi": 6.894757293168361}
_MODULUS_UNITS = {"mpa": 1.0, "n/mm2": 1.0, "gpa": 1e3, "pa": 1e-6, "kpa": 1e-3,
                  "psi": 0.006894757293168361, "ksi": 6.894757293168361,
                  "msi": 6894.757293168361}
_DENSITY_UNITS = {"kg/m3": 1.0, "g/cm3": 1000.0, "g/cc": 1000.0, "t/m3": 1000.0}

_INWARD_WORDS = frozenset({"normal", "in", "into", "inward", "inwards", "push"})
_OUTWARD_WORDS = frozenset({"out", "outward", "outwards", "pull"})
_AXIS = re.compile(r"^([+-]?)\s*([xyz])$")
_CONNECTORS = ("along", "towards", "toward", "direction", "dir", "in", "to", "@")


def _unit_key(text: str) -> str:
    return (
        text.strip().lower().replace(" ", "").replace("^", "")
        .replace("²", "2").replace("³", "3")
    )


#: '70,000' or '70 000 MPa': a number written with thousands separators.
_GROUPED = re.compile(r"^\s*[+-]?\d{1,3}(?:[, ]\d{3})+(?:\.\d*)?\s*([^\d,\s].*)?$")
_NOT_FINITE = re.compile(r"^[+-]?(?:nan|inf(?:inity)?)$", re.IGNORECASE)


def _quantity(text: Any, units: Mapping[str, float], role: str, example: str,
              exact: Mapping[str, float] | None = None, names: str = "") -> float:
    if not isinstance(text, str) or not text.strip():
        raise CaseError(f"{role}: type a number, e.g. {example}")
    if _GROUPED.match(text):
        plain = re.sub(r"(?<=\d)[, ](?=\d{3})", "", text.strip())
        raise CaseError(
            f"{role}: write {text.strip()!r} without thousands separators, e.g. {plain}"
        )
    if _NOT_FINITE.match(text.strip().split()[0]):
        raise CaseError(f"{role}: {text.strip()!r} is not a finite number; e.g. {example}")
    match = _QUANTITY.match(text)
    if not match:
        raise CaseError(f"{role}: {text.strip()!r} is not a number; e.g. {example}")
    value = float(match.group(1))
    unit = match.group(2)
    if unit:
        if exact and unit.strip() in exact:
            factor = exact[unit.strip()]
        else:
            key = _unit_key(unit)
            if key not in units:
                raise CaseError(
                    f"{role}: unknown unit {unit.strip()!r}; use one of "
                    f"{names or ', '.join(sorted(units))}, e.g. {example}"
                )
            factor = units[key]
        value *= factor
    if not math.isfinite(value):
        raise CaseError(f"{role}: {text.strip()!r} is not a finite number; e.g. {example}")
    return value


def parse_force(text: str) -> float:
    """Force in N from '500', '500 N', '5 kN', '2 MN', '100 lbf'. Bare number = N.

    Must be positive: the sign of a force belongs to its direction ('500 N -z').
    'mN' is refused rather than read as meganewtons.
    """
    if isinstance(text, str) and re.search(r"\dmN\s*$|\smN\s*$", text):
        raise CaseError(
            "force: 'mN' (millinewton) is not accepted; give N or kN, or MN for "
            "meganewtons, e.g. 500 N"
        )
    value = _quantity(text, _FORCE_UNITS, "force", "500 N or 5 kN", exact={"MN": 1e6},
                      names="N, kN, MN, lbf, kip")
    if not value > 0:
        raise CaseError(
            "force: give a positive magnitude and put the sign in the direction, "
            "e.g. 500 N with direction -z"
        )
    return value


def parse_pressure(text: str, *, role: str = "pressure",
                   example: str = "1.5 MPa or 200 kPa") -> float:
    """Pressure in MPa (= N/mm²). Bare number = MPa; also kPa, Pa, GPa, bar, psi.

    Positive pushes into the surface, negative is suction; zero is refused.
    `role` names the quantity in errors when the same units mean something
    else, e.g. a tensile yield.
    """
    value = _quantity(text, _PRESSURE_UNITS, role, example,
                      names="MPa, N/mm2, kPa, Pa, GPa, bar, psi, ksi")
    if value == 0:
        if role != "pressure":
            raise CaseError(f"{role} must not be zero; e.g. {example}")
        raise CaseError(
            "pressure must not be zero; positive pushes into the face (e.g. 1.5 MPa), "
            "negative is suction (e.g. -0.1 MPa)"
        )
    return value


def parse_modulus(text: str) -> float:
    """Young's modulus in MPa. Bare number = MPa (steel is 200000, not 200).

    '200 GPa', '29e6 psi', '29000 ksi' also work. Must be positive.
    """
    value = _quantity(text, _MODULUS_UNITS, "Young's modulus", "200 GPa or 200000 (MPa)",
                      names="MPa, GPa, N/mm2, kPa, Pa, psi, ksi, Msi")
    if not value > 0:
        raise CaseError("Young's modulus must be positive, e.g. 200 GPa or 200000 (MPa)")
    return value


def parse_density(text: str) -> float:
    """Density in kg/m³. Bare number = kg/m³; '7.85 g/cm3' also works. Positive."""
    value = _quantity(text, _DENSITY_UNITS, "density", "7850 (kg/m³) or 7.85 g/cm3",
                      names="kg/m3, g/cm3, g/cc, t/m3")
    if not value > 0:
        raise CaseError("density must be positive, e.g. 7850 (kg/m³) or 7.85 g/cm3")
    return value


def parse_vector(text: str) -> tuple[float, float, float]:
    """Three numbers: '0,0,-500', '0 0 -500', '[0,0,-500]', '(0 0 -500)'."""
    example = "0, 0, -500"
    if not isinstance(text, str) or not text.strip():
        raise CaseError(f"type three numbers x, y, z, e.g. {example}")
    body = text.strip()
    pairs = {"[": "]", "(": ")", "{": "}"}
    if body[0] in pairs:
        if not body.endswith(pairs[body[0]]):
            raise CaseError(f"unbalanced brackets in {text.strip()!r}; e.g. [{example}]")
        body = body[1:-1]
    parts = [p for p in re.split(r"[,;\s]+", body.strip()) if p]
    if len(parts) != 3:
        raise CaseError(
            f"{text.strip()!r} is not three numbers x, y, z; e.g. {example}"
        )
    values: list[float] = []
    for part in parts:
        if _NOT_FINITE.match(part):
            raise CaseError(f"{part!r} in {text.strip()!r} is not a finite number; e.g. {example}")
        if not re.fullmatch(_NUMBER, part):
            raise CaseError(f"{part!r} in {text.strip()!r} is not a number; e.g. {example}")
        number = float(part)
        if not math.isfinite(number):
            raise CaseError(f"{part!r} is not a finite number; e.g. {example}")
        values.append(number)
    return (values[0], values[1], values[2])


def _common_normal(faces: Sequence[FaceInfo]) -> tuple[float, float, float] | None:
    """The shared outward unit normal of planar faces that agree, else None."""
    if not faces or not all(bool(f.planar) for f in faces):
        return None
    normals = []
    for f in faces:
        n = [float(c) for c in f.normal]
        length = _norm(n)
        if not length > 0 or not math.isfinite(length):
            return None
        normals.append([c / length for c in n])
    first = normals[0]
    if any(sum(a * b for a, b in zip(first, n)) <= NORMALS_AGREE_COS for n in normals[1:]):
        return None
    total = [sum(n[i] for n in normals) for i in range(3)]
    length = _norm(total)
    # A meshed flat face's normal carries round-off (5e-18 beside -1): drop it,
    # or a saved case stores a force of (5.3e-15, 8.6e-16, -500) N.
    unit = [c / length for c in total]
    unit = [0.0 if abs(c) < 1e-9 else c for c in unit]
    length = _norm(unit)
    return (unit[0] / length, unit[1] / length, unit[2] / length)


def parse_direction(
    text: str, faces: Sequence[FaceInfo] = ()
) -> tuple[tuple[float, float, float], str]:
    """A load direction from the terminal, as (unit vector, source).

    'x' '+x' '-x' ... '-z' -> source 'axis'; three numbers -> 'vector';
    'normal' 'in' 'into' 'inward' 'push' -> minus the loaded faces' outward
    normal ('normal_inward'); 'out' 'outward' 'pull' -> plus it
    ('normal_outward'). The normal forms need `faces` (the loaded faces), all
    planar with agreeing normals: a curved face has no single normal, and the
    load a person means there is a pressure.
    """
    if not isinstance(text, str) or not text.strip():
        raise CaseError("type a direction: -z, x, normal, out, or three numbers like 0,0,-1")
    token = text.strip().lower()
    axis = _AXIS.match(token)
    if axis:
        index = "xyz".index(axis.group(2))
        vector = [0.0, 0.0, 0.0]
        vector[index] = -1.0 if axis.group(1) == "-" else 1.0
        return (vector[0], vector[1], vector[2]), "axis"
    if token in _INWARD_WORDS or token in _OUTWARD_WORDS:
        inward = token in _INWARD_WORDS
        if not faces:
            raise CaseError(
                f"direction {token!r} needs the loaded faces to find their normal; "
                "give an axis like -z or three numbers like 0,0,-1 instead"
            )
        normal = _common_normal(faces)
        if normal is None:
            tags = ", ".join(str(int(f.tag)) for f in faces)
            raise CaseError(
                f"direction {token!r} needs flat faces that all face the same way, "
                f"and face(s) {tags} do not (curved, or normals differ). For a load "
                "that pushes normal to a curved face, use a pressure load instead; "
                "otherwise give an axis like -z or three numbers"
            )
        sign = -1.0 if inward else 1.0
        vector3 = (sign * normal[0], sign * normal[1], sign * normal[2])
        return vector3, "normal_inward" if inward else "normal_outward"
    try:
        vector3 = parse_vector(text)
    except CaseError:
        raise CaseError(
            f"{text.strip()!r} is not a direction; type an axis (x, -y, +z), normal "
            "(push into the face), out (pull), or three numbers like 0,0,-1"
        ) from None
    return _unit(vector3, "direction", "0,0,-1"), "vector"


def parse_force_spec(
    text: str, faces: Sequence[FaceInfo] = ()
) -> tuple[tuple[float, float, float], str]:
    """A whole force from one line, as (vector in N, direction source).

    Ansys offers 'Define By: Vector' (magnitude + direction) and 'Components';
    both are one line here: '500 N -z', '5 kN along normal', '500 x' or
    components '0,0,-500' / '[0 0 -0.5] kN'. Components carry their own sign.
    """
    if not isinstance(text, str) or not text.strip():
        raise CaseError("type a force, e.g. 500 N -z (magnitude and direction) or 0,0,-500 (components in N)")
    body = text.strip()
    # Near misses, said plainly: a bare magnitude, two components, a NaN.
    numbers = [p for p in re.split(r"[,;\s\[\]()]+", body) if p]
    if numbers and all(_NOT_FINITE.match(p) or re.fullmatch(_NUMBER, p) for p in numbers) \
            and any(_NOT_FINITE.match(p) for p in numbers):
        raise CaseError(f"{body!r} is not a finite force; give numbers, e.g. 0,0,-500 or 500 N -z")
    if len(numbers) == 2 and all(re.fullmatch(_NUMBER, p) for p in numbers) and \
            re.search(r"[,;]", body):
        raise CaseError(
            f"{body!r} has two components; give three, x, y, z in N, e.g. 0,0,-500 "
            "(or a magnitude and a direction, e.g. 500 N -z)"
        )
    bare = re.fullmatch(rf"\s*({_NUMBER})\s*(kn|n|lbf|kips?|MN)?\s*", body, re.IGNORECASE)
    if bare:
        raise CaseError(
            f"{body!r} has a size but no direction; add one, e.g. {body} -z, "
            f"{body} normal (into a flat face), or give components like 0,0,-500"
        )
    # Components, optionally followed by a force unit.
    unit_match = re.match(r"^(.*?)\s*(kn|n|lbf|kips?|MN)\s*$", body, re.IGNORECASE)
    candidates = [(body, 1.0)]
    if unit_match and unit_match.group(1):
        unit = unit_match.group(2)
        factor = 1e6 if unit == "MN" else _FORCE_UNITS.get(unit.lower())
        if factor is not None:
            candidates.insert(0, (unit_match.group(1), factor))
    for numbers_text, factor in candidates:
        try:
            components = parse_vector(numbers_text)
        except CaseError:
            continue
        vector = tuple(c * factor for c in components)
        if not _norm(vector) > 0:
            raise CaseError("force components must not all be zero; e.g. 0,0,-500")
        return (vector[0], vector[1], vector[2]), "components"
    # Magnitude, optional unit, then a direction.
    match = re.match(rf"^\s*({_NUMBER})\s*(kn|n|lbf|kips?|MN)?\b\s*(.*)$", body, re.IGNORECASE)
    if not match or not match.group(3).strip():
        raise CaseError(
            f"{body!r} is not a force; type a magnitude and a direction, e.g. 500 N -z "
            "or 2 kN normal, or components in N, e.g. 0,0,-500"
        )
    magnitude = parse_force(match.group(1) + (match.group(2) or ""))
    rest = match.group(3).strip()
    try:
        direction, source = parse_direction(rest, faces)
    except CaseError as first_error:
        words = rest.split(None, 1)
        if len(words) == 2 and words[0].lower() in _CONNECTORS:
            direction, source = parse_direction(words[1], faces)
        else:
            raise first_error
    return (direction[0] * magnitude, direction[1] * magnitude, direction[2] * magnitude), source


def parse_face_list(text: str, known: Collection[int]) -> tuple[int, ...]:
    """Face numbers from '1', '1,8', '1 8', '2-4', 'faces 1 and 8'. Sorted, unique.

    Every number is checked against `known` (the part's face tags) when it is
    non-empty, so a typo is caught at the prompt rather than inside the solver.
    """
    if not isinstance(text, str) or not text.strip():
        raise CaseError("type face numbers from the face table, e.g. 1 or 1,8 or 2-4")
    body = text.strip().lower()
    body = re.sub(r"^faces?\b[:\s]*", "", body)
    body = body.replace("–", "-").replace("—", "-").replace("..", "-")
    body = re.sub(r"\s*-\s*", "-", body)
    tokens = [t for t in re.split(r"[,;\s]+|\band\b", body) if t and t != "and"]
    if not tokens:
        raise CaseError("type face numbers from the face table, e.g. 1 or 1,8 or 2-4")
    tags: set[int] = set()
    for token in tokens:
        span = re.fullmatch(r"(\d+)\s*-\s*(\d+)", token)
        if span:
            low, high = sorted((int(span.group(1)), int(span.group(2))))
            if high - low > 100000:
                raise CaseError(f"face range {token!r} is too long; check the numbers")
            tags.update(range(low, high + 1))
        elif re.fullmatch(r"\d+", token):
            tags.add(int(token))
        else:
            raise CaseError(
                f"{token!r} is not a face number; type numbers from the face table, "
                "e.g. 1 or 1,8 or 2-4"
            )
    if 0 in tags:
        raise CaseError("faces are numbered from 1; type numbers from the face table")
    if known:
        _check_known(tags, known, "that list")
    return tuple(sorted(tags))


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------

_TOP_KEYS = frozenset({"schema_version", "analysis", "supports", "loads", "material",
                       "gravity", "mesh"})
_SUPPORT_KEYS = frozenset({"type", "faces", "label"})
_FORCE_KEYS = frozenset({"type", "faces", "label", "vector_n", "magnitude_n", "direction"})
_PRESSURE_KEYS = frozenset({"type", "faces", "label", "pressure_mpa"})
_MATERIAL_KEYS = frozenset({"preset", "name", "youngs_modulus_mpa", "poisson_ratio",
                            "density_kg_m3", "tensile_yield_mpa", "tensile_ultimate_mpa"})
_GRAVITY_KEYS = frozenset({"enabled", "acceleration_mps2", "label"})
_MESH_KEYS = frozenset({"element_order", "refinement", "element_size_mm"})

_LEGACY_TOP_KEYS = frozenset({"fixed", "load", "loads", "material", "gravity", "analysis"})
_LEGACY_LOAD_KEYS = frozenset({"faces", "direction", "magnitude_n", "pressure_mpa"})
_LEGACY_GRAVITY_KEYS = frozenset({"enabled", "g"})

#: Named so a refusal can say what the analysis would need rather than "no".
_OTHER_ANALYSES = {
    "thermal": "a temperature field and expansion coefficients",
    "contact": "surfaces that come into and out of contact",
    "nonlinear": "plasticity or large displacement",
    "dynamic": "time and inertia",
    "transient": "time and inertia",
    "modal": "an eigenvalue solver for natural frequencies",
    "buckling": "a stability eigenvalue rather than a stress field",
    "fatigue": "a load history",
}

_OTHER_SUPPORTS = {
    "displacement": "prescribed displacements",
    "frictionless": "a frictionless support (normal-only constraint)",
    "cylindrical": "a cylindrical support (radial/axial/tangential constraints)",
    "remote": "a remote point",
    "elastic": "a foundation stiffness",
    "compression_only": "contact-like nonlinearity",
}

_OTHER_LOADS = {
    "moment": "a remote point to carry the moment",
    "remote_force": "a remote point",
    "bearing": "a bearing-load distribution",
    "bolt": "a bolt pretension section",
    "thermal": "a temperature field",
    "displacement": "a displacement support, not a load",
}


def _check_analysis(value: Any) -> None:
    if value is None:
        raise CaseError("analysis must be \"static_structural\"")
    token = _material_token(str(value))
    if token == STATIC_STRUCTURAL:
        return
    needs = _OTHER_ANALYSES.get(token)
    if needs:
        raise CaseError(
            f"{token} analysis needs {needs}; only static structural (linear "
            "elastic) is available. Remove \"analysis\" or set it to \"static_structural\""
        )
    raise CaseError(
        f"only \"static_structural\" analysis is available, not {value!r}"
    )


def _parse_material(spec: Any, role: str = "material") -> Material:
    if spec is None:
        raise CaseError(
            "the case needs a material: give material.preset (one of "
            f"{', '.join(MATERIAL_PRESETS)}) or material.youngs_modulus_mpa and "
            "material.poisson_ratio"
        )
    spec = _check_keys(spec, _MATERIAL_KEYS, role)
    given = {k: v for k, v in spec.items() if v is not None and k != "preset"}
    if spec.get("preset") is not None:
        if not isinstance(spec["preset"], str):
            raise CaseError(f"{role}.preset must be a preset name, e.g. \"structural_steel\"")
        base = material_from_key(spec["preset"])
        merged = base.as_dict()
        merged.update(given)
        name = merged.pop("name")
        merged.pop("key"), merged.pop("source")
        edited = any(
            merged[k] != getattr(base, k) for k in merged
        ) or name != base.name
        if not edited:
            return base
        if name == base.name:
            name = f"{base.name} (edited)"
        return custom_material(name=name, **merged)
    if "youngs_modulus_mpa" not in given:
        raise CaseError(
            "deformation needs Young's modulus: give material.preset or "
            "material.youngs_modulus_mpa (e.g. {\"preset\": \"structural_steel\"} or "
            "{\"youngs_modulus_mpa\": 200000, \"poisson_ratio\": 0.3})"
        )
    if "poisson_ratio" not in given:
        raise CaseError(
            "a custom material needs material.poisson_ratio as well as Young's "
            "modulus, e.g. 0.3 for steel or 0.33 for aluminium; or give material.preset"
        )
    name = given.pop("name", "Custom material")
    if not isinstance(name, str) or not name.strip():
        raise CaseError(f"{role}.name must be a non-empty string")
    return custom_material(name=name.strip(), **given)


def _parse_force(spec: Mapping[str, Any], role: str,
                 face_map: Mapping[int, FaceInfo] | None) -> ForceLoad:
    spec = _check_keys(spec, _FORCE_KEYS, role)
    faces = _face_tags(spec.get("faces"), f"{role}.faces")
    if face_map is not None:
        _check_known(faces, face_map.keys(), role)
    label = _label(spec.get("label"), "Force", role)
    has_vector = spec.get("vector_n") is not None
    has_polar = spec.get("magnitude_n") is not None or spec.get("direction") is not None
    if has_vector and has_polar:
        raise CaseError(
            f"{role}: give either vector_n (components in N) or magnitude_n with "
            "direction, not both"
        )
    if has_vector:
        vector = _vector3(spec["vector_n"], f"{role}.vector_n", "[0, 0, -500]")
        return ForceLoad(faces=faces, vector_n=vector, label=label)
    if spec.get("magnitude_n") is None:
        raise CaseError(
            f"{role} needs a size: vector_n in N (e.g. [0, 0, -500]) or magnitude_n "
            "with direction (e.g. 500 and \"-z\")"
        )
    magnitude = _real(spec["magnitude_n"], f"{role}.magnitude_n", "500")
    if not magnitude > 0:
        raise CaseError(
            f"{role}.magnitude_n must be positive; put the sign in the direction, "
            "e.g. magnitude_n 500 with direction \"-z\""
        )
    if spec.get("direction") is None:
        raise CaseError(
            f"{role} has no direction; give direction as an axis (\"-z\"), "
            "\"normal\"/\"out\", or three numbers like [0, 0, -1]. It is never assumed."
        )
    direction_spec = spec["direction"]
    if isinstance(direction_spec, str):
        loaded = [face_map[t] for t in faces] if face_map is not None else ()
        if direction_spec.strip().lower() in _INWARD_WORDS | _OUTWARD_WORDS and face_map is None:
            raise CaseError(
                f"{role}.direction {direction_spec!r} needs the part's faces to find the "
                "normal; give an axis like \"-z\" or three numbers instead"
            )
        direction, _ = parse_direction(direction_spec, loaded)
    else:
        direction = _unit(
            _vector3(direction_spec, f"{role}.direction", "[0, 0, -1]"),
            f"{role}.direction", "[0, 0, -1]",
        )
    vector = (direction[0] * magnitude, direction[1] * magnitude, direction[2] * magnitude)
    return ForceLoad(faces=faces, vector_n=vector, label=label)


def _parse_pressure(spec: Mapping[str, Any], role: str,
                    face_map: Mapping[int, FaceInfo] | None) -> PressureLoad:
    spec = _check_keys(spec, _PRESSURE_KEYS, role)
    faces = _face_tags(spec.get("faces"), f"{role}.faces")
    if face_map is not None:
        _check_known(faces, face_map.keys(), role)
    if spec.get("pressure_mpa") is None:
        raise CaseError(f"{role} needs pressure_mpa (MPa, positive pushes into the face), e.g. 1.5")
    pressure = _real(spec["pressure_mpa"], f"{role}.pressure_mpa", "1.5")
    return PressureLoad(faces=faces, pressure_mpa=pressure,
                        label=_label(spec.get("label"), "Pressure", role))


def _parse_load(spec: Any, role: str, face_map: Mapping[int, FaceInfo] | None) -> Load:
    if not isinstance(spec, Mapping):
        raise CaseError(
            f"{role} must be an object like {{\"type\": \"force\", \"faces\": [6], "
            "\"vector_n\": [0, 0, -500]}"
        )
    kind = spec.get("type")
    token = _material_token(kind) if isinstance(kind, str) else None
    if token == "force":
        return _parse_force(spec, role, face_map)
    if token == "pressure":
        return _parse_pressure(spec, role, face_map)
    if token == "gravity":
        raise CaseError(
            f"{role}: gravity is not a face load; set the top-level "
            "\"gravity\": {\"enabled\": true}"
        )
    if token in _OTHER_LOADS:
        raise CaseError(
            f"{role}: a {token} load needs {_OTHER_LOADS[token]}, which this solver "
            "does not have; use \"force\" (total N on faces) or \"pressure\" (MPa normal to faces)"
        )
    raise CaseError(
        f"{role}.type must be \"force\" or \"pressure\", not {kind!r}"
    )


def _parse_support(spec: Any, role: str,
                   face_map: Mapping[int, FaceInfo] | None) -> FixedSupport:
    spec = _check_keys(spec, _SUPPORT_KEYS, role)
    kind = spec.get("type", "fixed")
    token = _material_token(kind) if isinstance(kind, str) else None
    if token in _OTHER_SUPPORTS:
        raise CaseError(
            f"{role}: a {token} support needs {_OTHER_SUPPORTS[token]}, which this "
            "solver does not have; only \"fixed\" supports (all translations held) are available"
        )
    if token not in ("fixed", "fixed_support"):
        raise CaseError(f"{role}.type must be \"fixed\", not {kind!r}")
    faces = _face_tags(spec.get("faces"), f"{role}.faces")
    if face_map is not None:
        _check_known(faces, face_map.keys(), role)
    return FixedSupport(faces=faces, label=_label(spec.get("label"), "Fixed Support", role))


def _parse_gravity(spec: Any) -> Gravity | None:
    if spec is None:
        return None
    spec = _check_keys(spec, _GRAVITY_KEYS, "gravity")
    enabled = spec.get("enabled", True)
    if not isinstance(enabled, bool):
        raise CaseError("gravity.enabled must be true or false")
    if not enabled:
        return None
    acceleration = spec.get("acceleration_mps2")
    if acceleration is None:
        acceleration = (0.0, 0.0, -STANDARD_GRAVITY)
    vector = _vector3(acceleration, "gravity.acceleration_mps2", "[0, 0, -9.80665]")
    return Gravity(acceleration_mps2=vector,
                   label=_label(spec.get("label"), "Standard Earth Gravity", "gravity"))


def _parse_mesh(spec: Any) -> MeshSettings:
    if spec is None:
        return MeshSettings()
    spec = _check_keys(spec, _MESH_KEYS, "mesh")
    order = spec.get("element_order", 2)
    if isinstance(order, float) and order.is_integer():
        order = int(order)
    return MeshSettings(
        element_order=order,
        refinement="medium" if spec.get("refinement") is None else spec["refinement"],
        element_size_mm=spec.get("element_size_mm"),
    )


def _from_legacy(document: Mapping[str, Any]) -> dict[str, Any]:
    """The MCP `load_case` shape, converted to StaticStructuralCase/v1.

    Converted, not reinterpreted: anything the old shape left to a default that
    the full solve cannot guess (E, a load direction) is refused here.
    """
    document = _check_keys(document, _LEGACY_TOP_KEYS, "the load case")
    out: dict[str, Any] = {"schema_version": CASE_SCHEMA}
    if "analysis" in document:
        out["analysis"] = document["analysis"]

    if "fixed" not in document:
        raise CaseError(
            "the load case needs held faces: add \"fixed\": [1] (face numbers from "
            "the face table)"
        )
    fixed = document["fixed"]
    if isinstance(fixed, Mapping) and set(fixed) & {"centroid", "radial"}:
        raise CaseError(
            "centroid/radial selectors are for surrogate screening; a structural "
            "solve names fixed faces by number, e.g. \"fixed\": [1] (run "
            "`fea-structural inspect PART.step` for the face table)"
        )
    if isinstance(fixed, Mapping):
        _check_keys(fixed, {"faces"}, "fixed")
    out["supports"] = [{"type": "fixed", "faces": list(_face_tags(fixed, "fixed"))}]

    if "load" in document and "loads" in document:
        raise CaseError("give either \"load\" or \"loads\", not both")
    if "loads" in document:
        entries = document["loads"]
        if isinstance(entries, (str, bytes, Mapping)) or not isinstance(entries, Sequence):
            raise CaseError("\"loads\" must be a list of loads, e.g. [{\"faces\": [6], ...}]")
    elif "load" in document:
        entries = [document["load"]]
    else:
        entries = []
    loads: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        role = f"loads[{index}]" if "loads" in document else "load"
        entry = _check_keys(entry, _LEGACY_LOAD_KEYS, role)
        has_pressure = entry.get("pressure_mpa") is not None
        has_force = entry.get("magnitude_n") is not None
        if has_pressure and has_force:
            raise CaseError(f"{role}: give either magnitude_n or pressure_mpa, not both")
        if has_pressure:
            if entry.get("direction") is not None:
                raise CaseError(
                    f"{role}: a pressure with a direction was the screening shortcut "
                    "(p x area pushed along one direction). A real pressure acts "
                    "normal to the face: drop \"direction\", or give a force with "
                    "magnitude_n and direction instead"
                )
            loads.append({"type": "pressure", "faces": entry.get("faces"),
                          "pressure_mpa": entry["pressure_mpa"]})
        elif has_force:
            loads.append({"type": "force", "faces": entry.get("faces"),
                          "magnitude_n": entry["magnitude_n"],
                          "direction": entry.get("direction")})
        else:
            raise CaseError(f"{role} needs magnitude_n (N) with a direction, or pressure_mpa (MPa)")
    out["loads"] = loads

    out["material"] = document.get("material")

    gravity = document.get("gravity")
    if gravity is not None:
        gravity = _check_keys(gravity, _LEGACY_GRAVITY_KEYS, "gravity")
        enabled = gravity.get("enabled", False)
        if not isinstance(enabled, bool):
            raise CaseError("gravity.enabled must be true or false")
        if enabled:
            g = _real(gravity.get("g", STANDARD_GRAVITY), "gravity.g (m/s²)", "9.80665")
            if not g > 0:
                raise CaseError("gravity.g must be positive (m/s², acting along -Z), e.g. 9.80665")
            out["gravity"] = {"enabled": True, "acceleration_mps2": [0.0, 0.0, -g]}
        else:
            out["gravity"] = {"enabled": False}
    return out


def _is_legacy(document: Mapping[str, Any]) -> bool:
    legacy = {"fixed", "load"} & set(document)
    modern = {"supports", "schema_version", "mesh"} & set(document)
    if legacy and modern:
        raise CaseError(
            f"the case mixes the old load-case keys {sorted(legacy)} with "
            f"{CASE_SCHEMA} keys {sorted(modern)}; use \"supports\" and \"loads\" "
            "with \"type\" entries"
        )
    if legacy:
        return True
    loads = document.get("loads")
    if (
        "supports" not in document
        and isinstance(loads, Sequence) and not isinstance(loads, (str, bytes))
        and loads and all(isinstance(e, Mapping) and "type" not in e for e in loads)
    ):
        return True
    return False


def precheck_case(document: Mapping[str, Any]) -> None:
    """Check a case document before the part is read: keys, types, values,
    material, mesh -- everything except what needs the part's faces (face
    numbers, and "normal"/"out" directions, which are resolved later).

    Raises CaseError. Lets a broken case file fail in a moment instead of
    after the geometry has been read and meshed.
    """
    def defer_normals(value: Any) -> Any:
        if isinstance(value, Mapping):
            out = {}
            for key, item in value.items():
                if (key == "direction" and isinstance(item, str)
                        and item.strip().lower() in _INWARD_WORDS | _OUTWARD_WORDS):
                    out[key] = "-z"         # checked against the faces later
                else:
                    out[key] = defer_normals(item)
            return out
        if isinstance(value, list):
            return [defer_normals(item) for item in value]
        return value

    parse_case(defer_normals(document) if isinstance(document, Mapping) else document)


def parse_case(
    document: Mapping[str, Any], faces: Sequence[FaceInfo] | None = None
) -> StaticStructuralCase:
    """Read a StaticStructuralCase/v1 document (or the legacy MCP load case).

    With `faces` (cad_intake.Face or structural_mesh.FaceSummary), every face
    tag is checked against the part and "normal"/"into"/"out" directions are
    resolved; without it those directions are refused rather than guessed.
    Raises CaseError, never a bare KeyError/TypeError.
    """
    if not isinstance(document, Mapping):
        raise CaseError(
            f"the case must be a JSON object (parse the file first), not "
            f"{type(document).__name__}; see {CASE_SCHEMA} in the README"
        )
    if _is_legacy(document):
        document = _from_legacy(document)
    document = _check_keys(document, _TOP_KEYS, "the case")
    version = document.get("schema_version", CASE_SCHEMA)
    if version != CASE_SCHEMA:
        raise CaseError(
            f"schema_version {version!r} is not supported; this reads {CASE_SCHEMA!r}"
        )
    if "analysis" in document:
        _check_analysis(document["analysis"])

    face_map: dict[int, FaceInfo] | None = (
        {int(f.tag): f for f in faces} if faces is not None else None
    )

    supports_spec = document.get("supports")
    if (
        supports_spec is None
        or isinstance(supports_spec, (str, bytes, Mapping))
        or not isinstance(supports_spec, Sequence)
        or not supports_spec
    ):
        raise CaseError(
            "the case needs at least one fixed support, e.g. \"supports\": "
            "[{\"type\": \"fixed\", \"faces\": [1]}]"
        )
    supports = tuple(
        _parse_support(s, f"supports[{i}]", face_map) for i, s in enumerate(supports_spec)
    )

    loads_spec = document.get("loads")
    if loads_spec is None:
        loads_spec = []
    if isinstance(loads_spec, (str, bytes, Mapping)) or not isinstance(loads_spec, Sequence):
        raise CaseError(
            "\"loads\" must be a list, e.g. [{\"type\": \"force\", \"faces\": [6], "
            "\"vector_n\": [0, 0, -500]}]"
        )
    loads = tuple(_parse_load(s, f"loads[{i}]", face_map) for i, s in enumerate(loads_spec))

    return StaticStructuralCase(
        supports=supports,
        loads=loads,
        material=_parse_material(document.get("material")),
        gravity=_parse_gravity(document.get("gravity")),
        mesh=_parse_mesh(document.get("mesh")),
    )


__all__ = [
    "CASE_SCHEMA",
    "ELEMENT_ORDERS",
    "MATERIAL_ALIASES",
    "MATERIAL_PRESETS",
    "PRESET_CONFIDENCE",
    "REFINEMENT_LEVELS",
    "STANDARD_GRAVITY",
    "STATIC_STRUCTURAL",
    "CaseError",
    "FaceInfo",
    "FixedSupport",
    "ForceLoad",
    "Gravity",
    "Load",
    "Material",
    "MeshSettings",
    "PressureLoad",
    "StaticStructuralCase",
    "custom_material",
    "material_from_key",
    "parse_case",
    "parse_density",
    "parse_direction",
    "parse_face_list",
    "parse_force",
    "parse_force_spec",
    "parse_modulus",
    "parse_pressure",
    "precheck_case",
    "parse_vector",
]
