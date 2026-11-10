"""The questions a pre-processor asks, and the answers this surrogate can use.

Ansys, Abaqus and SimScale all begin the same way: which faces are held, what is
pushing on which faces, how hard, in what direction, what is the material, does
gravity matter. This module is that interview, with one difference that matters
more than the rest -- it says plainly which of those questions the model can
actually answer, and refuses the others instead of quietly ignoring them.

The sweep that trained the model applied exactly one uniform traction to a set of
tagged faces. `domain.load_pattern` records it as "single uniform traction on
tagged faces". So:

    supported   a force, or a pressure resolved into one, on faces, in a
                direction
    computed    self-weight, in closed form, only to decide whether it is
                negligible -- never screened
    refused     several simultaneous load sets, thermal, contact, nonlinear,
                dynamic, buckling, modal

A value the caller stated and a value proposed from geometry are different kinds
of thing, and the difference survives into the document. `build_graph` already
refuses a missing fixed mask outright, because "guessing one screens a load case
the caller did not ask for"; a proposed one is the same hazard with a softer
edge, so it travels as an assumption rather than as a fact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Mapping, Sequence


class LoadCaseError(ValueError):
    """The load case is incomplete, or asks for something out of domain."""


#: Above this fraction of the applied load, self-weight stops being a rounding
#: error and becomes a body load -- which the training set contains none of.
GRAVITY_NEGLIGIBLE_FRACTION = 0.05

#: mm^3 * kg/m^3 -> kg
_MM3_TO_M3 = 1.0e-9

STATIC_STRUCTURAL = "static_structural"

#: Named individually so a refusal can say what the analysis would need, rather
#: than "not supported". Each is a different solver, not a setting.
_OTHER_ANALYSES = {
    "thermal": "a temperature field and expansion coefficients",
    "contact": "surfaces that come into and out of contact",
    "nonlinear": "plasticity or large displacement",
    "dynamic": "time and inertia",
    "modal": "eigenvalues rather than a stress field",
    "buckling": "a stability eigenvalue rather than a stress field",
    "fatigue": "a load history",
}

_TOP_KEYS = frozenset({"fixed", "load", "loads", "material", "gravity", "analysis"})
_LOAD_KEYS = frozenset({"faces", "direction", "magnitude_n", "pressure_mpa"})
_MATERIAL_KEYS = frozenset(
    {"poisson_ratio", "density_kg_m3", "youngs_modulus_mpa", "name"}
)
_GRAVITY_KEYS = frozenset({"enabled", "g"})


@dataclass(frozen=True)
class Load:
    """One uniform traction. A force, or a pressure resolved into one."""

    faces: tuple[int, ...]
    magnitude_n: float
    direction: tuple[float, float, float]
    #: Set when the caller gave a pressure, so the document can say it did.
    pressure_mpa: float | None = None
    area_mm2: float | None = None


@dataclass(frozen=True)
class LoadCase:
    """A complete, screenable statement of what is being asked."""

    fixed: Mapping[str, Any]
    load: Load
    poisson_ratio: float
    #: Recorded and not used. Stress in a force-driven problem is independent of
    #: E exactly; dropping it silently would look like a bug.
    youngs_modulus_mpa: float | None = None
    density_kg_m3: float | None = None
    gravity_mps2: float | None = None
    self_weight_n: float | None = None
    material_name: str | None = None
    #: Which fields the caller actually stated. Everything else was proposed.
    stated: frozenset[str] = field(default_factory=frozenset)

    @property
    def loaded(self) -> Mapping[str, Any]:
        return {"faces": list(self.load.faces)}

    def assumptions(self) -> list[str]:
        """Everything a reader needs in order not to mistake this for a fact."""
        out: list[str] = []
        for name, description in (
            ("fixed", "held faces"),
            ("load_direction", "load direction"),
            ("poisson_ratio", "Poisson ratio"),
        ):
            if name not in self.stated:
                out.append(
                    f"{description} proposed, not stated by the caller -- "
                    "confirm before treating this as your load case"
                )
        if self.load.pressure_mpa is not None:
            out.append(
                f"pressure {self.load.pressure_mpa:g} MPa over "
                f"{self.load.area_mm2:g} mm^2 resolved to "
                f"{self.load.magnitude_n:g} N, which is the uniform traction "
                "the sweep applied"
            )
        if self.youngs_modulus_mpa is not None:
            out.append(
                f"Young's modulus {self.youngs_modulus_mpa:g} MPa recorded and "
                "not used: for a force-driven problem with zero-displacement "
                "constraints the stress field is independent of E exactly"
            )
        if self.self_weight_n is not None:
            share = self.self_weight_n / self.load.magnitude_n
            out.append(
                f"self-weight {self.self_weight_n:.4g} N against "
                f"{self.load.magnitude_n:g} N applied, {share:.2%}, treated as "
                "negligible and not screened"
            )
        if self.material_name:
            out.append(f"material named {self.material_name!r} by the caller")
        return out


def _number(value: Any, label: str) -> float:
    """A finite real number, or a refusal that names the field.

    `bool` is an int subclass, so `True` would otherwise arrive as 1 N. A
    non-finite value would pass every sign check (`inf > 0`, and NaN fails
    every comparison silently) and reach the solver as a load or a modulus.
    Numeric strings stay accepted, as `float()` always accepted them.
    """
    if isinstance(value, bool):
        raise LoadCaseError(f"{label} must be a number, not {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise LoadCaseError(f"{label} must be a number, not {value!r}") from None
    if not math.isfinite(number):
        raise LoadCaseError(f"{label} must be finite, not {value!r}")
    return number


def _unit(vector: Any, role: str) -> tuple[float, float, float]:
    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
        raise LoadCaseError(f"{role} direction must be three numbers")
    values = [_number(v, f"{role} direction component") for v in vector]
    if len(values) != 3:
        raise LoadCaseError(f"{role} direction must be three numbers")
    norm = sum(v * v for v in values) ** 0.5
    if norm == 0.0:
        raise LoadCaseError(f"{role} direction must not be the zero vector")
    return (values[0] / norm, values[1] / norm, values[2] / norm)


def _faces(spec: Any, role: str) -> tuple[int, ...]:
    if isinstance(spec, Mapping) and "faces" in spec:
        spec = spec["faces"]
    if isinstance(spec, bool):
        raise LoadCaseError(f"{role} must name faces by tag")
    if isinstance(spec, int):
        spec = [spec]
    if not isinstance(spec, Sequence) or isinstance(spec, (str, bytes)) or not spec:
        raise LoadCaseError(f"{role} must name at least one face, as a list of tags")
    tags = [_tag(t, role) for t in spec]
    # Order kept, repeats dropped. A face named twice is one face: counting it
    # twice doubled the area a pressure was resolved over, while the solver's
    # node mask spread the result over the real area once -- twice the stated
    # pressure, applied silently.
    return tuple(dict.fromkeys(tags))


def _tag(value: Any, role: str) -> int:
    """One face tag. `int()` alone turned face 1.7 into face 1."""
    if isinstance(value, bool):
        raise LoadCaseError(f"{role} face tags must be integers, not {value!r}")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return int(number)
        raise LoadCaseError(
            f"{role} face tags must be integers, not {value!r}; use the tags "
            "listed by surrogate_inspect_cad"
        )
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            pass
    raise LoadCaseError(
        f"{role} face tags must be integers, not {value!r}; use the tags "
        "listed by surrogate_inspect_cad"
    )


def _known_faces(faces: Sequence[int], solid, role: str) -> None:  # noqa: ANN001
    """Refuse a tag the part does not have, as a load-case error.

    Checked against `solid.faces` rather than through `solid.face()`, which
    raises the CAD layer's own error type: an unknown tag is a mistake in the
    load case, and the caller should be able to catch it as one.
    """
    known = [f.tag for f in solid.faces]
    present = set(known)
    unknown = [t for t in faces if t not in present]
    if unknown:
        raise LoadCaseError(
            f"{role} names face {', '.join(map(str, unknown))}, which this part "
            f"does not have; its faces are {', '.join(map(str, known))}"
        )


def _check_keys(value: Any, allowed: frozenset[str], role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LoadCaseError(f"{role} must be an object")
    unknown = set(value) - allowed
    if unknown:
        raise LoadCaseError(
            f"unexpected {sorted(unknown)} in {role}; it carries {sorted(allowed)}"
        )
    return value


def parse(document: Mapping[str, Any], solid=None) -> LoadCase:  # noqa: ANN001
    """Read a load case, refusing anything the model cannot answer.

    `solid` supplies the face areas a pressure needs and the volume self-weight
    needs. Without it, a pressure or a gravity request is refused rather than
    guessed at.
    """
    document = _check_keys(document, _TOP_KEYS, "the load case")

    analysis = str(document.get("analysis", STATIC_STRUCTURAL)).lower()
    if analysis != STATIC_STRUCTURAL:
        needs = _OTHER_ANALYSES.get(analysis)
        raise LoadCaseError(
            f"{analysis} analysis needs {needs}, and the training set contains "
            "none of it"
            if needs
            else f"only {STATIC_STRUCTURAL} is supported, not {analysis!r}"
        )

    if "loads" in document and "load" in document:
        raise LoadCaseError("give either 'load' or 'loads', not both")
    if "loads" in document:
        entries = document["loads"]
        if isinstance(entries, Mapping):
            raise LoadCaseError(
                "'loads' must be a list of load objects, not a single object; "
                "give one load as 'load': {...} or as 'loads': [{...}]"
            )
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise LoadCaseError(
                "'loads' must be a list holding one load object, e.g. "
                "'loads': [{\"faces\": [6], \"magnitude_n\": 500}]"
            )
        if not entries:
            raise LoadCaseError(
                "'loads' is empty; a load case needs one load, e.g. "
                "'loads': [{\"faces\": [6], \"magnitude_n\": 500}]"
            )
        if len(entries) != 1:
            raise LoadCaseError(
                "exactly one load set is supported. Fields superpose in linear "
                "elasticity, but the screened quantity is a critical-distance "
                "maximum, and the maximum of a sum is not the sum of the "
                "maxima, so two load sets cannot be combined after the fact. "
                "Screen them separately."
            )
        load_spec = entries[0]
    elif "load" in document:
        load_spec = document["load"]
    else:
        raise LoadCaseError("a load case needs a load")
    load_spec = _check_keys(load_spec, _LOAD_KEYS, "the load")

    if "fixed" not in document:
        raise LoadCaseError(
            "a load case needs held faces. An unconstrained body has rigid-body "
            "freedom and no stress solution, and guessing which faces are held "
            "screens a load case nobody asked for."
        )

    stated = {"fixed"}
    fixed = document["fixed"]
    held_faces: tuple[int, ...] | None = None    # None for a geometric selector
    if isinstance(fixed, Mapping) and set(fixed) & {"centroid", "radial"}:
        fixed_spec: Mapping[str, Any] = dict(fixed)
    else:
        held_faces = _faces(fixed, "fixed")
        fixed_spec = {"faces": list(held_faces)}

    faces = _faces(load_spec.get("faces"), "the load")
    if solid is not None:
        # Held faces are not checked here: an absent held face is refused by
        # the face selector when the problem is posed on the mesh, with its own
        # message, and the solve path's tests pin that.
        _known_faces(faces, solid, "the load")
    if held_faces is not None and set(faces) <= set(held_faces):
        # Every loaded node would be clamped, the solver drops loads on
        # constrained freedoms, and the answer is a part with no stress in it
        # that looks perfectly healthy.
        raise LoadCaseError(
            f"the load acts only on held faces ({', '.join(map(str, faces))}); "
            "a load on a fixed face goes straight into the support and stresses "
            "nothing. Load a face that is not held."
        )
    if "direction" in load_spec:
        direction = _unit(load_spec["direction"], "the load")
        stated.add("load_direction")
    else:
        direction = (0.0, 0.0, -1.0)

    pressure = load_spec.get("pressure_mpa")
    magnitude_n = load_spec.get("magnitude_n")
    if pressure is not None and magnitude_n is not None:
        raise LoadCaseError("give either magnitude_n or pressure_mpa, not both")
    if pressure is not None:
        pressure = _number(pressure, "pressure_mpa")
        if solid is None:
            raise LoadCaseError(
                "a pressure needs the loaded faces' area; inspect the CAD first"
            )
        area: float | None = float(sum(solid.face(t).area for t in faces))
        magnitude = float(pressure) * float(area)
    else:
        area = None
        magnitude = (
            0.0 if magnitude_n is None else _number(magnitude_n, "magnitude_n")
        )
    if not magnitude > 0.0:
        raise LoadCaseError("the load must be a positive force or pressure")

    material = _check_keys(document.get("material") or {}, _MATERIAL_KEYS, "material")
    raw_poisson = material.get("poisson_ratio")
    if raw_poisson is None:
        poisson = 0.3
    else:
        poisson = _number(raw_poisson, "material.poisson_ratio")
        if not -1.0 < poisson < 0.5:
            raise LoadCaseError(
                f"material.poisson_ratio {poisson:g} is not a linear elastic "
                "material: it must lie strictly between -1 and 0.5"
            )
        stated.add("poisson_ratio")

    modulus = material.get("youngs_modulus_mpa")
    if modulus is not None:
        modulus = _number(modulus, "material.youngs_modulus_mpa")
        if not modulus > 0.0:
            raise LoadCaseError("material.youngs_modulus_mpa must be positive")

    density = material.get("density_kg_m3")
    if density is not None:
        density = _number(density, "material.density_kg_m3")
        if not density > 0.0:
            # A negative density gives a negative self-weight, which passes
            # the negligibility test however heavy the part is.
            raise LoadCaseError("material.density_kg_m3 must be positive")

    raw_gravity = document.get("gravity")
    if raw_gravity is None:
        raw_gravity = {}
    if not isinstance(raw_gravity, Mapping):
        raise LoadCaseError(
            "gravity must be an object such as {\"enabled\": true, \"g\": 9.80665}"
            f", not {raw_gravity!r}"
        )
    # Key-checked like every other block: a misspelt 'enable' used to switch
    # the self-weight check off without a word.
    gravity = _check_keys(raw_gravity, _GRAVITY_KEYS, "gravity")
    enabled = gravity.get("enabled", False)
    if not isinstance(enabled, bool):
        raise LoadCaseError(
            f"gravity.enabled must be true or false, not {enabled!r}"
        )
    g = None
    if gravity.get("g") is not None:
        g = _number(gravity["g"], "gravity.g")
        if not g > 0.0:
            raise LoadCaseError(
                "gravity.g is the magnitude of the acceleration in m/s^2 and "
                "must be positive"
            )
    self_weight = None
    if enabled:
        if density is None:
            raise LoadCaseError("gravity needs material.density_kg_m3")
        if solid is None:
            raise LoadCaseError(
                "gravity needs the part's volume; inspect the CAD first"
            )
        g = 9.80665 if g is None else g
        weight = float(density) * float(solid.volume) * _MM3_TO_M3 * g
        share = weight / magnitude
        if share >= GRAVITY_NEGLIGIBLE_FRACTION:
            raise LoadCaseError(
                f"self-weight is {weight:.4g} N against {magnitude:g} N applied, "
                f"{share:.1%} of the load. Every sample in the training set is a "
                "single uniform traction on tagged faces with no body load "
                "anywhere, so a self-weight this large sits outside the domain. "
                "Either screen without gravity and read the result as covering "
                "the applied load only, or solve the part properly."
            )
        self_weight = weight

    return LoadCase(
        fixed=fixed_spec,
        load=Load(
            faces=faces,
            magnitude_n=magnitude,
            direction=direction,
            pressure_mpa=float(pressure) if pressure is not None else None,
            area_mm2=area,
        ),
        poisson_ratio=poisson,
        youngs_modulus_mpa=modulus,
        density_kg_m3=density,
        gravity_mps2=g if enabled else None,
        self_weight_n=self_weight,
        material_name=material.get("name"),
        stated=frozenset(stated),
    )


def propose(solid) -> dict[str, Any]:  # noqa: ANN001
    """A starting point, explicitly not an answer.

    The largest planar face is offered as the support and the one most nearly
    facing away from it as the load. That is a fair opening guess for a bracket
    and a poor one for a pulley, which is exactly why it comes back as a proposal
    with the open questions beside it rather than applied.
    """
    import numpy as np

    # Prefer planar faces, but do not require them. A real machined part can
    # have none at all: the OpenCASCADE test component `component8` is 21 faces
    # and every one of them curved.
    planar = [f for f in solid.faces if f.planar]
    candidates = planar or list(solid.faces)
    support = candidates[0]
    others = [f for f in candidates if f.tag != support.tag]
    opposed = (
        min(others, key=lambda f: float(np.dot(f.normal, support.normal)))
        if others
        else support
    )
    kind = "planar" if support.planar else "curved"

    return {
        "proposed": {
            "fixed": {"faces": [support.tag]},
            "load": {
                "faces": [opposed.tag],
                "direction": [-float(v) for v in opposed.normal],
            },
            "material": {"poisson_ratio": 0.3},
            "analysis": STATIC_STRUCTURAL,
        },
        "why": {
            "fixed": (
                f"face {support.tag} is the largest {kind} face "
                f"({support.area:.4g} mm^2)"
                + ("" if planar else "; this part has no planar face at all, so "
                   "the guess is weaker than usual")
            ),
            "load": (
                f"face {opposed.tag} faces most nearly away from it, so a load "
                "pushing inward there is the commonest arrangement"
            ),
        },
        "still_needed": [
            "confirm or replace the held faces: holding the wrong face screens a "
            "different question and the result will look perfectly healthy",
            "confirm or replace the loaded faces and the direction",
            "magnitude_n, or pressure_mpa -- no default is meaningful",
            "material.poisson_ratio if not 0.3; the model is fitted over "
            "0.27 to 0.42, which spans steel, aluminium and most engineering "
            "polymers",
            "material.density_kg_m3 and gravity.enabled, if self-weight may "
            "be a non-trivial fraction of the applied load",
        ],
    }


__all__ = [
    "GRAVITY_NEGLIGIBLE_FRACTION",
    "STATIC_STRUCTURAL",
    "Load",
    "LoadCase",
    "LoadCaseError",
    "parse",
    "propose",
]
