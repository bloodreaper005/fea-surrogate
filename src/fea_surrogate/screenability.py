"""When the surrogate may speak about a static structural case, and why not.

The surrogate was trained on one family at a time, each with exactly one load
case: one uniform traction on the faces the family's `tag()` picks, along one
fixed axis, Poisson 0.27-0.42, parts of a measured size. Three measured facts
make anything wider unsafe (reports/surrogate_model.json):

    direction-blind   on a bracket held at face 1 and loaded at face 6, the
                      solved critical-distance stress was 208.6 MPa along -z,
                      3.9 MPa along +x and 32.7 MPa along +y; the screening
                      interval was 165-313 MPa for all three. Wrong by 42x, and
                      it looked healthy.
    Poisson-blind     tau moved under 2% across nu 0.0-0.49.
    not scale-free    the same bracket scaled to 33/132/529/2645 mm gave
                      tau 1279/3978/2608/1.67e6, where physics says constant.

So screening is allowed only when the case *is* a family's trained load case:
same held faces, same loaded faces, the trained axis (either sign: von Mises is
sign-invariant), Poisson in range, size in that family's training range, one
load, one body, no meaningful body load. Anything else is reported as "not
screenable" with every failing reason, and the solve still runs. Most real load
cases will fail this gate; that is the honest outcome, not a bug.

Import-light by contract (stdlib + numpy): `fea_surrogate.predict` imports
torch at module top, so the untransferable-family list is a local copy and
`tests/test_screenability.py` checks the two agree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .face_selectors import FAMILY_FACE_PRESETS, FaceSelectionError, select_faces
from .load_case import GRAVITY_NEGLIGIBLE_FRACTION
from .structural_case import (
    CaseError,
    ForceLoad,
    PressureLoad,
    StaticStructuralCase,
    parse_direction,
)


#: Training range of the bounding-box diagonal L (mm) per family, measured from
#: the `length_mm` of every sample in data/sweep. The model is not scale
#: invariant, so a part outside its family's range is outside the domain even
#: though the declared characteristic-length bound (1-10000 mm) would accept it.
FAMILY_LENGTH_RANGES_MM: dict[str, tuple[float, float]] = {
    "bracket": (58.0, 367.0),
    "flange": (89.0, 328.0),
    "pulley": (114.0, 571.0),
    "stepped_shaft": (60.0, 246.0),
    "welded_t": (134.0, 357.0),
    "spring": (31.0, 208.0),
}

#: The calibration `set_sha256` those ranges were measured against. Weights
#: calibrated on another sweep may have seen other sizes, so the ranges above
#: stop meaning anything and screening is refused until they are re-measured.
RANGES_MEASURED_ON_SET = "2226539405671f9897ad2823fc6984363c89712e95fecce52895240a69aa2335"

#: Local copy of `predict.UNTRANSFERABLE_FAMILIES` (names only; predict imports
#: torch). A test keeps the two in step.
UNTRANSFERABLE_FAMILIES = frozenset({"spring"})

#: The calibration's declared Poisson range, used when no domain is supplied.
DEFAULT_POISSON_RANGE = (0.27, 0.42)

#: |cos| between the load and the preset axis. 0.98 is about 11 degrees: a
#: tolerance for a slightly skewed CAD axis, not for a different load case.
DIRECTION_COS_MIN = 0.98

#: mm^3 * kg/m^3 -> kg
_MM3_TO_M3 = 1.0e-9


@dataclass(frozen=True)
class ScreeningVerdict:
    """Whether the surrogate may screen this case, and every reason it may not.

    `legacy_document` is the `load_case.parse` shape of the one load, set only
    when `available`: it drives both the screening and the P1 label-solver
    crosscheck, so the two answer the same single-traction question.
    """

    available: bool
    reasons: tuple[str, ...]
    family: str | None
    matching_families: tuple[str, ...]
    force_n: float | None
    direction: tuple[float, float, float] | None
    fixed_faces: tuple[int, ...]
    loaded_faces: tuple[int, ...]
    legacy_document: dict[str, Any] | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reasons": list(self.reasons),
            "family": self.family,
            "matching_families": list(self.matching_families),
            "force_n": self.force_n,
            "direction": list(self.direction) if self.direction else None,
            "fixed_faces": list(self.fixed_faces),
            "loaded_faces": list(self.loaded_faces),
        }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _faces_by_tag(mesh) -> dict[int, Any]:  # noqa: ANN001 - AnalysisMesh
    return {int(f.tag): f for f in getattr(mesh, "faces", ())}


def _axis_name(vector: Sequence[float]) -> str:
    for index, name in enumerate("XYZ"):
        if abs(float(vector[index])) > 0.999:
            return ("+" if float(vector[index]) > 0 else "-") + name
    return "(" + ", ".join(f"{float(c):.3g}" for c in vector) + ")"


def _tags(tags: Sequence[int]) -> str:
    return "[" + ", ".join(str(int(t)) for t in tags) + "]"


def mesh_volume_mm3(mesh) -> float | None:  # noqa: ANN001 - AnalysisMesh
    """Part volume from the mesh corners (exact for straight-sided tets).

    `AnalysisMesh` carries no volume of its own; a `volume_mm3` attribute wins
    when an object has one. None when neither is available, so a caller can
    refuse rather than guess a self-weight.
    """
    given = getattr(mesh, "volume_mm3", None)
    if given is not None:
        return float(given)
    try:
        import numpy as np

        nodes = np.asarray(mesh.nodes, dtype=float)
        tets = np.asarray(mesh.elements)[:, :4]
        a, b, c, d = (nodes[tets[:, i]] for i in range(4))
        det = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)
        volume = float(np.abs(det).sum() / 6.0)
    except Exception:
        return None
    return volume if math.isfinite(volume) and volume > 0 else None


def _poisson_range(domain: Mapping[str, Any] | None) -> tuple[float, float]:
    spec = (domain or {}).get("poisson_ratio")
    try:
        if isinstance(spec, Mapping):
            return float(spec["min"]), float(spec["max"])
        if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)) and len(spec) == 2:
            return float(spec[0]), float(spec[1])
    except (KeyError, TypeError, ValueError):
        pass
    return DEFAULT_POISSON_RANGE


def _single_traction(
    case: StaticStructuralCase, mesh  # noqa: ANN001
) -> tuple[float | None, tuple[float, float, float] | None, str | None]:
    """(force N, unit direction, None) for the one load, or (None, None, reason).

    A pressure counts only on flat faces sharing one normal: there it is one
    uniform traction |p|*A along -sign(p)*n, exactly what the sweep applied. On
    a bore it pushes in every direction and has no single-traction equivalent.
    """
    if len(case.loads) != 1:
        return None, None, None
    load = case.loads[0]
    if isinstance(load, ForceLoad):
        return load.magnitude_n, load.direction, None
    if isinstance(load, PressureLoad):
        by_tag = _faces_by_tag(mesh)
        missing = [t for t in load.faces if t not in by_tag]
        if missing:
            return None, None, (
                f"pressure faces {_tags(missing)} are not faces of this mesh"
            )
        faces = [by_tag[t] for t in load.faces]
        word = "into" if load.pressure_mpa > 0 else "out"
        try:
            direction, _ = parse_direction(word, faces)
        except CaseError:
            return None, None, (
                f"the pressure on faces {_tags(load.faces)} acts on curved or "
                "non-parallel faces, so it pushes in many directions; the "
                "surrogate was trained on one uniform traction along one axis"
            )
        area = float(sum(float(f.area) for f in faces))
        return abs(load.pressure_mpa) * area, direction, None
    return None, None, f"unsupported load type {type(load).__name__}"


def _preset_faces(family: str, mesh, role: str) -> tuple[int, ...]:  # noqa: ANN001
    return tuple(sorted(int(t) for t in select_faces(FAMILY_FACE_PRESETS[family][role], mesh, role)))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(sum(float(x) ** 2 for x in a))
    nb = math.sqrt(sum(float(x) ** 2 for x in b))
    if not na > 0 or not nb > 0:
        return 0.0
    return sum(float(x) * float(y) for x, y in zip(a, b)) / (na * nb)


# --------------------------------------------------------------------------
# public
# --------------------------------------------------------------------------


def matching_families(case: StaticStructuralCase, mesh) -> tuple[str, ...]:  # noqa: ANN001
    """Families whose trained load case this is: same faces, same axis (either sign).

    For a suggestion at the prompt only; `check` still applies every rule.
    Untransferable families are never suggested.
    """
    force_n, direction, _ = _single_traction(case, mesh)
    if force_n is None or direction is None:
        return ()
    fixed, loaded = case.fixed_faces(), case.loaded_faces()
    found: list[str] = []
    for family in sorted(FAMILY_FACE_PRESETS):
        if family in UNTRANSFERABLE_FAMILIES:
            continue
        try:
            if _preset_faces(family, mesh, "fixed") != fixed:
                continue
            if _preset_faces(family, mesh, "loaded") != loaded:
                continue
        except (FaceSelectionError, KeyError, ValueError, TypeError, AttributeError):
            continue
        if abs(_cosine(direction, FAMILY_FACE_PRESETS[family]["direction"])) >= DIRECTION_COS_MIN:
            found.append(family)
    return tuple(found)


def check(
    case: StaticStructuralCase,
    mesh,  # noqa: ANN001 - structural_mesh.AnalysisMesh
    *,
    family: str | None,
    domain: Mapping[str, Any] | None = None,
    calibration_set_sha256: str | None = None,
) -> ScreeningVerdict:
    """Apply every rule; collect every failure rather than stopping at the first.

    A person told only the first reason fixes it and hits the next; told all of
    them, they can see at once whether the case can ever be screened.
    """
    reasons: list[str] = []
    family = (family or "").strip().lower() or None
    fixed, loaded = case.fixed_faces(), case.loaded_faces()

    # (h) one body
    solids = int(getattr(mesh, "solid_count", 1) or 1)
    if solids != 1:
        reasons.append(
            f"the file holds {solids} solids; the surrogate has only seen single parts"
        )

    # (a) exactly one load; (b) a force, or a pressure on flat parallel faces
    force_n, direction, traction_reason = _single_traction(case, mesh)
    if not case.loads:
        reasons.append(
            "gravity is the only load; the surrogate was trained on one traction "
            "on faces and no body loads"
        )
    elif len(case.loads) > 1:
        reasons.append(
            f"{len(case.loads)} loads; the surrogate was trained on exactly one "
            "traction, and screenings of separate loads cannot be added (the "
            "maximum of a sum is not the sum of the maxima)"
        )
    elif traction_reason:
        reasons.append(traction_reason)

    # (a) gravity only while self-weight is negligible
    if case.gravity is not None and force_n:
        volume = mesh_volume_mm3(mesh)
        density = case.material.density_kg_m3
        if volume is None or density is None:
            reasons.append(
                "gravity is on and the self-weight could not be computed; the "
                "surrogate saw no body loads"
            )
        else:
            weight = density * volume * _MM3_TO_M3 * case.gravity.magnitude_mps2
            share = weight / force_n
            if share >= GRAVITY_NEGLIGIBLE_FRACTION:
                reasons.append(
                    f"gravity: self-weight {weight:.3g} N is {share:.1%} of the "
                    f"{force_n:.4g} N load; the surrogate saw no body loads "
                    f"(allowed below {GRAVITY_NEGLIGIBLE_FRACTION:.0%})"
                )

    # (c) a named, known, transferable family
    preset = None
    if family is None:
        suggestion = matching_families(case, mesh)
        hint = (
            f"; this case matches the {suggestion[0]!r} training case, so "
            f"--family {suggestion[0]} would let it screen"
            if len(suggestion) == 1 else ""
        )
        reasons.append(
            "no surrogate family named; the model cannot tell load directions "
            "apart, so it may only speak on a family's own trained load case"
            + hint
        )
    elif family not in FAMILY_FACE_PRESETS:
        reasons.append(
            f"unknown family {family!r}; known families are "
            f"{', '.join(sorted(FAMILY_FACE_PRESETS))}"
        )
    elif family in UNTRANSFERABLE_FAMILIES:
        reasons.append(
            f"family {family!r} is refused: the shipped weights were fitted on "
            "geometry real CAD does not reproduce (measured 0.006 of the solved "
            "value), so its screening is not reported"
        )
    else:
        preset = FAMILY_FACE_PRESETS[family]

    if preset is not None and family is not None:
        # (d) the family's own held and loaded faces
        for role, have in (("fixed", fixed), ("loaded", loaded)):
            noun = "held" if role == "fixed" else "loaded"
            try:
                want = _preset_faces(family, mesh, role)
            except (FaceSelectionError, KeyError, ValueError, TypeError, AttributeError) as exc:
                reasons.append(
                    f"the {family} training case's {noun} faces cannot be found on "
                    f"this part ({exc})"
                )
                continue
            if want != have:
                reasons.append(
                    f"{noun} faces {_tags(have)} differ from the {family} training "
                    f"case, which {'holds' if role == 'fixed' else 'loads'} faces "
                    f"{_tags(want)}"
                )
        # (e) along the trained axis, either sign
        trained = preset["direction"]
        if direction is not None:
            cosine = abs(_cosine(direction, trained))
            if cosine < DIRECTION_COS_MIN:
                angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
                reasons.append(
                    f"load direction {_axis_name(direction)} is {angle:.0f} degrees "
                    f"off the {family} training axis {_axis_name(trained)} (either "
                    "sign); the model is blind to direction and would answer for "
                    "the trained one"
                )
        # (g) size inside that family's training range
        span = FAMILY_LENGTH_RANGES_MM.get(family)
        diagonal = float(getattr(mesh, "diagonal_mm", float("nan")))
        if span is None:
            reasons.append(f"no measured training size range for family {family!r}")
        elif not (math.isfinite(diagonal) and span[0] <= diagonal <= span[1]):
            reasons.append(
                f"size: the part's diagonal is {diagonal:.4g} mm, outside the "
                f"{family} training range {span[0]:g}-{span[1]:g} mm; the model "
                "is not scale-invariant"
            )
        if calibration_set_sha256 is not None and calibration_set_sha256 != RANGES_MEASURED_ON_SET:
            reasons.append(
                "size ranges were measured on a different calibration set "
                f"({RANGES_MEASURED_ON_SET[:12]}..., this model cites "
                f"{str(calibration_set_sha256)[:12]}...); re-measure "
                "FAMILY_LENGTH_RANGES_MM for these weights"
            )

    # (f) Poisson inside the trained range
    low, high = _poisson_range(domain)
    nu = case.material.poisson_ratio
    if not low <= nu <= high:
        reasons.append(
            f"Poisson's ratio {nu:g} is outside the trained {low:g}-{high:g}"
        )

    available = not reasons
    legacy = None
    if available and force_n is not None and direction is not None:
        legacy = _legacy_document(case, force_n, direction)
    return ScreeningVerdict(
        available=available and legacy is not None,
        reasons=tuple(reasons),
        family=family,
        matching_families=matching_families(case, mesh),
        force_n=force_n,
        direction=tuple(direction) if direction is not None else None,  # type: ignore[arg-type]
        fixed_faces=fixed,
        loaded_faces=loaded,
        legacy_document=legacy,
    )


def _legacy_document(
    case: StaticStructuralCase, force_n: float, direction: Sequence[float]
) -> dict[str, Any]:
    """The `load_case.parse` shape of a screenable case.

    Always a force: a planar pressure has already been resolved into |p|*A
    along -sign(p)*n, which is the traction the sweep applied. Gravity travels
    so `parse` re-applies its own 5% rule with the CAD volume.
    """
    material = case.material
    document: dict[str, Any] = {
        "analysis": "static_structural",
        "fixed": {"faces": list(case.fixed_faces())},
        "load": {
            "faces": list(case.loaded_faces()),
            "magnitude_n": float(force_n),
            "direction": [float(c) for c in direction],
        },
        "material": {
            key: value
            for key, value in (
                ("name", material.name),
                ("poisson_ratio", material.poisson_ratio),
                ("youngs_modulus_mpa", material.youngs_modulus_mpa),
                ("density_kg_m3", material.density_kg_m3),
            )
            if value is not None
        },
    }
    if case.gravity is not None:
        document["gravity"] = {"enabled": True, "g": case.gravity.magnitude_mps2}
    return document


__all__ = [
    "DEFAULT_POISSON_RANGE",
    "DIRECTION_COS_MIN",
    "FAMILY_LENGTH_RANGES_MM",
    "RANGES_MEASURED_ON_SET",
    "UNTRANSFERABLE_FAMILIES",
    "ScreeningVerdict",
    "check",
    "matching_families",
    "mesh_volume_mm3",
]
