"""`fea-structural`: an Ansys-Mechanical-style static structural analysis from the terminal.

    fea-structural PART.step              interview, confirm, solve, report
    fea-structural inspect PART.step      numbered face table and picture only
    fea-structural demo                   writes an L-bracket and solves it

The interview follows the order of a Mechanical setup tree -- geometry, mesh,
material, Fixed Support, loads, gravity -- and behaves like a careful
pre-processor rather than a form: every prompt shows its default, Enter accepts
it, "?" explains the question, and a bad answer gets one line saying why and the
same question again. Nothing is solved until a boxed summary has been confirmed.

Two promises the output keeps:

* solved numbers and the surrogate's screening never share a line; the
  screening block is headed as an estimate and only appears when
  `screenability.check` says the case is one the surrogate was trained on;
* nothing is written next to the STEP file unless --out says so.

Import-light by contract: stdlib (plus the light case/run modules) at the top,
so `--help` works in a base install; the mesher, solver and renderer load when a
run starts. Exit codes: 0 ok, 2 bad input or missing extras, 3 refused,
4 solve failed, 130 interrupted.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from . import structural_run as sr
from .structural_case import (
    MATERIAL_PRESETS,
    REFINEMENT_LEVELS,
    STANDARD_GRAVITY,
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
    precheck_case,
)


EXIT_OK = 0
EXIT_INPUT = 2
EXIT_REFUSED = 3
EXIT_FAILED = 4
EXIT_INTERRUPTED = 130

PROG = "fea-structural"
STEPS = 6
DEMO_FORCE_N = 1000.0

#: Modules a run needs that the base install does not have.
RUN_EXTRAS = ("gmsh", "scipy", "matplotlib")
INSPECT_EXTRAS = ("gmsh",)


class MissingExtras(RuntimeError):
    """A heavy dependency is not installed; the message says how to get it."""


class OutputError(RuntimeError):
    """An output folder or file cannot be written; the message names it."""


#: What the tables use beyond ASCII, and the ASCII it falls back to on a
#: console that cannot encode it (a Windows code page, a redirected stdout):
#: printing '┌' to cp1252 used to crash the run after meshing, as "solve failed".
_ASCII = str.maketrans({
    "┌": "+", "┐": "+", "└": "+", "┘": "+", "─": "-", "│": "|",
    "²": "^2", "³": "^3", "ν": "nu", "ρ": "rho", "—": "--", "–": "-", "·": "-",
    "×": "x", "≈": "~", "≤": "<=", "≥": ">=", "→": "->", "…": "...", "°": " deg",
})
_PROBE = "".join(sorted({key for key in map(chr, _ASCII)}))


def _encodes_all(stream: Any) -> bool:
    """True if `stream` can print every character the tables use."""
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        _PROBE.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def _ascii_safe(print_fn: Callable[..., Any], stream: Any) -> Callable[..., Any]:
    """`print_fn`, translating to ASCII when `stream` cannot encode the tables."""
    if _encodes_all(stream):
        return print_fn

    def safe(*args: Any, **kwargs: Any) -> Any:
        return print_fn(*(str(a).translate(_ASCII) for a in args), **kwargs)

    return safe


# --------------------------------------------------------------------------
# Console: prompts that re-ask, help on "?", defaults in brackets
# --------------------------------------------------------------------------


@dataclass
class Console:
    input_fn: Callable[[str], str]
    print_fn: Callable[..., Any]
    #: results.json escapes non-ASCII when stdout cannot encode it.
    ascii_json: bool = False

    def say(self, text: str = "") -> None:
        self.print_fn(text)

    def ask(
        self,
        question: str,
        parse: Callable[[str], Any],
        *,
        default: str | None = None,
        example: str | None = None,
        help: str | None = None,  # noqa: A002 - reads naturally at call sites
    ) -> Any:
        """Ask until `parse` accepts the answer. EOF/Ctrl-C propagate (exit 130)."""
        if default is not None and default != "":
            hint = f" [{default}]"
        elif example:
            hint = f" (e.g. {example})"
        else:
            hint = ""
        while True:
            raw = self.input_fn(f"{question}{hint}: ")
            if raw is None:
                raise EOFError
            text = str(raw).strip()
            if text == "?":
                for line in (help or "No more help for this one; type an answer or press Enter for the default.").splitlines():
                    self.say("    " + line)
                continue
            if not text:
                if default is None or default == "":
                    self.say("    ! an answer is needed here (type ? for help)")
                    continue
                text = str(default)
            try:
                return parse(text)
            except (CaseError, ValueError) as exc:
                self.say(f"    ! {exc}")

    def yes_no(self, question: str, *, default: bool, help: str | None = None) -> bool:  # noqa: A002
        def parse(text: str) -> bool:
            token = text.strip().lower()
            if token in ("y", "yes"):
                return True
            if token in ("n", "no"):
                return False
            raise CaseError("answer y or n")

        hint = "Y/n" if default else "y/N"
        while True:
            raw = self.input_fn(f"{question} [{hint}]: ")
            if raw is None:
                raise EOFError
            text = str(raw).strip()
            if text == "?":
                self.say("    " + (help or "y = yes, n = no, Enter = the capital letter"))
                continue
            if not text:
                return default
            try:
                return parse(text)
            except CaseError as exc:
                self.say(f"    ! {exc}")


def _terminal_width() -> int:
    return max(60, min(shutil.get_terminal_size((100, 24)).columns, 110))


def box(title: str, rows: Sequence[tuple[str, str]], width: int | None = None) -> list[str]:
    """A boxed summary drawn with box-drawing characters (they render in the
    macOS Terminal and iTerm); long values wrap inside the box."""
    width = width or min(_terminal_width(), 100)
    inner = width - 4
    label_width = max([len(label) for label, _ in rows] + [0]) + 2
    top = f"┌─ {title} " + "─" * max(0, width - len(title) - 5) + "┐"
    lines = [top]
    for label, value in rows:
        wrapped = textwrap.wrap(
            value, width=max(10, inner - label_width), subsequent_indent="  ",
            break_on_hyphens=False,
        ) or [""]
        for index, chunk in enumerate(wrapped):
            prefix = (label if index == 0 else "").ljust(label_width)
            lines.append("│ " + (prefix + chunk).ljust(inner) + " │")
    lines.append("└" + "─" * (width - 2) + "┘")
    return lines


# --------------------------------------------------------------------------
# Face table
# --------------------------------------------------------------------------


def _face_kind(face: Any) -> str:
    kind = getattr(face, "kind", None)
    if kind:
        return str(kind)
    return "plane" if getattr(face, "planar", False) else "curved"


def _normal_text(face: Any) -> str:
    n = [float(c) for c in face.normal]
    length = sum(c * c for c in n) ** 0.5
    if length < 0.2:
        return "(varies: curved)"
    n = [round(c / length, 2) + 0.0 for c in n]       # + 0.0 turns -0.0 into 0.0
    text = f"({n[0]:+.2f}, {n[1]:+.2f}, {n[2]:+.2f})"
    label = sr.axis_label(n, 0.999 if getattr(face, "planar", False) else 0.98)
    if not getattr(face, "planar", False):
        text = "~" + text
    return f"{text} {label}".rstrip()


def _snap(values: Sequence[float], tol: float) -> list[float]:
    """Round-off to 0: OpenCASCADE puts a planar face's centre at -3.9e-05 and a
    bore's bbox edge at -5.5e-14 on a 200 mm part, which reads as a real offset."""
    return [0.0 if abs(float(v)) < tol else float(v) for v in values]


def _bbox_text(face: Any, tol: float = 0.0) -> str:
    lo = getattr(face, "bbox_min", None)
    hi = getattr(face, "bbox_max", None)
    if lo is None or hi is None:
        return ""
    return "  ".join(
        f"{axis} {sr.fmt(a, 4)}..{sr.fmt(b, 4)}"
        for axis, a, b in zip("xyz", _snap(lo, tol), _snap(hi, tol))
    )


def _noise_tolerance(mesh: Any) -> float:
    """1e-6 of the part's size: far below the 4 figures the table prints."""
    size = getattr(mesh, "diagonal_mm", None)
    if not size:
        corners = [c for f in mesh.faces for c in (*getattr(f, "bbox_min", ()), *getattr(f, "bbox_max", ()))]
        size = max((abs(float(c)) for c in corners), default=1.0)
    return 1e-6 * float(size)


def face_table(mesh: Any) -> list[str]:
    """Every face, largest first: the numbers the interview and the picture use."""
    tol = _noise_tolerance(mesh)
    faces = sorted(mesh.faces, key=lambda f: -float(f.area))
    header = (
        f"  {'ID':>4}  {'type':<13} {'area mm²':>10}  {'centre (mm)':<26} "
        f"{'outward normal':<25} bbox (mm)"
    )
    lines = [header, "  " + "-" * (len(header) - 2)]
    for face in faces:
        lines.append(
            f"  {int(face.tag):>4}  {_face_kind(face):<13} {sr.fmt(face.area, 4):>10}  "
            f"{sr.fmt_vec(_snap(face.centroid, tol), 4):<26} {_normal_text(face):<25} "
            f"{_bbox_text(face, tol)}"
        )
    return lines


# --------------------------------------------------------------------------
# Small parsers for the interview and the flags
# --------------------------------------------------------------------------


def _letter(index: int) -> str:
    letters = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(ord("A") + rem) + letters
    return letters


def parse_poisson(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise CaseError(f"{text!r} is not a number; Poisson's ratio is about 0.3 for metals") from None
    if not -1.0 < value < 0.5:
        raise CaseError("Poisson's ratio must lie between -1 and 0.5 (exclusive); metals are about 0.3")
    return value


def parse_strength(text: str) -> float:
    value = parse_pressure(text, role="tensile yield", example="250 (MPa) or 36 ksi")
    if not value > 0:
        raise CaseError("a strength must be positive, e.g. 250 (MPa)")
    return value


def _optional(parse: Callable[[str], Any]) -> Callable[[str], Any]:
    def wrapped(text: str) -> Any:
        if text.strip().lower() in ("none", "-", "no", "n/a"):
            return None
        return parse(text)

    return wrapped


def parse_material_flag(text: str) -> Material:
    """`steel`, `aluminium`, ... or `E_MPA,NU[,DENSITY]` (units allowed on E)."""
    if "," in text:
        parts = [p.strip() for p in text.split(",")]
        if len(parts) not in (2, 3):
            raise CaseError(
                "--material takes a preset name (steel, aluminium, ...) or E,NU or "
                "E,NU,DENSITY, e.g. 70000,0.33 or '70 GPa,0.33,2700'"
            )
        density = parse_density(parts[2]) if len(parts) == 3 else None
        return custom_material(parse_modulus(parts[0]), parse_poisson(parts[1]),
                               density_kg_m3=density)
    try:
        return material_from_key(text)
    except CaseError:
        raise CaseError(
            f"--material: no material preset {text.strip()!r}; presets are "
            f"{', '.join(MATERIAL_PRESETS)} (also steel, aluminium, titanium...). For "
            "anything else give E and Poisson's ratio: --material 70000,0.33, or with a "
            "density for gravity: --material '70 GPa,0.33,2700'"
        ) from None


def _faces_by_tag(mesh: Any) -> dict[int, Any]:
    return {int(f.tag): f for f in mesh.faces}


def _clean_faces(faces: Sequence[Any]) -> list[Any]:
    """Face summaries with round-off removed from the normals (-2e-18 -> 0),
    so the confirmation box reads (0, 0, -1) rather than (-2.322e-18, ...)."""
    from types import SimpleNamespace

    cleaned = []
    for face in faces:
        normal = [float(c) for c in face.normal]
        normal = [0.0 if abs(c) < 1e-9 else c for c in normal]
        cleaned.append(SimpleNamespace(tag=face.tag, area=face.area, planar=face.planar,
                                       normal=tuple(normal)))
    return cleaned


def parse_force_flag(text: str, mesh: Any) -> ForceLoad:
    """`FACES:FX,FY,FZ` (N) or `FACES:MAGNITUDE:DIRECTION`, e.g. 6:0,0,-500 or 6:500:-z."""
    by_tag = _faces_by_tag(mesh)
    parts = text.split(":")
    if len(parts) not in (2, 3) or not all(p.strip() for p in parts):
        raise CaseError(
            f"--force {text!r}: use FACES:FX,FY,FZ or FACES:MAGNITUDE:DIRECTION, "
            "e.g. 6:0,0,-500 or 6:500:-z or '6,7:2 kN:normal'"
        )
    faces = parse_face_list(parts[0], by_tag.keys())
    infos = [by_tag[t] for t in faces]
    if len(parts) == 2:
        vector, _ = parse_force_spec(parts[1], infos)
    else:
        magnitude = parse_force(parts[1])
        direction, _ = parse_direction(parts[2], infos)
        vector = tuple(magnitude * c for c in direction)
    return ForceLoad(faces=faces, vector_n=vector)


def parse_pressure_flag(text: str, mesh: Any) -> PressureLoad:
    """`FACES:MPA`, e.g. 3:1.5 or '3:200 kPa'. Positive pushes into the face."""
    parts = text.split(":")
    if len(parts) != 2 or not all(p.strip() for p in parts):
        raise CaseError(
            f"--pressure {text!r}: use FACES:PRESSURE, e.g. 3:1.5 (MPa) or '3:200 kPa'"
        )
    faces = parse_face_list(parts[0], _faces_by_tag(mesh).keys())
    return PressureLoad(faces=faces, pressure_mpa=parse_pressure(parts[1]))


def mesh_settings_from_args(args: argparse.Namespace, base: MeshSettings | None = None) -> MeshSettings:
    base = base or MeshSettings()
    size = args.element_size
    if size is not None and not (size > 0 and size < float("inf")):
        raise CaseError(f"--element-size must be a positive number of mm, e.g. 3, not {size:g}")
    return MeshSettings(
        element_order=args.order if args.order is not None else base.element_order,
        refinement=args.mesh if args.mesh is not None else base.refinement,
        element_size_mm=(
            args.element_size if args.element_size is not None
            else (None if args.mesh is not None else base.element_size_mm)
        ),
    )


def parse_mesh_choice(text: str, current: MeshSettings) -> MeshSettings:
    token = text.strip().lower()
    if token in REFINEMENT_LEVELS:
        return MeshSettings(element_order=current.element_order, refinement=token)
    number = token[:-2].strip() if token.endswith("mm") else token
    try:
        size = float(number)
    except ValueError:
        raise CaseError(
            f"{text!r}: type coarse, medium or fine, or an element size in mm like 3 mm"
        ) from None
    if not (size > 0 and size < float("inf")):
        raise CaseError("the element size must be a positive number of mm, e.g. 3 mm")
    return MeshSettings(element_order=current.element_order,
                        refinement=current.refinement, element_size_mm=size)


def _mesh_choice_text(settings: MeshSettings) -> str:
    if settings.element_size_mm is not None:
        return f"{sr.fmt(settings.element_size_mm, 3)} mm"
    return settings.refinement


def screenable_families() -> list[str]:
    """Families the surrogate can screen here: the face presets, less those
    whose weights do not transfer to other parts (the spring)."""
    from .face_selectors import FAMILY_FACE_PRESETS
    from .screenability import UNTRANSFERABLE_FAMILIES

    return sorted(set(FAMILY_FACE_PRESETS) - set(UNTRANSFERABLE_FAMILIES))


def parse_family(text: str) -> str | None:
    from .screenability import UNTRANSFERABLE_FAMILIES

    token = text.strip().lower().replace("-", "_").replace(" ", "_")
    if token in ("none", "no", "-", "skip"):
        return None
    names = screenable_families()
    if token in names:
        return token
    if token in UNTRANSFERABLE_FAMILIES:
        raise CaseError(
            f"the {token} family cannot be screened: its trained weights do not "
            f"transfer to other parts; choose one of {', '.join(names)}, or none"
        )
    raise CaseError(
        f"no surrogate family {text.strip()!r}; choose one of {', '.join(names)}, or none"
    )


def _load_case_file(path: str) -> dict[str, Any]:
    file = Path(path).expanduser()
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as exc:
        raise CaseError(f"cannot read the case file {file}: {exc.strerror or exc}") from None
    try:
        if file.suffix.lower() == ".toml":
            import tomllib

            return tomllib.loads(text)
        document = json.loads(text)
    except ValueError as exc:
        raise CaseError(f"{file} is not valid {'TOML' if file.suffix.lower() == '.toml' else 'JSON'}: {exc}") from None
    if not isinstance(document, dict):
        raise CaseError(f"{file} must hold one JSON object (a StaticStructuralCase/v1 case)")
    return document


def _mesh_settings_of_document(document: dict[str, Any]) -> MeshSettings | None:
    spec = document.get("mesh")
    if not isinstance(spec, dict):
        return None
    order = spec.get("element_order", 2)
    if isinstance(order, float) and order.is_integer():
        order = int(order)
    return MeshSettings(
        element_order=order,
        refinement=spec.get("refinement") or "medium",
        element_size_mm=spec.get("element_size_mm"),
    )


# --------------------------------------------------------------------------
# The interview
# --------------------------------------------------------------------------

HELP = {
    "material": (
        "Pick a number from the list, a name (steel, aluminium, titanium...), or c\n"
        "for a custom material. Values are typical, not a specification: a design\n"
        "decision needs the supplier's data sheet. E sets deformation only; with\n"
        "these supports and loads stress does not depend on E."
    ),
    "fixed": (
        "Faces held in place: all three translations are zero on every node of\n"
        "them (Ansys 'Fixed Support'). Use the numbers from the face table and the\n"
        "picture: 1   or   1,8   or   1 8   or   2-4."
    ),
    "kind": (
        "f = Force: a total force in N, spread evenly over the faces' area.\n"
        "p = Pressure: MPa acting normal to the surface at every point (positive\n"
        "pushes in). On a curved face such as a bore, use a pressure."
    ),
    "force": (
        "Magnitude and direction:   500 N -z    2 kN +x    500 normal (into a flat face)\n"
        "or components in N:        0,0,-500    [0 0 -0.5] kN\n"
        "Units: N, kN, MN, lbf. Directions: x -x +y ... or three numbers, normal/in\n"
        "(push into the face), out (pull). The sign of a force lives in its direction.\n"
        "p makes this load a pressure instead; back returns to the force/pressure question."
    ),
    "pressure": (
        "MPa (= N/mm²); positive pushes into the face, negative pulls (suction).\n"
        "Units: MPa, kPa, Pa, bar, psi, N/mm2. Example: 1.5   or   200 kPa.\n"
        "f makes this load a force instead; back returns to the force/pressure question."
    ),
    "gravity": (
        "Adds self-weight (density x volume x g) as a body load. Usually negligible\n"
        "for small machined parts under kN loads; matters for large or light-loaded parts."
    ),
    "gravity_direction": (
        "Which way is down in the CAD model: -z is the usual convention, but check\n"
        "how the part was modelled. Axis (-z, +y...) or three numbers."
    ),
    "family": (
        "The surrogate screens only a family's own trained load case (same held and\n"
        "loaded faces, same axis, size and Poisson ratio in range). Name the family\n"
        "the part belongs to, or none. The full solve runs either way."
    ),
    "mesh": (
        "coarse / medium / fine pick an element size from the part's thinnest wall;\n"
        "or type a size like 3 mm. Finer = slower and closer to converged."
    ),
}


@dataclass(frozen=True)
class InterviewResult:
    case: StaticStructuralCase
    family: str | None
    mesh: Any
    #: The person accepted a mesh above the large-DOF threshold at the mesh question.
    allow_large: bool = False


@dataclass(frozen=True)
class Defaults:
    """Answers already given as flags; shown as the defaults in the interview."""

    material: Material | None = None
    fixed: tuple[int, ...] | None = None
    loads: tuple[Any, ...] = ()
    gravity: bool = False
    family: str | None = None


def _material_menu() -> list[str]:
    lines = ["Material  (typical values; use your supplier's data sheet for design decisions)"]
    for index, m in enumerate(MATERIAL_PRESETS.values(), 1):
        rho = f"ρ {sr.fmt(m.density_kg_m3)} kg/m³" if m.density_kg_m3 else ""
        strength = (
            f"yield {sr.fmt(m.tensile_yield_mpa)} MPa" if m.tensile_yield_mpa
            else f"no yield (ultimate {sr.fmt(m.tensile_ultimate_mpa)} MPa)"
            if m.tensile_ultimate_mpa else "no yield given"
        )
        lines.append(
            f"  {index:>2}  {m.name:<18} E {sr.fmt(m.youngs_modulus_mpa):>6} MPa  "
            f"ν {m.poisson_ratio:.2f}  {rho:<17} {strength}"
        )
    lines.append("   c  custom: E and ν, density and yield optional")
    return lines


def _ask_material(console: Console, given: Material | None) -> Material:
    presets = list(MATERIAL_PRESETS.values())
    for line in _material_menu():
        console.say(line)
    default = "1"
    if given is not None:
        default = (str(presets.index(given) + 1) if given in presets else "given")
        if default == "given":
            console.say(f"  given: {given.name}, E {sr.fmt(given.youngs_modulus_mpa)} MPa, "
                        f"ν {sr.fmt(given.poisson_ratio)}")

    def parse(text: str) -> Material | str:
        token = text.strip().lower()
        if token == "given" and given is not None:
            return given
        if token in ("c", "custom"):
            return "custom"
        if token.isdigit():
            index = int(token)
            if 1 <= index <= len(presets):
                return presets[index - 1]
            raise CaseError(f"pick 1-{len(presets)} from the list, a material name, or c for custom")
        try:
            return material_from_key(text)
        except CaseError:
            raise CaseError(
                f"no material {text.strip()!r}: pick 1-{len(presets)} from the list, a name "
                "such as steel or aluminium, or c to type in a custom material"
            ) from None

    choice = console.ask("Material", parse, default=default, help=HELP["material"])
    if isinstance(choice, Material):
        console.say(f"    -> {choice.name}: E {sr.fmt(choice.youngs_modulus_mpa)} MPa, "
                    f"ν {sr.fmt(choice.poisson_ratio)}")
        return choice

    modulus = console.ask(
        "  Young's modulus E (MPa, or with a unit)", parse_modulus, example="70 GPa or 70000",
        help="Bare numbers are MPa: steel is 200000 MPa = 200 GPa. Units: MPa, GPa, psi, ksi.",
    )
    console.say(f"    -> E = {sr.fmt(modulus)} MPa")
    if modulus < 1000:
        console.say(f"    note: that is {sr.fmt(modulus)} MPa; type a unit (e.g. 200 GPa) if you meant GPa")
    nu = console.ask("  Poisson's ratio ν", parse_poisson, default="0.3",
                     help="About 0.3 for metals, 0.33 aluminium, 0.4+ for polymers; must be < 0.5.")
    if not 0.27 <= nu <= 0.42:
        console.say("    note: outside 0.27-0.42 the surrogate will not screen; the solve is unaffected")
    density = console.ask("  Density (kg/m³; needed only for gravity)", _optional(parse_density),
                          default="none", help="e.g. 7850 steel, 2700 aluminium, or 7.85 g/cm3.")
    strength = console.ask("  Tensile yield (MPa; none = no safety factor)", _optional(parse_strength),
                           default="none", help="e.g. 250 for structural steel. Used only for yield / stress.")
    name = console.ask("  Name", lambda t: t.strip(), default="Custom material")
    return custom_material(modulus, nu, name=name, density_kg_m3=density,
                           tensile_yield_mpa=strength)


def _area(faces: Sequence[int], by_tag: dict[int, Any]) -> float:
    return float(sum(float(by_tag[t].area) for t in faces))


def _echo_load(console: Console, load: Any, by_tag: dict[int, Any]) -> None:
    area = _area(load.faces, by_tag)
    where = sr.fmt_faces(load.faces)
    if isinstance(load, ForceLoad):
        label = sr.axis_label(load.direction)
        console.say(
            f"    -> {sr.fmt(load.magnitude_n)} N along {sr.fmt_vec(load.direction, 3)}"
            f"{' ' + label if label else ''} on {where}: {sr.fmt(area)} mm², "
            f"traction {sr.fmt(load.magnitude_n / area)} MPa"
        )
        return
    sense = "into the surface" if load.pressure_mpa > 0 else "suction, out of the surface"
    text = f"    -> {sr.fmt(load.pressure_mpa)} MPa normal to {where} ({sense}), {sr.fmt(area)} mm²"
    try:
        direction, _ = parse_direction("into" if load.pressure_mpa > 0 else "out",
                                       [by_tag[t] for t in load.faces])
        text += (f"; equivalent {sr.fmt(abs(load.pressure_mpa) * area)} N along "
                 f"{sr.fmt_vec(direction, 3)}")
    except CaseError:
        text += "; curved or non-parallel faces, so the net force depends on the shape"
    console.say(text)


#: Typed at a load's faces/force/pressure prompt: back to "force or pressure?".
_BACK_WORDS = frozenset({"back", "b"})
_BACK = object()
_SWITCH = object()


def _ask_loads(console: Console, mesh: Any, fixed: tuple[int, ...]) -> list[Any]:
    """Loads A, B, ...: each kind, faces, then size. 'back' at any of a load's
    prompts returns to the kind question, the other kind's letter switches
    (p at the force prompt, f at the pressure one), and 'undo' after a load
    removes it -- a curved face answered with a force must not trap the user."""
    by_tag = _faces_by_tag(mesh)
    loads: list[Any] = []

    def parse_kind(text: str) -> str:
        token = text.strip().lower()
        if token in ("f", "force"):
            return "force"
        if token in ("p", "pressure"):
            return "pressure"
        raise CaseError("type f for a force or p for a pressure")

    def parse_more(text: str) -> str:
        token = text.strip().lower()
        if token in ("y", "yes"):
            return "more"
        if token in ("n", "no"):
            return "done"
        if token in ("u", "undo"):
            return "undo"
        raise CaseError("answer y, n, or undo to remove the last load")

    while True:
        letter = _letter(1 + len(loads))
        kind = console.ask(f"Load {letter}: [f]orce or [p]ressure", parse_kind, default="f",
                           help=HELP["kind"])

        def parse_faces(text: str) -> Any:
            if text.strip().lower() in _BACK_WORDS:
                return _BACK
            tags = parse_face_list(text, by_tag.keys())
            if set(tags) <= set(fixed):
                raise CaseError(
                    f"{sr.fmt_faces(tags)} {'is' if len(tags) == 1 else 'are all'} fixed, so a "
                    "load there goes straight into the support; pick faces free to move"
                )
            return tags

        faces = console.ask(f"  {letter} faces", parse_faces, example="6 or 6,7",
                            help=HELP["fixed"].replace("held in place", "carrying this load")
                            + "\nback returns to the force/pressure question.")
        if faces is _BACK:
            continue
        shared = sorted(set(faces) & set(fixed))
        if shared:
            console.say(f"    note: {sr.fmt_faces(shared)} also fixed; that share of the load "
                        "goes straight into the support")
        infos = [by_tag[t] for t in faces]

        def parse_force_answer(text: str) -> Any:
            token = text.strip().lower()
            if token in _BACK_WORDS:
                return _BACK
            if token in ("p", "pressure"):
                return _SWITCH
            try:
                return parse_force_spec(text, infos)
            except CaseError as exc:
                if "pressure load instead" in str(exc):
                    raise CaseError(f"{exc}. Type p to make this load a pressure, or back") from None
                raise

        def parse_pressure_answer(text: str) -> Any:
            token = text.strip().lower()
            if token in _BACK_WORDS:
                return _BACK
            if token in ("f", "force"):
                return _SWITCH
            return parse_pressure(text)

        load: Any = None
        while load is None:
            if kind == "force":
                answer = console.ask(
                    f"  {letter} force: magnitude and direction, or components in N",
                    parse_force_answer,
                    example="500 N -z, 2 kN normal, or 0,0,-500; p = a pressure instead",
                    help=HELP["force"],
                )
            else:
                answer = console.ask(
                    f"  {letter} pressure (MPa, + pushes into the face)", parse_pressure_answer,
                    example="1.5 or 200 kPa; f = a force instead", help=HELP["pressure"],
                )
            if answer is _BACK:
                break
            if answer is _SWITCH:
                kind = "pressure" if kind == "force" else "force"
                continue
            load = (ForceLoad(faces=faces, vector_n=answer[0]) if kind == "force"
                    else PressureLoad(faces=faces, pressure_mpa=answer))
        if load is None:
            continue
        _echo_load(console, load, by_tag)
        loads.append(load)
        while True:
            choice = console.ask(f"Add another load? (y, n, or undo to remove {letter})",
                                 parse_more, default="n")
            if choice != "undo":
                break
            removed = loads.pop()
            console.say(f"    -> removed {letter} ({type(removed).__name__.replace('Load', '').lower()} "
                        f"on {sr.fmt_faces(removed.faces)})")
            if not loads:
                choice = "more"         # a case needs at least one load here
                break
            letter = _letter(len(loads))
        if choice == "done":
            return loads


def _gravity_direction(text: str) -> tuple[float, float, float]:
    vector, _ = parse_direction(text)
    return vector


def interview(
    console: Console,
    mesh: Any,
    *,
    settings: MeshSettings,
    remesh: Callable[[MeshSettings], Any],
    screen: bool = True,
    defaults: Defaults = Defaults(),
) -> InterviewResult:
    """Material, Fixed Support, loads, gravity, surrogate family, mesh: in Ansys order."""
    by_tag = _faces_by_tag(mesh)
    known = list(by_tag)

    console.say("")
    material = _ask_material(console, defaults.material)

    console.say("")
    fixed = console.ask(
        "A  Fixed Support faces", lambda t: parse_face_list(t, known),
        default=",".join(map(str, defaults.fixed)) if defaults.fixed else None,
        example="1 or 1,8 or 2-4", help=HELP["fixed"],
    )
    console.say(f"    -> A  Fixed Support on {sr.fmt_faces(fixed)}: "
                f"{sr.fmt(_area(fixed, by_tag))} mm²")

    console.say("")
    if defaults.loads:
        loads = list(defaults.loads)
        console.say("Loads from the command line:")
        for load in loads:
            _echo_load(console, load, by_tag)
    else:
        loads = _ask_loads(console, mesh, fixed)

    console.say("")
    gravity = None
    if console.yes_no("Include gravity (self-weight)?", default=defaults.gravity, help=HELP["gravity"]):
        direction = console.ask(
            "  Gravity direction (proposed: -z, check it is down for this model)",
            _gravity_direction, default="-z", help=HELP["gravity_direction"],
        )
        gravity = Gravity(acceleration_mps2=tuple(STANDARD_GRAVITY * c for c in direction))
        if material.density_kg_m3 is None:
            density = console.ask(
                "  Density (kg/m³) - gravity needs it; none = no gravity", _optional(parse_density),
                example="7850", help="e.g. 7850 steel, 2700 aluminium, or 7.85 g/cm3; "
                                     "none drops gravity.")
            if density is None:
                gravity = None
                console.say("    -> no gravity (no density given)")
            else:
                material = dataclasses.replace(material, density_kg_m3=density)

    case = StaticStructuralCase(
        supports=(FixedSupport(faces=fixed),), loads=tuple(loads), material=material,
        gravity=gravity, mesh=settings,
    )

    family = defaults.family
    if screen:
        console.say("")
        from .screenability import matching_families

        matches = matching_families(case, mesh)
        default = defaults.family or (matches[0] if len(matches) == 1 else "none")
        if len(matches) == 1 and not defaults.family:
            console.say(f"  This load case is the {matches[0]!r} family's trained load case.")
        family = console.ask(
            f"Surrogate family for screening ({', '.join(screenable_families())}, none)",
            parse_family, default=default, help=HELP["family"],
        )

    console.say("")
    mesh, chosen, allow_large = _ask_mesh(console, mesh, settings, remesh)
    if chosen != settings:
        case = dataclasses.replace(case, mesh=chosen)
    return InterviewResult(case=case, family=family, mesh=mesh, allow_large=allow_large)


def _mesh_estimate(mesh: Any, chosen: MeshSettings) -> tuple[float, int, int]:
    """(element size mm, elements, DOF) that `chosen` would give on this part,
    scaled from the mesh in hand (count ~ 1/size^3) -- without meshing."""
    import numpy as np

    from .structural_mesh import element_size_for

    if chosen.element_size_mm is not None:
        size = float(chosen.element_size_mm)
    else:
        extent = np.ptp(np.asarray(mesh.nodes, dtype=float), axis=0)
        area = getattr(mesh, "area_mm2", None) or sum(float(f.area) for f in mesh.faces)
        size = float(element_size_for(extent, getattr(mesh, "volume_mm3", None), area,
                                      chosen.refinement))
    ratio = (float(mesh.element_size_mm) / size) ** 3
    return size, int(round(int(mesh.element_count) * ratio)), int(round(int(mesh.dof_count) * ratio))


def _ask_mesh(console: Console, mesh: Any, settings: MeshSettings,
              remesh: Callable[[MeshSettings], Any]) -> tuple[Any, MeshSettings, bool]:
    """The mesh question, checked where it is asked: a size no solver could
    factor is refused and asked again, a solve above the large-DOF threshold is
    priced and confirmed here (not after "Proceed?"), and a size gmsh cannot
    mesh keeps the mesh in hand. Returns (mesh, settings, allow_large)."""
    from .structural_mesh import MAX_ESTIMATED_ELEMENTS, AnalysisMeshError

    threshold = sr.large_dof_threshold()

    def parse(text: str) -> MeshSettings:
        chosen = parse_mesh_choice(text, settings)
        if chosen != settings:
            size, elements, _ = _mesh_estimate(mesh, chosen)
            if elements > MAX_ESTIMATED_ELEMENTS:
                raise CaseError(
                    f"an element size of {sr.fmt(size, 3)} mm would give roughly "
                    f"{elements:,} elements, far more than a direct solver can factor; "
                    "pick a larger size or a refinement level"
                )
        return chosen

    while True:
        seconds = sr.estimate_seconds(mesh)
        now = (f"now {int(mesh.node_count):,} nodes, {int(mesh.dof_count):,} DOF"
               + (f", about {sr.fmt(seconds, 2)} s" if seconds else ""))
        chosen = console.ask(
            f"Mesh: coarse, medium, fine, or a size like 3 mm ({now})", parse,
            default=_mesh_choice_text(settings), help=HELP["mesh"],
        )
        allow_large = int(mesh.dof_count) > threshold and chosen == settings
        if chosen == settings:
            if allow_large and not _confirm_large(console, int(mesh.dof_count), chosen):
                continue
            return mesh, settings, allow_large
        _, _, dof = _mesh_estimate(mesh, chosen)
        if dof > threshold:
            if not _confirm_large(console, dof, chosen, estimated=True):
                continue
            allow_large = True
        console.say(f"    re-meshing ({chosen.describe()}) ...")
        started = time.perf_counter()
        try:
            fresh = remesh(chosen)
        except AnalysisMeshError as exc:
            console.say(f"    ! could not mesh that: {exc}")
            continue
        console.say(f"    -> {int(fresh.node_count):,} nodes, {int(fresh.element_count):,} elements, "
                    f"{int(fresh.dof_count):,} DOF ({time.perf_counter() - started:.1f} s)")
        if int(fresh.dof_count) > threshold and not allow_large:
            if not _confirm_large(console, int(fresh.dof_count), chosen):
                continue
            allow_large = True
        return fresh, chosen, allow_large


def _confirm_large(console: Console, dof: int, chosen: MeshSettings, *, estimated: bool = False) -> bool:
    from types import SimpleNamespace

    seconds = sr.estimate_seconds(SimpleNamespace(dof_count=dof))
    about = "about " if estimated else ""
    return console.yes_no(
        f"    {_mesh_choice_text(chosen)}: {about}{dof:,} DOF, above {sr.large_dof_threshold():,}; "
        f"solve about {sr.fmt(seconds, 2)} s and several GB of memory. Use it? "
        "(n = pick another mesh)", default=False,
    )


# --------------------------------------------------------------------------
# Shared run steps
# --------------------------------------------------------------------------


def _missing_extras(modules: Sequence[str]) -> list[str]:
    import importlib.util

    return [m for m in modules if importlib.util.find_spec(m) is None]


def _require(modules: Sequence[str]) -> None:
    missing = _missing_extras(modules)
    if missing:
        raise MissingExtras(
            f"missing {', '.join(missing)}: the analysis needs the inference and render "
            f"extras. Install them with: {sr.INSTALL_HINT}"
        )


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _out_dir(path: Path, given: str | None) -> Path:
    if given:
        return Path(given).expanduser().resolve()
    base = sr.default_out_dir(path, _stamp())
    candidate, n = base, 2
    while candidate.exists():
        candidate = base.with_name(f"{base.name}-{n}")
        n += 1
    return candidate


def _may_open(no_open: bool) -> bool:
    return (
        not no_open
        and not os.environ.get("CI")
        and sys.platform == "darwin"
        and sys.stdout.isatty()
    )


def _open(path: Path | None, no_open: bool) -> None:
    if path is None or not _may_open(no_open):
        return
    try:
        subprocess.Popen(["open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass


class _Clock:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    def __call__(self) -> float:
        return time.perf_counter() - self.start


def _step(console: Console, clock: _Clock, n: int | None, message: str) -> None:
    """'  12.3 s  [4/6] Solving ...': the clock is time since the solve began,
    so a long stage shows as the gap to the next line."""
    tag = f"[{n}/{STEPS}]" if n is not None else "     "
    console.say(f"  {clock():6.1f} s  {tag} {message}")


def _read_and_mesh(console: Console, path: Path, settings: MeshSettings, clock: _Clock,
                   steps: int = STEPS) -> tuple[Any, Any]:
    console.say(f"  [1/{steps}] Reading geometry: {path.name}")
    solid = sr.inspect_part(path)
    extent = [b - a for a, b in zip(solid.bbox_min, solid.bbox_max)]
    console.say(
        f"        {len(solid.faces)} faces, bounding box "
        f"{' x '.join(sr.fmt(e, 4) for e in extent)} mm, volume {sr.fmt(solid.volume, 4)} mm³"
        f"{'' if solid.solid_count == 1 else f', {solid.solid_count} solids'}"
        f"  ({clock():.1f} s)"
    )
    if solid.healed:
        console.say("        note: the file held no solid; its surfaces were sewn into one "
                    "(a repair within a tolerance, not an exact read)")
    console.say(f"  [2/{steps}] Meshing: {settings.describe()}")
    mesh = sr.mesh_part(path, settings)
    seconds = sr.estimate_seconds(mesh)
    console.say(
        f"        {int(mesh.node_count):,} nodes, {int(mesh.element_count):,} elements, "
        f"{int(mesh.dof_count):,} DOF, element size {sr.fmt(mesh.element_size_mm, 3)} mm"
        + (f"; solve about {sr.fmt(seconds, 2)} s" if seconds else "")
        + f"  ({clock():.1f} s)"
    )
    if getattr(mesh, "straight_sided", False):
        console.say("        note: curved elements were invalid here; straight-sided ones are used")
    return solid, mesh


def _show_faces(console: Console, mesh: Any, out_dir: Path, no_open: bool) -> Path | None:
    console.say("")
    console.say(f"Faces of the part ({len(mesh.faces)}), largest first. "
                "Use these numbers for supports and loads.")
    for line in face_table(mesh):
        console.say(line)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        image = sr.render_faces_image(mesh, out_dir / "faces.png")
    except Exception as exc:
        console.say(f"  (face picture not drawn: {type(exc).__name__}: {exc})")
        return None
    console.say(f"  Picture with the numbers: {image}")
    _open(image, no_open)
    return image


def _screening_rows(case: StaticStructuralCase, mesh: Any, family: str | None, screen: bool,
                    files: tuple[Any, Any, Any]) -> list[tuple[str, str]]:
    if not screen:
        return [("Screening", "off (--no-screen); the solve runs as usual")]
    verdict = sr.screening_verdict(case, mesh, family, files[1])
    if verdict.available:
        text = f"yes: this is the {verdict.family!r} family's trained load case"
        if files[0] is None or files[1] is None:
            text += "; but no trained surrogate was found (--weights/--calibration), so it will be skipped"
        return [("Screening", text)]
    rows = [("Screening", "not screenable (the solve still runs):")]
    rows += [("", "- " + reason) for reason in verdict.reasons]
    return rows


def _confirm_rows(path: Path, solid: Any, mesh: Any, case: StaticStructuralCase,
                  family: str | None, screen: bool, out_dir: Path,
                  files: tuple[Any, Any, Any]) -> list[tuple[str, str]]:
    extent = [b - a for a, b in zip(solid.bbox_min, solid.bbox_max)]
    rows: list[tuple[str, str]] = [
        ("Part", f"{path.name} - 1 solid, {len(mesh.faces)} faces, "
                 f"{' x '.join(sr.fmt(e, 4) for e in extent)} mm"
                 + (" (surfaces sewn)" if solid.healed else "")),
        ("Units", "mm, N, MPa, kg/m³, s"),
    ]
    describe = [line for line in case.describe(_clean_faces(mesh.faces))
                if not line.startswith("Mesh:")]
    for index, line in enumerate(describe):
        rows.append(("Setup" if index == 0 else "", line))
    seconds = sr.estimate_seconds(mesh)
    rows.append((
        "Mesh",
        f"{case.mesh.describe()}: {int(mesh.node_count):,} nodes, "
        f"{int(mesh.element_count):,} elements, {int(mesh.dof_count):,} DOF"
        + (f", solve about {sr.fmt(seconds, 2)} s" if seconds else ""),
    ))
    rows += _screening_rows(case, mesh, family, screen, files)
    rows.append(("Output", str(out_dir)))
    return rows


def _solve_and_report(
    console: Console,
    *,
    path: Path,
    mesh: Any,
    case: StaticStructuralCase,
    family: str | None,
    out_dir: Path,
    args: argparse.Namespace,
    allow_large: bool,
    json_print: Callable[..., Any] | None,
) -> int:
    clock = _Clock()
    console.say("")
    console.say("Solving")

    def progress(stage: str, message: str) -> None:
        number = {"screen": 3, "solve": 4, "converge": 4, "render": 5, "export": 6}.get(stage)
        if number is not None:
            _step(console, clock, number, message)
        elif stage.endswith("_detail") or stage == "done":
            _step(console, clock, None, message)

    run = sr.run_static_structural(
        path, case, out_dir=out_dir, family=family, screen=not args.no_screen,
        weights=args.weights, calibration=args.calibration, neighbours=args.neighbours,
        converge=args.converge, allow_large=allow_large, progress=progress, mesh=mesh,
        crosscheck=bool(getattr(args, "crosscheck", False)),
    )
    console.say("")
    for line in sr.format_summary(run).splitlines():
        console.say(line)
    board = run.files.get("board_image") if hasattr(run, "files") else None
    _open(board, args.no_open)
    if json_print is not None:
        json_print(json.dumps(run.as_document(), indent=2, ensure_ascii=console.ascii_json))
    return EXIT_OK


def _large_ok(console: Console, mesh: Any, args: argparse.Namespace, interactive: bool) -> bool:
    """Ask before a solve above the large-DOF threshold; refuse with --yes."""
    if args.allow_large or int(mesh.dof_count) <= sr.large_dof_threshold():
        return bool(args.allow_large)
    seconds = sr.estimate_seconds(mesh)
    if not interactive:
        return False        # the run refuses with the reason and exit 3
    return console.yes_no(
        f"This mesh has {int(mesh.dof_count):,} DOF (solve about {sr.fmt(seconds, 2)} s, "
        "a lot of memory). Solve anyway?", default=False,
    )


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def _run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Static structural analysis of one STEP part with 10-node tetrahedra: "
            "numbered faces, an interview (material, fixed supports, forces, pressures, "
            "gravity), then Equivalent (von-Mises) Stress and Total Deformation images, "
            "results.json, a ParaView .vtu and a results table."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(f"""\
            other commands:
              {PROG} inspect PART.step [--out DIR] [--no-open]   face table and picture only
              {PROG} demo [--out DIR] [--no-open] [--no-screen]   solve a generated L-bracket

            examples:
              {PROG} part.step                                     interview
              {PROG} part.step --fixed 1 --force 6:500:-z --material steel --yes
              {PROG} part.step --case case.json --yes --json > results.json

            exit codes: 0 ok, 2 bad input or missing extras, 3 refused, 4 solve failed,
            130 interrupted. Output goes to ./fea_results/<part>-<time>/ unless --out.
        """),
    )
    parser.add_argument("part", metavar="PART.step", help="the part (STEP, IGES or BREP), in mm")
    case = parser.add_argument_group("case (skip the interview)")
    case.add_argument("--case", metavar="FILE.json", help="a StaticStructuralCase/v1 file (JSON or TOML)")
    case.add_argument("--material", metavar="KEY|E_MPA,NU", help="steel, aluminium, ... or 70000,0.33")
    case.add_argument("--fixed", metavar="FACES", help="fixed support faces, e.g. 1 or 1,8 or 2-4")
    case.add_argument("--force", action="append", default=[], metavar="FACES:FX,FY,FZ|FACES:MAG:DIR",
                      help="a force in N, e.g. 6:0,0,-500 or 6:500:-z (repeatable)")
    case.add_argument("--pressure", action="append", default=[], metavar="FACES:MPA",
                      help="a pressure, + pushes into the face, e.g. 3:1.5 (repeatable)")
    case.add_argument("--gravity", action="store_true", help="add standard earth gravity along -Z")
    case.add_argument("--family", metavar="NAME", help="surrogate family for screening (bracket, flange, ...)")
    mesh = parser.add_argument_group("mesh")
    mesh.add_argument("--mesh", choices=REFINEMENT_LEVELS,
                      help="refinement (default medium; fine has about 5x the nodes of "
                           "medium and takes about 20x as long to solve)")
    mesh.add_argument("--element-size", type=float, metavar="MM", help="element size, overrides --mesh")
    mesh.add_argument("--order", type=int, choices=(1, 2), help="2 = 10-node tets (default), 1 = 4-node")
    run = parser.add_argument_group("run")
    run.add_argument("--out", metavar="DIR", help="output folder (default ./fea_results/<part>-<time>/)")
    run.add_argument("--no-screen", action="store_true", help="do not consult the surrogate")
    run.add_argument("--crosscheck", action="store_true",
                     help="also solve a screened case with the 4-node solver the surrogate's "
                          "labels came from (about 4 s on the sample bracket)")
    run.add_argument("--converge", action="store_true", help="re-solve at 0.67x element size and report the change")
    run.add_argument("--yes", action="store_true", help="no questions: needs a complete case from flags or --case")
    run.add_argument("--no-open", action="store_true", help="do not open images")
    run.add_argument("--save-case", metavar="FILE", help="also save the case to FILE (JSON)")
    run.add_argument("--json", action="store_true", help="print results.json to stdout (messages go to stderr)")
    run.add_argument("--allow-large", action="store_true", help="solve above the large-DOF threshold without asking")
    surrogate = parser.add_argument_group("surrogate files (default: env vars, then <repo>/models)")
    surrogate.add_argument("--weights", metavar="P")
    surrogate.add_argument("--calibration", metavar="P")
    surrogate.add_argument("--neighbours", metavar="P")
    return parser


def _inspect_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"{PROG} inspect", description="Print the numbered face table and draw faces.png.",
    )
    parser.add_argument("part", metavar="PART.step")
    parser.add_argument("--out", metavar="DIR")
    parser.add_argument("--no-open", action="store_true")
    return parser


def _demo_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"{PROG} demo",
        description=(
            "Write a demo L-bracket STEP into the output folder and solve the bracket "
            "family's trained load case on it: fixed at the x-min face, "
            f"{sr.fmt(DEMO_FORCE_N)} N along -Z on the x-max face, structural steel."
        ),
    )
    parser.add_argument("--out", metavar="DIR")
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--no-screen", action="store_true")
    return parser


def _case_flags_given(args: argparse.Namespace) -> bool:
    return bool(args.fixed or args.force or args.pressure or args.material or args.gravity)


def _check_output_dir(folder: Path) -> None:
    """The results folder can be made and written: checked before any work, so a
    bad --out costs a moment, not a mesh, an interview and a solve."""
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputError(f"cannot write the results to {folder}: {exc.strerror or exc}; "
                          "pass --out DIR") from None
    if not os.access(folder, os.W_OK | os.X_OK):
        raise OutputError(f"cannot write the results to {folder}: permission denied; "
                          "pass --out DIR")


def _check_case_target(target: Path) -> None:
    """--save-case FILE can be written (its folder is made if missing)."""
    if target.is_dir():
        raise OutputError(f"--save-case {target} is a folder; give a file name, "
                          f"e.g. {target / 'case.json'}")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputError(f"cannot write the case to {target}: {exc.strerror or exc}") from None
    if not os.access(target.parent, os.W_OK | os.X_OK) or (
            target.exists() and not os.access(target, os.W_OK)):
        raise OutputError(f"cannot write the case to {target}: permission denied")


def _write_case(case: StaticStructuralCase, target: Path) -> Path:
    target.write_text(json.dumps(case.as_document(), indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    return target


def _command_run(console: Console, args: argparse.Namespace, json_print: Callable[..., Any] | None) -> int:
    from .cad_intake import check_readable

    path = check_readable(Path(args.part).expanduser())
    if args.case and _case_flags_given(args):
        raise CaseError(
            "use either --case FILE or the case flags (--fixed, --force, --pressure, "
            "--material, --gravity), not both"
        )
    # Everything that does not need the part's faces is checked before the
    # geometry is read: a typo should cost a moment, not a mesh.
    family = parse_family(args.family) if args.family else None
    document = _load_case_file(args.case) if args.case else None
    if document is not None:
        precheck_case(document)
    base = _mesh_settings_of_document(document) if document else None
    settings = mesh_settings_from_args(args, base)
    material = parse_material_flag(args.material) if args.material else None
    if args.gravity and material is not None and not (material.density_kg_m3 or 0) > 0:
        raise CaseError(
            "--gravity needs the material's density: give it as a third value, e.g. "
            "--material '70 GPa,0.33,2700' (kg/m³), or use a preset such as steel"
        )
    interactive = not args.yes
    if not interactive and document is None:
        missing = []
        if not args.fixed:
            missing.append("--fixed FACES")
        if not (args.force or args.pressure or args.gravity):
            missing.append("--force or --pressure (or --gravity)")
        if missing:
            raise CaseError(
                f"--yes needs a complete case: add {' and '.join(missing)}, or give "
                "--case FILE.json; or drop --yes to be asked"
            )
    _require(RUN_EXTRAS)
    out_dir = _out_dir(path, args.out)
    _check_output_dir(out_dir)
    save_target = Path(args.save_case).expanduser() if args.save_case else None
    if save_target is not None:
        _check_case_target(save_target)

    clock = _Clock()
    console.say(f"{PROG}: static structural analysis of {path}")
    solid, mesh = _read_and_mesh(console, path, settings, clock)
    _show_faces(console, mesh, out_dir, args.no_open)

    case: StaticStructuralCase | None = None
    asked_large = False
    interviewed = False
    if document is not None:
        case = dataclasses.replace(parse_case(document, mesh.faces), mesh=settings)
    else:
        fixed = parse_face_list(args.fixed, _faces_by_tag(mesh).keys()) if args.fixed else None
        loads = tuple(
            [parse_force_flag(t, mesh) for t in args.force]
            + [parse_pressure_flag(t, mesh) for t in args.pressure]
        )
        complete = fixed is not None and (loads or args.gravity)
        if complete:
            if material is None:
                material = MATERIAL_PRESETS["structural_steel"]
                console.say("  material not given: Structural Steel (the Ansys default); "
                            "pass --material to change it")
            case = StaticStructuralCase(
                supports=(FixedSupport(faces=fixed),), loads=loads, material=material,
                gravity=Gravity() if args.gravity else None, mesh=settings,
            )
        elif not interactive:          # pragma: no cover - refused above, before meshing
            raise CaseError("--yes needs a complete case: add --fixed and a load, or --case FILE")
        else:
            def remesh(new: MeshSettings) -> Any:
                return sr.mesh_part(path, new)

            result = interview(
                console, mesh, settings=settings, remesh=remesh, screen=not args.no_screen,
                defaults=Defaults(material=material, fixed=fixed, loads=loads,
                                  gravity=bool(args.gravity), family=family),
            )
            case, family, mesh = result.case, result.family, result.mesh
            asked_large = result.allow_large
            interviewed = True

    files = sr.find_surrogate_files(args.weights, args.calibration, args.neighbours)
    console.say("")
    for line in box("A: Static Structural - check before solving",
                    _confirm_rows(path, solid, mesh, case, family, not args.no_screen, out_dir, files)):
        console.say(line)
    if save_target is not None:
        _write_case(case, save_target)
        console.say(f"  case saved to {save_target}")

    def keep_answers() -> str:
        """An interview declined at the end is not thrown away."""
        if save_target is not None:
            return f" The case is in {save_target}."
        if not interviewed:
            return " Use --save-case FILE to keep the case."
        saved = _write_case(case, out_dir / "case.json")
        return f" Your answers are saved: rerun with --case {saved}"

    if interactive and not console.yes_no("Proceed?", default=True,
                                          help="Y solves now; n stops without solving."):
        console.say("Stopped before solving; nothing was solved." + keep_answers())
        return EXIT_OK
    allow_large = asked_large or _large_ok(console, mesh, args, interactive)
    if interactive and not allow_large and int(mesh.dof_count) > sr.large_dof_threshold():
        console.say(f"Not solved: {int(mesh.dof_count):,} DOF is above "
                    f"{sr.large_dof_threshold():,}; use a coarser --mesh or a larger "
                    "--element-size, or --allow-large." + keep_answers())
        return EXIT_OK
    return _solve_and_report(console, path=path, mesh=mesh, case=case, family=family,
                             out_dir=out_dir, args=args, allow_large=allow_large,
                             json_print=json_print)


def _command_inspect(console: Console, args: argparse.Namespace) -> int:
    from .cad_intake import check_readable

    path = check_readable(Path(args.part).expanduser())
    _require(INSPECT_EXTRAS)
    out_dir = _out_dir(path, args.out)
    _check_output_dir(out_dir)
    clock = _Clock()
    _, mesh = _read_and_mesh(console, path, MeshSettings(), clock, steps=2)
    _show_faces(console, mesh, out_dir, args.no_open)
    console.say("")
    console.say(f"Next: {PROG} {args.part}   (or --fixed FACES --force FACES:500:-z --yes)")
    return EXIT_OK


class _QuietFd:
    """Silence fd 1/2 around OCC's STEP writer, which prints transfer banners
    straight from C++ (print() redirection cannot catch them)."""

    def __enter__(self) -> "_QuietFd":
        self.saved: list[tuple[int, int]] = []
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            null = os.open(os.devnull, os.O_WRONLY)
            for fd in (1, 2):
                self.saved.append((fd, os.dup(fd)))
                os.dup2(null, fd)
            os.close(null)
        except (OSError, ValueError):
            pass
        return self

    def __exit__(self, *exc: Any) -> None:
        for fd, copy in self.saved:
            try:
                os.dup2(copy, fd)
                os.close(copy)
            except OSError:
                pass


def write_demo_bracket(path: Path) -> Path:
    """An L-bracket (two fused boxes, as tests/test_end_to_end.py builds it):
    a 180 x 60 x 12 mm plate with a 12 x 60 x 90 mm upstand at x = 0."""
    import gmsh

    from .cad_intake import start_gmsh

    path.parent.mkdir(parents=True, exist_ok=True)
    with _QuietFd():
        start_gmsh()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("demo_bracket")
            plate = gmsh.model.occ.addBox(0, 0, 0, 180, 60, 12)
            upstand = gmsh.model.occ.addBox(0, 0, 0, 12, 60, 90)
            gmsh.model.occ.fuse([(3, plate)], [(3, upstand)])
            gmsh.model.occ.synchronize()
            gmsh.write(str(path))
        finally:
            gmsh.finalize()
    return path


def demo_case(mesh: Any) -> StaticStructuralCase:
    """The bracket family's trained load case, selected by the family's own
    face preset so the demo is screenable by construction."""
    from .face_selectors import FAMILY_FACE_PRESETS, select_faces

    preset = FAMILY_FACE_PRESETS["bracket"]
    fixed = tuple(sorted(select_faces(preset["fixed"], mesh, "fixed")))
    loaded = tuple(sorted(select_faces(preset["loaded"], mesh, "loaded")))
    direction = preset["direction"]
    return StaticStructuralCase(
        supports=(FixedSupport(faces=fixed),),
        loads=(ForceLoad(faces=loaded, vector_n=tuple(DEMO_FORCE_N * c for c in direction)),),
        material=MATERIAL_PRESETS["structural_steel"],
    )


def _command_demo(console: Console, args: argparse.Namespace) -> int:
    _require(RUN_EXTRAS)
    path_name = Path("demo_bracket.step")
    out_dir = _out_dir(path_name, args.out)
    _check_output_dir(out_dir)
    path = write_demo_bracket(out_dir / path_name.name)
    console.say(f"{PROG} demo: wrote an L-bracket to {path}")
    clock = _Clock()
    settings = MeshSettings()
    solid, mesh = _read_and_mesh(console, path, settings, clock)
    _show_faces(console, mesh, out_dir, args.no_open)
    case = demo_case(mesh)
    family = "bracket"
    run_args = argparse.Namespace(
        no_screen=args.no_screen, weights=None, calibration=None, neighbours=None,
        converge=False, no_open=args.no_open, allow_large=False, crosscheck=False,
    )
    files = sr.find_surrogate_files()
    console.say("")
    for line in box("A: Static Structural - demo case",
                    _confirm_rows(path, solid, mesh, case, family, not args.no_screen, out_dir, files)):
        console.say(line)
    return _solve_and_report(console, path=path, mesh=mesh, case=case, family=family,
                             out_dir=out_dir, args=run_args, allow_large=False, json_print=None)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _exit_code_for(exc: BaseException) -> tuple[int, str]:
    name = type(exc).__name__
    if isinstance(exc, (CaseError, MissingExtras, OutputError)):
        return EXIT_INPUT, f"error: {exc}"
    if name in ("CadFormatError", "AnalysisMeshError"):
        # The part or the element size asked for cannot be meshed: input.
        return EXIT_INPUT, f"error: {exc}"
    if isinstance(exc, OSError) and not isinstance(exc, (ImportError,)):
        # Every file a run reads is checked up front, so this is a write.
        where = f" {exc.filename}" if getattr(exc, "filename", None) else ""
        return EXIT_INPUT, f"error: cannot write{where}: {exc.strerror or exc}"
    if isinstance(exc, ImportError) and getattr(exc, "name", None) in (
        "gmsh", "scipy", "matplotlib", "torch", "torch_geometric",
    ):
        return EXIT_INPUT, f"error: missing {exc.name}: install the extras with {sr.INSTALL_HINT}"
    if isinstance(exc, sr.RunRefused) or name in ("CadIntakeError", "SolveCadError"):
        return EXIT_REFUSED, f"refused: {exc}"
    return EXIT_FAILED, f"solve failed: {name}: {exc}"


def main(
    argv: Sequence[str] | None = None,
    *,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[..., Any] = print,
) -> int:
    """Run the CLI; returns the exit code (0 ok, 2 input, 3 refused, 4 failed, 130 interrupted)."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    json_mode = "--json" in arguments
    # A console that cannot encode the tables (a Windows code page, stdout
    # redirected there) gets ASCII instead of a UnicodeEncodeError mid-run.
    plain_stdout = not _encodes_all(sys.stdout)
    if plain_stdout and print_fn is print and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")    # paths can hold anything
        except (ValueError, OSError):
            pass
    if json_mode:
        # stdout carries only results.json; everything a person reads goes to
        # stderr, prompts included (input() would write them to stdout).
        if input_fn is input:
            def input_fn(prompt: str) -> str:  # noqa: F811
                sys.stderr.write(prompt if _encodes_all(sys.stderr) else prompt.translate(_ASCII))
                sys.stderr.flush()
                return input()
        console = Console(input_fn, _ascii_safe(lambda *a, **k: print(*a, file=sys.stderr, **k),
                                                sys.stderr),
                          ascii_json=plain_stdout and print_fn is print)
        json_print: Callable[..., Any] | None = print_fn
    else:
        if input_fn is input and plain_stdout:
            def input_fn(prompt: str) -> str:  # noqa: F811
                return input(prompt.translate(_ASCII))
        console = Console(input_fn, _ascii_safe(print_fn, sys.stdout) if print_fn is print
                          else print_fn)
        json_print = None
    if input_fn is input and sys.stdin is not None and sys.stdin.isatty():
        try:
            import readline  # noqa: F401 - line editing for input()
        except ImportError:
            pass
    try:
        if arguments[:1] == ["inspect"]:
            args = _inspect_parser().parse_args(arguments[1:])
            return _command_inspect(console, args)
        if arguments[:1] == ["demo"]:
            args = _demo_parser().parse_args(arguments[1:])
            return _command_demo(console, args)
        args = _run_parser().parse_args(arguments)
        return _command_run(console, args, json_print)
    except (KeyboardInterrupt, EOFError):
        console.say("")
        console.say("Interrupted; nothing more was done.")
        return EXIT_INTERRUPTED
    except SystemExit as exc:          # argparse: --help (0) or a usage error (2)
        code = exc.code
        return code if isinstance(code, int) else (EXIT_OK if code is None else EXIT_INPUT)
    except Exception as exc:
        code, message = _exit_code_for(exc)
        console.say(message)
        if os.environ.get("FEA_STRUCTURAL_DEBUG"):
            import traceback

            console.say(traceback.format_exc())
        elif code == EXIT_FAILED:
            console.say("(set FEA_STRUCTURAL_DEBUG=1 for the traceback)")
        return code


if __name__ == "__main__":
    raise SystemExit(main())
