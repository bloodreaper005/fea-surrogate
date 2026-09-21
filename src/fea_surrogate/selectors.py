"""Say which part of a mesh is held and which is pushed on.

Declarative, because the caller is an agent sending JSON and cannot know how
gmsh numbered the faces of a part it did not mesh. The same vocabulary the
families use to tag their own geometry, so the server and the training data
describe boundary conditions the same way.

Three kinds, covering what the six families need:

    {"axis": "x", "at": "min"}            a planar extreme
    {"axis": "z", "between": [0.1, 0.9]}  a band, as a fraction of the extent
    {"radial": "inner"}                   by distance from the z axis

Free of torch: a selector is a description, and describing one should not need
the inference stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


AXES = {"x": 0, "y": 1, "z": 2}
DEFAULT_TOLERANCE = 0.02


class SelectorError(ValueError):
    """The selector cannot be understood, or selected nothing."""


@dataclass(frozen=True)
class Selector:
    """A rule picking vertices out of a mesh."""

    raw: Mapping[str, Any]

    def mask(self, vertices: np.ndarray) -> np.ndarray:
        spec = self.raw
        tolerance = float(spec.get("tolerance", DEFAULT_TOLERANCE))
        if not 0.0 < tolerance < 1.0:
            raise SelectorError("tolerance must be a fraction between 0 and 1")

        if "radial" in spec:
            return self._radial(vertices, str(spec["radial"]), tolerance)
        if "axis" not in spec:
            raise SelectorError("a selector needs either 'axis' or 'radial'")

        axis = AXES.get(str(spec["axis"]).lower())
        if axis is None:
            raise SelectorError(f"unknown axis {spec['axis']!r}; use x, y or z")
        column = vertices[:, axis]
        low, high = float(column.min()), float(column.max())
        span = (high - low) or 1.0

        if "between" in spec:
            lo, hi = spec["between"]
            return (column >= low + float(lo) * span) & (
                column <= low + float(hi) * span
            )
        at = str(spec.get("at", "min")).lower()
        if at == "min":
            return column <= low + tolerance * span
        if at == "max":
            return column >= high - tolerance * span
        raise SelectorError(f"unknown 'at' {at!r}; use min or max")

    @staticmethod
    def _radial(vertices: np.ndarray, where: str, tolerance: float) -> np.ndarray:
        radius = np.hypot(vertices[:, 0] - vertices[:, 0].mean(),
                          vertices[:, 1] - vertices[:, 1].mean())
        low, high = float(radius.min()), float(radius.max())
        span = (high - low) or 1.0
        if where == "inner":
            return radius <= low + max(tolerance, 0.15) * span
        if where == "outer":
            return radius >= high - max(tolerance, 0.15) * span
        raise SelectorError(f"unknown 'radial' {where!r}; use inner or outer")


def select(spec: Mapping[str, Any] | None, vertices: np.ndarray,
           label: str) -> np.ndarray:
    """Apply a selector, refusing one that picks nothing.

    A selector matching no vertex is a silent misdescription of the load case,
    which is worse than an error: the solve would proceed against geometry the
    caller did not intend.
    """
    if spec is None:
        raise SelectorError(f"no {label} selector given")
    mask = Selector(spec).mask(np.asarray(vertices, dtype=float))
    if not mask.any():
        raise SelectorError(f"the {label} selector matched no vertex")
    return mask


#: How each swept family tags its own geometry. Naming a family without
#: selectors reproduces the boundary conditions that family was trained under,
#: which is the only way a per-family band means anything.
FAMILY_PRESETS: dict[str, dict[str, Any]] = {
    "bracket": {
        "fixed": {"axis": "x", "at": "min"},
        "loaded": {"axis": "x", "at": "max"},
        "direction": [0.0, 0.0, -1.0],
    },
    "stepped_shaft": {
        "fixed": {"axis": "z", "at": "max"},
        "loaded": {"axis": "z", "at": "min"},
        "direction": [1.0, 0.0, 0.0],
    },
    "welded_t": {
        "fixed": {"axis": "z", "at": "min"},
        "loaded": {"axis": "z", "at": "max"},
        "direction": [1.0, 0.0, 0.0],
    },
    "spring": {
        "fixed": {"axis": "z", "at": "min"},
        "loaded": {"axis": "z", "at": "max"},
        "direction": [0.0, 0.0, -1.0],
    },
    "flange": {
        "fixed": {"axis": "z", "between": [0.15, 0.85]},
        "loaded": {"axis": "z", "at": "max"},
        "direction": [0.0, 0.0, -1.0],
    },
    "pulley": {
        "fixed": {"radial": "inner"},
        "loaded": {"radial": "outer"},
        "direction": [1.0, 0.0, 0.0],
    },
}


def resolve(
    family: str | None,
    fixed: Mapping[str, Any] | None,
    loaded: Mapping[str, Any] | None,
    direction: list[float] | None,
) -> tuple[Mapping[str, Any], Mapping[str, Any], list[float], str]:
    """Explicit selectors win; a named family supplies them otherwise.

    Returns the pair actually used and how they were arrived at, because a
    document has to say which load case it screened.
    """
    preset = FAMILY_PRESETS.get(family or "", {})
    resolved_fixed = fixed or preset.get("fixed")
    resolved_loaded = loaded or preset.get("loaded")
    resolved_direction = direction or preset.get("direction") or [0.0, 0.0, -1.0]

    if resolved_fixed is None or resolved_loaded is None:
        raise SelectorError(
            "no boundary conditions: give fixed and loaded selectors, or name "
            f"one of {', '.join(sorted(FAMILY_PRESETS))}"
        )
    if fixed or loaded:
        origin = "caller-supplied selectors"
    else:
        origin = f"{family!r} family preset"
    return resolved_fixed, resolved_loaded, list(resolved_direction), origin


__all__ = [
    "AXES",
    "DEFAULT_TOLERANCE",
    "FAMILY_PRESETS",
    "Selector",
    "SelectorError",
    "resolve",
    "select",
]
