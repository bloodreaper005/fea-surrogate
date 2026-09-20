"""Stepped shaft: two coaxial cylinders, loaded in bending at the free end.

The step is a stress raiser, which is the point: a family whose peak sits at a
geometric discontinuity rather than at the load teaches the model something a
prismatic beam cannot.
"""

from __future__ import annotations

import numpy as np

from ..family import Tagging, faces_where, log_uniform


NAME = "stepped_shaft"


class SteppedShaft:
    name = NAME

    def sample_params(self, rng: np.random.Generator) -> dict[str, float]:
        small = log_uniform(rng, 8.0, 40.0)
        return {
            "small_diameter_mm": small,
            "large_diameter_mm": small * float(rng.uniform(1.3, 2.2)),
            "small_length_mm": log_uniform(rng, 30.0, 150.0),
            "large_length_mm": log_uniform(rng, 20.0, 100.0),
        }

    def build(self, params: dict[str, float]) -> None:
        import gmsh

        occ = gmsh.model.occ
        ls = params["small_length_mm"]
        ll = params["large_length_mm"]
        thin = occ.addCylinder(0, 0, 0, 0, 0, ls, params["small_diameter_mm"] / 2)
        thick = occ.addCylinder(0, 0, ls, 0, 0, ll, params["large_diameter_mm"] / 2)
        occ.fuse([(3, thin)], [(3, thick)])

    def tag(self, params: dict[str, float]) -> Tagging:
        total = params["small_length_mm"] + params["large_length_mm"]
        tol = 0.02 * total
        fixed = faces_where(lambda c: abs(c[2] - total) < tol)
        loaded = faces_where(lambda c: abs(c[2]) < tol)
        if not fixed or not loaded:
            raise ValueError("stepped shaft tagging found no end face")
        return Tagging(fixed, loaded, (1.0, 0.0, 0.0))


__all__ = ["NAME", "SteppedShaft"]
