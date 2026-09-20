"""Welded T joint: a web standing on a base plate, loaded sideways.

The peak sits at the root of the joint, which is what `mechanics.welds` rates.
"""

from __future__ import annotations

import numpy as np

from ..family import Tagging, faces_where, log_uniform


NAME = "welded_t"


class WeldedT:
    name = NAME

    def sample_params(self, rng: np.random.Generator) -> dict[str, float]:
        return {
            "base_length_mm": log_uniform(rng, 100.0, 300.0),
            "base_width_mm": log_uniform(rng, 50.0, 150.0),
            "base_thickness_mm": log_uniform(rng, 6.0, 20.0),
            "web_thickness_mm": log_uniform(rng, 5.0, 18.0),
            "web_height_mm": log_uniform(rng, 40.0, 160.0),
        }

    def build(self, params: dict[str, float]) -> None:
        import gmsh

        occ = gmsh.model.occ
        bl = params["base_length_mm"]
        bw = params["base_width_mm"]
        bt = params["base_thickness_mm"]
        wt = params["web_thickness_mm"]
        wh = params["web_height_mm"]
        base = occ.addBox(0, 0, 0, bl, bw, bt)
        web = occ.addBox((bl - wt) / 2, 0, bt, wt, bw, wh)
        occ.fuse([(3, base)], [(3, web)])

    def tag(self, params: dict[str, float]) -> Tagging:
        bt = params["base_thickness_mm"]
        top = bt + params["web_height_mm"]
        fixed = faces_where(lambda c: abs(c[2]) < 0.05 * bt)
        loaded = faces_where(lambda c: abs(c[2] - top) < 0.05 * bt)
        if not fixed or not loaded:
            raise ValueError("welded T tagging found no base or web top")
        return Tagging(fixed, loaded, (1.0, 0.0, 0.0))


__all__ = ["NAME", "WeldedT"]
