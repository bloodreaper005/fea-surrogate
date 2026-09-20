"""Bolted flange: a plate carrying bolt holes, held at the holes.

Load passes from a face into four constrained holes, so the peak sits at a hole
edge. That is the geometry `mechanics.bolted_joints` reasons about, and the
stress concentration a plain plate never shows.
"""

from __future__ import annotations

import numpy as np

from ..family import Tagging, faces_where, log_uniform


NAME = "flange"


class Flange:
    name = NAME

    def sample_params(self, rng: np.random.Generator) -> dict[str, float]:
        length = log_uniform(rng, 80.0, 250.0)
        return {
            "length_mm": length,
            "width_mm": length * float(rng.uniform(0.4, 0.9)),
            "thickness_mm": log_uniform(rng, 6.0, 25.0),
            "hole_diameter_mm": log_uniform(rng, 5.0, 16.0),
            "margin_fraction": float(rng.uniform(0.12, 0.22)),
        }

    def build(self, params: dict[str, float]) -> None:
        import gmsh

        occ = gmsh.model.occ
        length = params["length_mm"]
        width = params["width_mm"]
        thickness = params["thickness_mm"]
        radius = params["hole_diameter_mm"] / 2
        mx = params["margin_fraction"] * length
        my = params["margin_fraction"] * width

        plate = occ.addBox(0, 0, 0, length, width, thickness)
        holes = [
            (3, occ.addCylinder(x, y, -1.0, 0, 0, thickness + 2.0, radius))
            for x in (mx, length - mx)
            for y in (my, width - my)
        ]
        occ.cut([(3, plate)], holes)

    def tag(self, params: dict[str, float]) -> Tagging:
        radius = params["hole_diameter_mm"] / 2
        thickness = params["thickness_mm"]
        # Hole walls are the only faces whose centroid sits inside the plate.
        fixed = faces_where(
            lambda c: 0.1 * thickness < c[2] < 0.9 * thickness
        )
        loaded = faces_where(lambda c: abs(c[2] - thickness) < 0.05 * thickness)
        if not fixed or not loaded:
            raise ValueError("flange tagging found no hole wall or top face")
        del radius
        return Tagging(fixed, loaded, (0.0, 0.0, -1.0))


__all__ = ["NAME", "Flange"]
