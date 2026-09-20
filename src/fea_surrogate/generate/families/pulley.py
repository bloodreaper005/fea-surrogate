"""Pulley or sheave: a grooved rim on a bore, pulled by belt tension.

The conveyor-drive family. Held at the bore and loaded transversely at the rim,
which is how a belt actually loads one; `mechanics.friction_drives` rates the
belt, and this rates the part carrying it.
"""

from __future__ import annotations

import numpy as np

from ..family import Tagging, face_radial_extent, log_uniform


NAME = "pulley"


class Pulley:
    name = NAME

    def sample_params(self, rng: np.random.Generator) -> dict[str, float]:
        outer = log_uniform(rng, 40.0, 200.0)
        return {
            "outer_radius_mm": outer,
            "bore_radius_mm": outer * float(rng.uniform(0.15, 0.35)),
            "width_mm": outer * float(rng.uniform(0.25, 0.7)),
            "groove_radius_mm": outer * float(rng.uniform(0.06, 0.14)),
        }

    def build(self, params: dict[str, float]) -> None:
        import gmsh

        occ = gmsh.model.occ
        outer = params["outer_radius_mm"]
        bore = params["bore_radius_mm"]
        width = params["width_mm"]
        groove = params["groove_radius_mm"]

        body = occ.addCylinder(0, 0, 0, 0, 0, width, outer)
        hole = occ.addCylinder(0, 0, -1.0, 0, 0, width + 2.0, bore)
        rim_groove = occ.addTorus(0, 0, width / 2, outer, groove)
        occ.cut([(3, body)], [(3, hole), (3, rim_groove)])

    def tag(self, params: dict[str, float]) -> Tagging:
        bore = params["bore_radius_mm"]
        outer = params["outer_radius_mm"]
        radii = face_radial_extent()

        # By radial extent, not centroid: a full cylinder's centre of mass sits
        # on its axis, so the bore and the rim are indistinguishable by centroid.
        fixed = tuple(sorted(t for t, r in radii.items() if r < bore * 1.25))
        loaded = tuple(sorted(t for t, r in radii.items() if r > outer * 0.85))
        if not fixed or not loaded:
            raise ValueError("pulley tagging found no bore or rim face")
        return Tagging(fixed, loaded, (1.0, 0.0, 0.0))


__all__ = ["NAME", "Pulley"]
