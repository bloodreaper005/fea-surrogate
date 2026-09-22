"""Helical compression spring, squashed along its axis.

The one family whose peak is torsional rather than bending, which is why
`mechanics.springs` rates it through the Wahl correction rather than M c / I.
Including it stops the model learning that stress always means bending.
"""

from __future__ import annotations

import numpy as np

from ..family import Tagging, faces_where, log_uniform


NAME = "spring"

_POINTS_PER_TURN = 24


class Spring:
    name = NAME

    def sample_params(self, rng: np.random.Generator) -> dict[str, float]:
        wire = log_uniform(rng, 2.0, 8.0)
        return {
            "wire_diameter_mm": wire,
            "coil_radius_mm": wire * float(rng.uniform(3.0, 7.0)),
            "pitch_mm": wire * float(rng.uniform(2.0, 4.5)),
            "turns": float(rng.uniform(2.5, 5.5)),
        }

    def build(self, params: dict[str, float]) -> None:
        import gmsh

        occ = gmsh.model.occ
        radius = params["coil_radius_mm"]
        wire = params["wire_diameter_mm"] / 2
        pitch = params["pitch_mm"]
        turns = params["turns"]

        steps = int(turns * _POINTS_PER_TURN)
        points = []
        for i in range(steps + 1):
            angle = 2.0 * np.pi * i / _POINTS_PER_TURN
            points.append(
                occ.addPoint(
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    pitch * angle / (2.0 * np.pi),
                )
            )
        path = occ.addWire([occ.addSpline(points)])
        # The profile plane must be normal to the helix tangent at the start,
        # which is (0, R, pitch/2pi) rather than the y axis. Sweeping a plane
        # that is merely close produces overlapping facets.
        tangent = np.array([0.0, radius, pitch / (2.0 * np.pi)])
        tangent /= np.linalg.norm(tangent)
        profile = occ.addCircle(
            radius, 0.0, 0.0, wire, zAxis=tangent.tolist()
        )
        face = occ.addPlaneSurface([occ.addCurveLoop([profile])])
        occ.addPipe([(2, face)], path)
        # `addPipe` does not consume its profile: the original plane surface
        # stays in the model, a hair away from the swept solid's bottom cap.
        # gmsh then meshes both -- measured on one sample, 24 nodes on the cap
        # and 25 on the leftover profile, 0.044 mm apart and sharing none --
        # so the bottom end of every swept spring carried twice the nodes the
        # top did. A STEP export drops the orphan, which is why a screened
        # spring and a swept spring were not the same part at all.
        occ.remove([(2, face)])

    def tag(self, params: dict[str, float]) -> Tagging:
        wire = params["wire_diameter_mm"]
        height = params["pitch_mm"] * params["turns"]
        tol = 0.6 * wire
        fixed = faces_where(lambda c: c[2] < tol)
        loaded = faces_where(lambda c: c[2] > height - tol)
        if not fixed or not loaded:
            raise ValueError("spring tagging found no end coil")
        return Tagging(fixed, loaded, (0.0, 0.0, -1.0))


__all__ = ["NAME", "Spring"]
