"""L-bracket: a cantilever with a mounting leg.

The reference family. It is the simplest thing that is still a real component,
and its loaded case reduces to Shigley eq. 3-24 in closed form, so anything
trained on it can be checked against an answer that owes nothing to the model.
"""

from __future__ import annotations

import numpy as np

from ..family import Tagging, faces_where, log_uniform


NAME = "bracket"


class Bracket:
    name = NAME

    def sample_params(self, rng: np.random.Generator) -> dict[str, float]:
        thickness = log_uniform(rng, 3.0, 20.0)
        return {
            "length_mm": log_uniform(rng, 40.0, 300.0),
            "width_mm": log_uniform(rng, 20.0, 150.0),
            "thickness_mm": thickness,
            "height_mm": log_uniform(rng, 30.0, 200.0),
        }

    def build(self, params: dict[str, float]) -> None:
        import gmsh

        length = params["length_mm"]
        width = params["width_mm"]
        thickness = params["thickness_mm"]
        height = params["height_mm"]

        arm = gmsh.model.occ.addBox(0.0, 0.0, 0.0, length, width, thickness)
        leg = gmsh.model.occ.addBox(0.0, 0.0, 0.0, thickness, width, height)
        gmsh.model.occ.fuse([(3, arm)], [(3, leg)])

    def tag(self, params: dict[str, float]) -> Tagging:
        length = params["length_mm"]
        thickness = params["thickness_mm"]
        tol = 0.05 * thickness

        # Held at the mounting face, loaded downward at the free end: a
        # cantilever, which is what makes the closed-form check available.
        fixed = faces_where(lambda c: abs(c[0]) < tol)
        loaded = faces_where(lambda c: abs(c[0] - length) < max(tol, 0.01 * length))
        if not fixed or not loaded:
            raise ValueError("bracket tagging found no fixed or loaded face")
        return Tagging(
            fixed_faces=fixed,
            loaded_faces=loaded,
            load_direction=(0.0, 0.0, -1.0),
        )

    @staticmethod
    def closed_form_peak_mpa(params: dict[str, float], force_n: float) -> float:
        """Root bending stress, Shigley eq. 3-24.

        Not used in training. It is the independent anchor a surrogate's
        interval has to contain, and the reason this family is first.
        """
        length = params["length_mm"]
        width = params["width_mm"]
        thickness = params["thickness_mm"]
        inertia = width * thickness**3 / 12.0
        return abs(force_n * length * (thickness / 2.0) / inertia)


__all__ = ["NAME", "Bracket"]
