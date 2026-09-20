"""The contract every component family implements.

A family is three things: a parameter sampler, a builder that puts a solid into
the active gmsh model, and a tagging rule that says which faces are held and
which are pushed on. Everything downstream — meshing, solving, feature
assembly — is shared, so adding a family is adding those three and nothing
else.

The feature schema below is where this departs from SFEM, deliberately. SFEM
encodes the fixed surface and the load *magnitude class* but never where the
load lands, so 50 samples of one geometry have identical inputs and different
answers, and no model can do better than predict their mean. Load location and
direction are first-class here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np


# [pos(3) | sdf(1) | normal(3) | is_fixed(1) | is_loaded(1) | load_dir(3) | nu(1)]
FEATURE_COLUMNS = 13

# Poisson's ratio is the one material property that does not drop out of the
# dimensionless target, so it is swept. Young's modulus does drop out and is
# deliberately not a feature.
POISSON_RANGE = (0.27, 0.42)


@dataclass(frozen=True)
class Tagging:
    """Which faces are held, which are loaded, and in what direction."""

    fixed_faces: tuple[int, ...]
    loaded_faces: tuple[int, ...]
    load_direction: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not self.fixed_faces:
            raise ValueError("a family must fix at least one face")
        if not self.loaded_faces:
            raise ValueError("a family must load at least one face")
        if np.linalg.norm(self.load_direction) == 0.0:
            raise ValueError("load direction must be nonzero")


@dataclass(frozen=True)
class Sample:
    """One drawn instance of a family, before it is meshed or solved."""

    family: str
    params: dict[str, float]
    tagging: Tagging
    poisson_ratio: float
    force_n: float
    metadata: dict[str, object] = field(default_factory=dict)


class Family(Protocol):
    """What a component family must provide."""

    name: str

    def sample_params(self, rng: np.random.Generator) -> dict[str, float]:
        """Draw one parameter set from this family's design space."""

    def build(self, params: dict[str, float]) -> None:
        """Add the solid to the active gmsh model. Does not synchronise."""

    def tag(self, params: dict[str, float]) -> Tagging:
        """Select fixed and loaded faces on the synchronised model."""


def face_centroids() -> dict[int, np.ndarray]:
    """Centroid of every surface in the synchronised model, by tag."""
    import gmsh

    return {
        tag: np.array(gmsh.model.occ.getCenterOfMass(2, tag))
        for _, tag in gmsh.model.getEntities(2)
    }


def faces_where(
    predicate, centroids: dict[int, np.ndarray] | None = None
) -> tuple[int, ...]:
    """Tags of faces whose centroid satisfies `predicate`.

    Selecting by centroid keeps a family's tagging readable and independent of
    how the kernel happens to number its faces, which changes with parameters.
    """
    centroids = face_centroids() if centroids is None else centroids
    return tuple(sorted(tag for tag, c in centroids.items() if predicate(c)))


def extreme_face(axis: int, *, lowest: bool = True) -> tuple[int, ...]:
    """The face furthest along an axis — the usual 'the surface it sits on'."""
    centroids = face_centroids()
    if not centroids:
        raise ValueError("model has no faces")
    key = min if lowest else max
    target = key(centroids.items(), key=lambda kv: kv[1][axis])
    return (target[0],)


def sample_poisson(rng: np.random.Generator) -> float:
    return float(rng.uniform(*POISSON_RANGE))


def sample_force(rng: np.random.Generator) -> float:
    """Log-uniform over a wide range.

    Magnitude is handled exactly by the dimensionless target, so this only has
    to span enough to keep the solve well conditioned.
    """
    return float(np.exp(rng.uniform(np.log(50.0), np.log(50_000.0))))


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


__all__ = [
    "FEATURE_COLUMNS",
    "POISSON_RANGE",
    "Family",
    "Sample",
    "Tagging",
    "extreme_face",
    "face_centroids",
    "faces_where",
    "log_uniform",
    "sample_force",
    "sample_poisson",
]
