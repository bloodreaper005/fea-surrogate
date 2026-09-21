"""Which feature construction a model was fitted with.

Deliberately free of torch and gmsh: a version is metadata, and the contract
side has to be able to name one without the inference stack installed.

A model is only meaningful alongside the features it was trained on, so these
are selected by what a calibration record declares rather than by whichever is
newest. Both versions are kept; the first is not deleted because it produced
results that the second is measured against.
"""

from __future__ import annotations


SFEM_V1 = "sfem/v1"
SWEEP_V2 = "sweep/v2"
DEFAULT_VERSION = SWEEP_V2

FEATURE_COLUMNS = {SFEM_V1: 12, SWEEP_V2: 13}

LAYOUTS = {
    SFEM_V1: "pos(3) sdf(1) normal(3) fixed_free(2) load_class_onehot(3)",
    SWEEP_V2: "pos(3) sdf(1) normal(3) is_fixed(1) is_loaded(1) load_dir(3) poisson(1)",
}

#: What each version was fitted on, and what it measured about itself. The band
#: is the headline difference and it has one cause: v1 encodes the load's
#: magnitude class and never where it lands, so samples differing only in
#: placement are identical on the input side.
PROVENANCE = {
    SFEM_V1: {
        "dataset": "cmudrc/SFEM",
        "head": "per-node field; peak taken as a max over nodes",
        "band_factor": 19.75,
        "encodes_load_location": False,
        "note": "single shared conformal band",
    },
    SWEEP_V2: {
        "dataset": "fea-surrogate sweep",
        "head": "graph-level; peak predicted directly",
        "band_factor": 1.91,
        "encodes_load_location": True,
        "note": "Mondrian per-family bands; shared band shown for comparison",
    },
}


class FeatureVersionError(ValueError):
    """The requested feature version is not one this package builds."""


def require_version(version: str) -> str:
    if version not in FEATURE_COLUMNS:
        known = ", ".join(sorted(FEATURE_COLUMNS))
        raise FeatureVersionError(f"unknown feature version {version!r}; known: {known}")
    return version


__all__ = [
    "DEFAULT_VERSION",
    "FEATURE_COLUMNS",
    "LAYOUTS",
    "PROVENANCE",
    "SFEM_V1",
    "SWEEP_V2",
    "FeatureVersionError",
    "require_version",
]
