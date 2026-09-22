"""Choose boundary-condition faces the way the training sweep chose them.

Every family in `generate/families/` tags boundary conditions as **faces**, by a
predicate on the face centroid — or, for the pulley, on radial extent. Every node
of a tagged face is then held or loaded.

The deployed path used node-coordinate selectors instead, and the two do not
agree. Measured over 600 fresh parts:

    family          deployed coverage    median pred/true
    spring                      0.00                0.00
    flange                      0.00                4.97
    bracket                     0.46                0.72

against a calibrated 0.90. The spring preset held 20 of 6144 nodes — a tangent
sliver of the helical surface — where training held the entire end-coil cross
section. The flange preset held 26.5% of nodes in a slab through the whole plate,
where training held only the faces whose centroid lies inside the plate.

So the selectors here are face predicates, in the same terms the tags use, with
their thresholds derived from the part's own measured geometry rather than from
sweep parameters a screened STEP does not carry.

Three kinds, one per tag rule that exists:

    {"centroid": {"axis": "z", "at": "min"}}        an end face
    {"centroid": {"axis": "z", "between": [a, b]}}  a face inside the extent
    {"radial": "inner"} / {"radial": "outer"}       a bore or a rim
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


class FaceSelectionError(ValueError):
    """The selector matched no face, or is not one this module understands."""


#: How close to an extreme a face centroid must sit to count as an end face, as a
#: fraction of the extent along that axis. The tags use 2-5% of a local dimension
#: (thickness, wire diameter); this is the geometric stand-in, loose enough to
#: catch an end face on a coarse part and tight enough to exclude a side wall.
END_FACE_TOLERANCE = 0.02

#: Pulley: `radial extent < bore * 1.25` and `> outer * 0.85` in the tag. Applied
#: here against the measured minimum and maximum radial extent instead.
BORE_FACTOR = 1.25
RIM_FACTOR = 0.85

_AXES = {"x": 0, "y": 1, "z": 2}


def _centroids(mesh) -> dict[int, np.ndarray]:  # noqa: ANN001 - ScaledMesh
    if not mesh.face_centroids:
        raise FaceSelectionError(
            "this mesh carries no face centroids; it was built before face "
            "selection existed, or by a version that does not record them"
        )
    return {tag: np.asarray(c, dtype=float) for tag, c in mesh.face_centroids.items()}


def _radial_extent(mesh, tag: int) -> float:  # noqa: ANN001
    """Largest distance from the part's axis to any node of the face.

    By extent rather than centroid, for the reason `pulley.py` records: a full
    cylinder's centre of mass sits on its axis, so a bore and a rim are
    indistinguishable by centroid.
    """
    nodes = mesh.face_nodes[tag]
    points = np.asarray(mesh.vertices, dtype=float)[nodes]
    centre = np.asarray(mesh.vertices, dtype=float).mean(axis=0)
    # The pulley's axis is x; radial distance is measured in the other two.
    return float(np.max(np.linalg.norm(points[:, 1:] - centre[1:], axis=1)))


def select_faces(spec: Mapping[str, Any], mesh, role: str) -> tuple[int, ...]:  # noqa: ANN001
    """Face tags matching `spec`. Never empty — an empty match is an error."""
    if not isinstance(spec, Mapping) or not spec:
        raise FaceSelectionError(f"{role} face selector must be a non-empty object")

    if "faces" in spec:
        wanted = tuple(int(t) for t in spec["faces"])
        missing = [t for t in wanted if t not in mesh.face_nodes]
        if missing:
            known = ", ".join(str(t) for t in sorted(mesh.face_nodes))
            raise FaceSelectionError(
                f"{role}: no face {missing}; this part has faces {known}"
            )
        return wanted

    if "centroid" in spec:
        inner = spec["centroid"]
        axis = _AXES.get(str(inner.get("axis", "")).lower())
        if axis is None:
            raise FaceSelectionError(f"{role}: centroid selector needs axis x, y or z")
        centroids = _centroids(mesh)
        values = np.array([centroids[t][axis] for t in sorted(centroids)])
        tags = sorted(centroids)
        lo, hi = float(values.min()), float(values.max())
        span = (hi - lo) or 1.0

        if "at" in inner:
            where = str(inner["at"]).lower()
            # The tags express their tolerances against a *local* dimension --
            # 0.05 * thickness, 0.6 * wire diameter -- not against an overall
            # span. `of: thinnest` reproduces that; `of: span` is the fallback
            # for a family whose tag really does scale with total length.
            fraction = float(inner.get("tolerance_fraction", END_FACE_TOLERANCE))
            basis = str(inner.get("tolerance_of", "span")).lower()
            if basis == "thinnest":
                if not mesh.thinnest:
                    raise FaceSelectionError(
                        f"{role}: this mesh does not carry a thinnest dimension"
                    )
                tol = fraction * float(mesh.thinnest)
            elif basis == "span":
                tol = fraction * span
            else:
                raise FaceSelectionError(
                    f"{role}: tolerance_of must be 'thinnest' or 'span'"
                )
            if where == "min":
                hit = [t for t, v in zip(tags, values) if v <= lo + tol]
            elif where == "max":
                hit = [t for t, v in zip(tags, values) if v >= hi - tol]
            else:
                raise FaceSelectionError(f"{role}: 'at' must be 'min' or 'max'")
        elif "between" in inner:
            low, high = (float(v) for v in inner["between"])
            hit = [
                t for t, v in zip(tags, values)
                if lo + low * span < v < lo + high * span
            ]
        else:
            raise FaceSelectionError(f"{role}: centroid selector needs 'at' or 'between'")

        if not hit:
            raise FaceSelectionError(
                f"{role}: no face centroid satisfies {inner}. This part's "
                f"centroids along that axis span {lo:.4g} to {hi:.4g}."
            )
        return tuple(sorted(hit))

    if "radial" in spec:
        where = str(spec["radial"]).lower()
        extents = {t: _radial_extent(mesh, t) for t in mesh.face_nodes}
        if not extents:
            raise FaceSelectionError(f"{role}: the mesh has no faces")
        smallest, largest = min(extents.values()), max(extents.values())
        if where == "inner":
            hit = [t for t, r in extents.items() if r < smallest * BORE_FACTOR]
        elif where == "outer":
            hit = [t for t, r in extents.items() if r > largest * RIM_FACTOR]
        else:
            raise FaceSelectionError(f"{role}: 'radial' must be 'inner' or 'outer'")
        if not hit:
            raise FaceSelectionError(f"{role}: no face at the {where} radius")
        return tuple(sorted(hit))

    raise FaceSelectionError(
        f"{role}: selector must carry 'faces', 'centroid' or 'radial', got "
        f"{sorted(spec)}"
    )


def face_mask(tags: Sequence[int], mesh) -> np.ndarray:  # noqa: ANN001
    """A node mask covering every node of the given faces.

    The same rule `generate/harness.py:_nodes_of_faces` applies during the sweep,
    so the deployed path and the training path select identically rather than by
    two rules that happen to agree on some families.
    """
    mask = np.zeros(len(mesh.vertices), dtype=bool)
    for tag in tags:
        mask[mesh.face_nodes[int(tag)]] = True
    if not mask.any():
        raise FaceSelectionError("the selected faces carry no mesh nodes")
    return mask


#: Face-based equivalents of each family's `tag()`, derived from geometry alone.
#: Compare against `generate/families/*.py`; the comment on each says which rule
#: it reproduces.
FAMILY_FACE_PRESETS: dict[str, dict[str, Any]] = {
    # bracket.py:49-50 — |c_x| < 0.05 * thickness, and |c_x - length| < tol.
    # Against the span instead, the tolerance is wide enough to pick up the
    # underside of the upstand as a second held face, which training does not.
    "bracket": {
        "fixed": {"centroid": {"axis": "x", "at": "min",
                               "tolerance_fraction": 0.05,
                               "tolerance_of": "thinnest"}},
        "loaded": {"centroid": {"axis": "x", "at": "max"}},
        "direction": [0.0, 0.0, -1.0],
    },
    # shaft.py:43-44 — fixed at the far end (c_z = total), loaded at c_z = 0
    "stepped_shaft": {
        "fixed": {"centroid": {"axis": "z", "at": "max"}},
        "loaded": {"centroid": {"axis": "z", "at": "min"}},
        "direction": [1.0, 0.0, 0.0],
    },
    # welded_t.py:44-45 — base underside, and web top
    "welded_t": {
        "fixed": {"centroid": {"axis": "z", "at": "min"}},
        "loaded": {"centroid": {"axis": "z", "at": "max"}},
        "direction": [1.0, 0.0, 0.0],
    },
    # spring.py:68-69 — the end-coil cross sections, not the helical surface
    "spring": {
        "fixed": {"centroid": {"axis": "z", "at": "min"}},
        "loaded": {"centroid": {"axis": "z", "at": "max"}},
        "direction": [0.0, 0.0, -1.0],
    },
    # flange.py:54-57 — every face whose centroid lies inside the plate
    # thickness, which is the outer rim as well as the hole walls, and the top
    # face. The docstring there says "hole walls are the only faces whose
    # centroid sits inside the plate"; that is not true of the geometry, and the
    # model was trained on the behaviour rather than the comment.
    "flange": {
        "fixed": {"centroid": {"axis": "z", "between": [0.1, 0.9]}},
        "loaded": {"centroid": {"axis": "z", "at": "max"}},
        "direction": [0.0, 0.0, -1.0],
    },
    # pulley.py:51-52 — by radial extent, never by centroid
    "pulley": {
        "fixed": {"radial": "inner"},
        "loaded": {"radial": "outer"},
        "direction": [1.0, 0.0, 0.0],
    },
}


__all__ = [
    "BORE_FACTOR",
    "END_FACE_TOLERANCE",
    "FAMILY_FACE_PRESETS",
    "RIM_FACTOR",
    "FaceSelectionError",
    "face_mask",
    "select_faces",
]
