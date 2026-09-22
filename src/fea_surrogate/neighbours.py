"""Find the training parts a screened part resembles, and say how closely.

The domain gate answers "does this query satisfy the bounds the model declares?"
That is necessary and it is not sufficient: a part can sit inside every declared
bound -- linear elastic, Poisson 0.30, a sensible size, one traction on tagged
faces -- and still be unlike anything in three thousand samples. The gate has no
way to see that, because the bounds say nothing about shape.

So this is a second, softer check that fails differently. It describes a part by
a dozen numbers that survive scale and rotation, finds the nearest training
samples in that space, and reports them **with their true solver answers**. Two
things come out of it:

    a warning    the nearest training part is further away than almost any
                 training part is from its own neighbour, so the estimate is
                 extrapolating however healthy the interval looks
    an argument  "this resembles these five parts, whose true stresses were
                 161, 155, 178, 149 and 166 MPa" -- which a person can push back
                 on, where a bare interval gives them nothing to hold

Never a refusal. Being unusual is not being wrong, and a hard gate on similarity
would refuse every genuinely new design, which is the population this is for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


#: Order matters: it is the on-disk layout of every stored descriptor.
DESCRIPTOR_NAMES = (
    "log_aspect_longest",     # log(e1/e3), e sorted descending
    "log_aspect_middle",      # log(e2/e3)
    "log_solidity",           # log(volume / bounding-box volume)
    "log_slenderness",        # log(2V/A / diagonal)
    "sqrt_fixed_fraction",    # sqrt so a small held patch is not crushed to zero
    "sqrt_loaded_fraction",
    "load_along_e1",          # load direction in the part's own principal frame
    "load_along_e2",
    "load_along_e3",
    "poisson_ratio",
)

#: A query further than this quantile of the training set's own
#: nearest-neighbour distances is called extrapolation.
EXTRAPOLATION_QUANTILE = 0.99

_EPS = 1e-12


class NeighbourIndexError(RuntimeError):
    """The index is missing, or was built for a different descriptor."""


def _surface_area(vertices: np.ndarray, tetrahedra: np.ndarray) -> float:
    """Area of the boundary: the faces that belong to exactly one tetrahedron."""
    corners = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)], dtype=np.int64)
    faces = np.sort(tetrahedra[:, corners].reshape(-1, 3), axis=1)
    _, first, counts = np.unique(faces, axis=0, return_index=True, return_counts=True)
    boundary = faces[first[counts == 1]]
    if boundary.size == 0:
        return 0.0
    a, b, c = (vertices[boundary[:, i]] for i in range(3))
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


def _volume(vertices: np.ndarray, tetrahedra: np.ndarray) -> float:
    a, b, c, d = (vertices[tetrahedra[:, i]] for i in range(4))
    return float(np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))).sum() / 6.0)


def describe(x: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    """A scale- and rotation-invariant descriptor of one screened problem.

    Built from the feature matrix rather than from the CAD, so the training set
    and a live query are described by exactly the same code -- the mistake that
    broke boundary-condition selection was two paths that were supposed to agree
    and quietly did not.

    `x` is the sweep/v2 layout: pos(3) sdf(1) normal(3) is_fixed(1) is_loaded(1)
    load_dir(3) poisson(1).
    """
    x = np.asarray(x, dtype=float)
    if x.shape[1] != 13:
        raise NeighbourIndexError(
            f"descriptors need the 13-column sweep/v2 layout, got {x.shape[1]}"
        )
    positions = x[:, :3]
    fixed = x[:, 7]
    loaded = x[:, 8]
    poisson = float(x[0, 12])

    # Everything below is measured in the part's own principal frame. An
    # axis-aligned bounding box is not rotation invariant -- the same part
    # arriving turned measures a different aspect ratio, solidity and
    # slenderness, so identical designs would be judged unlike each other.
    centred = positions - positions.mean(axis=0)
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    local = centred @ axes.T

    extent = local.max(axis=0) - local.min(axis=0)
    e = np.sort(np.maximum(extent, _EPS))[::-1]
    diagonal = float(np.linalg.norm(extent)) or 1.0

    volume = _volume(positions, np.asarray(tetrahedra, dtype=np.int64))
    area = _surface_area(positions, np.asarray(tetrahedra, dtype=np.int64))
    bbox_volume = float(np.prod(e)) or _EPS
    thinnest = (2.0 * volume / area) if area > 0.0 else float(e[2])

    # The load direction lives on the loaded nodes only; elsewhere it is zero.
    carrying = loaded > 0.5
    direction = x[carrying, 9:12].mean(axis=0) if carrying.any() else np.zeros(3)
    norm = float(np.linalg.norm(direction))
    direction = direction / norm if norm > 0 else direction
    in_frame = axes @ direction

    return np.array(
        [
            np.log(max(e[0] / e[2], _EPS)),
            np.log(max(e[1] / e[2], _EPS)),
            np.log(max(volume / bbox_volume, _EPS)),
            np.log(max(thinnest / diagonal, _EPS)),
            np.sqrt(float(fixed.mean())),
            np.sqrt(float(loaded.mean())),
            # Sign is arbitrary under SVD, so use magnitudes: a load pushing
            # along the long axis is the same problem either way round.
            abs(float(in_frame[0])),
            abs(float(in_frame[1])),
            abs(float(in_frame[2])),
            poisson,
        ],
        dtype=float,
    )


@dataclass(frozen=True)
class Neighbour:
    family: str
    index: int
    distance: float
    tau: float
    stress_mpa: float


@dataclass(frozen=True)
class Similarity:
    """What retrieval found, and whether it is reassuring."""

    neighbours: tuple[Neighbour, ...]
    nearest_distance: float
    extrapolation_threshold: float

    @property
    def extrapolating(self) -> bool:
        return self.nearest_distance > self.extrapolation_threshold

    def as_dict(self) -> dict[str, Any]:
        return {
            "nearest_distance": self.nearest_distance,
            "extrapolation_threshold": self.extrapolation_threshold,
            "extrapolating": self.extrapolating,
            "neighbours": [
                {
                    "family": n.family,
                    "index": n.index,
                    "distance": n.distance,
                    "true_stress_mpa": n.stress_mpa,
                }
                for n in self.neighbours
            ],
        }

    def notes(self) -> list[str]:
        if not self.neighbours:
            return []
        resembles = ", ".join(
            f"{n.family}-{n.index} ({n.stress_mpa:.4g} MPa)" for n in self.neighbours
        )
        out = [f"nearest training parts, with their solved stresses: {resembles}"]
        if self.extrapolating:
            out.append(
                f"the nearest training part is {self.nearest_distance:.2f} away "
                f"against a {EXTRAPOLATION_QUANTILE:.0%} training distance of "
                f"{self.extrapolation_threshold:.2f}: this part is unlike "
                "anything the model was fitted on, and the interval is "
                "extrapolating even though every declared bound is satisfied"
            )
        return out


class NeighbourIndex:
    """Descriptors of the training set, standardised, with their answers."""

    def __init__(
        self,
        descriptors: np.ndarray,
        families: Sequence[str],
        taus: np.ndarray,
        stresses: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        threshold: float,
    ) -> None:
        self.descriptors = descriptors
        self.families = list(families)
        self.taus = taus
        self.stresses = stresses
        self.mean = mean
        self.std = std
        self.threshold = threshold

    @classmethod
    def build(cls, samples: Iterable, quantile: float = EXTRAPOLATION_QUANTILE):  # noqa: ANN001
        rows, families, taus, stresses = [], [], [], []
        for sample in samples:
            rows.append(describe(sample.x, sample.tetrahedra))
            families.append(sample.family)
            taus.append(sample.tau)
            stresses.append(
                sample.tau * sample.force_n / (sample.length_mm ** 2)
            )
        if not rows:
            raise NeighbourIndexError("no samples to index")

        raw = np.array(rows, dtype=float)
        mean = raw.mean(axis=0)
        std = raw.std(axis=0)
        std[std < _EPS] = 1.0
        scaled = (raw - mean) / std

        # How far is a training part from its own nearest neighbour? That
        # distribution is what makes a query's distance interpretable.
        own = _pairwise_nearest(scaled)
        threshold = float(np.quantile(own, quantile))
        return cls(
            scaled, families, np.array(taus), np.array(stresses), mean, std, threshold
        )

    def query(self, descriptor: np.ndarray, k: int = 5) -> Similarity:
        scaled = (np.asarray(descriptor, dtype=float) - self.mean) / self.std
        distances = np.linalg.norm(self.descriptors - scaled, axis=1)
        order = np.argsort(distances)[: max(1, k)]
        return Similarity(
            neighbours=tuple(
                Neighbour(
                    family=self.families[i],
                    index=int(i),
                    distance=float(distances[i]),
                    tau=float(self.taus[i]),
                    stress_mpa=float(self.stresses[i]),
                )
                for i in order
            ),
            nearest_distance=float(distances[order[0]]),
            extrapolation_threshold=self.threshold,
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            descriptors=self.descriptors,
            families=np.array(self.families),
            taus=self.taus,
            stresses=self.stresses,
            mean=self.mean,
            std=self.std,
            threshold=np.array([self.threshold]),
            names=np.array(DESCRIPTOR_NAMES),
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "NeighbourIndex":
        source = Path(path)
        if not source.is_file():
            raise NeighbourIndexError(f"no neighbour index at {source}")
        with np.load(source, allow_pickle=False) as data:
            stored = tuple(str(n) for n in data["names"])
            if stored != DESCRIPTOR_NAMES:
                raise NeighbourIndexError(
                    "this index was built for a different descriptor; rebuild it"
                )
            return cls(
                descriptors=data["descriptors"],
                families=[str(f) for f in data["families"]],
                taus=data["taus"],
                stresses=data["stresses"],
                mean=data["mean"],
                std=data["std"],
                threshold=float(data["threshold"][0]),
            )


def _pairwise_nearest(scaled: np.ndarray, block: int = 512) -> np.ndarray:
    """Each row's distance to its nearest other row, in blocks to bound memory."""
    n = scaled.shape[0]
    out = np.empty(n, dtype=float)
    for start in range(0, n, block):
        stop = min(start + block, n)
        chunk = np.linalg.norm(scaled[start:stop, None, :] - scaled[None, :, :], axis=2)
        chunk[np.arange(stop - start), np.arange(start, stop)] = np.inf
        out[start:stop] = chunk.min(axis=1)
    return out


def build_from_sweep(root: str | Path, out: str | Path) -> dict[str, Any]:
    """Index a sweep directory. Run once, ship the result beside the weights."""
    from .generate.dataset import SweptDataset

    dataset = SweptDataset(root)
    index = NeighbourIndex.build(dataset)
    path = index.save(out)
    summary = {
        "samples": len(index.families),
        "set_sha256": dataset.set_sha256,
        "extrapolation_threshold": index.threshold,
        "descriptor": list(DESCRIPTOR_NAMES),
        "path": str(path),
    }
    Path(str(path) + ".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


__all__ = [
    "DESCRIPTOR_NAMES",
    "EXTRAPOLATION_QUANTILE",
    "Neighbour",
    "NeighbourIndex",
    "NeighbourIndexError",
    "Similarity",
    "build_from_sweep",
    "describe",
]
