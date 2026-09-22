"""Read a swept dataset back.

Deliberately numpy-only, so the shards can be inspected and tested without
torch installed. Converting to whatever a trainer wants is one line at the call
site.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class SweptSample:
    family: str
    x: np.ndarray            # (N, 13)
    y_nodal: np.ndarray      # (N,) von Mises, averaged onto nodes
    tetrahedra: np.ndarray   # (M, 4) local indices
    tau: float               # dimensionless critical-distance stress: the target
    tau_peak: float          # dimensionless raw peak, for comparison only
    peak: float              # raw peak von Mises in solver units
    poisson: float
    force_n: float
    length_mm: float

    def edge_index(self) -> np.ndarray:
        """Undirected edges of the tetrahedral mesh, deduplicated. (2, E)."""
        pairs = np.array(
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], dtype=np.int64
        )
        edges = self.tetrahedra[:, pairs].reshape(-1, 2)
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)
        return np.concatenate([edges, edges[:, ::-1]]).T


class SweptDataset:
    """Every sample in a sweep directory, read shard by shard."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"no manifest in {self.root}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.shards = [self.root / name for name in self.manifest["shards"]]

    def __len__(self) -> int:
        return int(self.manifest["samples"])

    @property
    def set_sha256(self) -> str:
        """The digest a calibration record must cite to be attributable."""
        return str(self.manifest["set_sha256"])

    def __iter__(self) -> Iterator[SweptSample]:
        for shard in self.shards:
            with np.load(shard, allow_pickle=False) as data:
                # Read each array once. Indexing an NpzFile decompresses the
                # whole array every time, so slicing inside the loop made this
                # quadratic: 500 samples took 794 seconds.
                x_all = data["x"]
                y_all = data["y_nodal"]
                tet_all = data["tetrahedra"]
                node_offsets = data["node_offsets"]
                tet_offsets = data["tet_offsets"]
                tau = data["tau"]
                tau_peak = data["tau_peak"] if "tau_peak" in data.files else tau
                peak = data["peak"]
                poisson = data["poisson"]
                force_n = data["force_n"]
                length_mm = data["length_mm"]
                family = data["family"]

            for i in range(len(tau)):
                lo, hi = node_offsets[i], node_offsets[i + 1]
                yield SweptSample(
                    family=str(family[i]),
                    x=x_all[lo:hi],
                    y_nodal=y_all[lo:hi],
                    # Shards store tetrahedra offset into the shard; make them
                    # local again so a sample stands on its own.
                    tetrahedra=tet_all[tet_offsets[i]:tet_offsets[i + 1]] - lo,
                    tau=float(tau[i]),
                    tau_peak=float(tau_peak[i]),
                    peak=float(peak[i]),
                    poisson=float(poisson[i]),
                    force_n=float(force_n[i]),
                    length_mm=float(length_mm[i]),
                )

    def taus(self) -> np.ndarray:
        """Every target, cheaply, without materialising the meshes."""
        values = []
        for shard in self.shards:
            with np.load(shard, allow_pickle=False) as data:
                values.append(data["tau"])
        return np.concatenate(values) if values else np.empty(0)


__all__ = ["SweptDataset", "SweptSample"]
