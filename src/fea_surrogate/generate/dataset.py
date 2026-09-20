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
    tau: float               # dimensionless peak, the training target
    peak: float              # peak von Mises in solver units
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
                node_offsets = data["node_offsets"]
                tet_offsets = data["tet_offsets"]
                for i in range(len(data["tau"])):
                    lo, hi = node_offsets[i], node_offsets[i + 1]
                    tlo, thi = tet_offsets[i], tet_offsets[i + 1]
                    yield SweptSample(
                        family=str(data["family"][i]),
                        x=data["x"][lo:hi],
                        y_nodal=data["y_nodal"][lo:hi],
                        # Shards store tetrahedra offset into the shard; make
                        # them local again so a sample stands on its own.
                        tetrahedra=data["tetrahedra"][tlo:thi] - lo,
                        tau=float(data["tau"][i]),
                        peak=float(data["peak"][i]),
                        poisson=float(data["poisson"][i]),
                        force_n=float(data["force_n"][i]),
                        length_mm=float(data["length_mm"][i]),
                    )

    def taus(self) -> np.ndarray:
        """Every target, cheaply, without materialising the meshes."""
        values = []
        for shard in self.shards:
            with np.load(shard, allow_pickle=False) as data:
                values.append(data["tau"])
        return np.concatenate(values) if values else np.empty(0)


__all__ = ["SweptDataset", "SweptSample"]
