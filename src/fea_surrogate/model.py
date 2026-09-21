"""The trained architectures, one per feature version.

A state dict does not carry its own shape, and the two versions differ in both
input width and head. Selecting by what the calibration record declares keeps a
model paired with the feature construction it was fitted with.

    sfem/v1   12 inputs, per-node head; the peak is a max over node predictions
    sweep/v2  13 inputs, graph-level head; the peak is predicted directly
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_max_pool, global_mean_pool

from .versions import FEATURE_COLUMNS, SFEM_V1, SWEEP_V2, require_version


class _Trunk(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([SAGEConv(hidden, hidden) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])

    def forward(self, data):  # noqa: ANN001
        h = self.inp(data.x)
        for conv, norm in zip(self.convs, self.norms):
            h = h + torch.nn.functional.silu(norm(conv(h, data.edge_index)))
        return h


class StressGNN(nn.Module):
    """sfem/v1. Predicts a per-node field."""

    def __init__(self, in_dim: int = 12, hidden: int = 128, layers: int = 5):
        super().__init__()
        self.trunk = _Trunk(in_dim, hidden, layers)
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )

    def forward(self, data):  # noqa: ANN001
        return self.out(self.trunk(data))


class PeakGNN(nn.Module):
    """sweep/v2. Predicts the peak directly from a pooled read of the graph.

    A maximum over noisy per-node estimates is a worse estimator of the peak
    than pooling the whole graph, and the peak is the scalar the schema carries.
    """

    def __init__(self, in_dim: int = 13, hidden: int = 128, layers: int = 5):
        super().__init__()
        self.trunk = _Trunk(in_dim, hidden, layers)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )

    def forward(self, data):  # noqa: ANN001
        h = self.trunk(data)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(h.shape[0], dtype=torch.long, device=h.device)
        pooled = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], 1)
        return self.head(pooled).reshape(-1)


ARCHITECTURES = {SFEM_V1: StressGNN, SWEEP_V2: PeakGNN}


def load_model(weights: str | Path, *, version: str = SWEEP_V2, device: str = "cpu"):
    require_version(version)
    model = ARCHITECTURES[version](FEATURE_COLUMNS[version])
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.to(device).eval()
    return model


__all__ = ["ARCHITECTURES", "PeakGNN", "StressGNN", "load_model"]
