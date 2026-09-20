"""The trained architecture.

Kept structurally identical to `notebooks/train_sfem.ipynb`. A state dict does
not carry its own shape, so a mismatch here fails to load or, worse, loads into
the wrong layout.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


# [pos(3) | sdf(1) | norm(3) | fixed_free(2) | load_onehot(3)]
INPUT_DIM = 12


class StressGNN(nn.Module):
    def __init__(self, in_dim: int = INPUT_DIM, hidden: int = 128, layers: int = 5):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([SAGEConv(hidden, hidden) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )

    def forward(self, data):  # noqa: ANN001 - torch_geometric Data
        h = self.inp(data.x)
        for conv, norm in zip(self.convs, self.norms):
            h = h + torch.nn.functional.silu(norm(conv(h, data.edge_index)))
        return self.out(h)


def load_model(weights: str | Path, *, device: str = "cpu") -> StressGNN:
    model = StressGNN()
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.to(device).eval()
    return model


__all__ = ["INPUT_DIM", "StressGNN", "load_model"]
