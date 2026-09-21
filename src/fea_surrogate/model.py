"""The trained architectures, one per feature version.

A state dict does not carry its own shape, and it does not carry its own
*names* either. The layer attributes below are flat — `inp`, `convs`, `norms` —
because that is what the training notebooks produced. Factoring the shared
trunk into a submodule renames every parameter to `trunk.*` and the checkpoint
stops loading, which is a refactor that looks harmless and is not. The
duplication between the two classes is deliberate and cheap by comparison.

    sfem/v1   12 inputs, per-node head; the peak is a max over node predictions
    sweep/v2  13 inputs, graph-level head; the peak is predicted directly
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_max_pool, global_mean_pool

from .versions import FEATURE_COLUMNS, SFEM_V1, SWEEP_V2, require_version


def _message_passing(module, data):  # noqa: ANN001
    """Residual SAGE trunk, shared as a function so parameter names stay flat."""
    h = module.inp(data.x)
    for conv, norm in zip(module.convs, module.norms):
        h = h + torch.nn.functional.silu(norm(conv(h, data.edge_index)))
    return h


class StressGNN(nn.Module):
    """sfem/v1. Predicts a per-node field."""

    def __init__(self, in_dim: int = 12, hidden: int = 128, layers: int = 5):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([SAGEConv(hidden, hidden) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )

    def forward(self, data):  # noqa: ANN001
        return self.out(_message_passing(self, data))


class PeakGNN(nn.Module):
    """sweep/v2. Predicts the peak directly from a pooled read of the graph.

    A maximum over noisy per-node estimates is a worse estimator of the peak
    than pooling the whole graph, and the peak is the scalar the schema carries.
    """

    def __init__(self, in_dim: int = 13, hidden: int = 128, layers: int = 5):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([SAGEConv(hidden, hidden) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )

    def forward(self, data):  # noqa: ANN001
        h = _message_passing(self, data)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(h.shape[0], dtype=torch.long, device=h.device)
        pooled = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], 1)
        return self.head(pooled).reshape(-1)


ARCHITECTURES = {SFEM_V1: StressGNN, SWEEP_V2: PeakGNN}


def load_model(weights: str | Path, *, version: str = SWEEP_V2, device: str = "cpu"):
    """Load a checkpoint, refusing a state dict that does not fit.

    `strict=True` is the point: a partial load would leave randomly initialised
    layers in place and still return a model that predicts numbers.
    """
    require_version(version)
    model = ARCHITECTURES[version](FEATURE_COLUMNS[version])
    state = torch.load(weights, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


__all__ = ["ARCHITECTURES", "PeakGNN", "StressGNN", "load_model"]
