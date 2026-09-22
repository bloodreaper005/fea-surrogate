"""Build the nearest-neighbour index and write it beside the weights.

    python -m tools.build_neighbour_index [sweep_dir] [out]

Run once per dataset. The index is what lets a screening run say which training
parts a query resembles, and whether it is further from all of them than a
training part usually is from its own neighbour.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fea_surrogate.neighbours import build_from_sweep  # noqa: E402


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "data" / "sweep"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "models" / "neighbours.npz"

    started = time.time()
    print(f"indexing {root}", flush=True)
    summary = build_from_sweep(root, out)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"done in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
