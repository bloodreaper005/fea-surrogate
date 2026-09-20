"""Generate a labelled dataset by sweeping every family.

Run as a module:

    python -m fea_surrogate.generate.sweep --out data/sweep --samples 2000

Samples are written in shards rather than one file each, resumed by counting
the shards already on disk, and recorded in a manifest carrying a digest of the
shard bytes. That digest is what `calibration.set_sha256` refers to: a coverage
figure is only meaningful if the set it was measured on can be identified.

Some parameter draws produce geometry the mesher or the solver rejects. Those
are counted and skipped rather than retried, because a family whose failures
cluster in one corner of its design space is telling you something about the
family, and silently resampling would hide it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .families import ALL_FAMILIES
from .harness import draw
from .solver import solve


FAMILY_BY_NAME = {family.name: family for family in ALL_FAMILIES}
DEFAULT_SHARD_SIZE = 200


@dataclass(frozen=True)
class Drawn:
    """One solved sample, flattened for storage."""

    family: str
    x: np.ndarray
    y_nodal: np.ndarray
    tetrahedra: np.ndarray
    peak: float
    tau: float
    poisson: float
    force_n: float
    length_mm: float
    params: dict[str, float]


def _one(family_name: str, seed: int) -> Drawn | None:
    """Draw, mesh and solve a single sample. Returns None if it was rejected."""
    try:
        family = FAMILY_BY_NAME[family_name]()
        meshed = draw(family, np.random.default_rng(seed))
        result = solve(meshed)
        peak = float(result.peak_von_mises)
        tau = float(result.dimensionless_peak())
        if not (np.isfinite(peak) and np.isfinite(tau) and peak > 0.0):
            return None
        return Drawn(
            family=family_name,
            x=meshed.x.astype(np.float32),
            y_nodal=result.von_mises.astype(np.float32),
            tetrahedra=meshed.tetrahedra.astype(np.int32),
            peak=peak,
            tau=tau,
            poisson=float(meshed.sample.poisson_ratio),
            force_n=float(meshed.sample.force_n),
            length_mm=float(meshed.characteristic_length_mm),
            params=dict(meshed.sample.params),
        )
    except Exception:
        # Deliberately broad: gmsh raises bare Exception for a rejected mesh,
        # and a worker that dies takes the whole sweep with it.
        return None


def _write_shard(path: Path, samples: list[Drawn]) -> None:
    """Concatenate a shard's samples with offsets, rather than ragged arrays."""
    node_counts = np.array([len(s.x) for s in samples], dtype=np.int64)
    tet_counts = np.array([len(s.tetrahedra) for s in samples], dtype=np.int64)
    node_offsets = np.concatenate([[0], np.cumsum(node_counts)])
    tet_offsets = np.concatenate([[0], np.cumsum(tet_counts)])

    np.savez_compressed(
        path,
        x=np.concatenate([s.x for s in samples]),
        y_nodal=np.concatenate([s.y_nodal for s in samples]),
        tetrahedra=np.concatenate(
            [s.tetrahedra + node_offsets[i] for i, s in enumerate(samples)]
        ),
        node_offsets=node_offsets,
        tet_offsets=tet_offsets,
        peak=np.array([s.peak for s in samples], dtype=np.float64),
        tau=np.array([s.tau for s in samples], dtype=np.float64),
        poisson=np.array([s.poisson for s in samples], dtype=np.float64),
        force_n=np.array([s.force_n for s in samples], dtype=np.float64),
        length_mm=np.array([s.length_mm for s in samples], dtype=np.float64),
        family=np.array([s.family for s in samples]),
    )


def _digest_of(paths: Iterable[Path]) -> str:
    """One digest over the shard bytes, in a stable order."""
    running = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.name):
        running.update(path.name.encode("utf-8"))
        running.update(hashlib.sha256(path.read_bytes()).digest())
    return running.hexdigest()


def run(
    out_dir: Path,
    *,
    samples: int,
    families: list[str],
    workers: int,
    shard_size: int = DEFAULT_SHARD_SIZE,
    seed: int = 0,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("shard-*.npz"))
    done = len(existing) * shard_size
    if done >= samples:
        print(f"{done} samples already present in {out_dir}; nothing to do")
        return _write_manifest(out_dir, families, shard_size, seed)

    # Round-robin so an interrupted sweep still has every family represented.
    plan = [
        (families[i % len(families)], seed * 1_000_003 + i)
        for i in range(done, samples)
    ]
    print(f"drawing {len(plan)} samples across {len(families)} families "
          f"on {workers} workers")

    started = time.time()
    shard: list[Drawn] = []
    index = len(existing)
    rejected = 0
    completed = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, name, s): (name, s) for name, s in plan}
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result is None:
                rejected += 1
            else:
                shard.append(result)
            if len(shard) >= shard_size:
                _write_shard(out_dir / f"shard-{index:04d}.npz", shard)
                index += 1
                shard = []
            if completed % 50 == 0:
                rate = completed / max(time.time() - started, 1e-9)
                print(f"  {completed}/{len(plan)}  rejected {rejected}  "
                      f"{rate:.1f}/s")

    if shard:
        _write_shard(out_dir / f"shard-{index:04d}.npz", shard)

    print(f"done in {(time.time()-started)/60:.1f} min, rejected {rejected}")
    return _write_manifest(out_dir, families, shard_size, seed, rejected=rejected)


def _write_manifest(
    out_dir: Path,
    families: list[str],
    shard_size: int,
    seed: int,
    *,
    rejected: int = 0,
) -> dict[str, Any]:
    shards = sorted(out_dir.glob("shard-*.npz"))
    counts: dict[str, int] = {}
    total = 0
    for path in shards:
        with np.load(path, allow_pickle=False) as data:
            names, freq = np.unique(data["family"], return_counts=True)
            for name, n in zip(names, freq):
                counts[str(name)] = counts.get(str(name), 0) + int(n)
            total += len(data["tau"])

    manifest = {
        "schema_version": "FeaSurrogateSweep/v1",
        "samples": total,
        "by_family": counts,
        "rejected": rejected,
        "shards": [p.name for p in shards],
        "shard_size": shard_size,
        "seed": seed,
        "families_requested": families,
        "set_sha256": _digest_of(shards),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--families",
        nargs="*",
        default=sorted(FAMILY_BY_NAME),
        choices=sorted(FAMILY_BY_NAME),
    )
    args = parser.parse_args(argv)

    manifest = run(
        args.out,
        samples=args.samples,
        families=list(args.families),
        workers=args.workers,
        shard_size=args.shard_size,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
