"""Measure the deployed surrogate's coverage against the solver, all families.

    python -m tools.deployed_coverage 100 6

This is the acceptance gate. Run it after any change to boundary-condition
selection, the feature builder, or the model, and require every family back near
its nominal 0.90 with median pred/true near 1.0.

It is not a unit test and does not belong in `tests/`: it needs a trained model,
a solver, and roughly twenty minutes. It is the only thing that measures the path
a user actually takes.

This is the path a user actually takes: a STEP file arrives with no face tags,
boundary conditions come from geometric selectors, and the model has to find the
same answer the solver does. The sweep that trained the model did none of that --
it tagged faces by construction -- so agreement here is not implied by the
training metrics, and the isolated-node defect proved that the hard way.

Geometry and solving happen in worker processes; screening happens serially in
the parent, because loading torch into every worker costs more than it saves.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("FEA_SURROGATE_VALIDATION_DIR", REPO / "data" / "validation"))
STEPS = WORK / "steps"
OUT = WORK / "deployed_coverage.json"

# Far from the training sweep's seeds (0..2999 from base 0), so every part here
# is geometry the model has never seen.
SEED_BASE = 5_000_000


def _families():
    from fea_surrogate.generate.families import ALL_FAMILIES

    return {f.name: f for f in ALL_FAMILIES}


def _one(family_name: str, seed: int) -> dict | None:
    """Draw, solve, and write the STEP a screening run will be given."""
    try:
        import gmsh

        from fea_surrogate.generate.harness import _gmsh_model, draw
        from fea_surrogate.generate.solver import solve

        family = _families()[family_name]()
        meshed = draw(family, np.random.default_rng(seed))
        result = solve(meshed)

        truth = float(result.critical_distance_stress())
        if not np.isfinite(truth) or truth <= 0.0:
            return None

        step = STEPS / f"{family_name}-{seed}.step"
        with _gmsh_model(family_name):
            family.build(dict(meshed.sample.params))
            gmsh.model.occ.synchronize()
            gmsh.write(str(step))

        return {
            "family": family_name,
            "seed": seed,
            "step": str(step),
            "force_n": float(meshed.sample.force_n),
            "length_mm": float(meshed.characteristic_length_mm),
            "truth_mpa": truth,
            "peak_mpa": float(result.peak_von_mises),
            "poisson": float(meshed.sample.poisson_ratio),
            "nodes": int(len(meshed.vertices)),
        }
    except Exception:
        # Same reason the sweep swallows these: gmsh raises bare Exception for a
        # rejected draw, and a dead worker takes the run with it.
        return None


def main() -> None:
    per_family = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    STEPS.mkdir(parents=True, exist_ok=True)

    names = sorted(_families())
    plan = [
        (name, SEED_BASE + i * 977 + k)
        for k, name in enumerate(names)
        for i in range(per_family)
    ]
    print(f"solving {len(plan)} parts across {len(names)} families "
          f"on {workers} workers", flush=True)

    started = time.time()
    solved: list[dict] = []
    rejected = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, n, s) for n, s in plan]
        for done, future in enumerate(as_completed(futures), 1):
            row = future.result()
            if row is None:
                rejected += 1
            else:
                solved.append(row)
            if done % 25 == 0:
                rate = done / max(time.time() - started, 1e-9)
                left = (len(plan) - done) / max(rate, 1e-9) / 60.0
                print(f"  solved {done}/{len(plan)}  rejected {rejected}  "
                      f"{rate:.2f}/s  ~{left:.1f} min left", flush=True)

    print(f"solved {len(solved)}, rejected {rejected}, "
          f"in {(time.time()-started)/60:.1f} min", flush=True)

    from fea_surrogate.predict import Surrogate, screen_step

    surrogate = Surrogate.load("models/sweep_gnn.pt", "models/calibration.json")
    print(f"screening through {surrogate.model_name}@{surrogate.model_version}",
          flush=True)

    rows = []
    screen_started = time.time()
    for i, row in enumerate(solved, 1):
        try:
            document = screen_step(
                surrogate,
                row["step"],
                force_n=row["force_n"],
                family=row["family"],
                characteristic_length_mm=row["length_mm"],
                # The sweep drew this per sample; screening with a default
                # would measure a part the solver never solved.
                poisson_ratio=row["poisson"],
            )
        except Exception as exc:
            rows.append({**row, "error": f"{type(exc).__name__}: {exc}"})
            continue
        prediction = document["predictions"][0]
        lower, upper = float(prediction["lower"]), float(prediction["upper"])
        truth = row["truth_mpa"]
        rows.append({
            **row,
            "lower": lower,
            "upper": upper,
            "centre": float(np.sqrt(max(lower, 1e-12) * upper)),
            "inside": bool(lower <= truth <= upper),
        })
        if i % 25 == 0:
            print(f"  screened {i}/{len(solved)}  "
                  f"{i/max(time.time()-screen_started,1e-9):.2f}/s", flush=True)

    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    report(rows)


def report(rows: list[dict]) -> None:
    ok = [r for r in rows if "inside" in r]
    errors = [r for r in rows if "error" in r]
    print()
    print(f"{'family':16}{'n':>5}{'coverage':>10}{'median p/t':>12}"
          f"{'p5 p/t':>9}{'p95 p/t':>9}{'errors':>8}")
    for name in sorted({r["family"] for r in rows}):
        group = [r for r in ok if r["family"] == name]
        bad = [r for r in errors if r["family"] == name]
        if not group:
            print(f"{name:16}{0:>5}{'-':>10}{'-':>12}{'-':>9}{'-':>9}{len(bad):>8}")
            continue
        inside = np.mean([r["inside"] for r in group])
        ratio = np.array([r["centre"] / r["truth_mpa"] for r in group])
        print(f"{name:16}{len(group):>5}{inside:>10.2f}"
              f"{np.median(ratio):>12.2f}{np.percentile(ratio, 5):>9.2f}"
              f"{np.percentile(ratio, 95):>9.2f}{len(bad):>8}")
    if ok:
        overall = np.mean([r["inside"] for r in ok])
        se = float(np.sqrt(0.9 * 0.1 / len(ok)))
        print(f"\noverall coverage {overall:.3f} on n={len(ok)}; "
              f"calibrated for 0.90, standard error {se:.3f}")
    for r in errors[:5]:
        print(f"  error {r['family']}-{r['seed']}: {r['error']}")


if __name__ == "__main__":
    os.chdir(REPO)
    main()
