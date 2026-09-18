"""Population curve: is there an interior optimum, and what shape is it?

    python popcurve.py runs_popsize.json

Score is elapsed time and the run ends only at extinction, so population has no
direct score value -- only option value (redundancy, reproduction, sensing)
against a direct energy cost.  That predicts an interior optimum rather than a
monotone curve, and the shape tells us which force dominates.

Mean score alone cannot distinguish these: a level that usually collapses early
but occasionally runs very long can share a mean with one that reliably reaches
the middle.  So this also reports the upper tail, which is what actually wins a
three-run evaluation.
"""

import argparse
import json
import statistics
import sys

import numpy as np

BOOTSTRAP = 20000
TARGETS = (1000.0, 1500.0, 2000.0)


def boot_ci(sample, stat, n=BOOTSTRAP, rng=None):
    rng = rng or np.random.default_rng(12345)
    arr = np.asarray(sample, dtype=float)
    draws = rng.choice(arr, size=(n, arr.size), replace=True)
    vals = stat(draws)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", help="raw runs json from tune.py compare")
    parser.add_argument("--order", nargs="+", default=None,
                        help="variant names in population order")
    args = parser.parse_args()

    with open(args.runs, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict):
        raw = raw["runs"]

    by_variant = {}
    for row in raw:
        by_variant.setdefault(row["variant"], []).append(row)

    names = args.order or sorted(by_variant)
    names = [n for n in names if n in by_variant]

    print(f"{'variant':<16}{'n':>4}{'mean':>8}{'median':>8}{'sd':>7}"
          f"{'95% CI of mean':>20}{'max':>8}")
    for name in names:
        scores = [r["score"] for r in by_variant[name]]
        lo, hi = boot_ci(scores, lambda d: d.mean(axis=1))
        print(f"{name:<16}{len(scores):>4}{statistics.mean(scores):>8.1f}"
              f"{statistics.median(scores):>8.1f}"
              f"{statistics.pstdev(scores):>7.1f}"
              f"{f'[{lo:7.1f},{hi:7.1f}]':>20}{max(scores):>8.1f}")

    print(f"\n{'variant':<16}" + "".join(f"{f'P>={int(t)}':>22}" for t in TARGETS))
    for name in names:
        scores = np.asarray([r["score"] for r in by_variant[name]], dtype=float)
        cells = []
        for target in TARGETS:
            rate = float((scores >= target).mean())
            lo, hi = boot_ci(scores, lambda d, t=target: (d >= t).mean(axis=1))
            cells.append(f"{rate:.2f} [{lo:.2f},{hi:.2f}]".rjust(22))
        print(f"{name:<16}" + "".join(cells))

    # Extinction is the only thing that ends a run, so deaths are the mechanism.
    print(f"\n{'variant':<16}{'births':>8}{'starved':>9}{'eaten':>8}"
          f"{'young':>7}{'survived':>10}{'starve/death':>14}")
    for name in names:
        rows = by_variant[name]
        births = sum(r["births"] for r in rows)
        starved = sum(r["starved"] for r in rows)
        eaten = sum(r["eaten"] for r in rows)
        young = sum(r["young_deaths"] for r in rows)
        survived = sum(1 for r in rows if r.get("alive", 0) > 0)
        deaths = starved + eaten
        print(f"{name:<16}{births:>8.0f}{starved:>9}{eaten:>8}{young:>7}"
              f"{survived:>10}{(starved / deaths if deaths else 0):>14.2f}")

    best = max(names, key=lambda n: statistics.mean(
        [r["score"] for r in by_variant[n]]))
    print(f"\nhighest mean: {best}")
    ref = names[0]
    ref_scores = [r["score"] for r in by_variant[ref]]
    print(f"\nresolution check against '{ref}':")
    rng = np.random.default_rng(999)
    a = np.asarray(ref_scores, dtype=float)
    for name in names[1:]:
        b = np.asarray([r["score"] for r in by_variant[name]], dtype=float)
        da = rng.choice(a, size=(BOOTSTRAP, a.size), replace=True).mean(axis=1)
        db = rng.choice(b, size=(BOOTSTRAP, b.size), replace=True).mean(axis=1)
        diff = db - da
        lo, hi = float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))
        verdict = "RESOLVED" if (lo > 0 or hi < 0) else "not resolvable"
        print(f"  {name:<16}{b.mean() - a.mean():>+8.1f}  [{lo:>+7.1f},{hi:>+7.1f}]  {verdict}")


if __name__ == "__main__":
    sys.exit(main())
