"""Plots score distributions from `tune.py compare --out runs.json`.

    python tune.py compare --config variants_x.json --replicates 3 --out runs.json
    python plot_distributions.py runs.json [more.json ...] --reference default

Four panels:
  1. per-variant score distribution (box + every individual run)
  2. empirical CDF, with the 1000-point target marked
  3. bootstrap 95% CI on each variant's mean difference vs the reference
  4. P(score >= target) per variant, with bootstrap CI

Point of the exercise: a 10-seed mean has a standard error near +-50 here, so
differences smaller than roughly 150 are not resolvable and must not be read as
effects. Panels 3 and 4 show that directly instead of hiding it in a summary.
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TARGET = 1000.0
BOOTSTRAP = 20000
RNG = np.random.default_rng(12345)


def load(paths):
    groups, order = {}, []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        tag = Path(path).stem.replace("runs_", "")
        for run in payload["runs"]:
            label = f'{tag}:{run["variant"]}' if len(paths) > 1 else run["variant"]
            if label not in groups:
                groups[label] = []
                order.append(label)
            groups[label].append(run["score"])
    return {label: np.asarray(groups[label], dtype=float) for label in order}, order


def boot_mean(sample, n=BOOTSTRAP):
    draws = RNG.choice(sample, size=(n, sample.size), replace=True)
    return draws.mean(axis=1)


def boot_rate(sample, target, n=BOOTSTRAP):
    draws = RNG.choice(sample, size=(n, sample.size), replace=True)
    return (draws >= target).mean(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+")
    parser.add_argument("--reference", default=None,
                        help="variant used as the comparison baseline")
    parser.add_argument("--target", type=float, default=TARGET)
    parser.add_argument("--out", default="score_distributions.png")
    args = parser.parse_args()

    groups, order = load(args.results)
    reference = args.reference if args.reference in groups else order[0]
    ref = groups[reference]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"Score distributions  (reference: {reference}, target: {args.target:.0f})",
        fontsize=14, fontweight="bold")

    # --- 1. distribution + raw runs -----------------------------------
    ax = axes[0][0]
    data = [groups[label] for label in order]
    ax.boxplot(data, tick_labels=order, showmeans=True, widths=0.55)
    for index, label in enumerate(order, start=1):
        sample = groups[label]
        jitter = RNG.normal(0.0, 0.045, sample.size)
        ax.plot(index + jitter, sample, "o", alpha=0.45, markersize=4,
                color="tab:blue")
    ax.axhline(args.target, color="tab:red", linestyle="--", linewidth=1.2,
               label=f"target {args.target:.0f}")
    ax.set_ylabel("score")
    ax.set_title("Every run (box = quartiles, triangle = mean)")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # --- 2. ECDF -------------------------------------------------------
    ax = axes[0][1]
    for label in order:
        sample = np.sort(groups[label])
        ax.step(sample, np.arange(1, sample.size + 1) / sample.size,
                where="post", label=f"{label} (n={sample.size})")
    ax.axvline(args.target, color="tab:red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("score")
    ax.set_ylabel("P(score <= x)")
    ax.set_title("Empirical CDF - overlap means the variants are the same")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- 3. bootstrap CI on the mean difference ------------------------
    ax = axes[1][0]
    labels, centres, lows, highs = [], [], [], []
    ref_boot = boot_mean(ref)
    for label in order:
        if label == reference:
            continue
        diff = boot_mean(groups[label]) - ref_boot
        labels.append(label)
        centres.append(diff.mean())
        lows.append(np.percentile(diff, 2.5))
        highs.append(np.percentile(diff, 97.5))
    positions = np.arange(len(labels))
    errors = [np.array(centres) - np.array(lows), np.array(highs) - np.array(centres)]
    colours = ["tab:green" if low > 0 else "tab:red" if high < 0 else "tab:gray"
               for low, high in zip(lows, highs)]
    ax.errorbar(centres, positions, xerr=errors, fmt="o", capsize=4,
                ecolor="tab:gray", linestyle="none", color="black", zorder=3)
    for pos, centre, colour in zip(positions, centres, colours):
        ax.plot(centre, pos, "o", color=colour, markersize=9, zorder=4)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_yticks(positions, labels)
    ax.set_xlabel(f"mean score difference vs {reference}")
    ax.set_title("Bootstrap 95% CI - grey CI crossing 0 means 'not resolvable'")
    ax.grid(alpha=0.3, axis="x")

    # --- 4. probability of clearing the target -------------------------
    ax = axes[1][1]
    rates, rlow, rhigh = [], [], []
    for label in order:
        boot = boot_rate(groups[label], args.target)
        rates.append((groups[label] >= args.target).mean())
        rlow.append(np.percentile(boot, 2.5))
        rhigh.append(np.percentile(boot, 97.5))
    positions = np.arange(len(order))
    errors = [np.array(rates) - np.array(rlow), np.array(rhigh) - np.array(rates)]
    ax.bar(positions, rates, color="tab:blue", alpha=0.75)
    ax.errorbar(positions, rates, yerr=errors, fmt="none", ecolor="black", capsize=4)
    ax.set_xticks(positions, order)
    ax.set_ylabel(f"P(score >= {args.target:.0f})")
    ax.set_title("Chance of clearing the target in a single run")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")

    print(f"\n{'variant':22s} {'n':>3s} {'mean':>8s} {'median':>8s} {'sd':>7s} "
          f"{'95% CI of mean':>20s} {'P>=target':>10s}")
    for label in order:
        sample = groups[label]
        boot = boot_mean(sample)
        print(f"{label:22s} {sample.size:3d} {sample.mean():8.1f} "
              f"{statistics.median(sample):8.1f} "
              f"{(sample.std(ddof=1) if sample.size > 1 else 0.0):7.1f} "
              f"[{np.percentile(boot, 2.5):7.1f},{np.percentile(boot, 97.5):7.1f}] "
              f"{(sample >= args.target).mean():10.2f}")

    print(f"\nresolution check against '{reference}':")
    for label in order:
        if label == reference:
            continue
        diff = boot_mean(groups[label]) - ref_boot
        low, high = np.percentile(diff, 2.5), np.percentile(diff, 97.5)
        verdict = ("better" if low > 0 else "worse" if high < 0
                   else "NOT RESOLVABLE - consistent with no effect")
        print(f"  {label:20s} {diff.mean():+8.1f}  [{low:+7.1f},{high:+7.1f}]  {verdict}")


if __name__ == "__main__":
    main()
