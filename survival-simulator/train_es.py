"""Joint search over the Hive tunables with a noise-tolerant evolution strategy.

    python train_es.py --generations 30 --workers 7
    python train_es.py --resume                       # continue a checkpoint
    python train_es.py --report                       # show progress so far

Why this and not gradient RL: a single episode costs 60-130 s wall and the
score has sd ~200 on a mean of ~750, so any method needing 1e5+ episodes or a
clean per-step reward is unaffordable here.  SNES only needs the *ranking* of a
dozen candidates per generation, which survives that noise.

Two design points that matter more than the optimiser:

* Common random numbers.  Every candidate in a generation runs the SAME seed
  set, so candidates are compared on identical maps and the map-to-map variance
  (which dwarfs the parameter effect) largely cancels.
* The seed set rotates between generations, so the search cannot win by
  overfitting one map.

Single-parameter sweeps at n=20 could not resolve anything below about +-120
points.  The bet here is that many small effects exist and only compound when
searched jointly.
"""

import argparse
import contextlib
import io
import json
import math
import os
import statistics
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

CHECKPOINT = "es_state.json"
BEST_PATH = "es_best.json"

# name, low, high, kind
SPEC = [
    ("SPAWN_MIN",         150.0, 430.0, "float"),
    ("POP_MIN",             2.0,   8.0, "int"),
    ("POP_MAX",             5.0,  16.0, "int"),
    ("NET_GROW",            0.0,   2.0, "float"),
    ("NET_SHRINK",         -1.5,   0.3, "float"),
    ("SPAWN_COOLDOWN",      1.0,  15.0, "float"),
    ("AGING_COOLDOWN",      0.1,   8.0, "float"),
    ("AGING_DUMP",          0.0,   1.0, "bool"),
    ("MOVE_ON",            10.0,  60.0, "float"),
    ("FRUIT_RANGE",       150.0, 600.0, "float"),
    ("FRUIT_VALUE",        20.0,  60.0, "float"),
    ("FRUIT_MIN_VALUE",     0.0,  20.0, "float"),
    ("FRUIT_HYSTERESIS",    0.0,  15.0, "float"),
    ("FRUIT_DANGER",        0.0, 120.0, "float"),
    ("FRUIT_DANGER_RANGE", 80.0, 320.0, "float"),
    ("TREE_YIELD",          0.5,   5.0, "float"),
    ("YIELD_CAP",          10.0,  45.0, "float"),
    ("PATROL_RANGE",      150.0, 500.0, "float"),
    ("TREE_HYSTERESIS",     0.0,  15.0, "float"),
    ("TREE_CLAIMED",        0.0,  90.0, "float"),
    ("TREE_DANGER",         0.0, 160.0, "float"),
    ("TREE_DANGER_RANGE",  80.0, 340.0, "float"),
    ("RISK_AVERSION",       0.0,   3.0, "float"),
    ("RISK_ENERGY",       120.0, 450.0, "float"),
    ("SCAN_RATE",           0.1,   1.2, "float"),
]

SEED_POOL = tuple(range(101, 141))


def decode(z):
    """Unbounded vector -> Hive kwargs, squashed into each tunable's range."""
    params = {}
    for value, (name, lo, hi, kind) in zip(z, SPEC):
        unit = 1.0 / (1.0 + math.exp(-float(value)))
        raw = lo + (hi - lo) * unit
        if kind == "int":
            params[name] = int(round(raw))
        elif kind == "bool":
            params[name] = bool(raw >= 0.5)
        else:
            params[name] = float(raw)
    return params


def _eval_one(job):
    params, seed, horizon = job
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    try:
        from local_playground import local_simulation
        with contextlib.redirect_stdout(io.StringIO()):
            result = local_simulation(verbose=False, seed=int(seed), max_time=horizon,
                                      log_every=0, diagnostics=False, **params)
        return float(result["score"]), None
    except Exception:
        # Never let a crash look like a merely bad configuration: a silent 0.0
        # would corrupt the ranking the whole search depends on.
        return 0.0, traceback.format_exc()[-700:]


def evaluate_population(pool, population, seeds, horizon):
    jobs, owner = [], []
    for index, z in enumerate(population):
        params = decode(z)
        for seed in seeds:
            jobs.append((params, int(seed), horizon))
            owner.append(index)
    scores = [[] for _ in population]
    failures = []
    futures = {pool.submit(_eval_one, job): position for position, job in enumerate(jobs)}
    for future in as_completed(futures):
        score, error = future.result()
        scores[owner[futures[future]]].append(score)
        if error is not None:
            failures.append(error)
    return np.array([statistics.mean(s) if s else 0.0 for s in scores]), failures


def utilities(size):
    """Rank-based NES weights: only the ordering of candidates is used."""
    ranks = np.arange(1, size + 1)
    raw = np.maximum(0.0, math.log(size / 2.0 + 1.0) - np.log(ranks))
    if raw.sum() <= 0:
        return np.full(size, 1.0 / size) - 1.0 / size
    return raw / raw.sum() - 1.0 / size


def save(state, path):
    with open(path, "w") as handle:
        json.dump(state, handle, indent=1)


def train(args):
    dims = len(SPEC)
    rng = np.random.default_rng(args.seed)

    if args.resume and os.path.exists(CHECKPOINT):
        state = json.load(open(CHECKPOINT))
        mu = np.array(state["mu"])
        sigma = np.array(state["sigma"])
        history = state["history"]
        start = state["generation"]
        print(f"resumed at generation {start}")
    else:
        mu = np.zeros(dims)
        sigma = np.full(dims, args.sigma)
        history = []
        start = 0

    population_size = args.population or (4 + int(3 * math.log(dims)))
    weights = utilities(population_size)
    eta_mu = 1.0
    eta_sigma = (3.0 + math.log(dims)) / (5.0 * math.sqrt(dims))

    print(f"{dims} tunables, population {population_size}, "
          f"{args.seeds_per_eval} seeds/candidate, horizon {args.horizon}")
    print(f"{population_size * args.seeds_per_eval} episodes per generation")

    pool = ProcessPoolExecutor(max_workers=args.workers)
    best_score, best_params = -1e18, None
    if history:
        best_score = max(row["best"] for row in history)
        best_params = json.load(open(BEST_PATH)) if os.path.exists(BEST_PATH) else None

    try:
        for generation in range(start, args.generations):
            began = time.time()
            # common random numbers within a generation, rotated between them
            seeds = list(rng.choice(SEED_POOL, size=args.seeds_per_eval, replace=False))
            noise = rng.standard_normal((population_size, dims))
            population = mu + sigma * noise
            scores, failures = evaluate_population(pool, population, seeds, args.horizon)
            if failures:
                print(f"  {len(failures)} episode(s) FAILED this generation; "
                      f"first traceback:\n{failures[0]}", flush=True)
                if len(failures) > population_size * args.seeds_per_eval // 2:
                    raise RuntimeError("most episodes are failing; fix before searching")

            order = np.argsort(-scores)
            shaped = np.zeros(population_size)
            shaped[order] = weights

            gradient_mu = (shaped[:, None] * noise).sum(axis=0)
            gradient_sigma = (shaped[:, None] * (noise ** 2 - 1.0)).sum(axis=0)
            mu = mu + eta_mu * sigma * gradient_mu
            sigma = sigma * np.exp(0.5 * eta_sigma * gradient_sigma)
            sigma = np.clip(sigma, 0.03, 3.0)

            top = int(order[0])
            if scores[top] > best_score:
                best_score = float(scores[top])
                best_params = decode(population[top])
                save(best_params, BEST_PATH)

            centre = decode(mu)
            history.append({"generation": generation, "best": float(scores[top]),
                            "mean": float(scores.mean()), "worst": float(scores.min()),
                            "seeds": [int(s) for s in seeds],
                            "seconds": round(time.time() - began, 1)})
            save({"generation": generation + 1, "mu": mu.tolist(),
                  "sigma": sigma.tolist(), "history": history,
                  "centre": centre}, CHECKPOINT)

            print(f"gen {generation:3d}  best {scores[top]:7.1f}  "
                  f"mean {scores.mean():7.1f}  worst {scores.min():7.1f}  "
                  f"sigma {sigma.mean():.3f}  [{time.time()-began:.0f}s]", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted; checkpoint kept")
    finally:
        pool.shutdown()

    print(f"\nbest mean-over-seeds {best_score:.1f}, written to {BEST_PATH}")
    if best_params:
        for key in sorted(best_params):
            print(f"  {key} = {best_params[key]}")
    print("\nThis score is on the SEARCH seeds at the SEARCH horizon. Confirm it on "
          "held-out seeds before believing it:\n"
          f"  python tune.py compare --config <variants.json> --replicates 2 "
          f"--seeds 11 12 13 14 15 16 17 18 19 20 --out runs_es.json")


def report():
    if not os.path.exists(CHECKPOINT):
        print("no checkpoint yet")
        return
    state = json.load(open(CHECKPOINT))
    print(f"generation {state['generation']}")
    print(f"{'gen':>4s} {'best':>8s} {'mean':>8s} {'worst':>8s} {'s':>6s}")
    for row in state["history"]:
        print(f"{row['generation']:4d} {row['best']:8.1f} {row['mean']:8.1f} "
              f"{row['worst']:8.1f} {row['seconds']:6.0f}")
    print("\ncentre of the search distribution:")
    for key in sorted(state["centre"]):
        print(f"  {key} = {state['centre'][key]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population", type=int, default=0, help="0 = 4 + 3*ln(dims)")
    parser.add_argument("--seeds-per-eval", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--sigma", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    report() if args.report else train(args)


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _here)
    os.chdir(_here)
    main()
