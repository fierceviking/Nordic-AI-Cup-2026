"""What makes a seed easy, and is "seed luck" even real?

    python seed_probe.py --seeds 1 2 3 ... --replicates 4

Two questions, because they have different consequences.

1. MAP COMPOSITION. `spawn_fruit_around_tree` rolls against the biome under the
   TREE: forest/grassland 0.1/s, swamp 0.08, desert 0.05, river 0.0. A map that
   is mostly river/desert produces a fraction of the food of a green one, before
   any policy runs. This is measured from the generated biome map alone, with no
   simulation, so it is nearly free.

2. VARIANCE DECOMPOSITION. The simulator is not reproducible across processes
   (`_get_local_objects` iterates sets hashed by id()), so the same seed does
   NOT give the same run. That means "seed 19 scored 1717" might be a property
   of the seed, or might be chaos that happened to land well. Running each seed
   several times separates them:

     between-seed variance >> within-seed  -> seeds genuinely differ, and it is
                                              worth asking what makes one good
     between ~= within                     -> there is no such thing as a good
                                              seed; it is run-to-run chaos and
                                              only the mean is addressable
"""

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed


def _map_stats(seed):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    from src.core import SimulationCore

    sim = SimulationCore(seed=seed)
    env = sim.env
    counts = defaultdict(int)
    step = 16                                   # sampling the grid is plenty
    for x in range(0, env.width, step):
        for y in range(0, env.height, step):
            counts[type(env.biome_map[x, y]).__name__] += 1
    total = sum(counts.values()) or 1
    rates = {n: env.biome_map[0, 0].fruit_spawn_rate for n in ()}   # placeholder
    # Weight each biome by its actual fruit rate to get one comparable number.
    yield_by_name = {"Forest_biome": 0.1, "Grassland_biome": 0.1,
                     "Swamp_biome": 0.08, "Desert_biome": 0.05, "River_biome": 0.0}
    productivity = sum(yield_by_name.get(n, 0.0) * c for n, c in counts.items()) / total
    return {
        "seed": seed,
        "trees_at_start": len(env.trees),
        "productivity": round(productivity, 4),
        "green_pct": round(100.0 * (counts["Forest_biome"] + counts["Grassland_biome"])
                           / total, 1),
        "river_pct": round(100.0 * counts["River_biome"] / total, 1),
        "desert_pct": round(100.0 * counts["Desert_biome"] / total, 1),
    }


def _run_one(job):
    seed, horizon = job
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    from src.core import SimulationCore
    from src.utils.controllers.expert_agent_policy import Hive

    sim = SimulationCore(seed=seed)
    hive = Hive()
    hive.reset()
    actions = []
    while True:
        state = sim.step(actions)
        actions = [(r.agent_id, r) for r in hive.act(state["observations"], state["sim_time"])]
        if state["num_agents"] == 0 or sim.env.time > horizon:
            break
    return seed, round(state["score"], 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(range(11, 21)))
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--maps-only", action="store_true")
    args = parser.parse_args()

    print("=== map composition (no simulation) ===")
    print(f"{'seed':>5}{'trees':>7}{'productivity':>14}{'green%':>9}"
          f"{'river%':>8}{'desert%':>9}")
    maps = {}
    for seed in args.seeds:
        info = _map_stats(seed)
        maps[seed] = info
        print(f"{info['seed']:5d}{info['trees_at_start']:7d}{info['productivity']:14.4f}"
              f"{info['green_pct']:9.1f}{info['river_pct']:8.1f}{info['desert_pct']:9.1f}")
    prods = [m["productivity"] for m in maps.values()]
    print(f"\n  productivity spread: {min(prods):.4f} to {max(prods):.4f} "
          f"({max(prods) / max(1e-9, min(prods)):.2f}x)")
    if args.maps_only:
        return

    jobs = [(s, args.horizon) for s in args.seeds for _ in range(args.replicates)]
    scores = defaultdict(list)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run_one, j) for j in jobs]
        for future in as_completed(futures):
            seed, score = future.result()
            scores[seed].append(score)
            print(json.dumps({"seed": seed, "score": score}), flush=True)

    print("\n=== per-seed scores ===")
    print(f"{'seed':>5}{'n':>4}{'mean':>9}{'sd':>8}{'min':>9}{'max':>9}"
          f"{'productivity':>14}")
    means = []
    within = []
    for seed in sorted(scores):
        vals = scores[seed]
        mean = statistics.mean(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        means.append(mean)
        if len(vals) > 1:
            within.append(statistics.pvariance(vals))
        print(f"{seed:5d}{len(vals):4d}{mean:9.1f}{sd:8.1f}{min(vals):9.1f}"
              f"{max(vals):9.1f}{maps[seed]['productivity']:14.4f}")

    between_sd = statistics.pstdev(means) if len(means) > 1 else 0.0
    within_sd = (statistics.mean(within) ** 0.5) if within else 0.0
    print("\n=== is 'seed luck' real? ===")
    print(f"  between-seed sd (of seed means): {between_sd:7.1f}")
    print(f"  within-seed  sd (same seed)    : {within_sd:7.1f}")
    if within_sd > 0:
        print(f"  ratio                          : {between_sd / within_sd:7.2f}")
    print("\n  ratio >> 1 -> seeds genuinely differ; ask what makes one good")
    print("  ratio ~ 1  -> no such thing as a good seed, only run-to-run chaos")

    if len(means) > 2:
        xs = [maps[s]["productivity"] for s in sorted(scores)]
        mx, my = statistics.mean(xs), statistics.mean(means)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, means))
        den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in means)) ** 0.5
        if den > 0:
            print(f"\n  correlation(map productivity, seed mean score): {num / den:+.2f}")
            print("  strong positive -> the map, not the policy, sets the score")


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _here)
    os.chdir(_here)
    main()
