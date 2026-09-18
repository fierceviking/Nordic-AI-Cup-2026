"""How many parents actually compete for each birth?

    python breed_funnel_probe.py --seeds 11 12 13 --horizon 1500

Measured elsewhere: the hive's breeding preference applies only -1.70 energy of
pressure on max_energy per selection event, against mutation steps of +-250 and
a net upward viability drift of +25.8. Reweighting `_fitness` therefore cannot
steer the trait.

The suspected reason is that selection needs something to select FROM. A birth
picks from `pool`, which survives four filters in _spawn_plan:

    every living agent
      -> post-move energy > 112 and off spawn cooldown      (cands)
      -> energy > SPAWN_MIN                                 (rich enough)
      -> mind.mode == "hold"                                (sitting on a patch)
      -> no predator within 320 units                       (safe site)

If the pool is almost always a single agent, then weight(A)=1 and weight(A)=1000
produce identical behaviour and the evolutionary channel is structurally
powerless, regardless of the fitness function.

This wraps _spawn_plan and recomputes each stage, so the policy is untouched.
"""

import argparse
import math
import os
import statistics
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed


def _probe_one(job):
    seed, horizon, params = job
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    from src.core import SimulationCore
    from src.utils.controllers.expert_agent_policy import Hive

    sim = SimulationCore(seed=seed)
    hive = Hive(**params)
    hive.reset()
    env = sim.env

    stages = Counter()
    opportunities = 0
    pool_sizes = Counter()
    pool_spread = []          # max_energy range within the pool, when >1
    original = hive._spawn_plan

    def wrapped(ctx, plans, t, pop):
        nonlocal opportunities
        # Replicates the filters in _spawn_plan; the policy itself is untouched.
        alive = len(ctx)
        cands = []
        for mind, st in ctx:
            after = plans[mind.aid]["after"]
            cooldown = hive.AGING_COOLDOWN if mind.aging else hive.SPAWN_COOLDOWN
            if after <= 112.0 or t - mind.last_spawn < cooldown:
                continue
            cands.append((mind, st, after))

        rich = [c for c in cands if c[2] > hive.SPAWN_MIN]
        holding = [c for c in rich if hive._good_birth_site(c[0])]
        safe = [c for c in holding
                if not any(math.hypot(p[0] - c[0].x, p[1] - c[0].y) < 320.0
                           for p in c[0].world.predators)]

        opportunities += 1
        stages["alive"] += alive
        stages["off_cooldown"] += len(cands)
        stages["rich_enough"] += len(rich)
        stages["holding"] += len(holding)
        stages["safe_site"] += len(safe)
        pool_sizes[min(len(safe), 4)] += 1
        if len(safe) > 1:
            vals = [c[1]["max_energy"] for c in safe]
            pool_spread.append(max(vals) - min(vals))

        return original(ctx, plans, t, pop)

    hive._spawn_plan = wrapped

    actions = []
    state = sim.step(actions)
    while True:
        actions = [(r.agent_id, r) for r in hive.act(state["observations"], state["sim_time"])]
        state = sim.step(actions)
        if state["num_agents"] == 0 or env.time > horizon:
            break

    n = max(1, opportunities)
    return {
        "seed": seed,
        "score": round(state["score"], 1),
        "opportunities": opportunities,
        "per_opportunity": {k: round(v / n, 2) for k, v in stages.items()},
        "pool_sizes": dict(pool_sizes),
        "mean_pool_spread": round(statistics.mean(pool_spread), 1) if pool_spread else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    parser.add_argument("--horizon", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--set", action="append", metavar="KEY=VALUE", default=[])
    args = parser.parse_args()

    params = {}
    for item in args.set:
        key, _, raw = item.partition("=")
        raw = raw.strip()
        params[key.strip()] = (raw.lower() == "true") if raw.lower() in ("true", "false") \
            else float(raw)
    print(f"overrides: {params or 'none (baseline)'}")

    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_probe_one, (s, args.horizon, params)) for s in args.seeds]
        for future in as_completed(futures):
            reports.append(future.result())
    reports.sort(key=lambda r: r["seed"])

    total = Counter()
    sizes = Counter()
    for r in reports:
        print(f"\n=== seed {r['seed']}  score {r['score']}  "
              f"{r['opportunities']} birth opportunities ===")
        for stage in ("alive", "off_cooldown", "rich_enough", "holding", "safe_site"):
            print(f"    {stage:14s}{r['per_opportunity'].get(stage, 0):6.2f} agents")
            total[stage] += r["per_opportunity"].get(stage, 0)
        for k, v in r["pool_sizes"].items():
            sizes[k] += v

    print("\n=== the parent-choice funnel (mean agents surviving each filter) ===")
    for stage in ("alive", "off_cooldown", "rich_enough", "holding", "safe_site"):
        print(f"  {stage:14s}{total[stage] / len(reports):6.2f}")

    n = sum(sizes.values()) or 1
    print("\n=== how many parents actually compete ===")
    for k in sorted(sizes):
        label = "4+" if k == 4 else str(k)
        print(f"  {label:>3} candidate(s): {sizes[k]:6d}  ({sizes[k] / n:5.1%})")
    one_or_none = (sizes.get(0, 0) + sizes.get(1, 0)) / n
    print(f"\n  births with NO real choice (0 or 1 candidate): {one_or_none:.1%}")
    spread = statistics.mean([r["mean_pool_spread"] for r in reports])
    print(f"  mean max_energy spread within a contested pool : {spread:.1f}")
    print("\n  If the no-choice share is high, weight(A)=1 and weight(A)=1000 are the")
    print("  same policy, and the evolutionary channel is structurally powerless.")


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _here)
    os.chdir(_here)
    main()
