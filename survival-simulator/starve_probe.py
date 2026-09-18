"""Are agents starving in place next to food they refused to walk to?

    python starve_probe.py --seeds 11 12 --horizon 1200

Observed in the viewer: hungry agents circling instead of feeding. The suspect
is the PRIORITIZE_FOOD affordability gate in _plan, which skips any fruit whose
trip cost is not provably payable:

    required = travel * WALK_COST / pen + ticks * drain + 1.0
    if required >= energy: continue

When no fruit passes, the agent falls through to tree patrol, reaches a trunk,
sets spinning=True and scans in place until it dies. This counts how often that
happens and how much food was within reach when it did.
"""

import argparse
import math
import os
import statistics
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

DT = 0.1
WALK_COST = 0.05
FRUIT_REACH = 4.0
BIOME_MOVE = {"forest": 1.0, "grassland": 1.0, "swamp": 0.5, "desert": 0.8, "river": 0.3}


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

    modes = Counter()
    hungry_ticks = 0
    hungry_with_food = 0        # food known and in range, but not targeted
    hungry_refused = 0          # and the affordability gate is why
    refused_gap = []            # how short of `required` the agent was
    refused_dist = []
    starved_beside_food = 0
    last_seen = {}

    actions = []
    state = sim.step(actions)
    while True:
        reqs = hive.act(state["observations"], state["sim_time"])
        actions = [(r.agent_id, r) for r in reqs]
        t = env.time

        for agent in env.agents:
            mind = hive.minds.get(agent.agent_id)
            if mind is None or not mind.localized:
                continue
            modes[mind.mode] += 1
            # "hungry" = under a fifth of capacity, i.e. also sprint-locked
            if agent.energy > agent.max_energy * 0.2:
                last_seen[agent.agent_id] = (agent.energy, mind.mode, 1e9)
                continue
            hungry_ticks += 1

            pen = BIOME_MOVE.get(
                env.biome_map[min(max(int(agent.x), 0), env.width - 1),
                              min(max(int(agent.y), 0), env.height - 1)].__class__.__name__
                .replace("_biome", "").lower(), 1.0)
            speed = agent.speed

            nearest = 1e9
            affordable = False
            best_gap = None
            for rec in mind.world.fruits.values():
                d = math.hypot(rec[0] - mind.x, rec[1] - mind.y)
                if d > hive.FRUIT_RANGE:
                    continue
                nearest = min(nearest, d)
                travel = max(0.0, d - FRUIT_REACH)
                ticks = math.ceil(travel / max(speed * pen, 1e-6))
                drain = DT + (0.01 * (agent.age + ticks * DT) if mind.aging else 0.0)
                required = travel * WALK_COST / pen + ticks * drain + 1.0
                gap = required - agent.energy
                if gap < 0:
                    affordable = True
                elif best_gap is None or gap < best_gap:
                    best_gap = gap
            if nearest < 1e9:
                hungry_with_food += 1
                if not affordable and mind.target_fruit is None:
                    hungry_refused += 1
                    if best_gap is not None:
                        refused_gap.append(best_gap)
                    refused_dist.append(nearest)
            last_seen[agent.agent_id] = (agent.energy, mind.mode, nearest)

        alive = {a.agent_id for a in env.agents}
        state = sim.step(actions)
        for aid in alive - {a.agent_id for a in env.agents}:
            energy, mode, nearest = last_seen.get(aid, (99.0, "?", 1e9))
            if energy < 20.0 and nearest < hive.FRUIT_RANGE:
                starved_beside_food += 1

        if state["num_agents"] == 0 or t > horizon:
            break

    total_modes = sum(modes.values()) or 1
    return {
        "seed": seed,
        "score": round(state["score"], 1),
        "hungry_ticks": hungry_ticks,
        "hungry_with_food": hungry_with_food,
        "hungry_refused": hungry_refused,
        "starved_beside_food": starved_beside_food,
        "median_gap": round(statistics.median(refused_gap), 1) if refused_gap else 0.0,
        "median_dist": round(statistics.median(refused_dist), 0) if refused_dist else 0.0,
        "hold_pct": round(100.0 * modes["hold"] / total_modes, 1),
        "modes": {k: round(100.0 * v / total_modes, 1) for k, v in modes.most_common()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12])
    parser.add_argument("--horizon", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=2)
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

    for r in reports:
        print(f"\n=== seed {r['seed']}  score {r['score']} ===")
        print(f"  hungry agent-ticks        : {r['hungry_ticks']}")
        print(f"  ...with fruit in range    : {r['hungry_with_food']}")
        print(f"  ...refused as unaffordable: {r['hungry_refused']}")
        print(f"  median shortfall          : {r['median_gap']} energy")
        print(f"  median distance to it     : {r['median_dist']} units")
        print(f"  died hungry beside fruit  : {r['starved_beside_food']}")
        print(f"  mode split                : {r['modes']}")

    hungry = sum(r["hungry_ticks"] for r in reports) or 1
    withf = sum(r["hungry_with_food"] for r in reports)
    refused = sum(r["hungry_refused"] for r in reports)
    print("\n=== totals ===")
    print(f"  hungry ticks with fruit in range : {withf} of {hungry} ({withf / hungry:.1%})")
    print(f"  of those, refused as unaffordable: {refused} ({refused / max(1, withf):.1%})")
    print(f"  died hungry beside fruit         : "
          f"{sum(r['starved_beside_food'] for r in reports)}")
    gaps = [r["median_gap"] for r in reports if r["median_gap"]]
    if gaps:
        print(f"  median energy shortfall          : {statistics.mean(gaps):.1f}")


if __name__ == "__main__":
    sys.exit(main())
