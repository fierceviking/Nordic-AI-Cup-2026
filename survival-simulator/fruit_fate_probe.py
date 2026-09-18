"""If a satiated agent skips a fruit, does anyone else actually get it?

    python fruit_fate_probe.py --seeds 11 12 13

Routing near-full agents around fruit only recovers energy if a hungrier agent
reaches that fruit before it rots (50 s after spawning). Otherwise the fruit
simply rots instead of being half-wasted, and the detour cost buys nothing.

There is no counterfactual to run, so this measures the closest observable
proxy: how many DISTINCT agents come within eating range of each fruit during
its life. If the average is ~1, skipping means rotting. If it is comfortably
above 1, the food is genuinely being contested and a skip transfers it.
"""

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

NEAR_FULL = 0.80          # fraction of max_energy above which a bite wastes a lot


def _probe_one(job):
    seed, horizon = job
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    from src.core import SimulationCore
    from src.utils.controllers.expert_agent_policy import Hive

    sim = SimulationCore(seed=seed)
    hive = Hive()
    hive.reset()
    env = sim.env

    visitors = defaultdict(set)       # id(fruit) -> agents that could have eaten it
    spawned = set()
    eaten_visitors = []               # distinct visitors for fruit that got eaten
    eaten_by_full_visitors = []       # ...specifically those eaten by a near-full agent
    rotted = eaten = 0
    contested_full = 0                # near-full bites where someone else also passed

    original_remove = env.remove_fruit
    pending = {}

    def patched_remove(fruit):
        key = id(fruit)
        if fruit.age > 100:
            pending[key] = ("rot", None)
        else:
            biter = None
            for agent in env.agents:
                if ((agent.x - fruit.x) ** 2 + (agent.y - fruit.y) ** 2
                        <= (agent.size + fruit.radius) ** 2):
                    biter = agent
                    break
            # The environment applies the min() clamp BEFORE calling us, so the
            # biter's energy is already post-meal here and is useless for
            # deciding whether it was full. Resolve that from the pre-step
            # snapshot instead.
            pending[key] = ("eat", None if biter is None else biter.agent_id)
        original_remove(fruit)

    env.remove_fruit = patched_remove

    actions = []
    while True:
        # Snapshot live fruit BEFORE the step, then test agents against those
        # positions AFTER it: agents move during agent_step and eat from the
        # post-move position, so sampling beforehand misses the eater entirely.
        live = [(id(f), f.x, f.y, f.radius) for f in env.fruits]
        for key, _, _, _ in live:
            spawned.add(key)
        pre_energy = {a.agent_id: (a.energy, a.max_energy) for a in env.agents}

        pending.clear()
        state = sim.step(actions)

        for key, fx, fy, radius in live:
            reach2 = (radius + 5.0) ** 2
            seen = visitors[key]
            for agent in env.agents:
                if ((agent.x - fx) ** 2 + (agent.y - fy) ** 2) <= reach2:
                    seen.add(agent.agent_id)

        for key, (fate, who) in pending.items():
            seen = len(visitors.pop(key, ()))
            if fate == "rot":
                rotted += 1
            else:
                eaten += 1
                eaten_visitors.append(seen)
                before = pre_energy.get(who) if who is not None else None
                if before is not None and before[0] >= before[1] * NEAR_FULL:
                    eaten_by_full_visitors.append(seen)
                    if seen > 1:
                        contested_full += 1

        actions = [(r.agent_id, r) for r in hive.act(state["observations"], state["sim_time"])]
        if state["num_agents"] == 0 or env.time > horizon:
            break

    n_full = len(eaten_by_full_visitors)
    return {
        "seed": seed,
        "score": round(state["score"], 1),
        "fruit_seen": len(spawned),
        "eaten": eaten,
        "rotted": rotted,
        "eaten_share": round(eaten / max(1, eaten + rotted), 3),
        "mean_visitors_eaten": round(statistics.mean(eaten_visitors), 2)
        if eaten_visitors else 0.0,
        "near_full_bites": n_full,
        "mean_visitors_near_full": round(statistics.mean(eaten_by_full_visitors), 2)
        if eaten_by_full_visitors else 0.0,
        "contested_near_full": contested_full,
        "contested_share": round(contested_full / max(1, n_full), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    parser.add_argument("--horizon", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_probe_one, (s, args.horizon)) for s in args.seeds]
        for future in as_completed(futures):
            report = future.result()
            reports.append(report)
            print(json.dumps(report), flush=True)

    eaten = sum(r["eaten"] for r in reports)
    rotted = sum(r["rotted"] for r in reports)
    full = sum(r["near_full_bites"] for r in reports)
    contested = sum(r["contested_near_full"] for r in reports)
    print("\n=== WOULD ANYONE ELSE HAVE EATEN IT? ===")
    print(f"  fruit eaten {eaten} | rotted {rotted} "
          f"({eaten / max(1, eaten + rotted):.1%} harvested)")
    print(f"  mean distinct agents in reach, per eaten fruit : "
          f"{statistics.mean([r['mean_visitors_eaten'] for r in reports]):.2f}")
    print(f"  ...for fruit taken by a NEAR-FULL agent        : "
          f"{statistics.mean([r['mean_visitors_near_full'] for r in reports]):.2f}")
    print(f"  near-full bites where someone else also passed : "
          f"{contested}/{full} ({contested / max(1, full):.1%})")
    print("\n  A share near 0% means skipping just lets it rot and the detour is wasted.")
    print("  A high share means the food is contested and a skip transfers it to a")
    print("  hungrier agent, which is the only way routing recovers energy.")


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _here)
    os.chdir(_here)
    main()
