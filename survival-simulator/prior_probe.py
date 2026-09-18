"""Does tree knowledge convert into fruit knowledge?

    python prior_probe.py --seeds 11 12 13 --horizon 1200 --set FRUIT_PRIOR=1.0

The hive knows ~27% of trees but only ~8% of fruit.  A tree observation is
predictive: the environment rolls `dt * biome.fruit_spawn_rate` per tree per
second, so a known patch implies a standing crop nobody has looked at.  This
measures whether routing on that prior actually finds more food, rather than
just scoring differently.

Reported per run:

  discovered    unique real fruit that ever came inside somebody's sensors
  eaten         meals, split by the mode that walked the agent into range.
                An agent flips to `fruit` the instant it senses food, so the
                mode at the moment of eating is always `fruit` and says
                nothing; what matters is the mode it held just before that.
                  patrol/hold -> the tree map delivered the find
                  explore     -> blind search delivered it
  tree visits   distinct times an agent closed on a real tree
  latency       seconds from first sighting a tree to standing on it.  Only
                counts trees that were eventually visited, so it is a lower
                bound, useful for comparison rather than in absolute terms.

With SCOUT enabled it also traces the causal chain a scout is supposed to
produce, because discovering trees is worthless unless the food arrives:

    scout finds tree -> tree enters the map -> forager visits it
    -> fruit found near it -> fruit eaten

so discoveries are counted per scout-second, split into productive trees
(ones that ever had fruit near them) and barren ones, and every meal is
attributed to whoever first found the tree it was eaten beside.
"""

import argparse
import math
import os
import statistics
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

AT_TREE = 46.0
MEAL_JUMP = 10.0         # energy gain no cost model can explain except a fruit
FRUIT_NEAR_TREE = 62.0   # fruit spawns 20-60 units from its tree
DT = 0.1


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

    discovered = set()
    meals_by_mode = Counter()
    meal_energy = 0.0
    prev_energy = {}
    approach = {}            # agent -> last mode held before it locked onto a fruit
    at_tree = set()          # (agent_id, tree id) pairs already counted
    tree_visits = 0
    latencies = []
    seen_visits = {}         # id(world) -> set of tree keys already timed
    pop_trace = []
    found_by = {}            # id(tree) -> "scout" | "forager", whoever saw it first
    productive = set()       # trees that ever had fruit beside them
    visited_trees = set()
    scout_seconds = 0.0
    meals_by_finder = Counter()

    actions = []
    state = sim.step(actions)
    while True:
        actions = [(r.agent_id, r) for r in hive.act(state["observations"], state["sim_time"])]
        state = sim.step(actions)
        t = env.time

        # --- what the sensors touched this tick -------------------------
        for agent in env.agents:
            reach = max(agent.hearing_radius, agent.vision_radius * 0.5)
            mind = hive.minds.get(agent.agent_id)
            scouting = mind is not None and mind.mode == "scout"
            if scouting:
                scout_seconds += DT
            for fruit in env.fruits:
                if math.hypot(fruit.x - agent.x, fruit.y - agent.y) <= reach:
                    discovered.add(id(fruit))
            for tree in env.trees:
                d = math.hypot(tree.x - agent.x, tree.y - agent.y)
                if d <= reach and id(tree) not in found_by:
                    found_by[id(tree)] = "scout" if scouting else "forager"
                if d < AT_TREE:
                    visited_trees.add(id(tree))
                    tag = (agent.agent_id, id(tree))
                    if tag not in at_tree:
                        at_tree.add(tag)
                        tree_visits += 1

        # a tree only counts as productive once it actually bears something
        for fruit in env.fruits:
            for tree in env.trees:
                if math.hypot(tree.x - fruit.x, tree.y - fruit.y) < FRUIT_NEAR_TREE:
                    productive.add(id(tree))
                    break

        # --- meals, attributed to the mode that produced them ------------
        for agent in env.agents:
            was = prev_energy.get(agent.agent_id)
            if was is not None and agent.energy - was > MEAL_JUMP:
                meals_by_mode[approach.get(agent.agent_id, "?")] += 1
                meal_energy += agent.energy - was
                # credit the meal to whoever first put this patch on the map
                best, best_d = None, FRUIT_NEAR_TREE
                for tree in env.trees:
                    d = math.hypot(tree.x - agent.x, tree.y - agent.y)
                    if d < best_d:
                        best, best_d = tree, d
                meals_by_finder[found_by.get(id(best), "unknown") if best else "no tree"] += 1
            prev_energy[agent.agent_id] = agent.energy
            mind = hive.minds.get(agent.agent_id)
            if mind is not None and mind.mode != "fruit":
                approach[agent.agent_id] = mind.mode

        # --- how long a tree waits between being seen and being worked ---
        for mind in hive.minds.values():
            world = mind.world
            timed = seen_visits.setdefault(id(world), set())
            for key, visit_t in world.visits.items():
                if key in timed:
                    continue
                rec = world.trees.get(key)
                if rec is not None:
                    timed.add(key)
                    latencies.append(max(0.0, visit_t - rec[3]))

        # predator kills and starvation are already counted by tune.py compare
        if abs(t % 50.0) < 0.05:
            pop_trace.append((round(t), len(env.agents), len(env.fruits), len(env.trees)))

        if state["num_agents"] == 0 or t > horizon:
            break

    real_fruit_total = len(discovered)
    scout_trees = [k for k, who in found_by.items() if who == "scout"]
    return {
        "seed": seed,
        "score": round(state["score"], 1),
        "discovered": real_fruit_total,
        "meals": sum(meals_by_mode.values()),
        "meal_energy": round(meal_energy),
        "by_mode": dict(meals_by_mode),
        "tree_visits": tree_visits,
        "latency_median": round(statistics.median(latencies), 1) if latencies else 0.0,
        "latency_n": len(latencies),
        "pop": pop_trace,
        "trees_found": len(found_by),
        "trees_found_by_scout": len(scout_trees),
        "scout_trees_productive": sum(1 for k in scout_trees if k in productive),
        "scout_trees_visited": sum(1 for k in scout_trees if k in visited_trees),
        "scout_seconds": round(scout_seconds, 1),
        "meals_by_finder": dict(meals_by_finder),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    parser.add_argument("--horizon", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--set", action="append", metavar="KEY=VALUE", default=[])
    args = parser.parse_args()

    params = {}
    for item in args.set:
        key, _, raw = item.partition("=")
        raw = raw.strip()
        if raw.lower() in ("true", "false"):
            params[key.strip()] = raw.lower() == "true"
        else:
            params[key.strip()] = float(raw)
    print(f"overrides: {params or 'none (baseline)'}")

    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_probe_one, (s, args.horizon, params)) for s in args.seeds]
        for future in as_completed(futures):
            reports.append(future.result())
    reports.sort(key=lambda r: r["seed"])

    for r in reports:
        print(f"\n=== seed {r['seed']}  score {r['score']} ===")
        print(f"  fruit discovered : {r['discovered']}")
        print(f"  meals            : {r['meals']}  ({r['meal_energy']} energy)")
        print(f"  meals by approach: {r['by_mode']}")
        print(f"  tree visits      : {r['tree_visits']}")
        print(f"  sight->visit     : median {r['latency_median']}s over {r['latency_n']} trees")
        print("   t  agents fruit trees")
        for (t, n, f, tr) in r["pop"]:
            if t % 200 == 0:
                print(f"  {t:5d} {n:6d} {f:5d} {tr:5d}")

    modes = Counter()
    for r in reports:
        modes.update(r["by_mode"])
    total_meals = sum(modes.values()) or 1
    tree_meals = modes["patrol"] + modes["hold"]
    print("\n=== totals ===")
    print(f"  mean score           : {statistics.mean(r['score'] for r in reports):.1f}")
    print(f"  fruit discovered     : {sum(r['discovered'] for r in reports)}")
    print(f"  meals                : {total_meals}")
    print(f"  energy eaten         : {sum(r['meal_energy'] for r in reports)}")
    print(f"  tree visits          : {sum(r['tree_visits'] for r in reports)}")
    print(f"  meals by approach    : {dict(modes)}")
    print(f"  tree-map meals       : {tree_meals} ({tree_meals / total_meals:.1%})")
    lat = [r["latency_median"] for r in reports if r["latency_n"]]
    if lat:
        print(f"  sight->visit median  : {statistics.mean(lat):.1f}s")

    scout_secs = sum(r["scout_seconds"] for r in reports)
    if scout_secs > 0:
        found = sum(r["trees_found_by_scout"] for r in reports)
        prod = sum(r["scout_trees_productive"] for r in reports)
        seen = sum(r["scout_trees_visited"] for r in reports)
        finders = Counter()
        for r in reports:
            finders.update(r["meals_by_finder"])
        print("\n=== scout causal chain ===")
        print(f"  scout-seconds spent  : {scout_secs:.0f}")
        print(f"  trees found by scouts: {found} of "
              f"{sum(r['trees_found'] for r in reports)} total")
        print(f"    of those productive: {prod}   later visited: {seen}")
        print(f"  productive trees per 100 scout-seconds: "
              f"{100.0 * prod / scout_secs:.2f}")
        print(f"  meals by tree finder : {dict(finders)}")
        scout_meals = finders["scout"]
        total = sum(finders.values()) or 1
        print(f"  meals at scout-found trees: {scout_meals} ({scout_meals / total:.1%})")


if __name__ == "__main__":
    sys.exit(main())
