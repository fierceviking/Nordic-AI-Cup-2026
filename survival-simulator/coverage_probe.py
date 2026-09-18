"""What does the Hive actually know about the map, and how fresh is it?

    python coverage_probe.py --seeds 11 12 --horizon 1200

Removing map omniscience from the oracle cost 642 points, so the binding
constraint looks like information acquisition rather than decision quality.
This measures the Hive's knowledge state directly:

  known      how many real trees/fruit the hive has a marker for
  fresh      how many of those were sensed recently enough to act on
             (fruit rots in ~50 s, trees die in ~58 s, so knowledge decays)
  accurate   markers that actually land on a real entity once the lineage
             frame is projected back into world coordinates
  sensed     share of the map inside somebody's sensors right now
  spread     mean pairwise agent distance, i.e. sweeping vs clustering

The hive navigates in a per-lineage frame with an unknown rotation, so every
comparison here re-projects that frame using a live agent as the anchor.
"""

import argparse
import json
import math
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

SAMPLE_EVERY = 25.0
TREE_MATCH = 45.0        # a tree marker this close to a real tree is "correct"
FRUIT_MATCH = 20.0
TREE_FRESH = 8.0         # seconds since last sighting that still counts as usable
FRUIT_FRESH = 3.0
GRID = 100.0             # map cell size for coverage accounting


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
    cols = int(env.width // GRID) + 1
    rows = int(env.height // GRID) + 1
    total_cells = cols * rows

    rows_out = []
    actions = []
    while True:
        state = sim.step(actions)
        actions = [(r.agent_id, r) for r in hive.act(state["observations"], state["sim_time"])]
        t = env.time
        if abs(t % SAMPLE_EVERY) > 0.05 or not env.agents:
            if state["num_agents"] == 0 or t > horizon:
                break
            continue

        # the biggest lineage is the one whose map is doing the work
        lineages = {}
        for agent in env.agents:
            mind = hive.minds.get(agent.agent_id)
            if mind is not None and mind.localized:
                lineages.setdefault(id(mind.world), []).append((agent, mind))
        if not lineages:
            if state["num_agents"] == 0 or t > horizon:
                break
            continue
        crew = max(lineages.values(), key=len)
        anchor_agent, anchor_mind = crew[0]
        world = anchor_mind.world

        rot = anchor_agent.direction - anchor_mind.th
        cos_r, sin_r = math.cos(rot), math.sin(rot)

        def project(lx, ly):
            dx, dy = lx - anchor_mind.x, ly - anchor_mind.y
            return (anchor_agent.x + cos_r * dx - sin_r * dy,
                    anchor_agent.y + sin_r * dx + cos_r * dy)

        def score_markers(records, actual, match, fresh_window):
            hits = fresh_hits = 0
            for entity in actual:
                best = None
                for rec in records:
                    px, py = project(rec[0], rec[1])
                    d = math.hypot(px - entity.x, py - entity.y)
                    if d <= match and (best is None or d < best[0]):
                        best = (d, rec)
                if best is not None:
                    hits += 1
                    if t - best[1][2] <= fresh_window:
                        fresh_hits += 1
            return hits, fresh_hits

        tree_hits, tree_fresh = score_markers(list(world.trees.values()),
                                              env.trees, TREE_MATCH, TREE_FRESH)
        fruit_hits, fruit_fresh = score_markers(list(world.fruits.values()),
                                                env.fruits, FRUIT_MATCH, FRUIT_FRESH)

        # what is inside anybody's sensors this instant
        sensed = set()
        for agent in env.agents:
            reach = max(agent.hearing_radius, agent.vision_radius * 0.5)
            cx, cy = agent.x, agent.y
            steps = int(reach // GRID) + 1
            for gx in range(-steps, steps + 1):
                for gy in range(-steps, steps + 1):
                    px, py = cx + gx * GRID, cy + gy * GRID
                    if 0 <= px < env.width and 0 <= py < env.height:
                        if math.hypot(px - cx, py - cy) <= reach:
                            sensed.add((int(px // GRID), int(py // GRID)))

        # cells the lineage has stood in recently, from the hive's own record
        recent = sum(1 for last in world.visited.values() if t - last <= 120.0)

        if len(env.agents) > 1:
            pairs = [math.hypot(a.x - b.x, a.y - b.y)
                     for i, a in enumerate(env.agents) for b in env.agents[i + 1:]]
            spread = statistics.mean(pairs)
            closest = min(pairs)
        else:
            spread = closest = 0.0

        rows_out.append({
            "t": round(t), "agents": len(env.agents),
            "real_trees": len(env.trees), "known_trees": tree_hits,
            "fresh_trees": tree_fresh,
            "real_fruit": len(env.fruits), "known_fruit": fruit_hits,
            "fresh_fruit": fruit_fresh,
            "markers_trees": len(world.trees), "markers_fruit": len(world.fruits),
            "sensed_cells_pct": round(100.0 * len(sensed) / total_cells, 1),
            "visited_recent_pct": round(100.0 * recent / total_cells, 1),
            "spread": round(spread), "closest": round(closest),
        })
        if state["num_agents"] == 0 or t > horizon:
            break

    return {"seed": seed, "score": round(state["score"], 1), "rows": rows_out}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12])
    parser.add_argument("--horizon", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--set", action="append", metavar="KEY=VALUE", default=[],
                        help="override a Hive tunable, repeatable")
    args = parser.parse_args()

    params = {}
    for item in args.set:
        key, _, raw = item.partition("=")
        raw = raw.strip()
        if raw.lower() in ("true", "false"):
            params[key.strip()] = raw.lower() == "true"
        else:
            params[key.strip()] = float(raw) if "." in raw else int(raw)
    if params:
        print(f"overrides: {params}")

    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_probe_one, (s, args.horizon, params)) for s in args.seeds]
        for future in as_completed(futures):
            reports.append(future.result())

    for report in sorted(reports, key=lambda r: r["seed"]):
        print(f"\n=== seed {report['seed']}  score {report['score']} ===")
        print(f"{'t':>5}{'agents':>7}{'trees kn/real':>15}{'fresh':>7}"
              f"{'fruit kn/real':>15}{'fresh':>7}{'sensed%':>9}{'visited%':>10}{'spread':>8}")
        for row in report["rows"]:
            if row["t"] % 100:
                continue
            trees = "{}/{}".format(row["known_trees"], row["real_trees"])
            fruit = "{}/{}".format(row["known_fruit"], row["real_fruit"])
            print(f"{row['t']:5d}{row['agents']:7d}{trees:>15}{row['fresh_trees']:7d}"
                  f"{fruit:>15}{row['fresh_fruit']:7d}"
                  f"{row['sensed_cells_pct']:9.1f}{row['visited_recent_pct']:10.1f}"
                  f"{row['spread']:8d}")

    every = [row for r in reports for row in r["rows"]]
    if every:
        print("\n=== averages across all samples ===")
        tk = sum(r["known_trees"] for r in every) / max(1, sum(r["real_trees"] for r in every))
        tf = sum(r["fresh_trees"] for r in every) / max(1, sum(r["real_trees"] for r in every))
        fk = sum(r["known_fruit"] for r in every) / max(1, sum(r["real_fruit"] for r in every))
        ff = sum(r["fresh_fruit"] for r in every) / max(1, sum(r["real_fruit"] for r in every))
        print(f"  trees the hive knows about : {tk:.1%}   still fresh: {tf:.1%}")
        print(f"  fruit the hive knows about : {fk:.1%}   still fresh: {ff:.1%}")
        print(f"  map inside sensors per tick: "
              f"{statistics.mean(r['sensed_cells_pct'] for r in every):.1f}%")
        print(f"  map walked in last 120 s   : "
              f"{statistics.mean(r['visited_recent_pct'] for r in every):.1f}%")
        print(f"  mean agent separation      : "
              f"{statistics.mean(r['spread'] for r in every):.0f} units "
              f"(map diagonal is 2000)")
        print(f"  closest pair, averaged     : "
              f"{statistics.mean(r['closest'] for r in every):.0f} units")
        print(f"  mean score                 : "
              f"{statistics.mean(r['score'] for r in reports):.1f}")


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _here)
    os.chdir(_here)
    main()
