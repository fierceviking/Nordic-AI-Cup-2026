"""Quantifies the three behaviours reported from watching the hive.

    python behaviour_probe.py --seeds 11 12 13

1. STUCK      ticks where the policy commanded real movement but the agent
              barely moved, i.e. _steer() failed against an obstacle.
2. IGNORED    known fruit the policy discarded because of FRUIT_RANGE /
              FRUIT_MIN_VALUE while the agent was idling on a patch.
3. PREDATION  for every agent actually eaten: was the predator ever in the
              hive's memory, and how close was it when first believed?
              Plus when predators actually appear over the run.
"""

import argparse
import contextlib
import io
import json
import math
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

STUCK_COMMAND = 2.0      # only judge ticks where we asked for real movement
STUCK_FRACTION = 0.25    # moved less than this share of what was commanded
CHARGE_DIST = 90.0       # predator charges unconditionally inside this


def _probe_one(job):
    seed, horizon = job
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    from src.core import SimulationCore
    from src.utils.controllers.expert_agent_policy import Hive, BIOME_MOVE

    sim = SimulationCore(seed=seed)
    hive = Hive()
    hive.reset()

    stuck_ticks = commanded_ticks = 0
    stuck_by_mode = {}
    ignored_samples = []          # nearest known fruit that was out of range
    idle_with_food = idle_ticks = 0
    eaten_records = []
    predator_timeline = []
    snapshots = {}

    eaten_ids = []
    original_kill = sim.env.kill_agent
    # target in TRUE coordinates: a blocked agent still dead-reckons as if it
    # moved, so the hive's own frame would report progress that never happened.
    pending = {}
    noprog_ticks = noprog_near_wall = tracked_ticks = 0
    noprog_by_mode = {}

    def patched_kill(agent):
        if agent in sim.env.agents:
            if any(math.hypot(p.x - agent.x, p.y - agent.y) < p.size + agent.size + 2
                   for p in sim.env.predators):
                eaten_ids.append(agent.agent_id)
        original_kill(agent)

    sim.env.kill_agent = patched_kill

    actions = []
    while True:
        before = {a.agent_id: (a.x, a.y, a.speed,
                               BIOME_MOVE.get(sim.env.biome_map[
                                   min(max(int(a.x), 0), sim.env.width - 1),
                                   min(max(int(a.y), 0), sim.env.height - 1)].type, 1.0))
                  for a in sim.env.agents}
        commanded = {aid: act.move_distance for aid, act in actions}

        eaten_ids.clear()
        state = sim.step(actions)

        # 3. anyone eaten this tick: what did the hive know a tick earlier?
        for aid in eaten_ids:
            snap = snapshots.get(aid)
            if snap is not None:
                eaten_records.append(snap)

        # 1. commanded movement vs what actually happened
        for agent in sim.env.agents:
            aid = agent.agent_id
            if aid not in before or aid not in commanded:
                continue
            want = commanded[aid]
            if want < STUCK_COMMAND:
                continue
            px, py, speed, pen = before[aid]
            expected = min(want, speed) * pen
            moved = math.hypot(agent.x - px, agent.y - py)
            commanded_ticks += 1
            if expected > 0.5 and moved < STUCK_FRACTION * expected:
                stuck_ticks += 1
                mind = hive.minds.get(aid)
                mode = getattr(mind, "mode", "?")
                stuck_by_mode[mode] = stuck_by_mode.get(mode, 0) + 1

        # 1b. did the agent actually close on the thing it was heading for?
        for agent in sim.env.agents:
            record = pending.get(agent.agent_id)
            if record is None:
                continue
            tx, ty, was, expected, mode, near_wall = record
            now = math.hypot(tx - agent.x, ty - agent.y)
            tracked_ticks += 1
            if expected > 0.5 and (was - now) < STUCK_FRACTION * expected:
                noprog_ticks += 1
                noprog_by_mode[mode] = noprog_by_mode.get(mode, 0) + 1
                if near_wall:
                    noprog_near_wall += 1
        pending.clear()

        requests = hive.act(state["observations"], state["sim_time"])
        actions = [(r.agent_id, r) for r in requests]
        issued = {r.agent_id: r for r in requests}

        # 2 and 3. per-agent snapshot taken after the policy has run
        for agent in sim.env.agents:
            mind = hive.minds.get(agent.agent_id)
            if mind is None:
                continue
            request = issued.get(agent.agent_id)
            record = None
            if mind.target_fruit is not None:
                record = mind.world.fruits.get(mind.target_fruit)
            elif mind.target_tree is not None:
                record = mind.world.trees.get(mind.target_tree)
            if record is not None and request is not None and request.move_distance >= STUCK_COMMAND:
                rot = agent.direction - mind.th
                cos_r, sin_r = math.cos(rot), math.sin(rot)
                dx, dy = record[0] - mind.x, record[1] - mind.y
                tx = agent.x + cos_r * dx - sin_r * dy
                ty = agent.y + sin_r * dx + cos_r * dy
                biome = sim.env.biome_map[min(max(int(agent.x), 0), sim.env.width - 1),
                                          min(max(int(agent.y), 0), sim.env.height - 1)]
                expected = min(request.move_distance, agent.speed) * BIOME_MOVE.get(biome.type, 1.0)
                near_wall = any(
                    min(abs(agent.x - x1), abs(agent.x - x2),
                        abs(agent.y - y1), abs(agent.y - y2)) < 45.0
                    for (x1, y1, x2, y2) in mind.edges)
                pending[agent.agent_id] = (tx, ty, math.hypot(tx - agent.x, ty - agent.y),
                                           expected, mind.mode, near_wall)
            true_pred = min((math.hypot(p.x - agent.x, p.y - agent.y)
                             for p in sim.env.predators), default=1e9)
            believed = min((math.hypot(p[0] - mind.x, p[1] - mind.y)
                            for p in mind.world.predators), default=1e9)
            snapshots[agent.agent_id] = {
                "mode": mind.mode,
                "true_pred": true_pred,
                "believed_pred": believed,
                "known_predators": len(mind.world.predators),
                "energy": agent.energy,
            }
            if mind.mode in ("hold", "patrol", "explore"):
                idle_ticks += 1
                out_of_range = [math.hypot(rec[0] - mind.x, rec[1] - mind.y)
                                for rec in mind.world.fruits.values()]
                far = [d for d in out_of_range if d > hive.FRUIT_RANGE]
                if far:
                    idle_with_food += 1
                    ignored_samples.append(min(far))

        if int(round(state["sim_time"] * 10)) % 250 == 0:
            predator_timeline.append((round(state["sim_time"]), len(sim.env.predators)))

        if state["num_agents"] == 0 or sim.env.time > horizon:
            break

    ambushed = [r for r in eaten_records if r["true_pred"] < CHARGE_DIST
                and r["believed_pred"] > 1e8]
    knew = [r for r in eaten_records if r["believed_pred"] < 1e8]
    fleeing = [r for r in eaten_records if r["mode"] == "flee"]
    return {
        "seed": seed,
        "score": round(state["score"], 1),
        "sim_time": round(sim.env.time, 1),
        "stuck_ticks": stuck_ticks,
        "commanded_ticks": commanded_ticks,
        "stuck_rate": round(stuck_ticks / max(1, commanded_ticks), 4),
        "stuck_by_mode": stuck_by_mode,
        "tracked_ticks": tracked_ticks,
        "noprogress_ticks": noprog_ticks,
        "noprogress_rate": round(noprog_ticks / max(1, tracked_ticks), 4),
        "noprogress_near_wall": noprog_near_wall,
        "noprogress_by_mode": noprog_by_mode,
        "idle_ticks": idle_ticks,
        "idle_with_out_of_range_food": idle_with_food,
        "idle_food_rate": round(idle_with_food / max(1, idle_ticks), 4),
        "median_ignored_distance": round(statistics.median(ignored_samples), 1)
                                   if ignored_samples else None,
        "eaten": len(eaten_records),
        "eaten_while_fleeing": len(fleeing),
        "eaten_predator_never_seen": len(ambushed),
        "eaten_predator_known": len(knew),
        "predator_timeline": predator_timeline[:13],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    parser.add_argument("--horizon", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_probe_one, (seed, args.horizon)) for seed in args.seeds]
        for future in as_completed(futures):
            report = future.result()
            reports.append(report)
            timeline = report.pop("predator_timeline")
            print(json.dumps(report), flush=True)
            report["predator_timeline"] = timeline

    def total(key):
        return sum(r[key] for r in reports)

    print("\n=== 1. STUCK AGAINST GEOMETRY ===")
    print(f"  {total('stuck_ticks')} of {total('commanded_ticks')} commanded ticks "
          f"({total('stuck_ticks')/max(1,total('commanded_ticks')):.2%}) moved "
          f"<{STUCK_FRACTION:.0%} of what was asked")
    merged = {}
    for report in reports:
        for mode, count in report["stuck_by_mode"].items():
            merged[mode] = merged.get(mode, 0) + count
    print(f"  by mode: {merged}")

    print("\n=== 1b. FAILED TO CLOSE ON ITS TARGET (true coordinates) ===")
    print(f"  {total('noprogress_ticks')} of {total('tracked_ticks')} targeted ticks "
          f"({total('noprogress_ticks')/max(1,total('tracked_ticks')):.2%}) closed "
          f"<{STUCK_FRACTION:.0%} of the distance they should have")
    print(f"  of those, {total('noprogress_near_wall')} had a remembered wall within 45 units "
          f"({total('noprogress_near_wall')/max(1,total('noprogress_ticks')):.0%})")
    merged = {}
    for report in reports:
        for mode, count in report["noprogress_by_mode"].items():
            merged[mode] = merged.get(mode, 0) + count
    print(f"  by mode: {merged}")

    print("\n=== 2. KNOWN FOOD DISCARDED WHILE IDLE ===")
    print(f"  {total('idle_with_out_of_range_food')} of {total('idle_ticks')} idle "
          f"agent-ticks ({total('idle_with_out_of_range_food')/max(1,total('idle_ticks')):.2%}) "
          f"had remembered fruit beyond FRUIT_RANGE")
    distances = [r["median_ignored_distance"] for r in reports if r["median_ignored_distance"]]
    if distances:
        print(f"  median distance to the nearest discarded fruit: "
              f"{statistics.median(distances):.0f} units (FRUIT_RANGE = 220)")

    print("\n=== 3. PREDATION ===")
    eaten = total("eaten")
    print(f"  eaten {eaten}")
    print(f"    already fleeing when caught : {total('eaten_while_fleeing')} "
          f"({total('eaten_while_fleeing')/max(1,eaten):.0%})")
    print(f"    predator was in memory      : {total('eaten_predator_known')} "
          f"({total('eaten_predator_known')/max(1,eaten):.0%})")
    print(f"    never detected, inside {CHARGE_DIST:.0f}u : {total('eaten_predator_never_seen')} "
          f"({total('eaten_predator_never_seen')/max(1,eaten):.0%})")

    print("\n=== PREDATOR COUNT OVER TIME ===")
    print("   t(s)  " + "  ".join(f"{t:5d}" for t, _ in reports[0]["predator_timeline"]))
    for report in reports:
        print(f"  seed{report['seed']:3d} "
              + "  ".join(f"{n:5d}" for _, n in report["predator_timeline"]))


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _here)
    os.chdir(_here)
    main()
