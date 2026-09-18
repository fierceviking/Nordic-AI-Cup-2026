"""Exactly how much harvested fruit energy is destroyed by eating while full.

    python overflow_probe.py --seeds 11 12 13

Eating is involuntary: `agent.energy = min(max_energy, energy + fruit.energy)`
fires on contact, so an agent at 480/500 that touches a ripe 60-energy fruit
annihilates 40 of it.  The clamp happens before assignment, so it cannot be
observed directly -- instead this reconstructs each agent's energy at the
instant of the bite by replaying the exact cost formulas the environment uses,
then reports the shortfall.
"""

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

DT = 0.1
WALK_COST = 0.05
SPRINT_COST = 0.5
SPAWN_COST = 100.0


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

    offered = wasted = 0.0
    bites = full_bites = 0
    headroom = []
    eaten_now = defaultdict(list)
    # A bite is only fixable by reassignment if the agent CHOSE that fruit.
    # Waste from walking over fruit en route is a pathing problem instead, so
    # the split decides which intervention is even applicable.
    targeted_waste = incidental_waste = 0.0
    targeted_bites = incidental_bites = 0

    original_remove = sim.env.remove_fruit

    def patched_remove(fruit):
        if fruit.age <= 100:                      # not rot: somebody ate it
            for agent in sim.env.agents:
                if ((agent.x - fruit.x) ** 2 + (agent.y - fruit.y) ** 2
                        <= (agent.size + fruit.radius) ** 2):
                    mind = hive.minds.get(agent.agent_id)
                    chosen = False
                    if mind is not None and mind.target_fruit is not None:
                        rec = mind.world.fruits.get(mind.target_fruit)
                        if rec is not None:
                            # The hive navigates in a lineage-local frame, so the
                            # target is compared against the agent's own pose in
                            # that frame rather than against world coordinates.
                            chosen = math.hypot(rec[0] - mind.x, rec[1] - mind.y) < 12.0
                    eaten_now[agent.agent_id].append((fruit.energy, chosen))
                    break
        original_remove(fruit)

    sim.env.remove_fruit = patched_remove

    actions = []
    while True:
        # energy at the top of the tick, before agent_step charges anything
        start = {a.agent_id: (a.energy, a.max_energy, a.speed, a.sprint_speed,
                              a.age, a.max_age) for a in sim.env.agents}
        commanded = {aid: act for aid, act in actions}
        eaten_now.clear()

        state = sim.step(actions)

        for aid, fruits in eaten_now.items():
            if aid not in start:
                continue
            energy, cap, speed, sprint, age, max_age = start[aid]
            act = commanded.get(aid)
            if act is not None:
                # replay update_entity_position's clamps and cost exactly
                distance = max(0.0, min(act.move_distance, sprint))
                if energy < cap / 5.0 and distance > speed:
                    distance = speed
                energy -= (distance * WALK_COST if distance <= speed
                           else speed * WALK_COST + (distance - speed) * SPRINT_COST)
                energy -= min(math.pi, abs(act.turn_angle)) / (2.0 * math.pi)
                if act.spawn_agent and energy > 100.0:
                    energy -= SPAWN_COST
            energy -= DT                                   # living drain
            if age + DT > max_age:
                energy -= 0.01 * (age + DT)                # aging surcharge
            for value, chosen in fruits:                   # bites resolve in order
                spare = max(0.0, cap - energy)
                taken = min(value, spare)
                offered += value
                wasted += value - taken
                bites += 1
                headroom.append(spare)
                if chosen:
                    targeted_bites += 1
                    targeted_waste += value - taken
                else:
                    incidental_bites += 1
                    incidental_waste += value - taken
                if value - taken > 0.5:
                    full_bites += 1
                energy = min(cap, energy + value)

        actions = [(r.agent_id, r) for r in hive.act(state["observations"], state["sim_time"])]
        if state["num_agents"] == 0 or sim.env.time > horizon:
            break

    return {
        "seed": seed,
        "score": round(state["score"], 1),
        "bites": bites,
        "offered": round(offered),
        "absorbed": round(offered - wasted),
        "wasted": round(wasted),
        "waste_share": round(wasted / max(1.0, offered), 4),
        "bites_with_waste": full_bites,
        "bites_with_waste_share": round(full_bites / max(1, bites), 4),
        "median_headroom": round(statistics.median(headroom), 1) if headroom else None,
        "mean_fruit_value": round(offered / max(1, bites), 1),
        "targeted_bites": targeted_bites,
        "incidental_bites": incidental_bites,
        "targeted_waste": round(targeted_waste),
        "incidental_waste": round(incidental_waste),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    parser.add_argument("--horizon", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_probe_one, (s, args.horizon)) for s in args.seeds]
        for future in as_completed(futures):
            report = future.result()
            reports.append(report)
            print(json.dumps(report), flush=True)

    offered = sum(r["offered"] for r in reports)
    wasted = sum(r["wasted"] for r in reports)
    bites = sum(r["bites"] for r in reports)
    spoiled = sum(r["bites_with_waste"] for r in reports)
    print("\n=== EATING WHILE FULL ===")
    print(f"  {bites} bites, {offered} energy offered, {offered - wasted} absorbed")
    print(f"  DESTROYED: {wasted}  ({wasted / max(1, offered):.1%} of everything harvested)")
    print(f"  bites that lost something: {spoiled}/{bites} ({spoiled / max(1, bites):.1%})")
    tw = sum(r["targeted_waste"] for r in reports)
    iw = sum(r["incidental_waste"] for r in reports)
    tb = sum(r["targeted_bites"] for r in reports)
    ib = sum(r["incidental_bites"] for r in reports)
    print("\n  --- what could actually be reassigned ---")
    print(f"  targeted   : {tb:5d} bites, {tw:6.0f} wasted "
          f"({tw / max(1.0, tw + iw):.1%} of waste)  <- fixable by assignment")
    print(f"  incidental : {ib:5d} bites, {iw:6.0f} wasted "
          f"({iw / max(1.0, tw + iw):.1%} of waste)  <- only fixable by pathing")
    print(f"  median headroom at the moment of biting: "
          f"{statistics.median([r['median_headroom'] for r in reports if r['median_headroom']]):.0f}"
          f"  (a ripe fruit is 60)")
    print(f"\n  a spawn costs 100 and yields a 75-energy child, so the waste above is "
          f"worth ~{wasted / 100.0:.0f} children across {len(reports)} runs")


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _here)
    os.chdir(_here)
    main()
