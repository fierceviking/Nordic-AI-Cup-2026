"""Upper bound: how long can a policy survive with PERFECT information?

    python oracle_probe.py --seeds 11 12 13
    python oracle_probe.py --ablate tree_age      # hide one privileged fact

The real Hive sees only (type, distance, angle) plus its own status, navigates
by dead reckoning, and cannot observe tree age, fruit ripeness, predator energy
or its own max_age.  This policy reads all of it straight out of `env`.

It still acts through the ordinary ActionRequest interface and pays the exact
same movement, turning and spawning costs, so it is bounded by the simulator's
physics -- only its *knowledge* is free.

Reading:
  oracle ~= 800   the environment itself caps near our current score, and the
                  leaderboard gap must come from somewhere other than policy.
  oracle >> 1500  the environment clearly supports it, and the gap is the
                  information or strategy our agent is missing.  The ablations
                  then say which piece of knowledge is worth the most.
"""

import argparse
import json
import math
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

WALK_COST = 0.05
DT = 0.1
CHARGE_DIST = 90.0          # predator charges regardless of facing inside this
TREE_MATURE = 20.0          # bears fruit from this age
TREE_DOOMED = 56.0          # and dies around 58
FRUIT_ROT = 100.0
AGENTS_PER_TREE = 0.6       # bearing trees yield ~0.1 fruit/s each


class _Req:
    __slots__ = ("agent_id", "move_distance", "move_direction", "turn_angle", "spawn_agent")

    def __init__(self, aid, distance, direction, turn, spawn):
        self.agent_id = aid
        self.move_distance = distance
        self.move_direction = direction
        self.turn_angle = turn
        self.spawn_agent = spawn


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _sensed(env, entity):
    """Is this entity inside some agent's hearing radius or vision cone?

    The Hive can only ever act on things it has personally sensed; the oracle
    otherwise reads the entire map. Wall occlusion is ignored, so this is still
    generous to the oracle.
    """
    for agent in env.agents:
        dx, dy = entity.x - agent.x, entity.y - agent.y
        d = math.hypot(dx, dy)
        if d <= agent.hearing_radius:
            return True
        if d <= agent.vision_radius:
            offset = abs(_wrap(math.atan2(dy, dx) - agent.direction))
            if offset <= agent.cone_angle / 2.0:
                return True
    return False


def _chain(agent, start, fruits, claimed, threats, horizon, ablate, depth=3):
    """Project a chain of fruit visits and return energy at the horizon.

    Deliberately uses ONLY the current state: no future fruit spawns, no RNG.
    Otherwise the planner would be scored on access to future randomness rather
    than on lookahead, which is the single variable under test.
    """
    x, y = agent.x, agent.y
    energy = agent.energy
    speed = max(agent.speed, 1e-6)
    spent = 0.0                       # seconds of the horizon consumed
    visited = set()
    nxt = start
    eaten = 0

    while nxt is not None and len(visited) < depth:
        d = math.hypot(nxt.x - x, nxt.y - y)
        leg = (d / speed) * DT
        if spent + leg > horizon:
            break
        if "fruit_value" in ablate:
            value = 32.0
        else:
            if nxt.age + 2.0 * (spent + leg) > FRUIT_ROT:
                break
            value = min(60.0, nxt.energy + 2.0 * (spent + leg))
        energy -= d * WALK_COST + leg          # travel plus living drain
        if energy <= 0.0:
            return -1e9, 0
        energy = min(agent.max_energy, energy + value)   # the clamp is real
        x, y, spent = nxt.x, nxt.y, spent + leg
        visited.add(id(nxt))
        eaten += 1

        nxt, best = None, -1e18
        for fruit in fruits:
            if id(fruit) in claimed or id(fruit) in visited:
                continue
            if any(math.hypot(p.x - fruit.x, p.y - fruit.y) < 150.0 for p in threats):
                continue
            score = -math.hypot(fruit.x - x, fruit.y - y)
            if score > best:
                nxt, best = fruit, score

    energy -= max(0.0, horizon - spent)        # coast out the rest at 1.0/s
    return energy, eaten


def oracle_actions(env, ablate, per_tree=AGENTS_PER_TREE, horizon=0.0,
                   objective="survival", stats=None):
    """One tick of decisions using whatever privileged state is not ablated."""
    actions = []
    claimed = set()

    # Hunting predators only. A resting one cannot move at all, and the real
    # agent has no way to know the difference.
    threats = []
    for predator in env.predators:
        resting = predator.resting and "predator_state" not in ablate
        if not resting:
            threats.append(predator)

    agents = sorted(env.agents, key=lambda a: a.energy)      # hungriest choose first
    # Fruit income is ~0.1/s per bearing tree, so the number of bearing trees is
    # the real carrying capacity.  Without this the oracle breeds into a famine.
    bearing = sum(1 for t in env.trees if TREE_MATURE <= t.age <= TREE_DOOMED)
    budget = max(2, int(bearing * per_tree))
    room = budget - len(env.agents)

    blind = "visibility" in ablate
    fruits = [f for f in env.fruits if not blind or _sensed(env, f)]
    trees = [t for t in env.trees if not blind or _sensed(env, t)]

    for agent in agents:
        speed, sprint = agent.speed, agent.sprint_speed
        can_sprint = agent.energy >= agent.max_energy / 5.0

        near, near_d = None, 1e18
        for predator in threats:
            d = math.hypot(predator.x - agent.x, predator.y - agent.y)
            if d < near_d:
                near, near_d = predator, d

        # --- run away -------------------------------------------------
        if near is not None and near_d < 210.0:
            to_pred = math.atan2(near.y - agent.y, near.x - agent.x)
            away = to_pred + math.pi
            # nudge along the wall rather than into it
            if agent.x < 60:
                away = _wrap(away) if math.cos(away) > 0 else 0.0
            elif agent.x > env.width - 60:
                away = math.pi
            if agent.y < 60:
                away = math.pi / 2 if math.sin(away) < 0 else away
            elif agent.y > env.height - 60:
                away = -math.pi / 2 if math.sin(away) > 0 else away
            distance = sprint if (near_d < 130.0 and can_sprint) else speed
            # facing the predator downgrades its charge to a slow pivot
            actions.append(_Req(agent.agent_id, distance, _wrap(away - agent.direction),
                                _wrap(to_pred - agent.direction), False))
            continue

        # --- best fruit, priced exactly -------------------------------
        best, best_val = None, 0.0
        projected = None
        if horizon > 0.0:
            # Same candidates, same information; only the valuation horizon
            # differs. A chain can justify a fruit that is unprofitable alone.
            for fruit in fruits:
                if id(fruit) in claimed:
                    continue
                if any(math.hypot(p.x - fruit.x, p.y - fruit.y) < 150.0 for p in threats):
                    continue
                end_energy, eaten = _chain(agent, fruit, fruits, claimed, threats,
                                           horizon, ablate)
                if end_energy <= -1e8:
                    continue
                value = end_energy
                if best is None or value > best_val:
                    best, best_val, projected = fruit, value, end_energy
            if best is not None and stats is not None:
                greedy_gain = min(agent.max_energy - agent.energy,
                                  60.0 if "fruit_value" not in ablate else 32.0)
                d0 = math.hypot(best.x - agent.x, best.y - agent.y)
                if greedy_gain - (d0 * WALK_COST + d0 / max(agent.speed, 1e-6) * DT) <= 0.0:
                    stats["unprofitable_now"] += 1
        else:
            for fruit in fruits:
                if id(fruit) in claimed:
                    continue
                d = math.hypot(fruit.x - agent.x, fruit.y - agent.y)
                ticks = d / max(speed, 1e-6)
                if "fruit_value" in ablate:
                    value = 32.0
                else:
                    # it keeps ripening while we walk, and rots at age 100
                    value = min(60.0, fruit.energy + 2.0 * ticks * DT)
                    if fruit.age + 2.0 * ticks * DT > FRUIT_ROT:
                        continue
                headroom = agent.max_energy - agent.energy
                gain = min(value, headroom)                  # the clamp is real
                cost = d * WALK_COST + ticks * DT
                if any(math.hypot(p.x - fruit.x, p.y - fruit.y) < 150.0 for p in threats):
                    continue
                if gain - cost > best_val:
                    best, best_val = fruit, gain - cost
        if best is not None:
            claimed.add(id(best))
            ang = math.atan2(best.y - agent.y, best.x - agent.x)
            d = math.hypot(best.x - agent.x, best.y - agent.y)
            spawn = _should_spawn(agent, threats, room)
            room -= int(spawn)
            actions.append(_Req(agent.agent_id, min(speed, d), _wrap(ang - agent.direction),
                                _wrap(ang - agent.direction), spawn))
            continue

        # --- otherwise sit on a tree that is actually fruiting ---------
        best_tree, best_score = None, -1e18
        for tree in trees:
            if "tree_age" not in ablate:
                if tree.age < TREE_MATURE or tree.age > TREE_DOOMED:
                    continue
            d = math.hypot(tree.x - agent.x, tree.y - agent.y)
            if any(math.hypot(p.x - tree.x, p.y - tree.y) < 200.0 for p in threats):
                continue
            crowd = sum(1 for other in env.agents
                        if other is not agent
                        and math.hypot(other.x - tree.x, other.y - tree.y) < 70.0)
            score = -d * WALK_COST - crowd * 40.0
            if score > best_score:
                best_tree, best_score = tree, score
        if best_tree is not None:
            d = math.hypot(best_tree.x - agent.x, best_tree.y - agent.y)
            ang = math.atan2(best_tree.y - agent.y, best_tree.x - agent.x)
            spawn = _should_spawn(agent, threats, room)
            room -= int(spawn)
            if d > 12.0:
                actions.append(_Req(agent.agent_id, min(speed, d - 10.0),
                                    _wrap(ang - agent.direction),
                                    _wrap(ang - agent.direction), spawn))
            else:
                actions.append(_Req(agent.agent_id, 0.0, 0.0, 0.35, spawn))
            continue

        spawn = _should_spawn(agent, threats, room)
        room -= int(spawn)
        actions.append(_Req(agent.agent_id, speed, 0.0, 0.25, spawn))
    return actions


def _should_spawn(agent, threats, room):
    """Spawn when the surplus would otherwise be destroyed, or death is near."""
    if agent.energy <= 160.0:
        return False
    if any(math.hypot(p.x - agent.x, p.y - agent.y) < 260.0 for p in threats):
        return False
    # knows its own hidden max_age: liquidate before the aging tax burns it
    if agent.age > agent.max_age - 5.0:
        return True
    if room <= 0:
        return False
    return agent.max_energy - agent.energy < 60.0


def _apply_mechanic(env, mechanic):
    """Relax one simulator constraint, leaving the controller untouched.

    This measures d(score)/d(mechanic): where the ceiling actually comes from,
    without inventing a policy to chase it.
    """
    if mechanic == "none":
        return

    original_move = env.update_entity_position

    if mechanic == "free_movement":
        def patched(entity, distance, direction=None, local_obstacles=None):
            before = entity.energy
            original_move(entity, distance, direction, local_obstacles)
            entity.energy = before          # refund travel only; turning still costs
        env.update_entity_position = patched

    elif mechanic == "no_sprint_gate":
        def patched(entity, distance, direction=None, local_obstacles=None):
            # The gate reads entity.max_energy/5 and nothing else in this call does.
            cap = entity.max_energy
            entity.max_energy = 0.0
            try:
                original_move(entity, distance, direction, local_obstacles)
            finally:
                entity.max_energy = cap
        env.update_entity_position = patched

    elif mechanic == "no_clamp":
        # Infinite storage would also disable sprinting, because the gate is
        # max_energy/5. Bypass the gate too, or this measures "nobody sprints".
        def patched(entity, distance, direction=None, local_obstacles=None):
            cap = entity.max_energy
            entity.max_energy = 0.0
            try:
                original_move(entity, distance, direction, local_obstacles)
            finally:
                entity.max_energy = cap
        env.update_entity_position = patched

    elif mechanic == "no_predators":
        env.predators = []
        env.spawn_predator = lambda *a, **k: None


def _enforce_mechanic(env, mechanic):
    """Per-tick upkeep for ablations that must be re-applied to new agents."""
    if mechanic == "no_clamp":
        for agent in env.agents:
            agent.max_energy = 1e9
    elif mechanic == "no_aging":
        for agent in env.agents:
            agent.max_age = 1e9
    elif mechanic == "no_predators" and env.predators:
        env.predators = []


def _run_one(job):
    seed, horizon, ablate, per_tree, lookahead, objective, mechanic = job
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    from src.core import SimulationCore

    sim = SimulationCore(seed=seed)
    deaths = {"eaten": 0, "starved": 0}
    original_kill = sim.env.kill_agent

    def patched_kill(agent, _env=sim.env, _orig=original_kill):
        if agent in _env.agents:
            hit = any((p.x - agent.x) ** 2 + (p.y - agent.y) ** 2
                      < (p.size + agent.size + 2) ** 2 for p in _env.predators)
            deaths["eaten" if hit else "starved"] += 1
        _orig(agent)

    sim.env.kill_agent = patched_kill
    _apply_mechanic(sim.env, mechanic)

    actions = []
    peak = 0
    stats = {"unprofitable_now": 0}
    while True:
        _enforce_mechanic(sim.env, mechanic)
        state = sim.step(actions)
        peak = max(peak, state["num_agents"])
        requests = oracle_actions(sim.env, ablate, per_tree, lookahead, objective, stats)
        actions = [(r.agent_id, r) for r in requests]
        if state["num_agents"] == 0 or sim.env.time > horizon:
            break
    return {"seed": seed, "score": round(state["score"], 1),
            "sim_time": round(sim.env.time, 1), "alive": state["num_agents"],
            "peak_agents": peak, "survived": sim.env.time > horizon - 1,
            "eaten": deaths["eaten"], "starved": deaths["starved"],
            "unprofitable_now": stats["unprofitable_now"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    parser.add_argument("--horizon", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--ablate", nargs="*", default=[],
                        choices=["tree_age", "fruit_value", "predator_state", "visibility"],
                        help="hide a privileged fact to price it")
    parser.add_argument("--per-tree", type=float, default=AGENTS_PER_TREE,
                        help="agents supported per bearing tree")
    # The single variable under test: 0 reproduces the greedy oracle exactly.
    parser.add_argument("--lookahead", type=float, default=0.0,
                        help="seconds of forward planning (0 = greedy)")
    parser.add_argument("--objective", choices=("survival", "population"),
                        default="survival")
    parser.add_argument("--mechanic", default="none",
                        choices=("none", "no_clamp", "no_aging", "free_movement",
                                 "no_predators", "no_sprint_gate"),
                        help="relax one simulator constraint to locate the ceiling")
    args = parser.parse_args()

    ablate = set(args.ablate)
    print(f"oracle with perfect information{' minus ' + ', '.join(sorted(ablate)) if ablate else ''}")
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run_one, (s, args.horizon, ablate, args.per_tree,
                                          args.lookahead, args.objective,
                                          args.mechanic))
                   for s in args.seeds]
        for future in as_completed(futures):
            report = future.result()
            reports.append(report)
            print(json.dumps(report), flush=True)

    scores = [r["score"] for r in reports]
    sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
    eaten = sum(r["eaten"] for r in reports)
    starved = sum(r["starved"] for r in reports)
    print(f"\nmean {statistics.mean(scores):.1f} | median {statistics.median(scores):.1f} "
          f"| sd {sd:.1f} | se {sd / max(1.0, math.sqrt(len(scores))):.1f} "
          f"| min {min(scores):.1f} | max {max(scores):.1f} "
          f"| survived {sum(r['survived'] for r in reports)}/{len(reports)}")
    print(f"deaths: eaten {eaten} | starved {starved} "
          f"({eaten / max(1, eaten + starved):.0%} predation)")
    print(f"Hive baseline on these seeds is ~772.")


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _here)
    os.chdir(_here)
    main()
