"""A decision policy expressed as a searchable ordered rule list.

The perception layer is NOT searched. Shared `World`, dead reckoning, fruit and
predator memory, and claims are reused from expert_agent_policy because each has
been measured as load-bearing: removing map visibility from the oracle cost 642
points, and a map-free reactive controller scored 389 against 777.

What IS searched is the decision layer on top: which condition fires, in what
order, and with what threshold. An individual is

    IF  cond_1(theta_1):  action_1
    ELIF cond_2(theta_2): action_2
    ...
    ELSE:                 default

which is a finite, interpretable program space rather than arbitrary behaviour.

Twenty-one hand-designed rules produced one shipped improvement, so the weakest
link in the loop is human guessing about which rule matters. This replaces that
with search.
"""

import math
import random
from typing import List, Tuple

from src.utils.controllers.expert_agent_policy import (
    AT_TREE, BIOME_MOVE, DT, FRUIT_REACH, TREE_REACH, Hive, _wrap,
)

# --- the searchable vocabulary ---------------------------------------------
# Conditions read only what a real agent can know: its own state, the shared
# map, and the population. Each carries one threshold the search also tunes.
CONDITIONS = (
    "always",
    "energy_below",        # energy < theta * max_energy
    "energy_above",        # energy > theta * max_energy
    "runway_below",        # seconds of life left < theta
    "predator_within",     # nearest known predator closer than theta
    "fruit_within",        # a claimable known fruit closer than theta
    "tree_within",         # a known tree closer than theta
    "at_tree",             # standing on a patch
    "age_above",           # age > theta (the aging tax starts at max_age 60-120)
    "pop_below",           # living lineage smaller than theta
    "time_after",          # sim time > theta; food halves every 300 s
    "cannot_sprint",       # energy < max_energy/5, the gate that gets agents eaten
)

ACTIONS = (
    "pursue_fruit",
    "goto_tree",
    "hold",
    "flee",
    "explore",
    "retreat_to_tree",     # walk to a patch rather than chase food
)

# Threshold ranges, chosen so a sampled value is always meaningful for its
# condition rather than silently inert.
RANGES = {
    "energy_below": (0.05, 0.9),
    "energy_above": (0.3, 0.98),
    "runway_below": (2.0, 60.0),
    "predator_within": (60.0, 400.0),
    "fruit_within": (40.0, 600.0),
    "tree_within": (40.0, 700.0),
    "age_above": (30.0, 120.0),
    "pop_below": (1.0, 10.0),
    "time_after": (0.0, 1500.0),
    "always": (0.0, 1.0),
    "at_tree": (0.0, 1.0),
    "cannot_sprint": (0.0, 1.0),
}

Rule = Tuple[str, float, str]          # (condition, threshold, action)


def random_rule(rng: random.Random) -> Rule:
    cond = rng.choice(CONDITIONS)
    lo, hi = RANGES[cond]
    return (cond, rng.uniform(lo, hi), rng.choice(ACTIONS))


def random_program(rng: random.Random, max_rules=5) -> List[Rule]:
    n = rng.randint(2, max_rules)
    return [random_rule(rng) for _ in range(n)]


def mutate(program: List[Rule], rng: random.Random) -> List[Rule]:
    """One structural or numeric edit, so neighbours stay comparable."""
    out = list(program)
    roll = rng.random()
    if roll < 0.35 and out:                       # nudge a threshold
        i = rng.randrange(len(out))
        cond, theta, action = out[i]
        lo, hi = RANGES[cond]
        span = (hi - lo) * 0.25
        out[i] = (cond, min(hi, max(lo, theta + rng.uniform(-span, span))), action)
    elif roll < 0.60 and out:                     # swap an action
        i = rng.randrange(len(out))
        cond, theta, _ = out[i]
        out[i] = (cond, theta, rng.choice(ACTIONS))
    elif roll < 0.75 and len(out) > 2:            # drop a rule
        out.pop(rng.randrange(len(out)))
    elif roll < 0.90 and len(out) < 6:            # insert a rule
        out.insert(rng.randrange(len(out) + 1), random_rule(rng))
    elif len(out) > 1:                            # reorder: priority matters
        i = rng.randrange(len(out))
        j = rng.randrange(len(out))
        out[i], out[j] = out[j], out[i]
    return out


class RuleHive(Hive):
    """Hive perception and bookkeeping; searched rules choose the action."""

    PROGRAM: List[Rule] = []

    def __init__(self, program=None, **overrides):
        super().__init__(**overrides)
        self.program = list(program) if program else list(self.PROGRAM)

    # -- condition evaluation ------------------------------------------
    def _holds(self, cond, theta, mind, st, t, pop, ctx_cache):
        energy, cap = st["energy"], st["max_energy"]
        if cond == "always":
            return True
        if cond == "energy_below":
            return energy < theta * cap
        if cond == "energy_above":
            return energy > theta * cap
        if cond == "cannot_sprint":
            return energy < cap / 5.0
        if cond == "runway_below":
            drain = 1.0 + (0.01 * st["age"] * 10.0 if mind.aging else 0.0)
            return energy / max(drain, 1e-6) < theta
        if cond == "age_above":
            return st["age"] > theta
        if cond == "pop_below":
            return pop < theta
        if cond == "time_after":
            return t > theta
        if cond == "predator_within":
            return ctx_cache["pred_d"] < theta
        if cond == "fruit_within":
            return ctx_cache["fruit_d"] < theta
        if cond == "tree_within":
            return ctx_cache["tree_d"] < theta
        if cond == "at_tree":
            return ctx_cache["tree_d"] < AT_TREE
        return False

    def _cache(self, mind, t):
        world = mind.world
        pred_d = min((math.hypot(p[0] - mind.x, p[1] - mind.y)
                      for p in world.predators), default=1e9)
        fruit_d = 1e9
        for key, rec in world.fruits.items():
            if (world, key) in self._claimed_fruits:
                continue
            fruit_d = min(fruit_d, math.hypot(rec[0] - mind.x, rec[1] - mind.y))
        tree_d = min((math.hypot(r[0] - mind.x, r[1] - mind.y)
                      for r in world.trees.values()), default=1e9)
        return {"pred_d": pred_d, "fruit_d": fruit_d, "tree_d": tree_d}

    # -- decision ------------------------------------------------------
    def _plan(self, mind, st, t):
        # Fall back to the measured-good Hive whenever no rule fires, so the
        # search is scored on what its rules ADD rather than on reimplementing
        # foraging from scratch.
        if not mind.localized or not self.program:
            return super()._plan(mind, st, t)

        cache = self._cache(mind, t)
        pop = max(1, len(self.minds))
        chosen = None
        for cond, theta, action in self.program:
            if self._holds(cond, theta, mind, st, t, pop, cache):
                chosen = action
                break
        if chosen is None or chosen in ("pursue_fruit", "goto_tree"):
            return super()._plan(mind, st, t)      # let the Hive do its job

        energy, max_energy = st["energy"], st["max_energy"]
        speed, sprint = st["speed"], st["sprint_speed"]
        pen = BIOME_MOVE.get(st["biome"], 1.0)
        world = mind.world
        cmd = mdir = turn = 0.0
        heading = None
        spinning = False

        if chosen == "flee":
            if world.predators:
                px, py = min(world.predators,
                             key=lambda p: math.hypot(p[0] - mind.x, p[1] - mind.y))[:2]
                to_pred = math.atan2(py - mind.y, px - mind.x)
                away = self._steer(mind, to_pred + math.pi, speed * pen)
                # Facing the predator downgrades its charge to a slow pivot.
                cmd, mdir, heading = speed, _wrap(away - mind.th), to_pred
                mind.mode = "flee"
            else:
                spinning, mind.mode = True, "hold"
        elif chosen == "explore":
            a = self._steer(mind, self._explore_dir(mind, t), speed * pen)
            cmd, mdir, heading = speed, _wrap(a - mind.th), a
            mind.mode = "explore"
        elif chosen == "retreat_to_tree" and world.trees:
            key = min(world.trees,
                      key=lambda k: math.hypot(world.trees[k][0] - mind.x,
                                               world.trees[k][1] - mind.y))
            rec = world.trees[key]
            mind.target_tree, mind.target_fruit = key, None
            cmd, mdir, heading = self._goto(mind, rec[0], rec[1], speed, pen, TREE_REACH)
            mind.mode = "patrol"
            if cmd <= 0.0:
                spinning, heading, mind.mode = True, None, "hold"
        else:                                       # "hold", or nothing to do
            spinning, mind.mode = True, "hold"

        if spinning:
            turn = self.SCAN_RATE * mind.spin
        elif heading is not None:
            turn = _wrap(heading - mind.th)
            turn = 0.0 if abs(turn) < 0.03 else max(-0.9, min(0.9, turn))

        # Identical cost formula to the environment; the population controller
        # reads `after`, so it must stay exact.
        d_eff = max(0.0, min(cmd, sprint))
        if energy < max_energy / 5.0 and d_eff > speed:
            d_eff = speed
        cost = (d_eff * 0.05 if d_eff <= speed
                else speed * 0.05 + (d_eff - speed) * 0.5)
        turn_cost = min(math.pi, abs(turn)) / (2.0 * math.pi)
        return {"cmd": cmd, "mdir": mdir, "turn": turn, "cost": cost + turn_cost,
                "move_cost": cost, "turn_cost": turn_cost,
                "d_eff": d_eff, "pen": pen, "after": energy - cost - turn_cost}
