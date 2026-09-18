"""Burst controller: navigates on raw relative observations, no world model.

    MEASURED RESULT: REFUTED.  Do not ship this.  Six seeds, horizon 3000,
    against expert_agent_policy.Hive (mean 777.0): BURST_STEPS=1 scored 388.6,
    BURST_STEPS=3 scored 594.6, and POP_CAP=20 scored 349.7.  It looks strong
    over a short horizon -- at horizon 400 it held 15 agents where the Hive
    averages a die-off -- because early trees are dense enough that reacting to
    whatever is currently visible is sufficient.  Once tree count decays
    (N ~ 107 * 0.5**(t/600)) finding the next patch needs remembered positions,
    and a controller with no map cannot relocate.  Kept only as the executable
    record of the two findings below.

Finding 1: ``step_environment`` applies every ``(agent_id, action)`` pair::

    for agent_id, action in actions:
        if agent_id not in env.agents_dict:
            continue
        env.agent_step(agent_id, ...)

No de-duplication, so repeating an ``agent_id`` runs ``agent_step`` again
*inside the same tick*: no sim time passes and predators do not move.  Measured
directly: one action of ``move_distance=10`` displaces 10 units, twenty
displace 200, and four ``spawn_agent`` actions produce four children in one
tick.

    RULES WARNING: README.md specifies "a list of ActionRequests, one for each
    agent", so BURST_STEPS > 1 is outside the documented contract.

Finding 2, and the reason the exploit is worthless anyway: bursting buys no
energy discount.  ``update_entity_position`` charges 0.05 per unit whether the
distance arrives in one action or twenty (measured 0.0505/unit either way), so
a 14-step burst costs ~70 energy/second and starves the hive faster than the
travel time it saves.  It moves position in zero time; it does not move it
cheaply.

Third mechanic, which does hold: ``agent_step`` applies move, then turn, then
spawn.  Emitting the turn and spawn only on the final action of a burst leaves
every earlier move sharing one heading, so a relative bearing taken from an
observation stays valid for the whole burst.
"""

import math
from typing import Dict, List

from src.utils.DTOs import ActionRequest

TWO_PI = 2.0 * math.pi
DT = 0.1
WALK_COST = 0.05
SPAWN_COST = 100.0


def _wrap(angle: float) -> float:
    return (angle + math.pi) % TWO_PI - math.pi


class BurstHive:
    # Set to 1 to stay within the documented one-action-per-agent contract.
    BURST_STEPS = 14

    FLEE_RANGE = 230.0      # predator distance that cancels foraging
    FLEE_MARGIN = 120.0     # extra ground to put down when running
    FRUIT_STOP = 3.0        # stop short so the eat check still triggers
    TREE_HOLD = 26.0        # orbit radius while waiting on a patch to fruit
    SCAN_TURN = 0.55        # radians swept when not committed to a bearing
    ENERGY_FLOOR = 18.0     # never burst into starvation
    SPAWN_AT = 330.0        # surplus above this is banked as a child
    SPAWN_COOLDOWN = 3.0
    POP_CAP = 14

    def __init__(self, **overrides) -> None:
        for key, value in overrides.items():
            if not hasattr(type(self), key):
                raise AttributeError(f"unknown tunable {key!r}")
            setattr(self, key, value)
        self.reset()

    def reset(self) -> None:
        self.last_time = -1.0
        self.last_spawn: Dict[int, float] = {}
        self.heading_bias: Dict[int, float] = {}
        self.debug: Dict[str, object] = {}
        self.mode_ticks: Dict[str, int] = {}
        # Shape-compatible with Hive so local_playground's diagnostics run.
        self.minds: Dict[int, object] = {}
        self.deaths: List[tuple] = []
        self.target_pop = 0.0
        self.income_ema = 0.0
        self.last_state: Dict[int, tuple] = {}
        self.last_mode: Dict[int, str] = {}
        self.budget = {"income": 0.0, "move": 0.0, "turn": 0.0, "live": 0.0,
                       "spawn": 0.0, "age": 0.0, "ticks": 0.0, "agent_ticks": 0.0}

    def _record(self, agent_states: List[dict], t: float) -> None:
        alive = set()
        for state in agent_states:
            aid = int(state["agent_id"])
            alive.add(aid)
            self.last_state[aid] = (t, float(state["age"]), float(state["energy"]))
        for aid in [a for a in self.last_state if a not in alive]:
            when, age, energy = self.last_state.pop(aid)
            self.deaths.append((when, age, energy, self.last_mode.pop(aid, "?"), False))
            self.last_spawn.pop(aid, None)
            self.heading_bias.pop(aid, None)

    @staticmethod
    def _split(observations):
        predators, fruits, trees = [], [], []
        for obs in observations:
            kind = obs.get("type", "")
            kind = kind if kind in ("Fruit", "Predator", "Tree") else kind.capitalize()
            if kind == "Predator":
                predators.append((obs["distance"], obs["angle"]))
            elif kind == "Fruit":
                fruits.append((obs["distance"], obs["angle"]))
            elif kind == "Tree":
                trees.append((obs["distance"], obs["angle"]))
        return predators, fruits, trees

    def _emit(self, aid: int, distance: float, bearing: float, speed: float,
              turn: float, spawn: bool) -> List[dict]:
        """Chop a travel distance into walking-rate steps within one tick."""
        actions = []
        step_cap = max(speed, 1e-6)
        remaining = max(0.0, distance)
        budget = self.BURST_STEPS
        while remaining > 0.05 and budget > 0:
            step = min(step_cap, remaining)
            actions.append({"agent_id": aid, "move_distance": float(step),
                            "move_direction": float(bearing),
                            "turn_angle": 0.0, "spawn_agent": False})
            remaining -= step
            budget -= 1
        if actions and turn == 0.0 and not spawn:
            return actions
        actions.append({"agent_id": aid, "move_distance": 0.0,
                        "move_direction": 0.0, "turn_angle": float(turn),
                        "spawn_agent": bool(spawn)})
        return actions

    def act(self, agent_states: List[dict], sim_time: float) -> List[ActionRequest]:
        return [ActionRequest(**a) for a in self.act_dicts(agent_states, sim_time)]

    def act_dicts(self, agent_states: List[dict], sim_time: float) -> List[dict]:
        t = float(sim_time)
        if t + 1e-9 < self.last_time:
            self.reset()
        self.last_time = t
        population = len(agent_states)
        self._record(agent_states, t)
        self.budget["ticks"] += 1.0
        self.budget["agent_ticks"] += population
        self.budget["live"] += DT * population
        self.target_pop = float(self.POP_CAP)

        out: List[dict] = []
        modes: Dict[str, int] = {}
        for state in agent_states:
            aid = int(state["agent_id"])
            energy = float(state["energy"])
            speed = float(state["speed"])
            predators, fruits, trees = self._split(state["observations"])

            # Distance affordable this tick, leaving a floor to survive on.
            reach = max(0.0, (energy - self.ENERGY_FLOOR) / WALK_COST)
            reach = min(reach, self.BURST_STEPS * speed)

            spawn = (energy > self.SPAWN_AT
                     and population < self.POP_CAP
                     and t - self.last_spawn.get(aid, -1e9) >= self.SPAWN_COOLDOWN)

            turn = 0.0
            if predators:
                distance, bearing = min(predators)
                if distance < self.FLEE_RANGE:
                    # Time-free retreat: run directly away, far enough that the
                    # predator's 15/tick sprint cannot close before next tick.
                    travel = min(reach, distance + self.FLEE_MARGIN)
                    out.extend(self._emit(aid, travel, _wrap(bearing + math.pi),
                                          speed, 0.0, False))
                    modes["flee"] = modes.get("flee", 0) + 1
                    self.mode_ticks["flee"] = self.mode_ticks.get("flee", 0) + 1
                    self.last_mode[aid] = "flee"
                    continue

            if fruits:
                distance, bearing = min(fruits)
                travel = min(reach, max(0.0, distance - self.FRUIT_STOP))
                out.extend(self._emit(aid, travel, bearing, speed, 0.0, spawn))
                if spawn:
                    self.last_spawn[aid] = t
                    self.budget["spawn"] += SPAWN_COST
                modes["fruit"] = modes.get("fruit", 0) + 1
                self.mode_ticks["fruit"] = self.mode_ticks.get("fruit", 0) + 1
                self.last_mode[aid] = "fruit"
                continue

            if trees:
                distance, bearing = min(trees)
                if distance > self.TREE_HOLD:
                    travel = min(reach, distance - self.TREE_HOLD)
                    out.extend(self._emit(aid, travel, bearing, speed, 0.0, spawn))
                else:
                    # Sit on the patch and sweep the cone for the next fruit.
                    out.extend(self._emit(aid, 0.0, 0.0, speed, self.SCAN_TURN, spawn))
                if spawn:
                    self.last_spawn[aid] = t
                    self.budget["spawn"] += SPAWN_COST
                modes["patrol"] = modes.get("patrol", 0) + 1
                self.mode_ticks["patrol"] = self.mode_ticks.get("patrol", 0) + 1
                self.last_mode[aid] = "patrol"
                continue

            bias = self.heading_bias.get(aid, 0.0)
            if int(t * 10) % 90 == 0:
                bias = _wrap(bias + 1.1)
                self.heading_bias[aid] = bias
                turn = self.SCAN_TURN
            # Searching blind is charged per unit like any other travel, so it
            # stays at one walking step however large the burst budget is.
            out.extend(self._emit(aid, min(reach, speed), 0.0, speed, turn, spawn))
            if spawn:
                self.last_spawn[aid] = t
                self.budget["spawn"] += SPAWN_COST
            modes["explore"] = modes.get("explore", 0) + 1
            self.mode_ticks["explore"] = self.mode_ticks.get("explore", 0) + 1
            self.last_mode[aid] = "explore"

        self.debug = {"pop": population, "actions": len(out),
                      "burst": self.BURST_STEPS, "modes": modes}
        return out


_HIVE = BurstHive()


def action_decision_all(agent_states: List[dict], sim_time: float) -> List[ActionRequest]:
    return _HIVE.act(agent_states, sim_time)
