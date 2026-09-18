"""Deterministic hand-engineered controller for the survival simulator.

No learning involved: the policy is derived analytically from the simulator's
energy/aging/predator equations (see ``Hive`` docstring for the reasoning).
"""

import math
import random
from typing import Dict, List, Optional, Tuple

from src.utils.DTOs import ActionRequest

TWO_PI = 2.0 * math.pi
DT = 0.1

# Biome constants mirrored from src/elements/biome.py
BIOME_MOVE = {"forest": 1.0, "grassland": 1.0, "swamp": 0.5, "desert": 0.8, "river": 0.3}

WALK_COST = 0.05
SPRINT_COST = 0.5
SPAWN_COST = 100.0

# Predator model (src/elements/predator.py)
PRED_HEAR = 60.0
CHARGE_DIST = PRED_HEAR * 1.5          # below this it charges regardless of our facing
CHASE_DIST = CHARGE_DIST + 25.0        # run at matched speed from here in
BACKOFF_DIST = 260.0                   # walk away while facing an approaching one
WATCH_DIST = 320.0                     # merely bias foraging away
CLOSING_RATE = 2.0                     # units/tick that count as "hunting us"

AGENT_SIZE = 5.0
FRUIT_REACH = 4.0                      # get this close to the centre and we always eat
TREE_REACH = 7.0
AT_TREE = 46.0                         # close enough to be harvesting a patch

PRED_MEMORY = 4.0
FRUIT_MEMORY = 30.0
TREE_MEMORY = 70.0
EDGE_MEMORY = 10.0

# Displacement per tick above which a predator is taken to be sprinting rather
# than pivot-approaching. Biome move_penalty scales displacement, so this sits
# well under the nominal sprint of 15.
CHASE_SPEED = 8.0

# A mature tree drops ~0.1 fruit/s worth ~30 energy.
CELL = 120.0

# Fruit spawns at 20-60 units from its tree, so this annulus attributes a
# remembered fruit to the patch that produced it.
FRUIT_NEAR_TREE = 62.0

_OBS_TYPES = frozenset(("Fruit", "Agent", "Predator", "Tree", "Edge"))


def _wrap(a: float) -> float:
    return (a + math.pi) % TWO_PI - math.pi


def _point_seg_dist(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    dx, dy = x2 - x1, y2 - y1
    l2 = dx * dx + dy * dy
    if l2 <= 1e-9:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / l2
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


class World:
    """Shared map for one lineage (agents that share a dead-reckoning frame)."""

    def __init__(self) -> None:
        self.trees: Dict[int, List[float]] = {}   # key -> [x, y, last_seen]
        self.fruits: Dict[int, List[float]] = {}  # key -> [x, y, last_seen]
        self.visits: Dict[int, float] = {}        # tree key -> last time harvested
        # Fruit spawns from the biome cell the TREE stands in, so a river tree
        # never produces.  We cannot observe a tree's biome directly, but an
        # agent standing on the patch reports its own - close enough at this range.
        self.biomes: Dict[int, str] = {}          # tree key -> biome tagged on site
        self.dry: Dict[int, float] = {}           # tree key -> fruitless seconds spent on it
        self.predators: List[List[float]] = []    # [x, y, t, vx, vy, n, chase_since]
        self.visited: Dict[Tuple[int, int], float] = {}
        self._key = 0
        self._near_t = -1.0
        self._near: Dict[int, int] = {}

    def new_key(self) -> int:
        self._key += 1
        return self._key

    def fruit_near_trees(self, t: float, radius: float) -> Dict[int, int]:
        """Remembered fruit count per tree, computed once per tick for the hive."""
        if self._near_t == t:
            return self._near
        near: Dict[int, int] = {}
        r2 = radius * radius
        for frec in self.fruits.values():
            fx, fy = frec[0], frec[1]
            for key, rec in self.trees.items():
                dx, dy = rec[0] - fx, rec[1] - fy
                if dx * dx + dy * dy < r2:
                    near[key] = near.get(key, 0) + 1
        self._near_t, self._near = t, near
        return near

    def forget_tree(self, key: int) -> None:
        self.trees.pop(key, None)
        self.visits.pop(key, None)
        self.biomes.pop(key, None)
        self.dry.pop(key, None)

    def add_point(self, table: Dict[int, List[float]], x: float, y: float, t: float,
                  tol: float) -> int:
        best_key, best_d = -1, tol
        for key, rec in table.items():
            d = math.hypot(rec[0] - x, rec[1] - y)
            if d < best_d:
                best_key, best_d = key, d
        if best_key >= 0:
            rec = table[best_key]
            rec[0] += 0.4 * (x - rec[0])
            rec[1] += 0.4 * (y - rec[1])
            rec[2] = t
            return best_key
        key = self.new_key()
        # rec[3] is first-seen: trees cannot be aged from observations, so how
        # long we have known one is the only available proxy for its remaining life.
        table[key] = [x, y, t, t]
        return key

    def add_predator(self, x: float, y: float, t: float) -> None:
        for p in self.predators:
            if math.hypot(p[0] - x, p[1] - y) < 45.0:
                if t <= p[2]:
                    return
                dt_ticks = max(1.0, (t - p[2]) / DT)
                if dt_ticks <= 12.0:
                    p[3] = (x - p[0]) / dt_ticks
                    p[4] = (y - p[1]) / dt_ticks
                    p[5] += 1.0
                else:
                    p[5] = 1.0
                p[0], p[1], p[2] = x, y, t
                # A predator burns 2.55/tick sprinting from a wake threshold of
                # 100, so a chase cannot outlast ~3.9 s. Timing the sprint is
                # the only way to know how much of that budget is already gone.
                if math.hypot(p[3], p[4]) > CHASE_SPEED:
                    if p[6] < 0.0:
                        p[6] = t
                else:
                    p[6] = -1.0
                return
        self.predators.append([x, y, t, 0.0, 0.0, 1.0, -1.0])

    def expire(self, t: float) -> None:
        for key in [k for k, r in self.trees.items() if t - r[2] > TREE_MEMORY]:
            self.forget_tree(key)
        for key in [k for k, r in self.fruits.items() if t - r[2] > FRUIT_MEMORY]:
            del self.fruits[key]
        self.predators = [p for p in self.predators if t - p[2] <= PRED_MEMORY]


class Mind:
    """Per-agent state: dead-reckoned pose, energy bookkeeping and current job."""

    def __init__(self, aid: int, world: World, x: float, y: float, th: float,
                 localized: bool = True, born: float = 0.0) -> None:
        self.aid = aid
        self.world = world
        self.x, self.y, self.th = x, y, th
        # A spawned agent gets a *random* heading from the simulator, so its
        # frame is unknown until it sights an already localized hive member.
        self.localized = localized
        self.born = born
        self.sightings: List[Tuple[int, float, float, float]] = []
        self.prev_energy: Optional[float] = None
        self.last_spawn = -1e9
        self.prev_age = 0.0
        self.last_cost = 0.0
        self.aging = False
        self.age_hits = 0
        self.income = 0.0                 # leaky accumulator of harvested energy
        self.spin = 1.0
        self.explore_dir = 0.0
        self.explore_until = -1.0
        self.scout_until = -1.0
        self.target_tree: Optional[int] = None
        self.target_fruit: Optional[int] = None
        self.home: Optional[int] = None
        self.deflected = False
        self.stall = 0
        self.last_meal = born
        self.avoid: Dict[int, float] = {}   # target key -> time it becomes usable again
        self.mode = "init"
        self.last_state: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # t, age, energy
        self.edges: Dict[Tuple[int, int, int, int], float] = {}

    # -- perception -------------------------------------------------------
    def to_world(self, dist: float, ang: float) -> Tuple[float, float]:
        a = self.th + ang
        return self.x + dist * math.cos(a), self.y + dist * math.sin(a)

    def blocked(self, px: float, py: float, clearance: float = AGENT_SIZE + 2.0) -> bool:
        for (x1, y1, x2, y2) in self.edges:
            if _point_seg_dist(px, py, x1, y1, x2, y2) < clearance:
                return True
        return False


class Hive:
    """Hive-mind controller.

    Key facts the policy is built on (all read off the simulator source):

    * ``move_direction`` is relative to the agent heading, so an agent can run
      away from a predator while still *facing* it.  A predator only charges
      when ``|rel_dir| > pi/2`` (agent facing away) or when it is closer than
      ``1.5 * hearing_radius = 90``.  Facing it downgrades the chase to a slow
      pivot that the predator cannot afford for long (it burns 2.55 energy per
      tick and only carries ~100).
    * Aging (``energy -= 0.01 * age`` per tick past a hidden 60-120s max age)
      caps any individual near age ~100-160, so ~30 generations are needed to
      reach t=3000.  Spawning turns 100 energy into a fresh 75-energy agent,
      which is the only way to reset the age clock - dying agents therefore
      always dump their energy into a child.
    * Tree spawn rate decays as ``0.5**(t/300)``; the map holds ~107 trees at
      the start but only ~3 at the end, so map-wide food drops to ~8 energy/s.
      Trees are therefore treated as renewable patches and the hive runs a
      shared greedy orienteering loop over them (go to the patch with the best
      accumulated-yield minus travel-cost).  The shared visit table makes the
      agents spread over the patches without any explicit negotiation.
    * Travelling costs ``0.05/unit`` plus ``0.01/unit`` of elapsed living cost,
      so a ~30 energy fruit is worth a detour of up to ~450 units.  Sprinting
      costs ten times as much and is only used to break a chase.
    * Below ``max_energy/5`` an agent can no longer sprint, which is how most
      agents end up eaten, so reproduction must never push an agent under a
      solid energy reserve.
    """

    # --- tunables --------------------------------------------------------
    SPAWN_MIN = 300.0          # energy required for a routine spawn
    POP_MIN = 3
    POP_MAX = 8
    POP_HARD = 22              # ceiling for converting an aging agent's energy
    # `score += dt` fires every tick regardless of how many agents are alive, and
    # the run ends only at zero, so population carries NO score value -- it is
    # purely instrumental.  These two used to be hard-coded, which silently
    # floored the population at 3 and made a POP=1 policy impossible to express.
    RESCUE_POP = 2             # at or below this, spawn regardless of target
    POP_SLACK = 2              # how far above target replacement births may go
    # Parent eligibility. Requiring mode == "hold" removes 89% of the remaining
    # candidates (measured funnel: 6.03 alive -> 4.09 off-cooldown -> 2.54 rich
    # -> 0.27 holding), so 96.3% of births have at most ONE candidate and the
    # fitness weights are decorative -- there is nothing to choose between.
    # But "holding" is only a PROXY for "standing beside food", which is what a
    # 75-energy newborn actually needs. Testing the site directly decouples
    # eligibility from mode without moving the child: the environment always
    # spawns it 10-30 units from the parent.
    BREED_HOLD_ONLY = True     # False tests the birth site instead of the mode
    BREED_SITE_RADIUS = 60.0   # a known tree this close counts as a fed site
    INCOME_HIGH = 3.2
    INCOME_LOW = 1.8
    NET_GROW = 0.40            # net energy/agent/s that justifies another mouth
    NET_SHRINK = 0.00          # below this the hive is eating its own reserves
    SPAWN_COOLDOWN = 4.0       # seconds; without it one parent spawns every tick
    # Past max_age an agent burns 0.01*age per tick (~9-12/s) while earning ~8/s,
    # so it is strictly dying and that energy is only recoverable as children.
    # Measured age tax: 2.04 energy/agent/s, ~143 per lifetime = 1.4 lost children.
    AGING_COOLDOWN = 4.0       # own cooldown for aging agents; lower = faster dump
    AGING_DUMP = False         # let a dying agent liquidate up to POP_HARD
    TREE_YIELD = 2.2           # energy/s credited to an unharvested patch
    YIELD_CAP = 25.0           # seconds of accumulation worth believing
    PATROL_RANGE = 280.0       # normal working radius around an agent
    PATROL_FAR = 700.0         # used only when nothing is known nearby
    MOVE_ON = 35.0             # net energy that justifies leaving a patch
    FRUIT_RANGE = 220.0
    # Scoring weights for the fruit/tree value functions.  Hand-set originally;
    # exposed so train_es.py can search them.
    FRUIT_VALUE = 32.0         # credited energy for a fruit believed to be there
    FRUIT_MIN_VALUE = 6.0      # value floor below which a fruit is not worth it
    FRUIT_HYSTERESIS = 5.0     # bonus for the fruit already being chased
    FRUIT_DANGER = 60.0        # penalty for a fruit near a predator
    FRUIT_DANGER_RANGE = 200.0
    TREE_HYSTERESIS = 6.0      # bonus for the patch already being worked
    TREE_CLAIMED = 45.0        # penalty for a patch another agent claimed
    TREE_DANGER = 80.0
    TREE_DANGER_RANGE = 220.0
    RISK_AVERSION = 0.5        # extra weight on travel cost when energy is low
    RISK_ENERGY = 270.0        # energy above which travel is priced normally
    SCAN_RATE = 0.30           # radians/tick swept while holding a patch
    # A child is always born with 75 energy while the environment clamps anyone
    # below max_energy/5 to walking speed.  At the founder max_energy of 500
    # that threshold is 100, and traits barely mutate within a run (measured:
    # 1 of 45 newborns could sprint), so nearly every newborn spends its first
    # fruit unable to outrun a predator that closes at 15/tick against its 10.
    # Such an agent has to start retreating from much further out.
    NOSPRINT_MARGIN = 1.0      # 1.0 reproduces the previous behaviour
    # Predators are absent before t~50 and number only 2-4 by t~275, while tree
    # count peaks at t=0 and halves every 600 s.  The opening is therefore the
    # cheapest population a run will ever get.
    EARLY_BOOST = False
    EARLY_UNTIL = 150.0
    EARLY_POP = 14
    EARLY_COOLDOWN = 1.5
    # _steer() only rotates away from a heading when the step is blocked, so a
    # run of deflected ticks is the policy sliding along geometry it cannot get
    # past.  Measured: 9.1% of targeted ticks close <25% of the distance owed.
    UNSTICK = False
    STALL_TICKS = 8
    AVOID_TIME = 12.0
    # A Tree observation carries no age, and only trees aged 20-58 bear fruit,
    # so camping an immature trunk looks identical to camping a productive one
    # and costs ~1.5 energy/s indefinitely.  Going hungry this long while sat on
    # a patch is the only available evidence that it is barren.
    BARREN_PATIENCE = 0.0      # 0 disables; otherwise seconds without eating
    # Eating is involuntary -- `energy = min(max_energy, energy + fruit.energy)`
    # fires on contact -- so a nearly-full agent destroys the surplus.  Measured
    # over 3 runs: 39,334 of 173,027 harvested energy (22.7%) annihilated this
    # way, on 30.2% of all bites.  A spawn costs 100 and frees 100 of headroom,
    # so trading that surplus for a child is close to free.  Unlike the earlier
    # population pushes, this fires on forfeit energy, not on a population gap.
    OVERFLOW_SPAWN = False
    OVERFLOW_HEADROOM = 60.0   # a fully ripe fruit is 60
    OVERFLOW_COOLDOWN = 2.0
    # A fruit is only worth what the agent can still absorb: the environment
    # clamps with min(max_energy, energy + fruit.energy) and discards the rest.
    # The oracle prices every fruit this way; measured clamp loss is 22.7% of
    # everything harvested.
    HEADROOM_PRICING = False
    # Measured: sensors cover 7.5% of the map per tick, the hive knows 27.8% of
    # trees and 7.9% of fruit, and agents sometimes sit 3 units apart -- total
    # sensor overlap. TREE_CLAIMED only penalises the exact tree another agent
    # picked, not the region around it. This is a soft repulsion that decays
    # with distance, so agents still converge when there is a reason to.
    SPREAD = 0.0               # 0 disables; penalty at zero separation
    SPREAD_RANGE = 260.0       # separation beyond which hive-mates stop repelling
    # Expected-fruit routing.  The hive knows ~27% of trees but only ~8% of
    # fruit, and a tree observation predicts fruit it has never seen: the
    # environment rolls `dt * biome.fruit_spawn_rate` per tree per second using
    # the biome under the TREE, so a known patch has an estimable standing crop.
    # This replaces the flat TREE_YIELD accumulation term with that estimate.
    FRUIT_PRIOR = 0.0          # 0 disables; weight on expected unseen fruit
    PRIOR_LIFE = 50.0          # fruit rots 50 s after spawning, so older ones are gone
    PRIOR_ENERGY = 44.0        # mean energy of a fruit of uniform age (20->60 at 2/s)
    PRIOR_CAP = 3.0            # never believe in more than this many unseen fruit
    PRIOR_DEFAULT = 0.08       # assumed rate for a tree we have never stood on
    PRIOR_EVIDENCE = 0.0       # 0 disables; fruitless seconds that write a patch off
    # fruit_spawn_rate by biome, straight from src/elements/biome.py.  River is
    # 0.0: those trees are decoration and the old model valued them like forest.
    BIOME_YIELD = {"forest": 0.1, "grassland": 0.1, "swamp": 0.08,
                   "desert": 0.05, "river": 0.0}
    # Dedicated scouts.  `explore` mode already exists and already steers by
    # cell staleness, but it is only reachable when NO tree is known within
    # PATROL_FAR (700), which essentially never happens -- measured: 1 of 952
    # meals came from an exploring agent.  So the hive never deliberately grows
    # its map; the 27% of trees it knows are all incidental sightings picked up
    # while commuting between patches.  A scout is an agent taken off the
    # economy and spent on map growth instead.
    SCOUT = 0                  # scouts to maintain; 0 disables
    SCOUT_MIN_POP = 5          # never divert anyone below this population
    SCOUT_COMMIT = 25.0        # seconds a scout keeps the job before reassignment
    SCOUT_FEED = 140.0         # energy below which a scout goes back on the economy
    SCOUT_GRAB = 90.0          # a scout still takes fruit this close to its path
    # Predator chase budget. A predator sprints at 2.55/tick from a wake
    # threshold of max_energy/2 = 100, so it can chase for ~3.9 s and must then
    # rest ~3.3 s. The hive already notices a predator that HAS stopped
    # (p[5] >= 4 and vmag < 1.5) but never tracks how long one has been running,
    # so it keeps paying sprint costs against a chaser that is about to quit.
    # Fleeing is ~6% of agent-ticks at ~3.9 energy/tick, i.e. ~2.3 of the 4.29
    # energy/agent/s movement budget, against a net deficit of -0.76.
    # NOTE this is the opposite of the refuted `wary` variant: that bought
    # safety with foraging time, this spends less time fleeing.
    CHASE_BUDGET = 0.0         # 0 disables; seconds of sprint before we discount it
    CHASE_SPENT = 200.0        # effective distance added once the budget is spent
    PRIORITIZE_FOOD = True
    FOOD_AWARE_BREEDING = False
    # Territory allocation: bind each agent to one patch to cut travel.
    # MEASURED AND REFUTED -- leave TERRITORY off.  Movement is the largest
    # single cost (4.3 of 9.0 energy/agent/s) but it is profitable: income/move
    # is about 8.27/4.29 = 1.9, so confining agents cuts income roughly twice
    # as fast as it cuts travel.  Ten seeds, horizon 3000: off 662.2,
    # HOME_RADIUS 140 -> 681.0, 95 -> 525.7, 70 -> 486.9.  The tight variant
    # cut deaths hard (eaten 245->77, starved 398->218) and still scored worst,
    # because births fell 593->245.  Score improves monotonically as the radius
    # is loosened, i.e. the unrestricted forager is already the optimum.
    TERRITORY = False
    HOME_RADIUS = 95.0         # fruit further than this from home is not worth it
    HOME_TRAVEL = 0.06         # energy/unit used when comparing candidate homes
    HOME_AGE_COST = 0.45       # penalty per second a tree has already been known
    HOME_SWITCH_GAIN = 40.0    # margin required before abandoning a live home

    def __init__(self, **overrides) -> None:
        for key, value in overrides.items():
            if not hasattr(type(self), key):
                raise AttributeError(f"unknown tunable {key!r}")
            setattr(self, key, value)
        self.rng = random.Random(0xC0FFEE)
        self.reset()

    def reset(self) -> None:
        self.minds: Dict[int, Mind] = {}
        self.pending_spawns: List[int] = []
        self.last_time = -1.0
        self.income_ema = 0.0
        self.net_ema = 0.0
        self._net_sum = 0.0
        self._net_n = 0
        self.target_pop = float(self.POP_MAX)
        self.debug: Dict[str, object] = {}
        self._claimed_fruits: set = set()
        self._claimed_trees: set = set()
        self.deaths: List[Tuple[float, float, float, str, bool]] = []
        self.budget = {"income": 0.0, "move": 0.0, "turn": 0.0, "live": 0.0,
                       "spawn": 0.0, "age": 0.0, "ticks": 0.0, "agent_ticks": 0.0}
        self.mode_ticks: Dict[str, int] = {}

    # -- bookkeeping ------------------------------------------------------
    def _register(self, ids: List[int], t: float) -> None:
        new_ids = sorted(i for i in ids if i not in self.minds)
        for k, nid in enumerate(new_ids):
            if k < len(self.pending_spawns) and self.pending_spawns[k] in self.minds:
                p = self.minds[self.pending_spawns[k]]
                self.minds[nid] = Mind(nid, p.world, p.x, p.y, p.th, localized=False, born=t)
            else:
                self.minds[nid] = Mind(nid, World(), 0.0, 0.0, 0.0, born=t)
        self.pending_spawns = []
        alive = set(ids)
        for aid in [a for a in self.minds if a not in alive]:
            m = self.minds[aid]
            self.deaths.append((m.last_state[0], m.last_state[1], m.last_state[2],
                                m.mode, m.aging))
            del self.minds[aid]

    def _localize(self, t: float) -> None:
        """Recover the pose of agents whose frame is not yet known.

        ``rel_dir`` of an Agent observation is ``angle(observed -> observer) -
        observed.direction``, so one sighting of an already localized hive
        member pins down both the observer's position and its heading exactly.
        The same relation is used as a weak correction to stop the members of a
        lineage drifting apart after obstacle collisions.
        """
        for _ in range(2):
            for mind in self.minds.values():
                if mind.localized:
                    continue
                for oid, d, a, rd in mind.sightings:
                    other = self.minds.get(oid)
                    if other is None or not other.localized or other.world is not mind.world:
                        continue
                    ang = rd + other.th          # absolute angle other -> self
                    mind.x = other.x + d * math.cos(ang)
                    mind.y = other.y + d * math.sin(ang)
                    mind.th = _wrap(ang + math.pi - a)
                    mind.localized = True
                    break

        for mind in self.minds.values():
            if not mind.localized:
                if t - mind.born > 3.0:          # parent gone: start a fresh map
                    mind.world = World()
                    mind.x = mind.y = mind.th = 0.0
                    mind.localized = True
                continue
            for oid, d, a, rd in mind.sightings:
                other = self.minds.get(oid)
                if other is None or not other.localized or other.world is not mind.world:
                    continue
                ang = rd + other.th
                ex = other.x + d * math.cos(ang) - mind.x
                ey = other.y + d * math.sin(ang) - mind.y
                if math.hypot(ex, ey) < 120.0:
                    mind.x += 0.05 * ex
                    mind.y += 0.05 * ey
                    mind.th = _wrap(mind.th + 0.05 * _wrap(ang + math.pi - a - mind.th))
                break

    def _perceive(self, mind: Mind, st: dict, t: float) -> None:
        world = mind.world
        if not mind.localized:
            return              # frame unknown: writing to the map would poison it
        fruits: List[Tuple[float, float]] = []
        trees: List[Tuple[float, float]] = []

        for o in st["observations"]:
            ty = o["type"]
            if ty not in _OBS_TYPES:          # /verify posts lowercase types
                ty = ty.capitalize()
            if ty == "Edge":
                (sx, sy), (ex, ey) = o["coords"]
                c, s = math.cos(mind.th), math.sin(mind.th)
                mind.edges[(int(mind.x + sx * c - sy * s), int(mind.y + sx * s + sy * c),
                            int(mind.x + ex * c - ey * s), int(mind.y + ex * s + ey * c))] = t
                continue
            d, a = o["distance"], o["angle"]
            if ty == "Fruit":
                fruits.append((d, a))
            elif ty == "Tree":
                trees.append((d, a))
            elif ty == "Predator":
                x, y = mind.to_world(d, a)
                world.add_predator(x, y, t)

        for (d, a) in trees:
            x, y = mind.to_world(d, a)
            world.add_point(world.trees, x, y, t, 30.0)
        for (d, a) in fruits:
            x, y = mind.to_world(d, a)
            world.add_point(world.fruits, x, y, t, 6.0)

        # Anything inside the (omnidirectional) hearing radius that we did not
        # hear this tick is gone - drop it from memory.  ``add_point`` refreshes
        # last_seen, so a stale timestamp means "not sensed this tick".  The 1s
        # grace guards against another agent's dead-reckoning drift.
        hear = st["hearing_radius"] * 0.92
        for key in [k for k, r in world.trees.items()
                    if t - r[2] > 1.0 and math.hypot(r[0] - mind.x, r[1] - mind.y) < hear]:
            world.forget_tree(key)
        for key in [k for k, r in world.fruits.items()
                    if (t - r[2] > 0.05 and math.hypot(r[0] - mind.x, r[1] - mind.y) < 16.0)
                    or (t - r[2] > 1.0 and math.hypot(r[0] - mind.x, r[1] - mind.y) < hear)]:
            del world.fruits[key]

        # Standing next to a patch means its fruit is being harvested; publish
        # that so the rest of the hive routes itself elsewhere.
        near = (world.fruit_near_trees(t, FRUIT_NEAR_TREE)
                if self.FRUIT_PRIOR > 0.0 else {})
        for key, rec in world.trees.items():
            if math.hypot(rec[0] - mind.x, rec[1] - mind.y) < AT_TREE:
                world.visits[key] = t
                if self.FRUIT_PRIOR > 0.0:
                    world.biomes[key] = st["biome"]
                    # Sitting on a trunk that is producing nothing is the only
                    # evidence available that it is immature, spent or in a river.
                    if near.get(key):
                        world.dry[key] = 0.0
                    else:
                        world.dry[key] = world.dry.get(key, 0.0) + DT

        for key in [k for k, ts in mind.edges.items() if t - ts > EDGE_MEMORY]:
            del mind.edges[key]
        world.expire(t)
        world.visited[(int(mind.x // CELL), int(mind.y // CELL))] = t

    def _update_energy_model(self, mind: Mind, st: dict, t: float) -> None:
        """Infer hidden state (max_age reached, fruit income) from energy deltas."""
        energy, age = st["energy"], st["age"]
        if mind.prev_energy is None:
            mind.prev_energy = energy
            mind.prev_age = age
            return
        expected = mind.prev_energy - mind.last_cost - DT
        if mind.aging:
            expected -= 0.01 * age
        residual = energy - expected
        # Raw delta drives the population controller: it already contains every
        # cost, including the 100 spent on a spawn.
        self._net_sum += energy - mind.prev_energy
        self._net_n += 1
        if residual < -0.3 and not mind.aging:
            mind.age_hits += 1
            if mind.age_hits >= 2:
                mind.aging = True
        elif residual > 1.0:
            mind.income += residual
            mind.last_meal = t
            self.budget["income"] += residual
        mind.prev_energy = energy
        mind.prev_age = age

    # -- navigation helpers ----------------------------------------------
    def _expected_fruit(self, world: World, key: int, since: float, known: int) -> float:
        """Energy believed to be standing unseen around a known tree."""
        lam = self.BIOME_YIELD.get(world.biomes.get(key, ""), self.PRIOR_DEFAULT)
        if lam <= 0.0:
            return 0.0
        # Only the crop grown since the patch was last worked is still there,
        # and nothing outlives PRIOR_LIFE.
        n = lam * max(0.0, min(since, self.PRIOR_LIFE))
        # Fruit already in memory is priced by the fruit loop; do not pay twice.
        n = min(n - known, self.PRIOR_CAP)
        if n <= 0.0:
            return 0.0
        if self.PRIOR_EVIDENCE > 0.0:
            dry = world.dry.get(key, 0.0)
            if dry > 0.0:
                n *= max(0.0, 1.0 - dry / self.PRIOR_EVIDENCE)
        return self.FRUIT_PRIOR * n * self.PRIOR_ENERGY

    def _steer(self, mind: Mind, ang: float, step: float) -> float:
        """Rotate the desired heading until the step lands outside obstacles."""
        if not mind.edges:
            return ang
        if not mind.blocked(mind.x + step * math.cos(ang), mind.y + step * math.sin(ang)):
            return ang
        mind.deflected = True
        for i in range(1, 18):
            for sgn in (1.0, -1.0):
                a = ang + sgn * i * (math.pi / 18.0)
                if not mind.blocked(mind.x + step * math.cos(a), mind.y + step * math.sin(a)):
                    return a
        return ang

    def _goto(self, mind: Mind, tx: float, ty: float, max_speed: float, pen: float,
              stop: float = 0.0) -> Tuple[float, float, float]:
        """Return (command_distance, relative_direction, absolute_heading)."""
        dx, dy = tx - mind.x, ty - mind.y
        dist = math.hypot(dx, dy) - stop
        if dist <= 0.5:
            return 0.0, 0.0, math.atan2(dy, dx) if (dx or dy) else mind.th
        ang = math.atan2(dy, dx)
        step = min(dist, max_speed * pen)
        ang = self._steer(mind, ang, step)
        return min(max_speed, dist / max(pen, 1e-3)), _wrap(ang - mind.th), ang

    def _assign_scouts(self, ctx, t: float) -> None:
        """Keep SCOUT agents on map growth, rotating the job so nobody starves."""
        live = [(mind, st) for mind, st in ctx if mind.localized]
        if len(live) < self.SCOUT_MIN_POP:
            for mind, _ in live:
                mind.scout_until = -1.0
            return
        serving = [m for m, _ in live if t < m.scout_until]
        vacancies = self.SCOUT - len(serving)
        if vacancies <= 0:
            return
        # The richest agent loses the least by skipping meals, and rotating on a
        # commit window stops one agent scouting itself to death.
        idle = sorted((m for m, _ in live if t >= m.scout_until),
                      key=lambda m: (-(m.last_state[2] if m.last_state else 0.0), m.aid))
        for mind in idle[:vacancies]:
            mind.scout_until = t + self.SCOUT_COMMIT

    def _explore_dir(self, mind: Mind, t: float) -> float:
        if t < mind.explore_until:
            return mind.explore_dir
        world = mind.world
        best, best_score = mind.th, -1e18
        for i in range(16):
            a = mind.th + _wrap(i * TWO_PI / 16.0)
            score = 0.0
            ok = True
            for r in (140.0, 280.0, 420.0):
                px, py = mind.x + r * math.cos(a), mind.y + r * math.sin(a)
                if r <= 160.0 and mind.blocked(px, py, AGENT_SIZE + 10.0):
                    ok = False
                    break
                last = world.visited.get((int(px // CELL), int(py // CELL)), -1e5)
                score += min(400.0, t - last)
                for p in world.predators:
                    if math.hypot(p[0] - px, p[1] - py) < 200.0:
                        score -= 900.0
            if not ok:
                continue
            score -= 60.0 * abs(_wrap(a - mind.th))  # straight lines are cheap
            if score > best_score:
                best, best_score = a, score
        mind.explore_dir = best
        mind.explore_until = t + 3.0
        return best

    # -- per-agent decision ----------------------------------------------
    def _plan(self, mind: Mind, st: dict, t: float) -> dict:
        world = mind.world
        energy, max_energy = st["energy"], st["max_energy"]
        speed, sprint = st["speed"], st["sprint_speed"]
        pen = BIOME_MOVE.get(st["biome"], 1.0)
        # Energy burnt per unit of ground covered: movement plus elapsed living
        # cost.  A poor agent must value that far higher than a rich one - a
        # newborn holds 75 energy and dies in ~12 s of walking, so an empty trip
        # kills it while a fat agent merely loses a snack.
        unit_cost = (WALK_COST / pen + DT / (speed * pen)) * (
            1.0 + self.RISK_AVERSION * max(0.0, 1.0 - energy / self.RISK_ENERGY))

        home_rec = (world.trees.get(mind.home)
                    if self.TERRITORY and mind.home is not None else None)

        mind.deflected = False
        cmd = mdir = turn = 0.0
        heading: Optional[float] = None
        spinning = False

        if not mind.localized:
            # Hold still and scan until a hive member fixes our frame.
            mind.mode = "lost"
            cost = 0.13 / TWO_PI
            return {"cmd": 0.0, "mdir": 0.0, "turn": 0.13, "cost": cost,
                    "move_cost": 0.0, "turn_cost": cost,
                    "d_eff": 0.0, "pen": pen, "after": energy - cost}

        # --- threats ----------------------------------------------------
        # world.predators already holds everything sensed this tick.  A predator
        # that has stopped moving is out of energy and resting (it recovers at
        # 30/s from 0 to 100), so it is not worth burning energy over.
        threats = []                  # (effective distance, absolute angle, speed)
        for p in world.predators:
            dx, dy = p[0] - mind.x, p[1] - mind.y
            vmag = math.hypot(p[3], p[4])
            fx, fy = dx + p[3] * 12.0, dy + p[4] * 12.0   # where it is heading
            eff = min(math.hypot(dx, dy), math.hypot(fx, fy))
            if p[5] >= 4.0 and vmag < 1.5:
                eff += 250.0
            elif self.CHASE_BUDGET > 0.0 and p[6] >= 0.0 and t - p[6] > self.CHASE_BUDGET:
                # Out of sprint budget: it must break off within a tick or two,
                # so outrunning it now is paying for safety already guaranteed.
                eff += self.CHASE_SPENT
            # A predator that is not closing on us is not worth burning 5/s to
            # walk away from - there are ~30 of them roaming by the end.
            dist = math.hypot(dx, dy) or 1.0
            closing = -(dx * p[3] + dy * p[4]) / dist
            if closing < CLOSING_RATE and eff > CHASE_DIST:
                eff += BACKOFF_DIST
            threats.append((eff, math.atan2(dy, dx), vmag))
        threats.sort()
        nearest, near_a, near_v = threats[0] if threats else (1e9, 0.0, 0.0)

        # Cannot sprint => cannot outrun anything, so widen every threat radius.
        margin = 1.0 if energy > max_energy * 0.21 else self.NOSPRINT_MARGIN

        if nearest < BACKOFF_DIST * margin:
            rx = ry = 0.0
            for d, a, _ in threats:
                if d > WATCH_DIST * margin:
                    continue
                w = 1.0 / max(d, 25.0) ** 1.2
                rx -= w * math.cos(a)
                ry -= w * math.sin(a)
            for (x1, y1, x2, y2) in mind.edges:          # do not get pinned
                wd = _point_seg_dist(mind.x, mind.y, x1, y1, x2, y2)
                if wd < 80.0:
                    ax, ay = mind.x - (x1 + x2) * 0.5, mind.y - (y1 + y2) * 0.5
                    n = math.hypot(ax, ay) or 1.0
                    w = 0.6 / max(wd, 14.0) ** 1.2
                    rx += w * ax / n
                    ry += w * ay / n
            flee = math.atan2(ry, rx) if (rx or ry) else mind.th
            if nearest < CHASE_DIST and energy > max_energy * 0.21:
                # Outrunning is quadratically expensive: matching the chaser
                # (15/tick) costs 3.4 energy/tick, maxing out costs 5.5 for no
                # extra safety, since the predator runs out of energy first.
                need = max(12.0, near_v * 1.12 if near_v > 2.0 else 16.0) / max(pen, 1e-3)
                top = min(sprint, max(speed, need))
            else:
                top = speed
            heading = near_a                   # facing it downgrades chase to pivot
            flee = self._steer(mind, flee, top * pen)
            cmd, mdir = top, _wrap(flee - mind.th)
            mind.target_fruit = mind.target_tree = None
            mind.mode = "flee"
        else:
            danger = [(p[0], p[1]) for p in world.predators]
            # A scout is off the economy: it does not work patches, but food on
            # its path is still free energy and refusing it just starves it.
            scouting = (self.SCOUT > 0 and t < mind.scout_until
                        and energy > self.SCOUT_FEED)
            frange = self.SCOUT_GRAB if scouting else self.FRUIT_RANGE

            # --- reachable fruit ----------------------------------------
            # A remembered sighting decays in value: another agent may already
            # have eaten it, and a wasted walk is pure loss.
            best_key, best_val = None, self.FRUIT_MIN_VALUE
            if energy < max_energy - 8.0:
                for key, rec in world.fruits.items():
                    if (world, key) in self._claimed_fruits:
                        continue
                    if mind.avoid.get(key, -1e9) > t:
                        continue
                    d = math.hypot(rec[0] - mind.x, rec[1] - mind.y)
                    if d > frange:
                        continue
                    if home_rec is not None and math.hypot(
                            rec[0] - home_rec[0], rec[1] - home_rec[1]) > self.HOME_RADIUS:
                        continue
                    if self.PRIORITIZE_FOOD:
                        travel = max(0.0, d - FRUIT_REACH)
                        ticks = math.ceil(travel / max(speed * pen, 1e-6))
                        drain = DT + (0.01 * (st["age"] + ticks * DT) if mind.aging else 0.0)
                        required = travel * WALK_COST / pen + ticks * drain + 1.0
                        if required >= energy:
                            continue
                    trust = 1.0 if t - rec[2] < 0.15 else max(0.25, 1.0 - (t - rec[2]) / 20.0)
                    worth = self.FRUIT_VALUE
                    if self.HEADROOM_PRICING:
                        worth = min(worth, max(0.0, max_energy - energy))
                    val = worth * trust - unit_cost * d
                    if key == mind.target_fruit:
                        val += self.FRUIT_HYSTERESIS
                    for px, py in danger:
                        if math.hypot(px - rec[0], py - rec[1]) < self.FRUIT_DANGER_RANGE:
                            val -= self.FRUIT_DANGER
                    if val > best_val:
                        best_key, best_val = key, val

            if best_key is not None:
                rec = world.fruits[best_key]
                self._claimed_fruits.add((world, best_key))
                mind.target_fruit, mind.target_tree = best_key, None
                cmd, mdir, heading = self._goto(mind, rec[0], rec[1], speed, pen, FRUIT_REACH)
                mind.mode = "fruit"
            else:
                mind.target_fruit = None
                if scouting:
                    mind.target_tree = None
                    a = self._steer(mind, self._explore_dir(mind, t), speed * pen)
                    cmd, mdir, heading = speed, _wrap(a - mind.th), a
                    mind.mode = "scout"
                elif home_rec is not None:
                    mind.target_tree = mind.home
                    cmd, mdir, heading = self._goto(mind, home_rec[0], home_rec[1],
                                                    speed, pen, TREE_REACH)
                    mind.mode = "patrol"
                    if cmd <= 0.0:
                        spinning = True
                        heading = None
                        mind.mode = "hold"
                else:
                    # --- greedy orienteering over the tree patches ------------
                    # Work a local cluster: cycling ~3 nearby patches captures
                    # nearly all of their production for very little travel.
                    siblings = ([(m.x, m.y) for m in self.minds.values()
                                 if m is not mind and m.localized and m.world is world]
                                if self.SPREAD > 0.0 else ())
                    near = (world.fruit_near_trees(t, FRUIT_NEAR_TREE)
                            if self.FRUIT_PRIOR > 0.0 else {})
                    nkey, ndist, tkey, tval = None, 1e18, None, -1e18
                    for reach in (self.PATROL_RANGE, self.PATROL_FAR):
                        nkey, ndist, tkey, tval = None, 1e18, None, -1e18
                        for key, rec in world.trees.items():
                            d = math.hypot(rec[0] - mind.x, rec[1] - mind.y)
                            if d > reach:
                                continue
                            if mind.avoid.get(key, -1e9) > t:
                                continue
                            if d < ndist:
                                nkey, ndist = key, d
                            since = t - world.visits.get(key, t - 45.0)
                            if self.FRUIT_PRIOR > 0.0:
                                val = self._expected_fruit(world, key, since,
                                                           near.get(key, 0)) - unit_cost * d
                            else:
                                val = (self.TREE_YIELD * min(since, self.YIELD_CAP)
                                       - unit_cost * d)
                            if key == mind.target_tree:
                                val += self.TREE_HYSTERESIS
                            if (world, key) in self._claimed_trees:
                                val -= self.TREE_CLAIMED
                            for sx, sy in siblings:
                                sep = math.hypot(rec[0] - sx, rec[1] - sy)
                                if sep < self.SPREAD_RANGE:
                                    val -= self.SPREAD * (1.0 - sep / self.SPREAD_RANGE)
                            for px, py in danger:
                                if math.hypot(px - rec[0], py - rec[1]) < self.TREE_DANGER_RANGE:
                                    val -= self.TREE_DANGER
                            if val > tval:
                                tkey, tval = key, val
                        if nkey is not None:
                            break

                    at_patch = ndist < AT_TREE
                    # Move to another patch only when it is clearly worth the walk;
                    # otherwise sit on the nearest trunk and let it re-fruit.
                    goal = tkey if (tkey is not None
                                    and (tval > self.MOVE_ON or not at_patch)) else nkey
                    if goal is not None:
                        rec = world.trees[goal]
                        mind.target_tree = goal
                        self._claimed_trees.add((world, goal))
                        cmd, mdir, heading = self._goto(mind, rec[0], rec[1], speed, pen, TREE_REACH)
                        mind.mode = "patrol"
                        if cmd <= 0.0:
                            spinning = True
                            heading = None
                            mind.mode = "hold"
                    else:
                        mind.target_tree = None
                        a = self._steer(mind, self._explore_dir(mind, t), speed * pen)
                        cmd, mdir, heading = speed, _wrap(a - mind.th), a
                        mind.mode = "explore"

        # --- facing ------------------------------------------------------
        if spinning:
            # Turning costs |angle|/2pi, i.e. a fixed 1.0 energy per revolution
            # however fast it is done, so a quick sweep is cheap insurance: the
            # 60 degree vision cone only covers a sixth of the approaches.
            turn = self.SCAN_RATE * mind.spin
        elif heading is not None:
            turn = _wrap(heading - mind.th)
            turn = 0.0 if abs(turn) < 0.03 else max(-0.9, min(0.9, turn))

        if self.BARREN_PATIENCE > 0.0 and mind.mode == "hold":
            if t - mind.last_meal > self.BARREN_PATIENCE:
                if mind.target_tree is not None:
                    mind.avoid[mind.target_tree] = t + self.AVOID_TIME
                mind.target_tree = None
                mind.last_meal = t          # give the next patch a fair hearing

        if self.UNSTICK:
            if mind.deflected and mind.mode in ("fruit", "patrol"):
                mind.stall += 1
                if mind.stall >= self.STALL_TICKS:
                    blocked = mind.target_fruit if mind.mode == "fruit" else mind.target_tree
                    if blocked is not None:
                        mind.avoid[blocked] = t + self.AVOID_TIME
                    mind.target_fruit = mind.target_tree = None
                    mind.stall = 0
            else:
                mind.stall = 0
            for key in [k for k, until in mind.avoid.items() if until <= t]:
                del mind.avoid[key]

        # --- exact cost (identical formula to the environment) -------------
        d_eff = max(0.0, min(cmd, sprint))
        if energy < max_energy / 5.0 and d_eff > speed:
            d_eff = speed
        cost = (d_eff * WALK_COST if d_eff <= speed
                else speed * WALK_COST + (d_eff - speed) * SPRINT_COST)
        turn_cost = min(math.pi, abs(turn)) / TWO_PI
        return {"cmd": cmd, "mdir": mdir, "turn": turn, "cost": cost + turn_cost,
                "move_cost": cost, "turn_cost": turn_cost,
                "d_eff": d_eff, "pen": pen, "after": energy - cost - turn_cost}

    # -- reproduction ------------------------------------------------------
    # A child is always born with exactly 75 energy (environment.spawn_agent
    # passes max_energy but not energy) while sprinting is blocked below
    # max_energy/5.  Breeding max_energy up therefore leaves every newborn
    # unable to sprint until it has nearly tripled its energy, which is how
    # most of them get eaten.  max_energy == 375 puts the threshold exactly at
    # 75, so a child can sprint from birth; above that it is a liability.
    SPRINT_SAFE_MAX_ENERGY = 375.0
    # Breeding weights. These were inert while 90% of births had a single
    # candidate; with BREED_HOLD_ONLY off the pool is ~2.2 and steering works
    # (measured: max_energy moved -51 or +56 on command). Sensing is the
    # interesting target, not storage: removing map visibility from the oracle
    # cost 642 points, the largest effect measured anywhere in this project,
    # and hearing/vision each have 2x headroom (50->100, 200->400) worth 4x area.
    FIT_HEARING = 1.4
    FIT_VISION = 1.2
    FIT_CONE = 0.4
    FIT_STORAGE = 1.0
    FIT_OVERSHOOT = 1.2
    FIT_SPEED = 1.0
    FIT_SPRINT = 1.2

    def _fitness(self, st: dict) -> float:
        # Predation is the dominant cause of death, so weight the traits that
        # win a chase (sprint, sight) as highly as the foraging ones.
        me = st["max_energy"]
        storage = min(me, self.SPRINT_SAFE_MAX_ENERGY) / self.SPRINT_SAFE_MAX_ENERGY
        overshoot = max(0.0, me - self.SPRINT_SAFE_MAX_ENERGY) / self.SPRINT_SAFE_MAX_ENERGY
        return (st["hearing_radius"] / 100.0 * self.FIT_HEARING
                + storage * self.FIT_STORAGE - overshoot * self.FIT_OVERSHOOT
                + st["speed"] / 20.0 * self.FIT_SPEED
                + st["vision_range"] / 400.0 * self.FIT_VISION
                + st["sprint_speed"] / 40.0 * self.FIT_SPRINT
                + st["vision_angle"] / (math.pi / 2.0) * self.FIT_CONE)

    def _good_birth_site(self, mind: Mind) -> bool:
        """Is this agent standing somewhere a 75-energy newborn can survive?"""
        if mind.mode == "hold":
            return True
        if self.BREED_HOLD_ONLY:
            return False
        r2 = self.BREED_SITE_RADIUS * self.BREED_SITE_RADIUS
        for rec in mind.world.trees.values():
            dx, dy = rec[0] - mind.x, rec[1] - mind.y
            if dx * dx + dy * dy < r2:
                return True
        return False

    def _spawn_plan(self, ctx, plans, t: float, pop: int) -> set:
        # mind.income is a leaky accumulator of harvested fruit energy with a
        # 1/(1-0.985) = 66.7 tick (6.67 s) time constant, so dividing by 6.67
        # recovers energy/second.
        rate = sum(m.income for m, _ in ctx) / max(1, pop) / 6.67
        for m, _ in ctx:
            m.income *= 0.985
        self.income_ema += 0.004 * (rate - self.income_ema)

        # Population is driven by *net* energy, not gross income.  Gross income
        # runs ~8/agent/s while true spend runs ~8.7, so an income-driven
        # controller always inflates to the cap and starves the hive.
        if self._net_n:
            self.net_ema += 0.002 * ((self._net_sum / self._net_n) * 10.0 - self.net_ema)
        self._net_sum, self._net_n = 0.0, 0
        if self.net_ema > self.NET_GROW:
            self.target_pop = min(float(self.POP_MAX), self.target_pop + 0.015)
        elif self.net_ema < self.NET_SHRINK:
            self.target_pop = max(float(self.POP_MIN), self.target_pop - 0.015)
        target = int(round(self.target_pop))
        # No predators exist yet and tree cover is at its peak, so the opening
        # is the cheapest population the run will ever buy.
        early = self.EARLY_BOOST and t < self.EARLY_UNTIL
        if early:
            target = self.EARLY_POP

        spawners = set()
        cands = []
        for mind, st in ctx:
            after = plans[mind.aid]["after"]
            routine = self.EARLY_COOLDOWN if early else self.SPAWN_COOLDOWN
            cooldown = self.AGING_COOLDOWN if mind.aging else routine
            if after <= 112.0 or t - mind.last_spawn < cooldown:
                continue
            drain = 1.0 + (0.1 * st["age"] if mind.aging else 0.0)
            cands.append((mind.aging, after / drain, self._fitness(st), after, mind))

        # Past max_age an agent loses 0.1*age energy per second (~10/s) while
        # foraging no better than a newborn that loses 1/s, so replacing it is
        # strongly positive.  It must still respect the target though: firing on
        # every aging agent every tick inflates the population to POP_HARD and
        # starves the hive, which is what used to happen around t=150.  The
        # per-agent cooldown bounds the rate.
        for aging, life_left, _, _, mind in cands:
            if not aging:
                continue
            cap = (self.POP_HARD if (life_left < 15.0 or self.AGING_DUMP)
                   else target + self.POP_SLACK)
            if pop + len(spawners) < cap:
                spawners.add(mind.aid)

        # Routine growth: a child starts on 75 energy and dies in ~12 s of
        # walking, so it must be born sitting on a producing patch, never
        # mid-journey and never next to a predator.
        room = target - pop - len(spawners)
        if room > 0:
            pool = [c for c in cands
                    if c[4].aid not in spawners and c[3] > self.SPAWN_MIN
                    and self._good_birth_site(c[4])
                    and not any(math.hypot(p[0] - c[4].x, p[1] - c[4].y) < 320.0
                                for p in c[4].world.predators)]
            pool.sort(key=lambda c: (-c[2], -c[3], c[4].aid))
            for c in pool[:min(room, 4 if early else 2)]:
                spawners.add(c[4].aid)
        # Surplus about to hit the max_energy clamp is worth more as a child
        # than as nothing, so this pass ignores the population target.
        if self.OVERFLOW_SPAWN:
            for mind, st in ctx:
                if mind.aid in spawners or pop + len(spawners) >= self.POP_HARD:
                    continue
                after = plans[mind.aid]["after"]
                if after <= 112.0 or t - mind.last_spawn < self.OVERFLOW_COOLDOWN:
                    continue
                if st["max_energy"] - after < self.OVERFLOW_HEADROOM:
                    spawners.add(mind.aid)

        if pop <= self.RESCUE_POP:
            for _, _, _, after, mind in cands:
                if after > 170.0:
                    spawners.add(mind.aid)
        if self.FOOD_AWARE_BREEDING:
            return self._guard_births(ctx, plans, t, spawners, target)
        return spawners

    def _guard_births(self, ctx, plans, t: float, proposed: set, target: int) -> set:
        accepted = set()
        sites = []
        limit = min(self.POP_HARD, target + self.POP_SLACK)
        ordered = sorted(ctx, key=lambda item: (
            not item[0].aging, -item[1]["age"], -item[1]["energy"], item[0].aid))
        for mind, state in ordered:
            plan = plans[mind.aid]
            after = plan["after"]
            if (len(ctx) + len(accepted) >= limit or after <= 112.0
                    or t - mind.last_spawn < self.SPAWN_COOLDOWN):
                continue
            successor = any(other.world is mind.world and other.aid != mind.aid
                            and other_state["age"] < 40.0 for other, other_state in ctx)
            replacement = (state["age"] >= 60.0 and not successor
                           and after > max(170.0, SPAWN_COST + state["max_energy"] * 0.21 + 12.0))
            if mind.aid not in proposed and not replacement:
                continue
            emergency = len(ctx) <= 2 and mind.aging and not successor
            heading = mind.th + plan["mdir"]
            birth_x = mind.x + plan["d_eff"] * plan["pen"] * math.cos(heading)
            birth_y = mind.y + plan["d_eff"] * plan["pen"] * math.sin(heading)
            if any(world is mind.world and math.hypot(birth_x - pos_x, birth_y - pos_y) < 120.0
                   for world, pos_x, pos_y in sites):
                continue
            nearby_young = any(
                other.aid != mind.aid and other.world is mind.world
                and other_state["age"] < 15.0
                and math.hypot(birth_x - other.x, birth_y - other.y) < 120.0
                for other, other_state in ctx)
            trees = any(t - record[2] <= 1.0
                        and math.hypot(birth_x - record[0], birth_y - record[1]) < 80.0
                        for record in mind.world.trees.values())
            food = sum(t - record[2] <= 1.0
                       and math.hypot(birth_x - record[0], birth_y - record[1]) < 100.0
                       for record in mind.world.fruits.values())
            productive = trees and (mind.income / 6.67 > self.INCOME_LOW or food >= 2)
            danger = any(math.hypot(birth_x - record[0], birth_y - record[1]) < 320.0
                         for record in mind.world.predators)
            if not emergency and (not mind.localized or mind.mode in ("flee", "lost", "explore", "scout")
                                  or nearby_young or danger or not productive):
                continue
            accepted.add(mind.aid)
            sites.append((mind.world, birth_x, birth_y))
        return accepted

    def _assign_homes(self, ctx, t: float) -> None:
        """Give every agent its own patch, and hand it over before it dies.

        A tree's age is never observable, so ``rec[3]`` (when we first saw it)
        stands in for it: trees live ~58 s, so one we have known a long time is
        close to expiring and is worth trading for a freshly sighted one.
        """
        groups: Dict[int, List[Mind]] = {}
        for mind, _ in ctx:
            groups.setdefault(id(mind.world), []).append(mind)

        for minds in groups.values():
            trees = minds[0].world.trees
            order = sorted(minds, key=lambda m: m.aid)
            taken = set()
            for mind in order:
                if mind.home is not None and mind.home in trees and mind.home not in taken:
                    taken.add(mind.home)
                else:
                    mind.home = None
            if not trees:
                continue
            for mind in order:
                held = trees.get(mind.home) if mind.home is not None else None
                stay = (-self.HOME_AGE_COST * (t - held[3])
                        - self.HOME_TRAVEL * math.hypot(held[0] - mind.x, held[1] - mind.y)
                        if held is not None else -1e18)
                best, best_val = None, -1e18
                for key, rec in trees.items():
                    if key in taken:
                        continue
                    val = (-self.HOME_AGE_COST * (t - rec[3])
                           - self.HOME_TRAVEL * math.hypot(rec[0] - mind.x, rec[1] - mind.y))
                    if val > best_val:
                        best, best_val = key, val
                if best is None:
                    continue
                if held is None or best_val > stay + self.HOME_SWITCH_GAIN:
                    if mind.home is not None:
                        taken.discard(mind.home)
                    mind.home = best
                    taken.add(best)

    def _planning_order(self, ctx):
        if not self.PRIORITIZE_FOOD:
            return ctx
        return sorted(ctx, key=lambda item: (
            item[1]["energy"] / (
                WALK_COST * item[1]["speed"] + DT
                + (0.01 * item[1]["age"] if item[0].aging else 0.0)),
            item[0].aid))

    # -- main entry --------------------------------------------------------
    def act(self, agent_states: List[dict], sim_time: float) -> List[ActionRequest]:
        return [ActionRequest(**a) for a in self.act_dicts(agent_states, sim_time)]

    def act_dicts(self, agent_states: List[dict], sim_time: float) -> List[dict]:
        """Hot path used by the server: skips building pydantic models.

        The evaluation server allows 600 s of accumulated response time for
        30000 ticks, i.e. 20 ms per tick including network.
        """
        t = float(sim_time)
        if t + 1e-9 < self.last_time:
            self.reset()
        self.last_time = t

        self._register([st["agent_id"] for st in agent_states], t)

        for st in agent_states:
            mind = self.minds[st["agent_id"]]
            mind.sightings = [(o["id"], o["distance"], o["angle"], o["rel_dir"])
                              for o in st["observations"]
                              if o["type"].capitalize() == "Agent"
                              and "id" in o and "rel_dir" in o]
        self._localize(t)

        ctx = []
        for st in agent_states:
            mind = self.minds[st["agent_id"]]
            mind.last_state = (t, st["age"], st["energy"])
            self._update_energy_model(mind, st, t)
            self._perceive(mind, st, t)
            ctx.append((mind, st))

        plans = {}
        self._claimed_fruits = set()
        self._claimed_trees = set()
        if self.TERRITORY:
            self._assign_homes(ctx, t)
        if self.SCOUT > 0:
            self._assign_scouts(ctx, t)
        for mind, st in self._planning_order(ctx):
            plans[mind.aid] = self._plan(mind, st, t)
        spawners = self._spawn_plan(ctx, plans, t, len(ctx))
        actions = []
        for mind, st in ctx:
            p = plans[mind.aid]
            spawn = mind.aid in spawners
            mind.last_cost = p["cost"] + (SPAWN_COST if spawn else 0.0)
            if spawn:
                mind.last_spawn = t
                self.pending_spawns.append(mind.aid)
            actions.append({
                "agent_id": int(mind.aid),
                "move_distance": float(p["cmd"]),
                "move_direction": float(p["mdir"]),
                "turn_angle": float(p["turn"]),
                "spawn_agent": bool(spawn),
            })
            # dead reckoning for the next tick
            disp = p["d_eff"] * p["pen"]
            a = mind.th + p["mdir"]
            mind.x += disp * math.cos(a)
            mind.y += disp * math.sin(a)
            mind.th = _wrap(mind.th + p["turn"])

        if ctx:
            b = self.budget
            b["ticks"] += 1.0
            b["agent_ticks"] += len(ctx)
            b["live"] += DT * len(ctx)
            b["spawn"] += SPAWN_COST * len(spawners)
            for mind, st in ctx:
                p = plans[mind.aid]
                b["move"] += p["move_cost"]
                b["turn"] += p["turn_cost"]
                if mind.aging:
                    b["age"] += 0.01 * st["age"]
                self.mode_ticks[mind.mode] = self.mode_ticks.get(mind.mode, 0) + 1
            modes: Dict[str, int] = {}
            for mind, _ in ctx:
                modes[mind.mode] = modes.get(mind.mode, 0) + 1
            self.debug = {"pop": len(ctx), "target": round(self.target_pop, 1),
                          "income": round(self.income_ema, 2),
                          "net": round(self.net_ema, 2),
                          "energy": round(sum(s["energy"] for _, s in ctx) / len(ctx), 1),
                          "modes": modes}
        return actions


_HIVE = Hive()


def action_decision_all(agent_states: List[dict], sim_time: float) -> List[ActionRequest]:
    """Module level convenience wrapper around a singleton :class:`Hive`."""
    return _HIVE.act(agent_states, sim_time)
