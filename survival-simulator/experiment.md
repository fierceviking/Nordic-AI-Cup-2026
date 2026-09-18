# Survival simulator — experiment log

Engineering log for the hand-written controller in
[expert_agent_policy.py](src/utils/controllers/expert_agent_policy.py).
No machine learning is used: every rule is derived from the simulator's own
energy / aging / predator equations, then verified against measured runs.

## How to reproduce

```cmd
cd survival-simulator
python local_playground.py bench 1 2 3          # headless, multi-seed
python -c "from local_playground import local_simulation; local_simulation(verbose=False, seed=1, log_every=300, diagnostics=True)"
python local_playground.py                      # rendered, single run
```

`diagnostics=True` adds death-cause attribution, a per-agent energy budget, a
mode histogram and a dead-reckoning accuracy check. `sweep()` runs the same
seeds under several tunable settings.

## Scoring model and theoretical ceiling

Read off [environment.py](src/elements/environment.py) `non_agent_step`:

| Term | Value | Notes |
|---|---|---|
| Survival | `score += dt` every tick, unconditionally | 30000 ticks x 0.1 = **3000** |
| Fruit eaten | `+ fruit.energy / 1000` | 0.02–0.06 each |
| Agent eaten | `- agent.energy / 100` | up to -5.0 per kill |

The run stops when `num_agents == 0` or `time > 3000`. So **score ≈ survival
time in seconds**, and the theoretical maximum is ~3000 plus a small fruit
bonus (eating ~2000 fruits is worth only ~+70). Everything else is noise:
the entire optimisation problem is *keep at least one agent alive to t=3000*.

Current status: **mean 726 / 3000 over 10 held-out seeds**, 0/10 reach the time
limit — every run still ends in extinction, between t=490 and t=1054.

## Determinism caveat (important)

The README claims the simulation is deterministic per seed. It is **not**
reproducible across processes: `_get_local_objects` builds `set()`s of
`Agent` / `Fruit` / `Tree` objects, which hash by `id()`, so iteration order
depends on memory addresses. That changes which fruit gets eaten first and
the order of observations. Two runs of the identical policy on seed 1 gave
**1147.5** and **676.8**.

Consequence: single-seed comparisons are worthless. Differences below roughly
±300 points on one seed are not signal.

## Derived constants that drive the design

| Quantity | Derivation | Value |
|---|---|---|
| Living drain | `dt * energy_drain_rate`, all biomes 1.0 | 1.0 /s |
| Walking | `0.05 * distance`, distance capped by `speed` | 5.0 /s at full walk |
| Sprinting | `speed*0.05 + (d-speed)*0.5` | 55 /s at speed 20 |
| Travel cost | move + elapsed living cost | **0.06 / unit** |
| Aging | `energy -= 0.01*age` **per tick** past `max_age` (60–120 s) | 0.1*age /s |
| Lifespan, no food | newborn 75 E | dies at age ~97 |
| Lifespan, full 500 E | idle only | dies at age ~125 |
| Spawn | costs 100, child gets 75, age resets | 25 E lost per generation |
| Generations needed | 3000 / ~100 s | **~30** |
| Trees | spawn rate decays `0.5**(t/300)`, lifetime ~57 s | ~107 at t=0, ~27 at t=1200, ~3 at t=3000 |
| Map-wide food | late game | ~0.2 fruit/s ≈ 8 E/s |
| Predators | never die, `N ≈ 0.01*t` | 30 by t=3000 |

Two exploitable facts:

* `move_direction` is **relative to the agent's heading** (the environment adds
  `entity.direction`), contradicting the README. An agent can therefore retreat
  while still *facing* a predator.
* A predator only charges when `|rel_dir| > pi/2` (agent facing away) **or**
  when closer than `1.5 * hearing_radius = 90`. Facing it downgrades the chase
  to a pivot it cannot sustain — it burns 2.55 E/tick from a ~100 E budget, so
  it hunts ~4 s then rests ~3.3 s.

## Experiment log

| # | Version | Seed 1 score | Key diagnostic | Verdict |
|---|---|---|---|---|
| 1 | Camp on one tree + claims, pop cap 16 | **1147.5** / **676.8** | eaten 36, starved 163, mean age 40.9 | Baseline; huge variance |
| 2 | Patch orienteering, shared fruit memory | **858.0** | eaten 63, starved 153, mean age 39.7 | Chased stale fruit; 100% of agents in `fruit` mode |
| 3 | + per-tick claims, localization, pop cap 8 | **709.5** / **515.2** | 44% of deaths before age 25 | Energy deficit, high churn |

Held-out validation (seeds 11-20, full 3000 s) is the only trustworthy measure:

| Version | Mean | Min | Max | Survived |
|---|---|---|---|---|
| v3 defaults | 707.2 | 484.5 | 997.4 | 0/10 |
| + Optuna round 1 (`MOVE_ON`, `RISK_*`) + `max_energy` fitness fix | 689.8 | 442.6 | 995.6 | 0/10 |
| + net-energy population controller | 700.0 | 534.4 | 901.0 | 0/10 |
| **+ spawn-explosion fix (current)** | **726.3** | 489.9 | 1053.4 | 0/10 |
| Optuna round 2 "best" params | 633.0 | 440.8 | 926.7 | 0/10 |

### Run 1a — camp policy, seed 1 (no diagnostics)

```
Score:  111.99 | Agents: 16 | Predators: 0 | Trees: 83 | Fruits:  97 | Time:  100.0
Score:  227.10 | Agents: 16 | Predators: 0 | Trees: 64 | Fruits:  24 | Time:  200.0
Score:  337.43 | Agents: 13 | Predators: 2 | Trees: 70 | Fruits:  52 | Time:  300.0
Score:  448.26 | Agents: 17 | Predators: 2 | Trees: 57 | Fruits:  42 | Time:  400.0
Score:  546.23 | Agents:  5 | Predators: 4 | Trees: 50 | Fruits: 123 | Time:  500.0
Score:  650.15 | Agents:  8 | Predators: 5 | Trees: 46 | Fruits:  86 | Time:  600.0
Score:  750.95 | Agents:  5 | Predators: 5 | Trees: 40 | Fruits:  57 | Time:  700.0
Score:  850.68 | Agents:  8 | Predators: 5 | Trees: 29 | Fruits:  37 | Time:  800.0
Score:  951.50 | Agents:  4 | Predators: 6 | Trees: 28 | Fruits:  52 | Time:  900.0
Score: 1050.58 | Agents:  9 | Predators: 6 | Trees: 29 | Fruits:  50 | Time: 1000.0
Game over! Final Score: 1147.4634739579135
```

### Run 1b — identical code and seed, second process

```
Score:  111.63 | Agents: 16 | Predators: 0 | Trees: 75 | Fruits: 129 | Time: 100.0
Score:  231.18 | Agents: 16 | Predators: 0 | Trees: 72 | Fruits:  29 | Time: 200.0
Score:  342.99 | Agents: 16 | Predators: 2 | Trees: 61 | Fruits:  50 | Time: 300.0
Score:  452.93 | Agents: 11 | Predators: 3 | Trees: 63 | Fruits:  46 | Time: 400.0
Score:  554.27 | Agents: 12 | Predators: 4 | Trees: 46 | Fruits:  53 | Time: 500.0
Score:  646.60 | Agents:  4 | Predators: 5 | Trees: 39 | Fruits:  76 | Time: 600.0
Game over! Final Score: 676.7551086669561
deaths: eaten=36 starved=163 mean_age=40.9 score_lost_to_predators=26.9 target_pop=16.0 income_ema=6.62
```

Starvation outnumbers predation 4.5:1, and the mean age at death (40.9) is far
below the ~97 s a newborn survives doing nothing. Agents were burning energy on
movement faster than they earned it.

### Run 2 — patch orienteering, shared fruit memory

```
Score: 172.45 | Agents: 16 | Predators: 1 | Trees: 77 | Fruits: 103 | Time: 150.0 | {'pop': 16, 'target': 13.0, 'income': 8.74, 'energy': 225.0, 'modes': {'fruit': 16}}
Score: 320.23 | Agents: 14 | Predators: 3 | Trees: 57 | Fruits:  56 | Time: 300.0 | {'pop': 14, 'target': 13.0, 'income': 9.15, 'energy': 235.9, 'modes': {'flee': 1, 'fruit': 13}}
Score: 474.55 | Agents: 13 | Predators: 3 | Trees: 53 | Fruits:  37 | Time: 450.0 | {'pop': 13, 'target': 13.0, 'income': 8.29, 'energy': 141.8, 'modes': {'fruit': 11, 'flee': 2}}
Score: 624.27 | Agents: 10 | Predators: 6 | Trees: 50 | Fruits:  71 | Time: 600.0 | {'pop': 10, 'target': 13.0, 'income': 6.77, 'energy': 122.4, 'modes': {'fruit': 10}}
Score: 765.60 | Agents:  6 | Predators: 7 | Trees: 33 | Fruits:  39 | Time: 750.0 | {'pop':  6, 'target': 13.0, 'income': 8.98, 'energy': 131.5, 'modes': {'flee': 4, 'fruit': 2}}
Game over! Final Score: 858.0088547853451
deaths: eaten=63 starved=153 mean_age=39.7 score_lost_to_predators=82.5 target_pop=13.0 income_ema=5.00
```

The mode histogram is the finding: `{'fruit': 16}` — *every* agent was walking
to a fruit *every* tick and the patch logic never ran. Fruit memory is shared
across a lineage and was valued at `32 - 0.06*d` out to 400 units, so any stale
sighting anywhere looked profitable. Agents repeatedly walked 400 units
(24 energy) to fruit another agent had already eaten.

Fixes: cap the chase at 220 units, decay a sighting's value with its age,
and claim a fruit per tick so two agents never target the same one.

### Run 3 — current version, seed 1

```
Score: 332.68 | Agents: 8 | Predators: 3 | Trees: 61 | Fruits: 90 | Time: 300.0 | {'pop': 8, 'target': 8.0, 'income':  8.70, 'energy': 369.3, 'modes': {'fruit': 1, 'patrol': 7}}
Score: 595.55 | Agents: 8 | Predators: 8 | Trees: 46 | Fruits: 24 | Time: 600.0 | {'pop': 8, 'target': 8.0, 'income': 10.14, 'energy': 212.7, 'modes': {'patrol': 5, 'hold': 3}}
Game over! Final Score: 709.5049086228637
deaths: eaten=58 starved=115 mean_age=44.9 score_lost_to_predators=86.8
death modes={'explore': 1, 'fruit': 39, 'hold': 13, 'patrol': 68, 'flee': 52} died_before_age25=81/173 aged_out=37
energy per agent-second: income=8.27 move=4.81 turn=0.40 live=1.00 spawn=2.16 age=0.52
time in mode: patrol=38% fruit=37% hold=20% flee=5% explore=1%
```

Second process, same code and seed (and with the dead-reckoning metric fixed):

```
Game over! Final Score: 515.1882428928309
deaths: eaten=46 starved=86 mean_age=43.9 score_lost_to_predators=57.1
death modes={'explore': 1, 'fruit': 35, 'patrol': 51, 'hold': 10, 'flee': 35} died_before_age25=58/132 aged_out=32
dead-reckoning mean error per tick: 0.060 units (max 26.8)
energy per agent-second: income=7.89 move=4.58 turn=0.41 live=1.00 spawn=2.19 age=0.52
time in mode: fruit=41% patrol=40% hold=15% flee=4% explore=1%
```

## Hyperparameter search (Optuna)

[tune.py](tune.py) runs a TPE study over the `Hive` class tunables; a
`ProcessPoolExecutor` evaluates a fixed seed set per trial (common random
numbers). [tune_status.py](tune_status.py) reports a running study.

```cmd
python tune.py baseline                       # current defaults, seeds 11-20
python tune.py search --trials 60 --jobs 3 --horizon 1200 --study pop2 --storage sqlite:///optuna_pop2.db
python tune_status.py optuna_pop2.db
python tune.py validate --params best_params.json
```

Practical notes:

* A full 3000 s run takes ~350 s; env creation is only ~3 s, so there is no
  point short-circuiting the renderer. Runs that die early are cheap, so the
  search naturally spends its budget on good configurations.
* Search uses a shortened `--horizon` (1000-1200 s) purely to bound cost; the
  winners are then validated at the full 3000 s.
* **Workers must stay <= 7.** Map generation allocates a ~150 MB numpy
  temporary (`(10, 1600, 1200)` int64), and 12 concurrent runs exhausted RAM
  mid-search with `_ArrayMemoryError`.

### Round 1 — 6 params, 31 trials, horizon 1000, seeds 1-4

Top 8 trials spanned 617-738, while a *single* configuration produced
`[333, 793, 855, 552]` across its four seeds. Per-seed noise (~±100 s.e. on a
4-seed mean) is as large as the entire between-config spread, so individual
rankings are meaningless. Only consistent trends across the top trials are
worth reading:

| Param | Top-8 range | Default was | Read |
|---|---|---|---|
| `MOVE_ON` | 30.1 – 38.1 | 18 | **strong**: stay on a patch far longer |
| `RISK_AVERSION` | 0.08 – 1.04 | 2.0 | **strong**: drop the energy-scaled travel penalty |
| `RISK_ENERGY` | 230 – 310 | 220 | weak, centred ~270 |
| `POP_MAX` | 6 – 15 | 8 | no signal |
| `SPAWN_MIN` | 190 – 400 | 300 | no signal |
| `TREE_YIELD` | 1.0 – 4.0 | 2.2 | no signal |

Adopted `MOVE_ON=35`, `RISK_AVERSION=0.5`, `RISK_ENERGY=270`. Both strong
signals point the same way — movement was 4.58 of the 8.70 energy/agent-s
spend, and a simple high threshold beats my hand-tuned scaling.

`POP_MAX` showing no signal was itself the clue: the population controller was
overriding it (see the bug below).

### Round 2 — population params, 17 trials, horizon 1200

Re-run after the spawn-explosion fix, since that bug had flattened the
landscape. Best trial (746.5 on seeds 1-4) validated at **633.0** on the
held-out seeds 11-20 — *worse* than the untuned defaults at 726.3. Textbook
overfitting to four noisy seeds. **The searched parameters were discarded.**

### Verdict

Tuning was worth running but not for its parameters: three separate rounds of
changes all landed at the same ~700 mean. What the search actually bought was
the evidence that the tunables are *not* the bottleneck, which redirected the
effort into instrumenting the population dynamics — where the real bug was.

## Bugs found by instrumenting

1. **Children get a random heading.** `Creature.__init__` randomises
   `direction`, so a newborn does *not* inherit the parent's orientation. The
   controller had assumed it did, so every newborn navigated in a randomly
   rotated frame and corrupted the shared lineage map. Fixed by solving for a
   child's exact pose from one sighting of a known agent — observations of
   type `Agent` carry both `id` and `rel_dir`, which is enough to recover the
   observed agent's full pose. Unlocalized agents are now barred from writing
   to the shared map.
2. **Dead-reckoning metric was off by one tick.** It compared the *next*
   tick's prediction against the *current* tick's actual displacement,
   reporting 4.640 units/tick of error. Corrected, the real figure is
   **0.060 units/tick** (the 26.8 max comes from the environment rotating a
   blocked move in 10° steps, which is not predictable). Localization is sound
   and is not the cause of the remaining losses.
3. **Population explosion — the big one.** The rule "an aging agent converts
   its energy into a child" was gated only by `POP_HARD`, ignored `target_pop`
   entirely, and had no per-agent cooldown, so it re-fired *every tick* on
   *every* aging agent. When the initial cohort aged out together the
   population rocketed past its target of 8 straight to the hard cap:

   ```
   t= 50  pop  6  target 8  net +3.72  energy 389
   t=100  pop 17  target 8  net +2.79  energy 257
   t=150  pop 22  target 8  net +1.22  energy 287
   t=200  pop 22  target 4  net -2.06  energy 154
   t=450  pop  3  target 3  net -1.85  energy 134   -> extinct at t=524
   ```

   99 of 116 deaths were starvation. Fixed by gating the rule on `target + 2`
   (only bypassing to `POP_HARD` within 15 s of death) plus a 4 s per-agent
   `SPAWN_COOLDOWN`. Seed 11 went 497 -> 766 and total deaths halved, 116 -> 60.
4. **The population controller optimised the wrong quantity.** It grew the
   hive whenever *gross* fruit income exceeded 3.2/agent-s, but true spend is
   ~8.7/agent-s, so it always inflated to the cap. Now driven by measured
   **net** energy per agent-second (`NET_GROW` / `NET_SHRINK`), which is a
   genuine negative feedback loop.
5. **Breeding for `max_energy` was an own-goal.** `spawn_agent` passes
   `max_energy` to the child but *not* `energy`, so every child starts at
   exactly 75, while `update_entity_position` blocks sprinting below
   `max_energy / 5`. Breeding `max_energy` up therefore leaves newborns unable
   to sprint until they nearly triple their energy — and they were ~44% of all
   deaths. Fitness now peaks at `max_energy == 375`, where the sprint
   threshold is exactly 75 and a child can sprint from birth.


## Current energy budget (per agent-second)

| Term | Value |
|---|---|
| income (fruit) | +7.89 |
| move | -4.58 |
| turn | -0.41 |
| live | -1.00 |
| spawn | -2.19 |
| aging | -0.52 |
| **net** | **-0.81** |

This is the whole problem in one line. The hive runs a structural deficit and
covers it by letting agents die and replacing them — 132 deaths in 515 s at a
population of 8 is a mean lifespan of ~31 s against a no-food floor of ~97 s.

* Movement is 4.58 of the 8.70 spend, and agents are moving 81% of the time.
* Spawning is 2.19, i.e. ~0.17 spawns/s, purely to replace the churn.
* 44% of deaths occur before age 25 — newborns start with 75 E and die within
  ~13 s if they walk at full speed without finding food.

## Next steps

1. **Predation is now the leading killer** (33 of 60 deaths on seed 11, 43% of
   them in `flee` mode). Detection is the bottleneck: a predator senses at
   hearing 60 / vision 250 while an agent senses at 50 / 200, and agents face
   their direction of travel ~80% of the time, so a predator closing from
   behind is inside the 90-unit charge radius before it is ever seen.
2. **The aging tail is the largest controllable cost** at 3.0 energy/agent-s.
   An agent past `max_age` burns ~10/s while foraging no better than a newborn
   that burns 1/s, so replacing it is strongly positive — currently limited by
   needing >212 energy to spawn and keep a reserve.
3. Consider breeding `hearing_radius` harder (cap is `chunk_size/4 = 100`): it
   doubles both the fruit-detection radius while holding *and* the predator
   warning distance. Selection is currently weak because agents average only
   ~1-2 generations before dying.
4. Any future tuning must validate on >= 10 held-out seeds — round 2 showed a
   4-seed search winner losing 93 points on held-out data.

## Serving

[agent_server.py](agent_server.py) holds one persistent `Hive`; it carries the
world map, per-agent dead reckoning and the population controller across ticks,
and `Hive.act` resets itself when `sim_time` jumps backwards, so the three
back-to-back evaluation runs need no restart. Exceptions are caught and
downgraded to stand-still actions, because a missing response ends the run.

[server_smoketest.py](server_smoketest.py) drives a real simulation through the
HTTP endpoint and checks the response contract:

```cmd
python agent_server.py          # terminal 1
python server_smoketest.py      # terminal 2
```

Measured: 400 ticks at **10.2 ms** mean latency (server-side budget is 10 s per
response, 600 s accumulated), correct agent-id set per tick, and a verified
hive reset when a second run restarts `sim_time`.
