"""Measures where the food economy actually leaks, for a given controller.

    python mechanics_probe.py            # current Hive defaults
    python mechanics_probe.py --seeds 1 2 3 4

Reports, per run: what fraction of spawned fruit was harvested vs left to rot,
the energy each harvested fruit carried (20 at spawn, 60 once ripe), energy
thrown away by harvesting into a full stomach, and the population held against
the number of fruit-bearing trees actually alive.
"""

import argparse
import contextlib
import io
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

MATURE_TREE_AGE = 20.0      # trees only bear fruit from this age
RIPE_ENERGY = 60.0          # fruit energy cap
ROT_AGE = 100.0             # fruit age at which it is removed unharvested


def _probe_one(job):
    seed, horizon, params = job
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    from local_playground import local_simulation
    from src.utils.controllers.expert_agent_policy import Hive

    harvested, rotted, overflow = [], [], []
    samples = []

    class ProbedHive(Hive):
        def act_dicts(self, agent_states, sim_time):
            env = ProbedHive.env
            # Sampled before the environment mutates so eat events line up.
            if env is not None and abs(sim_time % 25.0) < 0.05:
                mature = sum(1 for tree in env.trees if tree.age >= MATURE_TREE_AGE)
                samples.append({
                    "t": round(sim_time),
                    "agents": len(env.agents),
                    "mature_trees": mature,
                    "fruits": len(env.fruits),
                    "predators": len(env.predators),
                    "mean_energy": round(statistics.mean(
                        [a.energy for a in env.agents]) if env.agents else 0.0, 1),
                })
            return super().act_dicts(agent_states, sim_time)

    ProbedHive.env = None
    hive = ProbedHive(**params)

    original_local_simulation = local_simulation

    def instrument(env):
        ProbedHive.env = env
        original_remove = env.remove_fruit

        def patched_remove(fruit):
            # The rot branch is the only caller once age has passed ROT_AGE;
            # every other removal is an agent eating it.
            if fruit.age > ROT_AGE:
                rotted.append(fruit.energy)
            else:
                harvested.append(fruit.energy)
                eaters = [a for a in env.agents
                          if (a.x - fruit.x) ** 2 + (a.y - fruit.y) ** 2
                          <= (a.size + fruit.radius) ** 2]
                for agent in eaters:
                    spare = agent.max_energy - agent.energy
                    overflow.append(max(0.0, fruit.energy - spare))
            original_remove(fruit)

        env.remove_fruit = patched_remove

    from src.core import SimulationCore
    original_create = SimulationCore._create_env

    def patched_create(self):
        env = original_create(self)
        instrument(env)
        return env

    SimulationCore._create_env = patched_create
    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = original_local_simulation(verbose=False, seed=seed, hive=hive,
                                               max_time=horizon, log_every=0,
                                               diagnostics=True)
    finally:
        SimulationCore._create_env = original_create

    spawned = len(harvested) + len(rotted)
    harvested_energy = sum(harvested)
    return {
        "seed": seed,
        "score": round(result["score"], 1),
        "sim_time": round(result["time"], 1),
        "fruit_spawned": spawned,
        "fruit_harvested": len(harvested),
        "fruit_rotted": len(rotted),
        "harvest_rate": round(len(harvested) / max(1, spawned), 3),
        "mean_harvest_energy": round(statistics.mean(harvested), 1) if harvested else 0.0,
        "median_harvest_energy": round(statistics.median(harvested), 1) if harvested else 0.0,
        "ripe_share": round(sum(e >= RIPE_ENERGY - 1 for e in harvested)
                            / max(1, len(harvested)), 3),
        "unripe_share": round(sum(e <= 25.0 for e in harvested)
                              / max(1, len(harvested)), 3),
        "harvested_energy": round(harvested_energy),
        "energy_left_to_rot": round(sum(rotted)),
        "overflow_wasted": round(sum(overflow)),
        "headroom_if_all_ripe": round(len(harvested) * RIPE_ENERGY - harvested_energy),
        "eaten": result["stats"]["eaten"],
        "starved": result["stats"]["starved"],
        "samples": samples,
        "wall": round(time.perf_counter() - started, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13, 14])
    parser.add_argument("--horizon", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    jobs = [(seed, args.horizon, {}) for seed in args.seeds]
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_probe_one, job) for job in jobs]
        for future in as_completed(futures):
            report = future.result()
            reports.append(report)
            trace = report.pop("samples")
            print(json.dumps(report), flush=True)
            report["samples"] = trace

    def total(key):
        return sum(r[key] for r in reports)

    harvested = total("fruit_harvested")
    print("SUMMARY " + json.dumps({
        "runs": len(reports),
        "mean_score": round(statistics.mean(r["score"] for r in reports), 1),
        "fruit_spawned": total("fruit_spawned"),
        "fruit_harvested": harvested,
        "harvest_rate": round(harvested / max(1, total("fruit_spawned")), 3),
        "mean_harvest_energy": round(total("harvested_energy") / max(1, harvested), 1),
        "ripe_share": round(statistics.mean(r["ripe_share"] for r in reports), 3),
        "unripe_share": round(statistics.mean(r["unripe_share"] for r in reports), 3),
        "harvested_energy": total("harvested_energy"),
        "energy_left_to_rot": total("energy_left_to_rot"),
        "overflow_wasted": total("overflow_wasted"),
        "headroom_if_all_ripe": total("headroom_if_all_ripe"),
    }), flush=True)

    print("\n t     agents  mature_trees  fruits  predators  mean_energy", flush=True)
    trace = max(reports, key=lambda r: r["sim_time"])
    print(f"(seed {trace['seed']}, the longest run)", flush=True)
    for row in trace["samples"]:
        if row["t"] % 100 == 0:
            print(f"{row['t']:5d}  {row['agents']:6d}  {row['mature_trees']:12d}"
                  f"  {row['fruits']:6d}  {row['predators']:9d}  {row['mean_energy']:11.1f}",
                  flush=True)


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _here)
    os.chdir(_here)
    main()
