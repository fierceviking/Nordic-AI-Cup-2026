"""Does selective breeding actually move the population's traits?

    python trait_probe.py --seeds 11 12 --horizon 1500

`Hive._fitness` already ranks candidate parents by trait quality, and already
penalises max_energy above SPRINT_SAFE_MAX_ENERGY (375) because the sprint gate
is max_energy/5 and a newborn always starts on exactly 75 energy.

But selection can only act on variation that exists. The environment mutates a
trait on only 10% of births (+-50%), so with ~100 births a run there may simply
be nothing to select. This measures whether the distribution moves at all:

  spread    min/mean/max of each trait across the living population
  drift     population mean vs the founder value
  mutants   how many living agents differ from the founder value

It also decomposes WHERE any steering signal dies, which the mean alone hides:

  selection differential  mean(max_energy of parents the hive CHOSE)
                          minus mean(max_energy of the living population).
                          If this is ~0 the hive is not selecting on the trait
                          at all, whatever its fitness weights say.
  mutation step           mean(child) minus mean(chosen parent). The environment
                          multiplies by uniform(0.5, 1.5) on 10% of births, so
                          this is the noise the selection differential must beat.

A differential near zero means the controller is not steering. A large
differential with no drift means mutation is drowning it. A differential that
does move the distribution, while score fails to follow, means the trait simply
does not pay -- three different conclusions.
"""

import argparse
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

FOUNDER = {"max_energy": 500.0, "hearing_radius": 50.0, "vision_radius": 200.0,
           "speed": 10.0, "sprint_speed": 20.0}
SAMPLE_EVERY = 100.0


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

    rows = []
    births = 0
    seen_ids = set()
    ever_mutated = {k: 0 for k in FOUNDER}
    differentials = []        # chosen-parent mean minus population mean
    mutation_steps = []       # newborn minus the parents chosen that tick

    actions = []
    state = sim.step(actions)
    while True:
        reqs = hive.act(state["observations"], state["sim_time"])
        actions = [(r.agent_id, r) for r in reqs]

        # Who the hive picked to breed, versus everyone alive. The gap between
        # them IS the selection pressure actually being applied.
        alive = {a.agent_id: a.max_energy for a in env.agents}
        chosen = [alive[r.agent_id] for r in reqs
                  if getattr(r, "spawn_agent", False) and r.agent_id in alive]
        if chosen and len(alive) > 1:
            differentials.append(statistics.mean(chosen) - statistics.mean(alive.values()))

        before = set(alive)
        state = sim.step(actions)
        t = env.time

        if chosen:
            newborn = [a.max_energy for a in env.agents if a.agent_id not in before]
            if newborn:
                mutation_steps.append(statistics.mean(newborn) - statistics.mean(chosen))

        for agent in env.agents:
            if agent.agent_id in seen_ids:
                continue
            seen_ids.add(agent.agent_id)
            births += 1
            for key, base in FOUNDER.items():
                if abs(getattr(agent, key) - base) > 1e-6:
                    ever_mutated[key] += 1

        if abs(t % SAMPLE_EVERY) < 0.05 and env.agents:
            row = {"t": round(t), "n": len(env.agents)}
            for key in FOUNDER:
                vals = [getattr(a, key) for a in env.agents]
                row[key] = (round(min(vals), 1), round(statistics.mean(vals), 1),
                            round(max(vals), 1))
            rows.append(row)

        if state["num_agents"] == 0 or t > horizon:
            break

    return {"seed": seed, "score": round(state["score"], 1), "births": births,
            "ever_mutated": ever_mutated, "rows": rows,
            "selection_differential": round(statistics.mean(differentials), 2)
            if differentials else 0.0,
            "mutation_step": round(statistics.mean(mutation_steps), 2)
            if mutation_steps else 0.0,
            "n_selection_events": len(differentials)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12])
    parser.add_argument("--horizon", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--set", action="append", metavar="KEY=VALUE", default=[])
    args = parser.parse_args()

    params = {}
    for item in args.set:
        key, _, raw = item.partition("=")
        params[key.strip()] = float(raw.strip())
    print(f"overrides: {params or 'none (baseline)'}")

    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_probe_one, (s, args.horizon, params)) for s in args.seeds]
        for future in as_completed(futures):
            reports.append(future.result())
    reports.sort(key=lambda r: r["seed"])

    for r in reports:
        print(f"\n=== seed {r['seed']}  score {r['score']}  births {r['births']} ===")
        print(f"  agents ever differing from the founder value:")
        for key, n in r["ever_mutated"].items():
            print(f"    {key:16s}{n:4d} of {r['births']}  ({n / max(1, r['births']):.1%})")
        print(f"  {'t':>5}{'n':>4}   max_energy min/mean/max      hearing min/mean/max")
        for row in r["rows"]:
            if row["t"] % 300:
                continue
            me = row["max_energy"]
            he = row["hearing_radius"]
            print(f"  {row['t']:5d}{row['n']:4d}   "
                  f"{me[0]:7.0f}{me[1]:7.0f}{me[2]:7.0f}      "
                  f"{he[0]:7.1f}{he[1]:7.1f}{he[2]:7.1f}")

    print("\n=== totals ===")
    total_births = sum(r["births"] for r in reports)
    for key in FOUNDER:
        n = sum(r["ever_mutated"][key] for r in reports)
        finals = [row[key][1] for r in reports for row in r["rows"][-1:]]
        drift = (statistics.mean(finals) - FOUNDER[key]) if finals else 0.0
        print(f"  {key:16s} mutated in {n:4d}/{total_births} births "
              f"({n / max(1, total_births):5.1%})   final mean drift "
              f"{drift:+.1f} from {FOUNDER[key]:.0f}")
    print("\n  Selection can only act on variation that exists. If the mutated share is")
    print("  tiny and the drift is ~0, breeding weights cannot matter in this simulator.")

    diffs = [r["selection_differential"] for r in reports]
    steps = [r["mutation_step"] for r in reports]
    print("\n=== where the steering signal goes ===")
    print(f"  selection differential (chosen parent - population), max_energy: "
          f"{statistics.mean(diffs):+.2f}")
    print(f"  mutation step (child - chosen parent), max_energy             : "
          f"{statistics.mean(steps):+.2f}")
    print(f"  selection events observed                                     : "
          f"{sum(r['n_selection_events'] for r in reports)}")
    print("\n  differential ~0  -> the hive is not steering this trait at all")
    print("  differential >> 0 but no drift -> mutation is drowning the selection")
    print("  drift happens but score flat  -> the trait simply does not pay")


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _here)
    os.chdir(_here)
    main()
