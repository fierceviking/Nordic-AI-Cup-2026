"""Search the rule space instead of guessing at it.

    python rule_search.py --generations 6 --population 48

Twenty-one hand-designed rules produced one shipped improvement, so the weakest
link in the loop is human intuition about which rule matters. This replaces that
with successive halving over the policy language in rule_policy.py.

Evaluation discipline, which matters more than the search itself given a per-run
sd of 130-250:

  * COMMON SEEDS within a stage. Every candidate in a stage sees the same seeds,
    so the comparison is paired even though absolute scores are noisy.
  * THE BASELINE RUNS IN EVERY STAGE, on those same seeds. A stage where the
    baseline happens to score badly must not promote candidates for free.
  * STAGE 1 IS A REJECTION FILTER ONLY. At 4 seeds a genuinely +100 policy can
    easily rank below baseline, so its ranking is not evidence; it only removes
    the obviously broken.
  * FRESH SEEDS FOR THE FINAL CHECK, so a winner cannot be one that merely
    exploits quirks of the search seeds.

Every candidate keeps a behavioural fingerprint, because a score with no
mechanism attached just restarts the guessing.
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

STAGE_SEEDS = {
    1: [101, 102, 103, 104],
    2: [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
    3: list(range(101, 131)),
}
FINAL_SEEDS = list(range(201, 231))          # never used during the search


def _evaluate(job):
    program, seed, horizon = job
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    from src.core import SimulationCore
    from src.utils.controllers.expert_agent_policy import Hive
    from rule_policy import RuleHive

    sim = SimulationCore(seed=seed)
    hive = RuleHive(program=program) if program else Hive()
    hive.reset()
    env = sim.env

    deaths = {"eaten": 0, "starved": 0}
    original_kill = env.kill_agent

    def patched_kill(agent, _env=env, _orig=original_kill):
        if agent in _env.agents:
            hit = any((p.x - agent.x) ** 2 + (p.y - agent.y) ** 2
                      < (p.size + agent.size + 2) ** 2 for p in _env.predators)
            deaths["eaten" if hit else "starved"] += 1
        _orig(agent)

    env.kill_agent = patched_kill

    distance = turning = sprint_ticks = 0.0
    modes = {}
    births = 0
    seen = set()
    # Predator cost dominates (+625) but the sprint gate does not explain it, so
    # separate the two ways a policy can lower it: meeting predators less often
    # (encounters) versus surviving the meetings it has (eaten / encounters).
    gated_exposure = 0
    encounters = 0
    encounters_gated = 0
    in_encounter = set()
    encounter_radius = 210.0

    actions = []
    while True:
        state = sim.step(actions)
        still_near = set()
        for agent in env.agents:
            if agent.agent_id not in seen:
                seen.add(agent.agent_id)
                births += 1
            gated = agent.energy < agent.max_energy / 5.0
            near = any(math.hypot(p.x - agent.x, p.y - agent.y) < encounter_radius
                       for p in env.predators)
            if near:
                still_near.add(agent.agent_id)
                if agent.agent_id not in in_encounter:
                    encounters += 1
                    if gated:
                        encounters_gated += 1
                if gated:
                    gated_exposure += 1
        in_encounter = still_near
        reqs = hive.act(state["observations"], state["sim_time"])
        for r in reqs:
            distance += r.move_distance
            turning += abs(r.turn_angle)
            agent = env.agents_dict.get(r.agent_id)
            if agent is not None and r.move_distance > agent.speed:
                sprint_ticks += 1
        for mind in hive.minds.values():
            modes[mind.mode] = modes.get(mind.mode, 0) + 1
        actions = [(r.agent_id, r) for r in reqs]
        if state["num_agents"] == 0 or env.time > horizon:
            break

    total_modes = sum(modes.values()) or 1
    return {
        "seed": seed,
        "score": round(state["score"], 1),
        "eaten": deaths["eaten"],
        "starved": deaths["starved"],
        "births": births,
        "distance": round(distance),
        "turning": round(turning),
        "sprint_ticks": int(sprint_ticks),
        "gated_exposure": gated_exposure,
        "encounters": encounters,
        "encounters_gated": encounters_gated,
        "lethality": round(deaths["eaten"] / encounters, 4) if encounters else 0.0,
        "modes": {k: round(100.0 * v / total_modes, 1) for k, v in modes.items()},
    }


def _score_stage(pool, programs, seeds, horizon):
    """Evaluate every candidate on every seed as one batch.

    Submitting per-candidate leaves workers idle whenever the seed count is
    below the worker count, which is most of stage 1.
    """
    futures = {}
    for idx, program in enumerate(programs):
        for seed in seeds:
            futures[pool.submit(_evaluate, (program, seed, horizon))] = idx
    runs = [[] for _ in programs]
    for future in as_completed(futures):
        runs[futures[future]].append(future.result())
    return runs


def _summarise(runs):
    n = max(1, len(runs))
    return {
        "mean": round(statistics.mean(r["score"] for r in runs), 1),
        "eaten": sum(r["eaten"] for r in runs) / n,
        "starved": sum(r["starved"] for r in runs) / n,
        "births": sum(r["births"] for r in runs) / n,
        "distance": round(sum(r["distance"] for r in runs) / n),
        "turning": round(sum(r["turning"] for r in runs) / n),
        "sprint": round(sum(r["sprint_ticks"] for r in runs) / n),
        "gated": round(sum(r["gated_exposure"] for r in runs) / n),
        "encounters": round(sum(r["encounters"] for r in runs) / n, 1),
        # Pooled, not a mean of per-run ratios: runs that end early otherwise
        # carry the same weight as runs with hundreds of encounters.
        "lethality": round(sum(r["eaten"] for r in runs)
                           / max(1, sum(r["encounters"] for r in runs)), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="rule_search.json")
    args = parser.parse_args()

    from rule_policy import random_program, mutate

    rng = random.Random(args.seed)
    # Score is 97-98% survival time and only pop==0 ends a run, so this is a
    # ruin problem: a death is nearly free at pop 6 and fatal at pop 1. Seeds
    # therefore cover (a) the ablation hierarchy -- predators +625, movement
    # +255 -- and (b) the ruin structure: risk posture conditioned on how thin
    # the lineage is, and on how far the world has decayed.
    seeded = [
        [("predator_within", 200.0, "flee"), ("always", 0.0, "pursue_fruit")],
        [("cannot_sprint", 0.0, "retreat_to_tree"), ("always", 0.0, "pursue_fruit")],
        [("cannot_sprint", 0.0, "flee"), ("predator_within", 180.0, "flee"),
         ("always", 0.0, "pursue_fruit")],
        [("energy_below", 0.3, "retreat_to_tree"), ("predator_within", 200.0, "flee"),
         ("always", 0.0, "pursue_fruit")],
        [("runway_below", 15.0, "pursue_fruit"), ("predator_within", 220.0, "flee"),
         ("always", 0.0, "pursue_fruit")],
        [("energy_below", 0.25, "hold"), ("always", 0.0, "pursue_fruit")],
        # --- ruin-shaped: last-survivor caution ---
        [("pop_below", 2.0, "flee"), ("always", 0.0, "pursue_fruit")],
        [("pop_below", 3.0, "flee"), ("predator_within", 200.0, "flee"),
         ("always", 0.0, "pursue_fruit")],
        [("pop_below", 2.0, "retreat_to_tree"), ("always", 0.0, "pursue_fruit")],
        [("pop_below", 3.0, "hold"), ("always", 0.0, "pursue_fruit")],
        # thin lineage AND exposed: the only state where ruin is imminent
        [("pop_below", 3.0, "flee"), ("predator_within", 300.0, "flee"),
         ("energy_below", 0.3, "retreat_to_tree"), ("always", 0.0, "pursue_fruit")],
        # --- ruin-shaped: phase change as the world decays ---
        [("time_after", 900.0, "retreat_to_tree"), ("always", 0.0, "pursue_fruit")],
        [("time_after", 600.0, "hold"), ("predator_within", 200.0, "flee"),
         ("always", 0.0, "pursue_fruit")],
        [("time_after", 1200.0, "flee"), ("always", 0.0, "pursue_fruit")],
        [("time_after", 900.0, "flee"), ("pop_below", 3.0, "flee"),
         ("always", 0.0, "pursue_fruit")],
    ]
    population = seeded + [random_program(rng) for _ in range(args.population - len(seeded))]

    history = []
    best = None
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for gen in range(args.generations):
            survivors = population
            for stage in (1, 2, 3):
                seeds = STAGE_SEEDS[stage]
                # Baseline is evaluated on the SAME seeds, in the same batch,
                # every stage: a stage where the seeds happen to be hostile
                # must not promote candidates for free.
                batch = _score_stage(pool, [None] + list(survivors), seeds, args.horizon)
                base = _summarise(batch[0])
                scored = [(_summarise(runs), prog)
                          for runs, prog in zip(batch[1:], survivors)]
                scored.sort(key=lambda kv: -kv[0]["mean"])
                keep = max(1, len(scored) // (4 if stage == 1 else 2))
                print(f"gen {gen} stage {stage}: baseline {base['mean']:.1f} | "
                      f"best {scored[0][0]['mean']:.1f} | keeping {keep}/{len(scored)}",
                      flush=True)
                survivors = [p for _, p in scored[:keep]]
                if stage == 3:
                    top, prog = scored[0]
                    history.append({"gen": gen, "baseline": base, "best": top,
                                    "program": prog})
                    if best is None or top["mean"] > best[0]["mean"]:
                        best = (top, prog)
                    print(f"  fingerprint vs baseline: "
                          f"dist {top['distance']} vs {base['distance']} | "
                          f"sprint {top['sprint']} vs {base['sprint']} | "
                          f"gated {top['gated']} vs {base['gated']} | "
                          f"eaten {top['eaten']:.0f} vs {base['eaten']:.0f} | "
                          f"starved {top['starved']:.0f} vs {base['starved']:.0f}",
                          flush=True)
                    print(f"  encounters {top['encounters']} vs {base['encounters']} | "
                          f"lethality {top['lethality']:.3f} vs {base['lethality']:.3f} "
                          f"(avoidance vs response geometry)", flush=True)
                    print(f"  program: {prog}", flush=True)

            children = []
            while len(children) < args.population:
                parent = rng.choice(survivors)
                children.append(mutate(parent, rng))
            population = survivors + children[:args.population - len(survivors)]

            with open(args.out, "w", encoding="utf-8") as handle:
                json.dump(history, handle, indent=2)

        if best is not None:
            print("\n=== FINAL CHECK on unseen seeds ===", flush=True)
            fresh = _score_stage(pool, [None, best[1]], FINAL_SEEDS, args.horizon)
            fresh_base, fresh_best = _summarise(fresh[0]), _summarise(fresh[1])
            print(f"  baseline  {fresh_base['mean']:.1f}")
            print(f"  candidate {fresh_best['mean']:.1f}")
            print(f"  delta     {fresh_best['mean'] - fresh_base['mean']:+.1f}")
            print(f"  encounters {fresh_best['encounters']} vs {fresh_base['encounters']}")
            print(f"  lethality  {fresh_best['lethality']:.3f} vs {fresh_base['lethality']:.3f}")
            print(f"  program   {best[1]}")
            history.append({"final": {"baseline": fresh_base, "candidate": fresh_best,
                                      "program": best[1]}})
            with open(args.out, "w", encoding="utf-8") as handle:
                json.dump(history, handle, indent=2)


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _here)
    os.chdir(_here)
    sys.exit(main())
