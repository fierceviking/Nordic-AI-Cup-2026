"""Paired A/B comparison on a deterministic environment.

With `det_env` installed a (policy, seed) pair always yields the same score, so
variant and baseline can be differenced SEED BY SEED. The seed effect -- which
is most of the variance, per-run sd 130-250 -- cancels exactly, and what remains
is the policy effect.

Unpatched, an effect needs roughly +150 to clear the noise at n=10. Paired on a
deterministic env, the resolution is set by the spread of the per-seed
DIFFERENCES instead, which is far smaller.

    python det_compare.py --variants variants.json --seeds 11 12 ... --horizon 3000

variants.json maps a name to Hive attribute overrides:
    {"baseline": {}, "pop3": {"POP_MIN": 3, "POP_MAX": 3}}

CAVEAT: the evaluator runs the UNPATCHED environment. A win here is a hypothesis
about the real one, not proof; confirm anything promising unpatched before
shipping it.
"""

import argparse
import json
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor


def _run(job):
    name, overrides, seed, salt, horizon = job
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import det_env
    det_env.install(salt=salt)
    from src.core import SimulationCore
    from src.utils.controllers.expert_agent_policy import Hive

    hive = Hive()
    for key, value in (overrides or {}).items():
        if not hasattr(hive, key):
            raise AttributeError(f"{name}: Hive has no setting {key}")
        setattr(hive, key, value)
    hive.reset()

    sim = SimulationCore(seed=seed)
    env = sim.env
    deaths = {"eaten": 0, "starved": 0}
    original = env.kill_agent

    def patched(agent, _e=env, _o=original):
        if agent in _e.agents:
            hit = any((p.x - agent.x) ** 2 + (p.y - agent.y) ** 2
                      < (p.size + agent.size + 2) ** 2 for p in _e.predators)
            deaths["eaten" if hit else "starved"] += 1
        _o(agent)

    env.kill_agent = patched

    actions = []
    while True:
        state = sim.step(actions)
        reqs = hive.act(state["observations"], state["sim_time"])
        actions = [(r.agent_id, r) for r in reqs]
        if state["num_agents"] == 0 or env.time > horizon:
            break
    return name, (seed, salt), round(state["score"], 6), deaths["eaten"], deaths["starved"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", required=True)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(range(11, 21)))
    parser.add_argument("--horizon", type=int, default=3000)
    parser.add_argument("--salts", type=int, default=3)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--reference", default="baseline")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(args.variants, encoding="utf-8") as handle:
        variants = json.load(handle)

    # Each (seed, salt) is an independent, replayable draw. Tie-break luck is
    # 77% of variance, so salts are the cheapest way to add real samples.
    cells = [(seed, salt) for seed in args.seeds for salt in range(args.salts)]
    jobs = [(name, overrides, seed, salt, args.horizon)
            for name, overrides in variants.items() for seed, salt in cells]
    scores = {name: {} for name in variants}
    deaths = {name: [0, 0] for name in variants}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, cell, score, eaten, starved in pool.map(_run, jobs):
            scores[name][cell] = score
            deaths[name][0] += eaten
            deaths[name][1] += starved

    ref = args.reference if args.reference in scores else next(iter(scores))
    print(f"\nreference: {ref}   seeds: {len(args.seeds)} x {args.salts} salts "
          f"= {len(cells)} runs/variant   horizon: {args.horizon}")
    print(f"{'variant':<22}{'mean':>9}{'paired d':>11}{'sd(d)':>9}"
          f"{'se(d)':>8}{'wins':>7}{'eaten':>8}{'starv':>8}")
    for name in scores:
        values = [scores[name][c] for c in cells]
        mean = statistics.mean(values)
        diffs = [scores[name][c] - scores[ref][c] for c in cells]
        dbar = statistics.mean(diffs)
        sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
        se = sd / (len(diffs) ** 0.5) if diffs else 0.0
        wins = sum(1 for d in diffs if d > 0)
        print(f"{name:<22}{mean:>9.1f}{dbar:>11.1f}{sd:>9.1f}{se:>8.1f}"
              f"{wins:>4}/{len(diffs):<2}{deaths[name][0]:>8}{deaths[name][1]:>8}")

    print("\n95% CI on the paired difference (t-ish, 1.96*se):")
    for name in scores:
        if name == ref:
            continue
        diffs = [scores[name][c] - scores[ref][c] for c in cells]
        dbar = statistics.mean(diffs)
        se = (statistics.stdev(diffs) / (len(diffs) ** 0.5)) if len(diffs) > 1 else 0.0
        lo, hi = dbar - 1.96 * se, dbar + 1.96 * se
        verdict = "RESOLVED" if lo > 0 or hi < 0 else "not resolvable"
        print(f"  {name:<22}{dbar:>9.1f}  [{lo:>8.1f},{hi:>8.1f}]  {verdict}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({n: {f"{s}_{k}": v for (s, k), v in d.items()}
                       for n, d in scores.items()}, handle, indent=1)
        print(f"\nraw scores -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
