"""How much of the score is luck rather than policy?

`det_env` can install any of several equally-valid orderings of the environment's
object sets. Physics is identical under all of them; only the arbitrary
tie-breaking differs. So running ONE policy on ONE seed across many salts
isolates pure luck.

This bounds every policy experiment we can run. If swapping an arbitrary
tie-break moves a seed by hundreds of points, then a policy change measured on
few runs is mostly measuring that, and -- since the competition scores the mean
of three runs -- so is the leaderboard.

    python luck_probe.py --seeds 11 12 ... --salts 6 --horizon 3000
"""

import argparse
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor


def _run(job):
    seed, salt, horizon = job
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import det_env
    det_env.install(salt=salt)
    from src.core import SimulationCore
    from src.utils.controllers.expert_agent_policy import Hive

    sim = SimulationCore(seed=seed)
    hive = Hive()
    hive.reset()
    env = sim.env
    actions = []
    while True:
        state = sim.step(actions)
        reqs = hive.act(state["observations"], state["sim_time"])
        actions = [(r.agent_id, r) for r in reqs]
        if state["num_agents"] == 0 or env.time > horizon:
            break
    return seed, salt, round(state["score"], 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(11, 21)))
    parser.add_argument("--salts", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    jobs = [(s, k, args.horizon) for s in args.seeds for k in range(args.salts)]
    table = {s: {} for s in args.seeds}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for seed, salt, score in pool.map(_run, jobs):
            table[seed][salt] = score

    print(f"\nONE policy (Hive), {len(args.seeds)} seeds x {args.salts} tie-break orderings")
    print(f"{'seed':>5}" + "".join(f"{'salt' + str(k):>10}" for k in range(args.salts))
          + f"{'mean':>10}{'sd':>9}{'range':>9}")
    within = []
    for seed in args.seeds:
        values = [table[seed][k] for k in range(args.salts)]
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        within.append(sd)
        print(f"{seed:>5}" + "".join(f"{v:>10.1f}" for v in values)
              + f"{statistics.mean(values):>10.1f}{sd:>9.1f}{max(values) - min(values):>9.1f}")

    seed_means = [statistics.mean(table[s][k] for k in range(args.salts)) for s in args.seeds]
    all_scores = [table[s][k] for s in args.seeds for k in range(args.salts)]
    pooled_within = (sum(v * v for v in within) / len(within)) ** 0.5
    between = statistics.stdev(seed_means) if len(seed_means) > 1 else 0.0

    print(f"\n  pooled WITHIN-seed sd (pure tie-break luck) : {pooled_within:7.1f}")
    print(f"  BETWEEN-seed sd (map/scenario effect)       : {between:7.1f}")
    print(f"  total sd over all runs                      : {statistics.stdev(all_scores):7.1f}")
    share = pooled_within ** 2 / (pooled_within ** 2 + between ** 2) * 100 if (pooled_within or between) else 0
    print(f"  => tie-break luck explains ~{share:.0f}% of score variance")
    print(f"\n  se of a 3-run evaluation mean, luck only    : {pooled_within / 3 ** 0.5:7.1f}")
    print("  (a policy gain smaller than ~2x that is undetectable by the competition itself)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
