"""Where does turn energy actually go, and would a dead-band recover any of it?

Turn cost is `min(pi, |a|) / (2*pi)` per tick and, unlike movement (which returns
~1.9 energy per unit spent) or spawning (which buys reproduction), it returns
nothing. The budget is income 8.27 vs spend 9.03 per agent-second, and R0
0.93 -> 0.959 needs about +0.32 -- the same order as the whole 0.40 turn line.

Measure before building: this reports the turn spend by mode, the magnitude
histogram, and what a dead-band would ACTUALLY save, counting only the turns it
would suppress. A dead-band also leaves heading error uncorrected, so the saving
is an upper bound on the benefit, not a prediction.

    python turn_probe.py --seeds 11 12 13 --salts 2
"""

import argparse
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

TWO_PI = 2.0 * math.pi
BANDS = (0.03, 0.08, 0.15, 0.25, 0.40)


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

    by_mode = defaultdict(float)
    ticks_by_mode = defaultdict(int)
    saved = {b: 0.0 for b in BANDS}
    total_turn = 0.0
    total_move = 0.0
    agent_ticks = 0
    actions = []
    while True:
        state = sim.step(actions)
        reqs = hive.act(state["observations"], state["sim_time"])
        for r in reqs:
            mind = hive.minds.get(r.agent_id)
            mode = mind.mode if mind else "?"
            a = abs(float(r.turn_angle))
            cost = min(math.pi, a) / TWO_PI
            by_mode[mode] += cost
            ticks_by_mode[mode] += 1
            total_turn += cost
            total_move += float(r.move_distance) * 0.05
            agent_ticks += 1
            for b in BANDS:
                if a < b:
                    saved[b] += cost
        actions = [(r.agent_id, r) for r in reqs]
        if state["num_agents"] == 0 or env.time > horizon:
            break
    return (seed, salt, state["score"], env.time, total_turn, total_move,
            agent_ticks, dict(by_mode), dict(ticks_by_mode), saved)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    parser.add_argument("--salts", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    jobs = [(s, k, args.horizon) for s in args.seeds for k in range(args.salts)]
    agg_mode = defaultdict(float)
    agg_ticks = defaultdict(int)
    agg_saved = {b: 0.0 for b in BANDS}
    turn = move = 0.0
    ticks = 0
    sim_time = 0.0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for (_s, _k, _score, t, tt, tm, at, bm, btk, sv) in pool.map(_run, jobs):
            turn += tt
            move += tm
            ticks += at
            sim_time += t
            for k, v in bm.items():
                agg_mode[k] += v
            for k, v in btk.items():
                agg_ticks[k] += v
            for b in BANDS:
                agg_saved[b] += sv[b]

    per_sec = turn / sim_time if sim_time else 0.0
    print(f"\n{len(jobs)} runs, {sim_time:.0f} agent-run seconds, {ticks} agent-ticks")
    print(f"  total turn energy : {turn:10.0f}   ({per_sec:.2f} per agent-second)")
    print(f"  total move energy : {move:10.0f}   ({move / sim_time:.2f} per agent-second)")
    print(f"  turn as share of move: {100.0 * turn / max(move, 1e-9):.1f}%")

    print("\n  turn energy by mode:")
    for mode in sorted(agg_mode, key=lambda m: -agg_mode[m]):
        share = 100.0 * agg_mode[mode] / max(turn, 1e-9)
        per_tick = agg_mode[mode] / max(agg_ticks[mode], 1)
        print(f"    {mode:<10}{agg_mode[mode]:>10.0f}{share:>7.1f}%"
              f"   {agg_ticks[mode]:>9} ticks   {per_tick:.4f}/tick")

    print("\n  energy a dead-band would suppress (UPPER bound - ignores the")
    print("  wasted movement from leaving heading error uncorrected):")
    for b in BANDS:
        share = 100.0 * agg_saved[b] / max(turn, 1e-9)
        print(f"    |turn| < {b:<5}{agg_saved[b]:>10.0f}{share:>7.1f}% of turn spend"
              f"   = {agg_saved[b] / sim_time:.3f} per agent-second")
    print("\n  target for R0 0.93 -> 0.959 is +0.32 per agent-second")
    return 0


if __name__ == "__main__":
    sys.exit(main())
