"""Is the simulator reproducible on this machine?

    python determinism_test.py --seed 11 --horizon 400 --runs 3

On Windows the same seed gave 1147.5 and 676.8, which forced every comparison
to n>=30 with a ~+-100 noise floor. The organisers state the simulation is
deterministic on a given OS, which would make that a Windows artefact and allow
paired comparisons instead.

Runs the identical policy on the identical seed several times IN SEPARATE
PROCESSES, because the suspected cause is `_get_local_objects` iterating sets
of entities hashed by id() -- addresses differ per process, so in-process
repeats would not test anything.
"""

import argparse
import json
import os
import subprocess
import sys


def _one(seed, horizon):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    from src.core import SimulationCore
    from src.utils.controllers.expert_agent_policy import Hive

    sim = SimulationCore(seed=seed)
    hive = Hive()
    hive.reset()
    actions = []
    while True:
        state = sim.step(actions)
        actions = [(r.agent_id, r) for r in hive.act(state["observations"], state["sim_time"])]
        if state["num_agents"] == 0 or sim.env.time > horizon:
            break
    return {"score": round(state["score"], 6), "time": round(sim.env.time, 3),
            "agents": state["num_agents"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        print(json.dumps(_one(args.seed, args.horizon)), flush=True)
        return

    results = []
    for i in range(args.runs):
        out = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--child",
             "--seed", str(args.seed), "--horizon", str(args.horizon)],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        line = [l for l in out.stdout.splitlines() if l.startswith("{")]
        if not line:
            print("child failed:", out.stderr[-500:])
            return 1
        results.append(json.loads(line[-1]))
        print(f"  run {i + 1}: {results[-1]}")

    scores = {r["score"] for r in results}
    print(f"\nseed {args.seed}, {args.runs} separate processes")
    if len(scores) == 1:
        print(f"  DETERMINISTIC - every run scored {results[0]['score']}")
        print("  => paired comparisons are valid here; n>=30 is not required")
    else:
        lo, hi = min(scores), max(scores)
        print(f"  NOT deterministic - scores {sorted(scores)}")
        print(f"  spread {hi - lo:.1f} ({(hi - lo) / max(1e-9, lo):.1%} of the lowest)")
        print("  => compare distributions, never paired runs")
    return 0


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _here)
    os.chdir(_here)
    sys.exit(main())
