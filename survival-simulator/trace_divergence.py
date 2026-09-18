"""Find the first tick where two identically-seeded runs diverge.

Canonical observation ordering took the spread from 26.1 to 0.3 at horizon 400,
leaving exactly two outcomes -- one binary branch. The environment alone is
deterministic (a no-op policy reproduces to 6 decimals), so the branch is in the
policy. At full horizon it amplifies to 213.6, which is the entire noise floor.

Each child hashes its per-tick decisions; the parent finds the first tick whose
hash differs between runs and dumps both sides.

    python trace_divergence.py --seed 11 --horizon 800 --runs 4
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

from determinism_sorted import _obs_key


def _digest(payload):
    return hashlib.sha1(json.dumps(payload, sort_keys=True,
                                   default=str).encode()).hexdigest()[:16]


def _child(seed, horizon, out_path, dump_tick):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    from src.core import SimulationCore
    from src.utils.controllers.expert_agent_policy import Hive

    sim = SimulationCore(seed=seed)
    hive = Hive()
    hive.reset()
    env = sim.env
    actions = []
    tick = 0
    trace = []
    dump = None
    while True:
        state = sim.step(actions)
        obs = state["observations"]
        for st in obs:
            st["observations"] = sorted(st["observations"], key=_obs_key)
        obs = sorted(obs, key=lambda s: s["agent_id"])

        if dump_tick is not None and tick == dump_tick:
            dump = {"agents": [{"id": s["agent_id"],
                                "x": None, "energy": round(s["energy"], 6),
                                "age": round(s["age"], 6),
                                "obs": [{k: (round(v, 6) if isinstance(v, float) else v)
                                         for k, v in o.items() if k != "coords"}
                                        for o in s["observations"]]}
                               for s in obs],
                    "minds": [{"aid": m.aid, "x": round(m.x, 6), "y": round(m.y, 6),
                               "th": round(m.th, 6), "mode": m.mode,
                               "localized": m.localized,
                               "trees": len(m.world.trees), "fruits": len(m.world.fruits),
                               "world": id(m.world) % 100000}
                              for m in hive.minds.values()]}

        acts = hive.act_dicts(obs, state["sim_time"])
        trace.append(_digest([{k: (round(v, 9) if isinstance(v, float) else v)
                               for k, v in a.items()} for a in acts]))
        actions = [(a["agent_id"], type("R", (), a)) for a in acts]
        actions = []
        for a in acts:
            from src.utils.DTOs import ActionRequest
            actions.append((a["agent_id"], ActionRequest(**a)))
        tick += 1
        if state["num_agents"] == 0 or env.time > horizon:
            break

    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({"score": round(state["score"], 6), "trace": trace, "dump": dump},
                  handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--horizon", type=int, default=800)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--dump-tick", type=int, default=None)
    parser.add_argument("--child", default=None)
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    if args.child:
        _child(args.seed, args.horizon, args.child, args.dump_tick)
        return 0

    runs = []
    for i in range(args.runs):
        out = os.path.join(here, f"_trace_{i}.json")
        cmd = [sys.executable, os.path.abspath(__file__), "--child", out,
               "--seed", str(args.seed), "--horizon", str(args.horizon)]
        if args.dump_tick is not None:
            cmd += ["--dump-tick", str(args.dump_tick)]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=here)
        if res.returncode != 0:
            print("child failed:", res.stderr[-900:])
            return 1
        with open(out, encoding="utf-8") as handle:
            runs.append(json.load(handle))
        print(f"  run {i}: score {runs[-1]['score']}  ticks {len(runs[-1]['trace'])}")

    base = runs[0]["trace"]
    first = None
    for i, other in enumerate(runs[1:], start=1):
        n = min(len(base), len(other["trace"]))
        for k in range(n):
            if base[k] != other["trace"][k]:
                print(f"\nrun 0 vs run {i}: first differing tick = {k}")
                first = k if first is None else min(first, k)
                break
        else:
            print(f"\nrun 0 vs run {i}: identical for all {n} shared ticks")

    if first is not None:
        print(f"\nEARLIEST DIVERGENCE: tick {first}")
        print("re-run with --dump-tick", first, "to inspect both sides")
    if args.dump_tick is not None:
        for i, r in enumerate(runs):
            path = os.path.join(here, f"_dump_{i}.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(r["dump"], handle, indent=1)
        print(f"dumps written for tick {args.dump_tick}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
