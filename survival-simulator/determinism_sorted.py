"""Does canonicalising observation ORDER make the run reproducible?

The sim is not deterministic across processes: `_get_local_objects` builds each
agent's observation list by iterating `set()`s whose elements hash by `id()`, so
memory layout decides the order. `oracle_probe` reproduces byte-for-byte only
because the oracle ignores observations and reads `env` directly.

The Hive does not ignore them: `_perceive` feeds them to `World.add_point`,
which MERGES sightings onto existing keys, so insertion order decides the stored
key and coordinates. If that is the dominant channel, sorting each observation
list into a canonical order should collapse the run-to-run spread.

Worth more than any policy tweak: the 130-250 per-run sd is what has made ~20
experiments unresolvable. Removing it changes the cost of every later test.

    python determinism_sorted.py --seed 11 --horizon 400 --runs 3 --sort
"""

import argparse
import json
import os
import subprocess
import sys


def _obs_key(o):
    # Full precision and every distinguishing field: rounding here would let
    # near-equal sightings tie and fall back on arrival order, which is the
    # very thing being canonicalised.
    ty = str(o.get("type", ""))
    if "coords" in o:
        (sx, sy), (ex, ey) = o["coords"]
        return (ty, 1, float(sx), float(sy), float(ex), float(ey), 0.0, -1)
    return (ty, 0, float(o.get("distance", 0.0)), float(o.get("angle", 0.0)),
            float(o.get("rel_dir", 0.0)), float(o.get("radius", 0.0)),
            float(o.get("energy", 0.0)), int(o.get("id", -1)))


def _child(seed, horizon, sort):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    if os.environ.get("DET_ENV") == "1":
        import det_env
        det_env.install()
    from src.core import SimulationCore
    from src.utils.controllers.expert_agent_policy import Hive

    sim = SimulationCore(seed=seed)
    hive = Hive()
    hive.reset()
    env = sim.env
    actions = []
    while True:
        state = sim.step(actions)
        obs = state["observations"]
        if sort:
            for st in obs:
                st["observations"] = sorted(st["observations"], key=_obs_key)
            obs = sorted(obs, key=lambda s: s["agent_id"])
        reqs = hive.act(obs, state["sim_time"])
        actions = [(r.agent_id, r) for r in reqs]
        if state["num_agents"] == 0 or env.time > horizon:
            break
    print(json.dumps({"score": round(state["score"], 6),
                      "time": round(env.time, 3),
                      "agents": state["num_agents"]}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--sort", action="store_true")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()

    if args.child:
        _child(args.seed, args.horizon, args.sort)
        return 0

    results = []
    for i in range(args.runs):
        cmd = [sys.executable, os.path.abspath(__file__), "--child",
               "--seed", str(args.seed), "--horizon", str(args.horizon)]
        if args.sort:
            cmd.append("--sort")
        out = subprocess.run(cmd, capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.abspath(__file__)))
        if out.returncode != 0:
            print("child failed:", out.stderr[-600:])
            return 1
        rec = json.loads(out.stdout.strip().splitlines()[-1])
        results.append(rec)
        print(f"  run {i + 1}: {rec}")

    scores = sorted(r["score"] for r in results)
    label = "SORTED observations" if args.sort else "raw observations"
    print(f"\nseed {args.seed}, {args.runs} separate processes, {label}")
    if len(set(scores)) == 1:
        print(f"  DETERMINISTIC - every run scored {scores[0]}")
    else:
        spread = scores[-1] - scores[0]
        print(f"  NOT deterministic - {scores}")
        print(f"  spread {spread:.1f} ({100.0 * spread / max(1e-9, scores[0]):.1f}% of lowest)")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
