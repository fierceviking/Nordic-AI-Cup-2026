"""Optuna search over the Hive tunables.

Runs are expensive (~350 s for a full 3000 s survival) but cheap when the hive
dies early, so the search naturally spends its budget on good configurations.
Each trial evaluates a fixed seed set (common random numbers) in parallel.

    python tune.py search --trials 60
    python tune.py validate --params best_params.json
    python tune.py compare --workers 4 --seeds 21 22 23 24 25 26 27 28 29 30
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

SEARCH_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)
VALIDATE_SEEDS = (11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
MAX_TIME = 3000
# Map generation allocates a ~150 MB numpy temporary per run, so 12 workers
# exhausted RAM part-way through a search.
WORKERS = 7
BEST_PATH = "best_params.json"


def _run_one(job):
    """Child process: one full simulation, stdout muted."""
    seed, params, max_time = job
    from local_playground import local_simulation
    with contextlib.redirect_stdout(io.StringIO()):
        res = local_simulation(verbose=False, seed=seed, max_time=max_time,
                               log_every=0, diagnostics=False, **params)
    return seed, res["score"], res["time"]


def evaluate(pool, params, seeds, max_time=MAX_TIME):
    jobs = [(s, params, max_time) for s in seeds]
    return list(pool.map(_run_one, jobs))


def suggest(trial):
    """The whole live tunable surface, including the rules added today.

    Single-parameter sweeps could not resolve anything below about +-120 points,
    so the premise here is that several sub-threshold effects only show up when
    searched together.  Booleans are searched too: each one gates a rule that
    measured neutral on its own.
    """
    params = {
        # population and reproduction
        "SPAWN_MIN": trial.suggest_float("SPAWN_MIN", 150.0, 430.0, step=10.0),
        "POP_MIN": trial.suggest_int("POP_MIN", 2, 8),
        "POP_MAX": trial.suggest_int("POP_MAX", 5, 16),
        "POP_HARD": trial.suggest_int("POP_HARD", 12, 26),
        "NET_GROW": trial.suggest_float("NET_GROW", 0.0, 2.0),
        "NET_SHRINK": trial.suggest_float("NET_SHRINK", -1.5, 0.3),
        "SPAWN_COOLDOWN": trial.suggest_float("SPAWN_COOLDOWN", 1.0, 15.0),
        "AGING_COOLDOWN": trial.suggest_float("AGING_COOLDOWN", 0.1, 8.0),
        "AGING_DUMP": trial.suggest_categorical("AGING_DUMP", [False, True]),
        # foraging
        "MOVE_ON": trial.suggest_float("MOVE_ON", 10.0, 60.0),
        "FRUIT_RANGE": trial.suggest_float("FRUIT_RANGE", 150.0, 600.0),
        "FRUIT_VALUE": trial.suggest_float("FRUIT_VALUE", 20.0, 60.0),
        "FRUIT_MIN_VALUE": trial.suggest_float("FRUIT_MIN_VALUE", 0.0, 20.0),
        "FRUIT_HYSTERESIS": trial.suggest_float("FRUIT_HYSTERESIS", 0.0, 15.0),
        "TREE_YIELD": trial.suggest_float("TREE_YIELD", 0.5, 5.0),
        "YIELD_CAP": trial.suggest_float("YIELD_CAP", 10.0, 45.0),
        "PATROL_RANGE": trial.suggest_float("PATROL_RANGE", 150.0, 500.0),
        "TREE_HYSTERESIS": trial.suggest_float("TREE_HYSTERESIS", 0.0, 15.0),
        "TREE_CLAIMED": trial.suggest_float("TREE_CLAIMED", 0.0, 90.0),
        # risk
        "FRUIT_DANGER": trial.suggest_float("FRUIT_DANGER", 0.0, 120.0),
        "TREE_DANGER": trial.suggest_float("TREE_DANGER", 0.0, 160.0),
        "RISK_AVERSION": trial.suggest_float("RISK_AVERSION", 0.0, 3.0),
        "RISK_ENERGY": trial.suggest_float("RISK_ENERGY", 120.0, 450.0),
        "SCAN_RATE": trial.suggest_float("SCAN_RATE", 0.1, 1.2),
        "NOSPRINT_MARGIN": trial.suggest_float("NOSPRINT_MARGIN", 1.0, 3.0),
        # rules added today, each neutral in isolation
        "OVERFLOW_SPAWN": trial.suggest_categorical("OVERFLOW_SPAWN", [False, True]),
        "BARREN_PATIENCE": trial.suggest_float("BARREN_PATIENCE", 0.0, 60.0),
        "EARLY_BOOST": trial.suggest_categorical("EARLY_BOOST", [False, True]),
        "UNSTICK": trial.suggest_categorical("UNSTICK", [False, True]),
    }
    if params["OVERFLOW_SPAWN"]:
        params["OVERFLOW_HEADROOM"] = trial.suggest_float("OVERFLOW_HEADROOM", 20.0, 200.0)
        params["OVERFLOW_COOLDOWN"] = trial.suggest_float("OVERFLOW_COOLDOWN", 0.5, 8.0)
    if params["EARLY_BOOST"]:
        params["EARLY_UNTIL"] = trial.suggest_float("EARLY_UNTIL", 50.0, 400.0)
        params["EARLY_POP"] = trial.suggest_int("EARLY_POP", 6, 20)
        params["EARLY_COOLDOWN"] = trial.suggest_float("EARLY_COOLDOWN", 0.5, 6.0)
    if params["UNSTICK"]:
        params["STALL_TICKS"] = trial.suggest_int("STALL_TICKS", 3, 20)
        params["AVOID_TIME"] = trial.suggest_float("AVOID_TIME", 5.0, 30.0)
    if params["POP_MAX"] < params["POP_MIN"]:
        params["POP_MAX"] = params["POP_MIN"]
    return params


def search(n_trials, n_jobs, study_name, storage, horizon):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    pool = ProcessPoolExecutor(max_workers=WORKERS)
    t_start = time.time()
    half = max(1, len(SEARCH_SEEDS) // 2)

    def objective(trial):
        params = suggest(trial)
        # Every trial runs the same seeds (common random numbers) so candidates
        # are compared on identical maps; map variance dwarfs parameter effects.
        scores = []
        for step, chunk in enumerate((SEARCH_SEEDS[:half], SEARCH_SEEDS[half:])):
            scores += [s for _, s, _ in evaluate(pool, params, chunk, horizon)]
            running = sum(scores) / len(scores)
            trial.report(running, step)
            # Half the seeds is usually enough to recognise a hopeless trial.
            if trial.should_prune():
                line = (f'[{time.time()-t_start:6.0f}s] trial {trial.number:3d} '
                        f'pruned at {len(scores)} seeds (mean {running:7.1f})')
                print(line, flush=True)
                with open("tune_progress.log", "a") as fh:
                    fh.write(line + "\n")
                raise optuna.TrialPruned()
        mean = sum(scores) / len(scores)
        line = (f'[{time.time()-t_start:6.0f}s] trial {trial.number:3d} '
                f'mean={mean:7.1f} scores={[round(s) for s in scores]}')
        print(line, flush=True)
        with open("tune_progress.log", "a") as fh:
            fh.write(line + "\n")
        return mean

    study = optuna.create_study(direction="maximize", study_name=study_name,
                                storage=storage, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=7,
                                                                   n_startup_trials=12),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8,
                                                                   n_warmup_steps=0))
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    pool.shutdown()

    print("\n=== best ===")
    print(f"value {study.best_value:.1f}")
    for k, v in sorted(study.best_params.items()):
        print(f"  {k} = {v}")
    with open(BEST_PATH, "w") as fh:
        json.dump(study.best_params, fh, indent=2, sort_keys=True)
    print(f"written to {BEST_PATH}")
    print("\nThis is the mean on the SEARCH seeds. Confirm on held-out seeds before\n"
          "believing it -- the search will happily fit noise:\n"
          f"  python tune.py validate --params {BEST_PATH}")

    print("\n=== top 10 ===")
    done = [t for t in study.trials if t.value is not None]
    for t in sorted(done, key=lambda t: -t.value)[:10]:
        print(f'{t.value:8.1f}  {t.params}')
    return study


def validate(params, seeds, label, horizon=MAX_TIME):
    pool = ProcessPoolExecutor(max_workers=WORKERS)
    results = evaluate(pool, params, seeds, horizon)
    pool.shutdown()
    scores = [s for _, s, _ in results]
    for seed, score, tm in sorted(results):
        print(f'  seed={seed:3d} score={score:8.1f} time={tm:7.1f}'
              f'{"  SURVIVED" if tm > MAX_TIME - 1 else ""}')
    mean = sum(scores) / len(scores)
    print(f'{label}: mean {mean:.1f} over {len(scores)} seeds '
          f'(min {min(scores):.1f}, max {max(scores):.1f}, '
          f'survived {sum(1 for _, _, t in results if t > MAX_TIME - 1)}/{len(scores)})')
    return mean


def _compare_one(job):
    label, seed, params, horizon, replicate = job
    from local_playground import local_simulation

    params = dict(params)
    controller = params.pop("__controller__", "hive")
    if controller == "burst":
        from src.utils.controllers.burst_agent_policy import BurstHive as Policy
    else:
        from src.utils.controllers.expert_agent_policy import Hive as Policy

    class MeasuredHive(Policy):
        def act_dicts(self, agent_states, sim_time):
            started = time.perf_counter()
            actions = super().act_dicts(agent_states, sim_time)
            timings.append(time.perf_counter() - started)
            return actions

    timings = []
    hive = MeasuredHive(**params)
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        result = local_simulation(verbose=False, seed=seed, hive=hive,
                                  max_time=horizon, log_every=0, diagnostics=True)
    timings.sort()
    deaths = result["stats"]
    return {"variant": label, "seed": seed, "replicate": replicate,
            "score": result["score"],
            "time": result["time"], "alive": result["agents"],
            "eaten": deaths["eaten"], "starved": deaths["starved"],
            "young_deaths": sum(death[1] < 25 for death in getattr(hive, "deaths", [])),
            "births": getattr(hive, "budget", {}).get("spawn", 0.0) / 100.0,
            "policy_mean_ms": statistics.mean(timings) * 1000.0,
            "policy_p95_ms": timings[min(len(timings) - 1, int(len(timings) * 0.95))] * 1000.0,
            "wall": time.perf_counter() - started}


def compare(seeds, horizon, workers, labels=None, config=None, replicates=1, out=None):
    variants = {
        "baseline": {"PRIORITIZE_FOOD": False, "FOOD_AWARE_BREEDING": False},
        "food": {"PRIORITIZE_FOOD": True, "FOOD_AWARE_BREEDING": False},
        "breeding": {"PRIORITIZE_FOOD": False, "FOOD_AWARE_BREEDING": True},
        "combined": {"PRIORITIZE_FOOD": True, "FOOD_AWARE_BREEDING": True},
    }
    if config is not None:
        with open(config) as handle:
            variants = json.load(handle)
    if labels is not None:
        variants = {label: variants[label] for label in labels}
    reports = []
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    jobs = [(label, seed, params, horizon, rep)
            for rep in range(replicates)
            for seed in seeds
            for label, params in variants.items()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_compare_one, job) for job in jobs]
        for future in as_completed(futures):
            report = future.result()
            reports.append(report)
            print(json.dumps(report), flush=True)
    if out is not None:
        with open(out, "w") as handle:
            json.dump({"horizon": horizon, "seeds": list(seeds),
                       "replicates": replicates, "variants": variants,
                       "runs": reports}, handle, indent=1)
        print(f"raw runs written to {out}", flush=True)
    print("Same seeds are not deterministic across processes; compare distributions, not paired wins.")
    for label in variants:
        group = [report for report in reports if report["variant"] == label]
        scores = [report["score"] for report in group]
        summary = {"variant": label, "runs": len(group),
                   "mean": statistics.mean(scores), "median": statistics.median(scores),
                   "minimum": min(scores), "maximum": max(scores),
                   "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                   "survived": sum(report["alive"] > 0 for report in group),
                   "mean_policy_ms": statistics.mean(report["policy_mean_ms"] for report in group)}
        for metric in ("eaten", "starved", "young_deaths", "births"):
            summary[metric] = sum(report[metric] for report in group)
        print("SUMMARY " + json.dumps(summary), flush=True)
    return reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["search", "validate", "baseline", "compare"])
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=1,
                    help="keep at 1; parallelism comes from the worker pool")
    ap.add_argument("--params", default=BEST_PATH)
    ap.add_argument("--study", default="survival")
    ap.add_argument("--storage", default="sqlite:///optuna_survival.db")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=VALIDATE_SEEDS)
    ap.add_argument("--variants", nargs="+")
    ap.add_argument("--config", help="JSON file of {variant_name: {tunable: value}}")
    ap.add_argument("--replicates", type=int, default=1,
                    help="repeat every seed this many times to expose run-to-run noise")
    ap.add_argument("--out", help="write raw per-run results to this JSON file")
    ap.add_argument("--horizon", type=int, default=MAX_TIME,
                    help="sim-seconds cap per run (shorter = faster search)")
    args = ap.parse_args()

    if args.mode == "compare":
        if not 1 <= args.workers <= WORKERS or args.horizon <= 0:
            ap.error(f"compare requires 1..{WORKERS} workers and a positive horizon")
        compare(args.seeds, args.horizon, args.workers, args.variants, args.config,
                args.replicates, args.out)
    elif args.mode == "search":
        search(args.trials, args.jobs, args.study, args.storage, args.horizon)
    elif args.mode == "baseline":
        validate({}, VALIDATE_SEEDS, "baseline (current defaults)", args.horizon)
    else:
        with open(args.params) as fh:
            params = json.load(fh)
        print(f"params: {params}")
        validate(params, VALIDATE_SEEDS, "tuned", args.horizon)


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _here)
    os.chdir(_here)
    main()
