"""Phase-0 go/no-go gate for a learned camera policy.

Question: does a *learned* camera policy beat the hardcoded loop at all, on
held-out frames, before we invest in a neural/RL upgrade? Pure in-process COCO
AP50 over a frozen detector-output cache. NO competition API call is made.

Reuses train_separate.cache_detector / fit_camera / rollout unchanged, and adds:
  * a RandomLegalPolicy floor (uniform over legal_actions),
  * multi-seed robustness of the fitted linear policy,
  * a temporal k-fold generalization probe (refit per fold).

Gate (printed as VERDICT): learned must beat BOTH the fixed CameraPolicy and the
random floor by >= --margin AP50 on the temporal diagnostic AND on median k-fold.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

LAB = Path(__file__).resolve().parent
ROOT = LAB.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LAB))

from train_separate import cache_detector, fit_camera, rollout, save_json, pin_reporting  # noqa: E402


def load_cache(path):
    records = json.loads(Path(path).read_text(encoding='utf-8'))
    cache = {}
    for record in records:
        cache[record['frame'], tuple(record['action'])] = record['predictions']
    return cache


class RandomLegalPolicy:
    """Uniform over the legal action set. No `coefficients` -> rollout calls it
    without `world`, matching a policy that needs no world state."""

    def __init__(self, seed):
        self.generator = np.random.default_rng(seed)

    def reset(self):
        pass

    def __call__(self, level, center_x, center_y, allowed_levels, maximum_delta):
        from camera_policy import legal_actions
        actions = legal_actions(level, center_x, center_y, allowed_levels, maximum_delta)
        if not actions:
            return None
        return actions[int(self.generator.integers(len(actions)))]


def fit_coefficients(cache, train_frames, seed, iterations):
    from scipy.optimize import differential_evolution
    from camera_policy import FEATURES, LearnedCameraPolicy

    generator = np.random.default_rng(seed)
    population = generator.uniform(-3, 3, size=(26, len(FEATURES)))
    population[0] = 0
    population[0, FEATURES.index('overview')] = 1

    def objective(coefficients):
        return -rollout(cache, train_frames, LearnedCameraPolicy(coefficients))['ap50']

    optimized = differential_evolution(
        objective, [(-3, 3)] * len(FEATURES), init=population, maxiter=iterations,
        seed=seed, workers=1, polish=False, tol=0, atol=0)
    return optimized.x


def ap(cache, frames, policy):
    return rollout(cache, frames, policy)['ap50']


def random_floor(cache, frames, seeds):
    return median(ap(cache, frames, RandomLegalPolicy(s)) for s in seeds)


def kfold(cache, frames, k, seed, iterations):
    from solution import CameraPolicy
    from camera_policy import LearnedCameraPolicy

    ordered = list(frames)
    folds = np.array_split(np.arange(len(ordered)), k)
    rows = []
    for i, block in enumerate(folds):
        held = [ordered[j] for j in block]
        train = [ordered[j] for j in range(len(ordered)) if j not in set(block.tolist())]
        coefficients = fit_coefficients(cache, train, seed, iterations)
        learned = ap(cache, held, LearnedCameraPolicy(coefficients))
        fixed = ap(cache, held, CameraPolicy())
        floor = random_floor(cache, held, [seed, seed + 1, seed + 2])
        rows.append({'fold': i, 'held': held, 'learned': learned, 'fixed': fixed, 'random': floor})
        print(f'  fold {i}: learned={learned:.4f} fixed={fixed:.4f} random={floor:.4f} '
              f'(held {held[0]}..{held[-1]})', flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detector', type=Path, default=ROOT / 'model' / 'v4s1.pt')
    parser.add_argument('--out', type=Path, default=LAB / 'runs' / 'camera_gate_v1')
    parser.add_argument('--iterations', type=int, default=40)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seeds', type=int, default=3, help='seed-robustness count for the canonical fit')
    parser.add_argument('--kfold', type=int, default=0, help='0 disables the k-fold probe')
    parser.add_argument('--margin', type=float, default=0.01)
    parser.add_argument('--imgsz', type=int, default=960)
    arguments = parser.parse_args()

    pin_reporting()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')

    from utils import frame_numbers
    from solution import CameraPolicy
    from camera_policy import LearnedCameraPolicy

    if not arguments.detector.is_file():
        raise FileNotFoundError(arguments.detector)
    arguments.out.mkdir(parents=True, exist_ok=True)
    frames = frame_numbers('helsinki')
    print(f'frames: {len(frames)} ({frames[0]}..{frames[-1]})', flush=True)

    cache_path = arguments.out / 'detector_cache.json'
    if cache_path.is_file():
        print(f'loading cache {cache_path}', flush=True)
        cache = load_cache(cache_path)
    else:
        print(f'building detector cache with {arguments.detector.name} ...', flush=True)
        started = time.perf_counter()
        cache = cache_detector(arguments.detector, frames, arguments.out, arguments.imgsz)
        print(f'cached in {time.perf_counter() - started:.1f}s', flush=True)

    # Canonical fit (train=frames[:16], diagnostic=frames[19:]); writes camera.json + metrics.
    print('=== canonical fit (train[:16] / diagnostic[19:]) ===', flush=True)
    fit_camera(cache, frames, arguments.detector, arguments.out, arguments.iterations,
               arguments.seed, arguments.imgsz)
    metrics = json.loads((arguments.out / 'policy_metrics.json').read_text(encoding='utf-8'))
    diag_fixed = metrics['comparisons']['temporal_diagnostic']['fixed']['ap50']
    diag_learned = metrics['comparisons']['temporal_diagnostic']['learned']['ap50']
    all_fixed = metrics['comparisons']['all']['fixed']['ap50']
    all_learned = metrics['comparisons']['all']['learned']['ap50']

    diagnostic_frames = frames[19:]
    diag_random = random_floor(cache, diagnostic_frames, [arguments.seed + i for i in range(3)])
    all_random = random_floor(cache, frames, [arguments.seed + i for i in range(3)])
    print(f'temporal_diagnostic: learned={diag_learned:.4f} fixed={diag_fixed:.4f} random={diag_random:.4f}',
          flush=True)
    print(f'all:                 learned={all_learned:.4f} fixed={all_fixed:.4f} random={all_random:.4f}',
          flush=True)

    # Seed robustness of the canonical fit (refit on train[:16], eval on diagnostic[19:]).
    seed_gaps = []
    if arguments.seeds > 1:
        print('=== seed robustness (fit train[:16], eval diagnostic[19:]) ===', flush=True)
        for s in range(arguments.seeds):
            coefficients = fit_coefficients(cache, frames[:16], s, arguments.iterations)
            learned = ap(cache, diagnostic_frames, LearnedCameraPolicy(coefficients))
            seed_gaps.append(learned - diag_fixed)
            print(f'  seed {s}: learned={learned:.4f} gap_vs_fixed={learned - diag_fixed:+.4f}', flush=True)
        print(f'  median gap vs fixed: {median(seed_gaps):+.4f}', flush=True)

    fold_rows = []
    if arguments.kfold:
        print(f'=== temporal {arguments.kfold}-fold generalization probe ===', flush=True)
        fold_rows = kfold(cache, frames, arguments.kfold, arguments.seed, arguments.iterations)
        med_learned = median(r['learned'] for r in fold_rows)
        med_fixed = median(r['fixed'] for r in fold_rows)
        med_random = median(r['random'] for r in fold_rows)
        print(f'  median: learned={med_learned:.4f} fixed={med_fixed:.4f} random={med_random:.4f}', flush=True)

    # Verdict
    beats_diag = (diag_learned >= diag_fixed + arguments.margin) and (diag_learned >= diag_random + arguments.margin)
    beats_fold = True
    if fold_rows:
        med_learned = median(r['learned'] for r in fold_rows)
        med_fixed = median(r['fixed'] for r in fold_rows)
        med_random = median(r['random'] for r in fold_rows)
        beats_fold = (med_learned >= med_fixed + arguments.margin) and (med_learned >= med_random + arguments.margin)
    verdict = 'PASS' if (beats_diag and beats_fold) else 'FAIL'
    print(f'\nVERDICT: {verdict} '
          f'(diagnostic beats fixed+random by margin: {beats_diag}; kfold: {beats_fold if fold_rows else "n/a"})',
          flush=True)

    summary = {
        'detector': str(arguments.detector), 'margin': arguments.margin,
        'iterations': arguments.iterations, 'seed': arguments.seed,
        'temporal_diagnostic': {'learned': diag_learned, 'fixed': diag_fixed, 'random': diag_random},
        'all': {'learned': all_learned, 'fixed': all_fixed, 'random': all_random},
        'seed_gaps_vs_fixed': seed_gaps, 'seed_gap_median': (median(seed_gaps) if seed_gaps else None),
        'kfold': fold_rows, 'verdict': verdict,
    }
    save_json(arguments.out / 'gate_summary.json', summary)
    print(f'wrote {arguments.out / "gate_summary.json"}', flush=True)


if __name__ == '__main__':
    main()
