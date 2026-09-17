"""Train detection first, then fit a separate causal camera action scorer.

Uses only provided/Helsinki-derived data. Camera fitting uses actual frozen
detector outputs and complete local AP50, not an oracle detector or box counts.
Helsinki splits overlap in scene/asset content: diagnostics are not city transfer.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

LAB = Path(__file__).resolve().parent
ROOT = LAB.parent
sys.path.insert(0, str(ROOT))
DEFAULT_OUT = LAB / 'runs' / 'separate_v1'


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def pin_reporting():
    os.environ.update(DRONE_ALTERNATES='2', DRONE_ALTERNATE_DAMPING='0.5',
                      DRONE_ASSIGN='0', DRONE_MAX_MISSES='3')


def cache_detector(weights, frames, directory, imgsz):
    from camera_policy import action_grid, region_for
    from simulate import render_view
    from solution import Detector
    from utils import load_frame

    logging.getLogger('ultralytics').setLevel(logging.ERROR)
    detector = Detector(weights=str(weights), imgsz=imgsz, confidence=0.05, device='0')
    cache = {}
    try:
        for number in frames:
            image = load_frame(number, 'helsinki')
            for action in action_grid():
                view, region = render_view(image, *action)
                assert tuple(region) == region_for(action)
                cache[number, action] = [
                    {'object_id': str(item['object_id']), 'confidence': float(item['confidence']),
                     'bbox': [float(value) for value in item['bbox']]}
                    for item in detector(view, region)]
            print(f'cached detector frame {number}: {len(cache)} frame/view pairs', flush=True)
    finally:
        detector._pool.shutdown(wait=True)
    save_json(directory / 'detector_cache.json', [
        {'frame': number, 'action': list(action), 'predictions': predictions}
        for (number, action), predictions in cache.items()])
    return cache


def rollout(cache, frames, policy):
    from camera_policy import region_for
    from local_evaluator import Camera
    from solution import WorldModel
    from utils import load_annotations
    import local_evaluator

    world = WorldModel()
    camera = Camera()
    predictions, trace, timings = {}, [], []
    for frame in frames:
        action = (camera.resolution_level, camera.center_x, camera.center_y)
        world.advance(frame)
        world.update(cache[frame, action], region_for(action))
        predictions[frame] = world.report()
        constraints = camera.constraints()
        started = time.perf_counter()
        if hasattr(policy, 'coefficients'):
            command = policy(*action, constraints['allowed_resolution_levels'],
                             constraints['maximum_center_delta'], world=world)
        else:
            command = policy(*action, constraints['allowed_resolution_levels'],
                             constraints['maximum_center_delta'])
        timings.append((time.perf_counter() - started) * 1000)
        trace.append({'frame': frame, 'view': list(action),
                      'next_view': list(command) if command is not None else None,
                      'reports': len(predictions[frame])})
        if command is not None:
            if not all(type(value) is int for value in command):
                raise AssertionError('camera fields must be integers')
            camera.apply(*command)
    truth = {frame: load_annotations(frame, 'helsinki') for frame in frames}
    with patch.object(local_evaluator, 'frame_numbers', return_value=list(frames)), \
            patch.object(local_evaluator, 'load_annotations', side_effect=lambda frame, scene: truth[frame]):
        score, per_class = local_evaluator.score('camera_fit', predictions)
    return {'ap50': float(score), 'per_class': per_class, 'trace': trace,
            'camera_ms_median': float(np.median(timings)),
            'camera_ms_p95': float(np.percentile(timings, 95)), 'camera_refusals': 0}


def fit_camera(cache, frames, detector_weights, out, iterations, seed, imgsz):
    from scipy.optimize import differential_evolution
    from camera_policy import FEATURES, SCHEMA_VERSION, LearnedCameraPolicy, file_hash
    from solution import CameraPolicy

    train_frames = frames[:16]
    diagnostic_frames = frames[19:]
    if not diagnostic_frames:
        raise ValueError('Need the complete supplied flight for temporal diagnostics')
    generator = np.random.default_rng(seed)
    population = generator.uniform(-3, 3, size=(26, len(FEATURES)))
    population[0] = 0
    population[0, FEATURES.index('overview')] = 1
    history = []

    def objective(coefficients):
        result = rollout(cache, train_frames, LearnedCameraPolicy(coefficients))
        history.append({'evaluation': len(history) + 1, 'train_ap50': result['ap50']})
        print(f'camera fit {len(history)}: train AP50={result["ap50"]:.4f}', flush=True)
        return -result['ap50']

    optimized = differential_evolution(objective, [(-3, 3)] * len(FEATURES),
                                       init=population, maxiter=iterations, seed=seed,
                                       workers=1, polish=False, tol=0, atol=0)
    coefficients = optimized.x
    policy_state = {'schema_version': SCHEMA_VERSION, 'features': list(FEATURES),
                    'coefficients': coefficients.tolist(), 'detector_sha256': file_hash(detector_weights),
                    'imgsz': imgsz, 'confidence': 0.05, 'seed': seed,
                    'reward': 'complete train-prefix macro COCO AP50 with frozen detector and WorldModel',
                    'optimizer': 'scipy.optimize.differential_evolution',
                    'train_frames': train_frames, 'diagnostic_frames': diagnostic_frames,
                    'limitations': 'Short overlapping Helsinki scene; no unseen-city guarantee; not an optimal-policy proof.'}
    save_json(out / 'camera.json', policy_state)
    results = {}
    for name, subset in [('train', train_frames), ('temporal_diagnostic', diagnostic_frames), ('all', frames)]:
        results[name] = {
            'fixed': rollout(cache, subset, CameraPolicy()),
            'learned': rollout(cache, subset, LearnedCameraPolicy.from_checkpoint(out / 'camera.json', detector_weights)),
        }
        print(f'{name}: fixed={results[name]["fixed"]["ap50"]:.4f} '
              f'learned={results[name]["learned"]["ap50"]:.4f}', flush=True)
    save_json(out / 'policy_metrics.json', {'history': history, 'comparisons': results})
    return policy_state


def train(arguments):
    from ultralytics import YOLO
    from camera_policy import file_hash
    from utils import frame_numbers

    if arguments.out.exists():
        raise FileExistsError(f'Choose a new --out to preserve existing results: {arguments.out}')
    if not arguments.initial.is_file():
        raise FileNotFoundError(arguments.initial)
    dataset = LAB / 'dataset_v5' / 'data.yaml'
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    arguments.out.mkdir(parents=True)
    settings = {'initial': str(arguments.initial.resolve()), 'initial_sha256': file_hash(arguments.initial),
                'dataset': str(dataset), 'epochs': arguments.epochs, 'lr0': arguments.lr,
                'optimizer': 'AdamW', 'imgsz': 960, 'seed': arguments.seed,
                'status': 'detector_training', 'policy_iterations': arguments.policy_iterations}
    save_json(arguments.out / 'run.json', settings)
    model = YOLO(str(arguments.initial.resolve()))
    print('STAGE 1: supervised detector fine-tune; Helsinki-derived positives AND negatives.', flush=True)
    model.train(data=str(dataset), project=str(arguments.out.resolve()), name='detector',
                exist_ok=False, epochs=arguments.epochs, imgsz=960, batch=arguments.batch,
                workers=arguments.workers, device=0, optimizer='AdamW', lr0=arguments.lr,
                lrf=0.1, warmup_epochs=0, cos_lr=True, patience=arguments.epochs,
                close_mosaic=min(3, arguments.epochs), mosaic=0.4, mixup=0.0,
                degrees=180.0, fliplr=0.5, flipud=0.5, scale=0.2, translate=0.1,
                shear=1.0, hsv_h=0.02, hsv_s=0.6, hsv_v=0.4,
                cache=False, val=True, plots=False, seed=arguments.seed)
    detector_weights = Path(model.trainer.best)
    if not detector_weights.is_file():
        raise FileNotFoundError('Detector training did not produce best weights')
    actual_lr = model.trainer.args.lr0
    if model.trainer.args.optimizer != 'AdamW' or not np.isclose(actual_lr, arguments.lr):
        raise AssertionError('Optimizer or requested learning rate was overridden')
    settings.update(status='camera_training', detector=str(detector_weights.resolve()),
                    detector_sha256=file_hash(detector_weights), actual_lr0=float(actual_lr))
    save_json(arguments.out / 'run.json', settings)
    del model
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    frames = frame_numbers('helsinki')
    print('STAGE 2: frozen detector view cache, then direct AP camera-policy fitting.', flush=True)
    cache = cache_detector(detector_weights, frames, arguments.out, 960)
    policy = fit_camera(cache, frames, detector_weights, arguments.out,
                        arguments.policy_iterations, arguments.seed, 960)
    manifest = {'schema_version': 1, 'detector': str(detector_weights.resolve()),
                'detector_sha256': file_hash(detector_weights),
                'camera': str((arguments.out / 'camera.json').resolve()),
                'camera_sha256': file_hash(arguments.out / 'camera.json'),
                'imgsz': 960, 'confidence': 0.05,
                'reporting': {'alternates': 2, 'alternate_damping': 0.5, 'assign': 0, 'max_misses': 3},
                'training_reward': policy['reward'], 'status': 'trained_pending_manual_validation'}
    settings['status'] = 'complete'
    save_json(arguments.out / 'run.json', settings)
    save_json(arguments.out / 'experiment.json', manifest)
    print(f'COMPLETE: {arguments.out / "experiment.json"}', flush=True)
    print('Use api.py --experiment separate-v1 or separate-control. No competition call was made.')


def checks():
    import tempfile
    from camera_policy import FEATURES, SCHEMA_VERSION, LearnedCameraPolicy, action_grid, legal_actions, file_hash
    from local_evaluator import Camera
    from solution import WorldModel

    assert len(action_grid()) == 59
    for action in action_grid():
        camera = Camera(*action)
        constraints = camera.constraints()
        choices = legal_actions(*action, constraints['allowed_resolution_levels'], constraints['maximum_center_delta'])
        assert action in choices
        for choice in choices:
            Camera(*action).apply(*choice)
            assert all(type(value) is int for value in choice)
    overview = np.zeros(len(FEATURES))
    overview[FEATURES.index('overview')] = 1
    policy = LearnedCameraPolicy(overview)
    world = WorldModel(np.eye(3))
    world.advance(0)
    assert policy(0, 1920, 1080, [0, 1], 2203, world) == (0, 1920, 1080)
    assert np.all(policy.ages[0] == 0) and np.all(policy.ages[2] == 60)
    world.advance(4)
    policy(2, 1920, 1080, [1, 2], 551, world)
    assert np.min(policy.ages[2]) == 0 and np.max(policy.ages[2]) == 60
    world.advance(0)
    policy(0, 1920, 1080, [0, 1], 2203, world)
    assert np.all(policy.ages[2] == 60)
    snapshot = policy.ages.copy()
    actions = legal_actions(0, 1920, 1080, [0, 1], 2203)
    features = policy.features(actions, 0, 1920, 1080, 2203, world)
    assert features.shape == (len(actions), len(FEATURES)) and np.isfinite(features).all()
    assert np.array_equal(snapshot, policy.ages)
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        weights = root / 'detector.bin'
        weights.write_bytes(b'test detector')
        saved = root / 'camera.json'
        save_json(saved, {'schema_version': SCHEMA_VERSION, 'features': list(FEATURES),
                          'coefficients': overview.tolist(), 'detector_sha256': file_hash(weights)})
        loaded = LearnedCameraPolicy.from_checkpoint(saved, weights)
        np.testing.assert_array_equal(loaded.coefficients, overview)
        weights.write_bytes(b'changed detector')
        try:
            LearnedCameraPolicy.from_checkpoint(saved, weights)
        except ValueError:
            pass
        else:
            raise AssertionError('Mismatched detector checkpoint was accepted')
    print('PASS: legal action grid, coverage age/reset, finite causal features, reload and detector hash guard.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'train'])
    parser.add_argument('--initial', type=Path, default=ROOT / 'model' / 'v4s1.pt')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--policy-iterations', type=int, default=6)
    parser.add_argument('--seed', type=int, default=0)
    arguments = parser.parse_args()
    if min(arguments.epochs, arguments.batch) < 1 or min(arguments.workers, arguments.policy_iterations) < 0:
        parser.error('Invalid training counts')
    if not np.isfinite(arguments.lr) or arguments.lr <= 0:
        parser.error('--lr must be finite and positive')
    pin_reporting()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    checks()
    if arguments.command == 'train':
        train(arguments)


if __name__ == '__main__':
    main()