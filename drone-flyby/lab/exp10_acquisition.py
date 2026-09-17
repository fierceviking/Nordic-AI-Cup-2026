"""Offline geometry-only test of survey -> L2 inspection -> causal memory.

Oracle proposals and perfect L2 verification make this an optimistic feasibility
study, not a detector benchmark. No image generation, training, or API calls.
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))

import local_evaluator
from dtos import IMAGE_HEIGHT, IMAGE_WIDTH, OBJECT_CLASSES
from exp01_bounds import H_MEAN, RasterPolicy, clip, full_raster_centers, top_band_centers, warp_box
from utils import frame_numbers, load_annotations


def intersection_fraction(box, region):
    width = max(0.0, min(box[2], region[2]) - max(box[0], region[0]))
    height = max(0.0, min(box[3], region[3]) - max(box[1], region[1]))
    area = max(1e-9, (box[2] - box[0]) * (box[3] - box[1]))
    return width * height / area


def overlap(box, other):
    width = max(0.0, min(box[2], other[2]) - max(box[0], other[0]))
    height = max(0.0, min(box[3], other[3]) - max(box[1], other[1]))
    area = (box[2] - box[0]) * (box[3] - box[1])
    other_area = (other[2] - other[0]) * (other[3] - other[1])
    return width * height / max(1e-9, area + other_area - width * height)


class Observation:
    def __init__(self, box, frame, name=None):
        self.box = list(box)
        self.frame = frame
        self.name = name

    def at(self, frame, homography):
        return warp_box(np.linalg.matrix_power(homography, frame - self.frame), self.box)


def legal_move(camera, target_level, target):
    constraints = camera.constraints()
    level = max(camera.resolution_level - 1, min(camera.resolution_level + 1, target_level))
    assert level in constraints['allowed_resolution_levels']
    bounds = next(item for item in constraints['center_bounds'] if item['resolution_level'] == level)
    if level == 0:
        return 0, 1920, 1080
    target_x = np.clip(target[0], bounds['minimum_center_x'], bounds['maximum_center_x'])
    target_y = np.clip(target[1], bounds['minimum_center_y'], bounds['maximum_center_y'])
    delta_x, delta_y = target_x - camera.center_x, target_y - camera.center_y
    distance = math.hypot(delta_x, delta_y)
    limit = constraints['maximum_center_delta'] * 0.995
    if distance > limit:
        target_x = camera.center_x + delta_x * limit / distance
        target_y = camera.center_y + delta_y * limit / distance
    target_x = np.clip(target_x, bounds['minimum_center_x'], bounds['maximum_center_x'])
    target_y = np.clip(target_y, bounds['minimum_center_y'], bounds['maximum_center_y'])
    return level, int(round(target_x)), int(round(target_y))


class AcquisitionPolicy:
    """Anonymous geometry only; reserve periodic overview and exploration views."""

    def __init__(self, homography, survey_every=6, max_inspect=5):
        self.homography = homography
        self.survey_every = survey_every
        self.max_inspect = max_inspect
        self.pending = []
        self.examined = []
        self.target = None
        self.target_start = 0
        self.last_survey = -survey_every
        self.returning = False
        self.sweep = RasterPolicy(2, top_band_centers(2))

    def observe(self, frame, camera, proposals):
        for collection in (self.pending, self.examined):
            collection[:] = [item for item in collection if clip(item.at(frame, self.homography)) is not None]
        if camera.resolution_level == 0:
            self.last_survey = frame
            self.returning = False
        for box in proposals:
            if not any(overlap(box, item.at(frame, self.homography)) > 0.25
                       for item in self.pending + self.examined):
                self.pending.append(Observation(box, frame))
        if camera.resolution_level == 2:
            inspected = [item for item in self.pending
                         if intersection_fraction(item.at(frame, self.homography), camera.source_region) >= 0.8]
            self.examined.extend(inspected)
            self.pending = [item for item in self.pending if item not in inspected]
        if self.target not in self.pending:
            self.target = None

    def command(self, frame, camera, next_gap=1):
        if frame - self.last_survey >= self.survey_every:
            self.returning = True
            self.target = None
        if self.returning:
            return legal_move(camera, 0, (1920, 1080))
        if self.target is not None and frame - self.target_start >= self.max_inspect:
            self.target = None
            self.returning = True
            return legal_move(camera, 0, (1920, 1080))
        if self.target is None and self.pending:
            def utility(item):
                box = item.at(frame, self.homography)
                center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                travel = math.hypot(center[0] - camera.center_x, center[1] - camera.center_y)
                future = item.at(frame + 1, self.homography)
                speed = max(1.0, (future[1] + future[3] - box[1] - box[3]) / 2)
                lifetime = max(0.0, (IMAGE_HEIGHT - center[1]) / speed)
                return min(lifetime, 40.0) / (1.0 + travel / 551.0)

            self.target = max(self.pending, key=utility)
            self.target_start = frame
        if self.target is not None:
            box = self.target.at(frame + next_gap, self.homography)
            return legal_move(camera, 2, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
        request = SimpleNamespace(resolution_level=camera.resolution_level,
                                  center_x=camera.center_x, center_y=camera.center_y,
                                  allowed_levels=camera.constraints()['allowed_resolution_levels'],
                                  maximum_center_delta=camera.constraints()['maximum_center_delta'])
        return self.sweep(request)


def supplied_scene():
    truth = {frame: load_annotations(frame, 'helsinki') for frame in frame_numbers('helsinki')}
    return truth


def synthetic_scene(length=249, seed=0, density=10, clutter_ratio=0):
    """Annotation trajectories using Helsinki box sizes, not rendered imagery."""
    generator = np.random.default_rng(seed)
    sizes = {name: [] for name in OBJECT_CLASSES}
    for annotations in supplied_scene().values():
        for item in annotations:
            box = item['bbox']
            sizes[item['object_id']].append((box[2] - box[0], box[3] - box[1]))
    objects, clutter = [], []
    count = max(len(OBJECT_CLASSES), int(math.ceil(density * length / 32)))
    for index in range(count * (1 + clutter_ratio)):
        name = OBJECT_CLASSES[index % len(OBJECT_CLASSES)]
        width, height = np.median(sizes[name], axis=0)
        reference = int(generator.integers(0, length))
        center_x = float(generator.uniform(150, IMAGE_WIDTH - 150))
        center_y = IMAGE_HEIGHT / 2
        item = Observation([center_x - width / 2, center_y - height / 2,
                            center_x + width / 2, center_y + height / 2], reference, name)
        (objects if index < count else clutter).append(item)
    truth, decoys = {}, {}
    for frame in range(length):
        truth[frame] = [{'object_id': item.name, 'bbox': clipped}
                        for item in objects if (clipped := clip(item.at(frame, H_MEAN))) is not None]
        decoys[frame] = [clipped for item in clutter
                         if (clipped := clip(item.at(frame, H_MEAN))) is not None]
    return truth, decoys


def score_truth(truth, predictions):
    if not any(truth.values()):
        return None
    with patch.object(local_evaluator, 'frame_numbers', return_value=list(truth)), \
         patch.object(local_evaluator, 'load_annotations', side_effect=lambda frame, scene: truth[frame]):
        return local_evaluator.score('acquisition_geometry_only', predictions)[0]


def run_case(truth, decoys, policy_name, proposal_min_px=0, latency_ms=0,
             homography=H_MEAN, survey_every=6):
    camera = local_evaluator.Camera()
    policy = (AcquisitionPolicy(homography, survey_every) if policy_name == 'acquire' else
              RasterPolicy(2, top_band_centers(2) if policy_name == 'sweep-top' else full_raster_centers(2)))
    memory = []
    predictions = {frame: [] for frame in truth}
    next_frame = min(truth)
    levels = Counter()
    inspected = 0
    rejected = 0
    proposal_total = 0
    processed = 0
    gap = max(1, math.ceil(latency_ms / local_evaluator.FRAME_INTERVAL_MS))
    for frame, annotations in truth.items():
        if frame < next_frame:
            continue
        next_frame = frame + gap
        processed += 1
        levels[camera.resolution_level] += 1
        region = camera.source_region
        factor = (region[2] - region[0]) / 960
        proposals = [list(box) for box in [item['bbox'] for item in annotations] + decoys.get(frame, [])
                     if intersection_fraction(box, region) >= 0.8
                     and min(box[2] - box[0], box[3] - box[1]) / factor >= proposal_min_px]
        proposal_total += len(proposals)
        memory = [item for item in memory if clip(item.at(frame, homography)) is not None]
        if camera.resolution_level == 2:
            for item in annotations:
                if intersection_fraction(item['bbox'], region) < 0.8:
                    continue
                matches = [entry for entry in memory if entry.name == item['object_id']
                           and overlap(entry.at(frame, homography), item['bbox']) >= 0.25]
                if matches:
                    entry = max(matches, key=lambda entry: overlap(entry.at(frame, homography), item['bbox']))
                    entry.box, entry.frame = list(item['bbox']), frame
                else:
                    memory.append(Observation(item['bbox'], frame, item['object_id']))
                    inspected += 1
            rejected += sum(intersection_fraction(box, region) >= 0.8 for box in decoys.get(frame, []))
        predictions[frame] = [{'object_id': entry.name, 'bbox': clip(entry.at(frame, homography)),
                               'confidence': 1 / (1 + 0.01 * (frame - entry.frame))}
                              for entry in memory]
        if policy_name == 'acquire':
            policy.observe(frame, camera, proposals)
            command = policy.command(frame, camera, gap)
        else:
            request = SimpleNamespace(resolution_level=camera.resolution_level,
                                      center_x=camera.center_x, center_y=camera.center_y,
                                      allowed_levels=camera.constraints()['allowed_resolution_levels'],
                                      maximum_center_delta=camera.constraints()['maximum_center_delta'])
            command = policy(request)
        if command is not None:
            assert all(type(value) is int for value in command), command
            camera.apply(*command)
    result = dict(policy=policy_name, map50=score_truth(truth, predictions),
                  processed=processed, skipped=len(truth) - processed,
                  levels=dict(levels), new_confirmations=inspected,
                  clutter_inspections=rejected, proposals_seen=proposal_total,
                  camera_refusals=0)
    return result, predictions


def checks():
    identity = np.eye(3)
    camera = local_evaluator.Camera()
    camera.apply(*legal_move(camera, 2, (3000, 1000)))
    assert camera.resolution_level == 1
    camera.apply(*legal_move(camera, 2, (3000, 1000)))
    assert camera.resolution_level == 2
    empty, proposals = AcquisitionPolicy(identity), AcquisitionPolicy(identity)
    empty.observe(0, local_evaluator.Camera(), [])
    proposals.observe(0, local_evaluator.Camera(), [[1800, 950, 1840, 990]])
    assert not empty.pending and len(proposals.pending) == 1
    assert all(item.name is None for item in proposals.pending)
    truth = {frame: [{'object_id': OBJECT_CLASSES[0], 'bbox': [1800, 950, 1840, 990]}]
             for frame in range(12)}
    result, predictions = run_case(truth, {}, 'acquire', homography=identity)
    assert not predictions[0] and not predictions[1]
    assert any(predictions[frame] for frame in range(2, 12))
    _, slow = run_case(truth, {}, 'acquire', latency_ms=400, homography=identity)
    assert all(not slow[frame] for frame in range(1, 12, 2))
    _, no_targets = run_case({frame: [] for frame in truth}, {}, 'acquire', homography=identity)
    assert not any(no_targets.values())
    repeated = {frame: items + [{'object_id': OBJECT_CLASSES[0], 'bbox': [2000, 950, 2040, 990]}]
                for frame, items in truth.items()}
    _, multiple = run_case(repeated, {}, 'acquire', homography=identity)
    assert max(map(len, multiple.values())) == 2
    _, decoys_only = run_case({frame: [] for frame in truth},
                              {frame: [[1800, 950, 1840, 990]] for frame in truth},
                              'acquire', homography=identity)
    assert not any(decoys_only.values())
    clean_truth, _ = synthetic_scene(length=40, seed=3, clutter_ratio=0)
    cluttered_truth, _ = synthetic_scene(length=40, seed=3, clutter_ratio=2)
    assert clean_truth == cluttered_truth
    policy = AcquisitionPolicy(identity)
    policy.observe(0, local_evaluator.Camera(), [[1800, 950, 1840, 990]])
    inspection = local_evaluator.Camera(2, 1920, 1080)
    policy.observe(1, inspection, [])
    assert not policy.pending and policy.examined
    oracle = {frame: [dict(item, confidence=1.0) for item in items] for frame, items in truth.items()}
    assert abs(score_truth(truth, oracle) - 1) < 1e-9
    assert 0 < result['map50'] < 1
    print('PASS: legal moves, anonymous proposals, L2-only confirmation, causal memory, '
          'skipped frames, repeated classes, decoy rejection, fixed trajectories, oracle AP.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--frames', type=int, default=249)
    parser.add_argument('--density', type=int, default=10)
    parser.add_argument('--clutter-ratio', type=int, default=0)
    parser.add_argument('--proposal-min-px', type=float, default=0)
    parser.add_argument('--latency-ms', type=float, default=0)
    parser.add_argument('--out', type=Path, default=LAB / 'out' / 'acquisition_bound.json')
    arguments = parser.parse_args()
    if arguments.check:
        checks()
        return
    if min(arguments.seeds, arguments.frames, arguments.density) < 1 or min(
            arguments.clutter_ratio, arguments.proposal_min_px, arguments.latency_ms) < 0:
        parser.error('counts must be positive and stress settings nonnegative')
    cases = [('helsinki', supplied_scene(), {})]
    for seed in range(arguments.seeds):
        truth, decoys = synthetic_scene(arguments.frames, seed, arguments.density, arguments.clutter_ratio)
        cases.append((f'synthetic-{seed}', truth, decoys))
    results = []
    for name, truth, decoys in cases:
        for policy_name in ('sweep-top', 'sweep-full', 'acquire'):
            result, _ = run_case(truth, decoys, policy_name, arguments.proposal_min_px, arguments.latency_ms)
            result['scene'] = name
            results.append(result)
            print(f'{name:16s} {policy_name:12s} AP50={result["map50"]:.4f} '
                  f'views={result["levels"]} skipped={result["skipped"]}', flush=True)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    settings = vars(arguments).copy()
    settings['out'] = str(settings['out'])
    arguments.out.write_text(json.dumps(dict(settings=settings, results=results), indent=2), encoding='utf-8')
    print(f'Geometry-only optimistic study, NOT model/competition performance. Saved {arguments.out}')


if __name__ == '__main__':
    main()