"""Small camera action scorer trained independently of a frozen detector."""

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from solution import SOURCE_REGION_SIZES, center_bounds, warp_bbox


FEATURES = (
    'overview', 'level1', 'level2', 'hold', 'move_cost', 'scale_change',
    'unseen_fraction', 'mean_age', 'top_position', 'track_evidence',
    'small_track_evidence', 'uncertain_track_evidence', 'stale_track_evidence',
)
GRID_HEIGHT, GRID_WIDTH = 18, 32
AGE_LIMIT = 60.0
SCHEMA_VERSION = 1


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def action_grid():
    actions = [(0, 1920, 1080)]
    actions.extend((1, center_x, center_y)
                   for center_y in (540, 1080, 1620) for center_x in (960, 1920, 2880))
    actions.extend((2, center_x, center_y)
                   for center_y in range(270, 1891, 270) for center_x in range(480, 3361, 480))
    return tuple(actions)


def legal_actions(level, center_x, center_y, allowed_levels, maximum_delta):
    if not math.isfinite(maximum_delta) or maximum_delta < 0:
        raise ValueError('maximum_delta must be finite and nonnegative')
    actions = []
    for action in dict.fromkeys((*action_grid(), (int(level), int(center_x), int(center_y)))):
        target_level, target_x, target_y = action
        if target_level not in allowed_levels or abs(target_level - level) > 1:
            continue
        left, right, top, bottom = center_bounds(target_level)
        if not left <= target_x <= right or not top <= target_y <= bottom:
            continue
        if target_level != 0 and math.hypot(target_x - center_x, target_y - center_y) > maximum_delta:
            continue
        actions.append(action)
    return actions


def region_for(action):
    level, center_x, center_y = action
    width, height = SOURCE_REGION_SIZES[level]
    return (center_x - width // 2, center_y - height // 2,
            center_x + width // 2, center_y + height // 2)


def grid_slice(region):
    left, top, right, bottom = region
    return (slice(max(0, int(top * GRID_HEIGHT / 2160)),
                  min(GRID_HEIGHT, int(np.ceil(bottom * GRID_HEIGHT / 2160)))),
            slice(max(0, int(left * GRID_WIDTH / 3840)),
                  min(GRID_WIDTH, int(np.ceil(right * GRID_WIDTH / 3840)))))


class LearnedCameraPolicy:
    def __init__(self, coefficients):
        coefficients = np.asarray(coefficients, dtype=np.float64)
        if coefficients.shape != (len(FEATURES),) or not np.isfinite(coefficients).all():
            raise ValueError('Invalid learned camera coefficients')
        self.coefficients = coefficients.copy()
        self.reset()

    def reset(self):
        self.ages = np.full((3, GRID_HEIGHT, GRID_WIDTH), AGE_LIMIT, dtype=np.float32)
        self.last_frame = None

    @classmethod
    def from_checkpoint(cls, path, detector_path):
        state = json.loads(Path(path).read_text(encoding='utf-8'))
        if state.get('schema_version') != SCHEMA_VERSION or state.get('features') != list(FEATURES):
            raise ValueError('Camera feature schema mismatch')
        if state.get('detector_sha256') != file_hash(detector_path):
            raise ValueError('Camera checkpoint was trained with different detector weights')
        return cls(state['coefficients'])

    def observe(self, level, center_x, center_y, world):
        frame = world.last_frame
        if frame is None:
            raise ValueError('World must be advanced before camera inference')
        if self.last_frame is not None and frame < self.last_frame:
            self.reset()
        if self.last_frame is not None and frame > self.last_frame:
            steps = frame - self.last_frame
            transform = np.linalg.matrix_power(world.homography, steps)
            scale = np.diag([GRID_WIDTH / 3840, GRID_HEIGHT / 2160, 1.0])
            transform = scale @ transform @ np.linalg.inv(scale)
            self.ages = np.stack([
                cv2.warpPerspective(np.minimum(plane + steps, AGE_LIMIT), transform,
                                    (GRID_WIDTH, GRID_HEIGHT), flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=AGE_LIMIT)
                for plane in self.ages])
        rows, columns = grid_slice(region_for((level, center_x, center_y)))
        self.ages[:level + 1, rows, columns] = 0
        self.last_frame = frame

    def features(self, actions, level, center_x, center_y, maximum_delta, world):
        tracks = []
        for track in world.tracks:
            box = warp_bbox(world.homography, track.bbox)
            total = sum(track.votes.values())
            certainty = max(track.votes.values(), default=0) / total if total > 0 else 0
            tracks.append((box, float(track.score), certainty, track.age_since_seen))
        result = []
        for action in actions:
            target_level, target_x, target_y = action
            region = region_for(action)
            rows, columns = grid_slice(region)
            ages = self.ages[target_level, rows, columns]
            evidence = np.zeros(4, dtype=float)
            factor = (region[2] - region[0]) / 960
            for box, confidence, certainty, age in tracks:
                center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                if not region[0] <= center[0] <= region[2] or not region[1] <= center[1] <= region[3]:
                    continue
                short_side = max(0, min(box[2] - box[0], box[3] - box[1])) / factor
                evidence += confidence * np.array([1, float(short_side < 16),
                                                   1 - certainty, min(age / AGE_LIMIT, 1)])
            evidence = np.log1p(np.maximum(evidence, 0)) / 5
            result.append([
                float(target_level == 0), float(target_level == 1), float(target_level == 2),
                float(action == (level, center_x, center_y)),
                min(1, math.hypot(target_x - center_x, target_y - center_y) / max(1, maximum_delta)),
                abs(target_level - level), float(np.mean(ages >= AGE_LIMIT)),
                float(np.mean(ages) / AGE_LIMIT), 1 - target_y / 2160, *evidence,
            ])
        return np.asarray(result, dtype=np.float64)

    def __call__(self, level, center_x, center_y, allowed_levels, maximum_delta, world=None):
        if world is None:
            return None
        self.observe(level, center_x, center_y, world)
        actions = legal_actions(level, center_x, center_y, allowed_levels, maximum_delta)
        if not actions:
            return None
        features = self.features(actions, level, center_x, center_y, maximum_delta, world)
        return actions[int(np.argmax(features @ self.coefficients))]