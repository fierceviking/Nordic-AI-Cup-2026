"""Bound studies: how much does the camera policy cost a *perfect* detector?

Nothing here is a real solution. Each run replaces the detector with the ground
truth, restricted to whatever the camera could actually see, so the resulting
mAP is the ceiling for that camera policy. Comparing the ceilings says where the
effort belongs: in the detector, in the camera policy, or in the world model.

Runs:
  oracle-full          ground truth everywhere                       (scorer check)
  L0-static            camera parked at L0, perfect detection in view
  L1-*                 L1 patrol patterns, perfect detection in view
  L2-*                 L2 patrol patterns, perfect detection in view
  *-memory             the same, but detections persist and are warped
                       forward by the constant homography
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulate import FrameCache, replay, report  # noqa: E402
from utils import load_annotations  # noqa: E402

OUT = Path(__file__).resolve().parent / 'out'
H_MEAN = np.load(OUT / 'h_mean.npy')
SCENE = 'helsinki'


def warp_box(h, bbox):
    x1, y1, x2, y2 = bbox
    pts = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], np.float64)
    import cv2
    w = cv2.perspectiveTransform(pts, h).reshape(-1, 2)
    return [float(w[:, 0].min()), float(w[:, 1].min()),
            float(w[:, 0].max()), float(w[:, 1].max())]


def clip(bbox):
    x1, y1, x2, y2 = bbox
    x1, x2 = max(0.0, min(3840.0, x1)), max(0.0, min(3840.0, x2))
    y1, y2 = max(0.0, min(2160.0, y1)), max(0.0, min(2160.0, y2))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return [x1, y1, x2, y2]


# --------------------------------------------------------------------------- #
# Camera policies
# --------------------------------------------------------------------------- #

class StaticPolicy:
    """Never move."""

    def __init__(self, level=0):
        self.level = level

    def __call__(self, request):
        if request.resolution_level == self.level:
            return None
        if self.level in request.allowed_levels:
            if self.level == 0:
                return (0, 1920, 1080)
            return (self.level, 1920, 1080)
        return None


class RasterPolicy:
    """Step through a fixed list of centres, one legal hop at a time.

    The movement limit means a target can need several frames to reach, so the
    policy walks toward the current target and only advances the target once it
    has arrived.
    """

    def __init__(self, level: int, centers: List[Tuple[int, int]]):
        self.level = level
        self.centers = centers
        self.index = 0

    def __call__(self, request):
        level = self.level
        current_level = request.resolution_level
        # Climb one level per frame until we are at the working level. The
        # target centre has to be clamped into the *intermediate* level's
        # bounds, or the command is refused and the camera never gets there.
        if current_level != level:
            step = level if abs(level - current_level) <= 1 else current_level + (
                1 if level > current_level else -1)
            if step not in request.allowed_levels:
                return None
            if step == 0:
                return (0, 1920, 1080)
            tx, ty = self.centers[self.index]
            tx, ty = clamp_center(step, tx, ty)
            current = np.array([request.center_x, request.center_y], float)
            delta = np.array([tx, ty], float) - current
            distance = float(np.hypot(*delta))
            limit = request.maximum_center_delta * 0.98
            if distance > limit:
                tx, ty = current + delta / distance * limit
            return (step, int(round(tx)), int(round(ty)))

        target = np.array(self.centers[self.index], float)
        current = np.array([request.center_x, request.center_y], float)
        delta = target - current
        distance = float(np.hypot(*delta))
        limit = request.maximum_center_delta * 0.98
        if distance <= 2.0:
            self.index = (self.index + 1) % len(self.centers)
            target = np.array(self.centers[self.index], float)
            delta = target - current
            distance = float(np.hypot(*delta))
        if distance > limit:
            target = current + delta / distance * limit
        cx, cy = clamp_center(level, target[0], target[1])
        return (level, int(round(cx)), int(round(cy)))


def clamp_center(level: int, cx: float, cy: float) -> Tuple[float, float]:
    from utils import center_bounds_for_level
    min_x, max_x, min_y, max_y = center_bounds_for_level(level)
    return (min(max(cx, min_x), max_x), min(max(cy, min_y), max_y))


def top_band_centers(level: int) -> List[Tuple[int, int]]:
    """Centres that sweep the band where new objects enter the frame."""
    if level == 1:
        return [(960, 540), (1920, 540), (2880, 540), (1920, 540)]
    return [(480, 270), (1440, 270), (2400, 270), (3360, 270),
            (2400, 270), (1440, 270)]


def full_raster_centers(level: int) -> List[Tuple[int, int]]:
    if level == 1:
        return [(960, 540), (2880, 540), (2880, 1620), (960, 1620)]
    return [(480, 270), (1440, 270), (2400, 270), (3360, 270),
            (3360, 810), (2400, 810), (1440, 810), (480, 810),
            (480, 1350), (1440, 1350), (2400, 1350), (3360, 1350),
            (3360, 1890), (2400, 1890), (1440, 1890), (480, 1890)]


# --------------------------------------------------------------------------- #
# Oracle detector, restricted to what the camera can see
# --------------------------------------------------------------------------- #

class OracleSolver:
    """Perfect detection inside the current view. Optionally with memory.

    ``min_transmitted_px`` models what a detector can actually resolve: an
    object is only "detected" if its shorter side is at least this many pixels
    in the 960x540 image that was transmitted. Set it to 0 for a detector with
    no resolution limit at all.
    """

    def __init__(self, policy, memory: bool = False, min_visible: float = 0.6,
                 min_transmitted_px: float = 0.0):
        self.policy = policy
        self.memory = memory
        self.min_visible = min_visible
        self.min_transmitted_px = min_transmitted_px
        self.tracks: Dict[str, dict] = {}

    def __call__(self, request):
        x1, y1, x2, y2 = request.source_region_xyxy
        # source pixels per transmitted pixel: 4 at L0, 2 at L1, 1 at L2
        scale = (x2 - x1) / 960.0

        # 1. age the world model into this frame
        if self.memory:
            for track in self.tracks.values():
                track['bbox'] = warp_box(H_MEAN, track['bbox'])
                track['age'] += 1

        # 2. "detect": ground truth whose box is mostly inside the view
        for annotation in load_annotations(request.frame, SCENE):
            bx1, by1, bx2, by2 = annotation['bbox']
            ix1, iy1 = max(bx1, x1), max(by1, y1)
            ix2, iy2 = min(bx2, x2), min(by2, y2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area = max(1, (bx2 - bx1) * (by2 - by1))
            if inter / area < self.min_visible:
                continue
            if min(bx2 - bx1, by2 - by1) / scale < self.min_transmitted_px:
                continue
            self.tracks[annotation['object_id']] = {
                'object_id': annotation['object_id'],
                'bbox': [float(bx1), float(by1), float(bx2), float(by2)],
                'age': 0,
            }

        # 3. answer for the whole frame
        annotations = []
        for track in list(self.tracks.values()):
            bbox = clip(track['bbox'])
            if bbox is None or track['age'] > 40:
                self.tracks.pop(track['object_id'], None)
                continue
            annotations.append({
                'object_id': track['object_id'],
                'bbox': bbox,
                'confidence': float(max(0.05, 0.99 - 0.01 * track['age'])),
            })
        if not self.memory:
            self.tracks.clear()

        return annotations, self.policy(request)


def oracle_predictions():
    return {
        frame: [
            {'object_id': a['object_id'],
             'bbox': [float(c) for c in a['bbox']],
             'confidence': 1.0}
            for a in load_annotations(frame, SCENE)
        ]
        for frame in __import__('utils').frame_numbers(SCENE)
    }


def main():
    cache = FrameCache(SCENE)
    results = {}

    results['oracle-full (scorer check)'] = (
        report('oracle-full (scorer check)', oracle_predictions()), 0)

    # min_px = 0  : a detector with no resolution limit
    # min_px = 12 : a realistic small-object detector, needs >=12 px in the
    #               transmitted 960x540 image
    experiments = []
    for min_px in (0.0, 12.0):
        tag = 'ideal' if min_px == 0 else f'>={min_px:.0f}px'
        experiments += [
            (f'L0 static, no memory [{tag}]', StaticPolicy(0), False, min_px),
            (f'L0 static, memory [{tag}]', StaticPolicy(0), True, min_px),
            (f'L1 top-band patrol, no memory [{tag}]',
             RasterPolicy(1, top_band_centers(1)), False, min_px),
            (f'L1 top-band patrol, memory [{tag}]',
             RasterPolicy(1, top_band_centers(1)), True, min_px),
            (f'L1 full raster, memory [{tag}]',
             RasterPolicy(1, full_raster_centers(1)), True, min_px),
            (f'L2 top-band patrol, memory [{tag}]',
             RasterPolicy(2, top_band_centers(2)), True, min_px),
            (f'L2 full raster, memory [{tag}]',
             RasterPolicy(2, full_raster_centers(2)), True, min_px),
        ]

    for name, policy, memory, min_px in experiments:
        solver = OracleSolver(policy, memory=memory, min_transmitted_px=min_px)
        predictions, stats = replay(solver, SCENE, cache)
        results[name] = (report(name, predictions, stats), stats['refused'])

    print('\n\n=== summary ===')
    for name, (value, refused) in results.items():
        note = f'  ({refused} camera refusals)' if refused else ''
        print(f'  {value:.4f}  {name}{note}')


if __name__ == '__main__':
    main()
