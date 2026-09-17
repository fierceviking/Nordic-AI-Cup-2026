"""Measure the failure that actually cost the validation run: invented objects.

mAP on the real-frame validation set cannot see this. All 125 of those views
contain targets, so precision is only ever measured where something really
exists. What broke the competition run was the detector firing on empty ground
and on man-made clutter that is not a target - 93 predictions per frame where
about ten were real.

Two measurements, both of which any checkpoint can be scored on without
spending a validation attempt:

  **empty Helsinki crops** - views built from the supplied frames at offsets
  where no annotation overlaps. The true count is exactly zero, so every
  detection is a false positive and the number is exact.

  **the recorded validation sequence** - held out, different location, full of
  boats and cars. There is no ground truth, but the scene contains on the order
  of ten targets per frame, so the raw detection count is a direct proxy for
  how badly the detector is inventing things. Used for measurement only; it is
  never a training input.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))
sys.path.insert(0, str(LAB))

from dtos import SOURCE_REGION_SIZES  # noqa: E402
from simulate import render_view  # noqa: E402
from solution import Detector  # noqa: E402
from utils import frame_numbers, load_annotations, load_frame  # noqa: E402

SCENE = 'helsinki'
RECORDINGS = LAB / 'recordings'


def empty_views(count: int, seed: int = 7):
    """Views from the supplied frames containing no annotated object at all."""
    rng = random.Random(seed)
    numbers = frame_numbers(SCENE)
    frames = {f: load_frame(f, SCENE) for f in numbers}
    annotations = {f: load_annotations(f, SCENE) for f in numbers}
    out = []
    attempts = 0
    while len(out) < count and attempts < count * 200:
        attempts += 1
        # Level 0 is the whole frame and always contains objects.
        level = 1 if rng.random() < 0.25 else 2
        rw, rh = SOURCE_REGION_SIZES[level]
        frame = rng.choice(numbers)
        cx = rng.randint(rw // 2, 3840 - rw // 2)
        cy = rng.randint(rh // 2, 2160 - rh // 2)
        x1, y1 = cx - rw // 2, cy - rh // 2
        x2, y2 = x1 + rw, y1 + rh
        clear = True
        for annotation in annotations[frame]:
            bx1, by1, bx2, by2 = annotation['bbox']
            if not (bx2 < x1 or bx1 > x2 or by2 < y1 or by1 > y2):
                clear = False
                break
        if not clear:
            continue
        view, region = render_view(frames[frame], level, cx, cy)
        out.append((view, region))
    return out


def recorded_views(limit: int = 0):
    out = []
    if not RECORDINGS.is_dir():
        return out
    for sequence in sorted(p for p in RECORDINGS.iterdir() if p.is_dir()):
        meta_dir, view_dir = sequence / 'meta', sequence / 'views'
        if not meta_dir.is_dir():
            continue
        for meta_path in sorted(meta_dir.glob('*.json')):
            view_path = view_dir / (meta_path.stem + '.png')
            if not view_path.exists():
                continue
            with open(meta_path) as handle:
                meta = json.load(handle)
            out.append((view_path, meta['source_region_xyxy']))
            if limit and len(out) >= limit:
                return out
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--empty', type=int, default=150)
    parser.add_argument('--recorded', type=int, default=60)
    parser.add_argument('--confs', default='0.05,0.15,0.25,0.40')
    arguments = parser.parse_args()

    confs = [float(v) for v in arguments.confs.split(',')]
    label = Path(arguments.weights).stem

    print(f'=== {label} ===')
    print('\nfalse positives on object-free Helsinki crops '
          f'({arguments.empty} views, true count = 0)')
    views = empty_views(arguments.empty)
    print(f'  {"conf":>6s} {"FP/view":>9s} {"views clean":>12s} {"worst":>7s}')
    for conf in confs:
        detector = Detector(weights=arguments.weights, imgsz=960,
                            confidence=conf, drop_view_edge=False)
        counts = [len(detector(view, region)) for view, region in views]
        counts = np.array(counts)
        print(f'  {conf:6.2f} {counts.mean():9.2f} '
              f'{float((counts == 0).mean()) * 100:11.0f}% {counts.max():7d}')

    recorded = recorded_views(arguments.recorded)
    if recorded:
        print(f'\ndetections on the recorded validation sequence '
              f'({len(recorded)} frames, held out, ~10 real targets per frame)')
        print(f'  {"conf":>6s} {"det/frame":>11s} {"worst":>7s}')
        for conf in confs:
            detector = Detector(weights=arguments.weights, imgsz=960,
                                confidence=conf, drop_view_edge=False)
            counts = []
            for path, region in recorded:
                image = cv2.imread(str(path))
                if image is None:
                    continue
                counts.append(len(detector(image, region)))
            counts = np.array(counts)
            print(f'  {conf:6.2f} {counts.mean():11.2f} {counts.max():7d}')


if __name__ == '__main__':
    main()
