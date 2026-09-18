"""Can temporal fusion of L0 views substitute for actually visiting L1?

The camera only ever transmits one 960x540 view. At L0 that view covers the
whole frame but is downsampled 4x, so a 30 px object arrives as ~7 px. At L1 it
arrives as ~15 px, but each L1 view covers only a quarter of the frame.

Consecutive L0 views of the same ground differ by a sub-pixel phase shift, so
shift-and-add fusion along the flight homography can in principle reconstruct a
1920x1080 image of the whole frame, which is exactly L1 sampling. Four 960x540
tiles of that reconstruction are pixel-for-pixel the four legal L1 views.

Three-way comparison at a fixed detector:

  l0_upsampled   one L0 view, bilinear 2x        lower bound, no temporal gain
  l0_fused       K L0 views, shift-and-add       the hypothesis
  l1_true        four genuinely captured L1 views  upper bound, unreachable in
                                                   one frame by the real camera

Known risk this is designed to expose: one homography aligns the ground plane.
Objects on rooftops sit at a different height, so parallax will smear exactly
the targets we care about. If fusion ghosts, l0_fused lands near or below
l0_upsampled.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

LAB = Path(__file__).resolve().parent
ROOT = LAB.parent
sys.path.insert(0, str(ROOT))

from dtos import IMAGE_HEIGHT, IMAGE_WIDTH  # noqa: E402
from simulate import render_view  # noqa: E402
from solution import DEFAULT_HOMOGRAPHY, Detector, iou  # noqa: E402
from utils import frame_numbers, load_annotations, load_frame  # noqa: E402

FUSED_SIZE = (1920, 1080)
TILE_ORIGINS = ((0, 0), (960, 0), (0, 540), (960, 540))
VIEW_TO_SOURCE = np.diag([4.0, 4.0, 1.0])
SOURCE_TO_FUSED = np.diag([0.5, 0.5, 1.0])


def tile_region(origin):
    """Source region of a fused tile, which is also a legal L1 view region."""
    x, y = origin
    return (2 * x, 2 * y, 2 * x + 1920, 2 * y + 1080)


def collect_views(frames, scene):
    """One L0 view and the four true L1 views per frame, then drop the frame."""
    level0, level1 = {}, {}
    for number in frames:
        image = load_frame(number, scene)
        level0[number] = render_view(image, 0, IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2)[0]
        level1[number] = [render_view(image, 1, (left + right) // 2, (top + bottom) // 2)[0]
                          for left, top, right, bottom in map(tile_region, TILE_ORIGINS)]
        del image
    return level0, level1


def fuse(views, frames, current, depth):
    """Shift-and-add the last ``depth`` L0 views into the current frame's grid."""
    total = np.zeros((FUSED_SIZE[1], FUSED_SIZE[0], 3), dtype=np.float32)
    weight = np.zeros((FUSED_SIZE[1], FUSED_SIZE[0], 1), dtype=np.float32)
    index = frames.index(current)
    for previous in frames[max(0, index - depth + 1):index + 1]:
        steps = current - previous
        motion = (np.linalg.matrix_power(DEFAULT_HOMOGRAPHY, steps) if steps
                  else np.eye(3))
        transform = SOURCE_TO_FUSED @ motion @ VIEW_TO_SOURCE
        source = views[previous].astype(np.float32)
        total += cv2.warpPerspective(source, transform, FUSED_SIZE,
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_CONSTANT)
        weight += cv2.warpPerspective(np.ones(source.shape[:2], np.float32),
                                      transform, FUSED_SIZE, flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT)[:, :, None]
    return np.clip(total / np.maximum(weight, 1e-3), 0, 255).astype(np.uint8)


def tiles_of(fused):
    return [(fused[y:y + 540, x:x + 960], tile_region((x, y))) for x, y in TILE_ORIGINS]


def match(truth, detections, threshold=0.5):
    """Greedy best-IoU matching, reporting located and correctly named counts."""
    remaining = sorted(detections, key=lambda d: -d['confidence'])
    located = named = 0
    confidences = []
    for target in truth:
        best, score = None, threshold
        for candidate in remaining:
            overlap = iou(candidate['bbox'], target['bbox'])
            if overlap >= score:
                best, score = candidate, overlap
        if best is None:
            continue
        remaining.remove(best)
        located += 1
        confidences.append(best['confidence'])
        named += int(best['object_id'] == target['object_id'])
    return located, named, confidences


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--weights', default=str(ROOT / 'model' / 'v4s1.pt'))
    parser.add_argument('--scene', default='helsinki')
    parser.add_argument('--depth', type=int, default=8, help='L0 views fused')
    parser.add_argument('--conf', type=float, default=0.05)
    parser.add_argument('--preview', type=int, default=None,
                        help='frame number to dump a visual comparison for')
    parser.add_argument('--out', type=Path, default=LAB / 'out' / 'temporal_fusion.json')
    arguments = parser.parse_args()
    if arguments.depth < 1:
        parser.error('--depth must be at least 1')

    frames = frame_numbers(arguments.scene)
    print(f'loading {len(frames)} frames of {arguments.scene}', flush=True)
    level0, level1 = collect_views(frames, arguments.scene)

    detector = Detector(weights=arguments.weights, imgsz=960,
                        confidence=arguments.conf, device='0')
    methods = ('l0_upsampled', 'l0_fused', 'l1_true')
    totals = {name: {'located': 0, 'named': 0, 'detections': 0, 'confidences': []}
              for name in methods}
    instances = 0
    try:
        for number in frames:
            truth = load_annotations(number, arguments.scene)
            instances += len(truth)
            single = fuse(level0, frames, number, 1)
            fused = fuse(level0, frames, number, arguments.depth)
            views = {
                'l0_upsampled': tiles_of(single),
                'l0_fused': tiles_of(fused),
                'l1_true': list(zip(level1[number], map(tile_region, TILE_ORIGINS))),
            }
            line = [f'frame {number:3d} ({len(truth):2d} gt)']
            for name in methods:
                detections = [d for view, region in views[name]
                              for d in detector(view, region)]
                located, named, confidences = match(truth, detections)
                totals[name]['located'] += located
                totals[name]['named'] += named
                totals[name]['detections'] += len(detections)
                totals[name]['confidences'].extend(confidences)
                line.append(f'{name} {located}/{named}')
            print('  '.join(line), flush=True)

            if arguments.preview == number:
                strip = np.hstack([single[:540, :960], fused[:540, :960],
                                   level1[number][0]])
                path = arguments.out.with_name(f'fusion_frame{number}.png')
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), strip)
                print(f'preview (upsampled | fused | true L1): {path}', flush=True)
    finally:
        detector._pool.shutdown(wait=True)

    summary = {'scene': arguments.scene, 'weights': arguments.weights,
               'depth': arguments.depth, 'confidence': arguments.conf,
               'frames': len(frames), 'instances': instances, 'methods': {}}
    print(f'\n{"method":14s} {"located":>9s} {"named":>9s} {"dets/frame":>11s} {"med conf":>9s}')
    for name in methods:
        record = totals[name]
        confidences = record.pop('confidences')
        record['located_recall'] = record['located'] / max(1, instances)
        record['named_recall'] = record['named'] / max(1, instances)
        record['detections_per_frame'] = record['detections'] / len(frames)
        record['median_matched_confidence'] = float(np.median(confidences)) if confidences else 0.0
        summary['methods'][name] = record
        print(f'{name:14s} {record["located_recall"]:9.3f} {record["named_recall"]:9.3f} '
              f'{record["detections_per_frame"]:11.1f} {record["median_matched_confidence"]:9.3f}')

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'\nwrote {arguments.out}')
    print('Recall here is class-agnostic localisation and same-class naming over '
          'the supplied scene only. It is not AP and not a transfer estimate.')


if __name__ == '__main__':
    main()
