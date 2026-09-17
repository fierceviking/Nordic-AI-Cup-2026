"""Turn one-frame labels into per-frame ground truth, and score against it.

An object is labelled once, in source pixels, in a chosen reference frame. The
flight is a single constant homography, so the same object in frame k is that
box warped by H^(k - reference).

Note H is *not* a constant pixel shift: it carries a scale term, so points near
the top of the frame move less than points near the bottom. Approximating it
with an average step drifts several pixels per frame, which at these object
sizes destroys IoU and makes the benchmark measure the labeller.

The scorer answers the question the competition will not: of the boxes we
actually sent, how many landed on a real object, and which classes did we find?
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))

from solution import warp_bbox  # noqa: E402

SOURCE_W, SOURCE_H = 3840, 2160
VIEW_W, VIEW_H = 960, 540
SCALE = SOURCE_W / VIEW_W                      # source px per Level-0 view px


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def truth_for_frame(labels: dict, frame_index: int) -> list:
    """Ground-truth boxes in source pixels for one frame."""
    homography = np.array(labels['homography'], float)
    out = []
    for item in labels['objects']:
        steps = frame_index - item['ref_frame']
        box = list(item['bbox'])
        if steps:
            matrix = np.linalg.matrix_power(homography, abs(steps))
            if steps < 0:
                matrix = np.linalg.inv(matrix)
            box = warp_bbox(matrix, box)
        # Drop it once it has left the frame entirely.
        if box[2] < 0 or box[0] > SOURCE_W or box[3] < 0 or box[1] > SOURCE_H:
            continue
        out.append({'object_id': item['object_id'], 'bbox': list(box)})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', required=True)
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--iou', type=float, default=0.50)
    parser.add_argument('--dump', default='', help='write per-frame truth here')
    parser.add_argument('--replay', default='',
                        help='score a replay_recording.py file instead of the '
                             'predictions the server actually sent')
    arguments = parser.parse_args()

    labels = json.loads(Path(arguments.labels).read_text())
    replay = (json.loads(Path(arguments.replay).read_text())
              if arguments.replay else None)
    directory = LAB / 'recordings' / arguments.sequence / 'meta'
    metas = sorted(directory.glob('*.json'))
    if not metas:
        raise SystemExit(f'no frames under {directory}')

    truth_total = 0
    matched = defaultdict(int)
    present = defaultdict(int)
    best_iou = defaultdict(float)
    hit_conf = defaultdict(list)
    # Class-agnostic: did any box land on the object, and what did we call it?
    found_any = defaultdict(int)
    agnostic_iou = defaultdict(float)
    called_it = defaultdict(Counter)
    dumped = {}

    for path in metas:
        meta = json.loads(path.read_text())
        truth = truth_for_frame(labels, meta['frame_index'])
        dumped[meta['frame_index']] = truth
        truth_total += len(truth)
        if replay is not None:
            predictions = replay.get(str(meta['frame_index']), [])
        else:
            predictions = meta.get('predictions', [])
        boxes = []
        for p in predictions:
            x1, y1, x2, y2 = p['bbox']
            boxes.append((p['object_id'],
                          [x1 * SOURCE_W, y1 * SOURCE_H, x2 * SOURCE_W, y2 * SOURCE_H],
                          float(p['confidence'])))
        for item in truth:
            present[item['object_id']] += 1
            best = 0.0
            best_conf = 0.0
            best_any = 0.0
            # Alternate-class emission puts several labels on one box, so
            # "closest box" is decided arbitrarily between identical geometry.
            # Rank by confidence among boxes that clear the threshold, which is
            # what AP orders by.
            winner = None
            winner_conf = -1.0
            for name, box, confidence in boxes:
                value = iou(item['bbox'], box)
                if value > best_any:
                    best_any = value
                if value >= arguments.iou and confidence > winner_conf:
                    winner, winner_conf = name, confidence
                if name != item['object_id']:
                    continue
                if value > best:
                    best, best_conf = value, confidence
            best_iou[item['object_id']] = max(best_iou[item['object_id']], best)
            agnostic_iou[item['object_id']] = max(agnostic_iou[item['object_id']],
                                                  best_any)
            if best >= arguments.iou:
                matched[item['object_id']] += 1
                hit_conf[item['object_id']].append(best_conf)
            if winner is not None:
                found_any[item['object_id']] += 1
                called_it[item['object_id']][winner] += 1

    print(f'{"class":16s} {"frames":>7s} {"hit":>6s} {"recall":>7s} '
          f'{"bestIoU":>8s} {"meanConf":>9s}')
    total_hits = 0
    for name in sorted(present):
        hits = matched.get(name, 0)
        total_hits += hits
        confidences = hit_conf.get(name, [])
        print(f'{name:16s} {present[name]:7d} {hits:6d} '
              f'{hits / present[name]:7.3f} {best_iou[name]:8.3f} '
              f'{np.mean(confidences) if confidences else 0.0:9.3f}')
    print(f'\nlabelled instances {truth_total}, matched at IoU>={arguments.iou}: '
          f'{total_hits} ({total_hits / max(1, truth_total):.3f})')
    print(f'classes ever hit   {sum(1 for n in present if matched.get(n))} '
          f'of {len(present)}')

    print('\n--- ignoring the label: did any box land on the object? ---')
    print(f'{"class":16s} {"anyhit":>7s} {"recall":>7s} {"bestIoU":>8s}  called it')
    for name in sorted(present):
        hits = found_any.get(name, 0)
        names = ', '.join(f'{k}:{v}' for k, v in called_it[name].most_common(3))
        print(f'{name:16s} {hits:7d} {hits / present[name]:7.3f} '
              f'{agnostic_iou[name]:8.3f}  {names or "-"}')

    if arguments.dump:
        Path(arguments.dump).write_text(json.dumps(
            {str(k): v for k, v in dumped.items()}))
        print(f'wrote per-frame truth to {arguments.dump}')


if __name__ == '__main__':
    main()
