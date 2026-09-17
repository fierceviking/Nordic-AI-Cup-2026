"""Sensitivity of AP to assumed true-positive confidence and injected clutter.

Recorded predictions are unlabelled, not verified false positives. They supply
class, score and size distributions for randomly placed injected boxes. True
boxes have perfect geometry and full recall with an assumed confidence band.
This is a ranking thought experiment, not an estimate or bound on real FP cost.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))

from local_evaluator import oracle_predictions, score  # noqa: E402
from utils import frame_numbers, load_annotations  # noqa: E402

SCENE = 'helsinki'
W, H = 3840, 2160


def observed_false_positives(sequence_dir: Path):
    """Unlabelled detections used only as an injection distribution."""
    rows = []
    for path in sorted((sequence_dir / 'meta').glob('*.json')):
        meta = json.loads(path.read_text())
        for prediction in meta.get('predictions', []):
            x1, y1, x2, y2 = prediction['bbox']
            rows.append((prediction['object_id'],
                         float(prediction['confidence']),
                         max(1.0, (x2 - x1) * W),
                         max(1.0, (y2 - y1) * H)))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', default='5198b3d8558141b2aa989a54da8d00f6')
    parser.add_argument('--rates', default='0,1,2,5,10,20,41')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tp-confidence', type=float, nargs=2,
                        default=(0.60, 0.92), metavar=('LOW', 'HIGH'),
                        help='assumed confidence range for perfect true boxes')
    arguments = parser.parse_args()
    low, high = arguments.tp_confidence
    if not 0 <= low <= high <= 1:
        parser.error('--tp-confidence must satisfy 0 <= LOW <= HIGH <= 1')
    pool = observed_false_positives(LAB / 'recordings' / arguments.sequence)
    if not pool:
        raise SystemExit('no recorded predictions found')

    frames = frame_numbers(SCENE)
    true_count = sum(len(load_annotations(f, SCENE)) for f in frames)
    classes = Counter(c for c, _, _, _ in pool)
    confidences = np.array([c for _, c, _, _ in pool])

    print(f'recorded detections : {len(pool)} over {len(list((LAB / "recordings" / arguments.sequence / "meta").glob("*.json")))} frames')
    print(f'confidence          : median {np.median(confidences):.3f}  '
          f'p90 {np.percentile(confidences, 90):.3f}  max {confidences.max():.3f}')
    print(f'top classes         : {", ".join(f"{k}:{v}" for k, v in classes.most_common(6))}')
    print(f'helsinki objects    : {true_count} over {len(frames)} frames '
          f'({true_count / len(frames):.1f}/frame)\n')

    rates = [int(v) for v in arguments.rates.split(',')]
    if any(rate < 0 for rate in rates):
        parser.error('--rates must be nonnegative')
    print(f'assumed TP confidence: [{low:.2f}, {high:.2f}]')
    print('Sensitivity test only: NOT a measured competition error budget.')
    print(f'{"fp/frame":>9} {"mAP (oracle conf 1.0)":>23} {"mAP (assumed conf)":>22}')
    for rate in rates:
        row = [rate]
        for realistic in (False, True):
            rng = random.Random(arguments.seed)
            confidence_rng = random.Random(arguments.seed + 1)
            predictions = oracle_predictions(SCENE)
            if realistic:
                for frame in predictions:
                    for detection in predictions[frame]:
                        detection['confidence'] = confidence_rng.uniform(low, high)
            for frame in frames:
                for _ in range(rate):
                    object_id, confidence, bw, bh = pool[rng.randrange(len(pool))]
                    x = rng.uniform(0, max(1.0, W - bw))
                    y = rng.uniform(0, max(1.0, H - bh))
                    predictions[frame].append({
                        'object_id': object_id,
                        'bbox': (x, y, x + bw, y + bh),
                        'confidence': confidence,
                    })
            row.append(score(SCENE, predictions)[0])
        print(f'{row[0]:>9} {row[1]:>23.4f} {row[2]:>22.4f}')


if __name__ == '__main__':
    main()
