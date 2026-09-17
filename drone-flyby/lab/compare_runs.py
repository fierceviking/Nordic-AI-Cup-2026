"""Compare what the solver actually reported across two recorded attempts.

The competition returns one number, so the only way to see *why* a score moved
is to look at what we sent. Confidence distribution matters as much as count:
COCO AP ranks predictions against each other, so a detection below every true
positive is nearly free, while a confident wrong one is not.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent


def summarise(sequence: str, label: str) -> None:
    directory = LAB / 'recordings' / sequence / 'meta'
    files = sorted(directory.glob('*.json'))
    if not files:
        print(f'--- {label}: no frames at {directory}')
        return

    rows, levels = [], Counter()
    for path in files:
        meta = json.loads(path.read_text())
        levels[meta['resolution_level']] += 1
        for prediction in meta.get('predictions', []):
            rows.append((prediction['object_id'], float(prediction['confidence'])))

    confidence = np.array([c for _, c in rows]) if rows else np.zeros(1)
    classes = Counter(name for name, _ in rows)
    print(f'--- {label} ---')
    print(f'  frames      {len(files)}   levels {dict(sorted(levels.items()))}')
    print(f'  detections  {len(rows)}  ({len(rows) / len(files):.1f} per frame)')
    print(f'  confidence  median {np.median(confidence):.3f}   '
          f'p90 {np.percentile(confidence, 90):.3f}   max {confidence.max():.3f}')
    print(f'  above 0.5   {(confidence > 0.5).sum():>6}')
    print(f'  above 0.7   {(confidence > 0.7).sum():>6}')
    print(f'  above 0.9   {(confidence > 0.9).sum():>6}')
    print(f'  classes     {len(classes)} distinct')
    print(f'  most common {", ".join(f"{k}:{v}" for k, v in classes.most_common(5))}')
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+',
                        help='sequence_id=label pairs, or bare sequence ids')
    arguments = parser.parse_args()
    for entry in arguments.runs:
        sequence, _, label = entry.partition('=')
        summarise(sequence, label or sequence[:8])


if __name__ == '__main__':
    main()
