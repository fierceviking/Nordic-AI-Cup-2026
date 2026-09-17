"""Which classes do we ever actually find, and how sure are we?

The score is a mean over classes, so a class we never detect confidently is a
flat zero no matter what else happens. Object sizes come from the Helsinki
annotations, which is the same asset set, so the table shows detection quality
against physical size - the relationship that decides whether more zoom would
help.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))

from utils import frame_numbers, load_annotations  # noqa: E402

METRES_PER_PIXEL = 13.8888889 / 66.86


def object_sizes():
    sizes = defaultdict(list)
    for frame in frame_numbers('helsinki'):
        for annotation in load_annotations(frame, 'helsinki'):
            x1, y1, x2, y2 = annotation['bbox']
            sizes[annotation['object_id']].append(max(x2 - x1, y2 - y1))
    return {k: float(np.median(v)) for k, v in sizes.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', required=True)
    arguments = parser.parse_args()

    directory = LAB / 'recordings' / arguments.sequence / 'meta'
    best = defaultdict(float)
    counts = Counter()
    strong = Counter()
    for path in sorted(directory.glob('*.json')):
        meta = json.loads(path.read_text())
        for prediction in meta.get('predictions', []):
            name = prediction['object_id']
            confidence = float(prediction['confidence'])
            counts[name] += 1
            best[name] = max(best[name], confidence)
            if confidence >= 0.60:
                strong[name] += 1

    sizes = object_sizes()
    print(f'{"class":16s} {"src px":>7s} {"at L0":>6s} {"metres":>7s} '
          f'{"max conf":>9s} {"n>=0.6":>7s} {"total":>7s}')
    for name in sorted(sizes, key=lambda k: -sizes[k]):
        px = sizes[name]
        print(f'{name:16s} {px:7.0f} {px / 4:6.1f} {px * METRES_PER_PIXEL:7.1f} '
              f'{best.get(name, 0.0):9.3f} {strong.get(name, 0):7d} '
              f'{counts.get(name, 0):7d}')

    found = [n for n in sizes if best.get(n, 0) >= 0.60]
    print(f'\nclasses with any detection >= 0.60 : {len(found)} of {len(sizes)}')
    if found:
        print(f'  smallest such class: '
              f'{min(found, key=lambda n: sizes[n])} '
              f'({sizes[min(found, key=lambda n: sizes[n])] * METRES_PER_PIXEL:.1f} m)')


if __name__ == '__main__':
    main()
