"""Compare the label size distributions of two generated datasets.

v2 added clipped objects to the labels, which was meant to fix the classes that
enter the frame at an edge. If it also created a population of two- and
three-pixel boxes, that would explain a regression on the smallest classes:
the detector is being taught to fire on fragments that carry no information.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))

from dtos import OBJECT_CLASSES  # noqa: E402

VIEW_W, VIEW_H = 960, 540


def collect(dataset: Path, split: str = 'train'):
    per_class = defaultdict(list)
    for path in (dataset / 'labels' / split).glob('*.txt'):
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            index = int(parts[0])
            w = float(parts[3]) * VIEW_W
            h = float(parts[4]) * VIEW_H
            per_class[OBJECT_CLASSES[index]].append((w, h))
    return per_class


def summarise(name: str, per_class):
    every = np.array([wh for boxes in per_class.values() for wh in boxes])
    shorter = np.minimum(every[:, 0], every[:, 1])
    print(f'\n=== {name} ===')
    print(f'  boxes {len(every)}')
    for threshold in (2, 4, 6, 8, 12):
        share = float((shorter < threshold).mean())
        print(f'  shorter side < {threshold:2d} px: {share * 100:5.2f}%  '
              f'({int((shorter < threshold).sum())} boxes)')
    return shorter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', required=True)
    parser.add_argument('--b', required=True)
    arguments = parser.parse_args()

    a = collect(Path(arguments.a))
    b = collect(Path(arguments.b))
    summarise(arguments.a, a)
    summarise(arguments.b, b)

    print(f'\n=== per class: share of boxes with shorter side < 6 px ===')
    print(f'  {"class":16s} {"A":>8s} {"B":>8s}   {"A median":>9s} {"B median":>9s}')
    for name in OBJECT_CLASSES:
        va = np.array(a.get(name, [[0, 0]]))
        vb = np.array(b.get(name, [[0, 0]]))
        sa = np.minimum(va[:, 0], va[:, 1])
        sb = np.minimum(vb[:, 0], vb[:, 1])
        print(f'  {name:16s} {float((sa < 6).mean()) * 100:7.2f}% '
              f'{float((sb < 6).mean()) * 100:7.2f}%   '
              f'{np.median(sa):9.1f} {np.median(sb):9.1f}')


if __name__ == '__main__':
    main()
