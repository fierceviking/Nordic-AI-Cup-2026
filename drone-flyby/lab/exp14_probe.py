"""Probe: does a real, human-verified asset match its Helsinki reference?

exp13 found retrieval rejects almost everything in the real views. Either the
assets do not match their references across scenes, or the scan missed them.
This takes two objects identified by eye in the recorded views - a helicopter
and the hangar - searches a window around each, and reports what the reference
bank says. A decisive small test of cross-scene appearance matching.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))

from exp12_separability import Encoder, at_sampling, crop  # noqa: E402
from exp13_retrieval import CONTEXT, build_bank  # noqa: E402


def probe(encoder, reference, labels, image, region, sizes, divisor, stride=3):
    """Best matches over a small search region, reported with class and score."""
    left, top, right, bottom = region
    results = []
    for window in sizes:
        half = window // 2
        boxes, patches = [], []
        for y in range(top + half, bottom - half, stride):
            for x in range(left + half, right - half, stride):
                patch = crop(image, [x - half, y - half, x + half, y + half], CONTEXT)
                if patch is not None:
                    patches.append(at_sampling(patch, divisor))
                    boxes.append((x, y, window))
        if not patches:
            continue
        scores = np.concatenate([encoder(patches[i:i + 512]) @ reference.T
                                 for i in range(0, len(patches), 512)])
        best = scores.max(axis=1)
        index = int(best.argmax())
        results.append((float(best[index]), labels[int(scores[index].argmax())],
                        boxes[index]))
    return sorted(results, reverse=True)


def main():
    import torch
    encoder = Encoder('cuda' if torch.cuda.is_available() else 'cpu')
    out = LAB / 'out'

    for name, filename, region, sizes, divisor, expected in (
            ('helicopter (L1 view)', 'scene_L1.png', (180, 0, 300, 100),
             (20, 30, 40, 55, 70), 2, 'helicopter'),
            ('hangar cluster (L0 view)', 'scene_L0.png', (630, 185, 745, 285),
             (8, 12, 18, 26, 36), 1, 'hangar'),
    ):
        image = cv2.imread(str(out / filename))
        if image is None:
            print(f'{name}: missing {filename}')
            continue
        reference, labels, _ = build_bank(encoder, divisor)
        found = probe(encoder, reference, labels, image, region, sizes, divisor)
        print(f'\n{name}  (expected {expected})')
        for score, label, (x, y, window) in found:
            mark = '  <-- correct' if label == expected else ''
            print(f'   window {window:3d}px at ({x:4d},{y:4d})   '
                  f'{label:16s} {score:.3f}{mark}')


if __name__ == '__main__':
    main()
