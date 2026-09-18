"""Quantify the domain gap between the training scene and the real flight.

Unlabelled. Compares what the detector was trained on (Helsinki source frames,
downsampled to the sampling the detector actually sees) against the views the
competition camera transmitted. If the real scene is dominated by content the
training scene never contains, that is a training-data problem rather than a
threshold problem.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))

from utils import frame_numbers, load_frame  # noqa: E402

RECORDINGS = LAB / 'recordings'


def describe(image):
    """Fractions of water-like, vegetation-like and built-up pixels, plus texture."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0].astype(int), hsv[:, :, 1], hsv[:, :, 2]
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texture = cv2.Laplacian(grey, cv2.CV_32F)
    local = cv2.blur(texture * texture, (17, 17))
    water = ((hue >= 90) & (hue <= 130) & (value < 150) & (local < 120))
    vegetation = (hue >= 30) & (hue <= 85) & (saturation > 60)
    built = (saturation < 55) & (value > 110)
    return {
        'water': float(water.mean()),
        'vegetation': float(vegetation.mean()),
        'built': float(built.mean()),
        'texture': float(np.median(local)),
        'saturation': float(saturation.mean()),
        'value': float(value.mean()),
    }


def summarise(name, samples):
    keys = ('water', 'vegetation', 'built', 'texture', 'saturation', 'value')
    print(f'{name:34s}' + ''.join(f'{np.mean([s[k] for s in samples]):>12.3f}' for k in keys))


def main():
    sequence = RECORDINGS / '14356d0b32754c4f9484a4bbb2d5b25d'

    helsinki = []
    for number in frame_numbers('helsinki'):
        frame = load_frame(number, 'helsinki')
        helsinki.append(describe(cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)))

    real = {0: [], 1: []}
    for path in sorted((sequence / 'meta').glob('*.json')):
        meta = json.loads(path.read_text(encoding='utf-8'))
        image = cv2.imread(str(sequence / 'views' / f'{path.stem}.png'))
        if image is not None:
            real[meta['resolution_level']].append(describe(image))

    header = ('water', 'vegetation', 'built', 'texture', 'saturation', 'value')
    print(f'{"":34s}' + ''.join(f'{h:>12s}' for h in header))
    summarise('helsinki (training scene, L0)', helsinki)
    summarise('competition L0', real[0])
    summarise('competition L1', real[1])

    water = np.array([s['water'] for s in real[0] + real[1]])
    print()
    print(f'competition views that are >25% water : {(water > 0.25).sum()}/{len(water)}')
    print(f'competition views that are >50% water : {(water > 0.50).sum()}/{len(water)}')
    print(f'helsinki frames that are >25% water   : '
          f'{sum(1 for s in helsinki if s["water"] > 0.25)}/{len(helsinki)}')


if __name__ == '__main__':
    main()
