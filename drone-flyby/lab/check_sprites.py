"""How clean are the extracted sprites, and are they usable on new backgrounds?

Compositing Helsinki sprites onto foreign backgrounds only helps if the cut-out
is tight. A loose mask carries a ring of Helsinki ground with it, and that ring
becomes a shortcut feature that will not exist in the competition scene, where
the objects are rendered into the image rather than pasted.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

LAB = Path(__file__).resolve().parent
SPRITES = LAB / 'out' / 'sprites'


def main():
    if not SPRITES.is_dir():
        raise SystemExit(f'no sprites under {SPRITES}')
    print(f'{"class":16s} {"n":>3s} {"has alpha":>10s} {"coverage":>9s} '
          f'{"border on":>10s} {"median px":>10s}')
    total = 0
    for folder in sorted(p for p in SPRITES.iterdir() if p.is_dir()):
        files = sorted(folder.glob('*.png'))
        total += len(files)
        alpha_count, coverage, border, sides = 0, [], [], []
        for path in files:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                continue
            sides.append(min(image.shape[:2]))
            if image.ndim == 3 and image.shape[2] == 4:
                alpha_count += 1
                mask = image[:, :, 3] > 127
                coverage.append(float(mask.mean()))
                edge = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
                border.append(float(edge.mean()))
        print(f'{folder.name:16s} {len(files):3d} {alpha_count:10d} '
              f'{np.mean(coverage) if coverage else float("nan"):9.3f} '
              f'{np.mean(border) if border else float("nan"):10.3f} '
              f'{np.median(sides) if sides else 0:10.0f}')
    print(f'\ntotal sprites {total}')
    print('coverage  = share of the sprite box the mask keeps (a tight cut-out '
          'of a compact object sits near 0.4-0.7)')
    print('border on = share of the sprite border still marked object; well '
          'above 0 means the mask runs off the edge and takes ground with it')


if __name__ == '__main__':
    main()
