"""Stitch a recorded flight into one ground-fixed strip.

The camera translates at a constant rate, so every Level-0 view is the same
ground shifted by a known offset. Undoing that shift lays the whole flight out
as a single image: every object in the scene appears exactly once, which is
what labelling needs. Level-0 views are a quarter of source resolution, so the
strip is for *locating* objects; identification uses the best view covering
each spot.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))

SOURCE_W, SOURCE_H = 3840, 2160


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--dy', type=float, default=None,
                        help='source px per frame; read from h_<seq>.npy if absent')
    parser.add_argument('--scale', type=float, default=1.0)
    arguments = parser.parse_args()

    root = LAB / 'recordings' / arguments.sequence
    metas = sorted((root / 'meta').glob('*.json'))
    if not metas:
        raise SystemExit(f'no frames under {root}')

    dy = arguments.dy
    if dy is None:
        path = LAB / 'out' / f'h_{arguments.sequence[:8]}.npy'
        if not path.exists():
            raise SystemExit(f'need --dy or {path}')
        homography = np.load(path)
        dy = float(homography[1, 2])
    print(f'ground motion {dy:.2f} source px/frame')

    views = []
    for path in metas:
        meta = json.loads(path.read_text())
        if meta.get('resolution_level') != 0:
            continue
        image_path = root / 'views' / (path.stem + '.png')
        if not image_path.exists():
            image_path = root / (path.stem + '.png')
        if not image_path.exists():
            continue
        views.append((meta['frame_index'], image_path))
    if not views:
        raise SystemExit('no Level-0 views found')
    views.sort()
    print(f'{len(views)} Level-0 views')

    probe = cv2.imread(str(views[0][1]))
    vh, vw = probe.shape[:2]
    # A Level-0 view is the whole frame, so view pixels are source/4.
    step = dy * vh / SOURCE_H
    span = int(abs(step) * (views[-1][0] - views[0][0])) + vh + 4
    canvas = np.zeros((span, vw, 3), np.uint8)
    written = np.zeros(span, bool)

    first = views[0][0]
    last = views[-1][0]
    for index, image_path in views:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        # Ground moves down the frame, so new terrain enters at the top and
        # later frames belong higher up the strip.
        top = int(round((last - index) * step))
        top = max(0, min(span - vh, top))
        rows = slice(top, top + vh)
        fresh = ~written[rows]
        if fresh.any():
            canvas[rows][fresh] = image[fresh]
            written[rows] = True

    if arguments.scale != 1.0:
        canvas = cv2.resize(canvas, None, fx=arguments.scale, fy=arguments.scale)
    out = LAB / 'out' / f'strip_{arguments.sequence[:8]}.jpg'
    cv2.imwrite(str(out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f'wrote {out}  {canvas.shape[1]}x{canvas.shape[0]}')
    print(f'strip row = view_row + ({last} - frame_index) * {step:.3f}')


if __name__ == '__main__':
    main()
