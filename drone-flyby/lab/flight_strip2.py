"""Stitch a recorded flight into one ground-fixed image, at any scale.

Every view carries the source region it came from, and the flight is a constant
translation, so a view's ground position is fully determined:

    ground_y = source_y + (last_index - frame_index) * dy

Level-1 views are half source resolution against Level-0's quarter, so they are
composited last and overwrite the coarser pixels. That matters for labelling:
at Level-0 scale a 30 px object is 7 px and invisible, which is why the first
strip only revealed the hangar.
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
    parser.add_argument('--dy', type=float, default=68.19)
    parser.add_argument('--scale', type=float, default=0.5,
                        help='output px per source px (0.5 = 1920 wide)')
    parser.add_argument('--levels', default='0,1')
    arguments = parser.parse_args()

    root = LAB / 'recordings' / arguments.sequence
    metas = sorted((root / 'meta').glob('*.json'))
    if not metas:
        raise SystemExit(f'no frames under {root}')

    levels = {int(v) for v in arguments.levels.split(',')}
    entries = []
    for path in metas:
        meta = json.loads(path.read_text())
        level = meta.get('resolution_level')
        if level not in levels:
            continue
        image_path = root / 'views' / (path.stem + '.png')
        if image_path.exists():
            entries.append((level, meta, image_path))
    if not entries:
        raise SystemExit('no matching views')

    last = max(m['frame_index'] for _, m, _ in entries)
    scale = arguments.scale
    height = int((SOURCE_H + arguments.dy * last) * scale) + 8
    width = int(SOURCE_W * scale)
    canvas = np.zeros((height, width, 3), np.uint8)

    # Coarse first, detailed last, so zoomed views win where they overlap.
    entries.sort(key=lambda e: e[0])
    for level, meta, image_path in entries:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        x1, y1, x2, y2 = meta['source_region_xyxy']
        ground_top = y1 + (last - meta['frame_index']) * arguments.dy
        tw = max(1, int(round((x2 - x1) * scale)))
        th = max(1, int(round((y2 - y1) * scale)))
        resized = cv2.resize(image, (tw, th), interpolation=cv2.INTER_AREA
                             if tw < image.shape[1] else cv2.INTER_LINEAR)
        left = int(round(x1 * scale))
        top = int(round(ground_top * scale))
        if top < 0 or left < 0 or top + th > height or left + tw > width:
            continue
        canvas[top:top + th, left:left + tw] = resized

    out = LAB / 'out' / f'strip{int(scale * 100)}_{arguments.sequence[:8]}.jpg'
    cv2.imwrite(str(out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])
    print(f'wrote {out}  {width}x{height}  (levels {sorted(levels)})')
    print(f'ground_y = source_y + ({last} - frame_index) * {arguments.dy}')
    print(f'source_x = strip_x / {scale}   source_ground_y = strip_y / {scale}')


if __name__ == '__main__':
    main()
