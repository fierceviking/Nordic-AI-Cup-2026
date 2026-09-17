"""Propose object locations on the ground-fixed strip, for labelling.

Detections above a confidence floor are mapped into ground coordinates and
clustered, then each cluster is cropped from the high-resolution strip. That
turns labelling into "identify this crop against the reference plate" rather
than scanning nine thousand rows by eye.

Locations we never detect cannot appear here, so this is a starting point for
labelling and not a survey of the scene.
"""

import argparse
import json
import sys
from collections import Counter
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
    parser.add_argument('--scale', type=float, default=0.5)
    parser.add_argument('--min-conf', type=float, default=0.60)
    parser.add_argument('--radius', type=float, default=140.0, help='source px')
    parser.add_argument('--cell', type=int, default=220)
    parser.add_argument('--cols', type=int, default=6)
    arguments = parser.parse_args()

    root = LAB / 'recordings' / arguments.sequence
    metas = sorted((root / 'meta').glob('*.json'))
    last = 0
    points = []
    for path in metas:
        meta = json.loads(path.read_text())
        last = max(last, meta['frame_index'])
    for path in metas:
        meta = json.loads(path.read_text())
        shift = (last - meta['frame_index']) * arguments.dy
        for prediction in meta.get('predictions', []):
            confidence = float(prediction['confidence'])
            if confidence < arguments.min_conf:
                continue
            x1, y1, x2, y2 = prediction['bbox']
            cx = (x1 + x2) / 2 * SOURCE_W
            cy = (y1 + y2) / 2 * SOURCE_H + shift
            points.append((cx, cy, prediction['object_id'], confidence,
                           (x2 - x1) * SOURCE_W, (y2 - y1) * SOURCE_H))
    if not points:
        raise SystemExit('no detections above the floor')

    clusters = []
    for cx, cy, name, confidence, w, h in sorted(points, key=lambda p: -p[3]):
        for cluster in clusters:
            if abs(cx - cluster['x']) < arguments.radius and \
               abs(cy - cluster['y']) < arguments.radius:
                cluster['votes'][name] += confidence
                cluster['n'] += 1
                cluster['w'] = max(cluster['w'], w)
                cluster['h'] = max(cluster['h'], h)
                break
        else:
            clusters.append({'x': cx, 'y': cy, 'votes': Counter({name: confidence}),
                             'n': 1, 'conf': confidence, 'w': w, 'h': h})

    clusters.sort(key=lambda c: -c['n'])
    strip = cv2.imread(str(LAB / 'out' /
                           f'strip{int(arguments.scale * 100)}_{arguments.sequence[:8]}.jpg'))
    if strip is None:
        raise SystemExit('build the strip first with flight_strip2.py')

    cell = arguments.cell
    tiles = []
    print(f'{"id":>3} {"n":>4} {"conf":>6} {"src_x":>7} {"ground_y":>9}  votes')
    for index, cluster in enumerate(clusters[:arguments.cols * 6]):
        sx = int(cluster['x'] * arguments.scale)
        sy = int(cluster['y'] * arguments.scale)
        half = cell // 2
        crop = strip[max(0, sy - half):sy + half, max(0, sx - half):sx + half]
        if crop.size == 0:
            continue
        tile = cv2.resize(crop, (cell, cell), interpolation=cv2.INTER_NEAREST)
        top = cluster['votes'].most_common(1)[0][0]
        cv2.putText(tile, f'{index}:{top}', (4, cell - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(tile, (0, 0), (cell - 1, cell - 1), (40, 40, 40), 1)
        tiles.append(tile)
        print(f'{index:>3} {cluster["n"]:>4} {cluster["conf"]:6.2f} '
              f'{cluster["x"]:7.0f} {cluster["y"]:9.0f}  '
              f'{", ".join(f"{k}:{v:.1f}" for k, v in cluster["votes"].most_common(3))}')

    rows = []
    for i in range(0, len(tiles), arguments.cols):
        row = tiles[i:i + arguments.cols]
        while len(row) < arguments.cols:
            row.append(np.zeros((cell, cell, 3), np.uint8))
        rows.append(cv2.hconcat(row))
    out = LAB / 'out' / f'proposals_{arguments.sequence[:8]}.jpg'
    cv2.imwrite(str(out), cv2.vconcat(rows), [cv2.IMWRITE_JPEG_QUALITY, 94])
    print(f'\nwrote {out}  ({len(tiles)} crops)')


if __name__ == '__main__':
    main()
