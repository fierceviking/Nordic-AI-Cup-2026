"""One crop per candidate track, in a grid, for fast triage.

Viewing tracks one at a time is slow. A real CGI model and a shrub are
distinguishable at a glance, so the efficient thing is to put one
representative crop of every track on a single sheet and mark them in a batch.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))

RECORDINGS = LAB / 'recordings'
W, H = 3840, 2160


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--tracks', default='')
    parser.add_argument('--cols', type=int, default=8)
    parser.add_argument('--cell', type=int, default=160)
    arguments = parser.parse_args()

    directory = LAB / 'out' / f'candidates_{arguments.sequence[:8]}'
    sheets = sorted(directory.glob('track*.jpg'))
    if arguments.tracks:
        wanted = {int(v) for v in arguments.tracks.split(',')}
        sheets = [p for p in sheets if int(p.stem[5:]) in wanted]
    if not sheets:
        raise SystemExit(f'no track sheets in {directory}')

    cell = arguments.cell
    tiles = []
    for path in sheets:
        strip = cv2.imread(str(path))
        if strip is None:
            continue
        # each strip is a row of 192px crops; take the middle one
        n = max(1, strip.shape[1] // 192)
        middle = strip[:, (n // 2) * 192:(n // 2 + 1) * 192]
        tile = cv2.resize(middle, (cell, cell), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(tile, (0, 0), (cell - 1, cell - 1), (40, 40, 40), 1)
        cv2.putText(tile, path.stem[5:], (4, cell - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)

    cols = arguments.cols
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros((cell, cell, 3), np.uint8))
        rows.append(cv2.hconcat(row))
    out = LAB / 'out' / f'triage_{arguments.sequence[:8]}.jpg'
    cv2.imwrite(str(out), cv2.vconcat(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f'wrote {out} ({len(tiles)} tracks, numbered)')


if __name__ == '__main__':
    main()
