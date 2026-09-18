"""Contact sheet of a recorded flight: what did the camera actually look at?

Unlabelled diagnostic. Every transmitted view as a thumbnail in flight order,
bordered by resolution level, so wasted looks (open water, empty ground) and
the camera's raster pattern are visible at a glance.
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
LEVEL_COLOUR = {0: (255, 160, 60), 1: (60, 200, 255), 2: (120, 255, 120)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', default='14356d0b32754c4f9484a4bbb2d5b25d')
    parser.add_argument('--columns', type=int, default=14)
    parser.add_argument('--width', type=int, default=208)
    parser.add_argument('--out', type=Path, default=LAB / 'out' / 'flight_contact.png')
    arguments = parser.parse_args()

    root = RECORDINGS / arguments.sequence
    metas = sorted((root / 'meta').glob('*.json'))
    if not metas:
        raise SystemExit(f'no recorded meta under {root}')

    cells, levels = [], {}
    height = int(arguments.width * 9 / 16)
    for path in metas:
        meta = json.loads(path.read_text(encoding='utf-8'))
        image = cv2.imread(str(root / 'views' / f'{path.stem}.png'))
        if image is None:
            continue
        level = meta['resolution_level']
        levels[level] = levels.get(level, 0) + 1
        cell = cv2.resize(image, (arguments.width, height), interpolation=cv2.INTER_AREA)
        cv2.rectangle(cell, (0, 0), (arguments.width - 1, height - 1),
                      LEVEL_COLOUR.get(level, (200, 200, 200)), 2)
        cv2.putText(cell, f"{meta['frame_index']} L{level} n={len(meta['predictions'])}",
                    (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(cell, f"{meta['frame_index']} L{level} n={len(meta['predictions'])}",
                    (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(cell)

    rows = []
    for start in range(0, len(cells), arguments.columns):
        chunk = cells[start:start + arguments.columns]
        while len(chunk) < arguments.columns:
            chunk.append(np.zeros_like(cells[0]))
        rows.append(cv2.hconcat(chunk))
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(arguments.out), cv2.vconcat(rows))
    print(f'{len(cells)} views, levels {dict(sorted(levels.items()))}')
    print(f'wrote {arguments.out}')


if __name__ == '__main__':
    main()
