"""A reference plate of the 16 known objects, cut from Helsinki at 1:1.

The evaluation scene uses the same assets, so labelling a candidate means
matching it against these rather than guessing. Each object is shown at source
resolution and again at quarter scale, which is how it looks in a Level-0 view.
"""

import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))

from utils import frame_numbers, load_annotations, load_frame  # noqa: E402

CELL = 150


def main():
    best = {}
    for frame in frame_numbers('helsinki'):
        for annotation in load_annotations(frame, 'helsinki'):
            x1, y1, x2, y2 = (int(v) for v in annotation['bbox'])
            area = (x2 - x1) * (y2 - y1)
            name = annotation['object_id']
            # Keep the largest, most central sighting of each object.
            if name not in best or area > best[name][1]:
                best[name] = (frame, area, (x1, y1, x2, y2))

    rows = []
    for name in sorted(best):
        frame, _, (x1, y1, x2, y2) = best[name]
        image = load_frame(frame, 'helsinki')
        pad = 12
        crop = image[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
        if crop.size == 0:
            continue
        native = cv2.resize(crop, (CELL, CELL), interpolation=cv2.INTER_NEAREST)
        # The same object as the detector sees it in a Level-0 view.
        small = cv2.resize(crop, (max(1, crop.shape[1] // 4),
                                  max(1, crop.shape[0] // 4)))
        small = cv2.resize(small, (CELL, CELL), interpolation=cv2.INTER_NEAREST)

        panel = cv2.hconcat([native, small])
        cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, CELL - 1), (50, 50, 50), 1)
        label = np.zeros((26, panel.shape[1], 3), np.uint8)
        cv2.putText(label, f'{name}  {x2 - x1}x{y2 - y1}px', (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        rows.append(cv2.vconcat([label, panel]))

    grid = []
    for i in range(0, len(rows), 4):
        chunk = rows[i:i + 4]
        while len(chunk) < 4:
            chunk.append(np.zeros_like(rows[0]))
        grid.append(cv2.hconcat(chunk))
    out = LAB / 'out' / 'object_reference.jpg'
    cv2.imwrite(str(out), cv2.vconcat(grid), [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f'wrote {out}  ({len(rows)} objects, native | level-0 scale)')


if __name__ == '__main__':
    main()
