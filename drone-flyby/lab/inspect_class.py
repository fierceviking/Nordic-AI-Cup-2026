"""Inspect one class: its raw crops and its extracted mattes side by side."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import frame_numbers, load_annotations, load_frame  # noqa: E402

OUT = Path(__file__).resolve().parent / 'out'
SCENE = 'helsinki'


def main(name: str):
    rows = []
    for frame in frame_numbers(SCENE):
        for annotation in load_annotations(frame, SCENE):
            if annotation['object_id'] != name:
                continue
            image = load_frame(frame, SCENE)
            x1, y1, x2, y2 = annotation['bbox']
            pad = 20
            crop = image[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad].copy()
            cv2.rectangle(crop, (x1 - max(0, x1 - pad), y1 - max(0, y1 - pad)),
                          (x2 - max(0, x1 - pad), y2 - max(0, y1 - pad)),
                          (0, 255, 255), 1)
            rows.append((frame, (x1, y1, x2, y2), crop))
    print(f'{name}: {len(rows)} annotated boxes')
    for frame, bbox, crop in rows:
        print(f'  frame {frame:3d} bbox={bbox} crop={crop.shape}')

    cell = 220
    canvas = np.full((cell + 20, cell * len(rows), 3), 30, np.uint8)
    for i, (frame, _, crop) in enumerate(rows):
        h, w = crop.shape[:2]
        scale = min(cell / w, cell / h)
        resized = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))),
                             interpolation=cv2.INTER_NEAREST)
        canvas[20:20 + resized.shape[0], i * cell:i * cell + resized.shape[1]] = resized
        cv2.putText(canvas, f'f{frame}', (i * cell + 2, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    path = OUT / f'inspect_{name}.png'
    cv2.imwrite(str(path), canvas)
    print('wrote', path)


if __name__ == '__main__':
    main(sys.argv[1])
