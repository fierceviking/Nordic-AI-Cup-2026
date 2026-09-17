"""Build a single montage of every class crop for quick visual inspection."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path(__file__).resolve().parent / 'out'


def montage(suffix: str, cell: int = 200, cols: int = 4) -> np.ndarray:
    paths = sorted((OUT / 'crops').glob(f'*_{suffix}.png'))
    rows = (len(paths) + cols - 1) // cols
    canvas = np.zeros((rows * (cell + 22), cols * cell, 3), np.uint8)
    for i, path in enumerate(paths):
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        scale = min(cell / w, cell / h)
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_NEAREST)
        r, c = divmod(i, cols)
        y0 = r * (cell + 22) + 22
        x0 = c * cell
        canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
        cv2.putText(canvas, path.stem.replace(f'_{suffix}', ''), (x0 + 2, y0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


if __name__ == '__main__':
    for suffix in ('src', 'L1', 'L0'):
        out = OUT / f'montage_{suffix}.png'
        cv2.imwrite(str(out), montage(suffix))
        print('wrote', out)
