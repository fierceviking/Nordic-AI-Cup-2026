"""Draw the generated labels onto the generated images, for eyeballing."""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtos import OBJECT_CLASSES  # noqa: E402

DATASET = Path(__file__).resolve().parent / 'dataset'
OUT = Path(__file__).resolve().parent / 'out'


def main(split='train', count=6):
    count = int(count)
    images = sorted((DATASET / 'images' / split).glob('*.jpg'))[:count]
    tiles = []
    for path in images:
        image = cv2.imread(str(path))
        h, w = image.shape[:2]
        label_path = DATASET / 'labels' / split / (path.stem + '.txt')
        for line in label_path.read_text().splitlines():
            index, cx, cy, bw, bh = line.split()
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
            x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 1)
            cv2.putText(image, OBJECT_CLASSES[int(index)], (x1, max(9, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(image)
    rows = [cv2.hconcat(tiles[i:i + 2]) for i in range(0, len(tiles) - 1, 2)]
    sheet = cv2.vconcat(rows)
    out = OUT / f'dataset_preview_{split}.jpg'
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print('wrote', out)


if __name__ == '__main__':
    main(*(sys.argv[1:] or []))
