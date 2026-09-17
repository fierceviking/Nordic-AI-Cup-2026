"""Convert a YOLO-format dataset to the COCO layout RF-DETR expects.

RF-DETR reads ``<root>/<split>/_annotations.coco.json`` with the images beside
it. Images are hard-linked rather than copied, so a 6,000-image dataset costs
kilobytes instead of gigabytes; the link falls back to a copy if the filesystem
refuses.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))

from dtos import OBJECT_CLASSES  # noqa: E402

SPLITS = {'train': 'train', 'val': 'valid'}


def link(source: Path, target: Path) -> None:
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def convert_split(root: Path, out: Path, split: str, destination: str) -> tuple:
    images_dir = root / 'images' / split
    labels_dir = root / 'labels' / split
    if not images_dir.is_dir():
        return 0, 0
    target = out / destination
    target.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    images, annotations = [], []
    annotation_id = 1
    for image_id, image_path in enumerate(sorted(images_dir.glob('*.*')), start=1):
        with Image.open(image_path) as handle:
            width, height = handle.size
        link(image_path, target / image_path.name)
        images.append({'id': image_id, 'file_name': image_path.name,
                       'width': width, 'height': height})

        label_path = labels_dir / f'{image_path.stem}.txt'
        if not label_path.exists():
            continue
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            index, cx, cy, bw, bh = int(parts[0]), *(float(v) for v in parts[1:])
            w, h = bw * width, bh * height
            x, y = (cx * width) - w / 2, (cy * height) - h / 2
            if w <= 0 or h <= 0:
                continue
            annotations.append({
                'id': annotation_id, 'image_id': image_id,
                # COCO category ids are 1-based.
                'category_id': index + 1,
                'bbox': [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                'area': round(w * h, 2), 'iscrowd': 0,
            })
            annotation_id += 1

    payload = {
        'info': {'description': f'drone-flyby {split}'},
        'licenses': [],
        'images': images,
        'annotations': annotations,
        'categories': [{'id': i, 'name': name, 'supercategory': 'object'}
                       for i, name in enumerate(OBJECT_CLASSES, start=1)],
    }
    (target / '_annotations.coco.json').write_text(json.dumps(payload))
    return len(images), len(annotations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='YOLO dataset directory')
    parser.add_argument('--out', required=True)
    arguments = parser.parse_args()

    root, out = Path(arguments.root), Path(arguments.out)
    for split, destination in SPLITS.items():
        count, boxes = convert_split(root, out, split, destination)
        print(f'{destination:<6} {count:>6} images  {boxes:>7} boxes')

    # RF-DETR looks for a test split; point it at the validation images.
    test = out / 'test'
    if not test.exists() and (out / 'valid').exists():
        test.mkdir(parents=True, exist_ok=True)
        for path in (out / 'valid').iterdir():
            link(path, test / path.name)
        print(f'{"test":<6} mirrors valid')
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
