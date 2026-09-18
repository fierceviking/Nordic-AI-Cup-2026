"""Helsinki as the camera actually delivers it, with real ground truth.

The composites in dataset_v7 paste cut-out sprites, so every object carries a
feathered seam and harmonised colour that the renderer never produced. These
views contain natively rendered objects at exactly the sampling the camera
transmits, which is what a localisation fine-tune should learn from.

Every legal tile at each level: one Level-0 view, four Level-1, sixteen
Level-2 per frame. Frames are split by number so the holdout is unseen terrain
rather than a neighbouring crop of the same ground.

    python lab/make_helsinki_views.py --holdout 5
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))
sys.path.insert(0, str(LAB))

from dtos import OBJECT_CLASSES, SOURCE_REGION_SIZES  # noqa: E402
from simulate import render_view  # noqa: E402
from utils import frame_numbers, load_annotations, load_frame  # noqa: E402

VIEW = (960, 540)
MIN_VISIBLE = 0.6
MIN_SIDE = 2.0


def centres(level):
    width, height = SOURCE_REGION_SIZES[level]
    xs = range(width // 2, 3840, width)
    ys = range(height // 2, 2160, height)
    return [(x, y) for y in ys for x in xs]


def to_view(bbox, region):
    left, top, right, bottom = region
    scale_x = VIEW[0] / (right - left)
    scale_y = VIEW[1] / (bottom - top)
    x1, y1, x2, y2 = bbox
    return [(x1 - left) * scale_x, (y1 - top) * scale_y,
            (x2 - left) * scale_x, (y2 - top) * scale_y]


def clipped(box):
    """Clip to the view and report how much of the object survived."""
    x1, y1, x2, y2 = box
    area = max(1e-6, (x2 - x1) * (y2 - y1))
    cx1, cy1 = max(0.0, x1), max(0.0, y1)
    cx2, cy2 = min(float(VIEW[0]), x2), min(float(VIEW[1]), y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return None, 0.0
    return [cx1, cy1, cx2, cy2], (cx2 - cx1) * (cy2 - cy1) / area


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', default='hv')
    parser.add_argument('--holdout', type=int, default=5,
                        help='last N frames become the validation split')
    parser.add_argument('--levels', default='0,1,2')
    arguments = parser.parse_args()

    levels = [int(v) for v in arguments.levels.split(',')]
    frames = frame_numbers('helsinki')
    if not 0 < arguments.holdout < len(frames):
        parser.error('--holdout must leave some training frames')
    validation = set(frames[-arguments.holdout:])
    root = LAB / f'dataset_{arguments.name}'
    index = {name: i for i, name in enumerate(OBJECT_CLASSES)}
    for split in ('train', 'val'):
        (root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (root / 'labels' / split).mkdir(parents=True, exist_ok=True)

    counts, boxes, empty = Counter(), Counter(), Counter()
    for number in frames:
        split = 'val' if number in validation else 'train'
        image = load_frame(number, 'helsinki')
        truth = load_annotations(number, 'helsinki')
        for level in levels:
            for cx, cy in centres(level):
                view, region = render_view(image, level, cx, cy)
                lines = []
                for annotation in truth:
                    box, visible = clipped(to_view(annotation['bbox'], region))
                    if box is None or visible < MIN_VISIBLE:
                        continue
                    if box[2] - box[0] < MIN_SIDE or box[3] - box[1] < MIN_SIDE:
                        continue
                    lines.append(
                        f'{index[annotation["object_id"]]} '
                        f'{(box[0] + box[2]) / 2 / VIEW[0]:.6f} '
                        f'{(box[1] + box[3]) / 2 / VIEW[1]:.6f} '
                        f'{(box[2] - box[0]) / VIEW[0]:.6f} '
                        f'{(box[3] - box[1]) / VIEW[1]:.6f}')
                    boxes[split] += 1
                stem = f'f{number:04d}_L{level}_{cx}_{cy}'
                cv2.imwrite(str(root / 'images' / split / f'{stem}.png'), view)
                (root / 'labels' / split / f'{stem}.txt').write_text(
                    '\n'.join(lines), encoding='utf-8')
                counts[split] += 1
                if not lines:
                    empty[split] += 1
        del image
        print(f'  frame {number}: {split}', flush=True)

    yaml = (f'path: {root.resolve().as_posix()}\ntrain: images/train\n'
            f'val: images/val\nnames:\n' +
            ''.join(f'  {i}: {n}\n' for i, n in enumerate(OBJECT_CLASSES)))
    (root / 'data.yaml').write_text(yaml, encoding='utf-8')
    (root / 'manifest.json').write_text(json.dumps({
        'levels': levels, 'holdout_frames': sorted(validation),
        'images': dict(counts), 'boxes': dict(boxes), 'empty': dict(empty),
        'note': 'native renders, real ground truth, no compositing'},
        indent=2), encoding='utf-8')
    print(f'\nimages {dict(counts)}   boxes {dict(boxes)}   empty {dict(empty)}')
    print(f'holdout frames {sorted(validation)}')
    print(f'wrote {root / "data.yaml"}')


if __name__ == '__main__':
    main()
