"""Build a validation split from the real frames, with the real annotations.

Training has been validating on synthetic composites drawn from the same 25
frames as the training images - the generator grading itself. `best.pt` is
selected on that metric, so a checkpoint that is good at finding pasted sprites
beats one that is good at finding objects.

This builds the alternative: the supplied frames put through the evaluator's
exact imaging chain (crop, INTER_AREA to 960x540), labelled from the supplied
annotations. Real pixels, real objects in real context, no paste seams.

It is **not held out** - the sprites were cut from these frames - so it is not a
generalisation estimate either. It is a better checkpoint-selection criterion
than synthetic data, and that is all it is for. The honest number still only
comes from the competition API.

Levels 0 and 1 are used, in the proportion the shipped camera policy actually
spends its time in (75 % / 25 %).
"""

import argparse
import sys
from pathlib import Path

import cv2

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))
sys.path.insert(0, str(LAB))

from dtos import OBJECT_CLASSES, SOURCE_REGION_SIZES  # noqa: E402
from simulate import render_view, tile_centers  # noqa: E402
from utils import frame_numbers, load_annotations, load_frame  # noqa: E402

SCENE = 'helsinki'
VIEW_W, VIEW_H = 960, 540
CLASS_INDEX = {name: i for i, name in enumerate(OBJECT_CLASSES)}
# Matches MINIMUM_VISIBLE in make_dataset.py, so the two agree on what counts
# as a visible object.
MINIMUM_VISIBLE = 0.35


def labels_for_view(annotations, region):
    """Real annotations mapped into one view, clipped, in YOLO format."""
    sx1, sy1, sx2, sy2 = region
    scale_x = VIEW_W / (sx2 - sx1)
    scale_y = VIEW_H / (sy2 - sy1)
    out = []
    for annotation in annotations:
        bx1, by1, bx2, by2 = annotation['bbox']
        cx1, cy1 = max(bx1, sx1), max(by1, sy1)
        cx2, cy2 = min(bx2, sx2), min(by2, sy2)
        if cx2 - cx1 < 1 or cy2 - cy1 < 1:
            continue
        visible = ((cx2 - cx1) * (cy2 - cy1)) / max(1, (bx2 - bx1) * (by2 - by1))
        if visible < MINIMUM_VISIBLE:
            continue
        x1 = (cx1 - sx1) * scale_x
        y1 = (cy1 - sy1) * scale_y
        x2 = (cx2 - sx1) * scale_x
        y2 = (cy2 - sy1) * scale_y
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        out.append((CLASS_INDEX[annotation['object_id']],
                    (x1 + x2) / 2 / VIEW_W, (y1 + y2) / 2 / VIEW_H,
                    (x2 - x1) / VIEW_W, (y2 - y1) / VIEW_H))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True,
                        help='dataset directory to write the real val split into')
    parser.add_argument('--levels', default='0,1')
    arguments = parser.parse_args()

    dataset = Path(arguments.out)
    image_dir = dataset / 'images' / 'realval'
    label_dir = dataset / 'labels' / 'realval'
    for directory in (image_dir, label_dir):
        directory.mkdir(parents=True, exist_ok=True)

    levels = [int(v) for v in arguments.levels.split(',')]
    written = 0
    with_objects = 0
    boxes = 0
    for frame in frame_numbers(SCENE):
        image = load_frame(frame, SCENE)
        annotations = load_annotations(frame, SCENE)
        for level in levels:
            for cx, cy in tile_centers(level):
                view, region = render_view(image, level, cx, cy)
                labels = labels_for_view(annotations, region)
                stem = f'f{frame:03d}_L{level}_{cx}_{cy}'
                cv2.imwrite(str(image_dir / f'{stem}.jpg'), view,
                            [cv2.IMWRITE_JPEG_QUALITY, 96])
                with open(label_dir / f'{stem}.txt', 'w') as handle:
                    for index, bx, by, bw, bh in labels:
                        handle.write(f'{index} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n')
                written += 1
                boxes += len(labels)
                with_objects += 1 if labels else 0

    print(f'wrote {written} real validation views '
          f'({with_objects} with objects, {written - with_objects} empty), '
          f'{boxes} boxes')
    for level in levels:
        tiles = len(tile_centers(level))
        print(f'  L{level}: {tiles} tiles x 25 frames = {tiles * 25} views')

    yaml_path = dataset / 'data_realval.yaml'
    with open(yaml_path, 'w') as handle:
        handle.write(f'path: {dataset.resolve().as_posix()}\n')
        handle.write('train: images/train\nval: images/realval\n')
        handle.write(f'nc: {len(OBJECT_CLASSES)}\n')
        handle.write('names:\n')
        for i, name in enumerate(OBJECT_CLASSES):
            handle.write(f'  {i}: {name}\n')
    print(f'wrote {yaml_path}')


if __name__ == '__main__':
    main()
