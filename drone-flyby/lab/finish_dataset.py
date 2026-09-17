"""Finish a dataset whose train split was cut short.

Generating 7 000 images takes long enough that a shared machine can make it the
bottleneck. This keeps whatever train images already exist, writes the matching
validation split, prunes any image whose label file is missing, and emits
``data.yaml``.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))
sys.path.insert(0, str(LAB))

import make_dataset as md  # noqa: E402
from dtos import OBJECT_CLASSES  # noqa: E402
from utils import frame_numbers, load_annotations, load_frame  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--val', type=int, default=300)
    parser.add_argument('--seed', type=int, default=9999)
    arguments = parser.parse_args()

    dataset = Path(arguments.out)
    md.DATASET = dataset

    # Drop any train image written without its label file.
    images = sorted((dataset / 'images' / 'train').glob('*.jpg'))
    removed = 0
    for path in images:
        if not (dataset / 'labels' / 'train' / (path.stem + '.txt')).exists():
            path.unlink()
            removed += 1
    kept = len(images) - removed
    print(f'train split: {kept} images ({removed} unpaired removed)')

    library = md.load_sprites()
    numbers = frame_numbers(md.SCENE)
    print('loading frames ...')
    frames = {f: load_frame(f, md.SCENE) for f in numbers}
    annotations = {f: load_annotations(f, md.SCENE) for f in numbers}

    print(f'writing {arguments.val} val images ...')
    val_histogram = md.write_split('val', arguments.val, frames, annotations,
                                   library, arguments.seed, (3, 8))

    with open(dataset / 'data.yaml', 'w') as handle:
        handle.write(f'path: {dataset.resolve().as_posix()}\n')
        handle.write('train: images/train\nval: images/val\n')
        handle.write(f'nc: {len(OBJECT_CLASSES)}\n')
        handle.write('names:\n')
        for i, name in enumerate(OBJECT_CLASSES):
            handle.write(f'  {i}: {name}\n')

    train_histogram = {name: 0 for name in OBJECT_CLASSES}
    for path in (dataset / 'labels' / 'train').glob('*.txt'):
        for line in path.read_text().splitlines():
            if line.strip():
                train_histogram[OBJECT_CLASSES[int(line.split()[0])]] += 1
    with open(dataset / 'stats.json', 'w') as handle:
        json.dump({'train': train_histogram, 'val': val_histogram}, handle, indent=2)

    print(f'\n{"class":16s} {"train":>7s} {"val":>6s}')
    for name in OBJECT_CLASSES:
        print(f'{name:16s} {train_histogram[name]:7d} {val_histogram[name]:6d}')
    print(f'{"TOTAL":16s} {sum(train_histogram.values()):7d} '
          f'{sum(val_histogram.values()):6d}')
    print(f'\nwrote {dataset / "data.yaml"}')


if __name__ == '__main__':
    main()
