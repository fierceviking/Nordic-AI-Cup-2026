"""Build the stage-2 view of a dataset: only the images that contain objects.

The two-stage recipe is: train once on positives **and** pure-background
negatives so the detector learns what "nothing here" looks like, then fine-tune
on the positives alone so its remaining capacity goes on localising and
classifying rather than on rejecting.

Nothing is copied. Ultralytics accepts a text file listing image paths, so this
just writes that list and a matching ``data.yaml``.
"""

import argparse
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))

from dtos import OBJECT_CLASSES  # noqa: E402


def collect(dataset: Path, split: str):
    positives, empties = [], []
    for image in sorted((dataset / 'images' / split).glob('*.jpg')):
        label = dataset / 'labels' / split / (image.stem + '.txt')
        if label.exists() and label.read_text().strip():
            positives.append(image)
        else:
            empties.append(image)
    return positives, empties


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--name', default='data_positives.yaml')
    arguments = parser.parse_args()

    dataset = Path(arguments.dataset).resolve()
    listings = {}
    for split in ('train', 'val'):
        positives, empties = collect(dataset, split)
        listing = dataset / f'{split}_positives.txt'
        listing.write_text('\n'.join(p.as_posix() for p in positives) + '\n')
        listings[split] = listing
        total = len(positives) + len(empties)
        print(f'{split}: {len(positives)} positive, {len(empties)} background '
              f'({len(empties) / max(1, total) * 100:.0f}% background) '
              f'-> {listing.name}')

    yaml_path = dataset / arguments.name
    with open(yaml_path, 'w') as handle:
        handle.write(f'path: {dataset.as_posix()}\n')
        handle.write(f'train: {listings["train"].name}\n')
        handle.write(f'val: {listings["val"].name}\n')
        handle.write(f'nc: {len(OBJECT_CLASSES)}\n')
        handle.write('names:\n')
        for i, name in enumerate(OBJECT_CLASSES):
            handle.write(f'  {i}: {name}\n')
    print(f'\nwrote {yaml_path}')


if __name__ == '__main__':
    main()
