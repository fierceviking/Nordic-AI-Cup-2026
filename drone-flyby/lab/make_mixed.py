"""Combine native Helsinki views with foreign-background composites.

Fine-tuning v7n on Helsinki views alone recovered Helsinki (0.22 -> 0.51) but
collapsed foreign-background accuracy (0.73 -> 0.31), below the model it was
meant to improve on. Background invariance survives only while foreign ground
stays in the mix, so the two sources are trained together rather than in
sequence.

    python lab/make_mixed.py --composites 1600
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))

from dtos import OBJECT_CLASSES  # noqa: E402


def copy_split(source, target, split, limit, generator, prefix):
    images = sorted((source / 'images' / split).glob('*.*'))
    if limit is not None and len(images) > limit:
        picked = generator.choice(len(images), size=limit, replace=False)
        images = [images[i] for i in sorted(picked)]
    kept = 0
    for path in images:
        label = source / 'labels' / split / f'{path.stem}.txt'
        if not label.is_file():
            continue
        shutil.copy2(path, target / 'images' / split / f'{prefix}{path.name}')
        shutil.copy2(label, target / 'labels' / split / f'{prefix}{path.stem}.txt')
        kept += 1
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', default='mix')
    parser.add_argument('--views', default='hv', help='native Helsinki view dataset')
    parser.add_argument('--composites', type=int, default=1600,
                        help='foreign-background images to mix in')
    parser.add_argument('--repeat-views', type=int, default=2,
                        help='times to repeat the small native set')
    parser.add_argument('--seed', type=int, default=0)
    arguments = parser.parse_args()

    generator = np.random.default_rng(arguments.seed)
    views = LAB / f'dataset_{arguments.views}'
    composites = LAB / 'dataset_v7'
    for source in (views, composites):
        if not (source / 'data.yaml').is_file():
            raise SystemExit(f'missing {source / "data.yaml"}')

    root = LAB / f'dataset_{arguments.name}'
    if root.exists():
        shutil.rmtree(root)
    for split in ('train', 'val'):
        (root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (root / 'labels' / split).mkdir(parents=True, exist_ok=True)

    counts = {}
    for split, limit in (('train', arguments.composites),
                         ('val', max(1, arguments.composites // 4))):
        native = 0
        for repeat in range(arguments.repeat_views):
            native += copy_split(views, root, split, None, generator, f'hv{repeat}_')
        foreign = copy_split(composites, root, split, limit, generator, 'fg_')
        counts[split] = {'native_helsinki': native, 'foreign_composites': foreign,
                         'total': native + foreign}

    yaml = (f'path: {root.resolve().as_posix()}\ntrain: images/train\n'
            f'val: images/val\nnames:\n' +
            ''.join(f'  {i}: {n}\n' for i, n in enumerate(OBJECT_CLASSES)))
    (root / 'data.yaml').write_text(yaml, encoding='utf-8')
    (root / 'manifest.json').write_text(json.dumps({
        'views': str(views), 'composites': str(composites),
        'repeat_views': arguments.repeat_views, 'counts': counts}, indent=2),
        encoding='utf-8')
    for split, value in counts.items():
        share = value['native_helsinki'] / max(1, value['total'])
        print(f'{split:6s} {value["total"]:5d} images   '
              f'native {value["native_helsinki"]:4d} ({share:.0%})   '
              f'foreign {value["foreign_composites"]:5d}')
    print(f'wrote {root / "data.yaml"}')


if __name__ == '__main__':
    main()
