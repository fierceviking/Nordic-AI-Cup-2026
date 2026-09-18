"""Can appearance retrieval find the assets, without a trained detector?

exp12 showed the 16 assets are 83% separable from each other at L0 sampling
using one reference each. That only matters if the same descriptor can also
reject background, which is the harder half: a nearest neighbour always returns
something, so the question is whether asset crops sit closer to the reference
bank than ordinary ground does.

Stage 1 measures that on Helsinki, where ground truth says which crops are
assets. Stage 2 applies the resulting threshold to the real competition views,
where no labels exist, and reports how much of the frame survives it.

    python lab/exp13_retrieval.py --scan 8
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))

from exp12_separability import Encoder, at_sampling, crop  # noqa: E402
from utils import frame_numbers, load_annotations, load_frame  # noqa: E402

RECORDINGS = LAB / 'recordings'
CONTEXT = 1.3


def build_bank(encoder, divisor):
    """One reference crop per class, from each object's first appearance."""
    seen, patches, labels, sides = set(), [], [], []
    for number in frame_numbers('helsinki'):
        annotations = [a for a in load_annotations(number, 'helsinki')
                       if a['object_id'] not in seen]
        if not annotations:
            continue
        image = load_frame(number, 'helsinki')
        for annotation in annotations:
            box = [int(v) for v in annotation['bbox']]
            patch = crop(image, box, CONTEXT)
            if patch is None:
                continue
            seen.add(annotation['object_id'])
            patches.append(at_sampling(patch, divisor))
            labels.append(annotation['object_id'])
            sides.append(min(box[2] - box[0], box[3] - box[1]))
        del image
    return encoder(patches), labels, sides


def helsinki_scores(encoder, reference, divisor, negatives_per_frame, seed):
    """Best similarity to the bank for asset crops and for random ground."""
    generator = np.random.default_rng(seed)
    positive, negative = [], []
    for number in frame_numbers('helsinki'):
        annotations = load_annotations(number, 'helsinki')
        image = load_frame(number, 'helsinki')
        boxes = [[int(v) for v in a['bbox']] for a in annotations]
        patches = [crop(image, b, CONTEXT) for b in boxes]
        patches = [at_sampling(p, divisor) for p in patches if p is not None]
        if patches:
            positive.extend((encoder(patches) @ reference.T).max(axis=1).tolist())

        sizes = [max(b[2] - b[0], b[3] - b[1]) for b in boxes] or [48]
        sampled = []
        while len(sampled) < negatives_per_frame:
            size = int(generator.choice(sizes))
            x = int(generator.integers(0, max(1, 3840 - size)))
            y = int(generator.integers(0, max(1, 2160 - size)))
            candidate = [x, y, x + size, y + size]
            if any(overlaps(candidate, b) for b in boxes):
                continue
            patch = crop(image, candidate, CONTEXT)
            if patch is not None:
                sampled.append(at_sampling(patch, divisor))
        negative.extend((encoder(sampled) @ reference.T).max(axis=1).tolist())
        del image
    return np.array(positive), np.array(negative)


def overlaps(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def auc(positive, negative):
    values = np.concatenate([positive, negative])
    order = values.argsort()
    ranks = np.empty(len(values), float)
    ranks[order] = np.arange(1, len(values) + 1)
    n, m = len(positive), len(negative)
    return (ranks[:n].sum() - n * (n + 1) / 2) / (n * m)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--divisor', type=int, default=4, help='4 = L0 sampling')
    parser.add_argument('--negatives', type=int, default=40)
    parser.add_argument('--scan', type=int, default=6, help='real views to scan')
    parser.add_argument('--stride', type=int, default=6)
    parser.add_argument('--windows', default='10,16,26',
                        help='view px; a median object is 12 px at L0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=Path, default=LAB / 'out' / 'retrieval.json')
    arguments = parser.parse_args()

    import torch
    encoder = Encoder('cuda' if torch.cuda.is_available() else 'cpu')
    reference, labels, sides = build_bank(encoder, arguments.divisor)
    print(f'bank: {len(labels)} classes, reference short sides '
          f'{min(sides)}-{max(sides)} source px')

    positive, negative = helsinki_scores(encoder, reference, arguments.divisor,
                                         arguments.negatives, arguments.seed)
    area = auc(positive, negative)
    print(f'\nHELSINKI  assets {len(positive)}  background {len(negative)}')
    print(f'  similarity to bank: assets {positive.mean():.3f} +/- {positive.std():.3f}   '
          f'background {negative.mean():.3f} +/- {negative.std():.3f}')
    print(f'  separability AUC  : {area:.3f}')
    for keep in (0.90, 0.80, 0.50):
        threshold = float(np.quantile(positive, 1 - keep))
        leak = float((negative >= threshold).mean())
        print(f'  keep {keep:.0%} of assets at t={threshold:.3f}  ->  '
              f'{leak:.1%} of background also passes')

    threshold = float(np.quantile(positive, 0.20))
    strict = float(np.quantile(positive, 0.50))
    summary = {'auc': area, 'threshold_keep80': threshold, 'threshold_keep50': strict,
               'positive_mean': float(positive.mean()),
               'negative_mean': float(negative.mean())}

    if arguments.scan:
        sequence = max((p for p in RECORDINGS.iterdir() if (p / 'views').is_dir()),
                       key=lambda p: len(list((p / 'views').glob('*.png'))))
        views = sorted((sequence / 'views').glob('*.png'))
        chosen = views[::max(1, len(views) // arguments.scan)][:arguments.scan]
        sizes = [int(v) for v in arguments.windows.split(',')]
        leak80 = float((negative >= threshold).mean())
        leak50 = float((negative >= strict).mean())
        print(f'\nREAL views from {sequence.name}, stride {arguments.stride}')
        print(f'  Helsinki background leaks {leak80:.2%} at t={threshold:.3f}, '
              f'{leak50:.2%} at t={strict:.3f}')
        print(f'  ~10 assets per frame would occupy roughly 0.5% of windows\n')
        print(f'  {"view":13s} {"win":>4s} {"windows":>8s} {"pass t80":>9s} '
              f'{"pass t50":>9s} {"best":>6s}')
        rows = []
        for path in chosen:
            image = cv2.imread(str(path))
            for window in sizes:
                patches, half = [], window // 2
                for y in range(half, image.shape[0] - half, arguments.stride):
                    for x in range(half, image.shape[1] - half, arguments.stride):
                        patch = crop(image, [x - half, y - half, x + half, y + half], CONTEXT)
                        if patch is not None:
                            patches.append(at_sampling(patch, 1))
                best = np.concatenate([(encoder(patches[i:i + 512]) @ reference.T).max(axis=1)
                                       for i in range(0, len(patches), 512)])
                rows.append({'view': path.stem, 'window': window,
                             'pass80': float((best >= threshold).mean()),
                             'pass50': float((best >= strict).mean()),
                             'best': float(best.max())})
                print(f'  {path.stem:13s} {window:4d} {len(patches):8d} '
                      f'{rows[-1]["pass80"]:8.2%} {rows[-1]["pass50"]:9.3%} '
                      f'{rows[-1]["best"]:6.3f}')
        summary['scan'] = rows
        summary['helsinki_leak80'] = leak80
        summary['helsinki_leak50'] = leak50

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'\nwrote {arguments.out}')


if __name__ == '__main__':
    main()
