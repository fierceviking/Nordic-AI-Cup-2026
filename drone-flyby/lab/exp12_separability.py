"""At what sampling do the 16 assets actually become distinguishable?

The whole propose-confirm-remember idea rests on one assumption: a crop at L2
(native) can be named when the same crop at L0 (4x downsampled) cannot. If that
is false, spending camera time at L2 buys nothing and the redesign is pointless.

No training and no labels beyond the supplied Helsinki ground truth. A frozen
ImageNet encoder embeds each ground-truth crop at three samplings, and a class
is assigned by nearest neighbour against a reference bank. The bank is built
from each object's FIRST appearance and tested on its later ones, so the
background under the object differs between bank and query: this measures
appearance matching, not background memorisation.

    python lab/exp12_separability.py
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))

from utils import frame_numbers, load_annotations, load_frame  # noqa: E402

PATCH = 128
SAMPLINGS = {'L2 native': 1, 'L1 half': 2, 'L0 quarter': 4}


def crop(image, bbox, context):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * context / 2
    left, top = int(round(cx - half)), int(round(cy - half))
    right, bottom = int(round(cx + half)), int(round(cy + half))
    height, width = image.shape[:2]
    patch = image[max(0, top):min(height, bottom), max(0, left):min(width, right)]
    if patch.size == 0:
        return None
    pad = (max(0, -top), max(0, bottom - height), max(0, -left), max(0, right - width))
    if any(pad):
        patch = cv2.copyMakeBorder(patch, *pad, cv2.BORDER_REFLECT_101)
    return patch


def at_sampling(patch, divisor):
    """Degrade to the resolution the camera would deliver, then standardise."""
    if divisor > 1:
        small = (max(1, patch.shape[1] // divisor), max(1, patch.shape[0] // divisor))
        patch = cv2.resize(patch, small, interpolation=cv2.INTER_AREA)
    return cv2.resize(patch, (PATCH, PATCH), interpolation=cv2.INTER_LINEAR)


class Encoder:
    def __init__(self, device):
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
        self.device = device
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier = torch.nn.Identity()
        self.model = model.eval().to(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, patches):
        batch = np.stack([cv2.cvtColor(p, cv2.COLOR_BGR2RGB) for p in patches])
        tensor = torch.from_numpy(batch).to(self.device).permute(0, 3, 1, 2).float() / 255
        tensor = (tensor - self.mean) / self.std
        features = self.model(tensor)
        return torch.nn.functional.normalize(features, dim=1).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--context', type=float, default=1.3)
    parser.add_argument('--out', type=Path, default=LAB / 'out' / 'separability.json')
    arguments = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = Encoder(device)
    print(f'encoder on {device}, context {arguments.context}')

    records = defaultdict(list)
    for number in frame_numbers('helsinki'):
        image = load_frame(number, 'helsinki')
        for annotation in load_annotations(number, 'helsinki'):
            patch = crop(image, [int(v) for v in annotation['bbox']], arguments.context)
            if patch is None:
                continue
            side = min(annotation['bbox'][2] - annotation['bbox'][0],
                       annotation['bbox'][3] - annotation['bbox'][1])
            records[annotation['object_id']].append((number, patch, float(side)))
        del image

    summary = {}
    print(f'\n{"sampling":12s} {"top-1":>8s} {"top-3":>8s} {"queries":>8s}   worst classes')
    for name, divisor in SAMPLINGS.items():
        bank, bank_labels, queries, query_labels, query_sides = [], [], [], [], []
        for label, entries in records.items():
            entries = sorted(entries, key=lambda e: e[0])
            bank.append(at_sampling(entries[0][1], divisor))
            bank_labels.append(label)
            for _, patch, side in entries[1:]:
                queries.append(at_sampling(patch, divisor))
                query_labels.append(label)
                query_sides.append(side / divisor)
        if not queries:
            continue

        reference = encoder(bank)
        predicted, correct3 = [], 0
        for start in range(0, len(queries), 64):
            chunk = encoder(queries[start:start + 64])
            similarity = chunk @ reference.T
            order = np.argsort(-similarity, axis=1)
            for row, position in enumerate(order):
                predicted.append(bank_labels[position[0]])
                correct3 += int(query_labels[start + row]
                                in [bank_labels[i] for i in position[:3]])

        hits = np.array([p == t for p, t in zip(predicted, query_labels)])
        per_class = defaultdict(list)
        for label, hit in zip(query_labels, hits):
            per_class[label].append(hit)
        worst = sorted(((np.mean(v), k) for k, v in per_class.items()))[:3]
        summary[name] = {
            'top1': float(hits.mean()), 'top3': correct3 / len(queries),
            'queries': len(queries),
            'per_class': {k: float(np.mean(v)) for k, v in per_class.items()},
            'median_query_px': float(np.median(query_sides)),
        }
        print(f'{name:12s} {hits.mean():8.3f} {correct3 / len(queries):8.3f} '
              f'{len(queries):8d}   ' + ', '.join(f'{k} {v:.2f}' for v, k in worst))

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'\nwrote {arguments.out}')
    print('Frozen ImageNet features, one reference per class, cosine nearest '
          'neighbour. A purpose-trained embedding would do better; this is a '
          'lower bound on how separable the assets are at each sampling.')


if __name__ == '__main__':
    main()
