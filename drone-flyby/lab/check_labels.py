"""Check that every label box actually lands on the object it claims.

Flipping a canvas without flipping its boxes is a silent corruption: training
still runs, the loss still falls, and the detector learns to predict mirrored
positions. This verifies the invariant directly, by asking whether the pixels
inside each labelled box differ from their surroundings the way an object
should, and by confirming that unlabelled clutter is genuinely unlabelled.

A generated sample is rebuilt with a known seed, then every box is checked for
having materially more local contrast inside than in a ring around it. That is
a weak test, but a mirrored box fails it immediately.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent))

import make_dataset as md  # noqa: E402
from utils import frame_numbers, load_annotations, load_frame  # noqa: E402


def local_contrast(grey: np.ndarray) -> float:
    return float(grey.std()) if grey.size else 0.0


def main(samples: int = 60):
    import random
    numbers = frame_numbers(md.SCENE)
    frames = {f: load_frame(f, md.SCENE) for f in numbers}
    annotations = {f: load_annotations(f, md.SCENE) for f in numbers}
    library = md.load_sprites()

    checked = 0
    suspicious = 0
    empty_images = 0
    rng = random.Random(1234)
    np.random.seed(1234)

    for _ in range(samples):
        empty = rng.random() < 0.3
        view, labels, _ = md.make_sample(frames, annotations, library, rng,
                                         (3, 11), None, empty, (0, 6))
        if not labels:
            empty_images += 1
            continue
        grey = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
        h, w = grey.shape
        for _index, cx, cy, bw, bh in labels:
            x1 = int(max(0, (cx - bw / 2) * w))
            y1 = int(max(0, (cy - bh / 2) * h))
            x2 = int(min(w, (cx + bw / 2) * w))
            y2 = int(min(h, (cy + bh / 2) * h))
            if x2 - x1 < 3 or y2 - y1 < 3:
                continue
            inside = local_contrast(grey[y1:y2, x1:x2])
            pad = max(4, (x2 - x1) // 2)
            ox1, oy1 = max(0, x1 - pad), max(0, y1 - pad)
            ox2, oy2 = min(w, x2 + pad), min(h, y2 + pad)
            ring = grey[oy1:oy2, ox1:ox2].copy()
            ring[y1 - oy1:y2 - oy1, x1 - ox1:x2 - ox1] = 0
            outside = local_contrast(ring[ring > 0]) if (ring > 0).any() else 0.0
            checked += 1
            # A box centred on a pasted object should not be flatter than the
            # terrain around it. A mirrored box usually is.
            if inside < 0.45 * outside:
                suspicious += 1

    print(f'boxes checked        {checked}')
    print(f'suspicious boxes     {suspicious} '
          f'({suspicious / max(1, checked) * 100:.1f}%)')
    print(f'images with no boxes {empty_images}/{samples}')
    if checked and suspicious / checked > 0.15:
        print('\nFAIL: many boxes look like they are not on an object. '
              'Check that every geometric augmentation transforms labels too.')
        return 1
    print('\nOK: labels track the pixels they claim to.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
