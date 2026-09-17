"""Classical machine learning: pyramid HOG + colour, linear model, sliding window.

The rung between hand-crafted thresholds and a learned detector - the
Dalal-Triggs recipe. A fixed 32x32 window is described by a histogram of
oriented gradients plus a coarse Lab colour layout; a linear classifier
separates the sixteen object classes from background; the classifier is slid
over an image pyramid and the peaks are kept.

Two implementation notes that decide whether this is feasible at all:

* descriptors are computed **densely per pyramid level** with a single
  ``HOGDescriptor.compute`` call over an explicit location list, not one call
  per window. Per-window calls are around fifty times slower and make the
  experiment impossible to run;
* colour statistics come from integral images, so they cost one pass per level.

Training patches come from the same synthetic dataset the YOLO uses, so the two
rungs are compared on identical data.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from detector_eval import FrameCache, iou_matrix, report_detector, run_detector  # noqa: E402
from dtos import OBJECT_CLASSES  # noqa: E402

LAB = Path(__file__).resolve().parent
DATASET = LAB / 'dataset'

PATCH = 32
BACKGROUND = len(OBJECT_CLASSES)
# Object side lengths, in transmitted pixels, that the window is mapped onto.
# Spans ta-ta at L1 (16x8) up to hangar at L2 (186x94).
SIZES = (14, 20, 28, 40, 58, 84, 120)
STRIDE = 8

# HOG geometry, matching the classic 32x32 window / 8x8 cell / 16x16 block
# layout: 4x4 cells, 3x3 overlapping blocks of 2x2 cells, 9 unsigned bins.
CELL = 8
BINS = 9
CELLS_PER_SIDE = PATCH // CELL             # 4
BLOCKS_PER_SIDE = CELLS_PER_SIDE - 1       # 3
HOG_DIMENSION = BLOCKS_PER_SIDE ** 2 * 4 * BINS    # 324
COLOUR_DIMENSION = 12                      # 2x2 cells x 3 Lab channels
DIMENSION = HOG_DIMENSION + COLOUR_DIMENSION


# --------------------------------------------------------------------------- #
# Dense HOG
# --------------------------------------------------------------------------- #
#
# OpenCV 5 dropped ``cv2.HOGDescriptor``, and a per-window Python implementation
# is far too slow for a sliding window over an image pyramid. Instead the nine
# orientation channels are integrated once per pyramid level, after which the
# histogram of any cell costs three additions and the whole window grid is one
# vectorised gather.

def orientation_integrals(grey: np.ndarray) -> np.ndarray:
    """Integral images of the nine orientation-weighted gradient channels."""
    g = grey.astype(np.float32) / 255.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=1)
    magnitude = np.hypot(gx, gy)
    angle = np.arctan2(gy, gx) % np.pi          # unsigned orientation
    index = np.minimum((angle / (np.pi / BINS)).astype(np.int32), BINS - 1)
    integrals = np.empty((BINS, grey.shape[0] + 1, grey.shape[1] + 1), np.float32)
    for b in range(BINS):
        integrals[b] = cv2.integral(np.where(index == b, magnitude, 0.0))
    return integrals


def _cell_sums(integrals: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Histogram of the CELLxCELL cell at each (x, y). Returns (N, BINS)."""
    x0, y0 = xs, ys
    x1, y1 = xs + CELL, ys + CELL
    total = (integrals[:, y1, x1] - integrals[:, y0, x1]
             - integrals[:, y1, x0] + integrals[:, y0, x0])
    return total.T


def hog_descriptors(grey: np.ndarray, locations: np.ndarray) -> np.ndarray:
    """HOG for every PATCHxPATCH window whose top-left corner is in locations."""
    if len(locations) == 0:
        return np.zeros((0, HOG_DIMENSION), np.float32)
    integrals = orientation_integrals(grey)
    xs = locations[:, 0].astype(np.int32)
    ys = locations[:, 1].astype(np.int32)

    cells = np.empty((CELLS_PER_SIDE, CELLS_PER_SIDE, len(locations), BINS),
                     np.float32)
    for row in range(CELLS_PER_SIDE):
        for column in range(CELLS_PER_SIDE):
            cells[row, column] = _cell_sums(integrals, xs + column * CELL,
                                            ys + row * CELL)

    blocks = []
    for row in range(BLOCKS_PER_SIDE):
        for column in range(BLOCKS_PER_SIDE):
            block = np.concatenate([cells[row, column], cells[row, column + 1],
                                    cells[row + 1, column],
                                    cells[row + 1, column + 1]], axis=1)
            # L2-Hys: normalise, clip, renormalise.
            block = block / (np.linalg.norm(block, axis=1, keepdims=True) + 1e-6)
            np.clip(block, 0, 0.2, out=block)
            block = block / (np.linalg.norm(block, axis=1, keepdims=True) + 1e-6)
            blocks.append(block)
    return np.concatenate(blocks, axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Dense descriptors over one image
# --------------------------------------------------------------------------- #

def colour_features(image: np.ndarray, locations: np.ndarray) -> np.ndarray:
    """Mean Lab of each quadrant of every 32x32 window, via an integral image."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    integral = cv2.integral(lab)                       # (h+1, w+1, 3)
    half = PATCH // 2
    area = float(half * half)
    out = np.empty((len(locations), COLOUR_DIMENSION), np.float32)
    column = 0
    for dy in (0, half):
        for dx in (0, half):
            x0 = locations[:, 0] + dx
            y0 = locations[:, 1] + dy
            x1, y1 = x0 + half, y0 + half
            total = (integral[y1, x1] - integral[y0, x1]
                     - integral[y1, x0] + integral[y0, x0])
            out[:, column:column + 3] = total / area
            column += 3
    return out


def _compute_hog(grey: np.ndarray, locations: np.ndarray) -> np.ndarray:
    return hog_descriptors(grey, np.asarray(locations, np.int32))


def dense_descriptors(image: np.ndarray, stride: int = STRIDE):
    """Descriptors for every 32x32 window on a stride grid. Returns (X, xy)."""
    h, w = image.shape[:2]
    if h < PATCH or w < PATCH:
        return np.zeros((0, DIMENSION), np.float32), np.zeros((0, 2), np.int32)
    xs = np.arange(0, w - PATCH + 1, stride)
    ys = np.arange(0, h - PATCH + 1, stride)
    grid = np.array([(int(x), int(y)) for y in ys for x in xs], np.int32)
    if len(grid) == 0:
        return np.zeros((0, DIMENSION), np.float32), grid
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return (np.hstack([_compute_hog(grey, grid), colour_features(image, grid)]),
            grid)


def describe_patches(patches: np.ndarray) -> np.ndarray:
    """Descriptors for a stack of exactly-32x32 BGR patches."""
    if len(patches) == 0:
        return np.zeros((0, DIMENSION), np.float32)
    out = np.empty((len(patches), DIMENSION), np.float32)
    chunk = 512
    for start in range(0, len(patches), chunk):
        block = patches[start:start + chunk]
        strip = np.concatenate(list(block), axis=1)        # 32 x (32n)
        locations = np.stack([np.arange(len(block)) * PATCH,
                              np.zeros(len(block), int)], 1).astype(np.int32)
        grey = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        out[start:start + len(block)] = np.hstack(
            [_compute_hog(grey, locations), colour_features(strip, locations)])
    return out


def crop_patch(image: np.ndarray, cx: float, cy: float, size: float):
    half = size / 2
    padded = cv2.copyMakeBorder(image, PATCH, PATCH, PATCH, PATCH,
                                cv2.BORDER_REFLECT_101)
    x1 = max(0, int(round(cx - half)) + PATCH)
    y1 = max(0, int(round(cy - half)) + PATCH)
    x2 = min(padded.shape[1], int(round(cx + half)) + PATCH)
    y2 = min(padded.shape[0], int(round(cy + half)) + PATCH)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return cv2.resize(padded[y1:y2, x1:x2], (PATCH, PATCH),
                      interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- #
# Training set
# --------------------------------------------------------------------------- #

def read_labels(path: Path, width: int, height: int):
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            out.append((int(parts[0]), float(parts[1]) * width,
                        float(parts[2]) * height, float(parts[3]) * width,
                        float(parts[4]) * height))
    return out


def build_patches(split: str, limit: int, rng, negatives_per_image: int = 20):
    image_dir = DATASET / 'images' / split
    label_dir = DATASET / 'labels' / split
    patches: List[np.ndarray] = []
    labels: List[int] = []
    for path in sorted(image_dir.glob('*.jpg'))[:limit]:
        image = cv2.imread(str(path))
        h, w = image.shape[:2]
        boxes = read_labels(label_dir / (path.stem + '.txt'), w, h)
        occupied = [(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
                    for _, cx, cy, bw, bh in boxes]
        for index, cx, cy, bw, bh in boxes:
            size = max(bw, bh) * 1.25
            if size < 8:
                continue
            for _ in range(2):
                patch = crop_patch(image,
                                   cx + rng.normal(0, max(1.0, bw * 0.08)),
                                   cy + rng.normal(0, max(1.0, bh * 0.08)),
                                   size * rng.uniform(0.9, 1.15))
                if patch is not None:
                    patches.append(patch)
                    labels.append(index)
        truth = np.array(occupied) if occupied else np.zeros((0, 4))
        for _ in range(negatives_per_image):
            size = float(rng.uniform(12, 110))
            cx, cy = float(rng.uniform(0, w)), float(rng.uniform(0, h))
            box = np.array([[cx - size / 2, cy - size / 2,
                             cx + size / 2, cy + size / 2]])
            if len(truth) and iou_matrix(box, truth).max() > 0.05:
                continue
            patch = crop_patch(image, cx, cy, size)
            if patch is not None:
                patches.append(patch)
                labels.append(BACKGROUND)
    return np.asarray(patches), np.asarray(labels)


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #

class PyramidDetector:
    def __init__(self, model, scaler, threshold: float = 0.50,
                 max_per_view: int = 80, sizes: Tuple[int, ...] = SIZES):
        self.model = model
        self.scaler = scaler
        self.threshold = threshold
        self.max_per_view = max_per_view
        self.sizes = sizes

    def raw(self, view: np.ndarray):
        detections = []
        for size in self.sizes:
            factor = PATCH / float(size)
            level = cv2.resize(view, None, fx=factor, fy=factor,
                               interpolation=cv2.INTER_AREA if factor < 1
                               else cv2.INTER_LINEAR)
            features, grid = dense_descriptors(level)
            if len(features) == 0:
                continue
            probabilities = self.model.predict_proba(self.scaler.transform(features))
            objects = probabilities[:, :BACKGROUND]
            scores = objects.max(1)
            for i in np.where(scores >= self.threshold)[0]:
                x, y = grid[i] / factor
                side = PATCH / factor
                detections.append((float(scores[i]), int(objects[i].argmax()),
                                   [float(x), float(y), float(x + side),
                                    float(y + side)]))
        detections.sort(key=lambda t: -t[0])
        return detections[:self.max_per_view]

    def __call__(self, view, region):
        sx1, sy1, sx2, sy2 = region
        fx = (sx2 - sx1) / view.shape[1]
        fy = (sy2 - sy1) / view.shape[0]
        return [(OBJECT_CLASSES[index],
                 [sx1 + b[0] * fx, sy1 + b[1] * fy,
                  sx1 + b[2] * fx, sy1 + b[3] * fy], score)
                for score, index, b in self.raw(view)]


def mine_hard_negatives(detector, images: int, per_image: int = 15):
    image_dir = DATASET / 'images' / 'train'
    label_dir = DATASET / 'labels' / 'train'
    patches = []
    for path in sorted(image_dir.glob('*.jpg'))[-images:]:
        image = cv2.imread(str(path))
        h, w = image.shape[:2]
        boxes = read_labels(label_dir / (path.stem + '.txt'), w, h)
        truth = np.array([[cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
                          for _, cx, cy, bw, bh in boxes]) if boxes else np.zeros((0, 4))
        found = 0
        for _, _, box in detector.raw(image):
            if found >= per_image:
                break
            if len(truth) and iou_matrix(np.array([box]), truth).max() > 0.2:
                continue
            patch = crop_patch(image, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2,
                               box[2] - box[0])
            if patch is not None:
                patches.append(patch)
                found += 1
    return np.asarray(patches) if patches else np.zeros((0, PATCH, PATCH, 3), np.uint8)


def fit(patches, labels):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    features = describe_patches(patches)
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(max_iter=1500, C=1.0, n_jobs=-1,
                               class_weight='balanced')
    scaled = scaler.transform(features)
    model.fit(scaled, labels)
    return model, scaler, float(model.score(scaled, labels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=int, default=800)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--mine-images', type=int, default=120)
    parser.add_argument('--threshold', type=float, default=0.50)
    arguments = parser.parse_args()

    rng = np.random.default_rng(0)
    print(f'descriptor: {HOG_DIMENSION} HOG + {COLOUR_DIMENSION} colour '
          f'= {DIMENSION} dims')
    started = time.perf_counter()
    patches, labels = build_patches('train', arguments.images, rng)
    print(f'{len(patches)} patches '
          f'({int((labels != BACKGROUND).sum())} positive, '
          f'{int((labels == BACKGROUND).sum())} background) '
          f'in {time.perf_counter() - started:.1f}s')

    model, scaler, accuracy = fit(patches, labels)
    print(f'round 0: train accuracy {accuracy:.4f}')
    detector = PyramidDetector(model, scaler, threshold=arguments.threshold)

    for round_index in range(arguments.rounds):
        started = time.perf_counter()
        mined = mine_hard_negatives(detector, arguments.mine_images)
        print(f'round {round_index + 1}: mined {len(mined)} false positives '
              f'in {time.perf_counter() - started:.1f}s')
        if not len(mined):
            break
        patches = np.concatenate([patches, mined])
        labels = np.concatenate([labels, np.full(len(mined), BACKGROUND)])
        model, scaler, accuracy = fit(patches, labels)
        print(f'         refit on {len(patches)} patches, '
              f'train accuracy {accuracy:.4f}')
        detector = PyramidDetector(model, scaler, threshold=arguments.threshold)

    cache = FrameCache()
    for level in (1, 2):
        started = time.perf_counter()
        predictions = run_detector(detector, level=level, cache=cache)
        elapsed = time.perf_counter() - started
        report_detector(f'L{level} | pyramid HOG+colour logistic regression',
                        predictions)
        print(f'  wall time {elapsed:.1f}s for 25 frames '
              f'({elapsed / 25 * 1000:.0f} ms/frame)')


if __name__ == '__main__':
    main()
