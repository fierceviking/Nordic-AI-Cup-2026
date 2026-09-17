"""Verify that consecutive frames are related by one constant homography.

The drone translates by a constant vector over planar ground with a fixed
camera orientation, so H(i -> i+1) should be the same matrix for every i. If it
is, a detection made once can be propagated through the rest of the sequence
for free, which is what makes the camera-control problem tractable.

Two independent checks:
  1. estimate H per frame pair from ORB matches on the full frames, and see how
     consistent the matrices are;
  2. use the estimated H to predict where annotated objects move, and compare
     against the ground truth.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import frame_numbers, load_annotations, load_frame  # noqa: E402

OUT = Path(__file__).resolve().parent / 'out'
OUT.mkdir(exist_ok=True)
SCENE = 'helsinki'


def estimate_h(image_a: np.ndarray, image_b: np.ndarray, n: int = 8000):
    """Homography mapping points in A to points in B."""
    grey_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    grey_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=n)
    kp_a, des_a = orb.detectAndCompute(grey_a, None)
    kp_b, des_b = orb.detectAndCompute(grey_b, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = matcher.knnMatch(des_a, des_b, k=2)
    good = [m for m, n_ in raw if m.distance < 0.75 * n_.distance]
    if len(good) < 12:
        return None, 0
    src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    h, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return h, int(mask.sum()) if mask is not None else 0


def warp_points(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, h).reshape(-1, 2)


def main():
    frames = frame_numbers(SCENE)
    homographies = {}

    print('=== per-pair homography from ORB ===')
    prev_image = load_frame(frames[0], SCENE)
    for a, b in zip(frames, frames[1:]):
        image_b = load_frame(b, SCENE)
        h, inliers = estimate_h(prev_image, image_b)
        prev_image = image_b
        if h is None:
            print(f'  {a}->{b}: FAILED')
            continue
        homographies[(a, b)] = h
        # what the homography does to the frame centre and the scale it implies
        c = warp_points(h, [[1920, 1080]])[0]
        print(f'  {a:2d}->{b:2d}: inliers {inliers:5d}  centre -> '
              f'({c[0]:7.1f}, {c[1]:7.1f})')

    hs = np.stack(list(homographies.values()))
    hs = hs / hs[:, 2:3, 2:3]
    print('\n=== consistency of H across pairs (normalised h33=1) ===')
    print('  mean:\n', np.array2string(hs.mean(0), precision=6, suppress_small=True))
    print('  std:\n', np.array2string(hs.std(0), precision=6, suppress_small=True))

    h_mean = hs.mean(0)
    np.save(OUT / 'h_mean.npy', h_mean)

    # ------------------------------------------------ check against annotations
    print('\n=== prediction error on annotated object centres ===')
    for label, get_h in (('per-pair H', lambda a, b: homographies.get((a, b))),
                         ('mean H', lambda a, b: h_mean)):
        errors = []
        for a, b in zip(frames, frames[1:]):
            h = get_h(a, b)
            if h is None:
                continue
            ann_a = {x['object_id']: x['bbox'] for x in load_annotations(a, SCENE)}
            ann_b = {x['object_id']: x['bbox'] for x in load_annotations(b, SCENE)}
            shared = sorted(set(ann_a) & set(ann_b))
            if not shared:
                continue
            pa = np.array([[(ann_a[k][0] + ann_a[k][2]) / 2,
                            (ann_a[k][1] + ann_a[k][3]) / 2] for k in shared])
            pb = np.array([[(ann_b[k][0] + ann_b[k][2]) / 2,
                            (ann_b[k][1] + ann_b[k][3]) / 2] for k in shared])
            pred = warp_points(h, pa)
            errors.extend(np.linalg.norm(pred - pb, axis=1))
        e = np.array(errors)
        print(f'  {label:12s} n={len(e):4d}  mean {e.mean():6.2f} px  '
              f'median {np.median(e):6.2f}  p90 {np.percentile(e, 90):6.2f}  '
              f'max {e.max():6.2f}')

    # ------------------------------------- how far can we propagate before drift
    print('\n=== cumulative drift when chaining the mean H ===')
    for span in (1, 2, 5, 10, 20):
        h_chain = np.linalg.matrix_power(h_mean, span)
        errors = []
        for a in frames:
            b = a + span
            if b > frames[-1]:
                continue
            ann_a = {x['object_id']: x['bbox'] for x in load_annotations(a, SCENE)}
            ann_b = {x['object_id']: x['bbox'] for x in load_annotations(b, SCENE)}
            shared = sorted(set(ann_a) & set(ann_b))
            if not shared:
                continue
            pa = np.array([[(ann_a[k][0] + ann_a[k][2]) / 2,
                            (ann_a[k][1] + ann_a[k][3]) / 2] for k in shared])
            pb = np.array([[(ann_b[k][0] + ann_b[k][2]) / 2,
                            (ann_b[k][1] + ann_b[k][3]) / 2] for k in shared])
            errors.extend(np.linalg.norm(warp_points(h_chain, pa) - pb, axis=1))
        e = np.array(errors)
        print(f'  span {span:2d}: n={len(e):4d} mean {e.mean():7.2f} px  '
              f'p90 {np.percentile(e, 90):7.2f}  max {e.max():7.2f}')

    # ------------------------------- does H also predict the box size change?
    print('\n=== box size prediction (warping the whole box) ===')
    ratios = []
    for a, b in zip(frames, frames[1:]):
        ann_a = {x['object_id']: x['bbox'] for x in load_annotations(a, SCENE)}
        ann_b = {x['object_id']: x['bbox'] for x in load_annotations(b, SCENE)}
        for k in sorted(set(ann_a) & set(ann_b)):
            x1, y1, x2, y2 = ann_a[k]
            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], float)
            w = warp_points(h_mean, corners)
            pw = w[:, 0].max() - w[:, 0].min()
            ph = w[:, 1].max() - w[:, 1].min()
            gx1, gy1, gx2, gy2 = ann_b[k]
            # only compare boxes not clipped by the frame edge
            if gx1 <= 1 or gy1 <= 1 or gx2 >= 3839 or gy2 >= 2159:
                continue
            if x1 <= 1 or y1 <= 1 or x2 >= 3839 or y2 >= 2159:
                continue
            ratios.append((pw / max(gx2 - gx1, 1), ph / max(gy2 - gy1, 1)))
    r = np.array(ratios)
    print(f'  n={len(r)}  width ratio {r[:, 0].mean():.4f} +- {r[:, 0].std():.4f}'
          f'   height ratio {r[:, 1].mean():.4f} +- {r[:, 1].std():.4f}')

    # --------------------------------------------- IoU of propagated boxes
    print('\n=== IoU of propagated boxes vs ground truth ===')
    for span in (1, 3, 5, 10):
        h_chain = np.linalg.matrix_power(h_mean, span)
        ious = []
        for a in frames:
            b = a + span
            if b > frames[-1]:
                continue
            ann_a = {x['object_id']: x['bbox'] for x in load_annotations(a, SCENE)}
            ann_b = {x['object_id']: x['bbox'] for x in load_annotations(b, SCENE)}
            for k in sorted(set(ann_a) & set(ann_b)):
                x1, y1, x2, y2 = ann_a[k]
                corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], float)
                w = warp_points(h_chain, corners)
                pb = [w[:, 0].min(), w[:, 1].min(), w[:, 0].max(), w[:, 1].max()]
                ious.append(iou(pb, ann_b[k]))
        i = np.array(ious)
        print(f'  span {span:2d}: n={len(i):4d} mean IoU {i.mean():.3f}  '
              f'frac>0.5 {float((i > 0.5).mean()):.3f}')


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


if __name__ == '__main__':
    main()
