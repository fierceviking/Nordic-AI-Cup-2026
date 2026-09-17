"""Measure the inter-frame homography of a recorded sequence.

The matrix in ``solution.py`` was measured on Helsinki. A different flight may
differ, and more importantly this is what lets one hand-drawn box become a
label on every frame the object appears in: annotate once, propagate through H.

Only Level-0 views are used. They cover the whole source frame, so the estimate
is in frame-global coordinates directly and no crop bookkeeping is needed.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))

from solution import DEFAULT_HOMOGRAPHY  # noqa: E402

RECORDINGS = LAB / 'recordings'


def load_level0(sequence: Path):
    """(frame number, L0 view) pairs, in order."""
    out = []
    for meta_path in sorted((sequence / 'meta').glob('*.json')):
        with open(meta_path) as handle:
            meta = json.load(handle)
        if meta.get('resolution_level') != 0:
            continue
        view = sequence / 'views' / (meta_path.stem + '.png')
        if view.exists():
            out.append((meta['frame'], view))
    return out


def estimate(image_a, image_b, n=4000):
    orb = cv2.ORB_create(nfeatures=n)
    kp_a, des_a = orb.detectAndCompute(image_a, None)
    kp_b, des_b = orb.detectAndCompute(image_b, None)
    if des_a is None or des_b is None:
        return None, 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = matcher.knnMatch(des_a, des_b, k=2)
    good = [p[0] for p in raw if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 20:
        return None, 0
    src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    h, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return h, int(mask.sum()) if mask is not None else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', required=True)
    arguments = parser.parse_args()

    sequence = RECORDINGS / arguments.sequence
    views = load_level0(sequence)
    print(f'{len(views)} Level-0 views (whole frame at 960x540)')

    # Views are 960x540 of a 3840x2160 frame, so a homography estimated on them
    # has to be conjugated by the 4x scale to become frame-global.
    scale = np.diag([4.0, 4.0, 1.0])
    inverse = np.diag([0.25, 0.25, 1.0])

    matrices, gaps = [], []
    previous_frame, previous_image = None, None
    for frame, path in views:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if previous_image is not None and frame - previous_frame == 1:
            h, inliers = estimate(previous_image, image)
            if h is not None and inliers >= 40:
                matrices.append(scale @ (h / h[2, 2]) @ inverse)
                gaps.append(inliers)
        previous_frame, previous_image = frame, image

    if not matrices:
        raise SystemExit('could not estimate any consecutive-frame homography')

    stack = np.stack(matrices)
    stack = stack / stack[:, 2:3, 2:3]
    mean = stack.mean(0)
    print(f'estimated from {len(matrices)} consecutive pairs, '
          f'median {int(np.median(gaps))} inliers\n')
    print('measured H (frame-global):')
    print(np.array2string(mean, precision=6, suppress_small=True))
    print('\nstd across pairs:')
    print(np.array2string(stack.std(0), precision=6, suppress_small=True))

    centre = np.array([[[1920.0, 1080.0]]])
    moved = cv2.perspectiveTransform(centre, mean)[0, 0]
    prior = cv2.perspectiveTransform(centre, DEFAULT_HOMOGRAPHY)[0, 0]
    print(f'\nframe centre moves per frame:')
    print(f'  this sequence   dx {moved[0] - 1920:+7.2f}  dy {moved[1] - 1080:+7.2f}')
    print(f'  Helsinki prior  dx {prior[0] - 1920:+7.2f}  dy {prior[1] - 1080:+7.2f}')
    print(f'  difference      {np.hypot(*(moved - prior)):.2f} px per frame')

    out = LAB / 'out' / f'h_{arguments.sequence[:8]}.npy'
    np.save(out, mean)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
