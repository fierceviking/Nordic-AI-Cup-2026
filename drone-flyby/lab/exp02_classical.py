"""Classical CV proposal generators - the "simple features" rung of the ladder.

The objects are CGI models composited onto a real orthophoto, which suggests
three signals that need no training at all:

  * **colour anomaly** - they are grey, white or flat camouflage in a scene of
    green vegetation, brown field and blue-black water;
  * **centre-surround contrast** - a compact region whose mean colour differs
    from the ring around it;
  * **structure** - sharp, geometric, high-frequency edges against a soft,
    natural texture.

Each variant is a class-agnostic proposal generator. It is scored on proposal
recall, because that is the ceiling any classifier bolted on top could reach.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from detector_eval import FrameCache, report_detector, run_detector  # noqa: E402

PLACEHOLDER = 'jammer'


def _to_source(bbox_view, region, view_w, view_h):
    x1, y1, x2, y2 = bbox_view
    sx1, sy1, sx2, sy2 = region
    fx = (sx2 - sx1) / view_w
    fy = (sy2 - sy1) / view_h
    return [sx1 + x1 * fx, sy1 + y1 * fy, sx1 + x2 * fx, sy1 + y2 * fy]


# --------------------------------------------------------------------------- #
# Variant A - Canny + contours (the shipped baseline, for reference)
# --------------------------------------------------------------------------- #

def baseline_edges(view, region, max_proposals=40):
    grey = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(grey, (3, 3), 0), 60, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), 1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        longest = max(w, h)
        if longest < 8 or longest > 320:
            continue
        ratio = cv2.contourArea(contour) / float(w * h or 1)
        out.append((ratio, (x, y, x + w, y + h)))
    out.sort(key=lambda t: -t[0])
    return [(PLACEHOLDER, _to_source(b, region, view.shape[1], view.shape[0]),
             min(0.30, 0.05 + 0.25 * s)) for s, b in out[:max_proposals]]


# --------------------------------------------------------------------------- #
# Variant B - centre-surround colour saliency, multi-scale
# --------------------------------------------------------------------------- #

def _center_surround(lab: np.ndarray, inner: int, outer_factor: int = 4) -> np.ndarray:
    inner = max(3, inner | 1)
    outer = max(inner + 2, inner * outer_factor | 1)
    centre = cv2.blur(lab, (inner, inner))
    surround = cv2.blur(lab, (outer, outer))
    return np.linalg.norm(centre - surround, axis=2)


def saliency_map(view: np.ndarray, scales=(5, 11, 21, 41)) -> np.ndarray:
    lab = cv2.cvtColor(view, cv2.COLOR_BGR2LAB).astype(np.float32)
    total = np.zeros(view.shape[:2], np.float32)
    for scale in scales:
        s = _center_surround(lab, scale)
        s /= (s.max() + 1e-6)
        total = np.maximum(total, s)
    return total


def greyness_map(view: np.ndarray) -> np.ndarray:
    """High where a pixel is achromatic - the signature of the grey CGI models."""
    hsv = cv2.cvtColor(view, cv2.COLOR_BGR2HSV).astype(np.float32)
    saturation = hsv[..., 1] / 255.0
    value = hsv[..., 2] / 255.0
    # bright and unsaturated, and not blown-out sky/water
    return np.clip((1.0 - saturation * 2.5), 0, 1) * np.clip(value * 1.5, 0, 1)


def _components_to_proposals(mask, view, region, min_side, max_side,
                             score_map, max_proposals, pad=1):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        longest = max(w, h)
        if longest < min_side or longest > max_side:
            continue
        if area < 4:
            continue
        patch = score_map[y:y + h, x:x + w]
        score = float(patch.mean())
        box = (max(0, x - pad), max(0, y - pad),
               min(view.shape[1], x + w + pad), min(view.shape[0], y + h + pad))
        out.append((score, box))
    out.sort(key=lambda t: -t[0])
    return [(PLACEHOLDER, _to_source(b, region, view.shape[1], view.shape[0]),
             float(np.clip(s, 0.01, 0.99)))
            for s, b in out[:max_proposals]]


def saliency_detector(view, region, percentile=99.0, min_side=6, max_side=220,
                      max_proposals=40):
    saliency = saliency_map(view)
    threshold = np.percentile(saliency, percentile)
    mask = saliency > max(threshold, 0.05)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                            np.ones((3, 3), np.uint8))
    return _components_to_proposals(mask, view, region, min_side, max_side,
                                    saliency, max_proposals)


# --------------------------------------------------------------------------- #
# Variant C - grey/achromatic anomaly
# --------------------------------------------------------------------------- #

def grey_detector(view, region, percentile=99.0, min_side=6, max_side=220,
                  max_proposals=40):
    grey = greyness_map(view)
    grey = cv2.GaussianBlur(grey, (5, 5), 0)
    threshold = np.percentile(grey, percentile)
    mask = grey > max(threshold, 0.15)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                            np.ones((5, 5), np.uint8))
    return _components_to_proposals(mask, view, region, min_side, max_side,
                                    grey, max_proposals)


# --------------------------------------------------------------------------- #
# Variant D - saliency AND structure, combined
# --------------------------------------------------------------------------- #

def structure_map(view: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    local = cv2.blur(magnitude, (9, 9))
    broad = cv2.blur(magnitude, (61, 61))
    return np.clip(local - broad, 0, None)


def combined_detector(view, region, percentile=98.5, min_side=5, max_side=240,
                      max_proposals=60):
    saliency = saliency_map(view)
    grey = cv2.GaussianBlur(greyness_map(view), (5, 5), 0)
    structure = structure_map(view)
    structure /= (structure.max() + 1e-6)
    score = np.maximum(saliency, np.maximum(grey, structure))
    threshold = np.percentile(score, percentile)
    mask = score > threshold
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                            np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), 1)
    return _components_to_proposals(mask, view, region, min_side, max_side,
                                    score, max_proposals)


# --------------------------------------------------------------------------- #
# Variant E - MSER
# --------------------------------------------------------------------------- #

def mser_detector(view, region, min_side=5, max_side=240, max_proposals=60):
    grey = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create()
    mser.setMinArea(16)
    mser.setMaxArea(40000)
    regions, boxes = mser.detectRegions(grey)
    out = []
    for (x, y, w, h) in boxes:
        longest = max(w, h)
        if longest < min_side or longest > max_side:
            continue
        out.append((float(w * h), (x, y, x + w, y + h)))
    out.sort(key=lambda t: -t[0])
    return [(PLACEHOLDER, _to_source(b, region, view.shape[1], view.shape[0]), 0.3)
            for _, b in out[:max_proposals]]


def main():
    cache = FrameCache()
    variants = [
        ('A canny baseline', baseline_edges),
        ('B colour saliency', saliency_detector),
        ('C grey anomaly', grey_detector),
        ('D saliency+grey+structure', combined_detector),
        ('E MSER', mser_detector),
    ]
    summary = []
    for level in (1, 2):
        for name, detector in variants:
            predictions = run_detector(detector, level=level, cache=cache)
            result = report_detector(f'L{level} | {name}', predictions,
                                     show_classes=(name.startswith('D')))
            summary.append((f'L{level} {name}', result['recall'],
                            result['proposals_per_frame']))
    print('\n\n=== proposal recall summary (class-agnostic, IoU 0.5) ===')
    for name, recall, per_frame in summary:
        print(f'  {recall:.4f}  ({per_frame:6.1f}/frame)  {name}')


if __name__ == '__main__':
    main()
