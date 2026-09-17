"""Measure a detector on its own, with the camera problem taken out.

Every frame is tiled at a fixed resolution level so the detector gets to see
all of it. Two numbers come out:

* **proposal recall** - the fraction of ground-truth objects for which *some*
  box was proposed at IoU >= 0.5, ignoring the class. This is the ceiling a
  classifier could reach on top of the proposals, and it is the right thing to
  optimise first.
* **mAP@0.50** - the real metric, once the detector also assigns classes.

A detector is any callable ``(view_bgr, source_region_xyxy) -> list of
(object_id, source_bbox_xyxy, confidence)``. ``object_id`` may be None for a
class-agnostic proposal generator.
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulate import FrameCache, evaluate, render_view, tile_centers  # noqa: E402
from utils import frame_numbers, load_annotations  # noqa: E402

SCENE = 'helsinki'


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    ax1, ay1, ax2, ay2 = boxes_a.T[:, :, None]
    bx1, by1, bx2, by2 = boxes_b.T[:, None, :]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def nms(boxes: List[dict], threshold: float = 0.45, class_agnostic: bool = False):
    """Greedy NMS over dicts with 'bbox' (xyxy), 'confidence', 'object_id'."""
    if not boxes:
        return []
    order = sorted(range(len(boxes)), key=lambda i: -boxes[i]['confidence'])
    arr = np.array([boxes[i]['bbox'] for i in order], float)
    keep: List[int] = []
    suppressed = np.zeros(len(order), bool)
    for i in range(len(order)):
        if suppressed[i]:
            continue
        keep.append(order[i])
        if i + 1 >= len(order):
            break
        ious = iou_matrix(arr[i:i + 1], arr[i + 1:])[0]
        for j, value in enumerate(ious, start=i + 1):
            if suppressed[j] or value < threshold:
                continue
            if class_agnostic or boxes[order[j]]['object_id'] == boxes[order[i]]['object_id']:
                suppressed[j] = True
    return [boxes[i] for i in keep]


def run_detector(
    detector: Callable,
    level: int = 1,
    scene: str = SCENE,
    cache: Optional[FrameCache] = None,
    frames: Optional[Sequence[int]] = None,
    nms_threshold: float = 0.45,
    overlap: float = 0.0,
) -> Dict[int, List[dict]]:
    """Tile every frame at ``level`` and collect detections in source pixels."""
    cache = cache or FrameCache(scene)
    frames = list(frames if frames is not None else frame_numbers(scene))
    centers = tile_centers(level) if overlap <= 0 else overlapped_tile_centers(level, overlap)
    out: Dict[int, List[dict]] = {}
    for number in frames:
        image = cache.frame(number)
        found: List[dict] = []
        for cx, cy in centers:
            view, region = render_view(image, level, cx, cy)
            for object_id, bbox, confidence in detector(view, region):
                found.append({'object_id': object_id,
                              'bbox': [float(c) for c in bbox],
                              'confidence': float(confidence)})
        out[number] = nms(found, nms_threshold, class_agnostic=False)
    return out


def overlapped_tile_centers(level: int, overlap: float) -> List[Tuple[int, int]]:
    from dtos import SOURCE_REGION_SIZES
    from utils import center_bounds_for_level
    width, height = SOURCE_REGION_SIZES[level]
    min_x, max_x, min_y, max_y = center_bounds_for_level(level)
    step_x = max(1, int(width * (1 - overlap)))
    step_y = max(1, int(height * (1 - overlap)))
    xs = sorted({*range(min_x, max_x + 1, step_x), max_x})
    ys = sorted({*range(min_y, max_y + 1, step_y), max_y})
    return [(x, y) for y in ys for x in xs]


def proposal_recall(predictions: Dict[int, List[dict]], scene: str = SCENE,
                    threshold: float = 0.5):
    """Class-agnostic recall, overall and per class."""
    hits = defaultdict(int)
    totals = defaultdict(int)
    for number, ground_truth in ((f, load_annotations(f, scene))
                                 for f in frame_numbers(scene)):
        proposals = np.array([p['bbox'] for p in predictions.get(number, [])], float)
        for annotation in ground_truth:
            totals[annotation['object_id']] += 1
            if len(proposals) == 0:
                continue
            gt = np.array([annotation['bbox']], float)
            if iou_matrix(gt, proposals).max() >= threshold:
                hits[annotation['object_id']] += 1
    overall = sum(hits.values()) / max(1, sum(totals.values()))
    per_class = {k: hits[k] / totals[k] for k in sorted(totals)}
    return overall, per_class


def report_detector(name: str, predictions: Dict[int, List[dict]],
                    scene: str = SCENE, show_classes: bool = True) -> dict:
    recall, per_class_recall = proposal_recall(predictions, scene)
    n = sum(len(v) for v in predictions.values())
    frames = max(1, len(predictions))
    try:
        map50, per_class_ap = evaluate(predictions, scene)
    except Exception as exc:                       # no predictions at all
        map50, per_class_ap = 0.0, {}
        print(f'  (scoring failed: {exc})')
    print(f'\n=== {name} ===')
    print(f'  proposals          {n} ({n / frames:.1f} per frame)')
    print(f'  proposal recall@.5 {recall:.4f}')
    print(f'  mAP@0.50           {map50:.4f}')
    if show_classes:
        print(f'  {"class":16s} {"recall":>7s} {"AP":>7s}')
        for cls in sorted(per_class_recall):
            print(f'  {cls:16s} {per_class_recall[cls]:7.3f} '
                  f'{per_class_ap.get(cls, 0.0):7.3f}')
    return {'recall': recall, 'map50': map50, 'proposals_per_frame': n / frames,
            'per_class_recall': per_class_recall, 'per_class_ap': per_class_ap}
