"""COCO mAP@0.50 restricted to a subset of frames.

``local_evaluator.score`` scores against every frame in the scene, so a frame
with no predictions costs recall. That is correct for a full attempt and wrong
for evaluating on a held-out subset, where the other frames should simply not
be part of the question.
"""

import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))

from dtos import IMAGE_HEIGHT, IMAGE_WIDTH, OBJECT_CLASSES  # noqa: E402
from utils import load_annotations  # noqa: E402


def _xyxy_to_xywh(bbox: Sequence[float]):
    x1, y1, x2, y2 = (float(c) for c in bbox)
    return x1, y1, x2 - x1, y2 - y1


def score_frames(predictions: Dict[int, List[dict]], frames: Sequence[int],
                 scene: str = 'helsinki'):
    """mAP@0.50 over exactly ``frames``, macro-averaged over classes present."""
    from faster_coco_eval import COCO, COCOeval_faster

    frames = list(frames)
    ground_truth = {f: load_annotations(f, scene) for f in frames}
    present = {a['object_id'] for anns in ground_truth.values() for a in anns}
    evaluated = tuple(name for name in OBJECT_CLASSES if name in present)
    if not evaluated:
        raise ValueError('no ground truth in the selected frames')

    image_id = {f: i for i, f in enumerate(frames, start=1)}
    category_id = {name: i for i, name in enumerate(OBJECT_CLASSES, start=1)}

    annotations = []
    next_id = 1
    for frame in frames:
        for annotation in ground_truth[frame]:
            x, y, w, h = _xyxy_to_xywh(annotation['bbox'])
            annotations.append({
                'id': next_id, 'image_id': image_id[frame],
                'category_id': category_id[annotation['object_id']],
                'bbox': [x, y, w, h], 'area': w * h, 'iscrowd': 0})
            next_id += 1

    coco_gt_dict = {
        'info': {'description': f'subset of {scene}'}, 'licenses': [],
        'images': [{'id': image_id[f], 'file_name': f'frame_{f:06d}.png',
                    'width': IMAGE_WIDTH, 'height': IMAGE_HEIGHT} for f in frames],
        'categories': [{'id': category_id[n], 'name': n, 'supercategory': 'object'}
                       for n in OBJECT_CLASSES],
        'annotations': annotations,
    }

    detections = []
    for frame in frames:
        for detection in predictions.get(frame, []):
            x, y, w, h = _xyxy_to_xywh(detection['bbox'])
            if w <= 0 or h <= 0:
                continue
            detections.append({
                'image_id': image_id[frame],
                'category_id': category_id[detection['object_id']],
                'bbox': [x, y, w, h], 'score': float(detection['confidence'])})

    if not detections:
        return 0.0, {name: 0.0 for name in evaluated}

    coco_gt = COCO(coco_gt_dict)
    coco_dt = coco_gt.loadRes(detections)
    evaluator = COCOeval_faster(coco_gt, coco_dt, 'bbox')
    evaluator.params.imgIds = list(image_id.values())
    evaluator.params.catIds = [category_id[n] for n in evaluated]
    evaluator.params.iouThrs = np.array([0.50])
    evaluator.evaluate()
    evaluator.accumulate()

    precision = evaluator.eval['precision']
    per_class = {}
    for index, name in enumerate(evaluated):
        column = precision[0, :, index, 0, -1]
        valid = column[column > -1]
        per_class[name] = float(np.clip(np.mean(valid) if valid.size else 0.0, 0, 1))
    overall = float(np.clip(sum(per_class.values()) / len(per_class), 0, 1))
    return overall, per_class
