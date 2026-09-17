"""Replay a recorded sequence through any detector and score it.

The recording fixes which views were requested, so the camera policy cannot be
changed - but the detector can. Feeding the same views to a different model
gives a like-for-like comparison on real evaluation terrain, which is the only
honest offline signal we have.

Two detectors are supported: the shipped YOLO via ``solution.Detector``, and an
RF-DETR checkpoint. Both are wrapped to the same interface, so the world model
and reporting are identical and only the detector differs.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))
sys.path.insert(0, str(LAB))

from dtos import OBJECT_CLASSES  # noqa: E402
from solution import Detector, Solver, WorldModel  # noqa: E402

SOURCE_W, SOURCE_H = 3840, 2160


class RFDetrDetector:
    """RF-DETR wrapped to the (view, region) -> list of dicts interface."""

    def __init__(self, checkpoint: str, size: str = 'small', resolution: int = 728,
                 confidence: float = 0.05):
        import rfdetr
        classes = {'nano': 'RFDETRNano', 'small': 'RFDETRSmall',
                   'medium': 'RFDETRMedium', 'large': 'RFDETRLarge'}
        factory = getattr(rfdetr, classes[size])
        self.model = factory(pretrain_weights=checkpoint, resolution=resolution)
        self.confidence = confidence
        self.model.optimize_for_inference()

    def __call__(self, view: np.ndarray, region):
        rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        detections = self.model.predict(rgb, threshold=self.confidence)
        sx1, sy1, sx2, sy2 = region
        height, width = view.shape[:2]
        fx = (sx2 - sx1) / width
        fy = (sy2 - sy1) / height

        names = None
        if getattr(detections, 'data', None):
            names = detections.data.get('class_name')

        out = []
        for i in range(len(detections.xyxy)):
            x1, y1, x2, y2 = detections.xyxy[i]
            if names is not None:
                name = str(names[i])
            else:
                index = int(detections.class_id[i]) - 1
                if not 0 <= index < len(OBJECT_CLASSES):
                    continue
                name = OBJECT_CLASSES[index]
            out.append({
                'object_id': name,
                'bbox': [sx1 + float(x1) * fx, sy1 + float(y1) * fy,
                         sx1 + float(x2) * fx, sy1 + float(y2) * fy],
                'confidence': float(detections.confidence[i]),
            })
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--detector', choices=['yolo', 'rfdetr'], default='yolo')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--resolution', type=int, default=728)
    parser.add_argument('--size', default='small')
    parser.add_argument('--conf', type=float, default=0.05)
    parser.add_argument('--raw', action='store_true',
                        help='skip the world model and score raw detections')
    parser.add_argument('--out', required=True)
    arguments = parser.parse_args()

    if arguments.detector == 'yolo':
        detector = Detector(weights=arguments.weights, confidence=arguments.conf)
    else:
        detector = RFDetrDetector(arguments.weights, arguments.size,
                                  arguments.resolution, arguments.conf)

    root = LAB / 'recordings' / arguments.sequence
    metas = sorted((root / 'meta').glob('*.json'))
    solver = Solver(detector=detector, world_factory=WorldModel)

    written = {}
    for path in metas:
        meta = json.loads(path.read_text())
        image = cv2.imread(str(root / 'views' / (path.stem + '.png')))
        if image is None:
            continue
        region = meta['source_region_xyxy']
        if arguments.raw:
            found = detector(image, region)
        else:
            annotations, _ = solver(
                arguments.sequence, meta['frame'], image, region,
                meta['resolution_level'], meta['center_x'], meta['center_y'],
                (0, 1, 2), 4000.0)
            found = annotations
        written[str(meta['frame_index'])] = [
            {'object_id': str(d['object_id']),
             'bbox': [float(d['bbox'][0]) / SOURCE_W, float(d['bbox'][1]) / SOURCE_H,
                      float(d['bbox'][2]) / SOURCE_W, float(d['bbox'][3]) / SOURCE_H],
             'confidence': float(d['confidence'])}
            for d in found]

    Path(arguments.out).write_text(json.dumps(written))
    total = sum(len(v) for v in written.values())
    print(f'replayed {len(written)} frames, {total} predictions '
          f'({total / max(1, len(written)):.1f} per frame) -> {arguments.out}')


if __name__ == '__main__':
    main()
