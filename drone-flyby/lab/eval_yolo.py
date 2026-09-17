"""Evaluate a trained YOLO checkpoint on the supplied scene.

Two separate measurements, deliberately kept apart:

* ``--detector`` tiles every frame at a fixed level so the detector sees all of
  it. That is detector quality with the camera problem removed.
* ``--pipeline`` runs the full solver - detector plus camera policy plus world
  model - through the same replay the local evaluator uses.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))
sys.path.insert(0, str(LAB))

from detector_eval import FrameCache, report_detector, run_detector  # noqa: E402
from dtos import OBJECT_CLASSES  # noqa: E402


class YoloDetector:
    """Adapter from an ultralytics model to the harness's detector signature."""

    def __init__(self, weights: str, confidence: float = 0.05,
                 iou: float = 0.55, imgsz: int = 960, half: bool = True,
                 device: int = 0, max_det: int = 300):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        self.half = half
        self.device = device
        self.max_det = max_det
        self.names = [OBJECT_CLASSES[i] for i in range(len(OBJECT_CLASSES))]
        # Warm up: the first inference is always the slowest and there is no
        # timing allowance for it during an attempt.
        self.model.predict(np.zeros((540, 960, 3), np.uint8), imgsz=imgsz,
                           device=device, half=half, verbose=False)

    def infer(self, view) -> List[Tuple[str, Tuple[float, float, float, float], float]]:
        result = self.model.predict(
            view, imgsz=self.imgsz, conf=self.confidence, iou=self.iou,
            device=self.device, half=self.half, verbose=False,
            max_det=self.max_det)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        return [(self.names[c], tuple(b), float(s))
                for b, s, c in zip(xyxy, confidences, classes)]

    def __call__(self, view, region):
        sx1, sy1, sx2, sy2 = region
        fx = (sx2 - sx1) / view.shape[1]
        fy = (sy2 - sy1) / view.shape[0]
        return [(name, [sx1 + x1 * fx, sy1 + y1 * fy,
                        sx1 + x2 * fx, sy1 + y2 * fy], score)
                for name, (x1, y1, x2, y2), score in self.infer(view)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=str(LAB / 'runs' / 'v1' / 'weights' / 'best.pt'))
    parser.add_argument('--conf', type=float, default=0.05)
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--levels', default='0,1,2')
    arguments = parser.parse_args()

    detector = YoloDetector(arguments.weights, confidence=arguments.conf,
                            imgsz=arguments.imgsz)
    cache = FrameCache()
    summary = []
    for level in [int(v) for v in arguments.levels.split(',')]:
        predictions = run_detector(detector, level=level, cache=cache)
        result = report_detector(f'L{level} | YOLO {Path(arguments.weights).parent.parent.name}',
                                 predictions)
        summary.append((level, result['recall'], result['map50'],
                        result['proposals_per_frame']))
    print('\n=== summary ===')
    for level, recall, map50, per_frame in summary:
        print(f'  L{level}: recall {recall:.4f}  mAP50 {map50:.4f}  '
              f'{per_frame:.1f} det/frame')


if __name__ == '__main__':
    main()
