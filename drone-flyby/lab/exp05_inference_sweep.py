"""Sweep the inference settings that are free to choose at run time.

The transmitted image is always 960x540. Running the detector at a larger
``imgsz`` upsamples it, which costs latency but gives the small classes more
pixels to be found in. This measures whether that trade is worth taking, and
what confidence floor to use, without retraining anything.
"""

import argparse
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))
sys.path.insert(0, str(LAB))

from detector_eval import FrameCache, report_detector, run_detector  # noqa: E402
from solution import Detector  # noqa: E402


class Adapter:
    """solution.Detector returns dicts; the harness wants tuples."""

    def __init__(self, detector):
        self.detector = detector

    def __call__(self, view, region):
        return [(d['object_id'], d['bbox'], d['confidence'])
                for d in self.detector(view, region)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--sizes', default='960,1280,1600')
    parser.add_argument('--levels', default='1')
    parser.add_argument('--conf', type=float, default=0.05)
    parser.add_argument('--edge', default='both', choices=('both', 'on', 'off'))
    parser.add_argument('--per-class', action='store_true')
    arguments = parser.parse_args()

    cache = FrameCache()
    rows = []
    edge_options = {'both': (True, False), 'on': (True,), 'off': (False,)}[arguments.edge]
    for imgsz in [int(v) for v in arguments.sizes.split(',')]:
        for drop_edge in edge_options:
            detector = Detector(weights=arguments.weights, imgsz=imgsz,
                                confidence=arguments.conf,
                                drop_view_edge=drop_edge)
            for level in [int(v) for v in arguments.levels.split(',')]:
                started = time.perf_counter()
                predictions = run_detector(Adapter(detector), level=level,
                                           cache=cache)
                elapsed = time.perf_counter() - started
                tiles = {0: 1, 1: 4, 2: 16}[level]
                result = report_detector(
                    f'L{level} imgsz={imgsz} edge-drop={drop_edge}',
                    predictions, show_classes=arguments.per_class)
                per_view = elapsed / (25 * tiles) * 1000
                print(f'  {per_view:.1f} ms per view')
                rows.append((level, imgsz, drop_edge, result['recall'],
                             result['map50'], per_view))

    print('\n=== summary ===')
    print(f'{"level":>5s} {"imgsz":>6s} {"edgedrop":>9s} {"recall":>7s} '
          f'{"mAP50":>7s} {"ms/view":>8s}')
    for level, imgsz, edge, recall, map50, ms in rows:
        print(f'{level:5d} {imgsz:6d} {str(edge):>9s} {recall:7.4f} '
              f'{map50:7.4f} {ms:8.1f}')


if __name__ == '__main__':
    main()
