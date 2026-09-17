"""Is the failure finding the objects, or naming them?

Everything so far has assumed class confusion, on the strength of a visual
triage. That has never been measured. ``proposal_recall`` is class-agnostic -
it asks only whether *some* box landed on the object - while mAP additionally
requires the right label. The gap between them is exactly the classification
tax, and running it per zoom level says whether more pixels would buy anything.

Use the terrain-holdout weights: a model that has to generalise is the only
honest stand-in for the evaluation scene.
"""

import argparse
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))
sys.path.insert(0, str(LAB))

from detector_eval import FrameCache, report_detector, run_detector  # noqa: E402
from solution import Detector  # noqa: E402


class Wrapped:
    """solution.Detector returns dicts; detector_eval wants tuples."""

    def __init__(self, detector):
        self.detector = detector

    def __call__(self, view, region):
        return [(d['object_id'], d['bbox'], d['confidence'])
                for d in self.detector(view, region)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=str(LAB / 'runs' / 'holdout' / 'weights' / 'best.pt'))
    parser.add_argument('--levels', default='0,1,2')
    parser.add_argument('--conf', type=float, default=0.05)
    parser.add_argument('--imgsz', type=int, default=960)
    arguments = parser.parse_args()

    detector = Wrapped(Detector(weights=arguments.weights, imgsz=arguments.imgsz,
                                confidence=arguments.conf, drop_view_edge=False))
    cache = FrameCache()

    summary = []
    for level in (int(v) for v in arguments.levels.split(',')):
        predictions = run_detector(detector, level=level, cache=cache)
        result = report_detector(f'level {level} (full coverage)', predictions,
                                 show_classes=(level == 0))
        summary.append((level, result))

    print('\n=== the classification tax ===')
    print(f'{"level":>6} {"recall":>9} {"mAP50":>9} {"lost to class":>15}')
    for level, r in summary:
        print(f'{level:>6} {r["recall"]:9.4f} {r["map50"]:9.4f} '
              f'{r["recall"] - r["map50"]:15.4f}')


if __name__ == '__main__':
    main()
