"""Does the detector generalise to terrain it has never seen?

Every number reported elsewhere in this log is measured on the same 25 frames
the training sprites and backgrounds were cut from, so none of it is a
generalisation estimate. The competition's own validation set is the only
honest test and it cannot be downloaded - the service pushes it to your
endpoint.

This is the closest honest proxy that can be built locally. The drone
translates, so late frames show ground that early frames never covered. Train
one model whose **backgrounds come only from frames 0-12**, then score it on
frames 18-24. Sprites still come from every frame, because object appearance is
not what is being held out: the same sixteen 3D models appear in every scene,
so having seen them is realistic. What is held out is the terrain.

Compare against the shipped model, which saw backgrounds from all 25 frames
including 18-24. If the held-out model scores about the same, terrain
memorisation is not doing the work and the reported numbers are less inflated
than they look. If it drops sharply, they are.
"""

import argparse
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))
sys.path.insert(0, str(LAB))

from detector_eval import FrameCache, proposal_recall, run_detector  # noqa: E402
from score_subset import score_frames  # noqa: E402
from solution import Detector  # noqa: E402

HELD_OUT = list(range(18, 25))
SEEN_BY_BOTH = list(range(0, 13))


class Adapter:
    def __init__(self, detector):
        self.detector = detector

    def __call__(self, view, region):
        return [(d['object_id'], d['bbox'], d['confidence'])
                for d in self.detector(view, region)]


def evaluate(weights, label, cache, frames, level=0, conf=0.05):
    detector = Detector(weights=weights, imgsz=960, confidence=conf,
                        drop_view_edge=False)
    predictions = run_detector(Adapter(detector), level=level, cache=cache,
                               frames=frames)
    map50, per_class = score_frames(predictions, frames)
    print(f'  {label:34s} mAP@0.50 = {map50:.4f}  over frames '
          f'{frames[0]}..{frames[-1]}')
    return map50, per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shipped', required=True,
                        help='model trained on backgrounds from all frames')
    parser.add_argument('--holdout', required=True,
                        help='model trained on backgrounds from frames 0-12 only')
    arguments = parser.parse_args()

    cache = FrameCache()
    results = {}

    print('\n=== frames 18-24: terrain the holdout model never trained on ===')
    results['shipped/held'] = evaluate(arguments.shipped, 'shipped (saw all terrain)',
                                       cache, HELD_OUT)
    results['holdout/held'] = evaluate(arguments.holdout, 'holdout (backgrounds 0-12)',
                                       cache, HELD_OUT)

    print('\n=== frames 0-12: terrain both models trained on (control) ===')
    results['shipped/seen'] = evaluate(arguments.shipped, 'shipped (saw all terrain)',
                                       cache, SEEN_BY_BOTH)
    results['holdout/seen'] = evaluate(arguments.holdout, 'holdout (backgrounds 0-12)',
                                       cache, SEEN_BY_BOTH)

    held_gap = results['shipped/held'][0] - results['holdout/held'][0]
    seen_gap = results['shipped/seen'][0] - results['holdout/seen'][0]
    print('\n=== reading ===')
    print(f'  gap on held-out terrain (18-24): {held_gap:+.4f}')
    print(f'  gap on shared terrain   (0-12):  {seen_gap:+.4f}')
    print(f'  terrain-memorisation effect:     {held_gap - seen_gap:+.4f}')
    print('\n  The control gap is whatever the two models differ by for reasons')
    print('  other than terrain (dataset size, seed). The difference between')
    print('  the two gaps is the part attributable to having seen the terrain.')

    print('\n=== per class on held-out terrain ===')
    shipped = results['shipped/held'][1]
    holdout = results['holdout/held'][1]
    print(f'  {"class":16s} {"shipped":>8s} {"holdout":>8s} {"delta":>8s}')
    for name in sorted(set(shipped) | set(holdout)):
        a, b = shipped.get(name, 0.0), holdout.get(name, 0.0)
        print(f'  {name:16s} {a:8.3f} {b:8.3f} {b - a:+8.3f}')


if __name__ == '__main__':
    main()
