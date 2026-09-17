"""Sweep the run-time constants that are not learned.

None of these need retraining: the detector confidence floor, how long an
unobserved track is kept, how fast its confidence decays, and the duplicate
suppression threshold. The metric ranks every prediction against every other
one, so how confidence is assigned to a thirty-frame-old extrapolation matters
as much as whether it is reported at all.
"""

import argparse
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))
sys.path.insert(0, str(LAB))

import solution  # noqa: E402
from eval_pipeline import LOOPS, as_simulator_solver, loop_policy  # noqa: E402
from simulate import FrameCache, evaluate, replay  # noqa: E402
from solution import Detector, Solver, WorldModel  # noqa: E402

POLICY = 'L0 x3 : L1 x1'


def run(detector, cache, world_kwargs=None):
    world_kwargs = world_kwargs or {}

    class Tuned(WorldModel):
        pass

    for key, value in world_kwargs.items():
        setattr(Tuned, key, value)

    solver = Solver(detector=detector, policy_factory=loop_policy(POLICY),
                    world_factory=Tuned, online_motion=True)
    predictions, _ = replay(as_simulator_solver(solver), cache=cache)
    return evaluate(predictions)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    arguments = parser.parse_args()
    cache = FrameCache()
    results = []

    print('--- detector confidence floor ---')
    for conf in (0.03, 0.05, 0.08, 0.12, 0.20, 0.30):
        detector = Detector(weights=arguments.weights, imgsz=960, confidence=conf)
        value = run(detector, cache)
        print(f'  conf {conf:.2f}: {value:.4f}')
        results.append((f'conf={conf}', value))

    detector = Detector(weights=arguments.weights, imgsz=960, confidence=0.05)

    print('--- how long to keep an unobserved track ---')
    for age in (10, 25, 45, 80):
        value = run(detector, cache, {'MAX_AGE_UNSEEN': age})
        print(f'  MAX_AGE_UNSEEN {age:3d}: {value:.4f}')
        results.append((f'max_age={age}', value))

    print('--- misses tolerated while in view ---')
    for misses in (1, 2, 3, 5):
        value = run(detector, cache, {'MAX_MISSES_IN_VIEW': misses})
        print(f'  MAX_MISSES_IN_VIEW {misses}: {value:.4f}')
        results.append((f'max_misses={misses}', value))

    print('--- association IoU gate ---')
    for gate in (0.15, 0.30, 0.45):
        value = run(detector, cache, {'MATCH_IOU': gate})
        print(f'  MATCH_IOU {gate:.2f}: {value:.4f}')
        results.append((f'match_iou={gate}', value))

    print('--- confidence decay per unobserved frame ---')
    original = WorldModel.report
    for decay in (0.0, 0.02, 0.045, 0.10):
        def report(self, maximum=400, _decay=decay):
            out = []
            for track in self.tracks:
                bbox = solution.clip_to_frame(track.bbox)
                if bbox is None:
                    continue
                staleness = 1.0 / (1.0 + _decay * track.age_since_seen)
                support = min(1.0, 0.55 + 0.15 * track.hits)
                out.append({'object_id': track.object_id, 'bbox': bbox,
                            'confidence': float(min(1.0, max(1e-4,
                                track.score * staleness * support)))})
            out.sort(key=lambda d: -d['confidence'])
            kept = []
            for candidate in out:
                if any(candidate['object_id'] == other['object_id'] and
                       solution.iou(candidate['bbox'], other['bbox']) > 0.55
                       for other in kept):
                    continue
                kept.append(candidate)
                if len(kept) >= maximum:
                    break
            return kept
        WorldModel.report = report
        value = run(detector, cache)
        print(f'  decay {decay:.3f}: {value:.4f}')
        results.append((f'decay={decay}', value))
    WorldModel.report = original

    print('\n=== best of each group ===')
    for name, value in sorted(results, key=lambda kv: -kv[1])[:8]:
        print(f'  {value:.4f}  {name}')


if __name__ == '__main__':
    main()
