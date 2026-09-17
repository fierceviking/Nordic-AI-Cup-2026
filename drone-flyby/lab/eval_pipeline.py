"""Run the full solver - detector, world model, camera policy - over a scene.

This is the number that matters: it is what ``local_evaluator.py`` would print,
without needing a server. Variants let each piece be ablated so the
contribution of the world model and of the camera policy can be read off
separately.
"""

import argparse
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))
sys.path.insert(0, str(LAB))

from simulate import FrameCache, replay, report  # noqa: E402
from solution import (AttentionPolicy, CameraPolicy, Detector, Solver,  # noqa: E402
                      WorldModel)


class NoMemoryWorldModel(WorldModel):
    """Ablation: report only what is detected in the current view."""

    def advance(self, frame):
        self.tracks = []
        self.last_frame = frame


FULL = (0, 1920, 1080)

# Candidate camera loops. A loop entry is (level, center_x, center_y).
LOOPS = {
    'L0 static': (FULL,),
    'L0/L1 alternating quadrants': (
        FULL, (1, 960, 540), FULL, (1, 2880, 540),
        FULL, (1, 2880, 1620), FULL, (1, 960, 1620)),
    'L0/L1 alternating, top band only': (
        FULL, (1, 960, 540), FULL, (1, 2880, 540)),
    'L0 x3 : L1 x1': (
        FULL, FULL, FULL, (1, 960, 540),
        FULL, FULL, FULL, (1, 2880, 540),
        FULL, FULL, FULL, (1, 2880, 1620),
        FULL, FULL, FULL, (1, 960, 1620)),
    'L1 quadrant raster': (
        (1, 960, 540), (1, 2880, 540), (1, 2880, 1620), (1, 960, 1620)),
    'L1 top-band patrol': (
        (1, 960, 540), (1, 1920, 540), (1, 2880, 540), (1, 1920, 540)),
    'L2 top-band patrol': (
        (2, 480, 270), (2, 1440, 270), (2, 2400, 270), (2, 3360, 270),
        (2, 2400, 270), (2, 1440, 270)),
}


def loop_policy(name):
    return lambda: CameraPolicy(LOOPS[name])


def attention_policy(name):
    return lambda: AttentionPolicy(LOOPS[name])


def as_simulator_solver(solver):
    def run(request):
        return solver(
            sequence_id=request.sequence_id,
            frame=request.frame,
            view=request.image,
            region=request.source_region_xyxy,
            level=request.resolution_level,
            center_x=request.center_x,
            center_y=request.center_y,
            allowed_levels=request.allowed_levels,
            maximum_delta=request.maximum_center_delta,
        )
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=str(LAB / 'runs' / 'v1' / 'weights' / 'best.pt'))
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--conf', type=float, default=0.08)
    parser.add_argument('--variant', default='all')
    parser.add_argument('--verbose', action='store_true')
    arguments = parser.parse_args()

    detectors = {
        True: Detector(weights=arguments.weights, imgsz=arguments.imgsz,
                       confidence=arguments.conf, drop_view_edge=True),
        False: Detector(weights=arguments.weights, imgsz=arguments.imgsz,
                        confidence=arguments.conf, drop_view_edge=False),
    }
    cache = FrameCache()

    # name -> (policy, world model, online motion, drop view-edge detections)
    variants = {}
    for loop_name in LOOPS:
        variants[loop_name] = (loop_policy(loop_name), WorldModel, True, True)
    variants['L0/L1 alternating quadrants, keep view-edge dets'] = (
        loop_policy('L0/L1 alternating quadrants'), WorldModel, True, False)
    variants['L0/L1 alternating quadrants, prior H only'] = (
        loop_policy('L0/L1 alternating quadrants'), WorldModel, False, True)
    variants['L0/L1 alternating quadrants, NO world model'] = (
        loop_policy('L0/L1 alternating quadrants'), NoMemoryWorldModel, True, True)
    variants['L0 static, NO world model'] = (
        loop_policy('L0 static'), NoMemoryWorldModel, True, True)
    variants['attention: L0 + zoom on contested track'] = (
        attention_policy('L0/L1 alternating quadrants'), WorldModel, True, True)

    if arguments.variant != 'all':
        variants = {k: v for k, v in variants.items() if arguments.variant in k}

    results = {}
    for name, (policy_factory, world_class, online, drop_edge) in variants.items():
        solver = Solver(detector=detectors[drop_edge],
                        policy_factory=policy_factory,
                        world_factory=world_class, online_motion=online)
        predictions, stats = replay(as_simulator_solver(solver), cache=cache,
                                    verbose=arguments.verbose)
        results[name] = report(name, predictions, stats)

    print('\n\n=== pipeline summary ===')
    for name, value in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f'  {value:.4f}  {name}')


if __name__ == '__main__':
    main()
