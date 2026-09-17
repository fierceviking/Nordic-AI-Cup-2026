"""Drive the camera policy for a long sequence and assert every command is legal.

The supplied scene is 25 frames; an attempt is 250. A policy bug that only
shows up after the loop has wrapped a few times, or after a command is refused,
would not appear locally. This runs the policy against the evaluator's own
``Camera`` for 500 frames and fails loudly on any refusal.

Also checks the two rules that are easy to get wrong: Level 0 and Level 2
cannot reach each other directly, and a Level-0 request must use the exact
frame centre.
"""

import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))
sys.path.insert(0, str(LAB))

from dtos import ALLOWED_RESOLUTION_LEVELS, MAXIMUM_CENTER_DELTA_PIXELS  # noqa: E402
from eval_pipeline import LOOPS  # noqa: E402
from local_evaluator import Camera, CameraRejection  # noqa: E402
from solution import CameraPolicy  # noqa: E402


def check(loop_name, loop, frames=500):
    policy = CameraPolicy(loop)
    camera = Camera()
    refusals = []
    visits = {}
    for frame in range(frames):
        requested = policy(
            camera.resolution_level, camera.center_x, camera.center_y,
            list(ALLOWED_RESOLUTION_LEVELS[camera.resolution_level]),
            MAXIMUM_CENTER_DELTA_PIXELS[camera.resolution_level])
        if requested is None:
            continue
        level, cx, cy = requested
        for value in (level, cx, cy):
            assert isinstance(value, int) and not isinstance(value, bool), \
                f'{loop_name}: non-int in requested view {requested}'
        try:
            camera.apply(level, cx, cy)
        except CameraRejection as exc:
            refusals.append((frame, requested, str(exc)))
        key = (camera.resolution_level, camera.center_x, camera.center_y)
        visits[key] = visits.get(key, 0) + 1

    status = 'OK ' if not refusals else 'FAIL'
    print(f'{status} {loop_name:36s} {frames} frames, '
          f'{len(refusals)} refusals, {len(visits)} distinct positions')
    for frame, requested, reason in refusals[:5]:
        print(f'       frame {frame}: {requested} -> {reason}')
    if not refusals:
        share = {k: v / frames for k, v in sorted(visits.items())}
        for (level, cx, cy), fraction in sorted(share.items(),
                                                key=lambda kv: -kv[1])[:6]:
            print(f'       L{level} ({cx:4d},{cy:4d})  {fraction * 100:5.1f}% of frames')
    return not refusals


if __name__ == '__main__':
    ok = True
    for name, loop in LOOPS.items():
        ok &= check(name, loop)
    ok &= check('default (shipped)', None)
    print('\nall policies legal' if ok else '\nSOME POLICIES EMIT ILLEGAL COMMANDS')
    sys.exit(0 if ok else 1)
