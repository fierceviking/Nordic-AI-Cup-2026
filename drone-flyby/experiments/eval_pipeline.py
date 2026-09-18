"""Fast in-process pipeline evaluator: replays helsinki through solution.predict
WITHOUT HTTP, applying the real camera dynamics and the real COCO scorer.

Mirrors local_evaluator.py offline mode but calls predict() directly, so we can
iterate on the detector/tracker/policy quickly. Optionally saves an annotated
video-like montage.
"""
import argparse, sys, time
sys.path.insert(0, '..')

from dtos import IMAGE_WIDTH, IMAGE_HEIGHT
from local_evaluator import (Camera, render_view, build_request, score, CameraRejection)
from utils import frame_numbers, load_frame
from dtos import DroneFlybyPredictRequestDto
import solution


def run(scene='helsinki'):
    frames = frame_numbers(scene)
    cam = Camera()
    feedback = None
    preds = {}
    # reset per-sequence state
    solution._worlds.clear(); solution._policies.clear()
    t_infer = []
    for fi, frame in enumerate(frames):
        img = load_frame(frame, scene)
        enc = render_view(img, cam)
        payload = build_request(frame, fi, cam, enc, feedback)
        req = DroneFlybyPredictRequestDto.model_validate(payload)
        t0 = time.time()
        resp = solution.predict(req)
        t_infer.append((time.time() - t0) * 1000)
        preds[frame] = [
            {'object_id': a.object_id,
             'bbox': (a.bbox[0]*IMAGE_WIDTH, a.bbox[1]*IMAGE_HEIGHT,
                      a.bbox[2]*IMAGE_WIDTH, a.bbox[3]*IMAGE_HEIGHT),
             'confidence': float(a.confidence)} for a in resp.annotations]
        if resp.requested_view is not None:
            rv = resp.requested_view
            try:
                cam.apply(rv.resolution_level, rv.center_x, rv.center_y)
                feedback = None
            except CameraRejection as exc:
                feedback = {'frame': frame, 'requested_view':
                            {'resolution_level': rv.resolution_level,
                             'center_x': rv.center_x, 'center_y': rv.center_y},
                            'reason': str(exc)}
    m, ap = score(scene, preds)
    import statistics
    print(f'\ninfer ms: mean {statistics.mean(t_infer):.0f} median {statistics.median(t_infer):.0f} max {max(t_infer):.0f}')
    print('AP@0.50 by class')
    for n, v in sorted(ap.items(), key=lambda x: -x[1]):
        print(f'  {n:16s} {v:.3f}')
    print(f'\nCOCO mAP@0.50: {m:.3f}')
    return m


if __name__ == '__main__':
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='helsinki')
    a = ap.parse_args()
    run(a.scene)
