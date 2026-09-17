"""Where does the per-request time actually go?

The offline harness measures the model. The evaluation service measures the
round trip, and on the supplied scene the two differ by a factor of three. This
breaks one request down into its parts so the difference is attributable rather
than guessed at.
"""

import base64
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))
sys.path.insert(0, str(LAB))

from dtos import DroneFlybyPredictRequestDto  # noqa: E402
from local_evaluator import Camera, build_request, render_view  # noqa: E402
from utils import load_frame  # noqa: E402


def timed(label, function, repeats=20):
    function()                                   # warm
    started = time.perf_counter()
    for _ in range(repeats):
        result = function()
    elapsed = (time.perf_counter() - started) / repeats * 1000
    print(f'  {label:38s} {elapsed:7.2f} ms')
    return result, elapsed


def main():
    import example
    from solution import Detector, MotionEstimator, WorldModel

    image = load_frame(0, 'helsinki')
    camera = Camera()
    encoded = render_view(image, camera)
    payload = build_request(0, 0, camera, encoded, None)
    body = json.dumps(payload)
    print(f'request payload: {len(body) / 1024:.0f} KB\n')

    print('per-request cost breakdown')
    timed('json.loads', lambda: json.loads(body))
    parsed = json.loads(body)
    request, _ = timed('pydantic validate',
                       lambda: DroneFlybyPredictRequestDto.model_validate(parsed))
    view, _ = timed('base64 + PNG decode',
                    lambda: cv2.imdecode(
                        np.frombuffer(base64.b64decode(payload['view']['image']),
                                      np.uint8), cv2.IMREAD_COLOR))

    detector = Detector(weights=str(LAB.parent / 'model' / 'best.pt'), imgsz=960)
    region = payload['view']['source_region_xyxy']
    timed('YOLO inference', lambda: detector(view, region))

    motion = MotionEstimator()
    motion.update(0, view, region)
    timed('ORB motion residual', lambda: motion.update(1, view, region))

    world = WorldModel()
    detections = detector(view, region)
    timed('world model update + report',
          lambda: (world.update(detections, region), world.report()))

    timed('example.predict (everything)', lambda: example.predict(request), 10)


if __name__ == '__main__':
    main()
