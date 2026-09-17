"""Measure the cold-start cost of the endpoint, request by request.

The evaluation service gives each request 3333 ms and emits a frame every
333 ms, so a slow first answer is expensive twice over: it can be recorded as
an error, and every frame that goes by while it is being computed is scored as
a frame with no detections. This sends the same payload repeatedly and prints
each round trip so the cold-start is visible instead of hidden in a mean.
"""

import json
import sys
import time
from pathlib import Path

import requests

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))

from local_evaluator import Camera, build_request, render_view  # noqa: E402
from utils import load_frame  # noqa: E402

URL = 'http://localhost:9053/predict'


def main(n=8):
    image = load_frame(0, 'helsinki')
    camera = Camera()
    payload = build_request(0, 0, camera, render_view(image, camera), None)
    body = json.dumps(payload).encode()
    headers = {'Content-Type': 'application/json'}
    session = requests.Session()

    # Open the connection first, so the handshake is not charged to request 1.
    session.get('http://localhost:9053/', timeout=10)

    print(f'payload {len(body) / 1024:.0f} KB')
    for i in range(n):
        payload['frame'] = i
        payload['frame_index'] = i
        body = json.dumps(payload).encode()
        started = time.perf_counter()
        response = session.post(URL, data=body, headers=headers, timeout=30)
        elapsed = (time.perf_counter() - started) * 1000
        count = len(response.json()['annotations'])
        print(f'  request {i}: {elapsed:8.1f} ms  ({count} detections)')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
