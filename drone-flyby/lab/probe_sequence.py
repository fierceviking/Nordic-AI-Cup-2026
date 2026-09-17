"""Replay the scene over HTTP and print the time of every single frame.

``local_evaluator`` reports a mean and a max; when the max is twenty times the
median, the mean is not the interesting number and the max needs a frame number
attached to it.
"""

import json
import sys
import time
from pathlib import Path

import requests

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))

from local_evaluator import Camera, CameraRejection, build_request, render_view  # noqa: E402
from utils import frame_numbers, load_frame  # noqa: E402

URL = 'http://localhost:9053/predict'


def main():
    session = requests.Session()
    session.get('http://localhost:9053/', timeout=10)
    camera = Camera()
    headers = {'Content-Type': 'application/json'}
    times = []
    client_times = []
    for index, frame in enumerate(frame_numbers('helsinki')):
        image = load_frame(frame, 'helsinki')
        payload = build_request(frame, index, camera,
                                render_view(image, camera), None)
        # Split the round trip the way local_evaluator does not: the harness
        # times session.post(json=payload), which charges its own serialisation
        # of a 1.4 MB body to the server.
        serialise_started = time.perf_counter()
        body = json.dumps(payload).encode()
        serialise_ms = (time.perf_counter() - serialise_started) * 1000
        started = time.perf_counter()
        response = session.post(URL, data=body, headers=headers, timeout=30)
        elapsed = (time.perf_counter() - started) * 1000
        times.append(elapsed)
        client_times.append(serialise_ms)
        parsed = response.json()
        view = parsed.get('requested_view')
        print(f'  frame {frame:3d} L{camera.resolution_level} '
              f'({camera.center_x:4d},{camera.center_y:4d}) '
              f'server {elapsed:7.1f} ms  client-serialise {serialise_ms:7.1f} ms '
              f' {len(parsed["annotations"]):3d} annotations')
        if view:
            try:
                camera.apply(view['resolution_level'], view['center_x'],
                             view['center_y'])
            except CameraRejection as exc:
                print(f'    REFUSED: {exc}')
    ordered = sorted(times)
    print(f'\n  server round trip: median {ordered[len(ordered) // 2]:.0f} ms   '
          f'p90 {ordered[int(len(ordered) * 0.9)]:.0f} ms   max {ordered[-1]:.0f} ms')
    ordered = sorted(client_times)
    print(f'  harness serialise: median {ordered[len(ordered) // 2]:.0f} ms   '
          f'max {ordered[-1]:.0f} ms')


if __name__ == '__main__':
    main()
