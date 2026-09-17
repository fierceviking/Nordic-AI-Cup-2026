"""Measure the endpoint through the public tunnel, not over localhost.

The evaluation service reaches us across the internet, so the number that
decides whether frames get dropped is the round trip through Cloudflare, not
the 58 ms measured on loopback.

This machine's router cannot resolve freshly created ``trycloudflare.com``
subdomains, so the hostname is pinned to an address with a ``getaddrinfo``
override. TLS still uses the real hostname for SNI, so the request path is
genuinely the public one.
"""

import argparse
import json
import socket
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))

from local_evaluator import Camera, CameraRejection, build_request, render_view  # noqa: E402
from utils import frame_numbers, load_frame  # noqa: E402


def pin_host(hostname: str, address: str) -> None:
    """Resolve one hostname to a fixed address, leaving everything else alone."""
    original = socket.getaddrinfo

    def patched(host, port, family=0, type=0, proto=0, flags=0):
        if host == hostname:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (address, port))]
        return original(host, port, family, type, proto, flags)

    socket.getaddrinfo = patched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True, help='public /predict URL')
    parser.add_argument('--resolve', help='pin the host to this IP')
    parser.add_argument('--frames', type=int, default=12)
    arguments = parser.parse_args()

    hostname = urlparse(arguments.url).hostname
    if arguments.resolve:
        pin_host(hostname, arguments.resolve)

    session = requests.Session()
    headers = {'Content-Type': 'application/json'}
    root = arguments.url.rsplit('/', 1)[0]
    started = time.perf_counter()
    session.get(root, timeout=30)
    print(f'connection + TLS handshake: {(time.perf_counter() - started) * 1000:.0f} ms')

    camera = Camera()
    times = []
    for index, frame in enumerate(frame_numbers('helsinki')[:arguments.frames]):
        payload = build_request(frame, index, camera,
                                render_view(load_frame(frame, 'helsinki'), camera),
                                None)
        body = json.dumps(payload).encode()
        began = time.perf_counter()
        response = session.post(arguments.url, data=body, headers=headers,
                                timeout=60)
        elapsed = (time.perf_counter() - began) * 1000
        times.append(elapsed)
        parsed = response.json()
        print(f'  frame {frame:3d} L{camera.resolution_level} '
              f'{elapsed:8.1f} ms  HTTP {response.status_code}  '
              f'{len(parsed.get("annotations", []))} annotations')
        view = parsed.get('requested_view')
        if view:
            try:
                camera.apply(view['resolution_level'], view['center_x'],
                             view['center_y'])
            except CameraRejection as exc:
                print(f'    REFUSED: {exc}')

    ordered = sorted(times)
    print(f'\n  median {ordered[len(ordered) // 2]:.0f} ms   '
          f'p90 {ordered[int(len(ordered) * 0.9)]:.0f} ms   max {ordered[-1]:.0f} ms')
    print(f'  frame interval is 333 ms; budget per request is 3333 ms')
    over = sum(1 for t in times if t > 333)
    print(f'  {over}/{len(times)} requests slower than one frame interval')


if __name__ == '__main__':
    main()
