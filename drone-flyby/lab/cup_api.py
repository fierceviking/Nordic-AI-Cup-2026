"""Talk to the Nordic AI Cup API: verify, validate, and check status.

The API key is never printed, logged, or written to disk by this script. It is
read from, in order:

  1. the ``NORDIC_API_KEY`` environment variable;
  2. a ``.nordic_key`` file next to this use case (git-ignored).

Endpoints used (auth is the ``x-token`` header):

    GET  /api/v1/usecases/drone-flyby/status
    POST /api/v1/usecases/drone-flyby/verify          {"url": ...}
    POST /api/v1/usecases/drone-flyby/validate/queue  {"url": ...}
    GET  /api/v1/usecases/drone-flyby/validate/queue/{uuid}
    GET  /api/v1/usecases/drone-flyby/validate/queue/{uuid}/attempt

There is deliberately no ``evaluate`` command here. Evaluation is one attempt
per team, for the whole competition, and is not something a script should be
able to fire by accident.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

BASE = 'https://cases.nordicaicup.com/api/v1/usecases/drone-flyby'
KEY_FILE = Path(__file__).resolve().parent.parent / '.nordic_key'


def read_key() -> str:
    token = os.environ.get('NORDIC_API_KEY', '').strip()
    if token:
        return token
    if KEY_FILE.exists():
        token = KEY_FILE.read_text(encoding='utf-8-sig').strip()
        if token:
            return token
    print('No API key found. Either set the environment variable:\n'
          '    $env:NORDIC_API_KEY = "<your team key>"\n'
          f'or write the key (and nothing else) into:\n    {KEY_FILE}',
          file=sys.stderr)
    raise SystemExit(2)


def headers():
    return {'x-token': read_key(), 'Content-Type': 'application/json'}


def show(response):
    print(f'HTTP {response.status_code}')
    try:
        body = response.json()
    except ValueError:
        print(response.text[:2000])
        return None
    # The verify response embeds a full base64 frame; keep it out of the log.
    if isinstance(body, dict) and isinstance(body.get('sample'), dict):
        sample = body['sample']
        view = sample.get('view')
        if isinstance(view, dict) and isinstance(view.get('image'), str):
            view['image'] = f'<{len(view["image"])} base64 chars elided>'
    print(json.dumps(body, indent=2)[:4000])
    return body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command',
                        choices=('status', 'verify', 'validate', 'poll'))
    parser.add_argument('--url', help='your public /predict endpoint')
    parser.add_argument('--uuid', help='queued attempt uuid, for poll')
    parser.add_argument('--watch', action='store_true',
                        help='keep polling until the attempt finishes')
    arguments = parser.parse_args()

    if arguments.command == 'status':
        show(requests.get(f'{BASE}/status', headers=headers(), timeout=30))
        return

    if arguments.command in ('verify', 'validate'):
        if not arguments.url:
            raise SystemExit('--url is required')
        if not arguments.url.rstrip('/').endswith('/predict'):
            print('warning: the URL is used exactly as given, path included. '
                  'It usually has to end in /predict.', file=sys.stderr)
        path = 'verify' if arguments.command == 'verify' else 'validate/queue'
        body = show(requests.post(f'{BASE}/{path}', headers=headers(),
                                  json={'url': arguments.url}, timeout=120))
        if arguments.command == 'validate' and isinstance(body, dict):
            uuid = body.get('queued_attempt_uuid')
            if uuid and arguments.watch:
                poll(uuid, watch=True)
        return

    if arguments.command == 'poll':
        if not arguments.uuid:
            raise SystemExit('--uuid is required')
        poll(arguments.uuid, watch=arguments.watch)


def poll(uuid: str, watch: bool = False):
    while True:
        queued = requests.get(f'{BASE}/validate/queue/{uuid}',
                              headers=headers(), timeout=30)
        print(f'--- queue entry --- HTTP {queued.status_code}')
        try:
            entry = queued.json()
            print(json.dumps(entry, indent=2)[:1500])
        except ValueError:
            entry = {}
            print(queued.text[:500])

        attempt = requests.get(f'{BASE}/validate/queue/{uuid}/attempt',
                               headers=headers(), timeout=30)
        if attempt.status_code == 200:
            print('--- attempt ---')
            try:
                print(json.dumps(attempt.json(), indent=2)[:3000])
            except ValueError:
                print(attempt.text[:1000])
            return
        if not watch:
            print(f'--- attempt --- HTTP {attempt.status_code} (not finished)')
            return
        time.sleep(15)


if __name__ == '__main__':
    main()
