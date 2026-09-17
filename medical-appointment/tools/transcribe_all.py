"""Transcribe every supplied conversation once and cache the result.

    python tools/transcribe_all.py --model large-v3

Experiments re-run the answering logic hundreds of times; transcription is the
expensive half and should only happen once per (audio, model) pair.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from solution import asr  # noqa: E402
from utils import AUDIO_DIRECTORY  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=asr.DEFAULT_MODEL)
    parser.add_argument('--beam-size', type=int, default=5)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    files = sorted(
        AUDIO_DIRECTORY.glob('conversation_*.mp3'),
        key=lambda p: int(p.stem.rsplit('_', 1)[1]),
    )
    if args.limit:
        files = files[: args.limit]

    total_audio = 0.0
    total_wall = 0.0

    for index, path in enumerate(files, 1):
        key = path.stem  # conversation_sample_4
        if not args.force and asr.load_cached(key, args.model) is not None:
            print(f'[{index}/{len(files)}] {key}: cached')
            continue

        started = time.time()
        transcript = asr.transcribe_file(
            str(path), model_name=args.model, beam_size=args.beam_size
        )
        elapsed = time.time() - started
        asr.save_cached(key, transcript, args.model)

        total_audio += transcript.duration
        total_wall += elapsed
        print(
            f'[{index}/{len(files)}] {key}: {transcript.duration:6.1f}s audio '
            f'in {elapsed:6.1f}s wall ({elapsed / max(transcript.duration, 1e-6):.2f}x), '
            f'{len(transcript.words)} words'
        )

    if total_wall:
        print(
            f'\ntotal {total_audio:.0f}s audio in {total_wall:.0f}s '
            f'({total_wall / max(total_audio, 1e-6):.3f} x real time)'
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
