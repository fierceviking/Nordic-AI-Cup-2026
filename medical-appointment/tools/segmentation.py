"""How much does finer utterance segmentation raise the ceiling?

    python tools/segmentation.py --model parakeet

An annotated passage is a clause, median 2.9 s. Units are currently split at
sentence ends and pauses, which makes them sentence-sized and caps the oracle at
0.64. Splitting at commas and clause-opening conjunctions should move the unit
closer to the thing being annotated -- without touching selection, which is the
part nothing has managed to improve.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from solution import asr  # noqa: E402
from solution.windows import build_windows, split_units  # noqa: E402
from utils import (gold_evidence, group_questions_by_conversation,  # noqa: E402
                   temporal_iou)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=asr.DEFAULT_MODEL)
    args = parser.parse_args()

    settings = [
        (pause, clause)
        for clause in (False, True)
        for pause in (0.25, 0.35, 0.45, 0.55)
    ]

    print(f'{"pause":>7}{"clause":>8}{"units":>8}{"median s":>10}'
          f'{"unit oracle":>13}{"window oracle":>15}')
    print('-' * 62)

    for pause, clause in settings:
        unit_best, window_best, counts, lengths = [], [], [], []

        for audio_filename, rows in group_questions_by_conversation():
            transcript = asr.load_cached(Path(audio_filename).stem, args.model)
            if transcript is None:
                raise SystemExit(f'transcribe first: {audio_filename}')

            units = split_units(transcript, pause=pause, split_on_clause=clause)
            windows = build_windows(units, max_units=3)
            counts.append(len(units))
            lengths.extend(unit.end - unit.start for unit in units)

            for row in rows:
                gold = gold_evidence(row)
                if not gold:
                    continue
                unit_best.append(max(
                    temporal_iou(gold, (unit.start, unit.end)) for unit in units
                ))
                window_best.append(max(
                    temporal_iou(gold, (window.start, window.end))
                    for window in windows
                ))

        print(f'{pause:>7.2f}{str(clause):>8}{np.mean(counts):>8.0f}'
              f'{np.median(lengths):>10.2f}'
              f'{np.mean(unit_best):>13.4f}{np.mean(window_best):>15.4f}')

    print('\ngold passages: median 2.88 s, mean 3.21 s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
