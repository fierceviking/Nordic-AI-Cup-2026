"""How good could the evidence half get, given the candidate spans we generate?

    python tools/retrieval_ceiling.py --model large-v3

Separates two failures that look the same in the final number: the right
passage never being a candidate, and the right passage being a candidate that
the ranker did not pick. Prints the oracle tIoU over the candidate set (pick
the best candidate per question with the gold span in hand) for a few window
and tightening settings.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from solution import asr  # noqa: E402
from solution.windows import build_windows, split_units  # noqa: E402
from utils import gold_evidence, group_questions_by_conversation, temporal_iou  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=asr.DEFAULT_MODEL)
    args = parser.parse_args()

    per_setting = {}

    for max_units in (1, 2, 3, 4):
        best_ious = []
        counts = []

        for audio_filename, rows in group_questions_by_conversation():
            key = Path(audio_filename).stem
            transcript = asr.load_cached(key, args.model)
            if transcript is None:
                raise SystemExit(f'transcribe first: {key}')

            units = split_units(transcript)
            windows = build_windows(units, max_units=max_units)
            counts.append(len(windows))

            for row in rows:
                gold = gold_evidence(row)
                if not gold:
                    continue
                best_ious.append(max(
                    temporal_iou(gold, (window.start, window.end))
                    for window in windows
                ))

        per_setting[max_units] = (
            sum(best_ious) / len(best_ious),
            statistics.mean(counts),
            sum(1 for iou in best_ious if iou >= 0.5) / len(best_ious),
        )

    print('oracle window selection (upper bound on the evidence half)')
    print(f'  {"max_units":<10}{"mean tIoU":<12}{"candidates":<12}{"frac >= 0.5"}')
    for max_units, (mean_iou, candidates, hit_rate) in per_setting.items():
        print(f'  {max_units:<10}{mean_iou:<12.3f}{candidates:<12.0f}{hit_rate:.3f}')

    # How well a single unit that merely *contains* the gold centre does, which
    # is what a perfect ranker over utterances would give without tightening.
    contained = []
    for audio_filename, rows in group_questions_by_conversation():
        transcript = asr.load_cached(Path(audio_filename).stem, args.model)
        units = split_units(transcript)
        for row in rows:
            gold = gold_evidence(row)
            if not gold:
                continue
            centre = (gold[0] + gold[1]) / 2
            hits = [u for u in units if u.start <= centre <= u.end]
            if hits:
                contained.append(temporal_iou(gold, (hits[0].start, hits[0].end)))
            else:
                contained.append(0.0)
    print(f'\n  single utterance containing the gold centre: mean tIoU '
          f'{sum(contained) / len(contained):.3f}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
