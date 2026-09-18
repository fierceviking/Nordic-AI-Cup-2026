"""What is the best tIoU any span policy could reach, given these word timings?

    python tools/ceiling.py --model parakeet

Selection has been the bottleneck all along, but there is a ceiling above it set
by *what spans can be built at all*. This measures that ceiling at four
granularities, so effort goes where there is room rather than where it is
comfortable.
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

MAX_SPAN = 12.0


def best_word_span(words, gold) -> float:
    """Best tIoU over every contiguous run of words -- the timing ceiling."""
    starts = np.array([word.start for word in words])
    ends = np.array([word.end for word in words])

    best = 0.0
    for i in range(len(words)):
        # Runs are nested, so stop as soon as the run outgrows the budget.
        for j in range(i, len(words)):
            if ends[j] - starts[i] > MAX_SPAN:
                break
            iou = temporal_iou(gold, (float(starts[i]), float(ends[j])))
            if iou > best:
                best = iou
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=asr.DEFAULT_MODEL)
    args = parser.parse_args()

    scores = {
        'unit (single utterance)': [],
        'window (1-3 utterances)': [],
        'any word run': [],
        'gold snapped to word edges': [],
    }

    for audio_filename, rows in group_questions_by_conversation():
        transcript = asr.load_cached(Path(audio_filename).stem, args.model)
        if transcript is None:
            raise SystemExit(f'transcribe first: {audio_filename}')

        units = split_units(transcript)
        windows = build_windows(units, max_units=3)
        words = transcript.words
        edges = np.array(
            [word.start for word in words] + [words[-1].end for _ in (0,)]
        )

        for row in rows:
            gold = gold_evidence(row)
            if not gold:
                continue

            scores['unit (single utterance)'].append(max(
                temporal_iou(gold, (unit.start, unit.end)) for unit in units
            ))
            scores['window (1-3 utterances)'].append(max(
                temporal_iou(gold, (window.start, window.end))
                for window in windows
            ))
            scores['any word run'].append(best_word_span(words, gold))

            # How much the 80 ms frame grid alone costs, before any selection.
            start = float(edges[np.argmin(np.abs(edges - gold[0]))])
            end = float(edges[np.argmin(np.abs(edges - gold[1]))])
            scores['gold snapped to word edges'].append(
                temporal_iou(gold, (start, max(end, start + 0.05)))
            )

    print(f'{len(scores["unit (single utterance)"])} annotated spans\n')
    print(f'{"candidate granularity":<32}{"oracle tIoU":>12}')
    print('-' * 45)
    for name, values in scores.items():
        print(f'{name:<32}{np.mean(values):>12.4f}')

    print('\nscore if selection were perfect (accuracy held at 0.964):')
    for name, values in scores.items():
        print(f'  {name:<30}{0.4 * 0.964 + 0.6 * np.mean(values):>8.3f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
