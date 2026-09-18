"""Does entailment rank spans better when it only sees the window itself?

    python tools/nli_span.py --model parakeet

Experiment 4 found that reranking by entailment wrecked the evidence half, and
experiment 7b found entailment the worst single ranking signal (0.294). But in
both, the premise carried `context_units=2` of neighbouring utterances, so a
window scored well when its *neighbours* held the fact -- exactly the wrong
thing when the window is what gets emitted.

This scores the same candidates with and without that context, to separate "NLI
is bad at localisation" from "NLI was being asked about the wrong text".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from solution import asr  # noqa: E402
from solution.approaches.lexical import _inverse_document_frequency  # noqa: E402
from solution.approaches.neural import NeuralApproach  # noqa: E402
from solution.textutil import content_stems, declarative  # noqa: E402
from solution.windows import (build_windows, calibrate,  # noqa: E402
                              matched_run_span, split_units)
from utils import (gold_evidence, group_questions_by_conversation,  # noqa: E402
                   temporal_iou)

TOP_K = 10
CALIB = (0.92, -0.09, -0.48)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=asr.DEFAULT_MODEL)
    args = parser.parse_args()

    approach = NeuralApproach()
    rows_out: List[dict] = []

    for number, (audio_filename, rows) in enumerate(
        group_questions_by_conversation(), 1
    ):
        transcript = asr.load_cached(Path(audio_filename).stem, args.model)
        if transcript is None:
            raise SystemExit(f'transcribe first: {audio_filename}')

        units = split_units(transcript)
        windows = build_windows(units, max_units=3)
        stems = [set(content_stems(w.text)) for w in windows]
        idf = _inverse_document_frequency(units)

        for row in rows:
            gold = gold_evidence(row)
            if not gold:
                continue

            matches = approach.lexical.score_windows(
                row['question'], windows, stems, idf
            )
            if not matches:
                continue
            matches = sorted(matches, key=lambda m: m.score, reverse=True)[:TOP_K]
            statement = declarative(row['question'])

            bare, _ = approach._entailment(
                [match.window.text for match in matches],
                [statement] * len(matches),
            )
            with_context, _ = approach._entailment(
                [approach._premise(units, match) for match in matches],
                [statement] * len(matches),
            )

            spans = [
                calibrate(
                    matched_run_span(m.window, m.matched, units, 3), *CALIB
                )
                for m in matches
            ]
            rows_out.append({
                'iou': np.array([temporal_iou(gold, s) for s in spans]),
                'coverage': np.array([m.coverage for m in matches]),
                'bare': np.asarray(bare, dtype=float),
                'context': np.asarray(with_context, dtype=float),
            })

        if number % 10 == 0:
            print(f'  ...{number} conversations')

    def pick(score_fn) -> float:
        return float(np.mean([
            row['iou'][int(np.argmax(score_fn(row)))] for row in rows_out
        ]))

    def scaled(values):
        return (values - values.min()) / (np.ptp(values) + 1e-6)

    print(f'\n{len(rows_out)} questions, top-{TOP_K} candidates, '
          f'matched_run spans + calibration\n')
    print(f'{"ranking signal":<40}{"mean tIoU":>10}')
    print('-' * 52)
    print(f'{"argmax coverage (shipped)":<40}{pick(lambda r: r["coverage"]):>10.4f}')
    print(f'{"argmax entailment, with context":<40}{pick(lambda r: r["context"]):>10.4f}')
    print(f'{"argmax entailment, window only":<40}{pick(lambda r: r["bare"]):>10.4f}')

    for weight in (0.2, 0.3, 0.4, 0.5, 0.7):
        blended = pick(
            lambda r, w=weight: w * scaled(r['bare']) + (1 - w) * r['coverage']
        )
        print(f'{f"  blend {weight:.1f} window-NLI + coverage":<40}{blended:>10.4f}')

    print('-' * 52)
    print(f'{"oracle over these candidates":<40}'
          f'{float(np.mean([r["iou"].max() for r in rows_out])):>10.4f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
