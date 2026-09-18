"""Use entailment to veto a bad pick rather than to rank the good ones.

    python tools/veto.py --model mlx-large-v3-turbo

Entailment answers the yes/no half at 0.96 but ranks spans worse than word
overlap (experiment 14), and blending the two has failed repeatedly. Those are
both *ranking* uses. This is a different one: keep coverage as the ranker, walk
down its ordering, and take the first candidate the reader does not reject.

The distinction matters because the reader is good at a binary judgement on one
passage and bad at ordering ten of them -- which is exactly the asymmetry the
0.96 accuracy and the 0.48 ranking tIoU already showed.
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

TOP_K = 24
CALIB = (0.98, 0.22, 0.02)
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
DEPTHS = (5, 8, 10, 14, 18, 24)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=asr.DEFAULT_MODEL)
    parser.add_argument('--span-units', type=int, default=2)
    args = parser.parse_args()

    approach = NeuralApproach()
    records: List[dict] = []

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

            entail, _ = approach._entailment(
                [m.window.text for m in matches],
                [statement] * len(matches),
            )
            spans = [
                calibrate(
                    matched_run_span(m.window, m.matched, units,
                                     args.span_units, True),
                    *CALIB,
                )
                for m in matches
            ]
            records.append({
                'entail': np.asarray(entail, dtype=float),
                'ious': np.array([temporal_iou(gold, s) for s in spans]),
            })

        if number % 10 == 0:
            print(f'  ...{number} conversations')

    baseline = float(np.mean([record['ious'][0] for record in records]))
    oracle = float(np.mean([record['ious'].max() for record in records]))

    print(f'\n{len(records)} annotated questions, top-{TOP_K} candidates\n')
    print(f'{"rule":<46}{"mean tIoU":>10}')
    print('-' * 58)
    print(f'{"argmax coverage (shipped)":<46}{baseline:>10.4f}')
    print(f'{"oracle over these candidates":<46}{oracle:>10.4f}')

    header = ''.join(f'{threshold:>9.2f}' for threshold in THRESHOLDS)
    print(f'\nveto threshold across the top ->\n{"depth":>7}{header}')
    for depth in DEPTHS:
        cells = []
        for threshold in THRESHOLDS:
            total = []
            for record in records:
                scores = record['entail'][:depth]
                passing = np.flatnonzero(scores >= threshold)
                total.append(
                    record['ious'][int(passing[0])] if len(passing)
                    else record['ious'][0]
                )
            cells.append(float(np.mean(total)))
        print(f'{depth:>7}' + ''.join(f'{cell:>9.4f}' for cell in cells))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
