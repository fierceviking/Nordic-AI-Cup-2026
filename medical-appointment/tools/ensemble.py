"""Do the ranking methods fail on the same questions, or different ones?

    python tools/ensemble.py --model parakeet

Every ranker tried has lost to lexical coverage, but that is an argument about
which is *best*, not about whether they are *redundant*. Ensembling works by
variance reduction: it needs the errors to be uncorrelated, not any member to be
better.

So this measures three things in order, and each one gates the next:
  1. how often the methods pick the same passage,
  2. the oracle over methods -- the ceiling for any combiner, including a
     perfect one that knows which method to trust per question,
  3. what actual combiners get (majority region, median boundaries, and a
     union/intersection pair) against the best single method.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from solution import asr  # noqa: E402
from solution.approaches.lexical import _inverse_document_frequency  # noqa: E402
from solution.approaches.neural import NeuralApproach, QUERY_PREFIX  # noqa: E402
from solution.textutil import content_stems, declarative  # noqa: E402
from solution.windows import (build_windows, calibrate,  # noqa: E402
                              matched_run_span, split_units)
from utils import (gold_evidence, group_questions_by_conversation,  # noqa: E402
                   temporal_iou)

TOP_K = 10
CALIB = (0.92, -0.12, -0.50)
METHODS = ['coverage', 'dense', 'cross_encoder', 'entail']


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=asr.DEFAULT_MODEL)
    args = parser.parse_args()

    approach = NeuralApproach(use_reranker=True)
    records: List[Dict] = []

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

        window_vectors = approach.embedder.encode(
            [w.text for w in windows], normalize_embeddings=True,
            batch_size=64, show_progress_bar=False,
        )
        index_of = {id(w): i for i, w in enumerate(windows)}

        annotated = [row for row in rows if gold_evidence(row) is not None]
        if not annotated:
            continue

        statements = [declarative(row['question']) for row in annotated]
        question_vectors = approach.embedder.encode(
            [QUERY_PREFIX + statement for statement in statements],
            normalize_embeddings=True, batch_size=32, show_progress_bar=False,
        )

        for position, row in enumerate(annotated):
            gold = gold_evidence(row)

            matches = approach.lexical.score_windows(
                row['question'], windows, stems, idf
            )
            if not matches:
                continue
            matches = sorted(matches, key=lambda m: m.score, reverse=True)[:TOP_K]
            statement = statements[position]
            question_vector = question_vectors[position]

            scores = {
                'coverage': np.array([m.coverage for m in matches]),
                'dense': np.array([
                    float(question_vector @ window_vectors[index_of[id(m.window)]])
                    for m in matches
                ]),
                'cross_encoder': np.asarray(approach.reranker.predict(
                    [(statement, m.window.text) for m in matches],
                    batch_size=128, show_progress_bar=False,
                ), dtype=float),
                'entail': np.asarray(approach._entailment(
                    [m.window.text for m in matches],
                    [statement] * len(matches),
                )[0], dtype=float),
            }

            spans = [
                calibrate(matched_run_span(m.window, m.matched, units, 3), *CALIB)
                for m in matches
            ]
            ious = np.array([temporal_iou(gold, s) for s in spans])

            records.append({
                'picks': {name: int(np.argmax(value))
                          for name, value in scores.items()},
                'spans': spans, 'ious': ious, 'gold': gold,
            })

        if number % 10 == 0:
            print(f'  ...{number} conversations')

    print(f'\n{len(records)} annotated questions, top-{TOP_K} candidates\n')

    print('how often each method picks the same candidate as coverage:')
    for name in METHODS[1:]:
        agree = np.mean([
            record['picks'][name] == record['picks']['coverage']
            for record in records
        ])
        print(f'  {name:<16}{agree:.1%}')

    all_agree = np.mean([
        len({record['picks'][name] for name in METHODS}) == 1
        for record in records
    ])
    print(f'  all four agree  {all_agree:.1%}')

    print(f'\n{"method":<34}{"mean tIoU":>10}')
    print('-' * 46)
    per_method = {}
    for name in METHODS:
        values = np.array([
            record['ious'][record['picks'][name]] for record in records
        ])
        per_method[name] = values
        print(f'{name:<34}{values.mean():>10.4f}')

    stacked = np.column_stack([per_method[name] for name in METHODS])
    print('-' * 46)
    print(f'{"ORACLE over the four methods":<34}{stacked.max(axis=1).mean():>10.4f}')

    # Does agreement predict correctness? If not, no vote can exploit it.
    agreeing = np.array([
        len({record['picks'][name] for name in METHODS}) == 1
        for record in records
    ])
    print(f'\ncoverage tIoU when all four agree    '
          f'{per_method["coverage"][agreeing].mean():.4f} (n={agreeing.sum()})')
    print(f'coverage tIoU when they disagree     '
          f'{per_method["coverage"][~agreeing].mean():.4f} (n={(~agreeing).sum()})')

    print(f'\n{"combiner":<34}{"mean tIoU":>10}')
    print('-' * 46)

    starts = np.array([
        [record['spans'][record['picks'][name]][0] for name in METHODS]
        for record in records
    ])
    ends = np.array([
        [record['spans'][record['picks'][name]][1] for name in METHODS]
        for record in records
    ])
    golds = np.array([record['gold'] for record in records])

    def score(lo, hi) -> float:
        overlap = np.maximum(0.0, np.minimum(golds[:, 1], hi)
                             - np.maximum(golds[:, 0], lo))
        union = (np.maximum(golds[:, 1], hi) - np.minimum(golds[:, 0], lo))
        return float(np.mean(np.where(union > 0, overlap / union, 0.0)))

    print(f'{"median of the four boundaries":<34}'
          f'{score(np.median(starts, axis=1), np.median(ends, axis=1)):>10.4f}')
    print(f'{"mean of the four boundaries":<34}'
          f'{score(starts.mean(axis=1), ends.mean(axis=1)):>10.4f}')
    print(f'{"union (widest)":<34}'
          f'{score(starts.min(axis=1), ends.max(axis=1)):>10.4f}')
    print(f'{"intersection (narrowest)":<34}'
          f'{score(starts.max(axis=1), np.maximum(ends.min(axis=1), starts.max(axis=1) + 0.05)):>10.4f}')

    # Majority region: keep the span the most methods landed on.
    majority = []
    for record in records:
        counts: Dict[int, int] = {}
        for name in METHODS:
            counts[record['picks'][name]] = counts.get(record['picks'][name], 0) + 1
        winner = max(counts.items(), key=lambda item: (item[1], -item[0]))[0]
        majority.append(record['ious'][winner])
    print(f'{"majority vote on the candidate":<34}{np.mean(majority):>10.4f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
