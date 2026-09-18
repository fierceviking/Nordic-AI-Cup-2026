"""Does span selection need a fine-tuned model, or just a better ranking rule?

    python tools/span_ranker.py --build      # cache features (slow, once)
    python tools/span_ranker.py              # compare ranking rules

Accuracy is already 0.96, so the whole remaining score is in the evidence half:
picking, out of ~157 candidate windows, the one whose tightened span overlaps the
annotation. Lexical coverage gets 0.40 and a perfect chooser would get 0.70.

This measures how much of that gap frozen models can close when a *small* model
is fitted on top of their scores, rather than fine-tuning any of them. Every
number is leave-one-conversation-out, because the 10 questions about one
conversation share an ASR transcript, a topic and a speaker pair.
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
from solution.approaches.neural import NeuralApproach  # noqa: E402
from solution.approaches.lexical import _inverse_document_frequency  # noqa: E402
from solution.textutil import content_stems, declarative  # noqa: E402
from solution.windows import build_windows, split_units, tighten  # noqa: E402
from utils import (gold_evidence, group_questions_by_conversation,  # noqa: E402
                   temporal_iou)

CACHE = ROOT / 'runs' / 'span_features.npz'
TOP_K = 20

FEATURES = [
    'coverage', 'coverage_rank', 'coverage_gap_to_best',
    'dense', 'cross_encoder', 'entail', 'contra',
    'span_duration', 'window_duration', 'n_units', 'n_words',
    'position', 'gap_before', 'gap_after',
]


def build(model_name: str) -> None:
    approach = NeuralApproach(use_reranker=True)

    rows: List[List[float]] = []
    targets: List[float] = []
    groups: List[str] = []
    question_ids: List[int] = []

    conversations = group_questions_by_conversation()
    question_counter = 0

    for number, (transcript_id, questions) in enumerate(conversations, 1):
        key = Path(transcript_id).stem  # transcribe_all caches by file stem
        transcript = asr.load_cached(key, model_name)
        if transcript is None:
            raise SystemExit(f'no cached transcript for {key}; run transcribe_all')

        units = split_units(transcript)
        windows = build_windows(units, max_units=approach.max_units)
        window_stems = [set(content_stems(window.text)) for window in windows]
        idf = _inverse_document_frequency(units)

        window_vectors = approach.embedder.encode(
            [window.text for window in windows],
            normalize_embeddings=True, batch_size=64, show_progress_bar=False,
        )

        # Only the annotated positives carry a target to rank against.
        positives = [row for row in questions if gold_evidence(row) is not None]
        if not positives:
            continue

        statements = [declarative(row['question']) for row in positives]
        from solution.approaches.neural import QUERY_PREFIX

        question_vectors = approach.embedder.encode(
            [QUERY_PREFIX + statement for statement in statements],
            normalize_embeddings=True, show_progress_bar=False,
        )
        similarity = question_vectors @ window_vectors.T

        for index, row in enumerate(positives):
            matches = approach.lexical.score_windows(
                row['question'], windows, window_stems, idf
            )
            if not matches:
                continue

            matches = sorted(matches, key=lambda m: m.score, reverse=True)[:TOP_K]
            gold = gold_evidence(row)

            window_index = {id(window): i for i, window in enumerate(windows)}
            dense = np.array([
                similarity[index][window_index[id(match.window)]]
                for match in matches
            ])
            cross = np.asarray(approach.reranker.predict(
                [(statements[index], match.window.text) for match in matches],
                batch_size=128, show_progress_bar=False,
            ), dtype=float)

            entail, contra = approach._entailment(
                [approach._premise(units, match) for match in matches],
                [statements[index]] * len(matches),
            )

            best_coverage = max(match.coverage for match in matches)
            for rank, match in enumerate(matches):
                span = tighten(
                    match.window, match.matched, pad=approach.span_pad,
                    min_duration=approach.min_span, max_duration=approach.max_span,
                )
                first, last = match.window.first_unit, match.window.last_unit
                rows.append([
                    match.coverage,
                    rank,
                    best_coverage - match.coverage,
                    float(dense[rank]),
                    float(cross[rank]),
                    float(entail[rank]),
                    float(contra[rank]),
                    span[1] - span[0],
                    match.window.duration,
                    last - first + 1,
                    len(match.window.words),
                    match.window.start / max(transcript.duration, 1e-6),
                    match.window.start - units[first - 1].end if first > 0 else 5.0,
                    units[last + 1].start - match.window.end
                    if last + 1 < len(units) else 5.0,
                ])
                targets.append(temporal_iou(gold, span))
                groups.append(transcript_id)
                question_ids.append(question_counter)

            question_counter += 1

        print(f'[{number}/{len(conversations)}] {transcript_id}: '
              f'{len(positives)} positives, {len(rows)} rows so far')

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        CACHE,
        x=np.array(rows, dtype=float), y=np.array(targets, dtype=float),
        groups=np.array(groups), questions=np.array(question_ids),
    )
    print(f'\nwrote {CACHE} — {len(rows)} rows, {question_counter} questions')


def selected_tiou(scores, y, questions) -> float:
    """Mean tIoU when the highest-scoring candidate is emitted per question."""
    total = 0.0
    for question in np.unique(questions):
        mask = questions == question
        total += y[mask][int(np.argmax(scores[mask]))]
    return total / len(np.unique(questions))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--model', default=asr.DEFAULT_MODEL)
    args = parser.parse_args()

    if args.build or not CACHE.exists():
        build(args.model)

    data = np.load(CACHE, allow_pickle=True)
    x, y, groups, questions = data['x'], data['y'], data['groups'], data['questions']
    n_questions = len(np.unique(questions))
    print(f'\n{len(x)} candidates over {n_questions} annotated questions, '
          f'{len(np.unique(groups))} conversations\n')

    print(f'{"rule":<34} {"mean tIoU":>9}')
    print('-' * 45)

    oracle = np.mean([
        y[questions == q].max() for q in np.unique(questions)
    ])

    for name in ('coverage', 'dense', 'cross_encoder', 'entail'):
        column = FEATURES.index(name)
        print(f'{"argmax " + name:<34} '
              f'{selected_tiou(x[:, column], y, questions):>9.4f}')

    # A ranker fitted on frozen scores. Leave-one-conversation-out, so the
    # number is what a held-out conversation would actually get.
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    for label, make in (
        ('ridge (14 feats, LOCO)', lambda: Ridge(alpha=10.0)),
        ('gbm (14 feats, LOCO)',
         lambda: GradientBoostingRegressor(
             n_estimators=150, max_depth=2, learning_rate=0.05,
             subsample=0.8, random_state=0)),
    ):
        predictions = np.zeros(len(y))
        for group in np.unique(groups):
            train = groups != group
            scaler = StandardScaler().fit(x[train])
            model = make().fit(scaler.transform(x[train]), y[train])
            predictions[~train] = model.predict(scaler.transform(x[~train]))
        print(f'{label:<34} {selected_tiou(predictions, y, questions):>9.4f}')

    print('-' * 45)
    print(f'{"oracle (best candidate)":<34} {oracle:>9.4f}')

    scaler = StandardScaler().fit(x)
    weights = Ridge(alpha=10.0).fit(scaler.transform(x), y).coef_
    print('\nridge weights (standardised), refit on everything:')
    for name, weight in sorted(
        zip(FEATURES, weights), key=lambda p: -abs(p[1])
    ):
        print(f'  {name:<22} {weight:+.3f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
