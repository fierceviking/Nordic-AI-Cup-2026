"""Experiment 3: train and honestly score the feature classifier.

    python tools/train_feature_ml.py --model large-v3

39 conversations is not much, so the number that matters is the out-of-fold
one: GroupKFold by conversation, so no question is scored by a model that has
seen another question about the same audio. The fitted model is then retrained
on everything and written to models/feature_ml.joblib for serving.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from offline_eval import report, summarise  # type: ignore  # noqa: E402
from solution import asr  # noqa: E402
from solution.features import FEATURE_NAMES, FeatureExtractor  # noqa: E402
from solution.windows import tighten  # noqa: E402
from utils import gold_evidence, group_questions_by_conversation, temporal_iou  # noqa: E402


def build_dataset(model: str, max_units: int, length_penalty: float):
    extractor = FeatureExtractor(max_units=max_units, length_penalty=length_penalty)

    features: List[List[float]] = []
    labels: List[int] = []
    groups: List[str] = []
    rows_out = []

    for audio_filename, rows in group_questions_by_conversation():
        transcript = asr.load_cached(Path(audio_filename).stem, model)
        if transcript is None:
            raise SystemExit(f'transcribe first: {audio_filename}')

        prepared = extractor.prepare(transcript)

        for row in rows:
            vector, best, _ = extractor.features_for(
                row['question'], transcript, prepared
            )
            span = (
                tighten(best.window, best.matched, pad=0.3,
                        min_duration=2.0, max_duration=6.0)
                if best else None
            )

            features.append(vector)
            labels.append(int(row['label']))
            groups.append(row['transcript_id'])
            rows_out.append((row, span))

    return np.array(features), np.array(labels), np.array(groups), rows_out


def score_predictions(rows_out, predictions) -> dict:
    records = []
    for (row, span), prediction in zip(rows_out, predictions):
        gold = gold_evidence(row)
        label = int(row['label'])
        iou = temporal_iou(gold, span) if (label == 1 and gold) else None
        records.append({
            'question_id': row['question_id'],
            'transcript_id': row['transcript_id'],
            'question': row['question'],
            'question_type': row['question_type'],
            'label': label,
            'prediction': int(prediction),
            'correct': int(int(prediction) == label),
            'gold_start': gold[0] if gold else '',
            'gold_end': gold[1] if gold else '',
            'pred_start': span[0] if span else '',
            'pred_end': span[1] if span else '',
            'tiou': iou if iou is not None else '',
        })
    return summarise(records, [])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=asr.DEFAULT_MODEL)
    parser.add_argument('--max-units', type=int, default=3)
    parser.add_argument('--length-penalty', type=float, default=0.0)
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()

    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, groups, rows_out = build_dataset(
        args.model, args.max_units, args.length_penalty
    )
    print(f'dataset {X.shape[0]} questions x {X.shape[1]} features')

    candidates = {
        'logreg': make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)
        ),
        'logreg_c0.2': make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=0.2)
        ),
        'hgb': HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, learning_rate=0.08,
            min_samples_leaf=10, l2_regularization=1.0, random_state=0,
        ),
    }

    best_name, best_score = None, -1.0
    for name, model in candidates.items():
        out_of_fold = np.zeros(len(y), dtype=int)
        splitter = GroupKFold(n_splits=args.folds)

        for train_index, test_index in splitter.split(X, y, groups):
            model.fit(X[train_index], y[train_index])
            out_of_fold[test_index] = model.predict(X[test_index])

        summary = score_predictions(rows_out, out_of_fold)
        print(report(summary, f'\n{name}  (out-of-fold, GroupKFold by conversation)'))

        if summary['score'] > best_score:
            best_name, best_score = name, summary['score']

    print(f'\nbest out-of-fold: {best_name} at {best_score:.3f}')

    final = candidates[best_name]
    final.fit(X, y)

    if hasattr(final, 'coef_') or hasattr(final[-1], 'coef_'):
        estimator = final[-1] if hasattr(final, 'steps') else final
        print('\nlogistic weights (standardised features)')
        for name, weight in sorted(
            zip(FEATURE_NAMES, estimator.coef_[0]),
            key=lambda item: abs(item[1]), reverse=True,
        ):
            print(f'  {name:<22} {weight:+.3f}')

    path = ROOT / 'models' / 'feature_ml.joblib'
    path.parent.mkdir(exist_ok=True)
    joblib.dump({'model': final, 'features': FEATURE_NAMES}, path)
    print(f'\nwrote {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
