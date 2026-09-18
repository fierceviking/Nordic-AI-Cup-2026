"""Is the difference between two runs real, or an artefact of 39 conversations?

    python tools/compare.py runs/a.csv runs/b.csv

Paired bootstrap, resampling **conversations** rather than questions, because the
ten questions about one consultation share a transcript, a topic and a speaker
pair. Reports the difference in score with a confidence interval.

This is the check that should gate every accepted change. Experiment 14's
`skip_prompt` was adopted on an offline gain that was never tested this way, and
it then lost on validation.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import temporal_iou  # noqa: E402

ACCURACY_WEIGHT = 0.4
TIOU_WEIGHT = 0.6


def load(path: Path) -> Dict[str, List[dict]]:
    by_conversation: Dict[str, List[dict]] = defaultdict(list)
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            by_conversation[row['transcript_id']].append(row)
    return by_conversation


def score(rows: List[dict]) -> Tuple[float, float, int, int]:
    """(correct answers, summed tIoU, question count, annotated count)."""
    correct = sum(1 for row in rows if row['correct'] == '1')

    total_iou, annotated = 0.0, 0
    for row in rows:
        if not row.get('gold_start') or not row.get('gold_end'):
            continue
        annotated += 1
        gold = (float(row['gold_start']), float(row['gold_end']))
        if row.get('pred_start') and row.get('pred_end'):
            total_iou += temporal_iou(
                gold, (float(row['pred_start']), float(row['pred_end']))
            )
    return correct, total_iou, len(rows), annotated


def overall(per_conversation, keys) -> float:
    correct = sum(per_conversation[key][0] for key in keys)
    iou = sum(per_conversation[key][1] for key in keys)
    questions = sum(per_conversation[key][2] for key in keys)
    annotated = sum(per_conversation[key][3] for key in keys)

    accuracy = correct / max(questions, 1)
    mean_iou = iou / max(annotated, 1)
    return ACCURACY_WEIGHT * accuracy + TIOU_WEIGHT * mean_iou


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--samples', type=int, default=10000)
    args = parser.parse_args()

    left, right = load(args.baseline), load(args.candidate)
    keys = sorted(set(left) & set(right))

    left_stats = {key: score(left[key]) for key in keys}
    right_stats = {key: score(right[key]) for key in keys}

    base = overall(left_stats, keys)
    cand = overall(right_stats, keys)

    rng = np.random.default_rng(0)
    differences = np.empty(args.samples)
    for index in range(args.samples):
        drawn = rng.choice(len(keys), size=len(keys), replace=True)
        sample = [keys[i] for i in drawn]
        differences[index] = overall(right_stats, sample) - overall(left_stats, sample)

    low, high = np.percentile(differences, [2.5, 97.5])
    better = float(np.mean(differences > 0))

    print(f'{len(keys)} conversations\n')
    print(f'baseline   {args.baseline.name:<28}{base:.4f}')
    print(f'candidate  {args.candidate.name:<28}{cand:.4f}')
    print(f'\ndifference           {cand - base:+.4f}')
    print(f'95% CI               [{low:+.4f}, {high:+.4f}]')
    print(f'P(candidate better)  {better:.3f}')

    if low > 0:
        print('\nverdict: real at this sample size')
    elif high < 0:
        print('\nverdict: a real regression')
    else:
        print('\nverdict: NOT distinguishable from noise')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
