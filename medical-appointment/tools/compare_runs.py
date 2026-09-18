"""Is the difference between two runs real, or is it noise?

    python tools/compare_runs.py runs/parakeet_run_cal.csv runs/parakeet_shift_cal.csv

Offline moved 0.693 -> 0.704 while validation moved 0.702 -> 0.692. At least one
of those is noise, and comparing two means over 39 conversations cannot say
which. This does a *paired* comparison instead -- the same questions under both
configurations -- and bootstraps by conversation, because the ten questions
about one recording share an ASR transcript and a topic and are not independent
draws.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ACCURACY_WEIGHT = 0.4
TIOU_WEIGHT = 0.6


def load(path: Path):
    tiou: Dict[str, float] = {}
    correct: Dict[str, int] = {}
    group: Dict[str, str] = {}

    with open(path, encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            key = row['question_id']
            group[key] = row['transcript_id']
            correct[key] = int(row['correct'])
            if row.get('gold_start'):
                tiou[key] = float(row['tiou'] or 0.0)

    return tiou, correct, group


def score(keys_tiou, keys_correct) -> float:
    return (ACCURACY_WEIGHT * np.mean(keys_correct)
            + TIOU_WEIGHT * np.mean(keys_tiou))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--draws', type=int, default=10000)
    args = parser.parse_args()

    a_tiou, a_correct, groups = load(args.baseline)
    b_tiou, b_correct, _ = load(args.candidate)

    shared_span = sorted(set(a_tiou) & set(b_tiou))
    shared_all = sorted(set(a_correct) & set(b_correct))

    by_conversation = defaultdict(lambda: {'span': [], 'all': []})
    for key in shared_span:
        by_conversation[groups[key]]['span'].append(key)
    for key in shared_all:
        by_conversation[groups[key]]['all'].append(key)
    conversations = sorted(by_conversation)

    def full(tiou, correct) -> float:
        return score([tiou[k] for k in shared_span],
                     [correct[k] for k in shared_all])

    base = full(a_tiou, a_correct)
    cand = full(b_tiou, b_correct)

    print(f'baseline   {args.baseline.name}   score {base:.4f}')
    print(f'candidate  {args.candidate.name}   score {cand:.4f}')
    print(f'difference {cand - base:+.4f}\n')

    # Cluster bootstrap: resample whole conversations, not questions.
    rng = np.random.default_rng(0)
    differences = np.empty(args.draws)
    for draw in range(args.draws):
        picked = rng.choice(conversations, size=len(conversations), replace=True)
        span_keys, all_keys = [], []
        for name in picked:
            span_keys.extend(by_conversation[name]['span'])
            all_keys.extend(by_conversation[name]['all'])
        differences[draw] = (
            score([b_tiou[k] for k in span_keys], [b_correct[k] for k in all_keys])
            - score([a_tiou[k] for k in span_keys], [a_correct[k] for k in all_keys])
        )

    low, high = np.percentile(differences, [2.5, 97.5])
    print(f'paired cluster bootstrap over {len(conversations)} conversations, '
          f'{args.draws} draws')
    print(f'  95% CI on the difference   [{low:+.4f}, {high:+.4f}]')
    print(f'  P(candidate is better)     {np.mean(differences > 0):.3f}')

    verdict = 'REAL' if low > 0 else ('HARMFUL' if high < 0 else 'NOT DISTINGUISHABLE')
    print(f'  verdict                    {verdict}')

    # What a 19-conversation validation run can resolve at all.
    per_conversation = []
    for name in conversations:
        keys = by_conversation[name]['span']
        if keys:
            per_conversation.append(np.mean([b_tiou[k] for k in keys]))
    spread = np.std(per_conversation, ddof=1)
    print(f'\nper-conversation mean tIoU: sd {spread:.3f}')
    print(f'  standard error over 19 conversations '
          f'{TIOU_WEIGHT * spread / np.sqrt(19):.4f}')
    print(f'  so a validation run resolves differences of roughly '
          f'+/-{2 * TIOU_WEIGHT * spread / np.sqrt(19):.3f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
