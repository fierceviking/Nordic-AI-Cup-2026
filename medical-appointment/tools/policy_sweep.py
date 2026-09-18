"""Segmentation x span policy, each cell with its own fitted calibration.

    python tools/policy_sweep.py --model parakeet

Clause-level units raise the window oracle (0.697 -> 0.752) but lower the
single-unit oracle, because a clause is about half the length of an annotated
passage. So the unit of emission has to change with the unit of segmentation.
Calibration is refitted per cell, leave-one-conversation-out: the boundary bias
belongs to the policy, and comparing uncalibrated spans would rank them wrong.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from solution import asr  # noqa: E402
from solution.approaches.lexical import (LexicalApproach,  # noqa: E402
                                         _inverse_document_frequency)
from solution.textutil import content_stems  # noqa: E402
from solution.windows import build_windows, split_units  # noqa: E402
from utils import (gold_evidence, group_questions_by_conversation,  # noqa: E402
                   temporal_iou)

SCALES = np.round(np.arange(0.6, 1.61, 0.1), 2)
OFFSETS = np.round(np.arange(-0.6, 0.61, 0.05), 3)


def matched_units(match, units) -> List[int]:
    """Indices of units inside the window that carry a matched word."""
    wanted = set(match.matched)
    hits = [
        index
        for index in range(match.window.first_unit,
                           min(match.window.last_unit, len(units) - 1) + 1)
        if any(token in wanted
               for word in units[index].words
               for token in content_stems(word.word))
    ]
    return hits


def make_span(match, units, policy: str) -> Tuple[float, float]:
    hits = matched_units(match, units)
    if not hits:
        return match.window.start, match.window.end

    if policy == 'matched unit':
        best, best_hits = hits[0], -1
        for index in hits:
            count = sum(
                1 for word in units[index].words
                for token in content_stems(word.word)
                if token in set(match.matched)
            )
            if count > best_hits:
                best, best_hits = index, count
        return units[best].start, units[best].end

    if policy.startswith('matched run'):
        cap = int(policy.split('<=')[1].rstrip(')'))
        first, last = hits[0], min(hits[-1], hits[0] + cap - 1)
        return units[first].start, units[last].end

    return match.window.start, match.window.end


def fit_calibration(spans, golds, groups) -> Tuple[float, float]:
    """(LOCO tIoU, fitted tIoU) after a per-cell 3-scalar correction."""
    starts = np.array([span[0] for span in spans])
    ends = np.array([span[1] for span in spans])
    gold_start = np.array([gold[0] for gold in golds])
    gold_end = np.array([gold[1] for gold in golds])

    grid = [(s, a, b) for s in SCALES for a in OFFSETS for b in OFFSETS]
    table = np.zeros((len(grid), len(spans)))

    centre = (starts + ends) / 2.0
    half = (ends - starts) / 2.0

    for index, (scale, start_offset, end_offset) in enumerate(grid):
        new_start = np.maximum(0.0, centre - half * scale + start_offset)
        new_end = np.maximum(new_start + 0.05, centre + half * scale + end_offset)
        overlap = np.maximum(
            0.0, np.minimum(gold_end, new_end) - np.maximum(gold_start, new_start)
        )
        union = np.maximum(gold_end, new_end) - np.minimum(gold_start, new_start)
        table[index] = np.where(union > 0, overlap / union, 0.0)

    means = table.mean(axis=1)
    held: List[float] = []
    for group in np.unique(groups):
        mask = groups == group
        inner = int(np.argmax(table[:, ~mask].mean(axis=1)))
        held.extend(table[inner][mask])

    return float(np.mean(held)), float(means.max())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=asr.DEFAULT_MODEL)
    args = parser.parse_args()

    lexical = LexicalApproach(max_units=3, length_penalty=0.0)
    policies = ['matched unit', 'matched run (<=2)', 'matched run (<=3)',
                'whole window']

    print(f'{"clause":>7}{"max_units":>11}{"policy":>20}'
          f'{"raw":>9}{"calibrated":>12}{"oracle":>9}')
    print('-' * 70)

    for clause in (False, True):
        for max_units in (3, 4):
            cached = []
            for audio_filename, rows in group_questions_by_conversation():
                transcript = asr.load_cached(
                    Path(audio_filename).stem, args.model
                )
                if transcript is None:
                    raise SystemExit(f'transcribe first: {audio_filename}')

                units = split_units(
                    transcript, pause=0.25, split_on_clause=clause
                )
                windows = build_windows(units, max_units=max_units)
                stems = [set(content_stems(w.text)) for w in windows]
                idf = _inverse_document_frequency(units)

                for row in rows:
                    gold = gold_evidence(row)
                    if not gold:
                        continue
                    matches = lexical.score_windows(
                        row['question'], windows, stems, idf
                    )
                    if not matches:
                        continue
                    cached.append((
                        max(matches, key=lambda m: m.score), units, gold,
                        Path(audio_filename).stem, windows,
                    ))

            for policy in policies:
                spans, golds, groups = [], [], []
                oracle = []
                for match, units, gold, group, windows in cached:
                    spans.append(make_span(match, units, policy))
                    golds.append(gold)
                    groups.append(group)
                    oracle.append(max(
                        temporal_iou(gold, (w.start, w.end)) for w in windows
                    ))

                raw = np.mean([
                    temporal_iou(g, s) for g, s in zip(golds, spans)
                ])
                loco, _ = fit_calibration(spans, golds, np.array(groups))
                print(f'{str(clause):>7}{max_units:>11}{policy:>20}'
                      f'{raw:>9.4f}{loco:>12.4f}{np.mean(oracle):>9.4f}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
