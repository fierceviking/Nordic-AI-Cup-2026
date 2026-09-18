"""Fit a global timestamp correction on top of a dumped run.

    python tools/calibrate_spans.py runs/parakeet_neural_base.csv

Three scalars are fitted -- ``width_scale`` about the span midpoint, then
``start_offset`` and ``end_offset`` -- by grid search on mean tIoU. SLUE Phase-2
(Shon et al., ACL 2023, appendix D.2) does exactly this for named entity
localization and reports optima that differ systematically by ASR architecture,
so the numbers have to be refitted whenever the ASR backend changes.

Scoring a correction costs nothing once a run is dumped, so the whole grid is
evaluated against the cached predictions rather than by re-running the model.
Every number is reported twice: fitted on all 39 conversations, and
leave-one-conversation-out, which is the one to believe.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load(path: Path):
    """Rows with an annotated span -- the only ones tIoU averages over."""
    golds: List[Tuple[float, float]] = []
    preds: List[Optional[Tuple[float, float]]] = []
    groups: List[str] = []

    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if not row.get('gold_start') or not row.get('gold_end'):
                continue

            golds.append((float(row['gold_start']), float(row['gold_end'])))
            groups.append(row['transcript_id'])

            if row.get('pred_start') and row.get('pred_end'):
                preds.append((float(row['pred_start']), float(row['pred_end'])))
            else:
                preds.append(None)

    return golds, preds, np.array(groups)


def transform(
    starts: np.ndarray,
    ends: np.ndarray,
    scale: float,
    start_offset: float,
    end_offset: float,
) -> Tuple[np.ndarray, np.ndarray]:
    centre = (starts + ends) / 2.0
    half = (ends - starts) / 2.0 * scale

    new_start = centre - half + start_offset
    new_end = centre + half + end_offset

    new_start = np.maximum(0.0, new_start)
    new_end = np.maximum(new_start + 0.05, new_end)
    return new_start, new_end


def iou(gold_start, gold_end, start, end) -> np.ndarray:
    overlap = np.minimum(gold_end, end) - np.maximum(gold_start, start)
    overlap = np.maximum(0.0, overlap)
    union = np.maximum(gold_end, end) - np.minimum(gold_start, start)
    return np.where(union > 0, overlap / union, 0.0)


def score_grid(golds, preds, grid) -> np.ndarray:
    """tIoU of every (scale, start_offset, end_offset) cell against every row."""
    gold_start = np.array([g[0] for g in golds])
    gold_end = np.array([g[1] for g in golds])

    # A question we produced no span for scores zero and cannot be rescued by
    # any correction, so it is carried as a constant-zero row.
    have = np.array([p is not None for p in preds])
    starts = np.array([p[0] if p else 0.0 for p in preds])
    ends = np.array([p[1] if p else 0.0 for p in preds])

    out = np.zeros((len(grid), len(golds)))
    for index, (scale, start_offset, end_offset) in enumerate(grid):
        new_start, new_end = transform(
            starts, ends, scale, start_offset, end_offset
        )
        out[index] = np.where(
            have, iou(gold_start, gold_end, new_start, new_end), 0.0
        )

    return out


def build_grid(scales, start_offsets, end_offsets):
    return [
        (scale, start_offset, end_offset)
        for scale in scales
        for start_offset in start_offsets
        for end_offset in end_offsets
    ]


def report(name, grid, table, groups) -> Tuple[float, float, float]:
    means = table.mean(axis=1)
    best = int(np.argmax(means))
    scale, start_offset, end_offset = grid[best]

    # Leave-one-conversation-out: pick the cell on 38 conversations, score it on
    # the 39th. The 10 questions in a conversation are not independent, so a
    # plain split would flatter the fit.
    held: List[float] = []
    for group in np.unique(groups):
        mask = groups == group
        inner = int(np.argmax(table[:, ~mask].mean(axis=1)))
        held.append(table[inner][mask].mean())

    loco = float(np.average(held, weights=[np.sum(groups == g)
                                           for g in np.unique(groups)]))

    print(f'{name}')
    print(f'  best cell        scale {scale:.2f}  '
          f'start {start_offset:+.3f}s  end {end_offset:+.3f}s')
    print(f'  mean tIoU (fit)  {means[best]:.4f}')
    print(f'  mean tIoU (LOCO) {loco:.4f}')
    return scale, start_offset, end_offset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()

    golds, preds, groups = load(args.run)
    print(f'{len(golds)} annotated spans over '
          f'{len(np.unique(groups))} conversations\n')

    baseline = score_grid(golds, preds, [(1.0, 0.0, 0.0)])
    print(f'uncorrected mean tIoU  {baseline.mean():.4f}\n')

    coarse = build_grid(
        np.round(np.arange(0.6, 2.01, 0.1), 2),
        np.round(np.arange(-0.6, 0.61, 0.04), 3),
        np.round(np.arange(-0.6, 0.61, 0.04), 3),
    )
    table = score_grid(golds, preds, coarse)
    scale, start_offset, end_offset = report('coarse', coarse, table, groups)

    fine = build_grid(
        np.round(np.arange(max(0.3, scale - 0.1), scale + 0.101, 0.02), 3),
        np.round(np.arange(start_offset - 0.04, start_offset + 0.041, 0.01), 3),
        np.round(np.arange(end_offset - 0.04, end_offset + 0.041, 0.01), 3),
    )
    table = score_grid(golds, preds, fine)
    print()
    scale, start_offset, end_offset = report('fine', fine, table, groups)

    print('\nApply with:')
    print(f'  --params \'{{"width_scale": {scale}, '
          f'"start_offset": {start_offset}, "end_offset": {end_offset}}}\'')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
