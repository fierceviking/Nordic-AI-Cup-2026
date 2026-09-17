"""Why the evidence half is losing points, on a dumped run.

    python tools/analyse_spans.py runs/neural.csv

Separates the two ways a span loses: pointing somewhere else entirely (tIoU 0)
and pointing at roughly the right place with the wrong extent.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from solution import asr  # noqa: E402
from solution.windows import split_units  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dump')
    parser.add_argument('--model', default=asr.DEFAULT_MODEL)
    parser.add_argument('--worst', type=int, default=12)
    args = parser.parse_args()

    with open(args.dump, newline='', encoding='utf-8') as f:
        rows = [row for row in csv.DictReader(f) if row['tiou'] != '']

    ious = [float(row['tiou']) for row in rows]
    gold_lengths = [float(row['gold_end']) - float(row['gold_start']) for row in rows]
    pred_lengths = [
        float(row['pred_end']) - float(row['pred_start'])
        for row in rows if row['pred_start'] != ''
    ]

    zero = [row for row in rows if float(row['tiou']) == 0.0]
    overlapping = [row for row in rows if float(row['tiou']) > 0]

    print(f'annotated yes questions   {len(rows)}')
    print(f'mean tIoU                 {statistics.mean(ious):.3f}')
    print(f'  tIoU = 0 (missed)       {len(zero)} ({len(zero) / len(rows):.1%})')
    print(f'  tIoU of the rest        '
          f'{statistics.mean([float(r["tiou"]) for r in overlapping]):.3f}')
    for bound in (0.1, 0.3, 0.5, 0.7):
        share = sum(1 for iou in ious if iou >= bound) / len(ious)
        print(f'  fraction >= {bound:<4}        {share:.3f}')

    print(f'\ngold length   mean {statistics.mean(gold_lengths):.2f}  '
          f'median {statistics.median(gold_lengths):.2f}')
    print(f'pred length   mean {statistics.mean(pred_lengths):.2f}  '
          f'median {statistics.median(pred_lengths):.2f}')

    # Of the ones that overlap: are we too wide, too narrow, or offset?
    wide = narrow = offset = 0
    for row in overlapping:
        gold_start, gold_end = float(row['gold_start']), float(row['gold_end'])
        pred_start, pred_end = float(row['pred_start']), float(row['pred_end'])
        inside = pred_start >= gold_start - 0.2 and pred_end <= gold_end + 0.2
        covers = pred_start <= gold_start + 0.2 and pred_end >= gold_end - 0.2
        if covers and not inside:
            wide += 1
        elif inside and not covers:
            narrow += 1
        else:
            offset += 1
    print(f'\nof the {len(overlapping)} overlapping spans: '
          f'{wide} cover the gold (too wide), {narrow} sit inside it '
          f'(too narrow), {offset} are offset')

    # Does the predicted centre land inside the gold span at all?
    centred = sum(
        1 for row in rows
        if row['pred_start'] != ''
        and float(row['gold_start']) <= (float(row['pred_start'])
                                         + float(row['pred_end'])) / 2
        <= float(row['gold_end'])
    )
    print(f'predicted centre inside the gold span: {centred}/{len(rows)} '
          f'({centred / len(rows):.1%})')

    print(f'\nworst {args.worst} (by tIoU), with what was said')
    transcripts = {}
    rows.sort(key=lambda row: float(row['tiou']))
    for row in rows[: args.worst]:
        key = f'conversation_{row["transcript_id"]}'
        if key not in transcripts:
            transcripts[key] = asr.load_cached(key, args.model)
        units = split_units(transcripts[key])

        print(f'\n  {row["question_id"]}  tIoU {float(row["tiou"]):.3f}')
        print(f'    Q      {row["question"]}')
        print(f'    gold   {row["gold_start"]}-{row["gold_end"]}  '
              f'| {_text_between(units, float(row["gold_start"]), float(row["gold_end"]))}')
        if row['pred_start'] != '':
            print(f'    pred   {float(row["pred_start"]):.2f}-'
                  f'{float(row["pred_end"]):.2f}  '
                  f'| {_text_between(units, float(row["pred_start"]), float(row["pred_end"]))}')

    return 0


def _text_between(units, start: float, end: float) -> str:
    hits = [
        unit.text for unit in units
        if min(unit.end, end) - max(unit.start, start) > 0.15
    ]
    return ' '.join(hits)[:220]


if __name__ == '__main__':
    raise SystemExit(main())
