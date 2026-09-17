"""Compact training progress: where a run is, and whether it is still improving.

Ultralytics' live output is a progress bar that scrolls; this is the same
information as a table, plus an ETA from the measured seconds per epoch.
"""

import argparse
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent


def read(path: Path):
    rows = [line.split(',') for line in path.read_text().splitlines() if line.strip()]
    header = [h.strip() for h in rows[0]]
    return header, [dict(zip(header, r)) for r in rows[1:]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, help='run name under lab/runs')
    parser.add_argument('--total', type=int, default=0, help='total epochs')
    parser.add_argument('--last', type=int, default=10)
    arguments = parser.parse_args()

    path = LAB / 'runs' / arguments.name / 'results.csv'
    if not path.exists():
        print(f'{arguments.name}: no results.csv yet (still starting up)')
        return

    _, rows = read(path)
    if not rows:
        print(f'{arguments.name}: no epochs finished yet')
        return

    print(f'{"epoch":>6s} {"box":>7s} {"cls":>7s} {"P":>7s} {"R":>7s} '
          f'{"mAP50":>7s} {"mAP50-95":>9s} {"s/epoch":>8s}')
    previous = 0.0
    shown = rows[-arguments.last:]
    for row in shown:
        elapsed = float(row['time'])
        index = rows.index(row)
        per_epoch = elapsed - (float(rows[index - 1]['time']) if index else 0.0)
        print(f'{row["epoch"]:>6s} {float(row["train/box_loss"]):7.4f} '
              f'{float(row["train/cls_loss"]):7.4f} '
              f'{float(row["metrics/precision(B)"]):7.4f} '
              f'{float(row["metrics/recall(B)"]):7.4f} '
              f'{float(row["metrics/mAP50(B)"]):7.4f} '
              f'{float(row["metrics/mAP50-95(B)"]):9.4f} {per_epoch:8.0f}')
        previous = elapsed

    done = int(rows[-1]['epoch'])
    total = arguments.total
    if total:
        per_epoch = float(rows[-1]['time']) / max(1, done)
        remaining = (total - done) * per_epoch
        print(f'\n  {done}/{total} epochs  '
              f'({done / total * 100:.0f}%)  '
              f'~{remaining / 60:.0f} min remaining '
              f'(ETA {time.strftime("%H:%M", time.localtime(time.time() + remaining))})')
    best = max(float(r['metrics/mAP50(B)']) for r in rows)
    print(f'  best mAP50 so far: {best:.4f} '
          f'(latest {float(rows[-1]["metrics/mAP50(B)"]):.4f})')


if __name__ == '__main__':
    main()
