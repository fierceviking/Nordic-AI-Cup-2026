"""Grid search over an approach's parameters against the cached transcripts.

    python tools/sweep.py --approach lexical --grid "{\"threshold\":[0.3,0.4,0.5]}"

Prints every cell sorted by score. With 39 conversations this over-fits
happily, so treat a win of less than ~0.01 as noise and prefer the setting
that is also defensible.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from offline_eval import evaluate  # type: ignore  # noqa: E402
from solution import asr  # noqa: E402
from solution.approaches import build  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approach', default='lexical')
    parser.add_argument('--model', default=asr.DEFAULT_MODEL)
    parser.add_argument('--grid', required=True,
                        help='JSON object of parameter -> list of values.')
    parser.add_argument('--fixed', default='{}',
                        help='JSON object of parameters held constant.')
    parser.add_argument('--top', type=int, default=15)
    args = parser.parse_args()

    grid = json.loads(args.grid)
    fixed = json.loads(args.fixed)
    names = list(grid)

    results = []
    for values in itertools.product(*(grid[name] for name in names)):
        params = dict(zip(names, values))
        approach = build(args.approach, **{**fixed, **params})
        summary = evaluate(approach, args.model)
        results.append((summary['score'], summary, params))
        print(f'  {json.dumps(params):<60} score {summary["score"]:.3f}  '
              f'acc {summary["accuracy"]:.3f}  tIoU {summary["mean_tiou"]:.3f}  '
              f'yes {summary["yes_rate"]:.3f}')

    results.sort(key=lambda item: item[0], reverse=True)
    print('\nbest')
    for score, summary, params in results[: args.top]:
        by_type = '  '.join(
            f'{name[:4]} {value:.3f}'
            for name, (value, _) in sorted(summary['by_type'].items())
        )
        print(f'  {score:.3f}  acc {summary["accuracy"]:.3f}  '
              f'tIoU {summary["mean_tiou"]:.3f}  {by_type}   {json.dumps(params)}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
