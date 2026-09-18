"""Where selection goes wrong, and what the better candidate looked like.

    python tools/diagnose_selection.py --model parakeet --worst 15

Eight ranking signals have failed to beat lexical coverage. Rather than guess a
ninth, this prints the cases where a much better candidate was sitting in the
shortlist, side by side with the one that was chosen, so the difference can be
read rather than assumed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from solution import asr  # noqa: E402
from solution.approaches.lexical import (LexicalApproach,  # noqa: E402
                                         _inverse_document_frequency)
from solution.textutil import content_stems  # noqa: E402
from solution.windows import (build_windows, calibrate,  # noqa: E402
                              matched_run_span, split_units)
from utils import (gold_evidence, group_questions_by_conversation,  # noqa: E402
                   temporal_iou)

TOP_K = 10
CALIB = (0.92, -0.09, -0.48)


def clip(text: str, width: int = 88) -> str:
    text = ' '.join(text.split())
    return text if len(text) <= width else text[: width - 1] + '…'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=asr.DEFAULT_MODEL)
    parser.add_argument('--worst', type=int, default=15)
    args = parser.parse_args()

    lexical = LexicalApproach(max_units=3, length_penalty=0.0)
    cases = []

    for audio_filename, rows in group_questions_by_conversation():
        transcript = asr.load_cached(Path(audio_filename).stem, args.model)
        if transcript is None:
            raise SystemExit(f'transcribe first: {audio_filename}')

        units = split_units(transcript)
        windows = build_windows(units, max_units=3)
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
            matches = sorted(matches, key=lambda m: m.score, reverse=True)[:TOP_K]

            spans = [
                calibrate(matched_run_span(m.window, m.matched, units, 3), *CALIB)
                for m in matches
            ]
            ious = np.array([temporal_iou(gold, s) for s in spans])

            chosen = 0
            best = int(np.argmax(ious))
            cases.append({
                'id': row['question_id'],
                'question': row['question'],
                'gold': gold,
                'chosen': matches[chosen], 'chosen_iou': ious[chosen],
                'chosen_span': spans[chosen],
                'best': matches[best], 'best_iou': ious[best],
                'best_rank': best,
                'regret': ious[best] - ious[chosen],
                'duration': transcript.duration,
            })

    cases.sort(key=lambda c: -c['regret'])
    total = np.mean([c['regret'] for c in cases])
    recoverable = [c for c in cases if c['regret'] > 0.1]

    print(f'{len(cases)} questions, top-{TOP_K} candidates')
    print(f'mean regret (oracle - chosen)      {total:.4f}')
    print(f'questions losing > 0.1 tIoU        {len(recoverable)} '
          f'({len(recoverable) / len(cases):.1%})')
    print(f'chosen was already best            '
          f'{sum(1 for c in cases if c["best_rank"] == 0)}\n')

    ranks = [c['best_rank'] for c in recoverable]
    print(f'where the better candidate sat (rank among top-{TOP_K}):')
    for rank in range(TOP_K):
        count = ranks.count(rank)
        if count:
            print(f'  rank {rank}: {"#" * count} {count}')

    print(f'\nposition in conversation (fraction), recoverable cases:')
    chosen_pos = np.mean([
        c['chosen'].window.start / c['duration'] for c in recoverable
    ])
    best_pos = np.mean([
        c['best'].window.start / c['duration'] for c in recoverable
    ])
    gold_pos = np.mean([c['gold'][0] / c['duration'] for c in recoverable])
    print(f'  chosen {chosen_pos:.3f}   better {best_pos:.3f}   gold {gold_pos:.3f}')

    print(f'\ncoverage, recoverable cases:')
    print(f'  chosen {np.mean([c["chosen"].coverage for c in recoverable]):.3f}'
          f'   better {np.mean([c["best"].coverage for c in recoverable]):.3f}')

    print(f'\n--- worst {args.worst} ---')
    for case in cases[: args.worst]:
        print(f'\n{case["id"]}   regret {case["regret"]:.3f}'
              f'   (chosen {case["chosen_iou"]:.3f} -> best {case["best_iou"]:.3f}'
              f' at rank {case["best_rank"]})')
        print(f'  Q       {clip(case["question"])}')
        print(f'  chosen  cov {case["chosen"].coverage:.2f} '
              f'{case["chosen_span"][0]:6.1f}-{case["chosen_span"][1]:6.1f} | '
              f'{clip(case["chosen"].window.text, 70)}')
        print(f'  better  cov {case["best"].coverage:.2f} '
              f'{case["gold"][0]:6.1f}-{case["gold"][1]:6.1f} | '
              f'{clip(case["best"].window.text, 70)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
