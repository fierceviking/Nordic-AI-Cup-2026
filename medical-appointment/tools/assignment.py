"""Make the ten questions compete for distinct passages.

    python tools/assignment.py --model parakeet

Nine attempts to rerank candidates *per question* have failed. This changes the
decoding rule instead: within a conversation the questions are answered by
different moments, so choosing all of them jointly -- one passage each -- uses
information that per-question argmax throws away. A window that two questions
both want is evidence that at least one of them is wrong.

This is the Segmental Posterior Decoding lesson from RESEARCH.md: on the same
network, replacing per-slot confidence with a globally normalised decision was
worth 11 points.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

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
CALIB = (0.92, -0.12, -0.50)


def overlaps(a, b) -> bool:
    return min(a[1], b[1]) - max(a[0], b[0]) > 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=asr.DEFAULT_MODEL)
    args = parser.parse_args()

    lexical = LexicalApproach(max_units=3, length_penalty=0.0)
    greedy, assigned, oracle = [], [], []
    gold_collisions = pred_collisions = pairs = 0

    for audio_filename, rows in group_questions_by_conversation():
        transcript = asr.load_cached(Path(audio_filename).stem, args.model)
        if transcript is None:
            raise SystemExit(f'transcribe first: {audio_filename}')

        units = split_units(transcript)
        windows = build_windows(units, max_units=3)
        stems = [set(content_stems(w.text)) for w in windows]
        idf = _inverse_document_frequency(units)

        positives = [row for row in rows if gold_evidence(row)]
        if len(positives) < 2:
            continue

        shortlists, spans, scores = [], [], []
        for row in positives:
            matches = lexical.score_windows(
                row['question'], windows, stems, idf
            )
            if not matches:
                shortlists.append([])
                spans.append([])
                scores.append([])
                continue
            matches = sorted(matches, key=lambda m: m.score, reverse=True)[:TOP_K]
            shortlists.append(matches)
            spans.append([
                calibrate(matched_run_span(m.window, m.matched, units, 3), *CALIB)
                for m in matches
            ])
            scores.append([m.coverage for m in matches])

        golds = [gold_evidence(row) for row in positives]

        for i in range(len(positives)):
            for j in range(i + 1, len(positives)):
                pairs += 1
                gold_collisions += overlaps(golds[i], golds[j])
                if spans[i] and spans[j]:
                    pred_collisions += overlaps(spans[i][0], spans[j][0])

        for index, row in enumerate(positives):
            if not spans[index]:
                greedy.append(0.0)
                oracle.append(0.0)
                continue
            greedy.append(temporal_iou(golds[index], spans[index][0]))
            oracle.append(max(
                temporal_iou(golds[index], span) for span in spans[index]
            ))

        # One row per question, one column per distinct candidate window.
        window_ids = sorted({
            id(match.window)
            for shortlist in shortlists for match in shortlist
        })
        column = {key: i for i, key in enumerate(window_ids)}
        cost = np.full((len(positives), len(window_ids)), -1.0)
        for index, shortlist in enumerate(shortlists):
            for match, value in zip(shortlist, scores[index]):
                cost[index, column[id(match.window)]] = value

        rows_idx, cols_idx = linear_sum_assignment(cost, maximize=True)
        chosen = {int(r): int(c) for r, c in zip(rows_idx, cols_idx)}

        for index, row in enumerate(positives):
            if not spans[index]:
                assigned.append(0.0)
                continue
            target = chosen.get(index)
            picked = spans[index][0]
            if target is not None:
                for match, span in zip(shortlists[index], spans[index]):
                    if column[id(match.window)] == target:
                        picked = span
                        break
            assigned.append(temporal_iou(golds[index], picked))

    print(f'{len(greedy)} annotated questions in multi-positive conversations\n')
    print(f'gold span pairs that overlap each other   '
          f'{gold_collisions}/{pairs} ({gold_collisions / pairs:.1%})')
    print(f'predicted span pairs that collide         '
          f'{pred_collisions}/{pairs} ({pred_collisions / pairs:.1%})\n')

    print(f'{"decoding rule":<40}{"mean tIoU":>10}')
    print('-' * 52)
    print(f'{"per-question argmax (shipped)":<40}{np.mean(greedy):>10.4f}')
    print(f'{"global one-passage-per-question":<40}{np.mean(assigned):>10.4f}')
    print(f'{"oracle over candidates":<40}{np.mean(oracle):>10.4f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
