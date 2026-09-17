"""Facts about the supplied data, before any modelling.

    python tools/inspect_data.py

Prints the shape of the annotated evidence spans (they decide 60% of the
score), the balance of question types, and the vocabulary of the questions.
"""

from __future__ import annotations

import collections
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import gold_evidence, group_questions_by_conversation  # noqa: E402


def main() -> int:
    groups = group_questions_by_conversation()
    rows = [row for _, group in groups for row in group]

    print(f'conversations {len(groups)}   questions {len(rows)}')

    types = collections.Counter(row['question_type'] for row in rows)
    labels = collections.Counter(row['answer'] for row in rows)
    print('by type  ', dict(types))
    print('by answer', dict(labels))

    spans = [gold_evidence(row) for row in rows]
    lengths = [end - start for span in spans if span for start, end in [span]]
    starts = [span[0] for span in spans if span]

    print(f'\nannotated spans {len(lengths)}')
    print(f'  length  min {min(lengths):.2f}  p25 {_q(lengths, .25):.2f}  '
          f'median {statistics.median(lengths):.2f}  p75 {_q(lengths, .75):.2f}  '
          f'p90 {_q(lengths, .90):.2f}  max {max(lengths):.2f}  '
          f'mean {statistics.mean(lengths):.2f}')
    print(f'  start   min {min(starts):.2f}  max {max(starts):.2f}')

    # What a fixed-width guess centred on the truth would score, as a ceiling
    # for "right place, wrong width".
    for width in (2, 3, 4, 5, 6, 8):
        ious = []
        for span in spans:
            if not span:
                continue
            centre = (span[0] + span[1]) / 2
            guess = (centre - width / 2, centre + width / 2)
            inter = max(0.0, min(span[1], guess[1]) - max(span[0], guess[0]))
            union = max(span[1], guess[1]) - min(span[0], guess[0])
            ious.append(inter / union if union > 0 else 0.0)
        print(f'  perfectly centred {width}s window -> mean tIoU '
              f'{sum(ious) / len(ious):.3f}')

    per_conversation = collections.Counter(
        row['transcript_id'] for row in rows
    )
    print(f'\nquestions per conversation: {set(per_conversation.values())}')

    positives = collections.Counter()
    for row in rows:
        if row['question_type'] == 'positive':
            positives[row['transcript_id']] += 1
    print(f'positives per conversation: min {min(positives.values())} '
          f'max {max(positives.values())}')

    print('\nsample questions by type')
    seen = collections.defaultdict(int)
    for row in rows:
        kind = row['question_type']
        if seen[kind] < 6:
            seen[kind] += 1
            print(f'  {kind:<14} {row["question"]}')

    return 0


def _q(values, fraction: float) -> float:
    values = sorted(values)
    index = min(len(values) - 1, int(fraction * len(values)))
    return values[index]


if __name__ == '__main__':
    raise SystemExit(main())
