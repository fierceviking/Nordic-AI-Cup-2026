"""Learn the boundary correction instead of using one global constant.

    python tools/train_boundaries.py --model parakeet

The three fitted scalars in `windows.calibrate` are an intercept-only model of
the boundary error: every span is nudged by the same amount. They were worth
+0.054 and +0.018 tIoU, the two largest wins in this log, which says the error
has a large systematic part.

This asks whether that part is *conditional* -- whether a one-utterance run needs
a different nudge from a three-utterance one, or a run that starts after a long
pause a different nudge from one that starts mid-turn. Ridge on a handful of
cheap features, leave-one-conversation-out, against the global constant as the
baseline to beat.

Unlike span selection (experiment 18, closed), this is well posed: the target is
a residual in seconds, not a choice among defensible passages.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

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

FEATURES = [
    'duration', 'n_units', 'n_words', 'starts_sentence', 'ends_sentence',
    'pause_before', 'pause_after', 'position', 'question_terms', 'coverage',
]


def collect(model_name: str):
    lexical = LexicalApproach(max_units=3, length_penalty=0.0)
    rows_out = []

    for audio_filename, rows in group_questions_by_conversation():
        transcript = asr.load_cached(Path(audio_filename).stem, model_name)
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
            match = max(matches, key=lambda m: m.score)
            span = matched_run_span(match.window, match.matched, units, 3)

            first = min(range(len(units)),
                        key=lambda i: abs(units[i].start - span[0]))
            last = min(range(len(units)),
                       key=lambda i: abs(units[i].end - span[1]))

            rows_out.append({
                'span': span,
                'gold': gold,
                'group': Path(audio_filename).stem,
                'x': [
                    span[1] - span[0],
                    last - first + 1,
                    sum(len(units[i].words) for i in range(first, last + 1)),
                    float(first == 0 or units[first - 1].text.strip()
                          .endswith(('.', '?', '!'))),
                    float(units[last].text.strip().endswith(('.', '?', '!'))),
                    min(5.0, span[0] - units[first - 1].end) if first > 0 else 5.0,
                    min(5.0, units[last + 1].start - span[1])
                    if last + 1 < len(units) else 5.0,
                    span[0] / max(transcript.duration, 1e-6),
                    len(set(content_stems(row['question']))),
                    match.coverage,
                ],
            })

    return rows_out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=asr.DEFAULT_MODEL)
    parser.add_argument('--alpha', type=float, default=10.0)
    args = parser.parse_args()

    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    data = collect(args.model)
    x = np.array([row['x'] for row in data], dtype=float)
    groups = np.array([row['group'] for row in data])
    spans = np.array([row['span'] for row in data], dtype=float)
    golds = np.array([row['gold'] for row in data], dtype=float)

    # What the boundary *should* have moved, in seconds.
    delta_start = golds[:, 0] - spans[:, 0]
    delta_end = golds[:, 1] - spans[:, 1]

    def iou(predicted) -> float:
        overlap = np.maximum(0.0, np.minimum(golds[:, 1], predicted[:, 1])
                             - np.maximum(golds[:, 0], predicted[:, 0]))
        union = (np.maximum(golds[:, 1], predicted[:, 1])
                 - np.minimum(golds[:, 0], predicted[:, 0]))
        return float(np.mean(np.where(union > 0, overlap / union, 0.0)))

    # A span that missed the passage entirely has a residual of tens of seconds,
    # which is a selection error wearing a boundary error's clothes. Fitting on
    # those makes the model chase the misses and ruin the hits.
    overlapping = (np.minimum(golds[:, 1], spans[:, 1])
                   > np.maximum(golds[:, 0], spans[:, 0]))

    print(f'{len(data)} annotated questions, {len(np.unique(groups))} '
          f'conversations')
    print(f'spans overlapping the annotation: {overlapping.sum()} '
          f'({overlapping.mean():.1%}) -- the rest are selection misses')
    print(f'boundary residual, overlapping only: '
          f'start {delta_start[overlapping].mean():+.3f}s '
          f'(sd {delta_start[overlapping].std():.3f}), '
          f'end {delta_end[overlapping].mean():+.3f}s '
          f'(sd {delta_end[overlapping].std():.3f})\n')

    print(f'{"boundary model":<40}{"mean tIoU":>10}')
    print('-' * 52)
    print(f'{"uncorrected":<40}{iou(spans):>10.4f}')

    shipped = np.array([calibrate(tuple(s), 0.92, -0.12, -0.50) for s in spans])
    print(f'{"global constant (shipped)":<40}{iou(shipped):>10.4f}')

    # Leave-one-conversation-out: fit the residual on 38, apply to the 39th.
    predicted = np.zeros_like(spans)
    for group in np.unique(groups):
        train = (groups != group) & overlapping
        scaler = StandardScaler().fit(x[train])

        start_model = Ridge(alpha=args.alpha).fit(
            scaler.transform(x[train]), delta_start[train]
        )
        end_model = Ridge(alpha=args.alpha).fit(
            scaler.transform(x[train]), delta_end[train]
        )

        held = scaler.transform(x[groups == group])
        new_start = spans[groups == group, 0] + start_model.predict(held)
        new_end = spans[groups == group, 1] + end_model.predict(held)
        new_start = np.maximum(0.0, new_start)
        new_end = np.maximum(new_start + 0.05, new_end)
        predicted[groups == group] = np.column_stack([new_start, new_end])

    print(f'{"conditional ridge (LOCO)":<40}{iou(predicted):>10.4f}')

    scaler = StandardScaler().fit(x[overlapping])
    start_model = Ridge(alpha=args.alpha).fit(
        scaler.transform(x[overlapping]), delta_start[overlapping]
    )
    end_model = Ridge(alpha=args.alpha).fit(
        scaler.transform(x[overlapping]), delta_end[overlapping]
    )
    print('\nwhich features move the boundary (standardised):')
    for name, a, b in sorted(
        zip(FEATURES, start_model.coef_, end_model.coef_),
        key=lambda t: -(abs(t[1]) + abs(t[2])),
    )[:6]:
        print(f'  {name:<18} start {a:+.3f}   end {b:+.3f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
