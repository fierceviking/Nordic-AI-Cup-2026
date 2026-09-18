"""Trim the filler off the edges of the span instead of nudging it globally.

    python tools/trim.py --model mlx-large-v3-turbo

The fitted `start_offset` on whisper is +0.22 s: emitted spans start too early,
consistently. That is what an utterance boundary does -- it opens on "So," or
"Well," or a backchannel, and the annotated passage starts at the content.

A global offset pays the average of that. Trimming pays the actual amount, per
span. Every variant gets its own calibration refitted afterwards, since the
residual bias changes once the edges move, and comparing on shared constants
would rank them wrong.
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
from solution.textutil import STOPWORDS, content_stems, normalise  # noqa: E402
from solution.windows import (build_windows, matched_run_span,  # noqa: E402
                              split_units)
from utils import (gold_evidence, group_questions_by_conversation,  # noqa: E402
                   temporal_iou)

# Openers and backchannels that start an utterance but never carry the claim.
FILLER = frozenset({
    'so', 'well', 'um', 'uh', 'er', 'ah', 'oh', 'yeah', 'yes', 'no', 'okay',
    'ok', 'right', 'now', 'then', 'and', 'but', 'mm', 'mhm', 'hmm', 'sure',
    'alright', 'anyway', 'actually', 'basically', 'you', 'know', 'i', 'see',
    'good', 'great', 'thanks', 'thank', 'please', 'let', 'us', 'lets',
})

SCALES = np.round(np.arange(0.7, 1.31, 0.02), 2)
OFFSETS = np.round(np.arange(-0.5, 0.51, 0.02), 3)


def token(word) -> str:
    return normalise(word.word).strip()


def trim(words: Sequence, drop_filler: bool, drop_stopwords: bool):
    """Move both edges inward past words that cannot carry the claim."""
    if not words:
        return None

    first, last = 0, len(words) - 1

    def skippable(word) -> bool:
        text = token(word)
        if not text:
            return True
        if drop_filler and text in FILLER:
            return True
        if drop_stopwords and text in STOPWORDS:
            return True
        return False

    while first < last and skippable(words[first]):
        first += 1
    while last > first and skippable(words[last]):
        last -= 1

    return float(words[first].start), float(words[last].end)


def words_in(units, span) -> List:
    return [
        word for unit in units for word in unit.words
        if word.end > span[0] + 1e-6 and word.start < span[1] - 1e-6
    ]


def fit(spans, golds, groups) -> float:
    """LOCO tIoU after refitting the three calibration scalars for this variant."""
    starts = np.array([s[0] for s in spans])
    ends = np.array([s[1] for s in spans])
    gold_start = np.array([g[0] for g in golds])
    gold_end = np.array([g[1] for g in golds])

    centre = (starts + ends) / 2.0
    half = (ends - starts) / 2.0

    grid = [(s, a, b) for s in SCALES for a in OFFSETS for b in OFFSETS]
    table = np.zeros((len(grid), len(spans)), dtype=np.float32)

    for index, (scale, start_offset, end_offset) in enumerate(grid):
        new_start = np.maximum(0.0, centre - half * scale + start_offset)
        new_end = np.maximum(new_start + 0.05, centre + half * scale + end_offset)
        overlap = np.maximum(
            0.0, np.minimum(gold_end, new_end) - np.maximum(gold_start, new_start)
        )
        union = np.maximum(gold_end, new_end) - np.minimum(gold_start, new_start)
        table[index] = np.where(union > 0, overlap / union, 0.0)

    held: List[float] = []
    for group in np.unique(groups):
        mask = groups == group
        best = int(np.argmax(table[:, ~mask].mean(axis=1)))
        held.extend(table[best][mask])

    return float(np.mean(held))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=asr.DEFAULT_MODEL)
    parser.add_argument('--span-units', type=int, default=2)
    args = parser.parse_args()

    lexical = LexicalApproach(max_units=3, length_penalty=0.0)
    base, golds, groups, unit_store = [], [], [], []

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
            match = max(matches, key=lambda m: m.score)
            base.append(matched_run_span(
                match.window, match.matched, units, args.span_units, True
            ))
            golds.append(gold)
            groups.append(Path(audio_filename).stem)
            unit_store.append(units)

    groups = np.array(groups)
    print(f'{len(base)} annotated questions, span_units={args.span_units}\n')
    print(f'{"edge treatment":<40}{"LOCO tIoU":>10}')
    print('-' * 52)
    print(f'{"none (shipped)":<40}{fit(base, golds, groups):>10.4f}')

    for label, drop_filler, drop_stopwords in (
        ('trim filler openers/closers', True, False),
        ('trim all stopwords', False, True),
        ('trim filler and stopwords', True, True),
    ):
        variant = []
        for span, units in zip(base, unit_store):
            trimmed = trim(words_in(units, span), drop_filler, drop_stopwords)
            variant.append(trimmed if trimmed else span)
        print(f'{label:<40}{fit(variant, golds, groups):>10.4f}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
