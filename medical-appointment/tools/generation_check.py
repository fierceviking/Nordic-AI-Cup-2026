"""Were the questions written from the annotated spans?

    python tools/generation_check.py --model parakeet

Eleven ranking methods have now lost to IDF-weighted word overlap, including a
7B model reading the transcript with explicit instructions. That is too
consistent to be luck, and suggests the annotation is not "the passage a human
would cite" but "the passage the question was written from" -- these are
simulated consultations, so a generator plausibly picked a span first and wrote
the question from it.

If so, lexical overlap is not a proxy for the target, it is the generating
process, and no amount of semantic reasoning can beat it. That would also
explain why every gain so far has come from changing what is emitted rather than
which candidate is chosen.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from solution import asr  # noqa: E402
from solution.textutil import content_stems  # noqa: E402
from solution.windows import split_units  # noqa: E402
from utils import gold_evidence, group_questions_by_conversation  # noqa: E402


def words_between(transcript, start, end):
    return [
        word for word in transcript.words
        if word.end > start and word.start < end
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=asr.DEFAULT_MODEL)
    args = parser.parse_args()

    in_gold, in_best_other, in_transcript = [], [], []

    for audio_filename, rows in group_questions_by_conversation():
        transcript = asr.load_cached(Path(audio_filename).stem, args.model)
        if transcript is None:
            raise SystemExit(f'transcribe first: {audio_filename}')

        units = split_units(transcript)
        transcript_stems = set(content_stems(transcript.text))

        for row in rows:
            gold = gold_evidence(row)
            if not gold:
                continue

            wanted = set(content_stems(row['question']))
            if not wanted:
                continue

            gold_words = words_between(transcript, gold[0], gold[1])
            gold_stems = set(content_stems(
                ' '.join(word.word for word in gold_words)
            ))

            in_gold.append(len(wanted & gold_stems) / len(wanted))
            in_transcript.append(len(wanted & transcript_stems) / len(wanted))

            # The best any *other* utterance manages, for comparison.
            others = [
                len(wanted & set(content_stems(unit.text))) / len(wanted)
                for unit in units
                if unit.end <= gold[0] or unit.start >= gold[1]
            ]
            in_best_other.append(max(others) if others else 0.0)

    print(f'{len(in_gold)} annotated questions\n')
    print('fraction of the question\'s content words that appear...')
    print(f'  anywhere in the transcript        {np.mean(in_transcript):.3f}')
    print(f'  inside the annotated span         {np.mean(in_gold):.3f}')
    print(f'  in the best non-overlapping unit  {np.mean(in_best_other):.3f}')

    margin = np.array(in_gold) - np.array(in_best_other)
    print(f'\ngold beats every other utterance   '
          f'{float(np.mean(margin > 0)):.1%} of the time')
    print(f'gold ties or loses                 '
          f'{float(np.mean(margin <= 0)):.1%} of the time')
    print(f'mean margin                        {float(np.mean(margin)):+.3f}')

    for threshold in (0.5, 0.7, 0.9, 1.0):
        share = float(np.mean(np.array(in_gold) >= threshold))
        print(f'  span contains >= {threshold:.0%} of question words: {share:.1%}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
