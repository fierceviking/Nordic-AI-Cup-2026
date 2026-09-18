"""The evidence is the answer, not the question that prompted it.

    python tools/answer_shift.py --model parakeet

Diagnosis: where selection fails, the chosen window has *higher* lexical
coverage than the correct one (0.63 vs 0.44). A question asked in the room
("Do you know of any exposure?") shares its words with the question we are
asked ("Does the patient deny any known exposure?"), so coverage points at the
prompt while the annotation marks the reply.

This is the issue -> resolution structure from the decision-detection literature
(Hsueh & Moore, NAACL 2007). The cues are hand-written rather than learned,
because at 39 conversations a learned detector memorises topic words instead
(Karan et al., SIGDIAL 2021).
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


def interrogative(text: str) -> bool:
    return text.strip().endswith('?')


def question_fraction(match, units) -> float:
    """How much of the window that matched is itself a question being asked."""
    wanted = set(match.matched)
    last = min(match.window.last_unit, len(units) - 1)

    hit, asked = 0, 0
    for index in range(match.window.first_unit, last + 1):
        unit = units[index]
        if any(token in wanted
               for word in unit.words
               for token in content_stems(word.word)):
            hit += 1
            asked += interrogative(unit.text)
    return asked / hit if hit else 0.0


def shifted_span(match, units, max_units=3):
    """Span starting at the first non-interrogative unit at or after the match."""
    last = min(match.window.last_unit, len(units) - 1)
    wanted = set(match.matched)

    hits = [
        index for index in range(match.window.first_unit, last + 1)
        if any(token in wanted
               for word in units[index].words
               for token in content_stems(word.word))
    ]
    if not hits:
        return matched_run_span(match.window, match.matched, units, max_units)

    start = hits[0]
    # Walk past the prompt to the reply it elicited.
    while start < len(units) - 1 and interrogative(units[start].text):
        start += 1

    end = min(start + max_units - 1, len(units) - 1, max(hits[-1], start))
    return float(units[start].start), float(units[end].end)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=asr.DEFAULT_MODEL)
    args = parser.parse_args()

    lexical = LexicalApproach(max_units=3, length_penalty=0.0)
    cached = []

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
            cached.append((matches, units, gold))

    def evaluate(score_fn, span_fn) -> float:
        total = []
        for matches, units, gold in cached:
            scores = np.array([score_fn(m, units) for m in matches])
            pick = matches[int(np.argmax(scores))]
            total.append(temporal_iou(gold, calibrate(span_fn(pick, units), *CALIB)))
        return float(np.mean(total))

    plain_score = lambda m, u: m.coverage  # noqa: E731
    plain_span = lambda m, u: matched_run_span(m.window, m.matched, u, 3)  # noqa: E731

    print(f'{len(cached)} questions, top-{TOP_K} candidates\n')
    print(f'{"rule":<46}{"mean tIoU":>10}')
    print('-' * 58)
    print(f'{"argmax coverage, matched run (shipped)":<46}'
          f'{evaluate(plain_score, plain_span):>10.4f}')

    print(f'{"+ shift span past the prompt":<46}'
          f'{evaluate(plain_score, shifted_span):>10.4f}')

    for penalty in (0.15, 0.25, 0.35, 0.5, 0.75):
        scorer = (lambda m, u, p=penalty:
                  m.coverage * (1.0 - p * question_fraction(m, u)))
        print(f'{f"+ penalise question-matched windows {penalty:.2f}":<46}'
              f'{evaluate(scorer, plain_span):>10.4f}')

    print()
    for penalty in (0.25, 0.35, 0.5):
        scorer = (lambda m, u, p=penalty:
                  m.coverage * (1.0 - p * question_fraction(m, u)))
        print(f'{f"+ penalty {penalty:.2f} AND shifted span":<46}'
              f'{evaluate(scorer, shifted_span):>10.4f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
