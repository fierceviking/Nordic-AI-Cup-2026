"""Which hypothesis form should the NLI reader be given?

    python tools/hypothesis.py --model parakeet

``declarative()`` moves the auxiliary to a fixed offset of three words whenever
the subject starts with a determiner, which mangles roughly half the questions:
"Was the patient listened to...?" becomes "The patient listened was to...".

That text is the NLI hypothesis. Accuracy survived it because the answer is
``max(entail) >= threshold``, which only needs the right passage to score
highest *for a consistently mangled hypothesis*. Ranking needs the argmax to be
correct, which is exactly what this destroys.

Four forms are compared on both halves of the score at once.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from solution import asr  # noqa: E402
from solution.approaches.lexical import _inverse_document_frequency  # noqa: E402
from solution.approaches.neural import NeuralApproach  # noqa: E402
from solution.textutil import content_stems, declarative  # noqa: E402
from solution.windows import (build_windows, calibrate,  # noqa: E402
                              matched_run_span, split_units)
from utils import (gold_evidence, group_questions_by_conversation,  # noqa: E402
                   temporal_iou)

TOP_K = 8
CALIB = (0.92, -0.12, -0.50)
AUXILIARIES = {
    'is', 'are', 'was', 'were', 'do', 'does', 'did', 'has', 'have', 'had',
    'will', 'would', 'should', 'shall', 'can', 'could', 'may', 'might', 'must',
}
TAG = re.compile(
    r',\s*(is|isn\'t|are|aren\'t|was|wasn\'t|were|weren\'t|do|don\'t|does|'
    r'doesn\'t|did|didn\'t|has|hasn\'t|have|haven\'t|will|won\'t|can\'t|right|'
    r'correct)\s*(it|he|she|they|that|this)?\s*$',
    re.IGNORECASE,
)


def strip_tag(question: str) -> str:
    return TAG.sub('', question.strip().rstrip('?').strip()).strip()


def raw(question: str) -> str:
    return question.strip()


def drop_aux(question: str) -> str:
    """Delete the fronted auxiliary and keep the remaining word order."""
    text = strip_tag(question)
    words = text.split()
    if words and words[0].lower() in AUXILIARIES and len(words) > 2:
        words = words[1:]
    statement = ' '.join(words)
    return statement[:1].upper() + statement[1:] + '.'


def statement_only(question: str) -> str:
    """Strip the tag and the question mark, otherwise leave it alone."""
    text = strip_tag(question)
    return text[:1].upper() + text[1:] + '.'


FORMS = {
    'raw question': raw,
    'declarative() (shipped)': declarative,
    'drop the auxiliary': drop_aux,
    'strip tag only': statement_only,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=asr.DEFAULT_MODEL)
    parser.add_argument('--threshold', type=float, default=0.35)
    args = parser.parse_args()

    approach = NeuralApproach()
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
            matches = approach.lexical.score_windows(
                row['question'], windows, stems, idf
            )
            if not matches:
                continue
            matches = sorted(matches, key=lambda m: m.score, reverse=True)[:TOP_K]
            cached.append({
                'row': row,
                'matches': matches,
                'units': units,
                'gold': gold_evidence(row),
                'spans': [
                    calibrate(
                        matched_run_span(m.window, m.matched, units, 3), *CALIB
                    )
                    for m in matches
                ],
            })

    print(f'{len(cached)} questions, top-{TOP_K} candidates\n')
    print(f'{"hypothesis form":<28}{"accuracy":>10}{"tIoU(NLI rank)":>16}'
          f'{"tIoU(coverage)":>16}')
    print('-' * 72)

    for name, form in FORMS.items():
        correct, ranked, baseline = [], [], []

        for index in range(0, len(cached), 40):
            chunk = cached[index:index + 40]
            premises, hypotheses, owners = [], [], []
            for position, item in enumerate(chunk):
                statement = form(item['row']['question'])
                for match in item['matches']:
                    premises.append(match.window.text)
                    hypotheses.append(statement)
                    owners.append(position)

            entail, _ = approach._entailment(premises, hypotheses)
            entail = np.asarray(entail, dtype=float)
            owners = np.asarray(owners)

            for position, item in enumerate(chunk):
                scores = entail[owners == position]
                label = int(item['row']['label'])
                correct.append(int((scores.max() >= args.threshold) == bool(label)))

                if item['gold'] is not None:
                    ranked.append(temporal_iou(
                        item['gold'], item['spans'][int(np.argmax(scores))]
                    ))
                    baseline.append(temporal_iou(item['gold'], item['spans'][0]))

        print(f'{name:<28}{np.mean(correct):>10.4f}{np.mean(ranked):>16.4f}'
              f'{np.mean(baseline):>16.4f}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
