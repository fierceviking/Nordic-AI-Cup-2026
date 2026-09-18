"""Is tightening the span to the matched words helping or hurting?

    python tools/span_policy.py --model parakeet

The pipeline picks a candidate window and then shrinks it to the words that
matched the question. That was tuned against Whisper timings on CUDA. This
compares span policies under identical selection, and reports each one both
under the actual ranker (argmax lexical coverage) and under an oracle, so the
selection error and the shaping error can be told apart.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from solution import asr  # noqa: E402
from solution.approaches.lexical import (LexicalApproach,  # noqa: E402
                                         _inverse_document_frequency)
from solution.textutil import content_stems  # noqa: E402
from solution.windows import build_windows, split_units, tighten  # noqa: E402
from utils import (gold_evidence, group_questions_by_conversation,  # noqa: E402
                   temporal_iou)

TOP_K = 20


def policies(lexical: LexicalApproach, units) -> Dict[str, Callable]:
    def raw(match):
        return match.window.start, match.window.end

    def first_unit(match):
        unit = units[match.window.first_unit]
        return unit.start, unit.end

    def matched_unit(match):
        """The single utterance holding most of the matched words."""
        best, best_hits = match.window.first_unit, -1
        for index in range(match.window.first_unit, match.window.last_unit + 1):
            hits = sum(
                1 for word in units[index].words
                for token in content_stems(word.word)
                if token in set(match.matched)
            )
            if hits > best_hits:
                best, best_hits = index, hits
        return units[best].start, units[best].end

    return {
        'tighten (current)': lambda m: tighten(
            m.window, m.matched, pad=0.3, min_duration=2.0, max_duration=6.0),
        'tighten pad=0.15': lambda m: tighten(
            m.window, m.matched, pad=0.15, min_duration=1.0, max_duration=8.0),
        'raw window': raw,
        'first unit': first_unit,
        'matched unit': matched_unit,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=asr.DEFAULT_MODEL)
    args = parser.parse_args()

    lexical = LexicalApproach(max_units=3, length_penalty=0.0)
    selected: Dict[str, List[float]] = {}
    oracle: Dict[str, List[float]] = {}

    for audio_filename, rows in group_questions_by_conversation():
        transcript = asr.load_cached(Path(audio_filename).stem, args.model)
        if transcript is None:
            raise SystemExit(f'transcribe first: {audio_filename}')

        units = split_units(transcript)
        windows = build_windows(units, max_units=3)
        window_stems = [set(content_stems(window.text)) for window in windows]
        idf = _inverse_document_frequency(units)
        made = policies(lexical, units)

        for row in rows:
            gold = gold_evidence(row)
            if not gold:
                continue

            matches = lexical.score_windows(
                row['question'], windows, window_stems, idf
            )
            if not matches:
                continue
            matches = sorted(matches, key=lambda m: m.score, reverse=True)[:TOP_K]

            for name, make in made.items():
                spans = [make(match) for match in matches]
                selected.setdefault(name, []).append(
                    temporal_iou(gold, spans[0])
                )
                oracle.setdefault(name, []).append(
                    max(temporal_iou(gold, span) for span in spans)
                )

    print(f'{len(next(iter(selected.values())))} annotated questions, '
          f'top-{TOP_K} lexical candidates\n')
    print(f'{"span policy":<22}{"argmax coverage":>16}{"oracle":>10}')
    print('-' * 48)
    for name in selected:
        print(f'{name:<22}{np.mean(selected[name]):>16.4f}'
              f'{np.mean(oracle[name]):>10.4f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
