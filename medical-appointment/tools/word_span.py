"""Given the window we already pick, how much is left in the boundaries?

    python tools/word_span.py --model parakeet

Emitting a whole utterance caps the score near 0.80; word-level runs cap near
0.90. This holds selection fixed at what the shipped ranker actually chooses and
varies only the span cut from inside it, so the remaining headroom splits
cleanly into "pick a better window" and "cut a better span out of it".
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
from solution.textutil import content_stems, stem, tokenize  # noqa: E402
from solution.windows import (build_windows, calibrate,  # noqa: E402
                              matched_unit_span, split_units)
from utils import (gold_evidence, group_questions_by_conversation,  # noqa: E402
                   temporal_iou)

TOP_K = 20
MAX_SPAN = 10.0


def runs(words: Sequence, max_span: float = MAX_SPAN):
    """Every contiguous word run no longer than ``max_span`` seconds."""
    for i in range(len(words)):
        for j in range(i, len(words)):
            if words[j].end - words[i].start > max_span:
                break
            yield i, j


def hit_mask(words: Sequence, wanted: set) -> np.ndarray:
    return np.array([
        any(stem(token) in wanted for token in tokenize(word.word))
        for word in words
    ])


def minimal_cover(words: Sequence, wanted: set) -> Tuple[float, float]:
    """Shortest run containing every matched word in the window."""
    hits = np.flatnonzero(hit_mask(words, wanted))
    if not len(hits):
        return words[0].start, words[-1].end
    return words[hits[0]].start, words[hits[-1]].end


def densest(words: Sequence, wanted: set, idf: dict, penalty: float) -> Tuple[float, float]:
    """Run maximising matched IDF mass, with a mild penalty on duration."""
    mask = hit_mask(words, wanted)
    weights = np.array([
        max((idf.get(stem(token), 0.0) for token in tokenize(word.word)),
            default=0.0) * hit
        for word, hit in zip(words, mask)
    ])

    best, best_score = (words[0].start, words[-1].end), -1e9
    for i, j in runs(words):
        duration = words[j].end - words[i].start
        score = weights[i:j + 1].sum() - penalty * duration
        if score > best_score:
            best, best_score = (words[i].start, words[j].end), score
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=asr.DEFAULT_MODEL)
    parser.add_argument('--penalty', type=float, default=0.35)
    args = parser.parse_args()

    lexical = LexicalApproach(max_units=3, length_penalty=0.0)
    out: dict = {}

    def record(name: str, value: float) -> None:
        out.setdefault(name, []).append(value)

    for audio_filename, rows in group_questions_by_conversation():
        transcript = asr.load_cached(Path(audio_filename).stem, args.model)
        if transcript is None:
            raise SystemExit(f'transcribe first: {audio_filename}')

        units = split_units(transcript)
        windows = build_windows(units, max_units=3)
        window_stems = [set(content_stems(window.text)) for window in windows]
        idf = _inverse_document_frequency(units)

        for row in rows:
            gold = gold_evidence(row)
            if not gold:
                continue

            matches = lexical.score_windows(
                row['question'], windows, window_stems, idf
            )
            if not matches:
                continue
            match = max(matches, key=lambda m: m.score)
            words = match.window.words
            if not words:
                continue
            wanted = set(match.matched)

            shipped = calibrate(
                matched_unit_span(match.window, match.matched, units),
                0.92, -0.10, -0.50,
            )
            record('shipped (matched unit + calib)', temporal_iou(gold, shipped))

            record('minimal cover of matched words',
                   temporal_iou(gold, minimal_cover(words, wanted)))
            record('densest run (penalised)',
                   temporal_iou(gold, densest(words, wanted, idf, args.penalty)))

            record('ORACLE run inside chosen window', max(
                temporal_iou(gold, (words[i].start, words[j].end))
                for i, j in runs(words)
            ))
            record('ORACLE run anywhere', max(
                temporal_iou(gold, (transcript.words[i].start,
                                    transcript.words[j].end))
                for i, j in runs(transcript.words)
            ))

    print(f'{len(out["shipped (matched unit + calib)"])} annotated spans, '
          f'selection fixed at argmax lexical coverage\n')
    print(f'{"span cut from the chosen window":<34}{"mean tIoU":>10}')
    print('-' * 46)
    for name, values in out.items():
        print(f'{name:<34}{np.mean(values):>10.4f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
