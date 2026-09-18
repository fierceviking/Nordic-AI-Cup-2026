"""Turning a word-timestamped transcript into candidate evidence spans.

The annotated spans are short — median 2.9 seconds, mean 3.2 — and temporal IoU
punishes width hard: a perfectly centred 3-second window scores 0.669 against
the annotations, an 8-second one 0.392. So candidates are built at utterance
scale, not at segment or conversation scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .asr import Transcript, Word

# A pause longer than this inside a segment is treated as an utterance break.
PAUSE_SPLIT_SECONDS = 0.55
SENTENCE_END = tuple('.?!')

# Words that tend to open a new clause, used when splitting finer than sentences.
CLAUSE_STARTERS = frozenset({
    'and', 'but', 'so', 'then', 'because', 'while', 'although', 'though',
    'which', 'that', 'if', 'when', 'after', 'before', 'plus', 'also',
})


@dataclass
class Unit:
    """One utterance-sized piece of the transcript."""

    start: float
    end: float
    text: str
    words: List[Word]


@dataclass
class Window:
    """A candidate span: one or more consecutive units."""

    start: float
    end: float
    text: str
    first_unit: int
    last_unit: int
    words: List[Word]

    @property
    def duration(self) -> float:
        return self.end - self.start


def split_units(
    transcript: Transcript,
    pause: float = PAUSE_SPLIT_SECONDS,
    split_on_clause: bool = False,
) -> List[Unit]:
    """Split the transcript at sentence ends and at noticeable pauses.

    ``split_on_clause`` also breaks at commas and at a few conjunctions, which
    brings a unit closer to the size of an annotated passage (median 2.9 s) than
    a sentence is.
    """
    units: List[Unit] = []

    for segment in transcript.segments:
        words = segment.words
        if not words:
            if segment.text:
                units.append(
                    Unit(segment.start, segment.end, segment.text.strip(), [])
                )
            continue

        current: List[Word] = []
        for index, word in enumerate(words):
            if current and word.start - current[-1].end > pause:
                units.append(_unit(current))
                current = []

            if (split_on_clause and current
                    and word.word.strip().lower().lstrip(',') in CLAUSE_STARTERS):
                units.append(_unit(current))
                current = []

            current.append(word)

            stripped = word.word.strip()
            ends_sentence = stripped.endswith(SENTENCE_END)
            ends_clause = split_on_clause and stripped.endswith(',')
            is_last = index == len(words) - 1
            if ends_sentence or ends_clause or is_last:
                units.append(_unit(current))
                current = []

        if current:
            units.append(_unit(current))

    return [unit for unit in units if unit.text]


def _unit(words: Sequence[Word]) -> Unit:
    return Unit(
        start=float(words[0].start),
        end=float(words[-1].end),
        text=''.join(word.word for word in words).strip(),
        words=list(words),
    )


def build_windows(
    units: Sequence[Unit],
    max_units: int = 3,
    max_duration: float = 14.0,
) -> List[Window]:
    """Every run of up to ``max_units`` consecutive units."""
    windows: List[Window] = []

    for start_index in range(len(units)):
        words: List[Word] = []
        texts: List[str] = []

        for offset in range(max_units):
            end_index = start_index + offset
            if end_index >= len(units):
                break

            unit = units[end_index]
            words = words + unit.words
            texts.append(unit.text)

            start = units[start_index].start
            end = unit.end
            if end - start > max_duration and offset > 0:
                break

            windows.append(
                Window(
                    start=start,
                    end=end,
                    text=' '.join(texts),
                    first_unit=start_index,
                    last_unit=end_index,
                    words=list(words),
                )
            )

    return windows


def tighten(
    window: Window,
    matched_stems: Sequence[str],
    pad: float = 0.35,
    min_duration: float = 1.2,
    max_duration: float = 6.0,
) -> Tuple[float, float]:
    """Shrink a window to the words that actually matched.

    Returning the passage rather than the clip is most of the evidence score:
    the annotation is a phrase, so a window that spans three utterances loses
    two thirds of its IoU even when the right one is inside it.
    """
    from .textutil import stem as stem_token
    from .textutil import tokenize

    if not window.words:
        return window.start, window.end

    wanted = set(matched_stems)
    hits = [
        word for word in window.words
        if any(stem_token(token) in wanted for token in tokenize(word.word))
    ]

    if not hits:
        start, end = window.start, window.end
    else:
        start = min(word.start for word in hits) - pad
        end = max(word.end for word in hits) + pad

    start = max(window.start - pad, start)
    end = min(window.end + pad, end)

    if end - start < min_duration:
        centre = (start + end) / 2
        start, end = centre - min_duration / 2, centre + min_duration / 2

    if end - start > max_duration:
        centre = (start + end) / 2
        start, end = centre - max_duration / 2, centre + max_duration / 2

    return max(0.0, start), max(start + 0.05, end)


def matched_unit_span(
    window: Window,
    matched_stems: Sequence[str],
    units: Sequence[Unit],
) -> Tuple[float, float]:
    """The one utterance inside the window carrying most of the matched words.

    An annotated passage is a phrase, and an utterance is the closest thing the
    transcript has to one. Shrinking further, to just the words that matched,
    clips more than the narrowness wins back.
    """
    from .textutil import content_stems

    wanted = set(matched_stems)
    best_index, best_hits = window.first_unit, -1

    for index in range(window.first_unit, min(window.last_unit, len(units) - 1) + 1):
        hits = sum(
            1 for word in units[index].words
            for token in content_stems(word.word)
            if token in wanted
        )
        if hits > best_hits:
            best_index, best_hits = index, hits

    unit = units[best_index]
    return float(unit.start), float(unit.end)


def calibrate(
    span: Tuple[float, float],
    width_scale: float = 1.0,
    start_offset: float = 0.0,
    end_offset: float = 0.0,
) -> Tuple[float, float]:
    """Correct the systematic part of the boundary error.

    Utterance boundaries run past the annotated phrase by a fairly constant
    amount, so a scale about the midpoint and one offset per edge recover most
    of it. Fitted by ``tools/calibrate_spans.py``; refit whenever the ASR
    backend or the span policy changes, since the bias belongs to those.
    """
    centre = (span[0] + span[1]) / 2.0
    half = (span[1] - span[0]) / 2.0 * width_scale

    start = max(0.0, centre - half + start_offset)
    end = max(start + 0.05, centre + half + end_offset)
    return start, end


def matched_run_span(
    window: Window,
    matched_stems: Sequence[str],
    units: Sequence[Unit],
    max_units: int = 3,
    skip_prompt: bool = True,
) -> Tuple[float, float]:
    """First to last utterance in the window that carries a matched word.

    A claim is often stated across a clause boundary — "and fluconazole, 50
    milligrams, for seven days" — so the passage is a short run of utterances
    rather than exactly one. Capped, because an uncapped run degenerates to the
    whole window, which scores 0.28.

    ``skip_prompt`` walks past a question asked in the room to the reply it
    elicited. A question put to the patient shares its words with the question
    we are asked, so lexical matching lands on the prompt while the annotation
    marks the answer.
    """
    from .textutil import content_stems

    wanted = set(matched_stems)
    last_unit = min(window.last_unit, len(units) - 1)

    hits = [
        index for index in range(window.first_unit, last_unit + 1)
        if any(token in wanted
               for word in units[index].words
               for token in content_stems(word.word))
    ]
    if not hits:
        return float(window.start), float(window.end)

    first = hits[0]
    if skip_prompt:
        while first < len(units) - 1 and units[first].text.strip().endswith('?'):
            first += 1

    last = min(first + max_units - 1, len(units) - 1, max(hits[-1], first))
    return float(units[first].start), float(units[last].end)


def word_span(words: Sequence[Word]) -> Optional[Tuple[float, float]]:
    if not words:
        return None
    return float(words[0].start), float(words[-1].end)
