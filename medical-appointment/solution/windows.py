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


def split_units(transcript: Transcript) -> List[Unit]:
    """Split the transcript at sentence ends and at noticeable pauses."""
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
            if current and word.start - current[-1].end > PAUSE_SPLIT_SECONDS:
                units.append(_unit(current))
                current = []

            current.append(word)

            stripped = word.word.strip()
            ends_sentence = stripped.endswith(SENTENCE_END)
            is_last = index == len(words) - 1
            if ends_sentence or is_last:
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


def word_span(words: Sequence[Word]) -> Optional[Tuple[float, float]]:
    if not words:
        return None
    return float(words[0].start), float(words[-1].end)
