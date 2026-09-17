"""Experiments 1 and 2: IDF-weighted lexical retrieval, then rules on top.

Retrieval picks the window of transcript that best covers the question's
content words; the coverage score doubles as the yes/no decision. The rule
layer exists for the hard negatives, which retrieve *well* — they are the true
statement with one detail swapped — and so need a second look at numbers,
polarity and the swapped word itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from rapidfuzz import fuzz

from ..asr import Transcript
from ..textutil import (
    antonym_conflict,
    content_stems,
    extract_quantities,
    quantity_conflict,
    stem,
    tokenize,
)
from ..windows import Unit, Window, build_windows, split_units

Span = Tuple[float, float]


@dataclass
class Match:
    window: Window
    coverage: float
    matched: List[str]
    missing: List[str]
    score: float


class LexicalApproach:
    """Retrieve the best window, answer from how well it covers the question."""

    name = 'lexical'

    def __init__(
        self,
        threshold: float = 0.55,
        max_units: int = 3,
        length_penalty: float = 0.012,
        fuzzy_ratio: float = 88.0,
        use_rules: bool = False,
        tighten_span: bool = True,
        span_pad: float = 0.35,
        max_span: float = 6.0,
        min_span: float = 1.2,
        spans_for_no: bool = True,
    ) -> None:
        self.threshold = threshold
        self.max_units = max_units
        self.length_penalty = length_penalty
        self.fuzzy_ratio = fuzzy_ratio
        self.use_rules = use_rules
        self.tighten_span = tighten_span
        self.span_pad = span_pad
        self.max_span = max_span
        self.min_span = min_span
        self.spans_for_no = spans_for_no
        if use_rules:
            self.name = 'lexical_rules'

    # ------------------------------------------------------------------ #

    def predict(
        self, transcript: Transcript, questions: List[str]
    ) -> List[Tuple[bool, Optional[Span]]]:
        units = split_units(transcript)
        if not units:
            return [(True, None) for _ in questions]

        windows = build_windows(units, max_units=self.max_units)
        idf = _inverse_document_frequency(units)
        window_stems = [set(content_stems(window.text)) for window in windows]

        predictions: List[Tuple[bool, Optional[Span]]] = []
        for question in questions:
            match = self.best_match(question, windows, window_stems, idf)
            answer = self.decide(question, match)
            span = self.span_for(match)
            predictions.append((answer, span if (answer or self.spans_for_no) else None))

        return predictions

    # ------------------------------------------------------------------ #

    def score_windows(
        self,
        question: str,
        windows: Sequence[Window],
        window_stems: Sequence[set],
        idf: Dict[str, float],
    ) -> List[Match]:
        """Every window scored against the question, in window order."""
        query = _unique(content_stems(question, drop_frame=True))
        if not query:
            query = _unique(content_stems(question))
        if not query:
            return []

        weights = [idf.get(token, _DEFAULT_IDF) for token in query]
        total_weight = sum(weights) or 1.0

        matches: List[Match] = []
        for window, stems in zip(windows, window_stems):
            matched: List[str] = []
            weight = 0.0

            for token, token_weight in zip(query, weights):
                if token in stems:
                    matched.append(token)
                    weight += token_weight
                elif self.fuzzy_ratio and _fuzzy_in(token, stems, self.fuzzy_ratio):
                    matched.append(token)
                    weight += token_weight * 0.85

            coverage = weight / total_weight
            matches.append(Match(
                window=window,
                coverage=coverage,
                matched=matched,
                missing=[t for t in query if t not in matched],
                score=coverage - self.length_penalty
                * max(0.0, window.duration - 3.0),
            ))

        return matches

    def best_match(
        self,
        question: str,
        windows: Sequence[Window],
        window_stems: Sequence[set],
        idf: Dict[str, float],
    ) -> Optional[Match]:
        matches = self.score_windows(question, windows, window_stems, idf)
        if not matches:
            return None
        return max(matches, key=lambda match: match.score)

    def decide(self, question: str, match: Optional[Match]) -> bool:
        if match is None:
            return True

        if match.coverage < self.threshold:
            return False

        if self.use_rules and self._contradicted(question, match):
            return False

        return True

    def _contradicted(self, question: str, match: Match) -> bool:
        """The hard-negative tests: a swapped number or a flipped polarity."""
        context = match.window.text

        if quantity_conflict(extract_quantities(question),
                             extract_quantities(context)):
            return True

        if antonym_conflict(question, context):
            return True

        return False

    def span_for(self, match: Optional[Match]) -> Optional[Span]:
        if match is None:
            return None

        if not self.tighten_span:
            return match.window.start, match.window.end

        from ..windows import tighten

        return tighten(
            match.window,
            match.matched,
            pad=self.span_pad,
            min_duration=self.min_span,
            max_duration=self.max_span,
        )


# --------------------------------------------------------------------------- #

_DEFAULT_IDF = 3.0


def _unique(tokens: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


def _inverse_document_frequency(units: Sequence[Unit]) -> Dict[str, float]:
    """IDF over the utterances of this one conversation.

    Nothing carries over between requests, so the corpus is what arrived.
    """
    document_frequency: Dict[str, int] = {}
    for unit in units:
        for token in set(content_stems(unit.text)):
            document_frequency[token] = document_frequency.get(token, 0) + 1

    total = len(units)
    return {
        token: math.log(1 + total / (1 + frequency))
        for token, frequency in document_frequency.items()
    }


def _fuzzy_in(token: str, stems: set, ratio: float) -> bool:
    if len(token) < 5:
        return False
    return any(
        fuzz.ratio(token, candidate) >= ratio
        for candidate in stems
        if abs(len(candidate) - len(token)) <= 3
    )
