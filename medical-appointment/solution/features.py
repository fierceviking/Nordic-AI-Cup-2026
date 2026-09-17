"""Features for the learned answerer (experiment 3).

All of these are computable from the transcript and the question alone, and
all of them are cheap. The neural scores live in ``neural.py``; nothing here
needs a GPU.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .approaches.lexical import Match, LexicalApproach
from .asr import Transcript
from .textutil import (
    antonym_conflict,
    content_stems,
    extract_quantities,
    numbers_only,
    quantity_conflict,
)
from .windows import Window, build_windows, split_units

FEATURE_NAMES = [
    'coverage_best',
    'coverage_second',
    'coverage_gap',
    'coverage_mean_top3',
    'coverage_whole',
    'query_terms',
    'missing_terms',
    'missing_weight',
    'max_missing_idf',
    'best_duration',
    'quantity_in_question',
    'quantity_conflict',
    'quantity_match',
    'antonym_conflict',
    'question_length',
    'oov_fraction',
    'best_start_fraction',
]


class FeatureExtractor:
    """Shared retrieval front end plus the feature vector on top of it."""

    def __init__(self, max_units: int = 3, **lexical_kwargs) -> None:
        self.lexical = LexicalApproach(max_units=max_units, **lexical_kwargs)

    def prepare(self, transcript: Transcript):
        units = split_units(transcript)
        windows = build_windows(units, max_units=self.lexical.max_units)
        from .approaches.lexical import _inverse_document_frequency

        idf = _inverse_document_frequency(units)
        window_stems = [set(content_stems(window.text)) for window in windows]
        transcript_stems = set(content_stems(transcript.text))
        return units, windows, window_stems, idf, transcript_stems

    def features_for(
        self,
        question: str,
        transcript: Transcript,
        prepared,
    ) -> Tuple[List[float], Optional[Match], List[Match]]:
        units, windows, window_stems, idf, transcript_stems = prepared

        ranked = _rank(self.lexical, question, windows, window_stems, idf)
        best = ranked[0] if ranked else None

        query = set(content_stems(question, drop_frame=True)) or \
            set(content_stems(question))

        question_quantities = extract_quantities(question)
        context = best.window.text if best else ''
        context_quantities = extract_quantities(context)

        whole_matched = len(query & transcript_stems)
        oov = [token for token in query if token not in transcript_stems]

        coverage_best = best.coverage if best else 0.0
        coverage_second = ranked[1].coverage if len(ranked) > 1 else 0.0
        top3 = [match.coverage for match in ranked[:3]] or [0.0]

        missing = best.missing if best else list(query)
        missing_weight = sum(idf.get(token, 3.0) for token in missing)
        max_missing_idf = max(
            [idf.get(token, 3.0) for token in missing] or [0.0]
        )

        duration = transcript.duration or 1.0

        features = [
            coverage_best,
            coverage_second,
            coverage_best - coverage_second,
            sum(top3) / len(top3),
            whole_matched / max(len(query), 1),
            float(len(query)),
            float(len(missing)),
            missing_weight,
            max_missing_idf,
            best.window.duration if best else 0.0,
            float(bool(question_quantities)),
            float(quantity_conflict(question_quantities, context_quantities)),
            float(bool(
                numbers_only(question_quantities)
                & numbers_only(context_quantities)
            )),
            float(antonym_conflict(question, context)),
            float(len(question.split())),
            len(oov) / max(len(query), 1),
            (best.window.start / duration) if best else 0.0,
        ]

        return features, best, ranked


def _rank(
    lexical: LexicalApproach,
    question: str,
    windows: Sequence[Window],
    window_stems: Sequence[set],
    idf: Dict[str, float],
    top_k: int = 8,
) -> List[Match]:
    """Every window scored, best first, one per region of the audio."""
    matches = lexical.score_windows(question, windows, window_stems, idf)
    matches.sort(key=lambda match: match.score, reverse=True)
    return dedupe_overlapping(matches, top_k)


def dedupe_overlapping(matches: Sequence[Match], top_k: int) -> List[Match]:
    """Keep the best window per region rather than k overlapping copies."""
    kept: List[Match] = []
    for match in matches:
        if any(
            min(match.window.end, other.window.end)
            - max(match.window.start, other.window.start) > 0.5
            for other in kept
        ):
            continue
        kept.append(match)
        if len(kept) >= top_k:
            break
    return kept
