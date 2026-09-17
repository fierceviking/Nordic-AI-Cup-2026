"""Experiment 4: dense retrieval plus an NLI reader.

Two things the lexical system cannot do. Retrieval by embedding finds the
passage when the question and the transcript share no words ("listened with a
stethoscope" against "let me listen to your chest"). Entailment reads the
passage rather than matching it, which is the only way a hard negative —
lexically almost identical to the truth — comes out as *no*.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..asr import Transcript
from ..features import dedupe_overlapping
from ..textutil import content_stems, declarative
from ..windows import build_windows, split_units, tighten
from .lexical import LexicalApproach, Match

logger = logging.getLogger(__name__)

Span = Tuple[float, float]

EMBED_MODEL = 'BAAI/bge-base-en-v1.5'
NLI_MODEL = 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'
QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '

# Sweeps build one instance per parameter cell; the weights are the same every
# time, so they are loaded once per process.
_LOADED: dict = {}


def _load_embedder(name: str, device: str):
    key = ('embed', name, device)
    if key not in _LOADED:
        from sentence_transformers import SentenceTransformer

        _LOADED[key] = SentenceTransformer(name, device=device)
    return _LOADED[key]


def _load_nli(name: str, device: str):
    key = ('nli', name, device)
    if key not in _LOADED:
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)

        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(
            name,
            dtype=torch.float16 if device == 'cuda' else torch.float32,
        ).to(device).eval()
        _LOADED[key] = (tokenizer, model)
    return _LOADED[key]


class NeuralApproach:
    name = 'neural'

    def __init__(
        self,
        embed_model: str = EMBED_MODEL,
        nli_model: str = NLI_MODEL,
        alpha: float = 0.3,
        top_k: int = 6,
        entail_threshold: float = 0.35,
        context_units: int = 2,
        max_units: int = 3,
        span_pad: float = 0.3,
        min_span: float = 2.0,
        max_span: float = 6.0,
        use_nli: bool = True,
        spans_for_no: bool = True,
        device: Optional[str] = None,
    ) -> None:
        self.alpha = alpha
        self.top_k = top_k
        self.entail_threshold = entail_threshold
        self.context_units = context_units
        self.max_units = max_units
        self.span_pad = span_pad
        self.min_span = min_span
        self.max_span = max_span
        self.use_nli = use_nli
        self.spans_for_no = spans_for_no

        import torch

        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedder = _load_embedder(embed_model, self.device)
        self.lexical = LexicalApproach(max_units=max_units, length_penalty=0.0)

        self.nli = None
        if use_nli:
            self.nli_tokenizer, self.nli = _load_nli(nli_model, self.device)

            labels = self.nli.config.id2label
            self.entail_index = next(
                index for index, name in labels.items()
                if name.lower().startswith('entail')
            )
            self.contradiction_index = next(
                index for index, name in labels.items()
                if name.lower().startswith('contradict')
            )

    # ------------------------------------------------------------------ #

    def predict(
        self, transcript: Transcript, questions: List[str]
    ) -> List[Tuple[bool, Optional[Span]]]:
        units = split_units(transcript)
        if not units:
            return [(True, None) for _ in questions]

        windows = build_windows(units, max_units=self.max_units)
        window_stems = [set(content_stems(window.text)) for window in windows]

        from .lexical import _inverse_document_frequency

        idf = _inverse_document_frequency(units)

        window_vectors = self.embedder.encode(
            [window.text for window in windows],
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        )
        statements = [declarative(question) for question in questions]
        question_vectors = self.embedder.encode(
            [QUERY_PREFIX + statement for statement in statements],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        similarity = question_vectors @ window_vectors.T  # (questions, windows)

        shortlists: List[List[Match]] = []
        span_matches: List[Optional[Match]] = []

        for index, question in enumerate(questions):
            lexical_matches = self.lexical.score_windows(
                question, windows, window_stems, idf
            )
            if not lexical_matches:
                shortlists.append([])
                span_matches.append(None)
                continue

            # The span is chosen on lexical score alone: adding the dense score
            # or the entailment score moves the window off the phrase that was
            # annotated, and width is most of the evidence score.
            span_matches.append(max(lexical_matches, key=lambda m: m.score))

            cosine = similarity[index]
            scaled = (cosine - cosine.min()) / (np.ptp(cosine) + 1e-6)

            hybrid = [
                Match(match.window, match.coverage, match.matched, match.missing,
                      self.alpha * float(dense) + (1 - self.alpha) * match.coverage)
                for match, dense in zip(lexical_matches, scaled)
            ]
            hybrid.sort(key=lambda match: match.score, reverse=True)
            shortlists.append(dedupe_overlapping(hybrid, self.top_k))

        if not self.use_nli:
            return [
                self._answer_from_retrieval(shortlist, span_match)
                for shortlist, span_match in zip(shortlists, span_matches)
            ]

        premises, pairs = [], []
        for question_index, shortlist in enumerate(shortlists):
            for match in shortlist:
                premises.append(self._premise(units, match))
                pairs.append((question_index, match))

        entailment, contradiction = self._entailment(
            premises, [statements[index] for index, _ in pairs]
        )

        per_question: List[List[Tuple[Match, float, float]]] = [
            [] for _ in questions
        ]
        for (question_index, match), entail, contra in zip(
            pairs, entailment, contradiction
        ):
            per_question[question_index].append((match, float(entail), float(contra)))

        return [
            self._answer_from_nli(candidates, span_match)
            for candidates, span_match in zip(per_question, span_matches)
        ]

    # ------------------------------------------------------------------ #

    def _premise(self, units, match: Match) -> str:
        first = max(0, match.window.first_unit - self.context_units)
        last = min(len(units) - 1, match.window.last_unit + self.context_units)
        return ' '.join(unit.text for unit in units[first: last + 1])

    def _entailment(self, premises: Sequence[str], hypotheses: Sequence[str]):
        torch = self.torch
        entail, contra = [], []

        for start in range(0, len(premises), 32):
            batch_premises = list(premises[start: start + 32])
            batch_hypotheses = list(hypotheses[start: start + 32])

            encoded = self.nli_tokenizer(
                batch_premises, batch_hypotheses, return_tensors='pt',
                truncation=True, padding=True, max_length=256,
            ).to(self.device)

            with torch.inference_mode():
                logits = self.nli(**encoded).logits.float()

            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            entail.extend(probabilities[:, self.entail_index])
            contra.extend(probabilities[:, self.contradiction_index])

        return np.array(entail), np.array(contra)

    def _answer_from_retrieval(
        self, shortlist: Sequence[Match], span_match: Optional[Match]
    ) -> Tuple[bool, Optional[Span]]:
        if not shortlist or span_match is None:
            return True, None
        answer = max(match.coverage for match in shortlist) >= 0.45
        span = self._span(span_match)
        return answer, span if (answer or self.spans_for_no) else None

    def _answer_from_nli(
        self,
        candidates: Sequence[Tuple[Match, float, float]],
        span_match: Optional[Match],
    ) -> Tuple[bool, Optional[Span]]:
        if not candidates or span_match is None:
            return True, None

        answer = max(item[1] for item in candidates) >= self.entail_threshold
        span = self._span(span_match)
        return answer, span if (answer or self.spans_for_no) else None

    def _span(self, match: Match) -> Span:
        return tighten(
            match.window, match.matched, pad=self.span_pad,
            min_duration=self.min_span, max_duration=self.max_span,
        )
