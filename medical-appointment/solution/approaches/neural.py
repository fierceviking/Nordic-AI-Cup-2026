"""Experiment 4: dense retrieval plus an NLI reader.

Two things the lexical system cannot do. Retrieval by embedding finds the
passage when the question and the transcript share no words ("listened with a
stethoscope" against "let me listen to your chest"). Entailment reads the
passage rather than matching it, which is the only way a hard negative —
lexically almost identical to the truth — comes out as *no*.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..asr import Transcript
from ..features import dedupe_overlapping
from ..textutil import content_stems, declarative
from ..windows import (build_windows, calibrate, matched_run_span,  # noqa: E501
                       matched_unit_span, split_units, tighten)
from .lexical import LexicalApproach, Match

logger = logging.getLogger(__name__)

Span = Tuple[float, float]

EMBED_MODEL = 'BAAI/bge-base-en-v1.5'
NLI_MODEL = 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'
RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L6-v2'
QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '

# Sweeps build one instance per parameter cell; the weights are the same every
# time, so they are loaded once per process.
_LOADED: dict = {}


def best_device() -> str:
    """cuda, else Apple's Metal backend, else CPU. Overridable with TORCH_DEVICE."""
    override = os.environ.get('TORCH_DEVICE')
    if override:
        return override

    import torch

    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


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
            # float16 on MPS trips MPSGraph broadcast errors; only CUDA gets it.
            dtype=torch.float16 if device == 'cuda' else torch.float32,
        ).to(device).eval()
        _LOADED[key] = (tokenizer, model)
    return _LOADED[key]


def _load_reranker(name: str, device: str):
    key = ('rerank', name, device)
    if key not in _LOADED:
        from sentence_transformers import CrossEncoder

        _LOADED[key] = CrossEncoder(name, device=device, max_length=256)
    return _LOADED[key]


class NeuralApproach:
    name = 'neural'

    def __init__(
        self,
        embed_model: str = EMBED_MODEL,
        nli_model: str = NLI_MODEL,
        rerank_model: str = RERANK_MODEL,
        alpha: float = 0.3,
        top_k: int = 6,
        entail_threshold: float = 0.35,
        context_units: int = 2,
        max_units: int = 3,
        span_mode: str = 'matched_run',
        span_units: int = 2,
        skip_prompt: bool = True,
        span_pad: float = 0.3,
        min_span: float = 2.0,
        max_span: float = 6.0,
        # Fitted on mlx-large-v3-turbo by tools/calibrate_spans.py. The optimum
        # is a property of the ASR, not the case: parakeet wants -0.12/-0.50.
        width_scale: float = 0.98,
        start_offset: float = 0.22,
        end_offset: float = 0.02,
        use_nli: bool = True,
        use_reranker: bool = False,
        rerank_weight: float = 0.5,
        rerank_top: int = 20,
        nli_span_weight: float = 0.0,
        veto_threshold: float = 0.3,
        veto_top: int = 10,
        spans_for_no: bool = True,
        device: Optional[str] = None,
    ) -> None:
        self.alpha = alpha
        self.top_k = top_k
        self.entail_threshold = entail_threshold
        self.context_units = context_units
        self.max_units = max_units
        self.span_mode = span_mode
        self.span_units = span_units
        self.skip_prompt = skip_prompt
        self.span_pad = span_pad
        self.min_span = min_span
        self.max_span = max_span
        self.width_scale = width_scale
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.use_nli = use_nli
        self.use_reranker = use_reranker
        self.rerank_weight = rerank_weight
        self.rerank_top = rerank_top
        self.nli_span_weight = nli_span_weight
        self.veto_threshold = veto_threshold
        self.veto_top = veto_top
        self.spans_for_no = spans_for_no

        import torch

        self.torch = torch
        self.device = device or best_device()

        self.embedder = _load_embedder(embed_model, self.device)
        self.lexical = LexicalApproach(max_units=max_units, length_penalty=0.0)

        self.reranker = None
        if use_reranker:
            self.reranker = _load_reranker(rerank_model, self.device)

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
        span_pools: List[List[Match]] = []

        for index, question in enumerate(questions):
            lexical_matches = self.lexical.score_windows(
                question, windows, window_stems, idf
            )
            if not lexical_matches:
                shortlists.append([])
                span_matches.append(None)
                span_pools.append([])
                continue

            # The span is chosen on lexical score alone: adding the dense score
            # or the entailment score moves the window off the phrase that was
            # annotated, and width is most of the evidence score.
            span_matches.append(max(lexical_matches, key=lambda m: m.score))
            span_pools.append(
                sorted(lexical_matches, key=lambda m: m.score, reverse=True)[
                    : self.rerank_top
                ]
            )

            cosine = similarity[index]
            scaled = (cosine - cosine.min()) / (np.ptp(cosine) + 1e-6)

            hybrid = [
                Match(match.window, match.coverage, match.matched, match.missing,
                      self.alpha * float(dense) + (1 - self.alpha) * match.coverage)
                for match, dense in zip(lexical_matches, scaled)
            ]
            hybrid.sort(key=lambda match: match.score, reverse=True)
            shortlists.append(dedupe_overlapping(hybrid, self.top_k))

        if self.reranker is not None:
            span_matches = self._rerank_spans(
                statements, span_pools, span_matches
            )

        if self.nli is not None and self.veto_threshold > 0.0:
            span_matches = self._veto_spans(
                statements, span_pools, span_matches
            )

        if not self.use_nli:
            return [
                self._answer_from_retrieval(shortlist, span_match, units)
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
            self._answer_from_nli(candidates, span_match, units)
            for candidates, span_match in zip(per_question, span_matches)
        ]

    # ------------------------------------------------------------------ #

    def _rerank_spans(
        self,
        statements: Sequence[str],
        span_pools: Sequence[List[Match]],
        span_matches: List[Optional[Match]],
    ) -> List[Optional[Match]]:
        """Pick the span window with a cross-encoder as well as lexically.

        Lexical coverage cannot separate two mentions of the same fact, which is
        the largest single source of missed spans: the drug is named early and
        prescribed late, and both mentions match the question word for word.
        Every question's candidates go through in one batch.
        """
        pairs = [
            (statements[index], match.window.text)
            for index, pool in enumerate(span_pools)
            for match in pool
        ]
        if not pairs:
            return span_matches

        scores = self.reranker.predict(
            pairs, batch_size=128, show_progress_bar=False
        )

        out = list(span_matches)
        cursor = 0
        for index, pool in enumerate(span_pools):
            if not pool:
                continue

            raw = np.asarray(scores[cursor:cursor + len(pool)], dtype=float)
            cursor += len(pool)

            scaled = (raw - raw.min()) / (np.ptp(raw) + 1e-6)
            coverage = np.asarray([match.coverage for match in pool])
            blended = (
                self.rerank_weight * scaled
                + (1 - self.rerank_weight) * coverage
            )
            out[index] = pool[int(np.argmax(blended))]

        return out

    # ------------------------------------------------------------------ #

    def _veto_spans(
        self,
        statements: Sequence[str],
        span_pools: Sequence[List[Match]],
        span_matches: List[Optional[Match]],
    ) -> List[Optional[Match]]:
        """Walk down the lexical ordering to the first passage the reader accepts.

        Entailment ranks passages worse than word overlap but judges a single one
        well — 0.96 accuracy against 0.48 ranking tIoU. Used as a veto rather
        than a ranker it keeps coverage's ordering and only skips the picks the
        reader rejects, which moves about one span in ten.

        The premise is the window alone: with neighbouring utterances added, a
        window scores well when its neighbours hold the fact, which is the wrong
        question to ask about the text being emitted.
        """
        pools = [pool[: self.veto_top] for pool in span_pools]

        premises, hypotheses = [], []
        for index, pool in enumerate(pools):
            for match in pool:
                premises.append(match.window.text)
                hypotheses.append(statements[index])

        if not premises:
            return span_matches

        entailment, _ = self._entailment(premises, hypotheses)

        out = list(span_matches)
        cursor = 0
        for index, pool in enumerate(pools):
            if not pool:
                continue

            scores = np.asarray(entailment[cursor:cursor + len(pool)], dtype=float)
            cursor += len(pool)

            passing = np.flatnonzero(scores >= self.veto_threshold)
            if len(passing):
                out[index] = pool[int(passing[0])]

        return out

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
        span = self._span(span_match, units)
        return answer, span if (answer or self.spans_for_no) else None

    def _answer_from_nli(
        self,
        candidates: Sequence[Tuple[Match, float, float]],
        span_match: Optional[Match],
        units: Sequence,
    ) -> Tuple[bool, Optional[Span]]:
        if not candidates or span_match is None:
            return True, None

        entail = [item[1] for item in candidates]
        answer = max(entail) >= self.entail_threshold

        # Entailment knows which of two identical-sounding mentions actually
        # establishes the claim, which lexical coverage cannot. It also prefers
        # a wider premise, so how far to trust it is a weight, not a switch.
        if self.nli_span_weight > 0.0:
            scores = np.asarray(entail, dtype=float)
            scaled = (scores - scores.min()) / (np.ptp(scores) + 1e-6)
            coverage = np.asarray([item[0].coverage for item in candidates])
            blended = (
                self.nli_span_weight * scaled
                + (1 - self.nli_span_weight) * coverage
            )
            span_match = candidates[int(np.argmax(blended))][0]

        span = self._span(span_match, units)
        return answer, span if (answer or self.spans_for_no) else None

    def _span(self, match: Match, units: Sequence) -> Span:
        if self.span_mode == 'matched_run':
            span = matched_run_span(
                match.window, match.matched, units, self.span_units,
                self.skip_prompt,
            )
        elif self.span_mode == 'matched_unit':
            span = matched_unit_span(match.window, match.matched, units)
        else:
            span = tighten(
                match.window, match.matched, pad=self.span_pad,
                min_duration=self.min_span, max_duration=self.max_span,
            )
        return calibrate(
            span, self.width_scale, self.start_offset, self.end_offset
        )
