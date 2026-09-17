"""Experiment 5: a local LLM chooses between candidate passages.

The first version put the whole transcript in the prompt and asked for an
answer and a verbatim quote. It works, but ~1600 tokens of context times ten
questions fills a 12 GB card, Windows spills the KV cache into shared memory,
and a single conversation takes minutes instead of seconds — useless against a
60 second budget. See EXPERIMENTS.md.

So the LLM is given only what retrieval already shortlisted: the question and
six candidate passages, numbered. It picks the one that establishes the claim,
or says none of them do. Prompts are ~300 tokens, which fits comfortably, and
choosing between candidates is exactly the job lexical scoring fails at when a
drug is named three times in one consultation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Optional, Sequence, Tuple

from ..asr import Transcript, Word
from ..textutil import content_stems, tokenize
from ..windows import Unit, build_windows, split_units, tighten
from .lexical import LexicalApproach, Match

logger = logging.getLogger(__name__)

Span = Tuple[float, float]

LLM_MODEL = 'Qwen/Qwen3-4B-Instruct-2507'

SYSTEM_PROMPT = (
    'You judge a yes/no question about a recorded doctor-patient consultation, '
    'given the passages of the transcript that mention the subject.\n'
    'Answer "yes" only if one passage establishes exactly what the question '
    'states. A different dose, drug, duration or body part, or the opposite '
    'finding, means "no" even when the wording is close. A subject that is not '
    'there at all means "no".\n'
    'Give the number of the passage the answer is read off, and quote the '
    'shortest part of it that establishes the claim, copied word for word.\n'
    'Reply with JSON only: {"answer":"yes"|"no","passage":<number or null>,'
    '"quote":"<verbatim text or empty>"}'
)

_LOADED: dict = {}


def _load(name: str, device: str):
    if name not in _LOADED:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(name, padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(
            name,
            dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
            device_map=device,
        ).eval()
        _LOADED[name] = (tokenizer, model)
    return _LOADED[name]


class LLMApproach:
    name = 'llm'

    def __init__(
        self,
        model_name: str = LLM_MODEL,
        top_k: int = 6,
        context_units: int = 1,
        max_units: int = 3,
        max_new_tokens: int = 64,
        batch_size: int = 5,
        span_pad: float = 0.3,
        min_span: float = 2.0,
        max_span: float = 6.0,
        quote_min_ratio: float = 75.0,
        use_quote: bool = True,
        spans_for_no: bool = True,
        device: Optional[str] = None,
    ) -> None:
        import torch

        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer, self.model = _load(model_name, self.device)

        self.top_k = top_k
        self.context_units = context_units
        self.max_units = max_units
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.span_pad = span_pad
        self.min_span = min_span
        self.max_span = max_span
        self.quote_min_ratio = quote_min_ratio
        self.use_quote = use_quote
        self.spans_for_no = spans_for_no

        self.lexical = LexicalApproach(max_units=max_units, length_penalty=0.0)

    # ------------------------------------------------------------------ #

    def predict(
        self, transcript: Transcript, questions: List[str]
    ) -> List[Tuple[bool, Optional[Span]]]:
        units = split_units(transcript)
        if not units:
            return [(True, None) for _ in questions]

        shortlists = self.shortlist(units, questions)
        words = [word for unit in units for word in unit.words]

        replies = self._generate([
            self._prompt(question, units, shortlist)
            for question, shortlist in zip(questions, shortlists)
        ])

        predictions: List[Tuple[bool, Optional[Span]]] = []
        for shortlist, reply in zip(shortlists, replies):
            answer, choice, quote = parse_reply(reply)
            span = self._span(words, shortlist, choice, quote)
            predictions.append(
                (answer, span if (answer or self.spans_for_no) else None)
            )

        return predictions

    def shortlist(
        self, units: Sequence[Unit], questions: Sequence[str]
    ) -> List[List[Match]]:
        from ..features import dedupe_overlapping
        from .lexical import _inverse_document_frequency

        windows = build_windows(units, max_units=self.max_units)
        stems = [set(content_stems(window.text)) for window in windows]
        idf = _inverse_document_frequency(units)

        shortlists = []
        for question in questions:
            matches = self.lexical.score_windows(question, windows, stems, idf)
            matches.sort(key=lambda match: match.score, reverse=True)
            shortlists.append(dedupe_overlapping(matches, self.top_k))
        return shortlists

    # ------------------------------------------------------------------ #

    def _prompt(
        self, question: str, units: Sequence[Unit], shortlist: Sequence[Match]
    ) -> str:
        passages = '\n'.join(
            f'{index + 1}. {self._with_context(units, match)}'
            for index, match in enumerate(shortlist)
        )
        return self.tokenizer.apply_chat_template(
            [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content':
                    f'Question: {question}\n\nPassages:\n{passages}'},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _with_context(self, units: Sequence[Unit], match: Match) -> str:
        first = max(0, match.window.first_unit - self.context_units)
        last = min(len(units) - 1, match.window.last_unit + self.context_units)
        return ' '.join(unit.text for unit in units[first: last + 1])

    def _generate(self, prompts: Sequence[str]) -> List[str]:
        torch = self.torch
        replies: List[str] = []

        for start in range(0, len(prompts), self.batch_size):
            batch = list(prompts[start: start + self.batch_size])
            encoded = self.tokenizer(
                batch, return_tensors='pt', padding=True
            ).to(self.model.device)

            with torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                    or self.tokenizer.eos_token_id,
                )

            prompt_length = encoded['input_ids'].shape[1]
            for index in range(len(batch)):
                replies.append(self.tokenizer.decode(
                    generated[index][prompt_length:], skip_special_tokens=True
                ))

        return replies

    def _span(
        self,
        words: Sequence[Word],
        shortlist: Sequence[Match],
        choice: Optional[int],
        quote: str,
    ) -> Optional[Span]:
        if not shortlist:
            return None

        match = shortlist[0]
        if choice is not None and 1 <= choice <= len(shortlist):
            match = shortlist[choice - 1]

        if self.use_quote and quote:
            aligned = align_quote(words, quote, self.quote_min_ratio)
            if aligned is not None:
                return self._shape(aligned)

        return tighten(
            match.window, match.matched, pad=self.span_pad,
            min_duration=self.min_span, max_duration=self.max_span,
        )

    def _shape(self, span: Span) -> Span:
        start, end = span[0] - self.span_pad, span[1] + self.span_pad

        if end - start < self.min_span:
            centre = (start + end) / 2
            start, end = centre - self.min_span / 2, centre + self.min_span / 2
        if end - start > self.max_span:
            centre = (start + end) / 2
            start, end = centre - self.max_span / 2, centre + self.max_span / 2

        return max(0.0, start), max(start + 0.05, end)


# --------------------------------------------------------------------------- #

_JSON_RE = re.compile(r'\{.*?\}', re.DOTALL)


def parse_reply(reply: str) -> Tuple[bool, Optional[int], str]:
    """Read the model's JSON. Never raises: a guess beats an exception."""
    match = _JSON_RE.search(reply)
    if match:
        try:
            payload = json.loads(match.group(0))
            answer = str(payload.get('answer', '')).strip().lower()
            choice = payload.get('passage')
            return (
                answer.startswith('y'),
                int(choice) if isinstance(choice, (int, float)) else None,
                str(payload.get('quote') or '').strip(),
            )
        except (ValueError, TypeError):
            logger.debug('unparsable reply: %r', reply)

    lowered = reply.strip().lower()
    return lowered.startswith('yes') or '"yes"' in lowered, None, ''


def align_quote(
    words: Sequence[Word],
    quote: str,
    min_ratio: float = 75.0,
) -> Optional[Span]:
    """Find where a quoted passage sits in the audio.

    The model copies from the transcript but not always exactly, so this is a
    fuzzy sliding match over the word sequence rather than a string search.
    """
    from rapidfuzz import fuzz

    quote_tokens = tokenize(quote)
    if not quote_tokens or not words:
        return None

    word_tokens = [' '.join(tokenize(word.word)) for word in words]
    target = ' '.join(quote_tokens)
    length = len(quote_tokens)

    best_score, best_range = 0.0, None
    for width in {max(1, length - 2), length, length + 2}:
        for start in range(0, max(1, len(word_tokens) - width + 1)):
            candidate = ' '.join(
                token for token in word_tokens[start: start + width] if token
            )
            if not candidate:
                continue
            score = fuzz.ratio(target, candidate)
            if score > best_score:
                best_score = score
                best_range = (start, min(start + width, len(word_tokens)))

    if best_range is None or best_score < min_ratio:
        return None

    start_index, end_index = best_range
    return float(words[start_index].start), float(words[end_index - 1].end)
