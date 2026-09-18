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
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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


QUOTE_MODEL = 'mlx-community/Qwen3-4B-Instruct-2507-4bit'
QUOTE_PROMPT = '''Answer the yes/no questions using only the consultation below.
Return a JSON object keyed by question number (1-based). Each value must contain:
"answer": a JSON boolean, "unit": the transcript line number where the evidence
starts, and "quote": a verbatim, contiguous phrase copied from that line or its
immediate continuation. Do not paraphrase or give timestamps.

For a yes, quote the complete clause that establishes the specific claim,
including the relevant action, finding, dose, duration or negation. Do not quote
unrelated sentences. For a no, quote the contradicting detail if there is one;
if the subject is absent, use "unit": null and "quote": "".
Questions asked during the consultation are not evidence that their premise is
true: read the reply too. Use the passage that directly establishes the claim.
Different questions may legitimately share the same evidence.

Transcript (0-based line numbers):
{transcript}

Questions:
{questions}

Return JSON only. Example format:
{{"1": {{"answer": true, "unit": 7, "quote": "copied phrase"}}}}
'''

_QUOTE_WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix='quote-mlx')


def evidence_features(transcript: Transcript, question: str, span: Optional[Span]):
    import math

    if span is None:
        return [0.0] * 9
    text = ''.join(word.word for word in transcript.words
                   if span[0] <= (word.start + word.end) / 2 <= span[1])
    terms = set(content_stems(text))
    query = set(content_stems(question))
    matched = len(terms & query)
    return [
        math.log1p(span[1] - span[0]), math.log1p(len(tokenize(text))),
        matched / max(1, len(query)), matched / max(1, len(terms)),
        float(len(terms) <= 1), float(text.strip().endswith('?')),
        span[0] / max(1.0, transcript.duration),
        float(matched == 0), float(matched >= 2),
    ]


def pair_features(transcript: Transcript, question: str, first: Optional[Span],
                  second: Optional[Span], judge_second: bool):
    from utils import temporal_iou

    left = evidence_features(transcript, question, first)
    right = evidence_features(transcript, question, second)
    overlap = temporal_iou(first, second) if first is not None else 0.0
    return [other - own for own, other in zip(left, right)] + [
        float(judge_second), overlap, float(judge_second) * overlap,
    ]


def pair_advantage(features, model) -> float:
    return float(model['intercept'] + sum(
        weight * (value - mean) / scale
        for value, mean, scale, weight in zip(
            features, model['mean'], model['scale'], model['coef'], strict=True)
    ))


def exact_quote_spans(words: Sequence[Word], quote: str) -> List[Span]:
    tokens, owners = [], []
    for word_index, word in enumerate(words):
        word_tokens = tokenize(word.word)
        tokens.extend(word_tokens)
        owners.extend([word_index] * len(word_tokens))
    wanted = tokenize(quote)
    if not wanted:
        return []
    spans = []
    for start in range(len(tokens) - len(wanted) + 1):
        if tokens[start:start + len(wanted)] == wanted:
            spans.append((float(words[owners[start]].start),
                          float(words[owners[start + len(wanted) - 1]].end)))
    return spans


def quote_span(units: Sequence[Unit], unit_index: int, quote: str) -> Optional[Span]:
    if not quote.strip():
        return None
    words = ([word for unit in units[unit_index:unit_index + 3] for word in unit.words]
             if 0 <= unit_index < len(units) else [])
    local = exact_quote_spans(words, quote)
    if local:
        return local[0]
    global_matches = exact_quote_spans([word for unit in units for word in unit.words], quote)
    if len(global_matches) == 1:
        return global_matches[0]
    return align_quote(words, quote, min_ratio=90.0)


def parse_quotes(reply: str, units: Sequence[Unit], count: int):
    match = re.search(r'\{.*\}', reply, re.DOTALL)
    try:
        payload = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        payload = {}
        decoder = json.JSONDecoder()
        for entry in re.finditer(r'"(\d+)"\s*:\s*(?=\{)', reply):
            try:
                value, _ = decoder.raw_decode(reply[entry.end():])
            except json.JSONDecodeError:
                continue
            payload[entry.group(1)] = value
    if not isinstance(payload, dict):
        payload = {}

    predictions = []
    valid_count = 0
    for index in range(count):
        item = payload.get(str(index + 1), {})
        if not isinstance(item, dict) or type(item.get('answer')) is not bool:
            predictions.append((True, None))
            continue
        valid_count += 1
        anchor, quote = item.get('unit'), item.get('quote')
        span = (quote_span(units, anchor, quote)
                if type(anchor) is int and isinstance(quote, str) else None)
        predictions.append((item['answer'], span))
    return predictions, valid_count


class QuoteApproach:
    name = 'quote'

    def __init__(self, model_name: str = QUOTE_MODEL, max_tokens: int = 1200,
                 trace_path: Optional[str] = None, examples_path: Optional[str] = None,
                 example_count: int = 6, width_scale: float = 1.0,
                 start_offset: float = 0.0, end_offset: float = 0.0,
                 adjudicate: bool = False, refine_evidence: bool = False,
                 reasoned_choice: bool = False, review_transcript: bool = False,
                 raw_evidence_text: bool = False, pair_selector_path: Optional[str] = None):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.example_count = example_count
        self.examples = []
        if examples_path:
            self.examples = json.loads(Path(examples_path).read_text(encoding='utf-8'))
        self.calibration = (width_scale, start_offset, end_offset)
        self.adjudicate = adjudicate
        self.refine_evidence = refine_evidence
        self.reasoned_choice = reasoned_choice
        self.review_transcript = review_transcript
        self.raw_evidence_text = raw_evidence_text
        self.pair_selector = (json.loads(Path(pair_selector_path).read_text(encoding='utf-8'))
                      if pair_selector_path else None)
        self.trace_path = Path(trace_path) if trace_path else None
        self.last_reply = ''
        self.last_adjudication = ''
        self.last_valid_count = 0
        _QUOTE_WORKER.submit(self._load).result()

    def _load(self):
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        self.model, self.tokenizer = load(self.model_name)
        self.sampler = make_sampler(temp=0.0)
        self.baseline = None
        if self.adjudicate:
            from .neural import NeuralApproach
            self.baseline = NeuralApproach()

    def predict(self, transcript: Transcript, questions: List[str]):
        return _QUOTE_WORKER.submit(self._predict, transcript, questions).result()

    def _examples_prompt(self, questions: List[str]) -> str:
        if not self.examples or self.example_count <= 0:
            return ''
        queries = [set(content_stems(question)) for question in questions]
        def relevance(example):
            terms = set(content_stems(example['question']))
            return max(len(terms & query) / max(1, len(terms | query))
                       for query in queries)
        selected, seen_groups = [], set()
        for example in sorted(self.examples, key=relevance, reverse=True):
            if example['group'] in seen_groups:
                continue
            seen_groups.add(example['group'])
            selected.append('Transcript excerpt:\n' + example['context']
                            + '\nQuestion: ' + example['question']
                            + '\nOutput: ' + json.dumps({'1': example['output']}))
            if len(selected) == self.example_count:
                break
        return ('Examples from other consultations. Match their evidence specificity; '
                'do not use their facts for the new consultation.\n\n'
                + '\n\n'.join(selected) + '\n\nNew consultation:\n')

    def _predict(self, transcript: Transcript, questions: List[str]):
        from mlx_lm import generate
        from ..windows import calibrate

        units = split_units(transcript)
        if not units:
            return [(True, None) for _ in questions]
        content = QUOTE_PROMPT.format(
            transcript='\n'.join(f'[{index}] {unit.text}'
                                 for index, unit in enumerate(units)),
            questions='\n'.join(f'{index + 1}. {question}'
                                for index, question in enumerate(questions)),
        )
        prompt = self.tokenizer.apply_chat_template(
            [{'role': 'system', 'content': 'Extract evidence from the provided transcript.'},
             {'role': 'user', 'content': self._examples_prompt(questions) + content}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        started = time.perf_counter()
        self.last_adjudication = ''
        self.last_reply = generate(
            self.model, self.tokenizer, prompt=prompt, sampler=self.sampler,
            max_tokens=self.max_tokens, verbose=False,
        )
        predictions, self.last_valid_count = parse_quotes(self.last_reply, units, len(questions))
        predictions = [(answer, calibrate(span, *self.calibration) if span else None)
                   for answer, span in predictions]
        if self.baseline is not None:
            try:
                predictions = self._adjudicate(transcript, questions, predictions)
            except Exception:
                logger.exception('Evidence adjudication failed; retaining quote predictions')
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps({
                    'model': self.model_name, 'asr': transcript.model,
                    'questions': questions, 'reply': self.last_reply,
                    'adjudication': self.last_adjudication,
                    'valid_count': self.last_valid_count,
                    'seconds': time.perf_counter() - started,
                }) + '\n')
        logger.info('Quote extraction: %d/%d valid answers, %d grounded spans',
                    self.last_valid_count, len(questions),
                    sum(span is not None for _, span in predictions))
        return predictions

    def _adjudicate(self, transcript, questions, predictions):
        from mlx_lm import generate
        from ..windows import calibrate
        from utils import temporal_iou

        prior = self.baseline.predict(transcript, questions)
        pairs = []
        def context(span, calibration):
            if span is None:
                return '<no passage>'
            if self.raw_evidence_text:
                scale, start_offset, end_offset = calibration
                centre = (span[0] + span[1] - start_offset - end_offset) / 2
                half = (span[1] - span[0] - end_offset + start_offset) / (2 * scale)
                span = (max(0.0, centre - half - 1e-6), centre + half + 1e-6)
            before = ''.join(word.word for word in transcript.words
                             if span[0] - 5 <= (word.start + word.end) / 2 < span[0])
            core = ''.join(word.word for word in transcript.words
                           if span[0] <= (word.start + word.end) / 2 <= span[1])
            after = ''.join(word.word for word in transcript.words
                            if span[1] < (word.start + word.end) / 2 <= span[1] + 5)
            return before + ' <evidence>' + core + '</evidence> ' + after

        for index, ((answer, span), (_, other_span)) in enumerate(zip(predictions, prior)):
            overlap = temporal_iou(span, other_span) if span is not None else 0.0
            if not answer or (not self.review_transcript and overlap > 0.8):
                continue
            baseline_calibration = (self.baseline.width_scale, self.baseline.start_offset,
                                    self.baseline.end_offset)
            pairs.append(f'{index + 1}. {questions[index]}\nA: {context(span, self.calibration)}'
                         f'\nB: {context(other_span, baseline_calibration)}')
        if not pairs:
            return predictions
        rules = (
            'Select the better evidence passage for each claim below. Both candidates '
            'are excerpts from the SAME consultation; surrounding context is supplied, '
            'but only text inside <evidence> is cited. Prefer the direct, complete '
            'assertion of the requested fact over merely asking about it, naming the '
            'topic, or using vague pronouns. Prefer an explicit prescription or finding '
            'over a speculative discussion. Keep A if both equally support the claim. '
            'Do not change the claim or answer from medical knowledge. '
        )
        if self.review_transcript:
            units = split_units(transcript)
            rules = (
                'Review evidence for the true clinical claims below. Read the entire '
                'consultation before choosing. The two proposed citations can both '
                'refer to the wrong occurrence. Prefer the first direct clinical '
                'assessment, finding or agreed prescription that establishes the '
                'specific fact, not a later generic recap or a vague confirmation. '
                'A mere request or speculative discussion is not a completed action. '
                'For symptoms/history, cite the specific patient statement; for '
                'assessment/treatment, cite the clinician establishing the fact. '
                'Use a concise self-contained clause: bare yes/no or pronouns alone '
                'lose the finding they refer to. Do not add unrelated details.\n'
                'Return JSON keyed by question number with "choice":"A", "B", '
                'or "C". Use A/B for a good existing citation. Use C when a different '
                'passage or different phrase boundaries are needed, and also give '
                '"unit": its starting line number and "quote": the exact contiguous '
                'phrase from the transcript. Do not change the yes/no answers.\n\n'
                'Full transcript:\n'
                + '\n'.join(f'[{index}] {unit.text}' for index, unit in enumerate(units))
                + '\n\nClaims and proposed citations:\n'
            )
        elif self.refine_evidence:
            rules = (
                'Select and refine a self-contained evidence phrase for each claim. '
                'A and B are excerpts from the SAME consultation. The marked evidence '
                'is a proposed citation, not necessarily the correct boundaries. '
                'Use the surrounding context to resolve who or what was discussed. '
                'A bare yes/no, "none", "both" or "exactly" is not a self-contained '
                'citation: include the specific finding, medicine or plan it confirms. '
                'Do not include unrelated symptoms or details. Prefer a direct statement '
                'over a later paraphrase or generic recap. '
                'Return JSON mapping question number to {"choice":"A" or "B",'
                '"quote":"verbatim contiguous text copied from that excerpt"}. '
                'You may expand or shorten the marked phrase within the supplied '
                'excerpt, but do not invent text or mix A and B. Ignore the '
                '<evidence> tags when copying. No explanations.\n\n'
            )
        elif self.reasoned_choice:
            rules = (
                'Compare two proposed evidence citations for each claim. Both excerpts '
                'come from the same medical consultation. Only the words inside '
                '<evidence> are returned as evidence; outside text is context only. '
                'The claim is already judged true. Decide which citation most directly '
                'and specifically establishes it. A later generic restatement or a '
                'bare yes/no may refer to the fact without stating it. Prefer the '
                'specific statement naming the finding, treatment or action. A question '
                'with its confirmed premise can supply the specific evidence; the bare '
                'reply alone often cannot. A proposal is not a completed prescription. '
                'Do not favor A or B based on ordering. '
                'For each question, first compare what A and B actually establish '
                'in at most 30 words, then choose. Return JSON mapping the question '
                'number to {"comparison":"brief comparison", "choice":"A" or "B"}. '
                'Do not output any other text.\n\n'
            )
        else:
            rules += 'Return JSON mapping question number to "A" or "B", nothing else.\n\n'
        instruction = rules + '\n\n'.join(pairs)
        prompt = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': instruction}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        reply = generate(self.model, self.tokenizer, prompt=prompt,
                         sampler=self.sampler,
                         max_tokens=1000 if (self.refine_evidence or self.reasoned_choice
                                             or self.review_transcript) else 300,
                         verbose=False)
        self.last_adjudication = reply
        try:
            match = re.search(r'\{.*\}', reply, re.DOTALL)
            choices = json.loads(match.group(0)) if match else {}
        except json.JSONDecodeError:
            choices = {}
        if not isinstance(choices, dict):
            choices = {}
        result = []
        for index, (answer, span) in enumerate(predictions):
            original = span
            selection = choices.get(str(index + 1))
            if self.review_transcript and isinstance(selection, dict):
                if selection.get('choice') == 'B':
                    span = prior[index][1]
                elif (selection.get('choice') == 'C' and type(selection.get('unit')) is int
                      and isinstance(selection.get('quote'), str)):
                    alternative = quote_span(units, selection['unit'], selection['quote'])
                    if alternative is not None:
                        span = calibrate(alternative, *self.calibration)
            elif self.refine_evidence and isinstance(selection, dict):
                candidate = prior[index][1] if selection.get('choice') == 'B' else span
                quote = selection.get('quote')
                if candidate is not None and isinstance(quote, str):
                    words = [word for word in transcript.words
                             if candidate[0] - 5 <= (word.start + word.end) / 2
                             <= candidate[1] + 5]
                    grounded = exact_quote_spans(words, quote)
                    if len(grounded) == 1:
                        span = calibrate(grounded[0], *self.calibration)
            elif self.reasoned_choice and isinstance(selection, dict):
                if selection.get('choice') == 'B':
                    span = prior[index][1]
            elif not self.refine_evidence and selection == 'B':
                span = prior[index][1]
            if self.pair_selector is not None and answer:
                first_overlap = temporal_iou(span, original) if span is not None else 0.0
                second_overlap = temporal_iou(span, prior[index][1]) if span is not None else 0.0
                features = pair_features(transcript, questions[index], original,
                                         prior[index][1], second_overlap > first_overlap)
                span = (prior[index][1] if pair_advantage(features, self.pair_selector) > 0
                        else original)
            result.append((answer, span))
        return result
