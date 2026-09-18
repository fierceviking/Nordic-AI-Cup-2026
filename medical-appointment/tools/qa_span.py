"""Cut the span with an extractive QA model instead of a lexical rule.

    python tools/qa_span.py --model parakeet

Emitting the whole utterance caps out near 0.80; word-level rules score 0.34
because word-by-word lexical matching carries no signal. Extractive QA is the
one grounding architecture the literature says transfers here (VSLNet): predict
a start and an end over the token sequence, then read the word timestamps under
it. The model is used zero-shot -- no fine-tuning, per RESEARCH.md section 1.8.
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
from solution.approaches.neural import best_device  # noqa: E402
from solution.textutil import content_stems  # noqa: E402
from solution.windows import (build_windows, calibrate,  # noqa: E402
                              matched_unit_span, split_units)
from utils import (gold_evidence, group_questions_by_conversation,  # noqa: E402
                   temporal_iou)

QA_MODEL = 'deepset/roberta-base-squad2'


def word_offsets(words: Sequence) -> Tuple[str, List[Tuple[int, int]]]:
    """Window text plus the character range each word occupies in it.

    ``word.word`` keeps its leading space, so concatenating reproduces the
    spoken text exactly and the offsets stay aligned with the timings.
    """
    parts, offsets, cursor = [], [], 0
    for word in words:
        token = word.word
        parts.append(token)
        offsets.append((cursor, cursor + len(token)))
        cursor += len(token)
    return ''.join(parts), offsets


def span_from_chars(words, offsets, start: int, end: int):
    covered = [
        index for index, (begin, finish) in enumerate(offsets)
        if finish > start and begin < end
    ]
    if not covered:
        return None
    return float(words[covered[0]].start), float(words[covered[-1]].end)


def read_spans(name, device, payloads, max_answer_len, batch_size=16):
    """Best (start, end) character range per question.

    transformers 5 dropped the question-answering pipeline, so the start/end
    pointer is decoded here: mask everything that is not context, then take the
    best-scoring ordered pair within the length budget.
    """
    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForQuestionAnswering.from_pretrained(name)
    model = model.to(device).eval()

    out = []
    for begin in range(0, len(payloads), batch_size):
        batch = payloads[begin:begin + batch_size]
        encoded = tokenizer(
            [item['question'] for item in batch],
            [item['context'] for item in batch],
            return_offsets_mapping=True, truncation='only_second',
            max_length=384, padding=True, return_tensors='pt',
        )
        offsets = encoded.pop('offset_mapping').numpy()
        context_mask = np.array([
            [sid == 1 for sid in encoded.sequence_ids(i)]
            for i in range(len(batch))
        ])

        with torch.no_grad():
            logits = model(**{
                key: value.to(device) for key, value in encoded.items()
            })

        starts = logits.start_logits.float().cpu().numpy()
        ends = logits.end_logits.float().cpu().numpy()

        for i in range(len(batch)):
            start_scores = np.where(context_mask[i], starts[i], -1e9)
            end_scores = np.where(context_mask[i], ends[i], -1e9)

            best, best_score = None, -1e18
            for s in np.argsort(start_scores)[-20:]:
                for e in np.argsort(end_scores)[-20:]:
                    if e < s or e - s + 1 > max_answer_len:
                        continue
                    score = start_scores[s] + end_scores[e]
                    if score > best_score:
                        best, best_score = (int(s), int(e)), score

            if best is None:
                out.append(None)
            else:
                out.append((int(offsets[i][best[0]][0]),
                            int(offsets[i][best[1]][1])))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=asr.DEFAULT_MODEL)
    parser.add_argument('--qa-model', default=QA_MODEL)
    parser.add_argument('--context-units', type=int, default=1)
    parser.add_argument('--max-answer-len', type=int, default=30)
    args = parser.parse_args()

    device = best_device()

    lexical = LexicalApproach(max_units=3, length_penalty=0.0)
    payloads, golds, shipped, oracles = [], [], [], []

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

            # The reader sees a little more than it may point at: a fact is
            # often stated across a doctor/patient turn pair.
            first = max(0, match.window.first_unit - args.context_units)
            last = min(len(units) - 1, match.window.last_unit + args.context_units)
            words = [word for unit in units[first:last + 1] for word in unit.words]
            if not words:
                continue

            text, offsets = word_offsets(words)
            payloads.append({'question': row['question'], 'context': text})
            golds.append(gold)
            shipped.append(calibrate(
                matched_unit_span(match.window, match.matched, units),
                0.92, -0.10, -0.50,
            ))
            oracles.append((words, offsets))

    print(f'{len(payloads)} questions through {args.qa_model} on {device}\n')
    answers = read_spans(
        args.qa_model, device, payloads, args.max_answer_len
    )

    raw, calibrated = [], []
    for answer, gold, (words, offsets) in zip(answers, golds, oracles):
        span = (
            span_from_chars(words, offsets, answer[0], answer[1])
            if answer is not None else None
        )
        if span is None:
            raw.append(0.0)
            calibrated.append(0.0)
            continue
        raw.append(temporal_iou(gold, span))
        calibrated.append(temporal_iou(gold, calibrate(span, 1.0, -0.05, 0.10)))

    print(f'{"span source":<34}{"mean tIoU":>10}')
    print('-' * 46)
    print(f'{"shipped (matched unit + calib)":<34}'
          f'{np.mean([temporal_iou(g, s) for g, s in zip(golds, shipped)]):>10.4f}')
    print(f'{"extractive QA (raw)":<34}{np.mean(raw):>10.4f}')
    print(f'{"extractive QA (nudged)":<34}{np.mean(calibrated):>10.4f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
