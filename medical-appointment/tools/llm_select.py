"""Let a local LLM choose the evidence utterances, and see if reading beats matching.

    python tools/llm_select.py --model parakeet --limit 10

Ten attacks on span selection have produced one small win, and the nine failures
were all shallow scorers -- lexical overlap, cosine, a cross-encoder, an NLI
head. None of them read the conversation. Meanwhile the pipeline uses about 4
seconds of its 60 second budget.

The model is asked for utterance *indices*, never for seconds: LLMs hallucinate
timestamps rather than copying them, and the timings are already known exactly
once an utterance is named. All ten questions go in one call, because ten calls
would re-prefill the transcript ten times.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from solution import asr  # noqa: E402
from solution.approaches.lexical import (LexicalApproach,  # noqa: E402
                                         _inverse_document_frequency)
from solution.textutil import content_stems  # noqa: E402
from solution.windows import (build_windows, calibrate,  # noqa: E402
                              matched_run_span, split_units)
from utils import (gold_evidence, group_questions_by_conversation,  # noqa: E402
                   temporal_iou)

LLM_MODEL = 'mlx-community/Qwen2.5-7B-Instruct-4bit'
CALIB = (0.92, -0.12, -0.50)
MAX_RUN = 3

SYSTEM = (
    'You locate evidence in transcripts of doctor-patient consultations. '
    'You answer only with JSON.'
)

INSTRUCTION = """\
Below is a consultation transcript, one utterance per line, numbered.

{transcript}

For each question, name the utterance numbers that state the fact the question \
is about. Rules:
- Give a start and an end number, inclusive. Use the same number twice if one \
utterance is enough. Never span more than 3 utterances.
- Point at the utterance that ESTABLISHES the fact, not at a question asking \
about it. If someone asks "Any allergies?" and the reply is "No, none", the \
evidence is the reply.
- If the same subject is mentioned more than once, choose the mention that \
settles it -- the decision, the prescription, the result -- not the first \
time it came up.
- If the fact never appears, use null.

Questions:
{questions}

Reply with only a JSON object mapping each question number to [start, end] or \
null, like {{"1": [12, 13], "2": null}}."""

SHORTLIST_INSTRUCTION = """\
Below is a consultation transcript, one utterance per line, numbered.

{transcript}

Each question below comes with a shortlist of utterance numbers found by word \
matching. Exactly one of them is usually the real evidence; word matching often \
picks the wrong one. Choose the best for each question. Rules:
- Prefer the utterance that ESTABLISHES the fact over one that merely mentions \
the words. If a shortlisted utterance is a question being asked, the evidence is \
usually the reply that follows it, so prefer a later number.
- If the subject comes up more than once, choose the mention that settles it -- \
the decision, the prescription, the result.
- You may give a start and an end number to cover up to 3 consecutive \
utterances, starting from one of the shortlisted numbers.

Questions:
{questions}

Reply with only a JSON object mapping each question number to [start, end], \
like {{"1": [12, 13], "2": [31, 31]}}."""


def build_prompt(units, questions: List[str]) -> str:
    transcript = '\n'.join(
        f'[{index}] {unit.text}' for index, unit in enumerate(units)
    )
    listing = '\n'.join(
        f'{number}. {question}' for number, question in enumerate(questions, 1)
    )
    return INSTRUCTION.format(transcript=transcript, questions=listing)


def build_shortlist_prompt(units, questions: List[str], shortlists) -> str:
    transcript = '\n'.join(
        f'[{index}] {unit.text}' for index, unit in enumerate(units)
    )
    listing = '\n'.join(
        f'{number}. {question}\n   shortlist: '
        f'{", ".join(str(index) for index in candidates)}'
        for number, (question, candidates) in enumerate(
            zip(questions, shortlists), 1
        )
    )
    return SHORTLIST_INSTRUCTION.format(
        transcript=transcript, questions=listing
    )


def parse(reply: str, count: int, limit: int) -> Dict[int, Optional[Tuple[int, int]]]:
    out: Dict[int, Optional[Tuple[int, int]]] = {i: None for i in range(count)}

    match = re.search(r'\{.*\}', reply, re.S)
    if not match:
        return out
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return out

    for key, value in payload.items():
        try:
            index = int(str(key).strip()) - 1
        except ValueError:
            continue
        if not 0 <= index < count or value is None:
            continue

        if isinstance(value, (int, float)):
            value = [value, value]
        if not isinstance(value, list) or len(value) != 2:
            continue
        try:
            start, end = int(value[0]), int(value[1])
        except (TypeError, ValueError):
            continue

        start = max(0, min(start, limit - 1))
        end = max(start, min(end, limit - 1, start + MAX_RUN - 1))
        out[index] = (start, end)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=asr.DEFAULT_MODEL)
    parser.add_argument('--llm', default=LLM_MODEL)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--mode', choices=('free', 'shortlist', 'fewshot-quotes'),
                        default='shortlist')
    parser.add_argument('--shortlist', type=int, default=6)
    parser.add_argument('--max-tokens', type=int, default=400)
    args = parser.parse_args()

    if args.mode == 'fewshot-quotes':
        return run_fewshot(args)

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    print(f'loading {args.llm} ...')
    model, tokenizer = load(args.llm)
    sampler = make_sampler(temp=0.0)

    lexical = LexicalApproach(max_units=3, length_penalty=0.0)
    conversations = group_questions_by_conversation()
    if args.limit:
        conversations = conversations[: args.limit]

    llm_iou, base_iou, unit_oracle = [], [], []
    parsed_ok, total_questions, elapsed = 0, 0, 0.0

    for number, (audio_filename, rows) in enumerate(conversations, 1):
        transcript = asr.load_cached(Path(audio_filename).stem, args.model)
        if transcript is None:
            raise SystemExit(f'transcribe first: {audio_filename}')

        units = split_units(transcript)
        windows = build_windows(units, max_units=3)
        stems = [set(content_stems(w.text)) for w in windows]
        idf = _inverse_document_frequency(units)

        questions = [row['question'] for row in rows]

        # Candidate utterance indices per question, in lexical order, deduped.
        shortlists = []
        per_question_best = []
        for row in rows:
            matches = lexical.score_windows(row['question'], windows, stems, idf)
            matches = sorted(matches, key=lambda m: m.score, reverse=True)
            per_question_best.append(matches[0] if matches else None)

            seen, candidates = set(), []
            for match in matches:
                span = matched_run_span(
                    match.window, match.matched, units, MAX_RUN
                )
                index = min(
                    range(len(units)),
                    key=lambda i: abs(units[i].start - span[0]),
                )
                if index not in seen:
                    seen.add(index)
                    candidates.append(index)
                if len(candidates) >= args.shortlist:
                    break
            shortlists.append(candidates or [0])

        if args.mode == 'shortlist':
            user = build_shortlist_prompt(units, questions, shortlists)
        else:
            user = build_prompt(units, questions)

        prompt = tokenizer.apply_chat_template(
            [{'role': 'system', 'content': SYSTEM},
             {'role': 'user', 'content': user}],
            add_generation_prompt=True, tokenize=False,
        )

        started = time.time()
        reply = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=args.max_tokens, sampler=sampler, verbose=False,
        )
        elapsed += time.time() - started
        picks = parse(reply, len(rows), len(units))

        for index, row in enumerate(rows):
            gold = gold_evidence(row)
            if not gold:
                continue
            total_questions += 1

            best = per_question_best[index]
            base = (
                calibrate(matched_run_span(best.window, best.matched, units, 3), *CALIB)
                if best else None
            )
            base_iou.append(temporal_iou(gold, base) if base else 0.0)
            unit_oracle.append(max(
                temporal_iou(gold, (u.start, u.end)) for u in units
            ))

            pick = picks.get(index)
            if pick is None:
                llm_iou.append(base_iou[-1])  # fall back rather than score zero
                continue

            parsed_ok += 1
            span = calibrate(
                (units[pick[0]].start, units[pick[1]].end), *CALIB
            )
            llm_iou.append(temporal_iou(gold, span))

        print(f'[{number}/{len(conversations)}] {Path(audio_filename).stem}: '
              f'{len(units)} units, llm {np.mean(llm_iou):.3f} '
              f'vs base {np.mean(base_iou):.3f}')

    print(f'\n{total_questions} annotated questions over {len(conversations)} '
          f'conversations')
    print(f'utterances named by the model   {parsed_ok}/{total_questions}')
    print(f'llm time                        {elapsed / len(conversations):.1f} '
          f's/conversation\n')
    print(f'{"span source":<34}{"mean tIoU":>10}')
    print('-' * 46)
    print(f'{"shipped (lexical + matched run)":<34}{np.mean(base_iou):>10.4f}')
    print(f'{"LLM-chosen utterances":<34}{np.mean(llm_iou):>10.4f}')
    print(f'{"oracle over single utterances":<34}{np.mean(unit_oracle):>10.4f}')
    return 0


def run_fewshot(args) -> int:
    import csv
    from sklearn.model_selection import GroupKFold
    from solution.approaches.llm import QuoteApproach
    from tools.offline_eval import evaluate, summarise, report

    conversations = group_questions_by_conversation()
    examples = []
    for audio_filename, rows in conversations:
        transcript = asr.load_cached(Path(audio_filename).stem, args.model)
        if transcript is None:
            raise SystemExit(f'Missing transcript: {audio_filename}')
        units = split_units(transcript)
        for row in rows:
            gold = gold_evidence(row)
            if gold is None:
                continue
            chosen_words = [word for word in transcript.words
                            if gold[0] <= (word.start + word.end) / 2 <= gold[1]]
            if not chosen_words:
                continue
            first = next(index for index, unit in enumerate(units)
                         if any(word is chosen_words[0] for word in unit.words))
            last = next(index for index, unit in enumerate(units)
                        if any(word is chosen_words[-1] for word in unit.words))
            examples.append({
                'group': rows[0]['transcript_id'], 'question': row['question'],
                'context': '\n'.join(f'[{index}] {units[index].text}' for index in
                                     range(max(0, first - 2), min(len(units), last + 3))),
                'output': {'answer': True, 'unit': first,
                           'quote': ''.join(word.word for word in chosen_words).strip()},
            })

    approach = QuoteApproach(
        model_name=args.llm, max_tokens=1200,
        trace_path='runs/quote_fewshot_trace.jsonl',
        width_scale=0.98, start_offset=0.22, end_offset=0.02,
    )
    group_ids = np.array([rows[0]['transcript_id'] for _, rows in conversations])
    all_records, fold_times = [], []
    for fold, (train, held) in enumerate(GroupKFold(5).split(group_ids, groups=group_ids), 1):
        train_ids = set(group_ids[train])
        approach.examples = [example for example in examples if example['group'] in train_ids]
        assert not (set(group_ids[held]) & {example['group'] for example in approach.examples})
        summary = evaluate(approach, args.model, only=group_ids[held].tolist(),
                           dump=f'runs/quote_fewshot_fold{fold}.csv')
        all_records.extend(summary['records'])
        fold_times.extend([summary['latency_ms_mean']] * len(held))
        print(report(summary, f'Fold {fold}, no shared conversations'), flush=True)
        if args.limit and fold >= args.limit:
            break
    path = ROOT / 'runs' / 'quote_fewshot_oof.csv'
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_records[0]))
        writer.writeheader()
        writer.writerows(all_records)
    print(report(summarise(all_records, fold_times), 'Grouped few-shot result'))
    if not args.limit:
        output = ROOT / 'models' / 'quote_examples.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(examples, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
