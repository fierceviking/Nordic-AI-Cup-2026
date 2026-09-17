"""Score an approach against the supplied data without going near the server.

    python tools/offline_eval.py --approach lexical
    python tools/offline_eval.py --approach lexical --params "{\"threshold\":0.4}"
    python tools/offline_eval.py --approach lexical --dump runs/lexical.csv

Uses the cached transcripts, so an experiment costs seconds rather than
minutes. The headline numbers are computed with the evaluator's own helpers,
so a number here means what the same number means there — the only difference
is that transcription is not re-run and the network is not involved.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from solution import asr  # noqa: E402
from solution.approaches import build  # noqa: E402
from utils import gold_evidence, group_questions_by_conversation, temporal_iou  # noqa: E402

ACCURACY_WEIGHT = 0.4
TIOU_WEIGHT = 0.6

Span = Tuple[float, float]


def load_transcripts(model: str) -> Dict[str, asr.Transcript]:
    transcripts = {}
    for audio_filename, _ in group_questions_by_conversation():
        key = Path(audio_filename).stem
        transcript = asr.load_cached(key, model)
        if transcript is None:
            raise SystemExit(
                f'No cached transcript for {key} with model {model!r}. '
                f'Run tools/transcribe_all.py --model {model} first.'
            )
        transcripts[audio_filename] = transcript
    return transcripts


def evaluate(
    approach,
    model: str,
    verbose: bool = False,
    only: Optional[Sequence[str]] = None,
    dump: Optional[str] = None,
    limit: int = 0,
) -> dict:
    transcripts = load_transcripts(model)

    records = []
    latencies = []

    conversations = group_questions_by_conversation()
    if limit:
        conversations = conversations[:limit]

    for audio_filename, rows in conversations:
        transcript_id = rows[0]['transcript_id']
        if only and transcript_id not in only:
            continue

        questions = [row['question'] for row in rows]
        started = time.time()
        predictions = approach.predict(transcripts[audio_filename], questions)
        latencies.append((time.time() - started) * 1000)

        if len(predictions) != len(rows):
            raise RuntimeError(
                f'{approach.name} returned {len(predictions)} predictions for '
                f'{len(rows)} questions on {transcript_id}'
            )

        for row, (answer, span) in zip(rows, predictions):
            gold = gold_evidence(row)
            label = int(row['label'])
            iou = temporal_iou(gold, span) if (label == 1 and gold) else None
            records.append({
                'question_id': row['question_id'],
                'transcript_id': transcript_id,
                'question': row['question'],
                'question_type': row['question_type'],
                'label': label,
                'prediction': int(bool(answer)),
                'correct': int(int(bool(answer)) == label),
                'gold_start': gold[0] if gold else '',
                'gold_end': gold[1] if gold else '',
                'pred_start': span[0] if span else '',
                'pred_end': span[1] if span else '',
                'tiou': iou if iou is not None else '',
            })

    summary = summarise(records, latencies)

    if verbose:
        for record in records:
            mark = 'ok  ' if record['correct'] else 'WRONG'
            iou = f" tIoU {record['tiou']:.3f}" if record['tiou'] != '' else ''
            print(f'  {mark} {record["question_id"]:<26} '
                  f'{record["question_type"]:<14} '
                  f'said {"yes" if record["prediction"] else "no ":<3} '
                  f'wanted {"yes" if record["label"] else "no "}{iou}')

    if dump:
        path = Path(dump)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(f'wrote {path} ({len(records)} rows)')

    return summary


def summarise(records: List[dict], latencies: Sequence[float]) -> dict:
    total = len(records)
    correct = sum(record['correct'] for record in records)
    accuracy = correct / total if total else 0.0

    ious = [record['tiou'] for record in records if record['tiou'] != '']
    mean_tiou = sum(ious) / len(ious) if ious else 0.0

    by_type: Dict[str, List[int]] = collections.defaultdict(lambda: [0, 0])
    for record in records:
        bucket = by_type[record['question_type']]
        bucket[0] += record['correct']
        bucket[1] += 1

    yes_rate = sum(record['prediction'] for record in records) / total if total else 0.0

    return {
        'questions': total,
        'accuracy': accuracy,
        'mean_tiou': mean_tiou,
        'score': ACCURACY_WEIGHT * accuracy + TIOU_WEIGHT * mean_tiou,
        'by_type': {k: (v[0] / v[1], v[1]) for k, v in by_type.items()},
        'spans_missing': sum(
            1 for record in records
            if record['label'] == 1 and record['gold_start'] != ''
            and record['pred_start'] == ''
        ),
        'yes_rate': yes_rate,
        'latency_ms_mean': sum(latencies) / len(latencies) if latencies else 0.0,
        'records': records,
    }


def report(summary: dict, title: str = '') -> str:
    lines = []
    if title:
        lines.append(title)
    lines.append(f'  questions      {summary["questions"]}')
    lines.append(f'  accuracy       {summary["accuracy"]:.3f}')
    for question_type in ('positive', 'hard_negative', 'off_topic'):
        if question_type in summary['by_type']:
            value, count = summary['by_type'][question_type]
            lines.append(f'    {question_type:<14} {value:.3f}  (n={count})')
    lines.append(f'  mean tIoU      {summary["mean_tiou"]:.3f}')
    lines.append(f'  spans missing  {summary["spans_missing"]}')
    lines.append(f'  yes rate       {summary["yes_rate"]:.3f}')
    lines.append(f'  answer ms/conv {summary["latency_ms_mean"]:.0f}')
    lines.append(f'  SCORE          {summary["score"]:.3f}'
                 f'   (0.4*acc + 0.6*tIoU)')
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approach', default='lexical')
    parser.add_argument('--model', default=asr.DEFAULT_MODEL,
                        help='Which cached ASR transcripts to read.')
    parser.add_argument('--params', default='{}',
                        help='JSON keyword arguments for the approach.')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--dump', default=None,
                        help='Write per-question results to this CSV.')
    parser.add_argument('--limit', type=int, default=0,
                        help='Only the first N conversations. For smoke tests.')
    args = parser.parse_args()

    approach = build(args.approach, **json.loads(args.params))
    summary = evaluate(approach, args.model, verbose=args.verbose,
                       dump=args.dump, limit=args.limit)
    print(report(summary, f'\n{approach.name}  (asr={args.model}, '
                          f'params={args.params})'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
