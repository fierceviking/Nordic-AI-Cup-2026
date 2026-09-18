"""Learn which candidate the annotator would have marked.

    python tools/train_span_model.py --model parakeet

Eleven ranking methods have lost to word overlap. All of them *reasoned* about
which passage establishes the claim; none of them learned the convention the
annotator actually used, which experiment 17 showed is not recoverable from the
question by reasoning (the gold span holds 45% of the question's content words
and ties or loses to some other utterance 47% of the time).

So this trains rather than reasons: a small cross-encoder regresses the tIoU a
candidate would *actually earn* under the shipped emission rule. Scored strictly
out-of-fold with GroupKFold by conversation -- questions from one consultation
share a transcript, a topic and a speaker pair, so an ordinary split would leak.

The backbone is deliberately small (22M) and the run is short, because at 39
independent conversations a larger model memorises drug names rather than
learning the convention (Karan et al., SIGDIAL 2021).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from solution import asr  # noqa: E402
from solution.approaches.lexical import (LexicalApproach,  # noqa: E402
                                         _inverse_document_frequency)
from solution.approaches.neural import best_device  # noqa: E402
from solution.textutil import content_stems, declarative  # noqa: E402
from solution.windows import (build_windows, calibrate,  # noqa: E402
                              matched_run_span, split_units)
from utils import (gold_evidence, group_questions_by_conversation,  # noqa: E402
                   temporal_iou)

BACKBONE = 'cross-encoder/ms-marco-MiniLM-L6-v2'
CALIB = (0.92, -0.12, -0.50)
TOP_K = 20


def build_dataset(model_name: str):
    """(question, candidate text, coverage, tIoU, conversation, question id)."""
    lexical = LexicalApproach(max_units=3, length_penalty=0.0)
    rows_out = []
    question_id = 0

    for audio_filename, rows in group_questions_by_conversation():
        transcript = asr.load_cached(Path(audio_filename).stem, model_name)
        if transcript is None:
            raise SystemExit(f'transcribe first: {audio_filename}')

        units = split_units(transcript)
        windows = build_windows(units, max_units=3)
        stems = [set(content_stems(w.text)) for w in windows]
        idf = _inverse_document_frequency(units)

        for row in rows:
            gold = gold_evidence(row)
            if not gold:
                continue

            matches = lexical.score_windows(
                row['question'], windows, stems, idf
            )
            if not matches:
                continue
            matches = sorted(matches, key=lambda m: m.score, reverse=True)[:TOP_K]
            statement = declarative(row['question'])

            for match in matches:
                span = calibrate(
                    matched_run_span(match.window, match.matched, units, 3),
                    *CALIB,
                )
                rows_out.append({
                    'question': statement,
                    'text': match.window.text,
                    'coverage': match.coverage,
                    'target': temporal_iou(gold, span),
                    'group': Path(audio_filename).stem,
                    'qid': question_id,
                })
            question_id += 1

    return rows_out


def selected(scores, data) -> float:
    scores = np.asarray(scores)
    qids = np.array([row['qid'] for row in data])
    targets = np.array([row['target'] for row in data])

    total = [
        targets[qids == qid][int(np.argmax(scores[qids == qid]))]
        for qid in np.unique(qids)
    ]
    return float(np.mean(total))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=asr.DEFAULT_MODEL)
    parser.add_argument('--backbone', default=BACKBONE)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--objective', choices=('mse', 'listwise'),
                        default='listwise')
    parser.add_argument('--pair-selector', action='store_true')
    args = parser.parse_args()

    if args.pair_selector:
        return train_pair_selector(args.model)

    import torch
    from sklearn.model_selection import GroupKFold
    from torch.utils.data import DataLoader
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer)

    device = best_device()
    print(f'building candidates from {args.model} transcripts ...')
    data = build_dataset(args.model)
    groups = np.array([row['group'] for row in data])
    qids = np.array([row['qid'] for row in data])
    targets = np.array([row['target'] for row in data], dtype=np.float32)
    coverage = np.array([row['coverage'] for row in data], dtype=np.float32)

    print(f'{len(data)} candidates, {len(np.unique([r["qid"] for r in data]))} '
          f'questions, {len(np.unique(groups))} conversations, device {device}\n')

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    predictions = np.zeros(len(data), dtype=np.float32)

    splitter = GroupKFold(n_splits=args.folds)
    for fold, (train_index, test_index) in enumerate(
        splitter.split(np.arange(len(data)), groups=groups), 1
    ):
        model = AutoModelForSequenceClassification.from_pretrained(
            args.backbone, num_labels=1, ignore_mismatched_sizes=True,
        ).to(device).train()
        optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)

        def collate(batch):
            encoded = tokenizer(
                [data[i]['question'] for i in batch],
                [data[i]['text'] for i in batch],
                truncation=True, max_length=192, padding=True,
                return_tensors='pt',
            )
            return encoded, torch.tensor(targets[batch])

        if args.objective == 'listwise':
            # One batch per question: the task is argmax over that question's
            # candidates, so the loss should range over the same set.
            train_qids = np.unique(qids[train_index])
            batches = [
                np.intersect1d(train_index, np.flatnonzero(qids == qid))
                for qid in train_qids
            ]
            loader = [
                (collate(batch)[0], torch.tensor(targets[batch]))
                for batch in batches if len(batch) > 1
            ]
        else:
            loader = DataLoader(
                train_index, batch_size=args.batch_size, shuffle=True,
                collate_fn=collate,
            )

        for epoch in range(args.epochs):
            running = 0.0
            order = (np.random.permutation(len(loader))
                     if args.objective == 'listwise' else None)
            sequence = ([loader[i] for i in order]
                        if order is not None else loader)

            for encoded, batch_targets in sequence:
                encoded = {k: v.to(device) for k, v in encoded.items()}
                logits = model(**encoded).logits.squeeze(-1)
                batch_targets = batch_targets.to(device)

                if args.objective == 'listwise':
                    # Soft targets: tIoU normalised over the question's
                    # candidates, so near-ties are not forced apart.
                    weights = batch_targets / (batch_targets.sum() + 1e-6)
                    loss = -(weights * torch.log_softmax(logits, dim=0)).sum()
                else:
                    loss = torch.nn.functional.mse_loss(
                        torch.sigmoid(logits), batch_targets
                    )

                loss.backward()
                optimiser.step()
                optimiser.zero_grad()
                running += float(loss.detach())
            print(f'  fold {fold} epoch {epoch + 1}  '
                  f'loss {running / max(len(loader), 1):.4f}')

        model.eval()
        with torch.no_grad():
            for begin in range(0, len(test_index), 64):
                batch = test_index[begin:begin + 64]
                encoded = tokenizer(
                    [data[i]['question'] for i in batch],
                    [data[i]['text'] for i in batch],
                    truncation=True, max_length=192, padding=True,
                    return_tensors='pt',
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                logits = model(**encoded).logits.squeeze(-1)
                predictions[batch] = torch.sigmoid(logits).float().cpu().numpy()

    print(f'\n{"ranking rule":<44}{"mean tIoU":>10}')
    print('-' * 56)
    print(f'{"argmax coverage (shipped)":<44}{selected(coverage, data):>10.4f}')
    print(f'{"argmax trained cross-encoder (out-of-fold)":<44}'
          f'{selected(predictions, data):>10.4f}')

    for weight in (0.2, 0.3, 0.5, 0.7):
        blended = weight * predictions + (1 - weight) * coverage
        print(f'{f"  blend {weight:.1f} model + coverage":<44}'
              f'{selected(blended, data):>10.4f}')

    print('-' * 56)
    qids = np.array([row['qid'] for row in data])
    print(f'{"oracle over these candidates":<44}'
          f'{float(np.mean([targets[qids == q].max() for q in np.unique(qids)])):>10.4f}')
    return 0


def train_pair_selector(model_name: str) -> int:
    import csv
    import json
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from solution.approaches.llm import pair_features

    def load_run(name):
        with (ROOT / 'runs' / name).open() as handle:
            return {row['question_id']: row for row in csv.DictReader(handle)}

    first = load_run('quote30b_full.csv')
    second = load_run('whisper_veto.csv')
    judge = load_run('quote_judge_full.csv')
    features, targets, groups, ious, chosen = [], [], [], [], []
    for key, row in judge.items():
        if row['label'] != '1':
            continue
        transcript = asr.load_cached('conversation_' + row['transcript_id'], model_name)
        def interval(record):
            return ((float(record['pred_start']), float(record['pred_end']))
                    if record['pred_start'] else None)
        left = interval(first[key])
        left = calibrate(left, 0.98, 0.22, 0.02) if left else None
        right = interval(second[key])
        selected_span = interval(row)
        judge_second = temporal_iou(selected_span, right) > temporal_iou(selected_span, left)
        gold = (float(row['gold_start']), float(row['gold_end']))
        left_iou, right_iou = temporal_iou(gold, left), temporal_iou(gold, right)
        features.append(pair_features(transcript, row['question'], left, right, judge_second))
        targets.append(right_iou - left_iou)
        groups.append(row['transcript_id'])
        ious.append([left_iou, right_iou])
        chosen.append(temporal_iou(gold, selected_span))

    features, targets = np.asarray(features), np.asarray(targets)
    groups, ious = np.asarray(groups), np.asarray(ious)
    predictions = np.zeros(len(targets))
    for train, held in GroupKFold(5).split(features, groups=groups):
        scaler = StandardScaler().fit(features[train])
        model = Ridge(alpha=10.0).fit(scaler.transform(features[train]), targets[train])
        predictions[held] = model.predict(scaler.transform(features[held]))
    scored = ious[np.arange(len(ious)), (predictions > 0).astype(int)]
    utility_predictions = np.zeros(len(targets), dtype=int)
    for train, held in GroupKFold(5).split(features, groups=groups):
        informative = train[np.abs(targets[train]) > 1e-6]
        scaler = StandardScaler().fit(features[informative])
        model = LogisticRegression(C=0.1, max_iter=1000).fit(
            scaler.transform(features[informative]), targets[informative] > 0,
            sample_weight=np.abs(targets[informative]),
        )
        utility_predictions[held] = model.predict(scaler.transform(features[held]))
    utility_scored = ious[np.arange(len(ious)), utility_predictions]
    print('Paired candidates:', len(targets), 'across', len(np.unique(groups)), 'conversations')
    print('Validated judge tIoU:', np.mean(chosen))
    print('Grouped OOF selector tIoU:', np.mean(scored))
    print('Grouped OOF utility-weighted selector tIoU:', np.mean(utility_scored))
    print('Oracle tIoU:', np.mean(ious.max(axis=1)))
    scaler = StandardScaler().fit(features)
    model = Ridge(alpha=10.0).fit(scaler.transform(features), targets)
    payload = {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
               'coef': model.coef_.tolist(), 'intercept': float(model.intercept_)}
    output = ROOT / 'models' / 'pair_selector.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    np.savez(ROOT / 'runs' / 'pair_selector_oof.npz', predictions=predictions,
             tiou=scored, utility_tiou=utility_scored,
             features=features, targets=targets, ious=ious,
             baseline=np.asarray(chosen), groups=groups)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
