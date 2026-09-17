# Experiment log — medical-appointment

Running log of every approach tried, what it scored, and why. Newest sections
are appended at the bottom; nothing is deleted, including the things that did
not work.

**Metric (from `local_evaluator.py`)**

```
score = 0.4 * accuracy + 0.6 * mean_tIoU
```

`mean_tIoU` is averaged over the 195 questions whose gold answer is *yes*
(those are the only ones with an annotated span). Accuracy is over all 390.

---

## 0. Reading the problem

**What arrives:** one MP3 (mono, 128 kbps, 44.1 kHz, ~1–3.5 min) plus ten
yes/no questions about it. No transcript.

**What goes back:** ten booleans, and for each a `(start, end)` in seconds.

### Facts measured from the supplied data (`tools/inspect_data.py`)

| Fact | Value |
| --- | --- |
| Conversations / questions | 39 / 390 (10 per conversation) |
| Question types | 195 `positive`, 142 `hard_negative`, 53 `off_topic` |
| Answer balance | 195 yes / 195 no — exactly balanced |
| Annotated spans | 195, one per positive |
| Span length | min 0.16 s, median **2.88 s**, mean 3.21 s, p90 5.50 s, max 14.2 s |
| Positives per conversation | 3 to 7 |

**The span-width ceiling.** If a window of fixed width were centred perfectly
on every gold span, its mean tIoU would be:

| Window width | 2 s | 3 s | 4 s | 5 s | 6 s | 8 s |
| --- | --- | --- | --- | --- | --- | --- |
| mean tIoU | 0.630 | **0.669** | 0.628 | 0.561 | 0.496 | 0.392 |

So **width matters as much as position**. Returning a whole Whisper segment
(often 6–12 s) throws away half the evidence score even when the retrieval is
right. Target ~3 s, tightened to the words that actually matched.

### Two consequences that shape everything below

1. **Evidence is worth more than answers.** 0.6 of the score is tIoU, 0.4 is
   accuracy. Perfect accuracy with no spans scores 0.400; perfect spans with
   coin-flip answers scores 0.800.

2. **Spans can be returned for questions answered *no*.** `Statistics.record`
   in `local_evaluator.py` computes `temporal_iou(gold, predicted)` using the
   *gold* label to decide whether the question counts, not our answer:

   ```python
   if label != YES or gold is None:
       return 0.0
   iou = temporal_iou(gold, predicted)
   ```

   `validate_response` only requires that the two timestamps are both set or
   both `None`, and the DTO validator only checks list lengths. So a span
   volunteered alongside a *no* still earns tIoU if the gold answer was *yes* —
   it insures against our own false negatives. The README suggests `None` for a
   *no*, so this is a calculated bet on the ported scorer; it is behind the
   `spans_for_no` flag and measured both ways below (§ "Ablations").

### Where the difficulty actually is

- `off_topic` (53): the subject never appears — cheap to detect by retrieval
  score alone.
- `hard_negative` (142): the *same* subject with one detail swapped (dose,
  duration, side, drug, polarity). These retrieve **well**, so retrieval score
  cannot separate them. They need the passage to be read, not just found.
- `positive` (195): must be answered yes *and* localised.

A system that answers from topical overlap gets ~all 142 hard negatives wrong,
which caps accuracy at (195 + 53)/390 = 0.636.

---

## Setup

- Python 3.12 (conda env `mtp`), RTX 5070 (sm_120), torch 2.11+cu128,
  faster-whisper 1.2.1 / CTranslate2 4.8.2 on CUDA.
- CTranslate2 does not ship cuBLAS/cuDNN; `solution/asr.py` adds torch's
  `lib/` to the DLL search path, otherwise it dies with
  `Library cublas64_12.dll is not found`.
- `tools/transcribe_all.py` caches word-level transcripts under
  `transcripts/<model>/`, so an experiment costs seconds instead of minutes.
- `tools/offline_eval.py` scores an approach against the cached transcripts
  using the evaluator's own `temporal_iou` / `gold_evidence` helpers.
- `local_evaluator.py` over the real server is used to confirm the end number.

---

## Results so far

| # | Approach | Accuracy | mean tIoU | **Score** |
| --- | --- | --- | --- | --- |
| 0 | Supplied baseline (always yes, no spans) | 0.500 | 0.000 | 0.200 |
| 1 | ASR + IDF lexical retrieval | 0.782 | 0.435 | 0.574 |
| 2 | + number/polarity rules | 0.782 | 0.435 | 0.574 |
| 3 | + logistic regression on 17 features (out-of-fold) | 0.779 | 0.456 | 0.586 |
| 4 | + dense retrieval and NLI entailment | 0.969 | 0.456 | **0.661** |

---

## Experiment 0 — the supplied baseline

`always_yes`, reproduced offline to check the harness agrees with
`local_evaluator.py`.

```
accuracy 0.500   positive 1.000  hard_negative 0.000  off_topic 0.000
mean tIoU 0.000  (195 spans missing)
SCORE 0.200
```

Matches the README's stated floor exactly, so the offline harness and the
evaluator agree.

## ASR — what everything else is built on

`faster-whisper large-v3`, CUDA, `word_timestamps=True`, `vad_filter=True`,
`condition_on_previous_text=False`, and a medical `initial_prompt` to bias
doses and drug names.

```
39 conversations, 4767 s of audio, transcribed in 304 s  =  0.064 x real time
```

A two-minute conversation costs ~8 s of the 60 s per-conversation budget, which
leaves the whole rest of the budget for reading it. Quality is high enough that
numbers survive: `100 mg daily for 2 weeks`, `four daily doses of one million
IU`. Drug names are the weak point — `Airomir` comes out as `Aromir`, `Pamol`
as `PAMEL` — which is why the lexical layer matches fuzzily rather than exactly.

`condition_on_previous_text=False` is deliberate: with it on, Whisper loops on
these conversational recordings and the timings drift.

## Experiment 1 — IDF-weighted lexical retrieval

The transcript is split into utterances (sentence ends, plus any pause longer
than 0.55 s), candidate windows are every run of up to *k* consecutive
utterances, and each is scored by how much of the question's content-word IDF
mass it covers. Coverage above a threshold means *yes*; the span is the winning
window, tightened to the words that actually matched.

Answer:

```
accuracy 0.738   positive 0.590  hard_negative 0.845  off_topic 1.000
mean tIoU 0.411
SCORE 0.542
```

Then swept (`tools/sweep.py`):

| Sweep | Best |
| --- | --- |
| threshold 0.20 → 0.60 | **0.45** (0.560); 0.55 was answering *no* far too often |
| `max_units` 1–4, length penalty 0–0.06 | `max_units=3`, penalty **0** (0.574) |
| span shaping | `min_span=2.0`, `max_span≈6`, `pad≈0.3` (tIoU 0.435) |

Two things learned:

- **The length penalty was a mistake.** Penalising long windows during
  *ranking* picks a worse passage; the width is better fixed afterwards, by
  tightening, than paid for during retrieval.
- **Do not let spans get too short.** `min_span=2.0` beat 0.8 by +0.03 tIoU.
  The gold median is 2.9 s, and when the centre is only approximately right a
  wider guess overlaps more than a narrow one. Precision that is not accurate
  is not worth having.

The shape of the errors is exactly what the README warns about: `off_topic`
1.000, `hard_negative` 0.69–0.87 only by answering *no* so often that
`positive` collapses. Retrieval score alone cannot tell a hard negative from a
positive, because a hard negative retrieves just as well.

## Experiment 2 — rules for the hard negatives

Two hand-written checks on the retrieved passage:

- **Quantity conflict** — parse every `(value, unit)` in the question and in
  the passage, canonicalising spellings (`milligrams`/`mg`, `micrograms`/`mcg`)
  and spelled-out numbers (`one million IU` → `(1000000, 'iu')`). If the
  question states a value the passage contradicts in the same unit, answer no.
- **Polarity conflict** — a list of flip pairs (`increase`/`reduce`,
  `normal`/`abnormal`, `continue`/`stop`, `left`/`right`, `before`/`after`,
  `empty`/`full`, …). If the question uses one side and the passage the other,
  answer no.

```
hard_negative 0.845 -> 0.859 at threshold 0.55
overall SCORE 0.542 -> 0.543
```

**Barely moved.** The rules fire on the handful of hard negatives that swap a
number or an obvious antonym, but most swap something else — a drug for
another drug, a body part, a fact that was simply never agreed. Hand-written
rules do not scale to that, and each new rule risks a false *no* on a positive.
Kept anyway: they are cheap, and they survive as features in experiment 3.

## Experiment 3 — a classifier over lexical features

17 features per question: coverage of the best window, of the second, the gap
between them, coverage over the whole transcript, out-of-vocabulary fraction,
the IDF mass of the question terms that were *not* found, whether the question
states a quantity, whether that quantity matched or conflicted, the polarity
flag, window duration and position, question length.

Scored **out-of-fold** with `GroupKFold` by conversation (5 folds), so no
question is judged by a model that has seen another question about the same
audio.

| Model | Accuracy | mean tIoU | Score |
| --- | --- | --- | --- |
| Logistic regression, C=1 | 0.779 | 0.456 | **0.586** |
| Logistic regression, C=0.2 | 0.777 | 0.456 | 0.585 |
| Hist gradient boosting | 0.741 | 0.456 | 0.570 |

Weights (standardised) say what the model actually uses:

```
coverage_mean_top3   +1.096      antonym_conflict   -0.469
quantity_match       +0.618      quantity_conflict  -0.337
question_length      -0.546      oov_fraction       -0.275
```

Notes:

- The classifier is worth ~+0.01 over the tuned threshold on accuracy, and no
  more. With 142 hard negatives that retrieve like positives, there is no
  decision boundary in these features that separates them; the model mostly
  learns a better-calibrated version of "is this covered".
- Gradient boosting is *worse* than logistic regression, which is what 390
  samples over 39 groups should do to a tree ensemble.
- `question_length` having a real weight is a warning: it is a dataset artefact
  (off-topic questions are phrased longer), not a property of the audio. A
  model leaning on it would not transfer.
- The tIoU rose to 0.456 purely from the swept span-shaping parameters, not
  from anything the classifier did.

**Conclusion: the answer half cannot be solved without reading the passage.**

## Experiment 4 — dense retrieval and an NLI reader

Two neural pieces, both local:

- `BAAI/bge-base-en-v1.5` embeds every candidate window and the question
  (rewritten as a declarative statement first, since roughly half the questions
  are tag questions).
- `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` scores
  entailment between the passage (premise) and that statement (hypothesis).

The answer is *yes* when the best entailment probability over the shortlist
clears a threshold.

| Variant | Accuracy | mean tIoU | Score |
| --- | --- | --- | --- |
| Dense retrieval only, no NLI | 0.772 | 0.431 | 0.567 |
| + NLI, span from the entailed window | 0.962 | 0.332 | 0.584 |
| + NLI, span from the *lexically* best window | **0.969** | 0.456 | **0.661** |

`hard_negative` accuracy goes **0.711 → 0.979**. That is the whole story of
this experiment: entailment reads the passage, so "200 mg daily" against a
transcript that says "100 mg" comes out as *not entailed*.

Three things worth recording:

- **Reranking by entailment wrecks the evidence half** (tIoU 0.456 → 0.332).
  The NLI model prefers a wider premise with more context around the fact,
  which is exactly the wrong thing to hand back as an annotated span. So the
  two halves were decoupled: *answer* from entailment over a hybrid shortlist,
  *span* from lexical ranking alone.
- **Dense retrieval did not improve localisation.** Pure lexical ranking
  (alpha 0) gives tIoU 0.456; alpha 0.3–0.7 gives 0.431–0.434. It does help the
  *answer* slightly (0.938 → 0.969) by putting a better passage in front of the
  NLI model, so it is kept in the shortlist only.
- **Context helps the reader, not the span.** `context_units=2` (one utterance
  either side of the window in the premise) beat 0 by +0.06 accuracy: a fact is
  often stated across a doctor/patient turn pair.

### Where the remaining points are

With accuracy at 0.969, accuracy is worth at most another +0.012 of score. The
evidence half is worth +0.16:

```
mean tIoU                 0.456
  tIoU = 0 (missed)       45 / 195  (23.1%)
  tIoU of the rest        0.593
  fraction >= 0.5         0.497
predicted centre inside the gold span: 132/195 (67.7%)
of the 150 overlapping spans: 42 too wide, 46 too narrow, 62 offset
oracle over the candidate windows: 0.717
```

So candidate generation is fine (0.717 available) and *selection* is the
problem. Reading the misses, they are almost all the same failure: the subject
is mentioned **more than once**, and lexical scoring picks the wrong mention.

```
Q     Is Pamol one of the medicines requested?
gold  182.44-184.98  I am creating prescriptions for both PAMOL and ibumetin now
pred   41.22- 43.22  PAMOL and IBUMETIN.
```

Both mentions match the question word for word. Only something that understands
*which* mention establishes the claim can choose between them — which is
experiment 5.

