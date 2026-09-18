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

---

# Part two — the move to Apple Silicon

Everything above ran on an RTX 5070. From here the target machine is an **M3 Pro
(36 GB, macOS 26.6)**, which changes the stack rather than the ideas. Literature
behind the decisions below is in [RESEARCH.md](RESEARCH.md).

## Setup

- `pyenv virtualenv 3.11.13 dm`; interpreter `~/.pyenv/versions/dm/bin/python`.
- torch 2.14.0 (MPS available and built), mlx 0.32.2, transformers 5.17.0,
  sentence-transformers 6.0.1, parakeet-mlx, mlx-whisper, `brew install ffmpeg`.
- Always `export PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false`;
  `TORCH_DEVICE` picks the torch device (`best_device()` is cuda > mps > cpu).

## Experiment 6 — porting the ASR off CTranslate2

**faster-whisper cannot be kept.** CTranslate2 has no Metal backend — its
backends are MKL, oneDNN, OpenBLAS, Ruy and Apple Accelerate on CPU plus
cuBLAS/ROCm on GPU, and `ValueError: unsupported device mps` has been open since
Nov 2023. On CPU, `large-v3` needs an estimated 66–145 s for a two-minute
conversation, against a 60 s budget. Every request would time out.

`solution/asr.py` was rewritten around a `BACKENDS` table so the backend is a
name, not a rewrite: `parakeet` (default), `parakeet-v3`, `mlx-large-v3-turbo`,
`mlx-large-v3`. The `Word` / `Segment` / `Transcript` dataclasses and the
`transcripts/<model>/` cache are unchanged, so every approach, tool and sweep
kept working untouched.

**Parakeet emits sub-word tokens**, not words — `' M'`, `'or'`, `'ning'`, `','`.
`_tokens_to_words` merges them: a token starting with a space opens a word,
everything else attaches to the current one. The leading space has to survive,
because `windows._unit` rebuilds utterance text by concatenating `word.word`.

```
39 conversations, 4769 s of audio, transcribed in 87 s  =  0.018 x real time
```

That is **3.5x faster than large-v3 was on the 5070** (0.064x), and the longest
conversation costs 5.3 s. Chosen for timestamp quality as much as speed:
parakeet-tdt is a transducer, so its timings are frame alignments and monotonic,
without Whisper's habit of placing a full stop seconds after the speech.

| Metric | CUDA large-v3 | parakeet (M3 Pro) |
| --- | --- | --- |
| accuracy | 0.969 | 0.964 |
| mean tIoU | 0.456 | 0.403 |
| **SCORE** | **0.661** | **0.627** |
| oracle over candidate windows | 0.717 | 0.697 |

Accuracy survives the ASR swap (−0.005). The tIoU drop is **not** an ASR
regression — it is that `span_pad` / `min_span` / `max_span` were tuned against
Whisper timings, and Parakeet's are tighter. Recovered below, with interest.

**Device.** DeBERTa-v3-large NLI was expected to be pathological on Metal
(disentangled attention leans on `torch.gather` and boolean masking, the weakest
MPS kernels). Measured, it is merely unexciting: **1919 ms/conversation on MPS
against 4547 ms on CPU, 2.4x, identical scores.** Kept on MPS. Whole pipeline is
~2.2 s ASR + ~1.9 s answering ≈ **4 s of the 60 s budget**, so there is room to
spend and no reason yet to replace the reader.

## Experiment 7 — three things that did not work

Recorded because each looked obviously right.

**7a. Cross-encoder reranking of the span.** `ms-marco-MiniLM-L6-v2` over the
top-20 lexical candidates, blended with coverage:

| `rerank_weight` | 0.0 | 0.3 | 0.5 | 0.7 | 1.0 |
| --- | --- | --- | --- | --- | --- |
| mean tIoU | **0.403** | 0.354 | 0.354 | 0.354 | 0.354 |

Worse at every non-zero weight, and flat across them — the cross-encoder simply
has a different argmax. It is trained on MS MARCO passage *relevance*, which
rewards a passage carrying enough context to look informative; the annotation is
the opposite, a bare phrase. Exactly the failure mode experiment 4 hit with NLI
reranking (tIoU 0.456 → 0.332), from a different model for the same reason.

**7b. A learned ranker over frozen features** (`tools/span_ranker.py`). 14
features per candidate — lexical coverage and its rank, dense cosine,
cross-encoder score, NLI entailment and contradiction, span and window duration,
unit and word counts, normalised position, silence before and after — over 3900
candidates from the 195 annotated questions, scored leave-one-conversation-out:

| rule | mean tIoU |
| --- | --- |
| argmax coverage | **0.4026** |
| argmax dense | 0.3721 |
| argmax cross-encoder | 0.3543 |
| argmax entailment | 0.2936 |
| ridge, 14 features, LOCO | 0.3889 |
| gradient boosting, 14 features, LOCO | 0.3896 |
| oracle over the same candidates | 0.5210 |

**Nothing beats plain lexical coverage, including combinations of everything.**
The fitted weights are tiny (largest standardised coefficient 0.071), which is
the model saying the features barely correlate with tIoU. This is the concrete
answer to "would fine-tuning help": the frozen scores do not contain the missing
signal, so there is nothing for a light head to learn from — and at 39
independent conversations a heavier fit would mostly memorise drug names, which
is the documented failure mode for decision detection in dialogue (Karan et al.,
SIGDIAL 2021).

**7c. Global timestamp calibration, applied to tightened spans.** Only
0.4026 → 0.4240 LOCO, with the optimum pinned at the edge of the grid. A single
offset cannot fix heterogeneous errors, and the tightened spans were wrong in
every direction at once: of 149 overlapping spans, 48 too wide, 36 too narrow,
65 offset. Revisited in experiment 9, where it works.

## Experiment 8 — the span policy was the bug

`tools/span_policy.py` holds selection fixed (argmax lexical coverage over the
top-20 candidates) and varies only what is emitted, so shaping error and
selection error separate:

| span policy | argmax coverage | oracle |
| --- | --- | --- |
| `tighten` (shipped) | 0.4026 | 0.5210 |
| `tighten`, pad 0.15 | 0.3725 | 0.4719 |
| raw window | 0.2722 | **0.6434** |
| first unit of the window | 0.0332 | 0.5929 |
| **matched unit** | **0.4360** | 0.5788 |

*matched unit* is the single utterance inside the winning window carrying most of
the matched words — about ten lines of code, no model.

Two things fall out. **Tightening to the matched words was costing 0.033 tIoU**:
an annotated passage is a phrase and an utterance is the closest thing the
transcript has to one, so shrinking further clips more than the narrowness wins
back. And the raw window has the *highest* ceiling (0.643) but the *worst*
realised score (0.272) — a reminder that a candidate set the ranker cannot
exploit is worth nothing.

```
accuracy 0.964   mean tIoU 0.436   SCORE 0.647
```

## Experiment 9 — calibration, now that the errors are systematic

Refitting the three scalars of experiment 7c on *matched-unit* spans
(`tools/calibrate_spans.py`, grid search on mean tIoU, leave-one-conversation-out):

| | uncorrected | corrected (fit) | corrected (LOCO) |
| --- | --- | --- | --- |
| tightened spans (exp 7c) | 0.4026 | 0.4252 | 0.4240 |
| **matched-unit spans** | 0.4360 | 0.4912 | **0.4898** |

```
width_scale 0.92   start_offset -0.10 s   end_offset -0.50 s
```

The same technique that bought +0.021 on tightened spans buys **+0.054** here,
and the LOCO number is within 0.0014 of the fit, so it is not overfitting three
parameters to 195 spans. The reason is visible in the fitted values: `end_offset`
is −0.50 s and `start_offset` only −0.10 s, i.e. **utterances systematically run
past the annotated phrase at the end and start almost on it**. That is a
consistent bias, which is what a global offset can fix and what the heterogeneous
tightening errors were not. Matches SLUE Phase-2's finding that the optimal
offset is a property of the ASR architecture and has to be refitted per backend.

```
accuracy 0.964   mean tIoU 0.491   SCORE 0.680
```

## Results, part two

| # | Approach | Accuracy | mean tIoU | **Score** |
| --- | --- | --- | --- | --- |
| 4 | CUDA large-v3 + dense retrieval + NLI | 0.969 | 0.456 | 0.661 |
| 6 | parakeet on MPS, same logic | 0.964 | 0.403 | 0.627 |
| 7a | + cross-encoder span reranking | 0.964 | 0.354 | 0.598 |
| 7b | + learned ranker over 14 frozen features | — | 0.389 | — |
| 8 | + matched-unit span policy | 0.964 | 0.436 | 0.647 |
| 9 | + global boundary calibration | 0.964 | **0.491** | **0.680** |

### Where the remaining points are

Accuracy 0.964 is worth at most another **+0.014** of score. The evidence half is
worth **+0.13** against the achievable ceiling:

```
mean tIoU                                    0.491
oracle, matched-unit over top-20 candidates  0.579
oracle, raw window over all candidates       0.643
```

So selection is still the whole game: the right utterance is in the shortlist and
is not being picked. Every scoring signal tried so far — lexical, dense,
cross-encoder, entailment, and learned blends of all four — ranks it no better
than word overlap does.

## Experiment 10 — the first real validation, and the bug that hid it

Three validation attempts had come back **0.200**, finishing in 6–21 seconds for
19 conversations. 0.200 is exactly the always-yes floor, and no real system
transcribes 19 conversations in six seconds, so this was never a model result.

```
RuntimeError: There is no Stream(cpu, 1) in current thread.
  File "solution/asr.py", line 188, in _transcribe_parakeet
    result = model.transcribe(path, **kwargs)
  File "parakeet_mlx/parakeet.py", line 627, in generate
    mx.eval(features, lengths)
```

**MLX streams are thread-local.** The weights were loaded on the main thread at
import by `pipeline.warmup()`, but FastAPI runs a `def` endpoint in a threadpool
worker — a different thread, with no MLX stream. Every transcription raised;
`pipeline.answer` caught it, as it is designed to, and degraded to
`(True, None)` for all ten questions. The catch-all that exists to stop one bad
conversation costing ten marks had been quietly converting the entire attempt
into the baseline, and returning HTTP 200 while doing it.

Fixed by pinning every MLX call to one dedicated worker thread, so transcription
always happens on the thread the model was created on.

**The lesson is about testing, not about MLX.** Every number up to here came
from `tools/offline_eval.py`, which calls the approach in-process and never
touches the server. `local_evaluator.py` exists precisely to catch this class of
bug and had not been run against the real endpoint since the port.

| | score |
| --- | --- |
| validation attempts 1–3 (broken threading) | 0.200 |
| **validation attempt 4** | **0.698** |
| offline at the same commit | 0.680 |

**Offline tracks validation to within 0.018**, and validation is slightly
*higher*. That makes the offline harness trustworthy as a proxy, which is what
makes the sweeps below worth running at all.

## Experiment 11 — what the candidate granularity allows

Before chasing 0.85, the question is whether it exists. `tools/ceiling.py`
measures the best tIoU *any* selector could reach at four granularities:

| candidate granularity | oracle tIoU | score at accuracy 0.964 |
| --- | --- | --- |
| single utterance | 0.6433 | 0.772 |
| window of 1–3 utterances | 0.6972 | 0.804 |
| **any contiguous word run** | **0.8631** | **0.903** |
| gold snapped to word edges | 0.8612 | 0.902 |

Two conclusions, both structural.

**0.85 is unreachable with utterance-level spans.** Perfect selection over
utterances tops out at 0.772 and over 1–3 utterance windows at 0.804. The
shipped system emits an utterance, so its ceiling is 0.80 no matter how good the
ranker gets. Getting past that requires emitting *word runs*.

**The word timings themselves cap us at 0.86.** Snapping the gold span to the
nearest word boundaries — an oracle that knows the answer and only has to
quantise it — scores 0.8612, statistically identical to the best word run
(0.8631). So the annotation does not align to word edges, and ~0.14 of tIoU is
unreachable without sub-word timing. **0.903 is the real ceiling of this
pipeline**, and 0.85 means realising 94% of it.

## Experiment 12 — cutting word-level spans, three ways that failed

`tools/word_span.py` holds selection fixed at what the shipped ranker actually
picks and varies only the span cut from inside the chosen window:

| span cut from the chosen window | mean tIoU |
| --- | --- |
| shipped (matched unit + calibration) | **0.4912** |
| minimal cover of the matched words | 0.3396 |
| densest run by matched IDF mass, length-penalised | 0.3360 |
| *oracle* run inside the chosen window | 0.6417 |
| *oracle* run anywhere | 0.8624 |

The headroom splits cleanly, and it is worth writing down:

```
0.491  shipped
0.642  perfect boundaries inside the window we already choose   (+0.151)
0.862  perfect boundaries and perfect selection                 (+0.220)
```

So **boundary cutting alone is worth +0.09 of score** (to ~0.77) without
improving selection at all — more than anything else currently on the table.

But both hand-written word-level rules score ~0.34, far *below* simply emitting
the utterance. Word-by-word lexical matching has no signal: the question's
content words are scattered across the passage, so the minimal run covering them
is either a couple of words or most of the window, and neither is the phrase.

**Extractive QA does not rescue it either.** `tools/qa_span.py` runs
`deepset/roberta-base-squad2` zero-shot over the window, decoding a start/end
pointer and reading the word timestamps underneath — the VSLNet recipe, and the
one grounding architecture RESEARCH.md expected to transfer:

| span source | mean tIoU |
| --- | --- |
| shipped (matched unit + calibration) | **0.4912** |
| extractive QA, raw | 0.2960 |
| extractive QA, nudged | 0.2952 |

The failure is a task mismatch rather than a model failure. SQuAD readers are
trained to return the *shortest string that answers a factoid question* — "100
mg", "two weeks". The annotation is the clause that establishes the claim, a
median 2.88 s of speech, perhaps eight to ten words. The reader is pointing at
roughly the right place and returning something far too narrow, which is the
worst of both worlds under IoU. These questions are also yes/no, so there is no
extractive answer to find in the first place.

**Running total of things that do not work for span selection or shaping:**
dense retrieval, cross-encoder reranking, NLI reranking, a learned ranker over
14 frozen features, minimal lexical cover, IDF-densest run, and zero-shot
extractive QA. Emitting the utterance and correcting it with three fitted
scalars still beats all of them.

## Experiment 13 — segmentation against span policy

If a word run is too fine to select and a sentence is too coarse to emit, the
remaining lever is the size of the thing being emitted. `tools/segmentation.py`
first measures what finer units do to the ceiling:

| pause | clause split | units/conv | median unit | unit oracle | window oracle |
| --- | --- | --- | --- | --- | --- |
| 0.25 | no | 55 | 1.92 s | 0.6442 | 0.7003 |
| 0.55 | no | 53 | 2.00 s | 0.6433 | 0.6972 |
| 0.25 | **yes** | 77 | 1.36 s | 0.6058 | **0.7524** |
| 0.55 | yes | 76 | 1.36 s | 0.6053 | 0.7492 |

Splitting at commas and clause-opening conjunctions cuts the median unit to
1.36 s against a gold median of 2.88 s. That *lowers* the single-unit ceiling
(0.644 → 0.606) and *raises* the window ceiling (0.700 → 0.752): a clause is
about half an annotated passage, so the passage becomes a run of two.

`tools/policy_sweep.py` then crosses segmentation with what gets emitted,
**refitting the three calibration scalars inside every cell** and reporting
leave-one-conversation-out — the boundary bias belongs to the policy, so
comparing uncalibrated spans would rank them wrong:

| clause | max_units | policy | raw | calibrated | oracle |
| --- | --- | --- | --- | --- | --- |
| no | 3 | matched unit | 0.4351 | 0.4870 | 0.7003 |
| no | 3 | matched run (≤2) | 0.4445 | 0.4967 | 0.7003 |
| no | 3 | **matched run (≤3)** | 0.4553 | **0.5097** | 0.7003 |
| no | 3 | whole window | 0.2775 | 0.3644 | 0.7003 |
| no | 4 | matched run (≤3) | 0.4457 | 0.4970 | 0.7057 |
| yes | 3 | matched unit | 0.4133 | 0.4422 | 0.7524 |
| yes | 3 | matched run (≤3) | 0.4617 | 0.5042 | 0.7524 |
| yes | 4 | matched run (≤3) | 0.4594 | 0.4995 | 0.7631 |

Two results worth keeping.

**Emitting a short run of utterances beats emitting one** (+0.023 calibrated).
A claim is often stated across a clause boundary — *"and fluconazole, 50
milligrams, for 7 days, for the mouth"* — so the passage is a run, not a
sentence. The cap matters: uncapped, the run degenerates to the whole window and
collapses to 0.28.

**The higher ceiling from clause splitting is not collectable.** Clause units
give the best oracle in the table (0.7631) and *worse* realised scores than
sentence units at every policy. This is the same lesson as the raw window in
experiment 8 — a candidate set the ranker cannot exploit is worth nothing — and
it is now the third time selection has refused to improve.

Adopted: sentence units, `matched_run` capped at 3, calibration refitted
(`scale 0.92, start −0.09 s, end −0.48 s`).

```
accuracy 0.964   mean tIoU 0.512   SCORE 0.693
```

## Results, part two (updated)

| # | Approach | Accuracy | mean tIoU | **Score** |
| --- | --- | --- | --- | --- |
| 4 | CUDA large-v3 + dense retrieval + NLI | 0.969 | 0.456 | 0.661 |
| 6 | parakeet on MPS, same logic | 0.964 | 0.403 | 0.627 |
| 7a | + cross-encoder span reranking | 0.964 | 0.354 | 0.598 |
| 8 | + matched-unit span policy | 0.964 | 0.436 | 0.647 |
| 9 | + global boundary calibration | 0.964 | 0.491 | 0.680 |
| 10 | *(validated: 0.698)* | | | |
| 12 | extractive QA span cutting | 0.964 | 0.296 | 0.578 |
| 13 | + matched-run span policy, recalibrated | 0.964 | **0.512** | **0.693** |

## How far 0.85 actually is

```
0.512   shipped
0.642   perfect boundaries inside the window already chosen     score 0.771
0.752   best window oracle, clause units                        score 0.837
0.863   perfect boundaries and perfect selection                score 0.903
0.861   gold snapped to word edges -- the timing ceiling
```

**0.85 needs mean tIoU 0.77**, which is above every oracle in this table except
perfect word-level selection. It requires realising ~89% of the absolute ceiling
of the pipeline, when the current system realises 59% of it. There is no
combination of the span policies measured here that reaches it; it would need
the selection half to improve, and selection has now resisted seven distinct
attacks.

**A realistic target is 0.75–0.78**, reached by fixing the 46 questions (23.6%)
whose span currently scores exactly zero. Those are pure selection failures — the
subject is mentioned more than once and lexical coverage picks the wrong
mention. At the current mean tIoU of the non-zero spans (0.67), converting even
half of them is worth +0.077 tIoU, or +0.046 of score.

## Experiment 14 — the evidence is the answer, not the question

Validated at **0.702** (offline 0.693), so experiment 13 carried over.

### First, entailment was being asked about the wrong text

Experiments 4 and 7b both scored the NLI premise with `context_units=2`, then
emitted the *core* window — so a window scored well when its **neighbours** held
the fact. Scoring the window alone (`tools/nli_span.py`):

| ranking signal | mean tIoU |
| --- | --- |
| argmax coverage (shipped) | **0.5121** |
| argmax entailment, with context | 0.4330 |
| argmax entailment, window only | 0.4801 |
| best blend of window-NLI and coverage | 0.4823 |
| oracle over the same candidates | 0.6624 |

The hypothesis was right — dropping the context is worth **+0.047** — and still
not enough. Entailment remains a worse localiser than word overlap. Eighth
failure.

### Then, reading the failures instead of guessing again

`tools/diagnose_selection.py` prints the cases where a much better candidate sat
in the shortlist:

```
195 questions, top-10 candidates
mean regret (oracle - chosen)   0.1502
questions losing > 0.1 tIoU     50 (25.6%)
chosen was already best         144 (73.8%)

recoverable cases:  coverage  chosen 0.629   better 0.443
                    position  chosen 0.423   better 0.460   gold 0.477
```

**In the failures the chosen window has *higher* lexical coverage than the
correct one.** Coverage is not noisy here, it is actively wrong, which is why
every attempt to blend another score with it has failed: the blend is anchored
to a misleading signal.

Reading the cases shows why:

```
Q       Does the patient deny any known exposure to COVID-19?
chosen  "Is there anything you would connect it to? ... Do you know of…"
better  "No, none that I know of."

Q       Does the patient feel well?
chosen  "I have been a little nervous about it, to be honest. Understood…"
better  "Well, actually, I feel fine."

Q       Has the cause of the tiredness remained unexplained?
chosen  "It is. Neither of those is behind the tiredness. So, what is causing…"
better  "So, what is causing it? ... We have not identified…"
```

A question *asked in the room* shares its words with the question *we are asked*
— both are interrogative sentences about the same fact. Lexical coverage lands
on the prompt; the annotation marks the reply. This is the issue → **resolution**
structure from Hsueh & Moore (NAACL 2007): the resolution utterance establishes
the fact, and the issue utterance is what looks like it.

### The fix

`tools/answer_shift.py` tests two ways of using that, both hand-written per the
Karan et al. topic-bias warning:

| rule | mean tIoU |
| --- | --- |
| argmax coverage, matched run (shipped) | 0.5121 |
| **+ shift the span past the prompt** | **0.5296** |
| + penalise question-matched windows (0.15–0.75) | 0.5143–0.5154 |
| + penalty and shifted span | 0.5212 |

Only the span shift works. Penalising a window for matching an interrogative
*reranks*, and reranking has failed nine times now; the shift leaves selection
alone and changes only where the emitted span starts — walk forward past any
interrogative utterance to the reply it elicited. It is four lines.

Combining the two is *worse* than the shift alone, which is the same lesson
again: the penalty moves selection off a window whose neighbourhood was right.

Calibration refitted afterwards (`scale 0.92, start −0.12 s, end −0.50 s`).

```
accuracy 0.964   mean tIoU 0.530   SCORE 0.704
```

## Results, part two (updated)

| # | Approach | Accuracy | mean tIoU | **Score** | Validated |
| --- | --- | --- | --- | --- | --- |
| 4 | CUDA large-v3 + dense retrieval + NLI | 0.969 | 0.456 | 0.661 | |
| 6 | parakeet on MPS, same logic | 0.964 | 0.403 | 0.627 | |
| 7a | + cross-encoder span reranking | 0.964 | 0.354 | 0.598 | |
| 8 | + matched-unit span policy | 0.964 | 0.436 | 0.647 | |
| 9 | + global boundary calibration | 0.964 | 0.491 | 0.680 | **0.698** |
| 12 | extractive QA span cutting | 0.964 | 0.296 | 0.578 | |
| 13 | + matched-run span policy | 0.964 | 0.512 | 0.693 | **0.702** |
| 14 | + skip the prompt, recalibrated | 0.964 | **0.530** | **0.704** | |

Ten things have now been tried against span selection and **one** has worked.
The nine that failed all tried to *rerank* candidates — dense retrieval,
cross-encoder, NLI with and without context, a 14-feature learned ranker,
clause-level candidates, extractive QA, lexical cover, densest run. The one that
worked changed *what is emitted from the chosen candidate* and left the ranking
untouched. That is the pattern worth carrying forward.

## Experiment 15 — how much of this is noise

Validation returned **0.692** for the build that scored 0.704 offline, after
0.702 for the build that scored 0.693. Offline went up, validation went down, so
the first question is whether any of it is real.

Paired bootstrap over the 195 annotated questions, resampling conversations:

```
per-conversation tIoU sd        0.182
19-conversation validation      resolves to about +/- 0.05
experiment 13 -> 14 difference  +0.0107  95% CI [-0.0036, +0.0256]
P(candidate better)             0.931    NOT DISTINGUISHABLE
```

**The three validation scores — 0.698, 0.702, 0.692 — are one number with noise
on it.** Tuning has been happening inside the error bars for several
experiments. From here, only changes worth more than ~0.02 offline are worth
believing, and offline is the more sensitive instrument because it has twice the
conversations.

## Experiment 16 — a 7B model that reads the conversation

The pipeline uses about 4 s of its 60 s budget. Every method tried so far is a
shallow scorer; none of them read the transcript. `tools/llm_select.py` gives
`Qwen2.5-7B-Instruct-4bit` (MLX, 4-bit) the numbered transcript and asks for
utterance *indices* — never seconds, because LLMs hallucinate timestamps rather
than copying them, and the timings are known exactly once an utterance is named.
All ten questions go in one call, since ten calls would re-prefill the transcript
ten times.

The prompt states the failure mode found in experiment 14 explicitly: prefer the
reply over the question that prompted it, and prefer the mention that settles the
matter over the first one.

| mode | questions answered | mean tIoU | shipped |
| --- | --- | --- | --- |
| free choice over all ~50 utterances (8 conv) | 42/42 | 0.4860 | 0.5286 |
| constrained to the lexical shortlist (20 conv) | 101/101 | 0.5205 | 0.5628 |

**Worse both ways**, at 6–7 s per conversation. Parsing is not the problem —
every question got a well-formed answer. Constraining it to the shortlist, where
the oracle is 0.664, helped relative to free choice but still lost to taking the
shortlist's first entry.

Eleventh failure, and the most informative one: a model that genuinely
understands the conversation, told exactly what mistake to avoid, still picks
worse than IDF-weighted word overlap.

## Experiment 17 — why word overlap keeps winning

Eleven semantic losses is too consistent for chance. The hypothesis: these are
simulated consultations, so perhaps a generator picked a span and wrote the
question from it, making lexical overlap the generating process rather than a
proxy for it. `tools/generation_check.py` tests it:

| fraction of the question's content words appearing... | |
| --- | --- |
| anywhere in the transcript | 0.563 |
| inside the annotated span | 0.446 |
| in the best non-overlapping utterance elsewhere | 0.287 |

```
gold beats every other utterance   52.8%
gold ties or loses                 47.2%
span contains >= 50% of question words   51.8%
span contains >= 90% of question words    3.6%
```

**The hypothesis is wrong** — questions are not copied from their spans; the
span holds under half of the question's content words, and only 3.6% hold nearly
all of them.

What the numbers show instead is that **the task is intrinsically ambiguous at
span level**. Nearly half the time, some other utterance matches the question's
words at least as well as the true evidence does. The annotation picked one of
several defensible passages, and the convention by which it picked is not
recoverable from the question — not by embeddings, not by entailment, and not by
a 7B model told what to look for.

That reframes the remaining headroom. The window oracle of 0.70 is not
"available with a better ranker"; a large part of it is the scorer being lucky
about which of several plausible passages the annotator chose. It also explains
the pattern in experiments 8, 9, 13 and 14: **every gain has come from changing
what is emitted, and every loss from changing what is chosen.**

## Where this leaves the score

```
accuracy    0.964    of a possible 1.000, worth +0.014 of score
mean tIoU   0.530    window oracle 0.700, word-run oracle 0.863
SCORE       0.704 offline, 0.692-0.702 validated
```

- **0.85 requires mean tIoU 0.77.** That is above every oracle measured except
  perfect word-level selection, and would mean realising 89% of a ceiling of
  0.903 when eleven attempts have failed to move selection at all.
- **A realistic ceiling for this design is 0.75–0.78**, and reaching even that
  needs the selection half to improve, which is the part that has resisted
  everything.
- The remaining untried mechanism is **fine-tuning a cross-encoder on
  (question, candidate) -> tIoU** to learn the annotation convention rather than
  reason about it. Experiment 7b argues against it — frozen features carried
  almost no signal — but 7b fitted a linear model on *scores*, not a network on
  *text*, so it is not the same test. With 39 independent conversations it is a
  coin flip, and Karan et al. predict it memorises drug names.

## Experiment 18 — training a model to learn the convention

The last mechanism available. `tools/train_span_model.py` fine-tunes
`ms-marco-MiniLM-L6-v2` (22M) to predict the tIoU a candidate would *actually
earn* under the shipped emission rule, over the top-20 lexical candidates for
each of the 195 annotated questions — 3900 pairs. Scored strictly out-of-fold
with `GroupKFold` by conversation.

Two objectives, because the first is a formulation mismatch: pointwise MSE
optimises a value the metric never reads, while the task is an argmax over each
question's candidates.

| ranking rule | mean tIoU |
| --- | --- |
| **argmax coverage (shipped)** | **0.5300** |
| trained cross-encoder, pointwise MSE, out-of-fold | 0.5167 |
| trained cross-encoder, listwise soft targets, out-of-fold | 0.4791 |
| best blend of model and coverage | 0.5014 |
| oracle over the same candidates | 0.6809 |

Training loss falls cleanly — 0.088 → 0.029 over three epochs for MSE — so the
model fits the training conversations well and **does not transfer to held-out
ones**. Fixing the objective made it *worse*, which rules out the obvious
explanation: a listwise model is forced to commit to one candidate per question,
and when the convention is not learnable that commitment costs more than
hedging.

That is exactly what experiment 17 predicted. If the annotator chose among
several equally defensible passages by a rule not present in the question text,
there is nothing for a model to learn from 39 conversations, and a network with
22M parameters will fit the drug names instead — the failure Karan et al.
(SIGDIAL 2021) describe for decision detection in dialogue.

**Selection is closed.** Thirteen methods, spanning lexical, dense, cross-encoder,
entailment, hand-written cues, global assignment, a 7B instruction-tuned LLM, a
learned feature ranker, and two fine-tuned cross-encoders. IDF-weighted word
overlap wins every time.

## Where the score is lost (current build, 0.704)

```
                                     tIoU cost   score cost   share
selection   wrong passage chosen       0.170       0.102       34%
shaping     right passage, wrong cut   0.163       0.098       33%
irreducible annotation + word grid     0.137       0.082       28%
accuracy    14 wrong answers             --        0.014        5%
```

- **Selection (0.102)** — closed, per experiment 18. 43 of 195 spans (22.1%)
  score exactly zero; the other 152 average 0.680.
- **Shaping (0.098)** — the only part that has ever responded. Every gain in this
  log came from it. Length is now right (pred median 3.01 s against gold 2.88 s)
  but position is not: the predicted centre falls outside the gold span 36% of
  the time. Given the windows actually chosen, perfect cutting is worth 0.642
  against the current 0.530.
- **Irreducible (0.082)** — snapping the *gold span itself* to word boundaries
  scores 0.861, so 0.903 is the hard ceiling of this pipeline.
- **Accuracy (0.014)** — done.

## Experiment 19 — training the boundary correction

Selection is closed, so the remaining third is shaping, which is the only part
that has ever responded. `windows.calibrate` is an **intercept-only** model of the
boundary error — every span nudged by the same three numbers — and it produced
the two largest wins in this log. The obvious next step is to make it
*conditional*: a one-utterance run may need a different nudge from a
three-utterance one, or a run starting after a long pause from one starting
mid-turn.

`tools/train_boundaries.py` regresses the residual — how far each boundary
*should* have moved, in seconds — on ten cheap features (duration, unit and word
counts, whether the run starts or ends a sentence, pause before and after,
position, question length, coverage). Ridge, leave-one-conversation-out.

The first attempt collapsed to 0.056, and the reason is worth recording:

```
boundary residual (all 195):  start +5.115s (sd 24.704)
```

A span that missed the passage entirely has a residual of tens of seconds — a
**selection** error wearing a boundary error's clothes. Fitting on those makes
the model chase the 43 misses and ruin the 152 hits. Refitting only on spans that
actually overlap the annotation:

| alpha | conditional ridge (LOCO) | global constant |
| --- | --- | --- |
| 1 | 0.4424 | **0.5300** |
| 10 | 0.4494 | **0.5300** |
| 50 | 0.4627 | **0.5300** |
| 200 | 0.4887 | **0.5300** |

**Monotone in the regularisation strength**: the harder the conditional model is
pushed toward the intercept-only model, the better it does, and it never catches
it. There is no conditional structure in the boundary error that survives a
held-out conversation — the systematic part is a constant, and the global
calibration already captures it.

One caveat kept for honesty: ridge minimises squared error on the residual, while
the global constants were fitted to maximise tIoU directly. Those are different
objectives, and part of the gap is that mismatch rather than the conditioning
itself. The monotone trend is the real evidence, and it points the same way.

## Verdict

Nineteen experiments. The score decomposition explains what is left and why none
of it is reachable with this design:

| | cost | status |
| --- | --- | --- |
| selection | 0.102 | **closed** — 13 methods, incl. a 7B LLM and 2 fine-tuned cross-encoders |
| shaping | 0.098 | intercept-only calibration is optimal; conditioning fails (exp 19) |
| irreducible | 0.082 | hard ceiling — gold snapped to word edges is 0.861 |
| accuracy | 0.014 | 0.964, effectively done |

**0.704 offline / 0.692–0.702 validated is close to the practical ceiling of this
approach**, and the measurement itself resolves only to about ±0.05 on a
19-conversation validation (experiment 15). Further tuning is not distinguishable
from noise.

The two things that would genuinely move the score — neither available here —
are a sub-word timing source (the 0.082 irreducible term) and knowledge of the
annotator's convention for choosing among equally defensible passages (the 0.102
selection term, shown unlearnable from 39 conversations in experiments 17 and 18).

## Experiment 20 — majority vote and span ensembling

Every ranker has lost to coverage, but that is an argument about which is *best*,
not about whether they are *redundant*. Ensembling reduces variance; it needs the
errors to be uncorrelated, not any member to be better. `tools/ensemble.py` tests
the premise before the method.

**The premise holds — the methods really are independent:**

| agrees with coverage on the chosen candidate | |
| --- | --- |
| dense | 9.7% |
| cross-encoder | 9.2% |
| entailment | 23.6% |
| all four agree | 4.1% |

**And there is real headroom if a combiner could work:**

| method | mean tIoU |
| --- | --- |
| coverage | **0.5300** |
| dense | 0.4910 |
| cross-encoder | 0.4686 |
| entailment | 0.4834 |
| **oracle over the four** | **0.6042** |

Knowing which method to trust per question would be worth +0.074 tIoU. But every
combiner loses:

| combiner | mean tIoU |
| --- | --- |
| majority vote on the candidate | 0.5062 |
| median of the four boundaries | 0.4932 |
| union (widest) | 0.4388 |
| mean of the four boundaries | 0.4251 |
| intersection (narrowest) | 0.4215 |

The reason is in one line, and it is the most surprising number in this log:

```
coverage tIoU when all four methods agree   0.3743  (n=8)
coverage tIoU when they disagree            0.5367  (n=187)
```

**Agreement anticorrelates with correctness.** Consensus does not mark the easy
questions — it marks the ones where a single passage dominates every similarity
measure, which is exactly the case where a lexically obvious but wrong mention
(a recap, or the doctor asking the question) attracts all four methods at once.
Disagreement is the normal, healthy state.

So majority vote cannot work here: its founding assumption, that agreement
predicts correctness, is false in this data. Boundary averaging fails for the
compounding reason from experiment 11 — averaging spans that disagree produces
something wider than any of them, and IoU punishes width hard.

Fourteenth and fifteenth failures. Ensembling is closed too, and unlike the
others it is closed for a *measured* reason rather than an empirical one.

## Experiment 21 — the assumption nobody checked: the ASR

Every ceiling in this log up to here was measured on **parakeet transcripts
only**. That was an untested assumption carried through twenty experiments.

Re-measuring on `mlx-whisper large-v3-turbo`:

| oracle | parakeet | whisper-turbo |
| --- | --- | --- |
| single utterance | 0.6433 | **0.7042** |
| window of 1–3 utterances | 0.6972 | **0.8012** |
| any contiguous word run | 0.8631 | **0.9261** |
| gold snapped to word edges | 0.8612 | 0.8251 |

**This is not a WER effect.** The fraction of question content words appearing
anywhere in the transcript is essentially identical — 0.563 for parakeet, 0.569
for whisper. What differs is **segmentation**: Whisper decodes punctuation with a
language model, so its utterance boundaries land where the annotator's passages
end. Parakeet's transducer gives *better word timings* (0.8612 vs 0.8251 on the
snapping test) and *worse utterance boundaries*, and the second matters more
because candidates are built from utterances.

The window oracle moves **+0.104**, which is larger than every modelling gain in
this log combined.

Realised, after refitting:

```
whisper, shipped calibration (parakeet's)   0.705
whisper, refitted calibration               0.717
whisper, + span_units=2                     0.718
```

The fitted offsets **flip sign** — `start +0.22 s` against parakeet's `−0.12 s`.
That is SLUE Phase-2's finding reproduced exactly: the optimal offset is a
property of the ASR architecture, not of the task. **Refit calibration whenever
the ASR changes.**

Two things re-tested on the new transcripts and still rejected: clause-level
splitting raises the window oracle to 0.8423 but *lowers* the realised score
(0.5341 against 0.5439), and per-span edge trimming of filler openers loses to
the global offset (0.5186–0.5377 against 0.5554). The global constant continues
to beat everything conditional.

Cost: 0.066x real time against parakeet's 0.018x — about 8 s for a two-minute
conversation instead of 2.2 s. Still a small part of the 60 s budget.

## Experiment 22 — entailment as a veto, not a ranker

On whisper the error profile changes shape: shaping is much better (non-missed
spans average **0.717**, up from 0.680; predicted centre inside the gold span
69.7%, up from 64.1%) and selection regret halves (0.0873 from 0.1502). What
remains is concentrated in the **44 spans (22.6%) that score exactly zero**.

Every previous use of entailment was as a *ranker*, and it lost every time. But
the reader answers the yes/no half at 0.96 while ranking spans at 0.48 — it
judges one passage well and orders ten badly. So use it for what it is good at:
keep coverage's ordering, walk down it, and take the first candidate the reader
does not reject.

| veto depth | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 0.5578 | 0.5578 | 0.5578 | 0.5578 | 0.5578 | 0.5552 |
| 8 | **0.5747** | 0.5649 | 0.5652 | 0.5652 | 0.5652 | 0.5676 |
| 10 | 0.5733 | 0.5735 | **0.5738** | 0.5689 | 0.5652 | 0.5676 |
| 14 | 0.5733 | 0.5735 | 0.5738 | 0.5738 | 0.5688 | 0.5711 |
| 18 | 0.5685 | 0.5686 | 0.5690 | 0.5690 | 0.5639 | 0.5663 |

against 0.5554 for plain argmax coverage. A broad plateau rather than a spike,
and it moves only about 20 spans in 195 — it is not overriding the ranker, it is
catching the cases where the top passage is confidently wrong.

Shipped at depth 10, threshold 0.3, in the middle of the plateau.

```
accuracy 0.962   mean tIoU 0.574   SCORE 0.729
```

## Results

| # | Approach | Accuracy | mean tIoU | **Score** | Validated |
| --- | --- | --- | --- | --- | --- |
| 4 | CUDA large-v3 + dense retrieval + NLI | 0.969 | 0.456 | 0.661 | |
| 6 | parakeet on MPS | 0.964 | 0.403 | 0.627 | |
| 9 | + matched-unit spans + calibration | 0.964 | 0.491 | 0.680 | 0.698 |
| 13 | + matched-run spans | 0.964 | 0.512 | 0.693 | 0.702 |
| 14 | + skip the prompt | 0.964 | 0.530 | 0.704 | 0.692 |
| 21 | **whisper-turbo ASR**, recalibrated | 0.962 | 0.555 | 0.718 | |
| 22 | + entailment veto | 0.962 | **0.574** | **0.729** | |

Seventeen mechanisms tried against the evidence half; three worked — the span
policy, the global calibration, and now the veto — plus the ASR swap, which was
worth more than all of them.

## Experiment 23 — correcting the noise estimate, and what transfers

Validated at **0.706**, the best so far. The four validated scores are
`0.698, 0.702, 0.692, 0.706` — a spread of 0.014, far tighter than the +/- 0.05
experiment 15 predicted.

**Experiment 15's noise estimate was the right answer to the wrong question.**
The bootstrap resampled conversations, which estimates how much the score would
vary across *different random* 19-conversation sets. But the validation set is
**fixed**, and the pipeline is deterministic, so comparing two of our own runs on
it is a *paired* comparison with essentially no sampling noise. Run-to-run
differences are signal, not noise. The +/- 0.05 figure applies to comparing our
score against *another team's*, or against a different 19 conversations — not to
our own history.

Re-reading the history with that correction:

| # | change | offline | validated | change in validated |
| --- | --- | --- | --- | --- |
| 9 | matched-unit spans + calibration | 0.680 | 0.698 | — |
| 13 | matched-run spans | 0.693 | 0.702 | +0.004 |
| 14 | skip the prompt | 0.704 | **0.692** | **−0.010** |
| 21+22 | whisper-turbo + entailment veto | 0.729 | **0.706** | +0.014 |

Two findings, one of them unwelcome.

**`skip_prompt` gained offline and lost on validation.** +0.011 offline, −0.010
validated. It is the one change in this log derived by *reading failure cases*
and then checked on the same 39 conversations that suggested it — every other
accepted change was either leave-one-conversation-out fitted (calibration) or a
structural swap (the ASR). Compelling qualitative stories are exactly what
overfits at this sample size, and this is what that looks like.

**Offline gains transfer at roughly one sixth.** Offline moved +0.049 across
these four builds; validation moved +0.008. The structural change (ASR) is
almost certainly real; the fitted pieces — span policy, veto threshold,
`skip_prompt` — are carrying less than their offline numbers suggest.

The methodological rule for the rest of this case:

- Select on **offline, leave-one-conversation-out** — 39 conversations, more
  sensitive, and the protocol that has transferred.
- Treat **validation as a held-out check**, not a tuning signal. It is a fixed
  set, so repeatedly selecting against it fits it, and the evaluation set is a
  different 38 conversations.
- Distrust any rule that came from reading examples unless it survives LOCO.

### Which gains are actually real

`tools/compare.py` paired-bootstraps two dumped runs, resampling **conversations**
rather than questions. This is the check that should have gated every accepted
change:

| change | offline difference | 95% CI | P(better) | verdict |
| --- | --- | --- | --- | --- |
| **whisper-turbo + veto** vs parakeet build | **+0.0252** | **[+0.0065, +0.0450]** | 0.996 | **real** |
| `skip_prompt` on vs off | +0.0138 | [−0.0023, +0.0304] | 0.953 | not distinguishable |

The structural change clears the bar; the hand-written rule does not, which is
consistent with it being the one change that lost on validation.

`skip_prompt` is kept on — the point estimate is positive and removing it is no
better justified than keeping it — but it is recorded as **unproven**, not as one
of the wins. The wins that survive this test are the ASR swap, the span policy
and the calibration.

The general lesson, and the one worth carrying out of this case: **a compelling
explanation of a failure case is not evidence.** `skip_prompt` came with a clean
story (the evidence is the reply, not the prompt), matched a published finding
(Hsueh & Moore's issue → resolution structure), and improved the offline number.
It still did not survive a paired bootstrap or a held-out set. Structural changes
— a better ASR — beat clever rules at this sample size.












---

# Part two — the Apple Silicon rebuild

Everything above ran on an RTX 5070. The competition machine is now an **M3 Pro
(36 GB, macOS 26.6)** and there is no CUDA box any more, which invalidates the
ASR half of the stack outright. Literature backing every choice below is in
[RESEARCH.md](RESEARCH.md).

## Setup, second time

- pyenv virtualenv **`dm`** on Python 3.11.13 (3.13/3.14 have patchy arm64
  wheels for sentencepiece/av/numba). Interpreter
  `~/.pyenv/versions/dm/bin/python`.
- torch 2.14.0 (MPS available and built), mlx 0.32.2, transformers 5.17.0,
  sentence-transformers 6.0.1, parakeet-mlx, mlx-whisper, `brew install ffmpeg`.
- Always `PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false`.

**faster-whisper had to go, not be worked around.** CTranslate2 has no Metal
backend — its backends are MKL, oneDNN, OpenBLAS, Ruy and Apple Accelerate on
CPU plus cuBLAS/ROCm on GPU, and `ValueError: unsupported device mps` has been
open since Nov 2023 with no maintainer response. Building with
`WITH_ACCELERATE=ON` only speeds the *cpu* device by 10–15%; it does not create
an `mps` device. large-v3 on M3 Pro CPU is 66–145 s for two minutes of audio,
against a 60 s budget: every single request would have timed out.

## Experiment 6 — port the ASR to MLX

`solution/asr.py` rewritten around a `BACKENDS` registry so the rest of the
codebase never learns the difference; `Word` / `Segment` / `Transcript` and the
`transcripts/<model>/` cache contract are unchanged.

| Backend | Repo |
| --- | --- |
| `parakeet` (default) | `mlx-community/parakeet-tdt-0.6b-v2` |
| `mlx-large-v3-turbo` | `mlx-community/whisper-large-v3-turbo` |
| `mlx-large-v3` | `mlx-community/whisper-large-v3-mlx` |

Parakeet was chosen on timestamps, not on WER: its timings come from **TDT
transducer frame alignment**, so they are monotonic and free of Whisper's habit
of placing a full stop seconds after the speech it follows. It also happens to
lead the Open ASR leaderboard (mean WER 6.05 against large-v3-turbo's 7.83).

One wrinkle: Parakeet emits **sub-word** tokens (`' M'`, `'or'`, `'ning'`,
`','`), so `_tokens_to_words` merges them, opening a new word on a leading
space. The leading space has to survive, because `windows._unit` rebuilds
utterance text by concatenating `word.word` directly.

```
39 conversations, 4769 s of audio, transcribed in 87 s  =  0.018 x real time
```

That is **3.5x faster than large-v3 was on the 5070** (0.064x), and the longest
conversation costs 5.3 s.

| Run | Accuracy | mean tIoU | Score | ms/conv |
| --- | --- | --- | --- | --- |
| CUDA large-v3 + neural (experiment 4) | 0.969 | 0.456 | 0.661 | — |
| parakeet + neural, CPU | 0.964 | 0.403 | 0.627 | 4547 |
| **parakeet + neural, MPS** | **0.964** | **0.403** | **0.627** | **1919** |

Two things worth recording:

- **MPS is 2.4x faster than CPU for the DeBERTa-v3-large NLI reader, with
  identical scores.** The literature warned that DeBERTa-v2/v3's disentangled
  attention is pathological on Metal — `torch.gather` on relative-position
  buckets and `xsoftmax` boolean masking are among the weakest MPS kernels — and
  2.4x is indeed far short of the 8–15x a plain BERT would get. But it is a
  speedup, it is measured, and at 1.9 s it is not the constraint. The planned
  replacement of the reader is therefore **not needed**; keeping it also keeps
  the 0.964 accuracy.
- Total cost is now **~2.2 s ASR + ~1.9 s answering ≈ 4 s per conversation**
  against a 60 s budget. There is an enormous amount of headroom to spend.

Accuracy survived the ASR change intact (0.969 → 0.964). **tIoU did not**
(0.456 → 0.403) — the span-shaping constants were tuned against Whisper's
timings.

### Where the points are now

```
tools/retrieval_ceiling.py --model parakeet
  max_units  1     2     3     4
  oracle     0.643 0.685 0.697 0.703      (CUDA large-v3 was 0.717)
  single utterance containing the gold centre: 0.634
```

```
tools/analyse_spans.py runs/parakeet_neural_base.csv
  mean tIoU               0.403
  tIoU = 0 (missed)       46 / 195  (23.6%)
  tIoU of the rest        0.527
  fraction >= 0.5         0.446
  gold length  mean 3.21  median 2.88
  pred length  mean 3.04  median 2.36
  of the 149 overlapping: 48 too wide, 36 too narrow, 65 offset
  predicted centre inside the gold span: 122/195 (62.6%)
```

Candidate generation is fine — 0.697 is available and we are getting 0.403.
**Selection is the whole problem, and it is worth ~+0.18 of score.** Accuracy,
at 0.964, is worth at most another +0.014.

## Experiment 6a — global timestamp calibration

`tools/calibrate_spans.py`, new. Fits three scalars on a dumped run — a
`width_scale` about the span midpoint, then `start_offset` and `end_offset` —
by grid search on mean tIoU, coarse then fine, reported both fitted-on-all and
leave-one-conversation-out. This is SLUE Phase-2's procedure (ACL 2023, §D.2),
which found single-scalar offsets worth a great deal and systematically
different per ASR architecture.

```
uncorrected           0.4026
best cell   scale 0.82  start -0.640s  end -0.020s
mean tIoU (fit)       0.4252
mean tIoU (LOCO)      0.4240
```

**+0.021 tIoU, and LOCO confirms it is real rather than fitted noise** — but far
short of the +0.05–0.15 the SLUE result suggested. The reason is visible in the
error profile above: our errors are *heterogeneous* (48 too wide, 36 too narrow,
65 offset), and one global shift cannot fix errors that point in both
directions. It also wants the grid edge on `start_offset`, which is a hint that
it is compensating for something structural rather than a true clock offset.

Kept, but applied last and refitted after any selection change.

## Experiment 6b — re-sweeping span shape

The old constants (`span_pad=0.3`, `min_span=2.0`, `max_span=6.0`) were tuned
for Whisper. Re-swept against the new timings — using the **lexical** approach,
because the span in `neural` is chosen by lexical ranking alone and so is
independent of the NLI reader. That makes a sweep cell cost 9 ms instead of
1919 ms, a 200x saving, and the tIoU tracks (0.389 vs 0.403).

| Sweep | Best |
| --- | --- |
| `span_pad` 0.0–0.4 | flat; anything 0.1–0.4 is within 0.002 |
| `min_span` 0.8–2.8 | **2.8**, at the grid edge |
| `max_span` 3.5–6.0 | flat above 4.5 |

tIoU 0.389 → 0.397. **Shaping is not the bottleneck.** `min_span` pinning at the
edge says the same thing the "too narrow" count does: when the centre is only
approximately right, a wider guess overlaps more. That is a symptom of bad
selection, not a shaping parameter worth tuning.

## Experiment 7 — a cross-encoder for span selection (negative)

The diagnosed failure is that the subject is mentioned more than once and
lexical scoring picks the wrong mention. `cross-encoder/ms-marco-MiniLM-L6-v2`
(22.7M parameters, ~0.3 s for the whole conversation's pairs) was added to
rescore the top-20 lexical candidates per question, blended with lexical
coverage by `rerank_weight`.

| `rerank_weight` | Accuracy | mean tIoU | Score |
| --- | --- | --- | --- |
| **0.0** (lexical only) | 0.964 | **0.403** | **0.627** |
| 0.3 | 0.964 | 0.354 | 0.598 |
| 0.5 | 0.964 | 0.354 | 0.598 |
| 0.7 | 0.964 | 0.354 | 0.598 |
| 1.0 | 0.964 | 0.354 | 0.598 |

**Worse, and flat.** Two things to take from it:

- The identical score at every non-zero weight means the cross-encoder's
  distribution is peaked enough that it wins the argmax even at weight 0.3. A
  blend weight is not a way to take a little of this model's opinion.
- This is **the same trap experiment 4 recorded for NLI reranking** (tIoU 0.456
  → 0.332), for the same reason: a relevance reranker is trained to find the
  passage a query is *about*, and prefers a wider passage with the surrounding
  context. That is the correct answer for retrieval and the wrong answer for a
  2.9-second annotated phrase.

Abandoned rather than tuned. Any model scoring passage-level *relevance* is the
wrong instrument for choosing a span here; the instrument has to be something
that reads the passage — which, at 0.965 on hard negatives, the NLI reader
already demonstrably does.

## Experiment 24 - Verbatim Evidence Extraction (2026-09-17)

Serving defaults are unchanged: Whisper turbo + neural veto, last user-reported
validation **0.706**, offline **0.7289** on all 39 training conversations.
This experiment is available separately as `--approach quote`.

Method: show the complete numbered transcript and all ten questions to a local
MLX instruction model. Return booleans plus a verbatim evidence clause and its
line number. Map quoted text to word timestamps; do not generate seconds or
force the returned span to cover whole utterances. Model load and inference run
on one dedicated thread to avoid the earlier MLX thread-local stream failure.

Pilot: first eight supplied training conversations, 80 questions / 42 positives.
No competition API requests or hidden labels were used.

| Variant | Accuracy | mean tIoU | Score | Reader seconds/conv |
| --- | --- | --- | --- | --- |
| Existing neural-veto baseline, same eight | 0.950 | 0.5700 | 0.7220 | not remeasured |
| Qwen3-4B-Instruct-2507, line-anchored quote | 0.950 | 0.528 | 0.697 | 8.0 |
| Same model, recover unique exact quotes despite wrong line | 0.950 | 0.5440 | 0.7064 | 7.8 |

Missing positive spans fell from five to one with the parser repair. The model
sometimes copies the right phrase but gives the wrong line number. Exact global
recovery is allowed only for a unique match; repeated phrases retain their line
anchor. Remaining misses include selecting an earlier medicine mention instead
of the prescription confirmation. The 4B pilot does not beat the baseline.

Qwen3-30B-A3B-Instruct-2507-4bit was downloaded and tested with the same prompt
and parser. Raw replies and latency are recorded through `trace_path`.

### Full Results and Adjudication

| Variant | Questions | Accuracy | mean tIoU | Score | Reader s/conv |
| --- | --- | --- | --- | --- | --- |
| 4B quote, no boundary correction | 390 | 0.979 | 0.536 | 0.713 | 8.0 |
| 30B-A3B quote pilot | 80 | 1.000 | 0.609 | 0.766 | 8.9 |
| 30B-A3B quote, no boundary correction | 390 | 0.995 | 0.558 | 0.733 | 9.0 |
| 30B-A3B quote, corrected boundaries | 390 | 0.995 | 0.578 | 0.745 | replay |
| Six examples, 5 conversation-disjoint folds | 390 | 0.946 | 0.486 | 0.670 | 9.0 |
| Same example-conditioned replies, repaired outer JSON | 390 | 0.985 | 0.567 | 0.734 | replay |
| Pairwise adjudication pilot | 80 | 1.000 | 0.670 | 0.802 | 13.7 |
| **Pairwise adjudication, all conversations** | **390** | **0.995** | **0.619** | **0.7692** | **13.8** |

The few-shot path is rejected: even after recovering valid per-question objects
from malformed outer JSON, it loses to zero-shot extraction. Folds exclude the
complete held-out conversation, not just the target question. Calibration was
fixed from earlier experiments on this development corpus, so this is not a
fully untouched estimate of the complete system.

Adjudication: obtain one evidence span from the existing neural-veto system and
one from the 30B reader's quote. For positive answers where the spans disagree,
show both marked evidence phrases with five seconds of neighboring context to
the 30B reader. It picks A or B; answers remain those of the 30B reader. This
tests actual evidence text rather than pooling relevance scores or averaging
unrelated timestamps. Same model and prompt settings in pilot and full run.

The 31 conversations outside the pilot score **0.7602**, versus **0.7307** for
the prior system on the same subset. Full-corpus paired difference: **+0.0403**,
conversation-bootstrap 95% interval **[+0.0084, +0.0735]**. These are development
results after adaptive experimentation, not a guarantee of hidden-set scores.

Serving profile: `APPROACH=quote_judge` (new default), Whisper turbo, 30B-A3B
4-bit, no few-shot examples, quote calibration `(0.98, +0.22, +0.02)`,
adjudication enabled. `APPROACH=neural` retains the prior serving path.

Verification: all 390 outputs well formed, 388/390 answers correct, no missing
positive spans. Reader mean/p95/max **13.82/15.62/16.55 seconds**. Real HTTP
requests through FastAPI with `CACHE_TRANSCRIPTS=0` and HF networking disabled:
106.6-second audio **24.30 seconds** total; 231.9-second audio **30.76 seconds**.
Both returned HTTP 200 with grounded spans, not fallback guesses. Combined MLX
peak **19.73 GB**, excluding PyTorch and other process memory. Temporary test
server shut down cleanly. No competition evaluation was requested.

Hosted validation reported by the user on 2026-09-17: **0.728**, versus their
reported previous **0.707** (+0.021). Earlier entries recorded the previous
score as 0.706; no fresh status request was made to resolve the rounding/history
difference. The validation was run by the user, not this assistant.

## Experiment 25 - Self-Contained Evidence Refinement

Starting point: experiment 24, **0.7692 offline / 0.728 user-reported validation**.
The current span selector misses 30/195 annotated positives. An oracle choosing
between its two candidates reaches tIoU 0.6825 versus realised 0.6188, so there
is measurable remaining selection headroom within the existing candidate pair.

Hypothesis: some candidates are vague confirmations whose topic is outside the
cited span. An opt-in `refine_evidence` mode allows the adjudicator to copy a
self-contained phrase from the candidate's five-second neighborhood. Quotes
must match uniquely and exactly within that excerpt or the original quote
prediction is retained. Answer decisions are unchanged. The validated A/B
adjudicator remains the default while this experiment is measured.

### Results So Far

| Variant | Sample | Score | Reader s/conv |
| --- | --- | --- | --- |
| Validated A/B judge | first 8 / all 39 | 0.802 / 0.7692 | 13.7 / 13.8 |
| Self-contained phrase refinement | first 8 / all 39 | 0.805 / 0.770 | 16.7 / 14.9 |
| Brief reason before A/B choice, no preference for A | first 8 / all 39 | 0.815 / 0.771 | 15.7 / 15.6 |
| Full-transcript review with option C | conversations 9-16 | 0.728 | 19.8 |
| Validated profile on conversations 9-16 | same subset | 0.7549 | not remeasured |

The promising reasoned-choice pilot did not translate into a substantial full
corpus gain. Full-transcript review regressed and was stopped at the pilot.
All variants preserve the first-pass answer decisions. None is promoted.

A diagnostic specificity rule switched six spans when the selected phrase had
zero question-content terms and the alternative had at least two: tIoU
0.6188 -> 0.6363. This was designed after reading development failures and is
not independently validated. A compact Ridge selector over 12 paired evidence
features, fixed alpha=10 and five conversation-disjoint folds, achieved tIoU
**0.6329** versus 0.6188 for the judge. With the same 0.9949 accuracy, that
corresponds to **0.7777 score**. The two-candidate oracle is 0.6825 tIoU.
The trained selector remains an offline experiment; its all-data refit is not
enabled in the API. It is accessible via `pair_selector_path` for explicit
experiments; its serialized inference matches scikit-learn to 1e-12 on all
195 feature rows. Candidate generators and calibration were previously
selected on this corpus, so the grouped result is not a wholly untouched test.

The Ridge change improves nine conversations, worsens five and ties 25. A second
fixed objective, utility-weighted logistic regression (C=0.1, weights equal to
absolute candidate tIoU difference), regressed to **0.6125 tIoU** under the same
folds. No model was promoted. Exact full-corpus scores for the prompt variants
are refinement **0.769685** and reasoned choice **0.770772**, versus the
validated-profile development baseline **0.769247**: neither is a substantive
step toward 0.8.

Another local issue: candidate text was reconstructed from calibrated timestamps
using word midpoints. For 107/195 positive baseline spans this changed the text;
11 lost content terms, including `Pamol and Ibumetin` becoming `and Ibumetin`.
An opt-in `raw_evidence_text` path reverses calibration for display only and
leaves returned seconds unchanged. A mocked adjudication check passed. It also
exposed and fixed an exception when candidate A had no span: the IoU helper
does not accept `None` as its first argument. This robustness fix does not
change the measured corpus results, which had no missing positive spans.

Dense-backbone comparison completed: `mlx-community/Qwen3-32B-4bit` under the
same quote/A-B workflow, first eight conversations (80 questions / 42 positives).
Accuracy **0.9875**, mean tIoU **0.608**, score **0.760**, reader time **71.5 s/conv**.
The validated 30B-A3B profile scores **0.802** on the same pilot at **13.7 s/conv**.
The dense model is rejected: lower pilot quality and reader latency alone exceeds
the 60-second budget before ASR. Checkpoint downloaded successfully; no hosted
validation or evaluation was requested. API defaults remain the validated
30B-A3B model with experimental flags disabled.

### Corrections to Earlier Conclusions

- Word/utterance oracles constrain those candidate sets, not all audio systems.
  They do not prove 0.77 or 1.0 impossible, or that another team is cheating.
- The snapping script uses word starts plus the final word end, whereas the
  word-run oracle uses every word end. Their difference cannot establish which
  ASR has more accurate timestamps. Neither measures WER without reference text.
- Low lexical overlap does not prove the annotated passage is unknowable.
  Failed training runs test their particular data, objective and model only.
- A bootstrap after extensive model selection is not an independent held-out
  confirmation; a positive interval does not prove future validation gains.
- Two teams scored on the same validation examples also permit paired
  comparisons if per-example outcomes are available. Dataset sampling variation
  is not a fixed +/-0.05 floor on observing changes on that fixed set.


