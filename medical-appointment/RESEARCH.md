# Research notes — medical-appointment

Literature review behind the approach in [EXPERIMENTS.md](EXPERIMENTS.md). Compiled
from two agent-run sweeps of arXiv / ACL Anthology / model cards / project READMEs.
Kept as a reference for future work: every claim below has a citation, and the
"applicable?" verdicts are specific to this case's constraints.

**The constraints that decide everything:** gold spans have a **median of 2.9 s** in a
~120 s conversation; there are **39 training conversations / 195 annotated spans**;
`score = 0.4*accuracy + 0.6*mean_tIoU`; everything must run locally on an **M3 Pro
(MPS, no CUDA)** inside **60 s per conversation**.

---

## 1. The ten findings that matter

1. **This is not video moment retrieval.** A 2.9 s span in 120 s of audio means the task
   is "pick the right utterance, then read its word timestamps". The DETR / 2D-map /
   proposal literature is built for 30–150 s moments in long video and needs 10k+
   annotated moments. CASTELLA ([arXiv:2511.15131](https://arxiv.org/abs/2511.15131))
   needed 1,009 real recordings to gain 10 points. We have 39. **Rabbit hole.**
2. **A global learned timestamp offset is the highest-ROI lever in the literature.**
   SLUE Phase-2 (Shon et al., ACL 2023,
   [arXiv:2212.10525](https://arxiv.org/abs/2212.10525), §D.2) tunes a single scalar
   offset on a 20 ms grid over [−0.3, +0.3] s; the optimum is systematically
   architecture-dependent (CTC −0.08…−0.26 s; RNN-T +0.14…+0.30 s). They also found
   `incl_blank=True` — including trailing silence in the span — optimal for nearly all
   CTC models. Relaxing the word-F1 tolerance ρ from 1 to 0.8 gave up to 30% relative
   improvement, i.e. **most errors are small boundary offsets, not wrong content**.
3. **Forced alignment beats Whisper's native DTW timestamps.** WhisperX (Bain et al.,
   INTERSPEECH 2023, [arXiv:2303.00747](https://arxiv.org/abs/2303.00747)),
   word-segmentation precision/recall at a 200 ms collar:

   | System | AMI P/R | SWB P/R |
   |---|---|---|
   | wav2vec2.0 | 81.8 / 45.5 | 92.9 / 54.3 |
   | Whisper (native DTW) | 78.9 / 52.1 | 85.4 / 62.8 |
   | **WhisperX (forced alignment)** | **84.1 / 60.3** | **93.2 / 65.4** |

   Whisper's own timestamps lose to plain wav2vec2 on AMI precision. Alignment overhead
   is <10% of inference. VAD chunking alone was worth +6.9 recall.
4. **"Whisper Has an Internal Word Aligner"** (Yeh et al., ASRU 2025,
   [arXiv:2509.09987](https://arxiv.org/abs/2509.09987)) — filter to the attention heads
   that actually align and teacher-force with *characters* instead of wordpieces. No
   training, and more accurate than prior work at a strict 20–100 ms tolerance.
5. **Padding arithmetic.** For a correctly placed span of length `L` with symmetric pad
   `p`, `IoU = L/(L+2p)`; for a pure translation error `δ`, `IoU = (L−δ)/(L+δ)`.
   At L = 2.9 s: p = 0.5 → 0.744, p = 1.0 → 0.592; δ = 0.5 → 0.706, δ = 1.0 → 0.487.
   Since tIoU carries 0.6 weight, **0.1 tIoU ≈ 0.06 score ≈ 1.5 extra correct answers.**
   **Never hedge by widening.**
6. **Don't parameterize a span by its centre.** BAM-DETR (Lee & Byun, ECCV 2024,
   [arXiv:2312.00083](https://arxiv.org/abs/2312.00083)) identifies "center
   misalignment" as the core failure of (centre, length) heads and instead predicts *any
   anchor inside* the span, regressing both boundaries independently. Practically here:
   snap the left and right boundaries independently to word onsets, pause onsets,
   sentence ends and turn changes.
7. **Verification, not retrieval, is where yes/no accuracy lives.** VitaminC (Schuster et
   al., NAACL 2021, [arXiv:2103.08541](https://arxiv.org/abs/2103.08541)) is exactly the
   hard-negative problem — contrastive evidence pairs nearly identical in wording, one
   supporting and one not. Training on it gave **+10% on adversarial fact verification
   and +6% on adversarial NLI**.
8. **Do not fine-tune on 39 conversations.** "True Few-Shot Learning" (Perez et al.,
   NeurIPS 2021, [arXiv:2105.11447](https://arxiv.org/abs/2105.11447)): without held-out
   data, selection criteria "marginally outperform random selection and greatly
   underperform selection based on held-out examples", and "often prefer models that
   perform significantly worse than randomly-selected ones". Budget ≤5 free
   hyperparameters and validate leave-one-conversation-out.
9. **"Which mention establishes the claim" is the decision-detection literature.**
   Hsueh & Moore (NAACL 2007) decompose a decision into issue → **resolution** →
   agreement; the *resolution* utterance establishes the fact. Karan et al.
   (SIGDIAL 2021) is the warning: such models "rely more on topic specific words that
   decisions are about rather than on words that more generally indicate decision
   making" — so **hand-write the cues, don't train them**. Inoue et al. (AACL 2022):
   dialogue utterances are fragmentary, so **score with ±1 utterance of context but emit
   only the core span**.
10. **Calibrate, don't fit.** Temperature scaling (Guo et al., ICML 2017,
    [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)) — "a single-parameter variant
    of Platt scaling, surprisingly effective". Isotonic regression overfits at n=390 with
    39 groups. Always `GroupKFold` by conversation: the 10 questions about one
    conversation are not independent.

---

## 2. Quantitative anchors from SLUE — the closest published task

SLUE-SQA-5 spoken QA (frame-F1, ≈ our tIoU):

| System | Test | Verified test |
|---|---|---|
| pipeline-**oracle transcript** + DeBERTa | **62.3** | **70.3** |
| pipeline-nemo (`stt-en-contextnet-1024`) | 43.3 | 45.9 |
| pipeline-wav2vec2 | 39.6 | 40.1 |
| pipeline-whisper (71M) | 32.7 | 35.7 |
| E2E-DUAL (textless) | 21.8 | 23.1 |

SLUE-NEL (named entity **localization**), frame-F1 / word-F1(ρ=0.8):
oracle 89.0/90.0 · nemo 74.1/81.4 · wav2vec2 65.2/72.0 · E2E 56.3/59.6

**Reported failure modes — these are our failure modes:**
- Pipeline ≫ end-to-end, consistently. **Never build a textless/E2E span model.**
- The oracle-transcript gap is huge → **ASR quality is the dominant bottleneck.**
- Document WER ↔ frame-F1 Pearson **−0.89**; *question* WER correlates only weakly →
  spend compute on transcribing the conversation, not on the question.
- For *localization*, "model architecture and/or training objective play a significant
  role in alignment quality" beyond WER → **choosing an ASR for timestamps is a
  different decision from choosing one for WER**.
- CTC word-boundary tokens "may not be a reliable indicator of the true time stamps" —
  hence the offset hyperparameter in finding #2.
- Pipelines are low-precision/high-recall (57.8/78.8) because the reader is trained on
  clean text and run on ASR output → **calibrate the reader on ASR transcripts**.
  (HeySQuAD: training on transcribed questions gave +12.51%.)

## 3. Other transferable results

- **DCASE 2026 Task 6** ([arXiv:2609.12484](https://arxiv.org/abs/2609.12484)): baseline
  R1@0.7 = 13.56, top-3 = 48.59 (~3.5×). Stated sources of gain: stronger features,
  stronger detector, and **confidence calibration + ensembling across temporal
  resolutions**.
- **Segmental Posterior Decoding** ([arXiv:2609.16495](https://arxiv.org/abs/2609.16495)):
  same network, replaced per-slot confidence with a globally-normalized distribution over
  segmentations → **+10.91 R1@0.7**. *The scoring rule, not the network, was worth 11
  points.* Applied here: score candidates against each other (softmax within a
  conversation) rather than thresholding each independently.
- **Otani et al., BMVC 2020** ([arXiv:2009.00325](https://arxiv.org/abs/2009.00325)):
  query-agnostic prior-only predictors are competitive with SOTA on moment retrieval →
  **build the prior-only baseline first** (fixed span at the median relative position;
  anchor ± 1.45 s) or you cannot tell whether your system is real.
- **Lost in the Middle** ([arXiv:2307.03172](https://arxiv.org/abs/2307.03172)):
  long-context models systematically under-retrieve the middle → if an LLM is used, have
  it **select an utterance ID, never emit seconds**.
- **Wallace et al., EMNLP 2019** ([arXiv:1909.07940](https://arxiv.org/abs/1909.07940)):
  subword transformers are imprecise about numeric magnitude → keep the regex quantity
  check as a **hard override** over any NLI/LLM verdict.
- **VSLNet** ([arXiv:2004.13931](https://arxiv.org/abs/2004.13931)): the one grounding
  architecture that transfers — a SQuAD-style start/end pointer over the token sequence,
  mapped back to `(word[i].start, word[j].end)`.
- **Soft-NMS** ([arXiv:1704.04503](https://arxiv.org/abs/1704.04503)): not applicable
  directly (one span per question), but the score-decay idea works as **score-weighted
  merging** of overlapping top-k spans. Weighted-average only — never union.

## 4. Decision-utterance cues (hand-written, per finding #9)

| Cue | Examples | Signal |
|---|---|---|
| Commitment | `let's`, `I'll`, `we'll`, `I'm going to prescribe`, `so the plan is`, `I want you to`, imperatives, future tense | the establishing mention |
| Agreement | "yeah, let's do that" | decision now committed |
| Retraction | `actually`, `instead`, `rather than`, `hold off` | an earlier mention is void |
| Speaker role | doctor-uttered | decisions are the doctor's |
| Position | late in the conversation | plans occur near the visit end |
| Backchannel | "uh-huh", "mm" | exclude from spans |
| Disfluency | fillers, repairs | trim from span boundaries |

---

## 5. Apple Silicon stack

### ASR verdicts (2 min of audio)

| Option | Word TS | Time | Trustworthy? | Verdict |
|---|---|---|---|---|
| **`parakeet-mlx` (parakeet-tdt-0.6b-v2)** | native word/char/segment | **2–6 s** | yes — TDT frame alignment, monotonic, ~80 ms grid | **chosen** |
| `mlx-whisper` + `whisper-large-v3-turbo` | full OpenAI DTW | 6–15 s | mid-sentence only; bad at punctuation/post-silence | fallback |
| `whisper.cpp` Metal+CoreML `-dtw` | `t_dtw` centiseconds | 8–20 s | worse (missing OpenAI's boundary heuristics) | no-Python fallback |
| **`faster-whisper` large-v3 CPU int8** | yes | **66–145 s** | — | **blows the budget — what this repo used** |
| `WhisperX --device mps` | — | crashes | — | broken since 2023 |
| `lightning-whisper-mlx` | **none exposed** | 5–12 s | — | unusable here |
| `insanely-fast-whisper --device-id mps` | yes | 40–90 s fp32 | — | slow, OOM-prone |
| **+ `ctc-forced-aligner` re-pass** | yes | **+1–2.5 s** | **±20–50 ms** | highest-ROI add-on |

**CTranslate2 has no Metal/MPS backend** — its backends are MKL, oneDNN, OpenBLAS, Ruy
and Apple Accelerate (CPU) plus cuBLAS/ROCm (GPU).
[CTranslate2#1562](https://github.com/OpenNMT/CTranslate2/issues/1562) has been open
since Nov 2023 with no maintainer response. Building with `WITH_ACCELERATE=ON` only
speeds the *cpu* device by 10–15%; it does not create an `mps` device.

Open ASR leaderboard (mean WER ↓ / AMI): **parakeet-tdt-0.6b-v2 6.05 / 11.16** ·
distil-large-v3.5 7.21 / 14.63 · distil-large-v3 7.52 / 15.16 · large-v3-turbo 7.83 /
16.13. large-v3 → turbo costs ~+0.4–0.6 WER for ~5× speed (32 → 4 decoder layers).

⚠️ **turbo and distil have structurally weaker word timestamps** — DTW averages over
cross-attention heads and there are only 4 (turbo) or 2 (distil) decoder layers.
whisper.cpp ships **no DTW preset for distil**. Do not ship distil's native word timings.

### Forced alignment
- `torchaudio.functional.forced_align` / `pipelines.MMS_FA` were **deprecated in 2.8 and
  removed in 2.9** ([pytorch/audio#3902](https://github.com/pytorch/audio/issues/3902)).
- **`ctc-forced-aligner`** (BSD-2, maintained): ≥5× less memory, sentence/word/char
  granularity, any HF wav2vec2/HuBERT/MMS CTC checkpoint, `<star>` support for partial
  transcripts. ⚠️ its default model is **CC-BY-NC** — use `facebook/wav2vec2-base-960h`
  (Apache-2.0). Encoder on MPS, Viterbi on CPU.
- Boundaries land on 20 ms frame edges; per-word confidence comes free.
- CTC "peaky" behaviour clips word *ends* by 20–60 ms — the fitted `end_offset` absorbs it.
- MFA: skip. Kaldi/conda install, 30 s+ startup, and SLUE's own audit found 51/372 MFA
  entity alignments misaligned anyway.

### Readers on MPS

| Model | Params | ~120 pairs on MPS | Verdict |
|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L6-v2` | 22.7M | **0.15–0.4 s** | free; use for ranking |
| `cross-encoder/ms-marco-MiniLM-L12-v2` | 33M | 0.25–0.6 s | ties L6 (74.31 vs 74.30 NDCG) |
| `BAAI/bge-reranker-base` | 278M | 1.2–3 s | XLM-R, 250k vocab, Chinese benchmarks |
| `DeBERTa-v3-large-mnli-...` | 435M | 5–15 s | strong but slow on Metal |
| `mxbai-rerank-v2` | 0.5–1.5B | 15–45 s | generative; wrong tool |

DeBERTa-v2/v3's disentangled attention computes three attention terms with
`torch.gather` on relative-position buckets plus `xsoftmax` boolean masking — among the
weakest MPS kernels, several CPU-fallback. Expect only 1.5–3× over CPU versus 8–15× for
a plain BERT of the same size. *(Measured here: 2.4×, which turned out acceptable.)*
A reranker scores **relevance** and will happily rank a near-miss contradiction highly —
use it for ranking, never for entailment.

### Small local LLM (llama.cpp, Llama-2 7B Q4_0, TG128 tok/s)
M1 14.2 · M1 Pro 36.4 · M1 Max 61.2 · M2 21.9 · **M3 Pro 30.7** · M3 Max 66.3 ·
M4 24.1 · M4 Pro 50.7 · M4 Max 83.1. Mac mini M4 + llama3.2:3b Q4_K_M: PP512 530 t/s,
TG128 46 t/s. Budget: ~900 prompt + ~120 output tokens ⇒ **4–7 s on M3 Pro** at 3B Q4.
- **Ask all 10 questions in one call** — ten calls re-prefill the transcript ten times.
- `mlx-lm` has native prompt caching; llama.cpp server reuses the prefix cache.
- llama.cpp **GBNF** grammar (`root ::= "yes" | "no"`) is a logit mask, essentially free.
  Use `{9}`/`{0,N}` repetition syntax, never `x? x? x?` chains
  ([#4218](https://github.com/ggml-org/llama.cpp/issues/4218)). The JSON schema is *not*
  injected into the prompt — describe it there too.
- **Cheapest trick: don't generate.** One forward pass over `...Answer:` and compare
  `logits[" yes"]` vs `logits[" no"]` — single prefill, zero decode, calibrated score.

### Environment gotchas, ranked by damage
1. **libomp double-init** — torch and ctranslate2 each ship `libomp.dylib` →
   `OMP: Error #15`. Best fix: never import both in one process. `KMP_DUPLICATE_LIB_OK`
   suppresses it but can corrupt results.
2. **fp16 on MPS fails**: `input types 'tensor<...xf16>' and 'tensor<1xf32>' are not
   broadcast compatible / LLVM ERROR`. Use fp32 or bf16.
3. **No arm64 wheels** for `flash-attn`, `bitsandbytes`, `nvidia-*`, `triton`, `vllm`,
   `deepspeed`, `apex`. `nemo_toolkit[asr]` is Linux-only.
4. **ffmpeg required** by mlx-whisper, parakeet-mlx and ctc-forced-aligner.
5. **Python 3.13/3.14 arm64** has patchy wheels for sentencepiece/numba/av/coremltools —
   use **3.11**.
6. **Warm every model before the 60 s timer**: MLX kernel JIT, Core ML ANE compile, HF
   downloads.
7. `PYTORCH_ENABLE_MPS_FALLBACK=1`, `TOKENIZERS_PARALLELISM=false`.

---

## 6. Explicitly flagged rabbit holes

- Training any DETR-family grounding model (Moment-DETR / QD-DETR / CG-DETR / BAM-DETR /
  AM-DETR) — needs 10k+ annotated moments.
- Textless / end-to-end audio span prediction (DUAL-style) — 21.8 vs 43.3 frame-F1.
- Diffusion span refinement (MomentDiff) — no zero-shot path.
- Montreal Forced Aligner on macOS.
- Fine-tuning DeBERTa on 390 examples.
- Feeding the whole transcript to an LLM and asking for `(start, end)` in seconds.
- Training a decision-utterance classifier on 39 conversations (topic bias).
- Isotonic calibration at n=390 with 39 groups.
- Soft-NMS for span diversity — only one span per question is emitted.

## 7. References

**Temporal grounding** — [2107.09609](https://arxiv.org/abs/2107.09609) Moment-DETR ·
[2203.12745](https://arxiv.org/abs/2203.12745) UMT ·
[2303.13874](https://arxiv.org/abs/2303.13874) QD-DETR ·
[2311.08835](https://arxiv.org/abs/2311.08835) CG-DETR ·
[2307.02869](https://arxiv.org/abs/2307.02869) MomentDiff ·
[2312.00083](https://arxiv.org/abs/2312.00083) BAM-DETR ·
[1912.03590](https://arxiv.org/abs/1912.03590) 2D-TAN ·
[2004.13931](https://arxiv.org/abs/2004.13931) VSLNet ·
[2009.00325](https://arxiv.org/abs/2009.00325) Otani et al. ·
[1704.04503](https://arxiv.org/abs/1704.04503) Soft-NMS

**Audio moment retrieval** — [2409.15672](https://arxiv.org/abs/2409.15672) AM-DETR ·
[2511.15131](https://arxiv.org/abs/2511.15131) CASTELLA ·
[2609.12484](https://arxiv.org/abs/2609.12484) DCASE'26 Task 6 ·
[2609.16495](https://arxiv.org/abs/2609.16495) Segmental Posterior Decoding

**ASR timestamps** — [2303.00747](https://arxiv.org/abs/2303.00747) WhisperX ·
[2509.09987](https://arxiv.org/abs/2509.09987) Whisper's internal word aligner ·
[2007.09127](https://arxiv.org/abs/2007.09127) CTC-segmentation ·
[2305.13516](https://arxiv.org/abs/2305.13516) MMS

**Spoken QA** — [2212.10525](https://arxiv.org/abs/2212.10525) SLUE Phase-2 ·
[2203.04911](https://arxiv.org/abs/2203.04911) DUAL/NMSQA ·
[1804.00320](https://arxiv.org/abs/1804.00320) Spoken-SQuAD ·
[2304.13689](https://arxiv.org/abs/2304.13689) HeySQuAD

**Verification** — [1803.05355](https://arxiv.org/abs/1803.05355) FEVER ·
[2004.14974](https://arxiv.org/abs/2004.14974) SciFact ·
[2103.08541](https://arxiv.org/abs/2103.08541) VitaminC ·
[1909.07940](https://arxiv.org/abs/1909.07940) Do NLP models know numbers ·
[1910.06701](https://arxiv.org/abs/1910.06701) NumNet ·
[2111.09543](https://arxiv.org/abs/2111.09543) DeBERTa-v3

**Small data** — [1909.00161](https://arxiv.org/abs/1909.00161) zero-shot entailment ·
[2104.14690](https://arxiv.org/abs/2104.14690) EFL ·
[2105.11447](https://arxiv.org/abs/2105.11447) True few-shot ·
[1706.04599](https://arxiv.org/abs/1706.04599) Calibration

**Dialogue decisions** — Hsueh & Moore NAACL 2007 · Purver SIGDIAL 2008 · Frampton
EMNLP 2009 · Karan SIGDIAL 2021 · Inoue AACL 2022 ·
[2306.02022](https://arxiv.org/abs/2306.02022) ACI-Bench · MTS-Dialog EACL 2023 ·
[2307.03172](https://arxiv.org/abs/2307.03172) Lost in the Middle
