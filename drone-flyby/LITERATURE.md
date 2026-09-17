# Literature review

What published work does about this problem, with numbers. Produced by the
`Literature Scout` agent (`.github/agents/literature-scout.agent.md`), which can
be re-run to extend this file.

**Correction, 2026-09-17:** the original framing below overclaimed what our
experiments established. The "2.6-point FP ceiling" assumed perfect true boxes
and high true-positive confidence; it is not a real error budget. The partial
hand-labelled recording score is recall, not AP, with disputed aircraft labels.
Recommendations dismissing negatives, intermediate pretraining or camera
acquisition on those grounds are withdrawn. See [EXPERIMENTS.md](EXPERIMENTS.md),
section 26, for the controlled sensitivity check and current proposal. Published
cross-dataset gains are motivations for local tests, not predicted gains here.

## Second Scout - 2026-09-17

**Current synthesis; supersedes the prescriptive conclusions of the first
survey below.** This pass read primary methods, tables and limitations, rather
than treating abstract claims as a recipe. It covers the SODA survey/benchmark,
current SAM 3/3.1 and RF-DETR, reference-based CNOS, a recent cross-domain
prototype method, few-shot transfer, focus-and-detect, and error analysis.
This is a targeted survey, not an exhaustive September 2026 leaderboard.

### Frame the task correctly

Our problem combines small-object detection, few-example custom recognition,
scene shift, and causal camera control. The supplied flight contains one
instance per class seen repeatedly; adjacent frames are not independent shots
or independent test examples. The wire input is always 960x540; L0/L1/L2
provide different source detail, and the score is box AP50 for the full current
frame. Source: [README.md](README.md), Supplied Data / Resolution / Scoring.

Local evidence: [EXPERIMENTS.md](EXPERIMENTS.md), sections 26-30, records
confident detector-only roof errors, partial improvement from the binary
verifier, and the limitations of both synthetic metrics and provisional hand
labels. A large report count is not an error decomposition. We do not know the
leading team's method. External training imagery stays deferred, recorded
validation data stays out of fitting, and competition attempts are user-run.

There is **no single universal SOTA procedure** across these tasks. The common
functional decomposition is: define the concept, localize candidates at a useful
scale, score identity and non-target evidence, refine/deduplicate, then track.
A fine-tuned detector can learn several of these functions jointly; separate
SAM and Siamese networks are not obligatory. [S1, S2, S3, S4, S5, S6]

### Sources and measurement scope

Source IDs below refer to these titles and versions. AP is in percentage
points; **box AP50, box AP50:95, mask AP, and cgF1 are not interchangeable**.

| ID | Primary source and status | Data / scale / input actually checked |
|---|---|---|
| S1 | [Towards Large-Scale Small Object Detection: Survey and Benchmarks](https://arxiv.org/html/2207.14096v4), TPAMI 2023; arXiv:2207.14096v4 | SODA-A: 2,513 aerial images, 872,069 instances, 9 classes, average image 4761x2777; reported average Small-object size 14.75 px. Oriented boxes and area-defined size subsets. Benchmark crops 800x800 at stride 650, resizes to 1200x1200. |
| S2 | [CNOS: A Strong Baseline for CAD-based Novel Object Segmentation](https://arxiv.org/html/2307.11067v4), ICCV 2023 R6D Workshop; arXiv:2307.11067v4 | Seven core BOP datasets: LM-O, T-LESS, TUD-L, IC-BIN, ITODD, HB, YCB-V; 132 rigid objects in clutter. Test images resized to width 640 preserving aspect ratio; masked/padded descriptor crops 224x224. Typical object size and original image-size distribution not reported in the inspected method; do not assume tiny aerial objects. |
| S3 | [SAM 3: Segment Anything with Concepts](https://arxiv.org/html/2511.16719v2), arXiv:2511.16719v2, March 2026; [SAM 3.1 release](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md), March 27, 2026 | SA-Co, LVIS, ODinW13 and RF100-VL; high-quality concept training includes 5.2M images and 4M noun phrases. PE encoder plus prompt-conditioned DETR/mask heads; usual input 1008x1008. No validated sub-16-pixel custom-asset AP for our task. |
| S4 | [RF-DETR: Neural Architecture Search for Real-Time Detection Transformers](https://arxiv.org/html/2511.09554v2), ICLR 2026; arXiv:2511.09554v2 | COCO and 100 separately adapted RF100-VL datasets, not zero-shot transfer of one fitted detector to every city. DINOv2 + Objects365 pretraining + weight-sharing NAS. COCO N/S/M/L inputs 384/512/576/704 square; RF100-VL main results use dataset-specific configurations. Typical object sizes vary; COCO AP_S is an area category, not our max-side statistic. |
| S5 | [Learning Multi-Modal Prototypes for Cross-Domain Few-Shot Object Detection](https://arxiv.org/html/2602.18811v1), arXiv:2602.18811v1; abstract page says accepted to CVPR 2026 Findings | ArTaxOr, Clipart1k, DIOR aerial, DeepFish, NEU-DET, UODD, with 1/5/10 labelled target-domain instances per class. GroundingDINO Swin-B/BERT with text and visual branches. Finest prototype features at stride 8; exact input dimensions and typical object-pixel sizes not specified in inspected implementation text. |
| S6 | [Frustratingly Simple Few-Shot Object Detection](https://arxiv.org/html/2003.06957v1), ICML 2020; arXiv:2003.06957v1 | VOC, COCO and LVIS; Faster R-CNN, ResNet-101 FPN in main experiments. Base classes have abundant labels, novel classes few labels. Exact input resize and typical object-pixel sizes not verified here. This is a methodological baseline, not a 2026 leaderboard winner. |
| S7 | [Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection](https://arxiv.org/html/2202.06934v5), ICIP 2022; arXiv:2202.06934v5 | VisDrone2019-Detection and xView; objects include widths <1% of image width. VisDrone training slices 480-640 px, inference 640x640; xView training 300-500 px, inference 400x400. Resize to reported width 800-1333 preserving aspect ratio; 25% overlap setting. FCOS/VFNet/TOOD experiments. Original dimensions vary; not a fixed 960x540 input benchmark. |
| S8 | [AutoFocus: Efficient Multi-Scale Inference](https://openaccess.thecvf.com/content_ICCV_2019/html/Najibi_AutoFocus_Efficient_Multi-Scale_Inference_ICCV_2019_paper.html), ICCV 2019 | COCO test-dev, ResNet-101 detector and category-agnostic FocusPixels. Official abstract verified; exact pyramid resolutions, typical object size and detailed ablations not extracted. Kept as the direct architectural precedent for heatmap-to-crops. |
| S9 | [TIDE: A General Toolbox for Identifying Object Detection Errors](https://arxiv.org/html/2008.08115v2), ECCV 2020; arXiv:2008.08115v2 | Analysis of seven detection/segmentation models across COCO, VOC, Cityscapes and LVIS. Takes labelled ground truth and saved predictions; not a trained detector and has no prescribed input resolution or promised AP gain. |

### Theme 1: Proposals are not identities

**The reference-matching procedure is real, not an invented extra stage.** CNOS
onboards each object with 42 rendered viewpoints, generates masks with SAM or
FastSAM, blacks out the background, resizes/pads crops, and compares DINOv2
descriptors to reference descriptors using cosine similarity. Class score is
aggregated across templates; the top-five mean is a tested option. No retraining
on novel objects is needed in its CAD protocol. [S2, sections 3-4]

| CNOS measurement | Result and interpretation |
|---|---|
| Seven-dataset mean mask AP50:95 | FastSAM + PBR templates: **41.2**, versus Chen et al. **21.4**; +19.8 points. SAM + PBR: **40.4**. Not box AP50 and not our expected gain. [S2, Table 1] |
| Better template rendering with same SAM proposals | Pyrender **36.1 -> PBR 40.4**; +4.3 points. Reference appearance matters even with foundation features. [S2, Table 1] |
| More viewpoints on LM-O | FastSAM **39.7 -> 39.7** for 42 -> 162 views; no gain. SAM **39.6 -> 39.5**. More references are not automatically better. [S2, Table 3] |
| Similarity aggregation on LM-O | Across SAM/FastSAM, all-view mean **36.40**, max **39.40**, top-five mean **39.65**. Do not average incompatible viewpoints blindly. [S2, Table 4] |
| Runtime on V100 | SAM proposal stage **1.58 s**, matching **0.13 s**; FastSAM **0.22 s + 0.12 s**. These are that implementation/hardware, not RTX 5070 timings. [S2, Table 5] |

**What did not work:** the authors found that many residual failures were
DINOv2-based classification errors despite well-aligned masks. Their attempt
to infer pose using the global cls descriptor also failed. CAD-free matching
from a few photographs is discussed as an extension, not validated by the main
CAD benchmark. [S2, sections 4.2 and 4.4]

**Here:** inspect masks and compare matched-scale Helsinki references. A frozen
reference classifier is a sensible low-training baseline before a custom
Siamese network. However, our existing binary EfficientNet head is NOT CNOS: it
has no per-class reference comparison and keeps contextual pixels. CNOS does
not prove that background removal always helps, that our masks are reliable,
or that tiny-object proposals will be complete. Test crop-only versus masked
crops rather than silently replacing one with the other. [S2; local section 28]

### Theme 2: Prompting does not replace domain adaptation

SAM 3 detects all instances of a prompted concept using short text phrases,
positive/negative exemplar boxes, or both. Its presence head separates
image-level concept existence from conditional localization. That is distinct
from SAM 1/2's point/box-to-mask role. Its limitations explicitly name
**fine-grained aircraft types** as a difficult zero-shot case. [S3, sections 2-3,
appendices B and C]

| SAM 3 measurement | Result and conditions |
|---|---|
| RF100-VL box AP, original dataset names as prompts | Zero-shot **15.2**, 10-shot fine-tuning **36.5**. GroundingDINO-T **15.7 -> 33.7** in the same summary. Specialized prompts remain difficult; this is adaptation, not a free prompt-only gain. [S3, few-shot table / appendix F.3] |
| ODinW13 box AP | **61.0 -> 71.8** from zero-shot to 10-shot. Different dataset mix from RF100-VL; do not average away the difference. [S3, appendix F.3] |
| Hard negative phrases per image in a lighter ablation | **0 -> 30** gives **28.3 -> 43.0 cgF1** and image-level MCC **0.44 -> 0.68**. These are absent/confusable CONCEPT phrases, not 30 empty image crops. [S3, Table 8(b), appendix A.2] |
| Presence-head ablation | **50.7 -> 52.2 cgF1**; MCC **0.77 -> 0.82**, while positive micro-F1 **65.4 -> 63.4**. Better rejection can trade against localization/recall; not every component improves every metric. [S3, Table 8(a)] |

SAM 3 few-shot adaptation uses box labels with mask losses disabled, 40 epochs,
batch size 2 and one-tenth the standard learning rate. Thus boxes can be
sufficient for adaptation; acquiring perfect masks is not always prerequisite.
[S3, appendix F.3]

**Important evaluation caveat:** its one-exemplar AP+ experiment obtains the
exemplar from a ground-truth box in a positive query image. That is not evidence
that an isolated reference from another city will work equally well. Concepts
must be consistent between text and visual prompts. [S3, sections 2 and 6]

**Here:** generic prompts can propose aircraft or vehicles, but competition
names such as `ta-ta` require a verified visual definition. SAM output is a
candidate, not ground truth. A prompt/reference test must include views without
that target, not just attractive positive examples. The existing binary
verifier is only a partial step toward that test. [S3; local section 28]

### Theme 3: Use positive AND confusable-negative references

LMP constructs class prototypes from support RoIs and negative prototypes from
jittered TRAINING query boxes. It first adapts a visual branch, then jointly
trains visual/text branches using focal classification and box regression;
the branches are ensembled at inference. It does not require an extra Siamese
contrastive objective. Hard-negative query construction uses known training
boxes, not unavailable test ground truth. [S5, sections 3-4]

- Six-domain, 5-shot mean box mAP: **44.0 versus 40.4** for its cited
  fine-tuned GroundingDINO baseline; DIOR aerial **31.3 versus 29.6**.
  These are published cross-domain benchmark numbers, not ours. [S5, Table 1]
- Its finest stride-8 prototype layer gives **44.0** mean, versus **42.9** for
  stride 64 in its feature ablation. But NEU-DET and UODD prefer stride 16:
  highest spatial resolution is not universally best. [S5, appendix B Table 3]
- Three negative prototypes worked best in the ArTaxOr sensitivity experiment;
  five slightly degraded performance. Atypical support examples and dual-branch
  overhead are stated limitations. This is not a universal negative-count rule.
  [S5, section 4.2 and limitations]
- It is not best everywhere: DIOR 1-shot **17.2**, versus Domain-RAG **18.0**;
  NEU-DET 10-shot **25.7**, versus Domain-RAG **26.3**. [S5, Table 1]

Source caution: Table 1's 10-shot means are **46.6 vs 43.9**, a **2.7-point**
gap, while the accompanying prose says 2.1. Its appendix uses a separately
reported 5-shot baseline of 40.8 rather than Table 1's 40.4. Keep those settings
and inconsistencies visible; do not manufacture a clean universal gain. [S5]

**Here:** compare candidate features with a class-specific reference bank AND
verified negatives from allowed training imagery. Include difficult target
crops, partial views, and confusing target-to-target pairs; measure per-class
retention as well as rejection. Unlike LMP's target-domain supports, our support
scenery is Helsinki and the next city is different. No guarantee bridges that
extra shift. Negative references from recorded validation images would violate
our training boundary and are excluded. [S5; repository constraints]

### Theme 4: A specialist remains a legitimate baseline

RF-DETR's main RF100-VL table reports **60.9 box AP / 87.5 AP50** for Small,
versus YOLO11-S **56.4 / 82.5**. But those models are fine-tuned on each target
dataset; this is NOT one Helsinki-trained model transferred untouched to an
unseen city. The authors also report TensorRT/export and NMS discrepancies for
YOLO, so this is not proof that our architecture is intrinsically unsuitable.
[S4, Tables 1 and 4]

Useful ablations, not brand conclusions:

- On COCO, gentler hyperparameters lower **52.6 -> 51.6 AP**; adding DINOv2
  gives **53.6**, and additional Objects365 pretraining gives **54.3**. Benefits
  depend on representation AND training recipe. [S4, Table 5]
- Fixed COCO-derived RF-DETR-S architecture fine-tuned on RF100-VL reaches
  **60.2 AP**, versus dataset-specific NAS **60.9**. A usable fixed architecture
  does not require repeating their entire search. [S4, Table 18]
- Post-NAS COCO detection fine-tuning changes Small by only **+0.1 AP**; several
  segmentation variants do not improve. More stages are not inherently better.
  [S4, Tables 15-16]

TFA supplies an older controlled alternative: train base Faster R-CNN, freeze
the backbone/RPN/proposal features, then adapt only box classification and
regression on a balanced support set. Its reported VOC split-1 10-shot novel
AP50 is **56.0**, versus **45.5** for its full-fine-tune reimplementation. On
COCO 10-shot novel AP50:95, the narrower difference is **10.0 versus 9.2**.
Cosine classification helps most in extremely low-shot cases, not uniformly at
every shot count. [S6, Tables 2-3 and section 4]

**Here:** preserve a reproducible specialist control; explicit optimizer and
logged effective LR are essential. Our prior `optimizer=auto` experiment did
not test the claimed learning-rate reduction. Head-only adaptation and a
properly controlled low-backbone-LR experiment remain valid options, not settled
failures. RF-DETR itself remains abandoned at the user's request. [S4, S6;
EXPERIMENTS section 26]

### Theme 5: Scale is part of the method, not a post-processing trick

SAHI trains on image slices plus original images, infers on overlapping slices,
maps detections back, and merges duplicates. On VisDrone TOOD, **29.4 AP50**
full-image baseline becomes **34.7** with sliced+full inference+overlap, and
**43.5** with sliced fine-tuning as well. Large-object AP50 is **66.4** for
sliced training/full inference, **60.2** with slice-only overlapping inference,
and **65.4** with full-image inference restored. Small-object gains can cost
large-object recall. [S7, Table 1]

On xView, TOOD **2.1 -> 20.6 AP50** for sliced fine-tuning + overlapping sliced
inference; adding full-image inference changes **20.6 -> 20.4**, not a universal
win. Use those table values, not a single headline gain pooled across datasets.
[S7, Table 2]

AutoFocus predicts class-agnostic FocusPixels, forms compact FocusChips, then
runs the detector on selected finer-scale regions. Its official abstract
reports **47.9 AP / 68.3 AP50** on COCO test-dev, matching its multi-scale
baseline at **2.5x** the speed; **6.4 images/s** on Titan X Pascal. Exact crop
settings and finer ablations were not verified in this pass. [S8]

The SODA benchmark is another example of the high-resolution procedure: crop
before destructive resizing, retain fine features, and merge in original-image
coordinates. In its SODA-A comparison, RoI Transformer is **36.0 AP / 73.0
AP50**, versus Rotated RetinaNet **26.8 / 63.4**. These use oriented boxes and
different area rules; they are not directly comparable to our horizontal AP50.
[S1, sections V.A-B, Table IX]

**Here:** both AutoFocus and SAHI have access to the source image for all crops.
We do not. Actual L1/L2 capture is an action with travel and coverage costs.
Interpolated L0 crops may change network sampling but cannot recreate withheld
pixels. Our section-27 oracle experiment rejected one planner configuration,
not AutoFocus or all heatmap approaches. Keep perception and camera changes
separate in initial manual-validation comparisons. [S7, S8; README; section 27]

### Theme 6: Confirm tracks without borrowing future evidence

SAM 3 separates detection from tracking, suppresses tracks inconsistent with
detections, and refreshes memory from confident detections. Its SA-V test video
ablation changes **27.1 -> 30.3 cgF1** and **55.9 -> 58.0 pHOTA** when temporal
disambiguation is enabled. Neither metric is our box AP50. [S3, Table 38(a)]

Crucially, the described default confirmation procedure displays frame t only
after observing **t+15**. At our 3 fps this would require approximately **5 s**
of future imagery, and our server cannot receive that while holding the current
request. Use causal evidence from already received frames instead; do not claim
the published gain for a modified causal tracker. Changing crop coordinates
also prevents treating received views as an ordinary fixed-camera video. [S3,
appendix C.3; timing derived from README]

SAM 3.1 improves multiplexed tracking efficiency, but is not an across-the-board
accuracy upgrade: its release table gives LVVIS **36.3 -> 34.3 mAP** and OVIS
**60.5 -> 61.5 mAP**. The paper's multiplex **5.2x** speedup and release's
approximately **7x** at 128 objects use different comparison descriptions and
optimization bundles on H100; neither establishes latency on our GPU. [S3,
appendix H; linked official 3.1 release]

**Here:** tentative/confirmed states and conservative memory updates are useful
design patterns, but persistent roofs are persistent too. Temporal consistency
is not semantic verification. Reject/de-duplicate on reliable evidence and
measure the recall lost by delayed confirmation. [S3; engineering implication]

### Theme 7: General development procedure

This is a synthesis for this repository, not a verbatim recipe or guaranteed
result from any one paper:

1. **Define the target and trusted labels.** Document what counts as each asset,
   what its box should enclose, and whether shadows/partial objects count.
   Review masks and class assignments against the provided annotations. Keep
   uncertain recording labels uncertain; do not use model matches as automatic
   ground truth. Split at ground-region/sequence level where possible. [S1, S9]
2. **Build a reproducible single-view baseline.** Keep detector, resizing,
   confidence ranking, source-coordinate mapping, precision and model hash
   fixed. Evaluate the exact inference artifact, not float32 accuracy paired
   with a different accelerated model's speed. [S4, sections 4 and appendix A]
3. **Measure candidate recall and final precision separately.** High-recall
   proposals may be numerous internally; final reported predictions should
   have class and localization support. Diagnose background, class, localization,
   joint class/localization, duplicate and missed-object errors. TIDE weights
   these by individual AP changes, not raw counts; those changes do not add up
   to an error budget. Reliable ground truth is still required. [S9, section 2]
4. **Adapt with foreground and hard negatives.** Start from pretrained features,
   use a small verified reference bank or a controlled specialist fine-tune,
   and include the confusing TRAINING proposals the current system produces.
   Keep class balance, representative positives and held-out calibration. A
   fixed empty-image percentage is not established by these papers. [S2-S6]
5. **Handle real scale and causal memory.** Compare L0/L1/L2 sampling, proposal
   generation cost and crop verifier cost; add tracking only after single-view
   behavior is understood. Real zoom, interpolation and feature-map resolution
   are different controls. Background rejection and duplicate removal belong
   in the complete evaluation, not just attractive demo crops. [S1, S3, S7-S9]
6. **Expose one-variable API experiments.** Keep a control preset. Log model
   hashes, prompts/references, threshold, precision, per-stage counts, timing,
   camera feedback and frame gaps. Synthetic results remain diagnostics; after
   startup/protocol/timing checks, make runnable experiments available for the
   user's manual competition validation. No agent-submitted attempts and no
   recorded validation training. [Local workflow, EXPERIMENTS sections 29-30]

### Disagreements and limits

- **Generic prompting vs specialist fine-tuning:** SAM 3 can excel zero-shot on
  ODinW13 yet be weak zero-shot on specialized RF100-VL names. RF-DETR's high
  RF100-VL scores follow dataset adaptation. Those are different protocols,
  not evidence that one family wins universally. [S3, S4]
- **Freezing vs adapting:** TFA shows value in freezing a strong feature/proposal
  stack for related novel categories; LMP adapts to different domains and reports
  sensitivity to supports. Neither proves a fixed training policy for tiny CGI
  assets on unseen terrain. [S5, S6]
- **Better masks vs better recognition:** CNOS reports descriptor confusion
  despite good masks; SAM 3 explicitly separates presence from localization.
  Adding a segmenter alone is not the demonstrated rejection mechanism. [S2, S3]
- **Pixel-size comparisons:** SODA size is area-oriented, our logged 13.5 px is
  a median maximum side after downsampling, and COCO AP_S is an area bucket.
  Similar numbers do not define matched difficulty. [S1, S4; local section 0]
- The LMP feature-ablation baseline and mean-delta prose have inconsistencies
  noted above. SAM 3 appendix F.2 prose and its pmF1 table also do not support a
  simple universal "image prompts beat text" claim; that conclusion is not used.
- No inspected source tests our exact combination of these custom assets,
  training-only Helsinki support, per-frame limited camera views and full-frame
  AP50. No source tells us how another competition team obtained its score.

### Ranked experiments worth trying here

Expected effects below are directional hypotheses; **no defensible numerical
competition gain is available**. Preserve the camera and other components until
each perception change has its own manual result.

1. **Class-conditional reference scoring plus negative references on current
   proposals.** Expected effect: reduce high-ranking background mistakes and
   expose class-specific ambiguity. Reason: CNOS establishes frozen-reference
   matching; LMP adds evidence for confusable-negative prototypes. Begin with
   frozen DINOv2 references before a new Siamese training run; compare masked
   and unmasked crops. Risk: missed proposals stay missed and tiny crops remain
   information-poor. This is not the binary verifier already implemented. [S2, S5]
2. **Independent semantic/mask proposal comparison.** Expected effect: recover
   objects our YOLO does not propose, or provide cleaner crops. Compare SAM 3
   meaningful-text/exemplar proposals, or FastSAM-style class-free proposals,
   under the SAME downstream verifier. Reason: S2 and S3 use these paths, but
   both document recognition failures. Setup, checkpoint access and local
   latency must be checked before promising an API-ready implementation. [S2, S3]
3. **A genuine controlled specialist adaptation.** Expected effect: alter
   task-specific recognition without unnecessarily changing pretrained
   features. Explicit optimizer, real LR logs, matched data/schedule and
   head-only versus limited-backbone updates. Reason: TFA and RF-DETR ablations;
   our prior auto-optimizer comparison did not answer this. Do not restart
   abandoned RF-DETR without a new instruction. [S4, S6]
4. **Causal confirmation and camera ablations after the above.** Expected effect:
   trade stale/false reports against delayed recall and acquire more usable
   detail. Reason: SAM 3 temporal design and SODA/SAHI focus-on-scale methods;
   current local camera simulations are conditional and not real perception
   results. No future-frame lookahead and no fabricated L2 views. [S1, S3, S7, S8]

No serving changes, dependency installation or new training occurred in this
research pass. Runnable experiments remain subject to the user's manual score.

---

## 1. How far does AP fall with object size?

**Towards Large-Scale Small Object Detection: Survey and Benchmarks**
(arXiv:2207.14096, TPAMI 45(11):13467–13488, 2023). SODA-A: 2,513 aerial images,
872,069 instances, average resolution 4761×2777 — close to our 3840×2160. Its
"Small" objects average **14.75 px**; ours are 13.5 px at Level 0. The closest
published match to our regime.

| SODA-A, ResNet-50 | AP | AP50 | AP_eS (<12 px) | AP_Normal |
|---|---|---|---|---|
| RoI Transformer | **36.0** | 73.0 | **13.5** | 39.5 |
| Oriented R-CNN | 34.4 | 70.7 | 12.5 | 36.7 |
| Rotated RetinaNet | 26.8 | 63.4 | 9.1 | 28.2 |

**AP_eS 13.5 vs AP_Normal 39.5 — a 3× drop from size alone**, for the best
method available. Our 0.102 is not as anomalous as it first looked.

The mechanical reason, stated in the survey: for a 20×20 px box a **6 px
diagonal shift drops IoU from 100% to 32.5%**; 12 px drops it to 8.7%. At 40 px
the same shift leaves 56.6%. Localisation noise that is harmless at 60 px is
fatal at 15 px. This punishes AP@[.5:.95] far more than AP@0.50 — **our metric
is the forgiving one**, which is why we score anything at all.

### Negative results worth having

* **Deeper backbones do not help.** "Compared to ResNet-50, ResNet-101 only
  brings a slight improvement even degrades the performance." Swin-T *hurt*
  RPN-free detectors on SODA-A (Rotated RetinaNet −3.5).
* **Tiny-object label assignment is unsettled below ~12 px.** NWD is +6.7 AP on
  AI-TOD (arXiv:2110.13389) and RFLA +4.0 (arXiv:2208.08738, ECCV 2022) — but on
  SODA-D, RFLA raises overall AP 0.8 while **AP_eS drops 0.7**. At 13.5 px we sit
  exactly on the boundary where the evidence stops agreeing.

---

## 2. Slicing / tiling

**Slicing Aided Hyper Inference** (arXiv:2202.06934, ICIP 2022). Two separable
parts: sliced *inference* (SAHI) and sliced *fine-tuning* (SF).

xView validation, AP50 — the closest analogue, targets are very small:

| setup | AP50 | AP50-small |
|---|---|---|
| FCOS, full-image | 2.20 | **0.10** |
| TOOD, full-image | 2.10 | 0.10 |
| TOOD + SF + SAHI + overlap | **20.6** | **14.9** |

On VisDrone the cumulative gains are **+12.7 / +13.4 / +14.5 AP** for FCOS /
VFNet / TOOD. **The fine-tuning half is the bigger half** — for TOOD, +2.5 from
slicing at inference alone, +14.1 from SF+SAHI together. This is a *training*
change, not just an inference trick.

**When it does not help**, stated by the authors: large objects lose up to
**6.2 AP50** when slices are smaller than the objects, "caused by the false
positives predicted from slices that match large ground truth boxes." Their best
row keeps full-image inference in the NMS merge. For us that maps onto keeping
Level-0 whole-view inference — and puts `condor`, `hangar` and `large_launcher`
at risk if we do not.

*Unverified:* "The Power of Tiling for Small Object Detection" (CVPRW 2019) —
CVF serves only the abstract.

---

## 3. Copy-paste and synthetic data — the section that explains §14

### Does a detector trained over few backgrounds learn the background?

**Yes, and the canonical paper says so outright.** Dwibedi, Misra & Hebert,
**Cut, Paste and Learn** (arXiv:1708.01642, ICCV 2017):

> "even if we limit ourselves to the same type of scene, e.g., kitchens, the
> curation step **can lack diversity and create biases that do not hold in the
> test setting**."

> "Object detectors trained on hand annotated scenes **also need new negatives**
> to be able to perform well in newer scenes."

Their generator used **1,548 distinct backgrounds** for ~6,000 synthetic images.
**We used 25, all rural, against an urban evaluation scene.** That ratio is the
single clearest discrepancy between our pipeline and the method it descends
from.

### How much does background diversity matter, quantitatively?

**Training Deep Networks with Synthetic Data: Domain Randomization**
(arXiv:1804.06516, CVPRW 2018), KITTI car detection AP@0.5, ablating from 73.7:

| removed | AP | delta |
|---|---|---|
| fixed light | 67.6 | **−6.1** |
| no random object texture | 69.0 | **−4.7** |
| **4K instead of 8K textures** | 71.5 | **−2.2** |
| no data augmentation | 72.0 | −1.7 |
| no flying distractors | — | −1.1 |

**Merely halving the background pool costs 2.2 AP** at the top of the curve.
Extrapolating to 25 is not supported by the paper, but the direction is not in
doubt. The headline: 100K composites over **8K Flickr backgrounds, containing
zero real KITTI pixels**, reach **78.1 AP on real KITTI** — beating
photorealistic Virtual KITTI for R-FCN (71.5 vs 64.6) and SSD (46.3 vs 36.1).

Two further findings that bear on our recipe:

* **Dataset size saturates early** — "after only about 10K of training images
  with pretrained weights." More synthetic images is not the lever; more
  *diverse* ones is. Our 6,000 is already near the plateau.
* **Freezing early layers hurts** by as much as 13.5% (78.1 → 66.4), contra
  Hinterstoisser et al. (arXiv:1710.10710).

Their stated limitation is our failure mode in miniature: "**image context is
ignored** by our procedure, so that the structure inherent in parked cars is not
taken into account."

### Does blending realism matter?

**Blending *variety* matters; blending *quality* does not.** Cut-Paste-Learn,
GMU Kitchen mAP@0.5:

| blending | mAP |
|---|---|
| none | 65.9 |
| Gaussian blur | 68.9 |
| **Poisson (most realistic)** | **58.4** |
| all modes mixed | 72.4 |
| **same scene rendered with each mode** | **73.7** |

Poisson blending — the most physically plausible single mode — is **7.5 points
worse than no blending at all**. Rendering the same scene several times with
different blending is worth **+7.8**. "Ensuring only **patch-level realism**
provides enough training signal." Other deltas: no occlusion −10.6, no 3D
rotation −5.4, **adding distractors +2.5**.

### The disagreement, and why it does not apply to us

**Ghiasi et al., Simple Copy-Paste** (arXiv:2012.07177, CVPR 2021 Oral): "pasting
objects randomly is good enough"; +3.6 mask AP on LVIS rare categories.
**Dvornik et al.** (arXiv:1807.07428, ECCV 2018): "randomly pasting objects on
images **hurts** the performance, unless the object is placed in the right
context." **Structured Domain Randomization** (arXiv:1810.10093, ICRA 2019)
agrees with Dvornik.

These separate cleanly by regime. Dvornik and SDR augment an **already-large
in-domain real dataset**, where random context adds implausibility the model
would otherwise have learned correctly. Dwibedi, Tremblay and Ghiasi measure
**cross-domain transfer and long-tail classes**, where the alternative is
nothing. **We are unambiguously the second case**: 25 source frames, zero real
data from the target city. The context-modelling advice does not apply to us.

### The closest published analogue to our task

**Class-specific diffusion models improve military object detection in a
low-data domain** (arXiv:2604.18076, SPIE Defense + Security). 15 military
vehicle classes, **8 or 24 real images per class**, **RF-DETR** as detector.
LoRA-finetuned FLUX.1 synthetic data: **+8.0 mAP50 at 8 real samples**;
ControlNet edge conditioning adds **+4.1** more — but "**no additional benefit
when more real data is available**."

---

## 4. DETR vs YOLO at small sizes

### The 2022 case against DETR

SODA-D test, ResNet-50:

| method | AP | AP_eS |
|---|---|---|
| Cascade R-CNN (12 ep) | **31.2** | 14.1 |
| YOLOX (70 ep) | 26.7 | 13.6 |
| **Deformable DETR (50 ep)** | **19.2** | **6.3** |

Deformable DETR is **last** despite 4× the schedule: "**the sparse query paradigm
could not cover small objects adequately**."

### The 2026 case for DETR with foundation-model pretraining

**RF-DETR** (arXiv:2511.09554, ICLR 2026), on **RF100-VL** — 100 out-of-
distribution real-world datasets, the benchmark built for exactly our situation:

| model | AP | AP50 | **AP_S** |
|---|---|---|---|
| YOLOv8-N | 55.0 | 81.1 | **4.8** |
| YOLOv11-N | 55.5 | 81.3 | **4.7** |
| YOLOv11-M | 57.0 | 82.5 | **7.3** |
| YOLOv11-XL | 56.2 | 81.7 | **6.1** |
| D-FINE-N | 58.2 | 84.4 | 32.4 |
| **RF-DETR-N** | 57.8 | 85.1 | **30.1** |
| RF-DETR-2XL | 63.3 | 88.9 | 38.7 |

**A ~25 AP_S gap on out-of-distribution data, and scaling YOLO does not close
it** (nano 4.7 → x-large 6.1). We are running **YOLO11s** — squarely in the worst
cell of this table for our exact regime.

DINOv2 backbone initialisation alone is **+2.0–2.4 AP** and "significantly
improves detection accuracy on small datasets" (RF20-VL AP_S 33.8 → 37.8).

Author caveats to weigh: they could not reproduce YOLOv8/v11 mAP in TensorRT
("these models evaluate with multi-class NMS but only use single-class NMS in
inference"), so the YOLO AP_S figures may be pessimistic by an unknown amount.

Also relevant to our config: RF-DETR "limit[s] augmentations to horizontal flips
and random crops." Our stage-1 uses mosaic 1.0, mixup 0.10, shear 3.0, hsv_s
0.85 — far more aggressive than current SOTA-for-transfer.

**TinyFormer** (arXiv:2605.25046) argues both families fail for different
reasons: YOLO's "large-stride backbones may suppress tiny instances"; DETR
"reason[s] over coarse token grids, where tiny objects occupy only a few weak
tokens."

### Resolution is a nearly free lever

**Frozen High-Resolution Inference for Cross-City Object Detection**
(arXiv:2608.03136, AI City Challenge 2026 Track 6): RF-DETR-L trained at 704²,
inferred **frozen at 1120² with no parameter update**, AP **0.3272 → 0.3654
(+0.0382)**, "largest relative gain on small objects."

The same paper independently reproduces our §14 lesson: a warm-start fine-tune
raised **in-domain validation 0.767 → 0.789 while cross-city AP was worse** — "a
caution that **in-domain validation is an unreliable model-selection signal**."
Also a useful negative: gray-world normalisation did not meaningfully help.

---

## 5. Training with negative / background images

**The weakest-evidenced theme in the survey, stated plainly.** No controlled
ablation isolating "add N% pure-background images" for detection was found.

What exists: distractors are worth **+2.5 mAP** (Cut-Paste-Learn) and **+1.1 AP**
(domain randomisation). Background Mixup (arXiv:2202.13941) claims false-positive
reduction — magnitude unverified. The widely repeated **"0–10% background
images" guidance is framework folklore**; it was not found in the documentation
checked and no peer-reviewed source was located.

**Verdict: the literature does not justify another training run on this axis, and
our own `lab/exp08_fp_cost.py` caps the whole available gain at 2.6 points.** §15
is correctly superseded by §16. Distractors stay only because they are free.

---

## 6. Temporal aggregation

**MEGA** (arXiv:2003.12063, CVPR 2020), ImageNet VID:

| model | mAP |
|---|---|
| single frame | 75.4 |
| base (global + local aggregation) | 81.4 |
| **MEGA (+ long-range memory)** | **82.9** |

**+7.5 mAP**, and in the **online setting — previous frames only, our constraint
— still 81.9 (+6.5)**. Removing the global stage costs **−1.6**; that stage
"find[s] a distinct object from other frames that shares high semantic
similarity," which is the mechanism behind our `DRONE_ALTERNATES` class voting.

The published gain comes from aggregating **appearance features**, not boxes. We
already aggregate boxes, worth a great deal in our own ablation (0.8877 vs
0.5491 without). The literature says roughly 6 more points sit in evidence-level
aggregation.

Saturation and negatives: global references 5 → 20 moves mAP only 82.7 → 83.0;
adding *location* to global aggregation **hurts** (82.5 vs 82.9). No paper found
measures this on aerial imagery under a constant-homography flight.

---

## Historical Recommendations (Superseded)

The Second Scout above replaces this ranking. The following is retained as
history, not active advice or a statement of verified effects in this repo.

1. **Rebuild the background pool: 25 rural frames → thousands of diverse urban
   backgrounds.** Largest lever. Halving a pool costs −2.2 AP even at the top of
   the curve; 8K Flickr backgrounds with zero target-domain pixels reach 78.1 AP
   on real KITTI. Public overhead datasets (DOTA, VisDrone, xView, SODA-A) are
   motorways, marinas and car parks — a *third* location, so it does not
   compromise the held-out recording.
2. **Slicing-aided fine-tuning + sliced inference merged with full-image
   inference.** xView AP50 2.1 → 20.6, small-object 0.10 → 14.9. Keep Level-0
   whole-view inference in the merge or lose up to 6.2 AP50 on large classes.
3. **Move off YOLO11s to a DINOv2-pretrained DETR (RF-DETR).** The RF100-VL
   AP_S gap is ~25 points and YOLO does not scale out of it. Widest error bar on
   this list — Deformable DETR was *last* on SODA-D; the condition that separates
   the two results is internet-scale pretraining.
4. **Raise inference resolution without retraining.** Cheapest experiment here:
   +0.0382 AP cross-city from a frozen 1.6× upsample, no parameter update.
5. **Aggregate class evidence across frames.** Worth 0.0625 per class rescued
   under our macro-averaged scorer; MEGA's global stage is +1.6 alone.

### Explicitly not worth doing

* More empty/background images — published effect +1.1 to +2.5, our own ceiling
  2.6 points.
* Raising the confidence floor — measured 0.1018 → 0.1094.
* A deeper backbone — ResNet-101 ≤ ResNet-50 on both SODA benchmarks.
* NWD / RFLA label assignment — helps at 12–30 px, *loses* AP_eS below 12 px.
* Gray-world colour normalisation — reported as not meaningfully helping.

### Unverified claims

Tiling (CVPRW 2019) magnitudes; Simple Copy-Paste blending ablation; SDR's KITTI
table; Background Mixup's magnitude; the "0–10% background" rule of thumb.
