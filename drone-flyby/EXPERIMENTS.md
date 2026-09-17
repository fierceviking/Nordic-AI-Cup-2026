# Drone Flyby — Experiment Log

Running log of every approach tried, what it scored, and why.

Environment: conda env `dm` — `C:\Users\marti\miniconda3\envs\dm\python.exe` (Python 3.11.16).
Hardware: NVIDIA RTX 5070, 12 GB VRAM.

## Evidence Correction - 2026-09-17

Section 26 supersedes the causal conclusions and priorities in sections 16-25.
The ~0.295 recording metric is **recall on four provisional object labels, not
COCO mAP**, and several labels are disputed. It cannot establish that naming
is the main failure or rank complete detectors. The 2.6-point false-positive
"ceiling" is withdrawn: it depended on assumed high true-positive confidence.
The v5/v6 learning-rate comparison was also not as claimed: their logs show
`optimizer=auto` selected AdamW at 0.0005 in both cases. Historical observations
below are retained, but statements ruling out directions on these grounds are
not reliable. No competitors' training methods have been verified.

## Scoreboard

**The headline is the right-hand column.** Everything else on this page is
measured on the 25 frames the training data was cut from, and that number has
proved six times optimistic.

| # | approach | helsinki pipeline | competition | § |
|---|---|---|---|---|
| — | supplied Canny baseline | ~0 | — | 3 |
| 1 | colour saliency / greyness / MSER | 0.000 | — | 3 |
| 2 | HOG + colour + logistic regression | 0.000 | — | 5 |
| 3 | YOLO11s synthetic, L1 patrol camera | 0.802 | — | 6–7 |
| 4 | v1 detector, Level-0 camera + world model | 0.947 | — | 7 |
| 5 | v2 detector, edge-aware data | 0.960 | **0.1018** | 9, 14 |
| 6 | v4s1 + even L0/L1 loop + alternate classes | 0.936 | **0.1530** | 15, 20 |
| 7 | v5s2 (occlusion, blend variety, two-stage) | 0.945 | ~0.15 | 19 |
| 8 | v6gentle (light augmentation; requested LR overridden) | not measured here | not run | 24, 26 |

Provisional recording diagnostic (§22): micro recall at IoU 0.50 on four
hand-labelled objects, **not AP and not a validated benchmark**. The images were
not training inputs, but repeated development against them also means this is
no longer an untouched test. v6's 0.9438 is training-validation AP, not a measured
Helsinki full-pipeline result.

| model | benchmark | hangar recall | helicopter found |
|---|---|---|---|
| v2 | 0.295 | 0.789 | 0.559 |
| **v4s1** | **0.317** | 0.816 | 0.559 |
| v5s2 | 0.295 | 0.789 | 0.559 |
| v6gentle | 0.201 | 0.737 | 0.059 |

Ceilings, for scale: a *perfect* detector restricted to what the camera can see
reaches 0.95 with a world model and 0.26 without it (§2). The scorer returns
1.000 on ground truth (`local_evaluator.py --oracle`).

> ### ⚠ Read this before believing any local number
>
> **Every helsinki figure is measured on the same 25 frames the training data
> was built from.** The sprites were cut from these frames and pasted onto
> backgrounds taken from these frames. The model has seen these exact sixteen
> object instances, on this terrain, under this light.
>
> The "validation" split used during training is **synthetic, from the same
> generator and the same source frames** — only the seed differs. It is the
> generator grading itself.
>
> 0.960 locally was 0.102 in reality. §14 is that story; §16 is why; §22 is the
> benchmark built so it stops happening.

---

## 0. Problem analysis

### The task

A drone flies a straight line 600 m above a landscape, filming at 3 fps. The
source frame is 3840x2160. We never see it. We receive a 960x540 PNG of
wherever the camera is pointed, and we must answer with detections for the
**entire source frame**.

Two coupled sub-problems:

1. **Detection** — 16 classes of small, CGI-rendered military objects composited
   onto real aerial photography.
2. **Camera control** — we choose the next view (resolution level + centre),
   one move per frame, with movement limits.

Scored by **COCO mAP@IoU0.50, macro-averaged over the classes present**.

### Resolution / coverage trade-off

| Level | Source region | Coverage | Downsample | Views needed for full frame |
|---|---|---|---|---|
| 0 | 3840x2160 | 1/1 | 4x | 1 |
| 1 | 1920x1080 | 1/4 | 2x | 4 |
| 2 | 960x540 | 1/16 | 1x | 16 |

### Object sizes (Helsinki scene, mean over all annotations)

Sizes in **source pixels**, then as they appear in the transmitted 960x540 image
at each level.

| class | n boxes | src w x h | L0 (÷4) | L1 (÷2) | L2 (÷1) |
|---|---|---|---|---|---|
| ta-ta | 25 | 32 x 17 | 8 x 4 | 16 x 8 | 32 x 17 |
| small_launcher | 25 | 22 x 30 | 5 x 7 | 11 x 15 | 22 x 30 |
| jammer | 13 | 33 x 42 | 8 x 11 | 16 x 21 | 33 x 42 |
| medium_launcher | 10 | 40 x 43 | 10 x 11 | 20 x 21 | 40 x 43 |
| spacecraft | 23 | 45 x 47 | 11 x 12 | 22 x 24 | 45 x 47 |
| small_plane | 9 | 42 x 50 | 11 x 13 | 21 x 25 | 42 x 50 |
| tank | 25 | 50 x 46 | 13 x 12 | 25 x 23 | 50 x 46 |
| medium_plane | 5 | 56 x 45 | 14 x 11 | 28 x 23 | 56 x 45 |
| mine_roller | 2 | 53 x 60 | 13 x 15 | 27 x 30 | 53 x 60 |
| small_tower | 20 | 58 x 57 | 14 x 14 | 29 x 28 | 58 x 57 |
| large_tower | 19 | 62 x 62 | 16 x 16 | 31 x 31 | 62 x 62 |
| jet_plane | 22 | 77 x 79 | 19 x 20 | 39 x 40 | 77 x 79 |
| helicopter | 19 | 117 x 92 | 29 x 23 | 58 x 46 | 117 x 92 |
| large_launcher | 25 | 145 x 110 | 36 x 27 | 73 x 55 | 145 x 110 |
| hangar | 6 | 186 x 94 | 46 x 23 | 93 x 47 | 186 x 94 |
| condor | 11 | 174 x 152 | 43 x 38 | 87 x 76 | 174 x 152 |

The bottom half of that table is detectable at L0. The top half is not: `ta-ta`
is 8x4 pixels at L0 and `small_launcher` is 5x7. **Zoom is mandatory for the
small classes, and the small classes are worth exactly as much as the big ones
because the metric is macro-averaged per class.**

### Scene geometry — the decisive observation

`pose` advances by a constant `(dx, dy) = (2.412, 13.678)` metres per frame.
The ground is planar, the camera orientation is fixed, so **consecutive frames
are related by a single, constant homography**.

Objects enter at the top of the frame and flow downward:

- mean displacement per frame: `dx = +1.2 px` (radial, sign depends on side of
  the frame), `dy = +64 px`
- displacement grows from ~26 px/frame near the top to ~80 px/frame near the
  bottom — a forward-tilted (oblique) camera, not nadir
- objects also grow slightly as they descend the frame

An object is therefore visible for roughly **30–35 consecutive frames**.

This is the whole game: we do not have to re-detect an object every frame. We
detect it **once, near the top of the frame where it enters**, and then
**propagate it through the constant homography** for the remaining ~30 frames
while the camera looks somewhere else. Every frame we do that is a frame that
earns recall for free.

### Visibility per frame

8–12 of the 16 classes are visible in any given frame. Classes appear and
disappear as the drone passes over them.

### What the objects look like

Screenshots of the crops are in `lab/out/montage_*.png`. They are CGI models
composited onto real orthophotos. Some are high-contrast greys and whites
(`condor`, `spacecraft`, `ta-ta`) and pop against vegetation. Others are
camouflage green (`jammer`, `large_launcher`, `mine_roller`, `small_launcher`)
and are genuinely hard against grass and trees.

### Strategy that follows

1. A detector that runs on the transmitted view, strongest at L1/L2.
2. A **world model**: every detection is kept and warped forward each frame by
   the inter-frame homography, so the answer covers the full frame even though
   the camera only sees a fraction of it.
3. A **camera policy** that patrols the top band of the frame, where objects
   enter, so nothing is missed on the way in.

### Data budget

25 frames, 16 unique object instances, ~253 annotation boxes. That is nowhere
near enough to train a detector conventionally, so any deep-learning approach
has to be built on **synthetic composites** — cut the objects out, paste them
onto backgrounds drawn from the same scene, and simulate the L0/L1/L2
downsampling pipeline exactly.

---

## 1. Motion model — is the flight one constant homography?

`lab/motion_model.py`. Estimated `H(i -> i+1)` from ORB matches on the full
frames for all 24 consecutive pairs (~5000 RANSAC inliers each), then validated
it against the annotations.

Mean homography (normalised so `h33 = 1`), and its standard deviation across
the 24 pairs:

```
mean                                std
[[ 1.006333 -0.001266 -12.454043]   [[0.000510 0.000860 0.787193]
 [ 0.000198  1.011600  52.717630]    [0.000162 0.000967 0.767211]
 [ 0.000000 -0.000001  1.000000]]    [0.000000 0.000000 0.000000]]
```

The translation terms vary by less than 0.8 px over the whole sequence. **It is
one constant matrix.**

Accuracy of the *single mean* homography at predicting annotated object centres:

| | n | mean err | median | p90 | max |
|---|---|---|---|---|---|
| per-pair H | 243 | 2.03 px | 0.62 | 1.88 | 38.9 |
| mean H | 243 | 2.12 px | **0.73** | 1.88 | 39.7 |

There is no measurable benefit to re-estimating H per frame. The large max
errors are all boxes clipped by the frame edge.

Drift when the mean H is chained forward without any new observation:

| span (frames) | mean centre err | mean IoU vs truth | fraction IoU > 0.5 |
|---|---|---|---|
| 1 | 2.1 px | 0.918 | 0.959 |
| 3 | — | 0.859 | 0.943 |
| 5 | 5.3 px | 0.813 | 0.934 |
| 10 | 7.9 px | 0.726 | 0.855 |
| 20 | 15.2 px | — | — |

Warping the box corners also reproduces the size change: predicted/actual
width ratio `0.9999 ± 0.0125`, height ratio `1.0116 ± 0.0182`.

**Conclusion.** A detection can be carried forward ~10 frames on the motion
model alone and still clear IoU 0.50 about 85 % of the time. That is the
mechanism that makes it possible to answer for the whole frame while looking at
a quarter of it.

---

## 2. Bound study — what does the camera policy cost a perfect detector?

`lab/exp01_bounds.py`. The detector is replaced by the ground truth, restricted
to objects that are ≥ 60 % inside the current view. Every number is therefore a
**ceiling** for that camera policy, not a result.

Two detector models:

* **ideal** — no resolution limit, sees anything inside the view;
* **≥12px** — only detects an object whose shorter side is at least 12 px *in
  the transmitted 960x540 image*. This is a realistic small-object detector.

"memory" means detections persist and are warped forward each frame by the
constant homography from section 1.

| policy | ideal | ≥12px |
|---|---|---|
| L0 static, no memory | 1.0000 | 0.5557 |
| L0 static, memory | 1.0000 | 0.5718 |
| L1 top-band patrol, no memory | 0.2649 | 0.1850 |
| **L1 top-band patrol, memory** | **0.9511** | **0.6986** |
| L1 full raster, memory | 0.8544 | 0.6644 |
| L2 top-band patrol, memory | 0.8011 | 0.6527 |
| L2 full raster, memory | 0.7871 | 0.6392 |

Three things fall out of this, and they set the architecture:

1. **Memory is the single biggest lever.** Same policy, same detector:
   0.185 → 0.699. Without it, a zoomed camera can only ever answer for the
   quarter of the frame it is looking at, and the other three quarters score
   zero.
2. **L1 beats both L0 and L2.** L0 cannot resolve the small classes (0.57 with
   a realistic detector); L2 resolves everything but covers a sixteenth of the
   frame and cannot patrol fast enough to keep up. L1 is the balance point.
3. **Patrolling the top band beats rastering the whole frame.** Objects enter
   at the top and take ~30 frames to cross. Catching them on entry and then
   propagating is strictly better than sweeping everywhere and catching each
   object late.

Caveat: this scene is only 25 frames and several objects are already mid-frame
at frame 0, which a top-band patrol can never catch on entry. On the 250-frame
evaluation sequence the top-band advantage should be **larger** than it looks
here.

**Architecture decided:** L1 top-band patrol + a homography-propagated world
model + a detector that runs on the transmitted view.

---

## 3. Classical CV detection — simple features

`lab/exp02_classical.py`. Five training-free, class-agnostic proposal
generators, each scored on **class-agnostic proposal recall at IoU 0.50** with
the camera problem removed: every frame is tiled so the detector sees all of it.
Proposal recall is the right target here because it is the ceiling any
classifier bolted on top could reach.

| variant | signal |
|---|---|
| A | Canny edges + contours (the shipped baseline) |
| B | multi-scale centre–surround colour saliency in Lab |
| C | achromatic ("grey CGI") anomaly in HSV |
| D | max of saliency, greyness and local-minus-broad gradient energy |
| E | MSER |

### Results

| detector | level | proposals/frame | proposal recall@0.5 | mAP@0.50 |
|---|---|---|---|---|
| A canny baseline | L1 | 98 | 0.0425 | 0.000 |
| B colour saliency | L1 | 121 | 0.0039 | 0.000 |
| C grey anomaly | L1 | 148 | 0.0193 | 0.000 |
| D combined | L1 | 236 | 0.0000 | 0.000 |
| E MSER | L1 | 157 | 0.0116 | 0.000 |
| A canny baseline | L2 | 550 | 0.0849 | 0.000 |
| B colour saliency | L2 | 460 | 0.0000 | 0.000 |
| C grey anomaly | L2 | 543 | 0.0116 | 0.000 |
| D combined | L2 | 909 | 0.0039 | 0.000 |
| E MSER | L2 | 566 | 0.0811 | 0.000 |

### Verdict: dead end

The best variant finds 8.5 % of the objects while emitting **550 false
positives per frame**. Even a perfect classifier on top of those proposals would
top out at 0.085 mAP, and the 550 proposals/frame would have to be filtered
with essentially zero error.

Why it fails is not subtle. The background is real aerial photography of a
Finnish coastline: trees, rocks, roofs, shoreline, field boundaries, boats,
shadows. Every one of those is a compact, high-contrast, locally anomalous
region. "Different from its surroundings" describes thousands of places in every
frame and the sixteen objects are not the most different among them. Several of
the classes (`jammer`, `large_launcher`, `mine_roller`, `small_launcher`) are
camouflage green sitting on grass, which is the opposite of salient.

Harness check: `lab/check_harness.py` pushes the ground truth through the same
tiling and scoring path and returns recall 1.000 / mAP 1.000 at L0, L1 and L2,
so the numbers above are the detectors' and not the harness's.

**Conclusion.** Hand-crafted low-level features cannot separate these objects
from natural clutter. The appearance has to be learned.

---

## 4. Synthetic data — turning 25 frames into a training set

Learned detectors need data and there are 25 frames with 253 boxes. The way
around it is to treat the supplied scene as a **sprite library plus a
background library** and synthesise views.

### 4a. Sprite extraction — `lab/extract_sprites.py`

Only boxes are supplied, not masks, so the alpha has to be recovered. GrabCut
initialised from the box does it: the box is tight around a composited CGI
model, so the border is reliably background and the centre reliably foreground.

Details that mattered:

* the box is grown by 45 % so GrabCut has real background to learn from on all
  four sides;
* the central 30 % is forced foreground;
* anything outside the annotated box is erased afterwards;
* only the connected component containing the box centre is kept;
* a result covering less than 10 % or more than 98 % of the box is rejected as
  a failed matte;
* boxes touching the frame edge are skipped — they are partial objects;
* a 1 px feather stops the paste having a hard aliased border.

Every annotated box of every instance is harvested, not one per class: the same
model at 25 positions gives 25 slightly different renderings and all are useful
paste material.

**236 sprites from 253 boxes**, all 16 classes covered:

| class | kept/tried | | class | kept/tried |
|---|---|---|---|---|
| condor | 9/11 | | medium_launcher | 5/10 |
| hangar | 3/6 | | medium_plane | 4/5 |
| helicopter | 18/19 | | mine_roller | 2/2 |
| jammer | 12/13 | | small_launcher | 25/25 |
| jet_plane | 21/22 | | small_plane | 9/9 |
| large_launcher | 20/25 | | small_tower | 19/20 |
| large_tower | 18/19 | | spacecraft | 22/23 |
| ta-ta | 25/25 | | tank | 24/25 |

Contact sheet at `lab/out/sprites_preview.png`. The `hangar` matte looks like a
solid black blob; checking the raw crops (`lab/inspect_class.py hangar`) shows
that is correct — it is an unlit dark barrel-vault structure, not a matting
failure.

### 4b. Compositing — `lab/make_dataset.py`

Every generated image goes through **exactly the evaluator's imaging chain**:

```
pick a source region of size (960*f, 540*f) from a 3840x2160 frame
  -> paste augmented sprites into it at SOURCE resolution
  -> cv2.resize to 960x540 with INTER_AREA
  -> that is the training image
```

with `f = 4, 2, 1` for levels 0, 1, 2, sampled 15 % / 60 % / 25 %.

Pasting *before* the downsample is the point. It reproduces the sub-pixel blur
that makes `ta-ta` an 8x4 smudge at Level 0. Pasting crisp sprites onto an
already-downsampled background would train the model on objects that look
nothing like the ones it will be asked to find.

Real objects that fall entirely inside the chosen region keep their true
labels, so every image mixes genuine objects in genuine context with pasted
ones.

Augmentation, chosen from the physics rather than from a default recipe:

* **rotation 0–360°** — the camera looks down, so object heading is arbitrary.
* **scale 0.80–1.35 only** — altitude is fixed at 600 m in every capture, so an
  object's size on the ground is nearly fixed. Wide scale jitter would be
  actively wrong.
* **hard photometric jitter on both sprite and background** — gain, bias,
  per-channel tint, hue rotation up to ±25°, saturation 0.45–1.45x. The
  evaluation scene is a different place under different light, and background
  diversity is the real bottleneck: 25 overlapping frames cover only about
  1.7 frames' worth of unique ground.
* **post-downsample blur and sensor noise**, applied to the transmitted image,
  because that is where they happen.
* pasted sprites are rejected if they overlap an existing object.

Output: **6000 training images with 124 306 boxes, 600 validation images with
10 458 boxes**, every class between 4 682 and 10 329 instances. Preview at
`lab/out/dataset_preview_train.jpg`.

### Honest note on validation

The validation split is drawn from the same 25 frames, so it measures fit, not
generalisation to a new location. There is no honest held-out set available —
the evaluation scene is somewhere else entirely. Design choices were therefore
made to be robust rather than tuned: heavy colour augmentation, physically
motivated scale limits, and a preference for the simpler option wherever two
were close.

---

## 5. Classical machine learning — HOG + colour, linear model, sliding window

`lab/exp03_classical_ml.py`. The Dalal–Triggs rung: describe a 32x32 window
with a histogram of oriented gradients plus a coarse Lab colour layout, train a
multinomial logistic regression to separate the 16 classes from background,
slide it over a 7-level image pyramid, keep the peaks, and mine hard negatives.

Trained on patches drawn from the **same synthetic dataset the YOLO uses**, so
the two learned rungs are compared on identical data.

Two implementation notes:

* **OpenCV 5 removed `cv2.HOGDescriptor`.** It is reimplemented here with
  integral images of the nine orientation channels, which also makes the dense
  sliding window practical: 7 488 windows of a 960x540 image in 0.35 s, versus
  roughly fifty times that for a per-window loop.
* Colour statistics come from an integral image, one pass per pyramid level.

### Results

Trained on 34 162 patches (23 938 positive, 10 224 background), then two rounds
of hard-negative mining adding 900 mined false positives each.

```
round 0: train accuracy 0.4472
round 1: mined 900 false positives -> refit, train accuracy 0.4447
round 2: mined 900 false positives -> refit, train accuracy 0.4409
```

| level | proposals/frame | proposal recall@0.5 | mAP@0.50 | ms/frame |
|---|---|---|---|---|
| L1 | 286 | 0.0039 | 0.000 | 7 096 |
| L2 | 1 167 | 0.0000 | 0.000 | 35 206 |

### Verdict: also a dead end, and not a close one

**Training accuracy is 0.44 on a 17-way problem** (16 classes + background) and
it *falls* with each round of hard-negative mining. A linear model on HOG plus
colour cannot even fit the training set, let alone generalise. Hard negatives
make it worse because every negative added forces the shared linear boundary
to give up more positives.

The failure is the descriptor, not the classifier. HOG over a 32x32 window
summarises gradient orientation in sixteen 8x8 cells. At Level 1 a `ta-ta` is
16x8 px, which is two cells by one — there is no gradient structure at that
scale to histogram, and what little there is looks identical to a rock or a
roof ridge.

It is also **200x too slow**: 7 s per frame at Level 1 against a 333 ms frame
interval. A pyramid sliding window evaluating ~70 000 windows per view cannot
be made to fit the budget in any case.

Both classical rungs are now measured and both are zero. The problem needs a
detector that learns its own features at the scale the objects actually occupy.

---

## 6. Deep learning — YOLO on the synthetic views

`lab/train_yolo.py`, evaluated by `lab/eval_yolo.py` and
`lab/exp05_inference_sweep.py`.

`yolo11s`, pretrained on COCO, fine-tuned at `imgsz=960` (the transmitted image
is 960x540, so this is native resolution), batch 12, AdamW with a cosine
schedule, mosaic closed for the last 10 epochs.

Augmentation in the trainer is set from the same physics as the data
synthesis — `degrees=180` and both flips because the camera looks down and
heading is arbitrary, but `scale=0.25` only, because altitude is fixed.

### v1 — first model

15 epochs on the 6 000-image dataset. Synthetic validation reached
mAP50 0.938 / mAP50-95 0.693.

What matters is the **real** Helsinki frames. Detector measured with the camera
problem removed (every frame tiled so the detector sees all of it):

| level | proposal recall@0.5 | mAP@0.50 | ms per view |
|---|---|---|---|
| **L0 (whole frame, ÷4)** | **0.9614** | **0.9295** | ~24 |
| L1 (quarter, ÷2) | 0.9305 | 0.9039 | ~24 |
| L2 (sixteenth, ÷1) | 0.9035 | 0.8603 | ~20 |

Against the classical rungs on the same measurement:

| approach | proposal recall@0.5 | mAP@0.50 |
|---|---|---|
| Canny baseline (L2) | 0.085 | 0.000 |
| best classical CV (L2, MSER) | 0.081 | 0.000 |
| HOG + colour logistic regression | see §5 | see §5 |
| **YOLO v1, L0** | **0.961** | **0.930** |

### Inference-time sweep

The transmitted image is 960x540 whatever the level, so the only free choice is
what `imgsz` to run the detector at. Upsampling gives small objects more pixels
but does not add information.

| imgsz | proposal recall | mAP@0.50 | ms/view |
|---|---|---|---|
| **960 (native)** | **0.9305** | **0.9039** | 24 |
| 1280 | 0.9073 | 0.8817 | 25 |
| 1600 | 0.9073 | 0.8812 | 28 |

Upsampling is **worse**, not just slower. The model was trained at 960 and
matching the training resolution beats giving it interpolated pixels. Native it
is.

---

## 7. The result that inverted the plan — Level 0 beats zooming

The bound study in §2 predicted a Level-1 patrol would beat a parked Level-0
camera, on the assumption that a detector needs roughly 12 px to find an object
and `ta-ta` is 8x4 px at Level 0. **That assumption was wrong.** The trained
detector finds the small classes at Level 0 anyway.

Full pipeline, v1 detector, `lab/eval_pipeline.py`:

| camera policy | mAP@0.50 |
|---|---|
| **L0 static** | **0.9475** |
| L0 x3 : L1 x1 (quadrant cycle) | 0.9363 |
| L0/L1 alternating, top band only | 0.9357 |
| L0 static, **no world model** | 0.9269 |
| L0/L1 alternating quadrants | 0.9260 |
| L0/L1 alternating quadrants, prior H only | 0.9258 |
| L0/L1 alternating quadrants, keep view-edge detections | 0.9217 |
| L1 top-band patrol | 0.8020 |
| L1 quadrant raster | 0.7965 |
| L2 top-band patrol | 0.6341 |
| L0/L1 alternating quadrants, **no world model** | 0.5833 |

Why zooming loses, now that the detector is real rather than assumed:

1. **A zoomed camera answers for a fraction of the frame.** Every frame it is
   at Level 1, three quarters of the frame is a propagated guess rather than an
   observation. Level 0 observes all of it, every frame.
2. **Tiling costs accuracy of its own.** An object straddling a tile boundary
   is cut in half in both tiles, and four inferences produce four times the
   opportunities for a false positive. That shows up directly in §6: the same
   detector scores 0.930 at L0 and 0.904 tiled at L1.
3. **The downsample is gentler than it looks.** INTER_AREA over a 4x4 box is
   box-averaging, not decimation — an 8x4 px `ta-ta` at Level 0 still carries
   its silhouette, and a convolutional detector trained on exactly that
   degradation can use it.

### Ablations worth keeping

* **World model still earns its place.** Even with the camera parked at L0 and
  seeing everything, it is worth +0.021 (0.9475 vs 0.9269): it smooths over
  frames where the detector blinks, and it carries a confident identity across
  frames instead of re-deciding the class each time. When the camera *does*
  zoom, it is worth +0.343 (0.9260 vs 0.5833).
* **Online motion estimation helps, but only in the right form.** A full
  eight-parameter homography re-fitted from ORB matches made things *worse*
  (0.6887 vs 0.7204 in an early L1 run) because two consecutive views overlap
  in only part of the frame and a homography fitted to a strip extrapolates
  badly across the rest. Replacing it with a **two-parameter residual
  translation on top of the measured prior** flipped it to a gain
  (0.7431 vs 0.7204). Fewer degrees of freedom, fitted to the same strip,
  extrapolate safely.
* **Dropping detections that touch a view edge** is worth +0.004 in the hybrid
  (0.9260 vs 0.9217). A box cut off by the crop is not the box being scored —
  unless the crop edge is also a frame edge, where the ground truth is clipped
  too. That exception is what makes the rule safe to apply.

### Which policy to ship

`L0 static` is the best measured number, but the measurement is on the scene
the sprites were cut from: the model has seen these exact 16 instances, so its
Level-0 performance here is optimistic in a way that will not transfer.

`L0 x3 : L1 x1` costs **0.011** locally and buys a genuine 2x-linear-resolution
look at each quadrant roughly every 16 frames — about twice per object crossing.
That is real information the model cannot have overfitted away, and the world
model keeps whatever it finds. With one evaluation attempt on an unseen scene,
that is worth 1 % of a local score, so that is the shipped default.

---

## 8. Latency — a two-second bug that would have cost the attempt

`lab/profile_request.py`, `lab/probe_latency.py`, `lab/probe_sequence.py`.

Frames are emitted every 333 ms and only the newest one is ever sent, so a slow
answer does not just risk the 3333 ms timeout — it silently deletes every frame
that goes by while it is being computed, and each of those is scored as a frame
with no detections.

### Where the time goes

| stage | ms |
|---|---|
| `json.loads` of the 1 384 KB body | 1.6 |
| pydantic validation | 0.01 |
| base64 + PNG decode | 11.3 |
| YOLO inference | 16.5 |
| ORB motion residual | ~4 |
| world model update + report | 0.7 |
| **`example.predict`, everything** | **33** |

### The bug

Probing the live endpoint with identical repeated requests:

```
request 0:   2192.2 ms
request 1:    167.4 ms
request 2:     65.3 ms
request 3+:    ~55 ms
```

The model *was* warmed up — three blank inferences at import. But the server
answers requests on a **uvicorn worker thread**, and the first CUDA call on a
new thread costs about two seconds. Warming up on the importing thread warms
the wrong thread.

At 3 fps that first request costs roughly seven frames, and on a slower host it
would breach the 3333 ms budget outright. On a one-attempt evaluation that is
an expensive way to find out.

**Fix:** inference runs on a single dedicated `ThreadPoolExecutor(max_workers=1)`
thread, and the warm-up is submitted to that same executor during construction.
Every request is then served by an already-warm thread. It also serialises
access to the single GPU.

```
request 0:     42.2 ms   <- was 2192
request 1:    134.3 ms
request 2+:    ~50 ms
```

### After the fix

Every frame of the scene, timed from the socket (`lab/probe_sequence.py`):

```
server round trip: median 58 ms   p90 76 ms   max 84 ms
harness serialise: median  5 ms   max  9 ms
```

`local_evaluator.py` still reports `max 2109 ms`, on **frame index 0 only**.
That is the harness's own first `requests.Session` POST — this probe opens the
connection before timing and sees 43 ms for the same frame. It is client-side,
and the evaluation service's HTTP client will be warm and pooled.

At ~58 ms per frame against a 333 ms interval there is roughly **5x headroom**,
so there is budget available if a heavier model turns out to be worth it.

---

## 9. v2 — teaching the detector about partially visible objects

`medium_launcher` scored **0.000** with v1, at detector level, with a proposal
recall of zero. Inspecting it (`lab/inspect_class.py medium_launcher`) explains
why: it sits against the right-hand edge of the frame and **5 of its 10
annotated boxes are clipped by it**, at x = 3839.

The v1 dataset could not have taught the model to find it:

* `extract_sprites.py` skips boxes touching the frame edge, so only 5 of its 10
  boxes produced sprites;
* `make_dataset.py` labelled only objects that fell **entirely** inside the
  chosen region, so a clipped object was silently turned into an unlabelled
  one — an object present in the image with no box on it, which actively
  teaches the detector to ignore it.

That second point is not specific to `medium_launcher`. **Every** object enters
through the top edge of the source frame and is clipped for its first few
frames, and those are exactly the frames where a detection is most valuable,
because the world model then carries it for the next thirty.

### Changes for v2

1. Real objects overlapping the region are **labelled and clipped** when at
   least 35 % of their area is visible, instead of being dropped.
2. Pasted sprites are deliberately hung off the canvas edge 18 % of the time,
   subject to the same 35 % visibility rule, measured against the sprite's own
   opaque area rather than its rotation-padded bounding box.
3. Background flips (horizontal and vertical) were added, but only on canvases
   with no real labelled objects in them — flipping would invalidate the real
   boxes.
4. Background colour jitter was pulled back (hue ±16° rather than ±25°) after
   the preview showed purple foliage, which spends model capacity on terrain
   that cannot occur.
5. 7 000 / 700 images instead of 6 000 / 600.

This pairs with a matching **inference-side rule**: a detection touching a view
edge that is *not* a frame edge is discarded, because there the ground truth is
not clipped and a half-box would poison the world model. Where the view edge
*is* the frame edge, the clipped box is correct and is kept.

### Results — v2 wins, once it is actually trained

The first read of v2 was **wrong**, and the way it was wrong is worth keeping.

Stopped at epoch 17 of 45 (a dataloader ran out of memory on a shared machine),
v2 scored **0.8894** on the real frames against v1's 0.9295, with `ta-ta`
collapsing from 0.922 to 0.340. The obvious explanation was that labelling
clipped objects had created a population of degenerate two- and three-pixel
boxes that poisoned the smallest classes.

`lab/compare_label_sizes.py` tested that directly and **refuted it**:

| shorter side | v1 dataset | v2 dataset |
|---|---|---|
| < 4 px | 1.36 % | 1.46 % |
| < 6 px | 5.62 % | 5.75 % |
| < 8 px | 10.65 % | 10.75 % |
| < 12 px | 26.32 % | 26.75 % |

Near-identical, per class as well as overall. The clipped labels were not the
problem — v2 had simply seen 29 % fewer samples (3 739 x 17 vs 6 000 x 15) and
was mid-cosine-schedule. Training it out to epoch 45 reversed the result
entirely.

**Final, both on the real Helsinki frames, Level 0:**

| class | v1 (15 ep) | v2 (45 ep) |
|---|---|---|
| condor | 0.901 | **1.000** |
| hangar | 0.832 | 0.832 |
| helicopter | 1.000 | 1.000 |
| jammer | 0.994 | **1.000** |
| jet_plane | 0.950 | **0.985** |
| large_launcher | 1.000 | 1.000 |
| large_tower | **0.992** | 0.941 |
| **medium_launcher** | 0.563 | **0.693** |
| medium_plane | 1.000 | 1.000 |
| mine_roller | 0.835 | **1.000** |
| small_launcher | 0.994 | **1.000** |
| small_plane | 0.978 | **1.000** |
| small_tower | 1.000 | 1.000 |
| spacecraft | 0.950 | 0.950 |
| ta-ta | 0.922 | **1.000** |
| tank | 0.960 | 0.960 |
| **mAP@0.50** | **0.9295** | **0.9600** |
| proposal recall | 0.9614 | **0.9730** |

`medium_launcher` — the class the edge fix was built for, and the only one that
is clipped by the frame edge in half its appearances — went from 0.563 to
0.693. That is the change doing what it was designed to do.

Full pipeline with the v2 detector:

| camera policy | mAP@0.50 |
|---|---|
| L0 static | 0.9650 |
| L0 static, no world model | 0.9600 |
| **L0 x3 : L1 x1 (shipped)** | **0.9599** |
| L0/L1 alternating, top band | 0.9577 |
| L0/L1 alternating quadrants | 0.9461 |
| L1 top-band patrol | 0.8007 |
| L2 top-band patrol | 0.6368 |
| L0/L1 alternating, no world model | 0.5941 |

Latency fell to 22.8 ms mean. **v2 is promoted to `model/best.pt`.**

The lesson: a mid-schedule checkpoint is not evidence about a data change. The
first comparison was between epoch 17 of one run and epoch 15 of another, and
it pointed the wrong way with a plausible-sounding story attached.

---

## 10. Run-time parameter sweep — and why almost nothing was tuned

`lab/exp06_tuning.py`. Five run-time constants, none of which need retraining,
swept one at a time on the full pipeline with the v1 detector.

| parameter | values | mAP range |
|---|---|---|
| detector confidence floor | 0.03 … 0.30 | 0.9382 – 0.9414 |
| how long an unobserved track is kept | 10 … 80 frames | 0.9414 (identical) |
| misses tolerated while in view | 1 … 5 | 0.9392 – 0.9439 |
| association IoU gate | 0.15 … 0.45 | 0.9414 (identical) |
| confidence decay per unobserved frame | 0.0 … 0.10 | 0.9413 – 0.9416 |

**The whole surface is flat**: 0.938 to 0.944, a spread of 0.6 %, across every
parameter. Two of the five make no difference at all.

That is a consequence of the Level-0 policy. When the detector observes the
whole frame every frame, the world model has very little work left to do —
tracks are re-observed constantly, so how long they survive unobserved and how
their confidence decays barely enters the answer.

The scene has 16 instances. A 0.25 % difference on 16 instances is one
borderline box. **Tuning on that would be fitting noise**, so the values shipped
were chosen for robustness rather than for the local optimum:

* `DRONE_CONF = 0.05` — 0.9414 against a best of 0.9414. A low floor costs
  almost nothing in a globally ranked metric and buys recall if the detector is
  less confident on an unfamiliar scene.
* `MAX_MISSES_IN_VIEW = 3` — 0.9414 against 0.9439 for a value of 1. The
  0.0025 was given up deliberately: a tolerant setting keeps tracks alive
  through detector blinks, and blinks are exactly what a new location will
  produce more of.

---

## 11. Correctness checks

Three things are asserted rather than assumed, because each is a failure that
would be invisible in a score and expensive in an attempt.

| check | what it proves |
|---|---|
| `local_evaluator.py --oracle` | the scorer and the data agree: prints 1.000 |
| `lab/check_harness.py` | ground truth through the tiling harness returns recall 1.000 / mAP 1.000 at L0, L1 and L2, so a low detector score is the detector |
| `lab/check_camera_policy.py` | every camera loop, driven for 500 frames against the evaluator's own `Camera`, emits **0 refused commands** and only strict `int` fields |

The 500-frame policy check matters because the supplied scene is 25 frames and
an attempt is 250: a bug that only appears after the loop wraps, or after a
command is refused, would never show up locally. It also confirms the shipped
loop's duty cycle is what it is supposed to be — 75 % of frames at Level 0 and
6.2–6.4 % at each of the four Level-1 quadrants.

Two bugs were found this way and fixed:

* new tracks were charged with a "miss" on the frame they were created,
  because the miss loop ran after the append;
* state survived across attempts. `local_evaluator.py` uses the fixed
  `sequence_id` of `local`, so a second run inherited the first run's tracks
  and scored 0.897 instead of 0.945. Both the world model and the motion
  estimator now reset when the frame number goes backwards.

---

## 12. What was learned

**The ladder was worth climbing in order.** Hand-crafted features and a linear
model on HOG both returned exactly 0.000, and they returned it for
*different* reasons — saliency cannot separate CGI objects from a coastline
full of rocks and roofs, while HOG over a 32x32 window has nothing to describe
when the object is 16x8 px. Knowing both of those is what justifies the
synthetic-data effort that followed, rather than it being an assumption.

**The biggest win was not the detector.** It was noticing that the flight is a
single constant homography and building a world model on it. That is worth
+0.34 mAP when the camera zooms, and it is the difference between answering for
a quarter of the frame and answering for all of it.

**The second biggest win was a two-second thread bug.** Warming the model on
the importing thread does nothing for the worker thread that actually serves
requests. On an evaluation that emits a frame every 333 ms and allows one
attempt, that alone would have cost seven frames and risked a recorded error.
It was only visible because latency was probed per-request instead of as a mean.

**The stated framing of the problem was a hypothesis, not a fact.** The use
case is built around a zoom/coverage trade-off, and the bound study in §2 —
using a plausible 12 px detectability assumption — agreed that zooming should
win. The trained detector then read 8x4 px objects out of the full view anyway,
and parking at Level 0 beat every patrol by 0.15 mAP. The assumption was
reasonable and it was wrong, and the only reason that was caught is that the
policy was measured rather than assumed.

**Almost nothing else mattered.** Five run-time parameters swept across
sensible ranges moved the score by 0.6 % total. On 16 instances that is noise,
and the shipped values were chosen for robustness instead.

## Caveats, stated plainly

* **The local score is optimistic.** The detector's sprites were cut from these
  25 frames, so it has seen these exact 16 object instances, under this light,
  on this terrain. The evaluation scene is a different location.
* **Background diversity is the weakest link.** 25 heavily overlapping frames
  cover roughly 1.7 frames' worth of unique ground. Colour and flip
  augmentation mitigates it; it does not solve it.
* **There is no honest validation set.** Everything reported here is measured
  on the same 25 frames, including the synthetic validation split. Where a
  choice was close, the more conservative option was taken rather than the
  higher-scoring one — the Level-0-vs-hybrid camera policy is the clearest
  example.

## What to try next, in order of expected value

> Superseded — written when the local 0.960 was still believed. Kept because
> the reasoning is instructive: four of these five were tried and none moved
> the held-out benchmark (§24). The one that mattered, class evidence, is not
> on the list at all. The current priorities are at the end of §24.

1. **A larger detector.** The server answers in ~58 ms against a 333 ms frame
   interval, so there is roughly 5x latency headroom sitting unused.
   `yolo11m` or `yolo11l` at the same 960 input costs little of it.
2. **Longer training.** v1 was stopped at 15 epochs and v2 at 45 because the
   machine was shared. Neither had plateaued.
3. **Better mattes.** GrabCut leaves a halo of original terrain on several
   sprites, which is a cue the detector can learn and that will not exist in
   the evaluation scene. A segmentation model would cut them cleanly.
4. **An adaptive camera policy.** The shipped loop is fixed. Scoring each
   candidate view by the staleness and size of the tracks it would refresh is
   the obvious upgrade — though on the evidence of §7, expected gain is small
   while Level 0 dominates. *(Tried in §18: 0.7914 against 0.8877 for a blind
   raster. Coverage beats refinement while recall binds.)*
5. **Re-run the §2 bound study with the real detector's per-class detection
   rate at each level**, instead of the 12 px assumption, to see whether any
   class is genuinely zoom-only. That would turn the Level-0-versus-hybrid
   decision from a judgement call into a measurement.


---

## 13. Deployment — reaching the evaluation service

The service calls *us*, from the internet, so the endpoint has to be publicly
reachable. `lab/cup_api.py` wraps the competition API; the key is read from
`NORDIC_API_KEY` or a git-ignored `.nordic_key` file and is never printed.

The API (`https://cases.nordicaicup.com/api/openapi.json`, auth is the
`x-token` header) has **no endpoint that serves the validation images**. The
only way to obtain them is to record them as the service pushes them, which the
use case explicitly permits. `recorder.py` does that: enabled by
`DRONE_RECORD_DIR`, it writes each transmitted view as a PNG plus a JSON
sidecar with the geometry and our own answer, on a background thread with a
bounded queue so disk I/O can never cost a frame. Verified: 25/25 frames
captured with no change to score (0.941) or median latency (47 ms).

That recording is the most valuable thing available for the weakness identified
in §4 — a 249-frame sequence over a *different location* is exactly the
background diversity the 25 Helsinki frames cannot provide.

### Tunnel latency — measured, and a problem

Exposed the local server with a Cloudflare quick tunnel and measured the round
trip from outside (`lab/probe_tunnel.py`, which pins DNS because this machine's
router would not resolve a freshly created `trycloudflare.com` subdomain):

| leg | median |
|---|---|
| our server, measured on loopback | 39 ms |
| **through the Cloudflare quick tunnel** | **374 ms** |

The frame interval is 333 ms, so **11 of 12 requests were slower than one
frame** — roughly every second frame would be dropped, each one scored as a
frame with no detections.

Our own code is 39 ms of that; the tunnel is the other 335 ms. The measurement
is pessimistic — it is a hairpin that pushes the 1.4 MB body *up* a domestic
uplink to Frankfurt and back, whereas a real attempt sends it *down* to us from
the service — but the margin is not comfortable. **A cloud VM near the service,
as the use case recommends, is the right deployment, not a tunnel from a
workstation.**

### A failure mode worth knowing about

Partway through this work the server started answering every request in 4 ms
with **HTTP 200 and zero detections**. The detector had failed to load, and
nothing in the protocol distinguishes that from a model that simply finds
nothing. On a one-attempt evaluation it would look like a bad model rather than
a broken deployment.

`api.py` now has a `/health` route reporting `solver_loaded`, and logs a loud
banner at startup if the detector is missing. **Check `/health` before an
attempt.**

---

## 14. The real validation set — 0.960 locally, 0.102 in reality

Ran an actual validation attempt against the competition service and recorded
every frame it sent.

| | mAP@0.50 |
|---|---|
| synthetic validation split (the generator grading itself) | 0.954 |
| real Helsinki frames (training scene) | 0.960 |
| **competition validation set (held out, different location)** | **0.102** |

**That is the honest number, and it is a ninth of the local one.** Everything
reported in sections 6–12 is measured on data the model was built from, and the
gap between 0.96 and 0.10 is the size of that self-deception. The two prior
attempts on this team's account both scored 0.000, so 0.102 is at least
movement, but the local figures were never a prediction of anything.

### It was not a timing problem

The first suspicion was dropped frames — §13 measured 374 ms through the tunnel
against a 333 ms interval. The recording settles it:

```
248 frames recorded, indices 0..248, 1 skipped
camera levels: L0 180 (73%), L1 68 (27%)   <- the shipped loop, exactly as designed
```

One frame lost out of 249. The hairpin latency measurement was pessimistic, as
suspected: the real path sends the 1.4 MB *down* to us rather than up. Timing is
fine. The detector is the problem.

### What went wrong, from the recording

```
predictions    mean 93.1 per frame, min 29, max 147
confidence     median 0.107, p90 0.359
most predicted ta-ta 4742, jammer 4128, small_launcher 2188, mine_roller 1929
```

**93 predictions per frame**, where Helsinki averaged about 30 *including* the
world model's accumulated tracks. Drawing them onto the recorded views
(`lab/inspect_recording.py`) shows what they are: the validation scene is a
Danish coastal city — motorways, a marina full of moored yachts, industrial
estates, car parks, terraced housing with tiled roofs — and the detector is
boxing **boats, cars, greenhouses and rooftop plant**.

The real CGI objects are present and are being found: the black `hangar` model
is clearly detected in several frames at 0.84 confidence. They are simply
buried under a hundred false positives, and mAP is a precision-weighted
measure.

### Why: the training backgrounds had no man-made clutter

The 25 Helsinki frames are forest, field, water and a handful of buildings.
Nothing in them looks like a compact, bright, geometric man-made object except
the sixteen pasted targets. So "compact man-made thing on terrain" was a
*perfect* discriminator on the training distribution, and that is what the
detector learned. It is the cheapest hypothesis that fits the data, and the
data never punished it.

Put that detector over a marina and every hull matches.

This is the failure that the §4 note about background diversity was pointing
at, but the note said "mitigated, not solved" and the local score of 0.96 made
it easy to stop worrying. The synthetic validation split could never have
caught it: it is drawn from the same 25 frames.

### The cheap lever does not work

Raising the detector's confidence floor from 0.05 to 0.25 moved the score from
0.1018 to 0.1094. The false positives are not low-confidence noise — they are
confident detections of real objects that happen not to be targets. A threshold
cannot separate them.

### The tempting fix, and why it was rejected

The obvious move is to composite sprites onto the 248 recorded frames. The
rules permit keeping them, and they contain exactly the clutter that fools the
detector. It was built, and then **thrown away**.

Training on them destroys the only held-out measurement available. The next
validation score would no longer mean anything, and the failure this whole
section describes is precisely what happens when you optimise against data you
also trained on. With one evaluation attempt, an honest gauge is worth more
than a flattering number — and the evaluation sequence is a *third* location,
so fitting to this one city is not obviously useful anyway.

**The recordings stay a test set.** `lab/recordings/` is never a training
input. `make_dataset.py` has no option to make it one.

---

## 15. Two-stage training with real negatives

The fix has to come from the generator. Two things were missing, both free, and
both visible only in hindsight:

1. **Not one training image contained zero objects.** Every generated image had
   between three and eleven. The detector was never once shown that "nothing
   here" is a valid answer, so on unfamiliar ground it invents objects.
2. **No hard negatives.** Nothing man-made that is not a target.

### The recipe

Borrowed from a few-shot segmentation approach that beat a GAN-based one on the
same kind of data-starved problem: **train on positives together with pure
negatives, then fine-tune on positives alone.** Stage 1 spends capacity on
learning what to reject; stage 2 spends what is left on localising and
classifying.

| | stage 1 | stage 2 |
|---|---|---|
| data | positives + background images | positives only |
| lr0 | 0.01 | 0.0002 |
| warmup | 3 epochs | none |
| mosaic / mixup | 1.0 / 0.10 | 0.4 / 0 |
| scale, translate, shear | 0.30, 0.20, 3.0 | 0.20, 0.10, 1.0 |
| hsv h/s/v | 0.035 / 0.85 / 0.55 | 0.020 / 0.60 / 0.40 |

`lab/make_stage2.py` writes the stage-2 view as an image list plus a matching
`data.yaml`, so nothing is duplicated on disk.

### Generating negatives is harder than it sounds

A first attempt asked for 30 % background images and produced **1 in 60**. Two
reasons, both obvious afterwards:

* a Level-0 view is the *entire* frame, which always contains objects, so it
  can never be a background image;
* a Level-1 crop is a quarter of a frame holding eight to twelve annotated
  objects, so a random one nearly always catches something.

Background crops have to be *sought*: sample offsets until none of the frame's
annotations overlap, and draw them from Level 1 and Level 2 only (weighted
0.25 / 0.75). That raised the yield to about 20 %.

### Distractors

Flat-shaded geometric shapes — elongated hulls, boxes with a darker rim,
L-shapes, discs — in metallic greys, whites, navy and dark tones, at the same
source-pixel sizes as the real targets, rotated arbitrarily, pasted through the
same imaging chain and deliberately **left unlabelled**. Zero to seven per
image.

They are crude next to a real yacht. What they teach is the part that matters:
being a compact, bright, geometric thing is not sufficient to be a target.

### Checking the labels actually follow the pixels

`lab/check_labels.py`. Flipping a canvas without flipping its boxes is a silent
corruption — training still runs and the loss still falls. The check rebuilds
samples and verifies every labelled box has materially more local contrast
inside it than in a ring around it, which a mirrored box fails immediately.

```
boxes checked        948
suspicious boxes     12 (1.3%)
images with no boxes 12/60
```

1.3 % is the low-contrast classes (`hangar` is a near-black shape on dark
ground). Geometry is being applied to labels correctly — both in the generator,
where the background flip only happens on canvases with no real objects and
sprites are pasted after it, and in the trainer, where Ultralytics transforms
labels with the image.

### Results

| | stage 1 | stage 2 |
|---|---|---|
| training validation mAP50 | 0.9385 | 0.9599 |
| helsinki, full pipeline | 0.9357 | 0.9432 |
| competition | **0.1530** | ~0.15 |
| held-out benchmark (§22) | **0.317** | 0.295 |

Stage 2 improved every local number and improved nothing that mattered. Note
that the two stages were validated against *different* sets — stage 1 on 125
real Helsinki views, stage 2 on the synthetic split — so even the local
comparison is not like-for-like.

v4s1 remains the best model this project has produced, on both the competition
and the benchmark. It was not beaten by v5 (more augmentation), by v5s2 (a
second fine-tuning stage), or by v6gentle (preserved pretrained features).

The recipe is sound and the literature supports it (§17), but the axis it
targets — false positives — is worth 2.6 points in total (§16). It was aimed at
the wrong thing, and that was knowable before the run: the measurement in
`exp08_fp_cost.py` took minutes and came after.


---

## 16. Reading the scorer — and discovering we optimised the wrong thing

Section 15 was built on an assumption that had never been measured: that the
**41 detections per frame** on the validation sequence were what destroyed the
score. Before spending another training run on it, the assumption was tested.

### What the scorer actually does

```python
evaluated_classes = tuple(
    object_id for object_id in OBJECT_CLASSES if object_id in present_classes
)
...
return _clamp(sum(ap_by_class.values()) / len(ap_by_class)), ap_by_class
```

Two consequences, both load-bearing:

* **Classes absent from the ground truth are never evaluated.** Predictions for
  them are discarded before scoring. They are free.
* **The result is an unweighted mean over the classes that *are* present.** A
  class we never get right contributes a hard zero to that mean, no matter how
  well the others do.

Helsinki contains **all sixteen classes**, roughly one object each, most of them
visible in most frames. So each class is worth `1/16 = 0.0625`, and our 0.102 is

```
0.102 / 0.0625 = 1.6 classes out of 16
```

We are getting one and a half classes right and scoring zero on the rest. That
matches the manual triage exactly: `hangar` at 0.92, `jet_plane` at 0.75, and
little else.

### How much do false positives really cost? (`lab/exp08_fp_cost.py`)

Helsinki's labels are perfect, so start from an oracle that scores 1.000 and
inject false positives drawn from the classes, confidences and box sizes we
actually produced on the validation sequence. Two ranking regimes, because COCO
AP is order-sensitive.

| injected fp/frame | mAP (oracle conf 1.0) | mAP (realistic conf) |
|---|---|---|
| 0 | 1.0000 | 1.0000 |
| 5 | 1.0000 | 1.0000 |
| 10 | 1.0000 | 0.9861 |
| 20 | 1.0000 | 0.9820 |
| **41** | 1.0000 | **0.9740** |

**Our entire false-positive load costs about 2.6 points.** The reason is in the
recorded confidences: median **0.107**, p90 **0.359**. AP integrates precision
over recall, and boxes that rank below every true positive barely touch the
curve.

So the clutter monitor was measuring a quantity worth ~2.6 points while we were
losing ~90, and the v4 generator changes — empty images, distractors — target
that same near-irrelevant axis. They are not harmful. They are close to
pointless. The failure is **recall and class accuracy**, not precision.

### Why the classes are wrong: there are not enough pixels

| | source px | at Level 0 |
|---|---|---|
| p10 object | 30 | 7.5 |
| median object | 54 | **13.5** |
| p90 object | 151 | 37.8 |

At Level 0 the median object is **13.5 pixels across**, and from that we are
asking for a sixteen-way class label on terrain the model has never seen.
Helsinki is forest, water, ploughed fields and a few scattered houses; the
validation sequence is motorways, marinas, industrial estates and car parks.

### Consequence 1 — name the runners-up

Since a wrong low-confidence box is nearly free and a class scoring zero costs
0.0625, each track now also reports its second and third most-voted classes at
damped confidence (`DRONE_ALTERNATES`, default 2). Measured on the terrain
holdout model: **0.8477 → 0.8475**. The cost is real and negligible; the upside
is unproven but large, and it is bought with the world model's accumulated
class votes rather than a single ambiguous frame.

### Consequence 2 — the camera policy was tuned on the wrong scene

Section 7 concluded that Level 0 beats zooming. That was measured with a
detector trained on sprites cut from the very scene it was tested on. Repeating
the comparison with the **terrain-holdout** model — one that has to generalise,
which is the regime the evaluation puts us in — inverts the ranking:

| loop | holdout model | production model (in-domain) |
|---|---|---|
| **L0/L1 alternating quadrants** | **0.8877** | 0.9460 |
| `L0 x3 : L1 x1` (was shipped) | 0.8624 | 0.9600 |
| L0 static | 0.8475 | 0.9650 |
| L1 quadrant raster | 0.8223 | — |
| L2 top-band patrol | 0.6417 | — |
| L0/L1 alternating, **no world model** | 0.5491 | 0.5941 |

Even alternation now ships: **+0.025** against the previous loop under
generalisation pressure, for **−0.014** in-domain. Given the in-domain number
has already been shown to be nine times optimistic, that is the right side of
the trade. The last row is why the zoom is affordable at all — the world model
carries coverage across the frames the camera spends elsewhere, and without it
the whole approach collapses.

The general lesson, stated once: **when the local metric is known to be
inflated, tune against the degraded proxy, not the inflated number.**

---

## 17. What the literature says

Full survey with citations in [LITERATURE.md](LITERATURE.md), produced by the
`Literature Scout` agent in `.github/agents/`. The four findings that change
what we should do next:

1. **Our background pool is the outlier.** Cut-Paste-Learn (arXiv:1708.01642),
   the method this generator descends from, used **1,548 distinct backgrounds**
   for ~6,000 composites and warned that restricting to one scene type "can
   create biases that do not hold in the test setting." We used **25, all
   rural**. Domain randomisation (arXiv:1804.06516) measures **−2.2 AP for
   merely halving** a background pool from 8K to 4K, and shows 8K Flickr
   backgrounds with *zero* target-domain pixels training a detector to 78.1 AP
   on real KITTI. Same paper: synthetic dataset size **saturates around 10K
   images** — our 6,000 is already near the plateau, so generating *more* is
   not the lever, generating *more varied* is.

2. **0.102 is less anomalous than it looked.** On SODA-A (arXiv:2207.14096,
   TPAMI 2023), whose "small" objects average 14.75 px against our 13.5 px, the
   best method scores **AP 13.5 on the smallest objects versus 39.5 on normal
   ones — a 3× drop from size alone.**

3. **We are running the worst architecture for this regime.** On RF100-VL, 100
   out-of-distribution datasets, **every YOLO scores AP_S 4.7–7.3 while every
   DETR scores 30–39**, and scaling YOLO does not close the gap (arXiv:2511.09554).
   Counter-evidence, held honestly: Deformable DETR came **last** on SODA-D
   (AP_eS 6.3 vs Cascade R-CNN's 14.1). The separating condition appears to be
   internet-scale pretraining, which is also worth +2.0–2.4 AP on its own.

4. **Section 15 is superseded, and the literature agrees with §16.** Distractors
   are worth **+1.1 to +2.5 AP** in published ablations — the same order as the
   2.6-point ceiling `exp08_fp_cost.py` measured. No controlled ablation for
   "add N% empty images" could be found at all; the familiar "0–10% background
   images" guidance is folklore with no located source.

One independent confirmation worth recording: a cross-city detection study
(arXiv:2608.03136) reports a fine-tune whose **in-domain validation rose 0.767 →
0.789 while its cross-city score fell**, and concludes that in-domain validation
is an unreliable selection signal. That is §14 reproduced by someone else.

---

## 18. Two ideas from the literature that did not survive contact

Gathering outside data is permitted by the rules, but last year's winners used
only the supplied data, so the plan is to exhaust what the 25 frames can give
before importing anything. That rules out the survey's top recommendation
(thousands of urban backgrounds) for now and promotes the two cheapest
data-free levers. Both were tested on the terrain-holdout model, which is the
only proxy here that has to generalise. **Both lost.**

### Inference-time upsampling — rejected

The survey's cheapest suggestion: a frozen RF-DETR gained +0.038 AP cross-city
purely by inferring at 1.6× its training resolution (arXiv:2608.03136), and
SODA's own benchmark protocol upsamples 800×800 crops to 1200×1200.

| `imgsz` | holdout mAP |
|---|---|
| **960 (native, as trained)** | **0.8877** |
| 1280 | 0.8813 |
| 1536 | 0.8726 |

Monotonically worse. The published gain came from RF-DETR, whose resolution is
NAS-tuned and which trains across scales; our YOLO is trained at a fixed 960 on
960×540 views, so raising `imgsz` only introduces a train/test scale mismatch.
**A result borrowed from another architecture is not evidence about ours.**

This also disposes of SAHI-style slicing of the *transmitted* view: slicing a
960×540 image adds no pixels that were not already there. Only the camera can
buy real resolution, which is what Level 1 already does.

### Attention-driven camera — rejected

Since the failure is class accuracy, and the score is a mean over classes, a
zoom frame arguably belongs on whichever object we cannot yet *name* rather than
on whichever quadrant is next in a blind raster. `AttentionPolicy` picks the
in-frame track whose leading two class votes are closest, breaking ties toward
tracks with fewest observations.

| policy | holdout mAP |
|---|---|
| **fixed L0/L1 alternating raster** | **0.8877** |
| attention, re-target every zoom frame | 0.7914 |
| attention, target latched for 6 frames | 0.7981 |

Latching recovered a little — confirming the camera really was stuck in transit
under the movement limit — but the gap is far too large to be a tuning problem.
The reason is **discovery**: the policy can only zoom at tracks that already
exist, so it refines what was found while the raster keeps finding new things.
Coverage dominates refinement when recall is the binding constraint.

`AttentionPolicy` stays in `solution.py`, unshipped, because the idea becomes
right the moment recall stops being the bottleneck.

---

## 19. Two generator gaps the survey exposed

Both are free, both use only the supplied 25 frames, and both correspond to the
*largest* ablations in Cut-Paste-Learn (arXiv:1708.01642).

**Occlusion was forbidden.** `make_sample` used `boxes_overlap` to reject any
sprite landing on an already-placed one, so no training image ever contained one
object partly behind another. That paper's single biggest ablation is exactly
this: **−10.6 mAP for training without occlusion**. A quarter of pastes may now
overlap, capped at 35 % of the earlier object hidden (`OCCLUSION_PROBABILITY`,
`MAXIMUM_OCCLUDED`), and both labels are kept.

**Every sprite was pasted with one blend mode.** `paste()` did plain alpha
compositing, so a fixed edge artefact accompanied every single positive — a
free shortcut feature, present on targets and absent everywhere else. Blend mode
is now drawn per paste from hard / feathered / slightly blurred.

The counter-intuitive part of that paper is why **Poisson blending is not among
them**: as a single mode it scores 58.4 against 65.9 for no blending at all —
*worse than doing nothing* — while mixing modes reaches 73.7. What matters is
that no one artefact correlates with "target", not that any one looks real.

`check_labels.py` still passes after the change, with suspicious boxes rising
1.3 % → 1.6 %, which is the expected consequence of labelling objects that are
now partly covered.

---

## 20. Second validation run — 0.1018 → 0.1530

v4 stage-1 weights, the new even L0/L1 camera loop and `DRONE_ALTERNATES=2`,
all changed together. **The gain is therefore unattributable**, which is a
process failure worth naming rather than glossing: three changes went out in
one run because the run was cheap and the temptation was obvious.

In the units that matter, `0.1530 / 0.0625 = 2.4 of 16 classes`, up from 1.6.
We recovered roughly one class. **Thirteen and a half still score zero.**

### What we actually sent (`lab/compare_runs.py`)

| | v2 — 0.1018 | v4s1 — 0.1530 |
|---|---|---|
| detections per frame | 93.1 | **152.1** |
| confidence, median | 0.107 | **0.067** |
| above 0.7 | 181 | **58** |
| above 0.9 | 12 | **1** |
| L0 / L1 frames | 180 / 68 | 133 / 115 |

The better score came with **63 % more detections, lower median confidence, and
almost no high-confidence predictions at all** — one above 0.9 against twelve.
That is the third independent confirmation, after `exp08_fp_cost.py` and the
raised-floor test, that volume is close to free here and that pruning
detections is not where the score lives.

The 93 → 152 rise is about the size expected from alternate-class emission
firing on contested tracks, which is suggestive but not evidence.

---

## 21. One instance per class — a metric exploit that does not work

`run_metadata.json` is explicit:

```json
"total_objects": 16,
"object_totals": { "mine_roller": 1, "jet_plane": 1, "condor": 1, ... }
```

**Exactly one instance of every class.** Since AP is computed per class, a real
object ranked below several spurious boxes *of its own class* scores almost as
badly as never finding it. Capping reported boxes per class looked like a free
win. Measured on the terrain holdout:

| cap | mAP |
|---|---|
| **unlimited** | **0.8877** |
| 1 per class | 0.8813 |
| 2 per class | 0.8862 |
| 3 per class | 0.8877 (no-op) |

Monotonic, and never better than not doing it.

The reason is that a cap is **a bet on precision-at-1**. Keeping only the
leading box wins when it is the right one, but when it is wrong the class
scores a guaranteed zero, whereas leaving the extras in keeps the true box
somewhere in the ranking where it still earns partial credit. Even the holdout
model — much stronger than what we field on the evaluation scene — loses that
bet. The competition model gets about 2.4 classes of 16 right, so its leading
box is usually wrong and the bet is worse still.

This is the alternates result seen from the other side, and the two together
now say the same thing three ways: **under this metric extra low-confidence
boxes are cheap insurance, and discarding them forfeits recall that cannot be
recovered.** The code was reverted rather than left behind an unused flag.

### On pretraining elsewhere and fine-tuning here

Considered and rejected on the following reasoning. Someone scored 0.452, and
almost certainly on the supplied data, so the supplied data is sufficient for
three times our score — our deficit is methodological, not a shortage of data.
We also already pretrain elsewhere: `yolo11s.pt` is COCO-initialised. The
published mechanism for gains from external data at this scale is *backbone
representation quality* from self-supervised pretraining (DINOv2, +2.0-2.4 AP,
arXiv:2511.09554), which RF-DETR carries and which is already queued. An extra
hand-rolled detection stage on DOTA or VisDrone would mostly teach the backbone
that cars, roofs and boats are ordinary - which suppresses false positives, and
false positives are worth 2.6 points in total.

---

## 22. Building ground truth, because the local metric was useless

Three changes in a row — the v5 generator, two-stage fine-tuning, a new camera
loop — each shipped on a local score of 0.94 and each landed at 0.15. A proxy
that reads six times high cannot rank anything. The only way out was a held-out
benchmark on the terrain the evaluation actually uses.

Our own server recorded every view it answered, so the material was already on
disk. Labelling it is unusually cheap because of two facts:

* `run_metadata.json` gives **exactly one instance of each class**;
* the flight is one constant homography, measured on this sequence at
  **68.19 px/frame**.

So an object labelled *once* determines its position in every frame it appears
in. Sixteen annotations cover 244 frames.

**This is a test set and nothing else.** `make_dataset.py` has no code path
that reads `lab/recordings/`, and it is staying that way.

### The tooling

| script | what it does |
|---|---|
| `lab/flight_strip2.py` | composites Level-0 and Level-1 views into one ground-fixed image of the whole flight |
| `lab/object_reference.py` | the 16 assets cut from Helsinki, shown at native *and* Level-0 scale |
| `lab/propose_labels.py` | clusters confident detections into candidate locations and crops each |
| `lab/score_recording.py` | propagates a label by homography and scores any recording against it |
| `lab/label_check.py` | draws labels and predictions on frames, to check the labels |
| `lab/replay_recording.py` | replays a recording through any detector, so only the detector varies |

### Two bugs in the labelling itself

**The mosaic had the offset backwards.** Ground moves *down* the frame, so new
terrain enters at the *top* and later frames belong *higher* in the strip. The
first attempt banded horribly.

**Propagating by a constant step is wrong.** The homography carries a scale
term (`H[1,1] = 1.0135`), so points near the top of the frame move less than
points near the bottom. Approximating it with the average 17.05 view-px/frame
drifts several pixels per frame, which at these object sizes destroys IoU. The
first scoring run reported **8.8 % hangar recall**, which was entirely my own
drift. Propagating with `H^(k - ref)` instead is accurate to **2-7 source px
over 16 frames**, verified against independently segmented positions at frames
69 and 77.

`label_check.py` caught this by drawing the label next to our detections: the
orange boxes were on the hangar and the green label was not.

### What one good label is worth

The hangar is a near-black shape on grass, so it can be located by segmentation
rather than by eye — no circularity, since the detector plays no part in it.

| run | competition score | hangar recall | best IoU | mean conf |
|---|---|---|---|---|
| v2 | 0.1018 | 0.789 | 0.837 | 0.796 |
| v4s1 | 0.1530 | 0.816 | 0.840 | 0.856 |
| v5s2 | ~0.15 | 0.789 | 0.836 | 0.841 |

**Three models, indistinguishable on the object they find.** Whatever separates
0.10 from 0.15 is not detection quality on found objects.

### Hand-placing small objects does not work

Three placements, three failures — two `jet_plane` attempts and one
`medium_plane`, each offset by 40-120 source px, all caught by `label_check`.
Detection-cluster centroids are not labels either; one put a box 120 px off the
object. At 12-50 px, eyeballing downscaled imagery is not a labelling method,
and a benchmark that encodes the labeller's errors is worse than none because
it looks authoritative.

The hangar succeeded because it was *measured*. The generalisation of that is
**template matching against the Helsinki sprites**, which is also the
similarity machinery a metric-learning classifier needs — so building it serves
both purposes. That is the outstanding work.

---

## 23. What the benchmark says

With the class-agnostic diagnostic — *did any box land on the object, and what
did we call it?* — the failure finally has a shape.

| object | any box lands | what we called it |
|---|---|---|
| hangar | 30/38 (0.79) | `hangar` x30 |
| helicopter | 19/34 (0.56) | `helicopter` x11, **`medium_plane` x8** |
| medium_plane | 14/34 (0.41) | **`jet_plane` x13** |
| jet_plane | **0/33** | — (best IoU 0.378) |

**Localisation is not the problem.** Where we land on an object we land well,
IoU 0.83-0.84, and the hangar is named correctly every single time.

**Naming is the problem, and it is inconsistent frame to frame.** The same
helicopter is called `helicopter` in eleven frames and `medium_plane` in eight.

This also explains why `DRONE_ALTERNATES=2` coincided with 0.102 → 0.153:
emitting runner-up classes is a partial hedge against exactly this.

Two caveats kept in view: the aircraft labels are the unreliable ones, and the
`medium_plane` assignment is probably wrong — both candidate objects are
swept-wing jets, while the `medium_plane` reference is a cross-shaped propeller
aircraft. If those labels are swapped, part of this table is my error.

### A hypothesis that died cheaply

The hangar looks flat black in the strip, suggesting the assets might render
differently here than in Helsinki — which would break our sprites at source.
They do not:

| | mean | std | min | max |
|---|---|---|---|---|
| Helsinki hangar at Level-0 scale | 66.3 | 74.6 | 1 | 214 |
| evaluation hangar at Level-0 | 52.3 | 57.0 | 4 | 161 |

Same family. The flat look was an artefact of a quarter-resolution mosaic.

---

## 24. Six changes that moved nothing

All measured on the benchmark, all against the same 0.295 baseline.

| change | result |
|---|---|
| v5 generator (occlusion, blend variety) | 0.295 |
| two-stage fine-tune | 0.295 |
| global one-to-one class assignment | 0.295 (4 predictions differed) |
| track persistence 3 → 10 → 30 misses | 0.295, naming identical |
| merging overlapping tracks | 0.295 |
| gentle fine-tuning, low LR + light augmentation | **0.201** |

**Global assignment.** Since the scene holds one instance per class,
independent per-track argmax is unnecessarily weak: two tracks can claim the
same class. A Hungarian assignment forbids it. It changed four predictions,
because the confusion is *temporal* — one object named differently in different
frames — and the two confused objects are rarely in frame together.

**Track persistence and merging.** If one object spawned two tracks with
different labels, report-time deduplication would not catch it, since that only
removes duplicates *within* a class. Merging overlapping tracks is a genuine
correctness fix and is kept — it pools class evidence and cut output from 191
to 188 predictions per frame — but it did not move the score.

**Gentle fine-tuning** (`--lr0 0.0005 --light-aug`) tested whether our
`lr0=0.01` plus heavy augmentation destroys the COCO features that would carry
us across terrain. In-domain it matched: **0.9438 against 0.9484**. On the
benchmark it lost badly:

| | v5s2 | v6gentle |
|---|---|---|
| overall | **0.295** | 0.201 |
| helicopter, any box | **0.559** | 0.059 |
| predictions per frame | 191 | **54** |

The helicopter all but disappears. The mechanism is in the last row:
conservatism. Under a metric where the entire false-positive load costs 2.6
points and a missed class costs 6.25, emitting less is the wrong trade — the
fourth independent confirmation of that, after `exp08_fp_cost.py`, the raised
confidence floor, and the per-class cap.

So heavy augmentation earns its keep and the training recipe is not what holds
us back. A useful thing to have closed off.

### The pattern

Every one of these six reuses the detector's own class scores — aggregating
them differently, constraining them, preserving the features that produce them.
They fail the same way, which is one story rather than six disappointments:
**the class evidence itself is unreliable, and no downstream reasoning over it
helps.**

That is the argument, reached by elimination, for replacing the class decision
rather than post-processing it: a metric-learning head over crops, compared
against the 16 known references. A crop of a tank is a tank on any terrain,
which is precisely the bias that wrecks whole-frame detection here.

### RF-DETR — abandoned, and why it proves nothing

Three attempts. The first two died at step ~199 with no traceback and no
checkpoint: memory, not the model — RF-DETR rounds 952 up to 1088 and at batch
2 alongside another workload the OS killed it. At 728 with batch 1 it trained
stably and was abandoned mid-run in favour of the classifier work.

The DINOv2 warnings (`patch size 16 instead of 14`) are benign: RF-DETR-Small
is natively 512 with patch 16, and we fine-tune from `rf-detr-small.pth`, which
the message explicitly says is the case where it does not matter. Flagging them
as a confound was wrong.

`lab/replay_recording.py` carries an RF-DETR adapter, so the comparison can be
finished later. It is validated: replaying v5s2 reproduces its recorded run
exactly, 0.295 and the same confusion pattern, so any difference it reports is
the detector and nothing else.

---

## 25. Where this stands, and what is worth doing next

**The position.** Local 0.94, competition 0.15, benchmark 0.295. Someone else
is at 0.31 on the competition, almost certainly on the same supplied data, so
the gap is methodological rather than a shortage of data. Detection works —
where a box lands on an object it lands at IoU 0.84 and the hangar is named
correctly every time. Naming does not: the same object is called different
things in different frames.

**In priority order:**

1. **Template matching against the Helsinki sprites.** Unblocks everything.
   Hand-placing labels failed three times out of four; the one that worked was
   *measured*, not eyeballed. The same similarity machinery gives objective
   labels for the remaining twelve objects **and** is the foundation for (2),
   so it is not a detour.
2. **A metric-learning head over crops**, compared against the 16 known
   references, replacing the detector's class decision. Six attempts to
   re-aggregate the existing class scores all failed (§24); the evidence itself
   has to be replaced. A crop of a tank is a tank on any terrain, which is
   exactly the bias that wrecks whole-frame detection here.
3. **Global one-to-one assignment, revisited.** Worthless today (§24) because
   it operates on noisy scores, but with trustworthy per-crop similarities the
   one-instance-per-class constraint becomes a real constraint.
4. **Finish the RF-DETR comparison.** The adapter and the harness are ready;
   only the training run was abandoned.

**Not worth doing**, each with a measurement behind it: more empty images or
distractors (§16, 2.6-point ceiling); raising the confidence floor (0.1018 →
0.1094); inference-time upsampling (§18); an attention camera (§18); a per-class
cap (§21); gentler fine-tuning (§24). External background data is permitted by
the rules and is the literature's top recommendation (§17), but is parked while
the supplied data remains unexhausted.

**The methodological lesson, stated once.** Three changes shipped on a local
score that reads six times high, and all three landed at 0.15. The benchmark in
§22 cost a few hours and immediately showed that three consecutive models were
indistinguishable — which no amount of local mAP could have revealed. Build the
measurement before the next idea, not after.

---

## 26. Reset: Acquire Evidence Before Committing to a Class

User reports another team at 0.612 against our best reported 0.153. That shows
substantial headroom, not what their method or data must be. Their approach is
unknown. Reread the complete challenge README before choosing this direction.

### What the rules support

- Only one 960x540 view arrives per frame. L2 contains native source pixels;
  enlarging L0 or cropping it cannot recover the discarded detail.
- A response describes the whole current source frame. A confirmed object can
  contribute on later frames while the camera inspects something else.
- New views require legal moves. L0 to L2 takes two commands, and the target
  moves during those commands. The planner must use supplied constraints.
- AP50 rewards correct class, box, coverage and ranking together. Duplicates
  and high-ranking false positives matter. Skipped frames cost recall.
- One instance of each class is documented for Helsinki only. Do not force
  that count or a one-to-one class assignment on an unseen scene.
- No competition API calls. Recorded images/labels remain excluded from
  training. External imagery remains deferred. RF-DETR remains abandoned.

### Evidence we must not build on

The former FP-cost experiment inserted randomly located boxes sampled from
unlabelled recorded predictions around perfect Helsinki ground-truth boxes.
Its true-positive scores were assumed uniform in [0.60, 0.92]. That answers a
conditional ranking question, not how much our actual false positives cost.

Added `--tp-confidence LOW HIGH` to `lab/exp08_fp_cost.py`, using independent
fixed RNGs so injected boxes remain identical between confidence-band runs.
Seed 0, perfect geometry and full recall in every row:

| Injected boxes/frame | TP scores 0.60-0.92 | TP scores 0.10-0.30 |
|---|---|---|
| 0 | 1.0000 | 1.0000 |
| 41 | 0.9625 | 0.4785 |
| 93 | 0.9346 | 0.3159 |

This is a **sensitivity check**, not a prediction of competition AP. It
disproves the claimed universal 2.6-point FP-cost bound. The earlier 0.9740
used a shared RNG; the paired test separates score draws from box draws.
More low-confidence predictions do not establish better precision or recall.

Other corrections:

- AP 0.153 does not imply "2.4 classes found" or "13.6 classes at zero".
  Many different per-class precision-recall curves yield that mean.
- Maximum detection confidence does not measure whether a class was found.
- A best IoU from one trustworthy hangar label does not establish localisation
  quality for the other classes. Template matches would be proposals, not
  automatically correct labels either.
- Logs for v5s1, v5s2 and v6gentle explicitly say the requested `lr0` was
  ignored, then select `AdamW(lr=0.0005)`. The claimed 20x learning-rate change
  did not occur. Different augmentation and schedules remain confounded.
- Low performance of the existing AttentionPolicy does not reject learned
  heatmaps: it chases existing YOLO tracks at L1 and delegates alternate calls
  to a stateful raster. It is not the system proposed below.

### New, untested approach: survey, inspect, verify

Implement the user's dense heatmap idea as **active coarse-to-fine detection**,
not a second class-voting rule over YOLO tracks. No new network was trained in
this reset; the deployed model and camera policy were not changed.

1. **Survey with a class-agnostic foreground heatmap.** Use a pretrained small
   EfficientNet backbone with shallow, high-resolution features and a binary
   target/background head. Train from supplied sprites and masks, with the
   actual L0 downsampling applied after compositing. Its task is only to rank
   candidate regions, not read sixteen tiny class identities. Include target-
   free crops and ambiguous clutter from the provided data. Threshold for high
   proposal recall under a fixed candidate budget, not final reporting.
2. **Plan an inspection, not a one-frame glance.** Maintain a motion-warped map
   of examined ground. Rank legal views by expected improvement in correct
   future reports: candidate plausibility, expected benefit from extra pixels,
   remaining time in frame, and travel cost. Propagate candidate positions to
   the arrival frame, respect L0->L1->L2, and latch a target through acquisition.
   Cap inspection duration and reserve systematic sweeps of unseen/new ground;
   an imperfect heatmap must not permanently hide an object from inspection.
3. **Verify at native resolution against references.** At L2, use a shared
   crop encoder with multiple Helsinki reference views per class plus an
   explicit background/reject decision, and refine the box from the actual
   received pixels. Use rotation/scale variants and different-background
   positive pairs; choose between metric loss and ordinary classification on
   a fixed crop benchmark before making the loss itself a research project.
   Crops reduce context but do not eliminate background, lighting or viewpoint
   shifts. No forced nearest class and no forced one-object-per-class count.
4. **Report and track confirmed evidence.** Carry verified boxes/classes
   forward with estimated motion; reduce confidence with geometric uncertainty
   and stale observations. Do not let a weak later L0 guess overwrite strong
   L2 evidence automatically. Suppress duplicate same-class reports. Calibrate
   reporting confidence on held-out development data rather than assuming
   every extra class hypothesis is free. Preserve the old detector as a fixed
   comparison baseline, not an unmeasured production dependency.

Why this can improve the metric: an early successful inspection may support
correct reports for many subsequent frames, while a rejected clutter candidate
does not become a high-confidence false track. It trades camera time for better
evidence. It cannot recover already missed frames, and travel/latency can erase
the benefit. These are hypotheses to test, not promised gains or claims about
the leading team's method.

### First gates, before another long training run

1. **Acquisition feasibility upper bound.** In the local simulator only, expose
   class-free oracle proposals in the current view and allow perfect class/box
   confirmation only after a legal L2 inspection. Use causal track memory and
   score with the existing COCO scorer. Compare against the same verifier with
   fixed sweeps, including a 249-frame synthetic flight from provided assets
   so start-up costs on Helsinki's short clip do not dominate. Add controlled
   clutter proposals to stress the planner. Stop this direction if even this
   optimistic acquisition bound cannot beat the fixed-sweep baseline under
   the chosen size/latency assumptions. Oracle data stays in the test harness.
2. **Heatmap gate.** Measure proposal recall by object size at fixed candidate
   budgets (e.g. top 16 and top 32), plus candidates on target-free views. Split
   by source ground region with an overlap buffer, not randomly by adjacent
   frames. Synthetic held-out seeds alone do not measure scene transfer.
3. **Verifier gate.** Measure macro class accuracy, confusion and false accepts
   on foreground AND background crops, at L0/L1/L2 sampling. Different rendering
   seeds and held-out source regions are useful stress tests, not a substitute
   for genuinely different real scenes. The single asset/class in Helsinki
   prevents an independent held-out-instance test.
4. **Combined gate.** Compare same-view detector outputs first, then full legal
   camera replays with the same scorer and frame-drop model. Measure p95 compute
   and end-to-end timing against 333 ms/frame, not the 3333 ms request timeout.
   Record actual optimizer/LR, model hash, split, augmentation and camera config.
   Change one component at a time; do not replace the server on a recall-only
   result from four disputed labels.

Existing recordings can check detector/verifier behaviour on the views already
captured, but cannot supply an unrecorded L2 observation for a different camera
path. Do not fabricate those pixels. No calls to the competition service are
required or authorised for these gates.

---

## 27. Acquisition Feasibility: First Gate Tested

Implemented `lab/exp10_acquisition.py`, an isolated CPU-only geometry experiment.
No training, serving changes, recordings, external images, or competition API
calls. It reuses `local_evaluator.Camera`, its supplied movement constraints,
the existing box-warping helpers and the actual local COCO AP50 scorer.

### Contract and controls

- The acquisition policy sees anonymous proposal boxes, never their class,
  true/false status or hidden instance identity. Proposals must overlap the
  current view by at least 80 percent of their box area. A configurable minimum
  short side in transmitted pixels models limited proposal sensitivity.
- Perfect confirmation is possible only at L2, under the same visibility rule.
  Confirmations reveal correct boxes/classes; clutter is perfectly rejected.
  Confirmed observations propagate causally with the known homography. No
  reports are backfilled into past or skipped frames.
- The planner ranks pending regions by estimated remaining in-frame lifetime
  divided by travel cost. It predicts the next observation position, latches
  a target for at most five source frames, requests an overview after six
  frames, and uses a top-band sweep when no pending target remains. Examined
  candidate regions persist in motion-compensated memory.
- Baselines are fixed L2 top-band and full-frame sweeps. Both use exactly the
  same perfect verifier and causal reporting memory. This compares camera
  acquisition under an L2-only contract, NOT against YOLO's actual outputs.
- The synthetic cases are **annotation trajectories, not rendered imagery**:
  Helsinki median box dimensions, the measured Helsinki homography, random
  horizontal positions and crossing times, repeated instances of all classes.
  Default density parameter 10 yields 78 target trajectories per 249-frame
  case. Seeds 0, 1 and 2 are used. Objects crossing near sequence boundaries
  are naturally truncated; no independent-instance generalisation is tested.
- Clutter uses the same sizes and motion as targets and indistinguishable
  proposal treatment. Adding it preserves the exact target trajectories. The
  ratio is per generated target trajectory, not false positives per frame.
- A constant 400 ms response-time assumption at 333 ms/frame processes every
  second frame and scores 124 of 249 frames as empty. This is a deterministic
  frame-drop stress test, not a measurement of a real network or tunnel.

The oracle observation model is optimistic, but the planner is a simple
heuristic, not an optimal policy. These numbers are NOT a mathematical upper
bound on all possible acquisition strategies or a prediction of competition AP.

### Results

COCO AP50, mean over the three 249-frame synthetic cases:

| Conditions | L2 top-band sweep | L2 full sweep | Targeted acquisition |
|---|---|---|---|
| Perfect proposals, no clutter, no skips | **0.8174** | 0.3871 | 0.8140 |
| Two persistent clutter trajectories per target | **0.8174** | 0.3871 | 0.6967 |
| Proposals require short side >=6 transmitted px | **0.8174** | 0.3871 | 0.7846 |
| 400 ms response-time assumption | **0.2999** | 0.1621 | 0.2484 |

With clean proposals, targeted acquisition spends 32 views at L0, 63 at L1,
and 154 at L2; the top-band sweep spends 1, 1 and 247 respectively. The equal
scores in the clean case do not establish an advantage worth a new heatmap
training run. Unranked clutter diverts the target-driven policy; the sweep is
unaffected because its route ignores proposals and the verifier is ideal.

On the supplied 25-frame Helsinki clip, targeted acquisition wins instead:
**0.4045**, versus 0.2783 top-band and 0.3051 full sweep. With the >=6 px
proposal requirement its score falls to 0.2911. This illustrates sensitivity to
initial object placement and the short sequence horizon. These are not directly
comparable to section 2: the visibility threshold, confirmation contract and
memory implementation differ; all three methods in THIS test share them.

### Checks and reproducibility

`--check` passes assertions for legal one-step level changes, anonymous
proposals, no class output before L2, causal propagation, empty skipped-frame
outputs, repeated-class instances kept separately, decoy rejection, unchanged
target trajectories when clutter is added, and perfect-prediction COCO AP=1.
All 48 scenario/policy runs completed without an illegal camera command.

From `drone-flyby/`, using the `dm` interpreter:

```powershell
python lab/exp10_acquisition.py --check
python lab/exp10_acquisition.py --seeds 3 --frames 249
python lab/exp10_acquisition.py --clutter-ratio 2 --out lab/out/acquisition_clutter2.json
python lab/exp10_acquisition.py --proposal-min-px 6 --out lab/out/acquisition_min6.json
python lab/exp10_acquisition.py --latency-ms 400 --out lab/out/acquisition_latency400.json
```

Default output is `lab/out/acquisition_bound.json`; outputs include settings,
per-case AP, camera level counts, skipped frames and confirmation counters.

### Decision

**Do not train the heatmap for this planner yet.** It fails the proposed
long-flight improvement gate: near-tie with clean proposals, worse with clutter
or missing tiny proposals. This rejects this first heuristic/configuration,
not the entire class of heatmap-driven policies. Proposal ranking, adaptive
survey timing, more varied motion and noisy L2 verification remain untested.

The simpler L2 entry-band sweep plus verification is the baseline to beat in
the next acquisition experiment. Its high oracle score only shows geometric
opportunity under these assumptions. A real verifier still needs acceptable
class accuracy, box accuracy and clutter rejection before any deployment.

---

## 28. Binary Crop Verifier: Removes Some Clutter, Not Promoted

The user identified numerous confident rooftop predictions in the fresh v4s1
inference image at recorded source frame 211 (index 210, L1). These came from
the detector alone, not tracking. Tested an additional target/background crop
verifier offline in `lab/crop_verifier.py`.

### Data and model

- The base detector stays `model/v4s1.pt`, our highest reported competition
  checkpoint (0.153). Raw inference threshold 0.05; class names/boxes unchanged.
- All fitting, calibration and quantitative testing use `lab/dataset_v5`,
  generated from provided Helsinki data. No recorded validation pixels, labels
  or pseudo-labels enter fitting or threshold selection. External imagery is
  still deferred; only torchvision's pretrained ImageNet EfficientNet-B0
  checkpoint was downloaded from its official model host.
- Each proposal becomes a square crop with 1.25x context, resized to 128x128.
  Features come from a frozen EfficientNet-B0 with its classifier removed.
  A StandardScaler and binary logistic regression (C=0.1, balanced class weights,
  max_iter=1000) are fitted on TRAIN features only. This is a frozen-feature
  verifier, not backbone fine-tuning or a Siamese model.
- Positives are annotated target boxes and detector proposals with class-
  agnostic IoU >=0.50 to a labelled target. Hard negatives are detector proposals
  whose ENTIRE context crop has no overlap with any target annotation; ambiguous
  overlaps are excluded, not labelled background. Random target-free crops use
  target-like sizes. Negative safety depends on the completeness of synthetic
  annotations, which are not perfect segmentation ground truth.
- Sample: 240 train images, 40 calibration images, 40 test images, seed 0.
  Training has 11,524 crops: 4,348 annotation positives, 4,010 detector positives,
  286 hard negatives and 2,880 random negatives. Calibration has 1,627 crops;
  test has 1,865. Image paths are disjoint, but ALL splits share Helsinki scenery
  and source assets. This is not a held-out-city generalisation benchmark.
- A single rejection threshold is chosen at the calibration positives' fifth
  percentile: 0.1791661. Surviving confidence is detector confidence multiplied
  by verifier score. Balanced-logistic scores are NOT deployment-calibrated
  probabilities; multiplication is a ranking heuristic, tested as part of this
  complete filter. No test- or recording-based threshold tuning was performed.

### Local results

An initial 24-train/12-validation-image smoke test exercised mining, fitting,
checkpoint reload and AP scoring. The planned full sample then produced:

| Synthetic test measure | Result |
|---|---|
| Binary crop AP (not detection AP) | 0.9733 |
| Binary ROC AUC | 0.9475 |
| Annotated-positive retention | 95.03% (704 crops) |
| Localized detector-positive retention | 94.64% (634 crops) |
| Hard-negative rejection | 42.55% (20/47) |
| Random-negative rejection | 81.04% (389/480) |
| Detector COCO AP50, before filter | **0.8626** |
| Detector COCO AP50, after filter | **0.8179** |

The binary separation is useful, but filtering/rescoring costs 0.0447 detector
AP on this test. Large-launcher retention is only 87.14%, small-launcher 89.16%,
and tank 90.72% among the test positive crops, despite a pooled 95% calibration
target. High binary AP is not sufficient to justify deployment.

### Latency and verification

The first smoke test showed approximately 1.9 seconds per varying crop batch.
A focused timing test confirmed first-use costs for new batch sizes, followed
by ~9 ms on repeated sizes. Fixed padding to 64 crops and warm-up after loading
removed this risk in the tested thread. The full test measures **9.02 ms median,
11.60 ms p95** extra verifier time. A separate 65-crop check (two batches) took
29.8 ms. This is NOT an HTTP latency or serving-thread warm-up guarantee.

Checks cover border crops, source-to-view coordinate conversion, and rejecting
negative crops even when a tiny target is enclosed. Reloaded checkpoint scores
match sklearn predictions within 1e-5. The checkpoint includes encoder weights,
normalisation, classifier coefficients and crop geometry, so inference requires
no downloads or cloud services. The live server was not modified or restarted.

### Untuned recording preview

Applied the frozen checkpoint to the same eight pre-existing fresh v4s1 outputs,
using `lab/out/inference_v4s1_4eca91bf/predictions.json`. Source frame 211:

| Detection | Original confidence | Verifier score | Retained? |
|---|---|---|---|
| ta-ta, highest-ranked rooftop box | 0.691 | 0.995 | yes |
| ta-ta, next rooftop box | 0.689 | 0.012 | no |
| ta-ta, third rooftop box | 0.679 | near zero | no |
| large_launcher, visible bottom-left target | 0.669 | 0.998 | yes |
| large_tower candidate | 0.664 | 0.015 | no |

In total, 77/114 raw boxes survive the rejection gate. At display confidence
0.30, shown boxes fall from 24 to 12; this combines rejection AND rescoring,
not twelve verified false positives removed. The frame-70 hangar remains at
about 0.873. Other unverified boxes remain, including a high-confidence rooftop
prediction. These visuals show partial transfer, not a competition AP gain.

Artifacts:

- `lab/out/verifier_v1/verifier.pt`: reusable experimental checkpoint.
- `lab/out/verifier_v1/metrics.json`: crop and detector metrics, full split file
  lists, settings notes, base-detector path and SHA-256.
- `lab/out/verifier_v1_preview/frame_000211_comparison.png`: original above,
  filtered below; seven other frame comparisons in the same directory.
- `lab/out/verifier_v1_preview/predictions.json`: all per-proposal verifier
  scores, accepted predictions and before/after counts.

Reproduce from `drone-flyby/` using the `dm` interpreter (choose a fresh fit
output directory; existing experiments are not overwritten):

```powershell
python lab/crop_verifier.py check
python lab/crop_verifier.py fit --train-images 240 --val-images 80 --out lab/out/verifier_v1_repeat
python lab/crop_verifier.py preview --manifest lab/out/inference_v4s1_4eca91bf/predictions.json --checkpoint lab/out/verifier_v1/verifier.pt --out lab/out/verifier_v1_preview
```

### Decision

**Updated by user instruction: expose this experiment for manual competition
validation; do not use synthetic performance as a veto.** The synthetic AP
drop and missed positives remain diagnostics, not a reason to withhold a usable
API preset. Only 286 mined hard negatives were available in the training sample;
generalization remains unproven. The user runs competition validation and will
provide the result. Recorded validation data remains inference-only; the agent
makes no competition API calls. The baseline is not automatically replaced.

---

## 29. Manual Validation Presets: Verifier V1

New workflow: make each runnable inference experiment selectable through
`api.py`, verify startup, protocol correctness and local latency, then let the
user judge competition performance. Synthetic metrics are retained as
diagnostics, not promotion or availability gates. Do not automatically submit
an attempt or infer a competition score from the local checks.

From `drone-flyby/` in conda environment `dm`, stop the existing API with Ctrl+C
in its terminal first; these commands intentionally use the same port 9053:

```powershell
python api.py --experiment verifier-v1
```

Matched control, after stopping that server:

```powershell
python api.py --experiment v4s1
```

Both presets explicitly use `model/v4s1.pt`, imgsz 960, detector confidence
0.05, two alternate classes at damping 0.5, class assignment disabled, and
three tolerated misses. The existing camera loop and world model are unchanged.
Preset choices override those environment settings so stale weights from v5
cannot silently enter this comparison. Device and recording directory still
honour the environment. `baseline` preserves the normal DRONE_WEIGHTS selection
with no crop verifier. For `uvicorn api:app`, set DRONE_EXPERIMENT to the same
preset name before importing the app.

`verifier-v1` wraps the detector with `VerifiedDetector`, applying the existing
checkpoint's crop rejection/rescoring **before** the world model update. It uses
`lab/out/verifier_v1/verifier.pt`, threshold **0.17916610836982727**, rejecting
scores below that threshold and multiplying retained detector confidences by
their verifier scores. The API annotations contain only the documented fields;
internal verifier scores are not added to the response payload.

The verifier loads and warms its fixed-size batch on one persistent worker
thread; subsequent rescoring uses that same thread. Inference needs no cloud
service or pretrained-backbone download: the checkpoint contains the encoder.
Both model files remain in place, with no overwrite or default promotion.

`/health` now identifies:

- selected experiment and actual detector SHA-256;
- whether the verifier is enabled and loaded;
- verifier checkpoint path, hash, saved threshold, device and latest error;
- existing detector, camera and recording settings.

Detector SHA-256: `8e6a793c6e043f1e252c5a73c80e92f6c48d7eed8388a557824c75eb510caf0d`.
Verifier SHA-256: `a57d85f772197dd5a886fd95af4e1fc33640e711388ab43905b0ecb8093ac83e`.

Failed model/verifier startup is explicit: health is degraded and `/predict`
returns HTTP 503 instead of silently serving an unfiltered substitute or an
apparently healthy zero-detection experiment. A runtime verifier failure is
also visible via its last-error health field; it does not bypass the filter.

Validation performed locally using FastAPI TestClient and supplied Helsinki
pixels: experiment/control load, valid L0/L1/L2 normalized responses, request
echoing, verifier-only fields excluded, no-verifier control, degraded 503, and
unknown experiment rejection. Adapter checks assert rejection/rescoring,
unchanged input predictions, and the same thread for verifier load and use.
Actual verifier preset smoke times: 66.2, 139.2, 54.9, 52.6 ms across four
requests (median 60.6 ms). These are local smoke timings, not measured public
round trips or a production p95.

Port 9053 was occupied by the user's existing API process while these checks
ran. It was left untouched; a restart with the preset command is required to
activate the experiment. Manual competition result: awaiting the user.

---

## 30. Stricter Final Reporting Threshold

User observed 185-230 reports per frame and requested a stricter threshold.
Local `/health` confirmed the active `verifier-v1` experiment, detector floor
0.05, verifier threshold 0.17916610836982727, and two alternate classes. These
response counts include tracked and alternate-class boxes, not just fresh
detector output. The verifier was loaded, not silently bypassed.

Added `--report-conf` to `api.py` (or `DRONE_REPORT_CONF` for imported serving).
Default 0.0 retains previous finite-confidence behavior. Finite values in
[0, 1] are accepted; a prediction exactly at the floor is kept. The filter is
in `example._to_annotations`, after verification/tracking and before protocol
conversion, recording and the response. Track updates and camera behavior are
unchanged, and internal detections are not mutated. Nonfinite scores are dropped.
Health, startup and per-frame logs identify the reporting floor separately from
the detector threshold and verifier gate.

After the current validation finishes, stop the API with Ctrl+C and run in `dm`:

```powershell
python api.py --experiment verifier-v1 --report-conf 0.25
```

Control with identical weights and verification, after stopping that server:

```powershell
python api.py --experiment verifier-v1 --report-conf 0
```

The value 0.25 is a proposed manual test, not an optimized threshold. The
live server was neither restarted nor reconfigured during implementation.

Offline count check on the latest recorded snapshot
`6e2a0ac9379f4866ba4a51ecc0d9a8c5` (232 saved frames at inspection):

| Frame | Recorded reports | Reports >=0.25 |
|---|---|---|
| 216 | 223 | 13 |
| 217 | 230 | 16 |
| 218 | 225 | 13 |
| 219 | 186 | 13 |
| 220 | 185 | 12 |
| 221 | 211 | 15 |
| 222 | 216 | 23 |
| 223 | 203 | 20 |
| 224 | 192 | 20 |

Snapshot mean reports/frame: 116.2 -> 10.1. This is output filtering, not an
AP measurement. A final confidence cutoff removes lower-ranked outputs but
does not improve the ordering of retained predictions; under standard ranked
AP on fixed predictions it generally cannot help and may cost recall. It also
does not remove high-confidence roof errors. A stricter crop-verifier gate or
better rescoring is a separate experiment that changes which evidence reaches
tracking; neither was changed here. User will provide manual validation results.

Checks: inclusive boundary, nonfinite rejection, input preservation, zero/one
floor behavior, health value, and actual local TestClient `/predict` requests
at L0/L1/L2 with the loaded verifier. All responses passed schema/geometry
validation and contained only confidences >=0.25. No competition API calls,
training, new weights, or changes to the `half` precision setting were made.

---

## 31. Second Literature Scout: Procedure Before Model Selection

The user requested a fresh state-of-the-art survey and the general procedure.
The complete theme-organized review, source versions, datasets, input scales,
ablations, negative results and ranked experiments are in
[LITERATURE.md](LITERATURE.md#second-scout---2026-09-17). This is research only;
no new inference preset, training, installs, server changes or competition API
calls occurred. Existing experiments remain available for user-run validation.

### What the primary sources support

- **Propose, mask, describe, match is established.** CNOS, *A Strong Baseline
  for CAD-based Novel Object Segmentation* (ICCV 2023 R6D Workshop,
  [arXiv:2307.11067v4](https://arxiv.org/html/2307.11067v4)), uses SAM/FastSAM
  proposals and DINOv2 reference matching. Seven-BOP-dataset mask AP50:95 is
  41.2 versus 21.4 for its cited prior method. It still reports erroneous
  descriptor classification despite good masks; its main benchmark uses CAD
  renderings, not our small set of Helsinki crops. The number is not a forecast.
- **Prompt-only is not reliable fine-grained adaptation.** *SAM 3: Segment
  Anything with Concepts* ([arXiv:2511.16719v2](https://arxiv.org/html/2511.16719v2))
  reports RF100-VL box AP 15.2 zero-shot versus 36.5 with 10-shot fine-tuning,
  and explicitly identifies fine-grained aircraft types as a zero-shot weakness.
  Its hard-negative phrase ablation improves 28.3 -> 43.0 cgF1; this is NOT
  box AP and NOT an ablation of empty-image percentage.
- **Positive references need rejection evidence.** *Learning Multi-Modal
  Prototypes for Cross-Domain Few-Shot Object Detection* (CVPR 2026 Findings,
  [arXiv:2602.18811v1](https://arxiv.org/html/2602.18811v1)) combines visual class
  prototypes and hard-negative prototypes with text. Its 5-shot six-domain
  average is 44.0 versus 40.4 mAP for its cited GroundingDINO baseline;
  DIOR aerial is 31.3 versus 29.6. This uses labelled target-domain supports;
  our extra unseen-city shift remains untested. Source-table inconsistencies
  and non-winning dataset results are recorded in the full review.
- **Heatmap-to-crops is not a new invention, but camera acquisition differs.**
  *AutoFocus: Efficient Multi-Scale Inference* (ICCV 2019,
  [CVF abstract](https://openaccess.thecvf.com/content_ICCV_2019/html/Najibi_AutoFocus_Efficient_Multi-Scale_Inference_ICCV_2019_paper.html))
  predicts class-agnostic FocusPixels and processes finer-scale FocusChips.
  It matches its multi-scale baseline's 47.9 COCO AP at 2.5x speed. It already
  has the source image; our unobserved L2 pixels require legal camera moves.
- **Measure errors, not just output volume.** *TIDE: A General Toolbox for
  Identifying Object Detection Errors* (ECCV 2020,
  [arXiv:2008.08115v2](https://arxiv.org/html/2008.08115v2)) separates class,
  localization, combined, duplicate, background and missed-object errors by
  individual changes in AP. The diagnostic still requires reliable labels.

### Repository procedure and ranked next experiments

Define trusted target references/box conventions -> establish single-view
control -> compare candidate recall and class/background discrimination ->
adapt on permitted positive and hard-negative data -> refine and deduplicate
-> add causal temporal evidence -> test camera selection separately. Functions
can be joint inside a detector; separate networks are not mandatory.

1. **Class-conditional reference scoring plus negative references** on fixed
   current proposals. Expected direction: reduce high-ranking roof errors and
   discriminate similar target assets; magnitude unknown. This differs from
   the binary verifier of section 28. A frozen DINOv2 reference baseline avoids
   committing immediately to custom Siamese training. Compare contextual crops
   and masked crops because masks may erase tiny discriminative parts. [CNOS;
   Learning Multi-Modal Prototypes, above]
2. **Independent SAM 3 or FastSAM-style proposals with the same verifier.**
   Expected direction: improve proposal coverage or crop boundaries; magnitude
   unknown. It may add clutter and latency instead. Do not change the camera
   simultaneously. Check access, supported runtime and speed before claiming
   a deployable system. [CNOS; SAM 3, above]
3. **Controlled specialist fine-tuning** with explicit optimizer and effective
   LR logs, plus a frozen-feature/head-only control. The earlier auto-optimizer
   experiment did not rule it out. RF-DETR is not restarted: this survey does
   not override its abandonment. [*Frustratingly Simple Few-Shot Object Detection*,
   ICML 2020, arXiv:2003.06957; *RF-DETR: Neural Architecture Search for Real-Time
   Detection Transformers*, ICLR 2026, arXiv:2511.09554v2]
4. **Causal track confirmation and camera tests** after single-view comparisons.
   SAM 3's published default t+15 confirmation needs future frames and cannot
   be copied into this request protocol. Adapt the idea, not its measured gain.
   [SAM 3, appendix C.3]

No source establishes the best procedure for the exact challenge or reveals
other teams' methods. Synthetic diagnostics and visual inspection do not veto
API availability: once an experiment is runnable and passes local protocol
and timing checks, expose its named preset and matched control for the user's
manual validation. Never fit on recorded validation images or pseudo-labels.

---
