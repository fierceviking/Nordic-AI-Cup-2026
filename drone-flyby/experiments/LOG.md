# Drone-Flyby — Experiment Log

Goal: understand why competitors reach ~0.653 while a stateless per-image YOLO
scores ~0.153, and build a much stronger solution.

Scoring: COCO mAP@0.50, macro-averaged over the 16 classes present. One instance
per class. You answer for the WHOLE 3840x2160 source frame every frame, but only
receive a 960x540 crop (L0=÷4 whole frame, L1=÷2 quarter, L2=÷1 sixteenth). You
also steer the camera (1 move/frame, bounded step, 1 level-step/frame).

## Data facts (Helsinki, 25 frames, 16 classes, 1 instance each)
- Drone flies straight, ~13.9 m/frame, 600 m altitude. Objects flow through the
  frame, mostly +y (~52-80 px/frame) with small x drift (-11..+11 px/frame).
- Motion is NOT a single global translation: dy varies per object (parallax /
  different heights). But each object's motion is individually very stable
  (dy sd ~3-12px, dx sd ~0.2-2.7px).
- Object source sizes range ~19px (ta-ta, small_launcher) to ~166px (condor).

## Exp 1 — Motion extrapolation (constant velocity from 2 consecutive obs)
Fraction of predictions keeping IoU>=0.5 with GT after k frames:
  k=1: 0.99   k=2: 0.95   k=3: 0.88   k=4: 0.77   k=5: 0.59   k=6: 0.34
=> A memory that re-observes each object every ~3-4 frames can keep valid boxes
   for frames where the camera is looking elsewhere. This is exactly the info a
   stateless detector throws away.

## Exp 2 — Oracle architecture simulator (experiments/sim.py)
Perfect detector but can only "see" an object if its shorter side is >= MIN_ONSCREEN
transmitted px at the current level (models downsample). Same COCO scorer.

  MIN_ONSCREEN  stayL0   zoomOnly(L1 raster)  zoom+motionMemory
       9px      0.749         0.300               0.485
      14px      0.389         0.230               0.406
      20px      0.239         0.230               0.406
      28px      0.106         0.163               0.285
      40px      0.051         0.095               0.161

Interpretation:
- With a PERFECT detector, staying at L0 already gives 0.749. So the user's 0.153
  is a DETECTOR-QUALITY problem as much as an architecture problem: 0.153 matches
  "stay L0 + detector needs ~20-28px on-screen" (=> only ~80-110px source objects
  detected, i.e. the big ones).
- Once a (realistic) detector needs >=~18px on-screen, zoom+memory BEATS stay-L0
  and has a higher ceiling. The naive L1 raster policy here caps ~0.4-0.49; a
  smarter policy (target stale objects, use L2 for the tiniest) + better detector
  should go higher. The two smallest classes are undetectable even at L1 (need L2).

## Diagnosis
The 0.653 solutions almost certainly:
  1. Zoom (L1/L2) so objects are big enough for the detector to classify+localize.
  2. Maintain a persistent world/track memory + per-object motion model, and
     report ALL known objects projected into every frame (full-frame recall),
     not just what's in the current crop.
  3. Use a detector good enough on the resulting on-screen sizes.

## Exp 3 — Real detector, real pipeline (eval_pipeline.py, in-sample on helsinki)
Trained YOLO11s on wire-mimicking tiles (datagen.py: L0 whole frame + L1/L2 crops,
INTER_AREA to 960x540, GT remapped). solution.py = detector + motion-memory world
model + camera policy. Replayed helsinki through the REAL camera dynamics + COCO
scorer. Per-policy mAP@0.50:

  policy            mAP     note
  stale (L1 sweep)  0.297   zoom+memory, aim at stalest region
  l1sweep           0.228   raster L1
  hybrid (L0/P=4)   0.398   L0 refresh every 4th frame + L1 zoom between
  L0 (stay)         0.643   full frame every frame + tracking  <-- matches target

Per-class AP at pure L0 (the winner):
  helicopter 1.000  jet_plane 1.000  large_launcher 1.000  small_tower 1.000
  condor 1.000  tank 0.960  spacecraft 0.935  jammer 0.933
  hangar 0.663  large_tower 0.648  small_plane 0.627  medium_plane 0.505
  ta-ta 0.009  small_launcher 0.005  medium_launcher 0.000  mine_roller 0.000
Infer: median 33ms, first-frame ~4.5s (warmup — must preload at import).

### Re-diagnosis (important, overturns the "must zoom" hypothesis)
- DETECTOR QUALITY, not camera control, was the user's dominant missing lever.
  A properly trained detector that just STAYS at L0 and tracks already hits 0.643,
  i.e. the whole 0.153->0.65 gap is explained by the detector, not the architecture.
- Zoom policies did WORSE (0.30-0.40): leaving L0 sacrifices precise boxes on ~10-12
  strong classes (perfect at L0) to refresh only 1-2 small ones. Coverage loss >
  small-class gain. The oracle sim over-rewarded zoom because it assumed a PERFECT
  detector; the real detector is already ~perfect on big classes at L0.
- Only 4 small/rare classes fail at L0 (ta-ta, small_launcher, medium_launcher,
  mine_roller) — their detail is genuinely destroyed by the L0 4x downsample and
  can only be recovered by zoom. Lifting these is the remaining upside, but must be
  done WITHOUT dropping L0 coverage of the strong classes (via tracking).

### Caveats
- 0.643 is IN-SAMPLE (trained AND evaluated on helsinki). Real eval is a different
  scene/city with the same asset classes. Generalization is the real open risk, and
  cannot be measured with the one scene we have. Priority: a robust detector (bigger
  model, heavy aug, denser tiles) + the simplest inference (pure L0 + tracking).

## Exp 4 — Realtime cold-start bug (the offline->realtime gap)
Via the REAL HTTP harness (local_evaluator, localhost) the OFFLINE score matched
in-process exactly: 0.643, 25/25 camera moves applied, 0 timeouts. But --realtime
(3 fps clock, only the newest frame is ever sent) dropped to 0.302:
  frames sent 20 / skipped 5 / round-trip max 2094 ms.
Cause: the FIRST served frame stalled ~2 s, so the clock advanced and skipped the
opening 5 frames of the 25-frame sequence (=20% of all GT frames -> huge AP hit).
Steady state was fine (median 31 ms). The ~2 s is a CUDA/cuDNN *per-thread* init
cost: the startup warmup ran on the FastAPI startup thread, but uvicorn serves the
sync endpoint on an anyio *threadpool* thread, which was cold on its first call.

Fix (solution.py): pin ALL inference (load + warmup + serving) to ONE dedicated
worker thread (`_INFER_POOL`, max_workers=1), warmed 4x at startup. Every served
frame then reuses the already-warm thread, so the first real frame is fast.
NOTE: validation of this fix is left to the user (do not spend eval attempts;
do not run the api harness here).

## Environment
- conda env `dm`: python 3.11, cv2 5.0, torch 2.11+cu128 (RTX 5070), ultralytics
  8.4.154, faster_coco_eval 1.8.0.  Invoke: ~/miniconda3/envs/dm/python.exe

## Next
- Build/train a detector; measure detection quality vs on-screen size (the real
  critical path). Then wire zoom + motion-memory pipeline and score end-to-end.
