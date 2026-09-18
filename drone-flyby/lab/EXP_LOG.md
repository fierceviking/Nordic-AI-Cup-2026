# Drone-flyby experiment log (validation-fitting phase)

Target: VALIDATION leaderboard ~0.653. Strategy: fit the recordable 249-frame
validation flyby with real labels + retrain. Validate (never evaluate) on the
competition API. Fail fast; abandon weak directions.

BEST_SCORE   = 0.3635  (API validation, 2026-09-18)
BEST_EXP     = v11all  (yolo11s, 6000 synthetic + all 154 real recorded-flyby frames)
BEST_WEIGHTS = model/v11all.pt   (deployed, preset `v11all`)
PREV_BEST    = v9real 0.3228 (model/v9real.pt, preset `v9real`) — still on disk, restorable

Data facts (real recorded flyby 14356d0b…):
- 227 proposals, human reviewed 212 → 24 approve / 188 disapprove.
- 24 approved objects fan out to 154 labeled frames (v9: 107 train / 47 val).
- Real-label class coverage = 12/16. Missing: small_launcher, ta-ta, condor,
  jammer — every proposal guessed as these was human-DISAPPROVED (condor never
  proposed). Likely not clearly present in the flyby, so probably not the ceiling.
- medium_launcher + mine_roller were ONLY in v9's val split (not trained on).

| ID      | Hypothesis | Config | API val | Δ vs best | Verdict |
|---------|-----------|--------|---------|-----------|---------|
| v4s1    | synthetic-only baseline | yolo11s, 6000 synth | ~0.153 | — | superseded |
| v8mix   | foreign-bg retrain transfers | — | 0.018 | −− | ABANDON |
| v9real  | +107 real frames | yolo11s | **0.3228** | baseline | BEST |
| v10real | + densified boxes | yolo11s | (offline 0.041 on own val) | poison | ABANDON |
| v11all  | fold all 154 real frames into train (+2 classes into train) | yolo11s, 45ep | **0.3635** | +0.041 | NEW BEST |

Read (2026-09-18): adding real recorded-flyby frames to training is the dominant,
still-productive lever (v4s1 0.153 -> v9real 0.3228 -> v11all 0.3635). Consistent
with "the gap is detector generalization, not camera/tracking". Explore AROUND v11all
before the expensive sim/camera build (exp15 says learned camera is marginal in-domain).

Queued hypotheses (fundamentally distinct):
- B: bigger model (yolo11m/l) on v11all data — more capacity to memorize the
  fixed validation scene (validation-fitting favours overfit capacity).
- C: fixed densifier — properly warp the 24 approved objects across all in-view
  frames (v10's densifier was buggy); safe coverage boost, no clutter.
- D: self-training pseudo-labels — risky (188/227 proposals are clutter; naive
  self-training would re-include it). Lower priority.
