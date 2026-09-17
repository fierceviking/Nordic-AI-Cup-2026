# Solution

What is in this folder beyond the supplied use case, how to run it, and how to
retrain it. The reasoning and the measurements behind every choice are in
[EXPERIMENTS.md](EXPERIMENTS.md).

## The short version

| piece | what it does |
|---|---|
| `solution.py` | Detector, world model, camera policy. All the logic. |
| `example.py` | Adapter between the wire protocol and `solution.py`. |
| `model/best.pt` | The trained detector. Required at import. |
| `lab/` | Everything used to get there: analysis, data synthesis, training, evaluation, profiling. |
| `EXPERIMENTS.md` | The log of every approach tried and what it scored. |

Three ideas carry the result:

1. **The flight is one constant homography.** The drone translates at a fixed
   rate over planar ground with a fixed camera, so frame *i* and frame *i+1*
   are related by a single matrix, measured once and accurate to a median of
   0.73 px. A detection can be carried forward for ten frames on it and still
   clear IoU 0.50 about 85 % of the time.
2. **So keep a world model.** Every response must cover the whole source frame.
   Tracked objects are warped forward each frame whether or not the camera is
   looking at them. Worth +0.34 mAP when the camera zooms, +0.02 when it does
   not.
3. **Then mostly do not zoom.** The detector reads the small classes out of the
   4x-downsampled full view well enough that answering for the entire frame
   every frame beats observing a quarter of it in detail. This was the
   surprise; see §7 of the log.

## Running it

```cmd
pip install -r requirements.txt
python api.py
```

Then, in a second terminal:

```cmd
python local_evaluator.py              # offline, every frame
python local_evaluator.py --realtime   # with the 3 fps clock
python local_evaluator.py --oracle     # scorer sanity check, prints 1.000
```

The detector loads and warms up at import, so the first request is not charged
for it. Give the server ~20 s to start before pointing the evaluator at it.

### Configuration

Read from the environment, so nothing needs editing to redeploy:

| variable | default | meaning |
|---|---|---|
| `DRONE_WEIGHTS` | `model/best.pt` | detector checkpoint |
| `DRONE_DEVICE` | `0` | CUDA device index, or `cpu` |
| `DRONE_IMGSZ` | `960` | inference size; 960 is native and measured best |
| `DRONE_CONF` | `0.08` | detector confidence floor |

### Deploying

```cmd
docker build -t drone-flyby .
docker run --gpus all -p 9053:9053 drone-flyby
```

Submit `http://<your-host>:9053/predict` — the path matters.

**Check `/health` before every attempt.** It reports `solver_loaded`. A server
whose detector failed to load still answers every request with HTTP 200 and an
empty annotation list, which scores zero and is indistinguishable from a bad
model. This happened here once; the guard exists because of it.

Deploy on a **cloud VM**, not a tunnel from a workstation. Measured through a
Cloudflare quick tunnel, the round trip was 374 ms against a 333 ms frame
interval — 11 of 12 requests slower than one frame, so roughly every second
frame would be dropped and scored as empty. The detector itself is 39 ms of
that; the rest is transport. See §13 of the experiment log.

## Talking to the competition API

`lab/cup_api.py` wraps it. The key is read from `NORDIC_API_KEY` or from a
git-ignored `.nordic_key` file, and is never printed or logged.

```cmd
python lab\cup_api.py status
python lab\cup_api.py verify   --url https://<host>/predict
python lab\cup_api.py validate --url https://<host>/predict --watch
```

There is deliberately no `evaluate` command: that is one attempt per team for
the whole competition and should not be reachable by a typo.

## Recording the validation sequence

The API serves no images — the service pushes them to your endpoint, and the
use case explicitly allows keeping them. Set `DRONE_RECORD_DIR` and every
transmitted view is written as a PNG with a JSON sidecar (geometry plus our own
answer), on a background thread with a bounded queue so it can never cost a
frame.

```cmd
set DRONE_RECORD_DIR=lab\recordings
python api.py
```

A 249-frame validation sequence over a *different location* is the single most
useful thing available for the weakness in the training data: the 25 supplied
frames cover about 1.7 frames' worth of unique ground, and background diversity
is what limits generalisation.

## Retraining

Everything runs from the 25 supplied frames. Order matters.

```cmd
python lab\extract_sprites.py                              # 1. cut 236 RGBA sprites
python lab\make_dataset.py --train 7000 --val 700 --out lab\dataset_v2
python lab\train_yolo.py --data lab\dataset_v2\data.yaml --name v2 --epochs 45
copy lab\runs\v2\weights\best.pt model\best.pt
```

Synthesis is CPU-bound and takes a while. If it has to be interrupted,
`lab\finish_dataset.py --out lab\dataset_v2 --val 300` keeps the train images
already written, drops any unpaired ones, adds a validation split and writes
`data.yaml`.

## Measuring

The lab has three levels of evaluation, deliberately separate, because they
answer different questions.

```cmd
python lab\check_harness.py             # push ground truth through: must print 1.000
python lab\exp05_inference_sweep.py --weights model\best.pt --levels 0,1,2
                                        # detector alone, camera problem removed
python lab\eval_pipeline.py --weights model\best.pt
                                        # full solver, every camera policy and ablation
python lab\probe_sequence.py            # per-frame latency over HTTP
```

`lab/eval_pipeline.py` replicates `local_evaluator.py` in-process — same camera
rules, same scorer, no HTTP — which is what makes it practical to compare a
dozen variants.

## A caution about the numbers

The local score is measured on the same 25 frames the training sprites were cut
from. The model has seen these exact 16 object instances, so **the local number
is optimistic and should not be read as a prediction of the evaluation score**.

Where a choice was close, the more robust option was taken rather than the
higher-scoring one. The clearest case is the camera policy: parking at Level 0
scored 0.9475 and the shipped `L0 x3 : L1 x1` loop scored 0.9363, and the loop
was shipped anyway, because a 2x linear resolution advantage on each quadrant
is real information that cannot be overfitted away, and there is only one
evaluation attempt.
