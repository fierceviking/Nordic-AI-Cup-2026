"""Monitor a model against the recorded competition frames during training.

There is no ground truth for the recorded sequence, so mAP cannot be computed
on it. The failure that matters is measurable anyway: the detector emitted
**93 boxes per frame** on that scene where about a dozen objects exist. That
count needs no labels.

What is reported, per call:

* ``det_per_frame``  - boxes at the deployed confidence floor. The number to
  watch. It was 93; a healthy detector should be near the object count.
* ``det_conf25``     - boxes above 0.25, i.e. confident mistakes rather than
  noise. Raising the threshold did not fix the original failure, so this
  matters more than the raw count.
* ``empty_frames``   - frames answered with nothing at all. Zero of these on a
  cluttered scene is itself a warning sign.
* ``median_conf`` / ``p90_conf``

**This is a readout, not a selection criterion.** Choosing checkpoints by it
would start fitting the held-out set, which is the mistake this whole exercise
is trying to avoid. `best.pt` stays selected on the real-frame validation split.
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))

RECORDINGS = LAB / 'recordings'


def sample_views(directory: Path = RECORDINGS, count: int = 48) -> List[Path]:
    """A fixed, evenly spaced sample of recorded views, so runs are comparable."""
    views: List[Path] = []
    if not directory.is_dir():
        return views
    for sequence in sorted(p for p in directory.iterdir() if p.is_dir()):
        views.extend(sorted((sequence / 'views').glob('*.png')))
    if not views:
        return views
    step = max(1, len(views) // count)
    return views[::step][:count]


class ClutterMonitor:
    """Counts what a model finds on terrain that should hold almost nothing."""

    def __init__(self, views: Optional[List[Path]] = None, confidence: float = 0.05,
                 imgsz: int = 960, out: Optional[Path] = None):
        self.views = views if views is not None else sample_views()
        self.confidence = confidence
        self.imgsz = imgsz
        self.out = out
        self._images = None

    def available(self) -> bool:
        return len(self.views) > 0

    def images(self):
        if self._images is None:
            import cv2
            self._images = [cv2.imread(str(p)) for p in self.views]
            self._images = [i for i in self._images if i is not None]
        return self._images

    def measure(self, model, epoch: int = -1) -> dict:
        images = self.images()
        if not images:
            return {}
        counts, confident, confidences, empty = [], [], [], 0
        for i in range(0, len(images), 8):
            batch = images[i:i + 8]
            results = model.predict(batch, imgsz=self.imgsz, conf=self.confidence,
                                    iou=0.55, verbose=False, max_det=300)
            for result in results:
                scores = (result.boxes.conf.cpu().numpy()
                          if result.boxes is not None and len(result.boxes)
                          else np.zeros(0))
                counts.append(len(scores))
                confident.append(int((scores >= 0.25).sum()))
                confidences.extend(scores.tolist())
                empty += 1 if len(scores) == 0 else 0

        record = {
            'epoch': epoch,
            'frames': len(counts),
            'det_per_frame': round(float(np.mean(counts)), 2),
            'det_conf25': round(float(np.mean(confident)), 2),
            'empty_frames': empty,
            'median_conf': round(float(np.median(confidences)), 4) if confidences else 0.0,
            'p90_conf': round(float(np.percentile(confidences, 90)), 4) if confidences else 0.0,
        }
        if self.out:
            self.out.parent.mkdir(parents=True, exist_ok=True)
            with open(self.out, 'a') as handle:
                handle.write(json.dumps(record) + '\n')
        return record


def attach(model, every: int = 5, out: Optional[Path] = None,
           confidence: float = 0.05, imgsz: int = 960):
    """Report clutter behaviour every ``every`` epochs during training."""
    monitor = ClutterMonitor(confidence=confidence, imgsz=imgsz, out=out)
    if not monitor.available():
        print('[clutter] no recordings found; monitoring disabled')
        return None

    print(f'[clutter] monitoring {len(monitor.views)} recorded validation views '
          f'every {every} epochs (readout only, not used for checkpoint selection)')

    def on_fit_epoch_end(trainer):
        epoch = int(trainer.epoch) + 1
        if epoch % every and epoch != trainer.epochs:
            return
        try:
            record = monitor.measure(trainer.ema.ema if trainer.ema else trainer.model,
                                     epoch)
            if record:
                print(f'[clutter] epoch {epoch:3d}  '
                      f'det/frame {record["det_per_frame"]:7.2f}  '
                      f'conf>=0.25 {record["det_conf25"]:6.2f}  '
                      f'empty {record["empty_frames"]:3d}/{record["frames"]}  '
                      f'median conf {record["median_conf"]:.3f}')
        except Exception as exc:
            print(f'[clutter] monitor failed at epoch {epoch}: {exc}')

    model.add_callback('on_fit_epoch_end', on_fit_epoch_end)
    return monitor


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--conf', type=float, default=0.05)
    parser.add_argument('--count', type=int, default=48)
    arguments = parser.parse_args()

    from ultralytics import YOLO

    monitor = ClutterMonitor(views=sample_views(count=arguments.count),
                             confidence=arguments.conf)
    if not monitor.available():
        raise SystemExit('no recorded views under lab/recordings')
    record = monitor.measure(YOLO(arguments.weights))
    print(f'{arguments.weights}')
    for key, value in record.items():
        print(f'  {key:16s} {value}')
    print('\n  for reference, the model that scored 0.102 on the competition '
          'emitted ~93 boxes per frame on this scene')
