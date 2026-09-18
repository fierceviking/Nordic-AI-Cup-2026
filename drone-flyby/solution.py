"""Drone-flyby solution: zoom detector + motion-model world memory + camera policy.

Why this beats a stateless per-image detector:

* You must answer for the WHOLE source frame every frame, but the camera only
  shows a 960x540 crop. A stateless detector can only report what is inside the
  crop, so it either stays at L0 (everything tiny, small objects unrecognisable)
  or zooms (loses coverage of the rest of the frame). Either way most of the
  per-frame ground truth is missed.

* Objects are static on the ground and the drone flies a straight line, so each
  object drifts across the full frame by a nearly-constant per-object velocity
  (measured: dy sd ~3-12 px/frame). A constant-velocity track keeps IoU>=0.5 for
  ~3-4 frames after the last observation.

So we ZOOM to see objects big enough for the detector, remember every object in a
world model keyed in full-frame source pixels, predict each object forward with
its own velocity, and report ALL of them every frame -- inside the crop or not.
The camera policy points where observations are going stale.
"""
import logging
import math
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from dtos import (
    MAXIMUM_CENTER_DELTA_PIXELS,
    SOURCE_REGION_SIZES,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    OBJECT_CLASSES,
    DroneFlybyPredictionDto,
    DroneFlybyPredictRequestDto,
    DroneFlybyPredictResponseDto,
    RequestedViewDto,
)
from utils import clip_bbox_to_frame, decode_view, view_bbox_to_global

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Config (overridable via env)
# --------------------------------------------------------------------------- #
WEIGHTS = os.environ.get('DF_WEIGHTS', os.path.join(
    os.path.dirname(__file__), 'experiments', 'runs', 'detect', 'runs', 'det_s', 'weights', 'best.pt'))
DET_CONF = float(os.environ.get('DF_CONF', '0.20'))
DET_IMGSZ = int(os.environ.get('DF_IMGSZ', '960'))
MAX_TRACK_GAP = int(os.environ.get('DF_MAX_GAP', '6'))   # drop track after this many unseen frames
CONF_DECAY = float(os.environ.get('DF_DECAY', '0.85'))    # per-frame confidence decay for predictions
POLICY = os.environ.get('DF_POLICY', 'l0')                # 'l0' | 'hybrid' | 'stale' | 'l1sweep' | 'lr'

# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
# All GPU inference (load, warmup, serving) runs on ONE dedicated thread. The
# API endpoint is a sync FastAPI route, which uvicorn dispatches on an anyio
# threadpool; CUDA/cuDNN pay a first-call initialisation cost *per thread*, so
# if warmup and serving happened on different threads the first served frame
# would stall ~2s -- and at 3 fps the realtime clock would skip the opening
# frames of the sequence (measured: 0.643 -> 0.302). Pinning every inference to
# one pre-warmed thread makes the first real frame as fast as the rest.
_INFER_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix='df-infer')
WARMUP_ITERS = int(os.environ.get('DF_WARMUP_ITERS', '4'))


class Detector:
    _lock = threading.Lock()
    _model = None

    @classmethod
    def _load_and_warm(cls):
        from ultralytics import YOLO
        logger.info('Loading detector %s', WEIGHTS)
        m = YOLO(WEIGHTS)
        dummy = np.zeros((540, 960, 3), dtype=np.uint8)
        for _ in range(max(1, WARMUP_ITERS)):
            m.predict(dummy, imgsz=DET_IMGSZ, conf=DET_CONF, verbose=False, device=0)
        return m

    @classmethod
    def get(cls):
        if cls._model is None:
            with cls._lock:
                if cls._model is None:
                    # Load + warm ON the inference thread, so that thread is the
                    # one that pays (and caches) the CUDA/cuDNN init.
                    cls._model = _INFER_POOL.submit(cls._load_and_warm).result()
        return cls._model

    @classmethod
    def _detect_sync(cls, image_bgr) -> List[Tuple[str, Tuple[float, float, float, float], float]]:
        m = cls.get()
        h, w = image_bgr.shape[:2]
        res = m.predict(image_bgr, imgsz=DET_IMGSZ, conf=DET_CONF, iou=0.5,
                        verbose=False, device=0)[0]
        out = []
        if res.boxes is None:
            return out
        for b in res.boxes:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            out.append((OBJECT_CLASSES[cls_id],
                        (x1 / w, y1 / h, x2 / w, y2 / h), conf))
        return out

    @classmethod
    def detect(cls, image_bgr) -> List[Tuple[str, Tuple[float, float, float, float], float]]:
        """Return (class_name, view_normalized_xyxy, confidence) for a 960x540 view.

        Routed through the single inference thread so serving never hits a cold
        thread (see _INFER_POOL note above)."""
        return _INFER_POOL.submit(cls._detect_sync, image_bgr).result()


# --------------------------------------------------------------------------- #
# World model (per-object motion tracks in full-frame SOURCE pixels)
# --------------------------------------------------------------------------- #
def _iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class Track:
    cls: str
    bbox: Tuple[float, float, float, float]      # last observed, source px
    vel: Tuple[float, float, float, float]        # per-frame
    last_frame: int
    conf: float
    hits: int = 1

    def predict(self, frame: int) -> Tuple[float, float, float, float]:
        k = frame - self.last_frame
        return tuple(self.bbox[i] + self.vel[i] * k for i in range(4))


class WorldModel:
    """Tracks every object seen so far and predicts positions for unseen frames."""

    def __init__(self):
        self.tracks: List[Track] = []

    def update(self, frame: int, detections):
        """detections: list of (class, source_xyxy, conf)."""
        used = set()
        for cls, box, conf in detections:
            # associate to best same-class track by IoU of predicted position
            best_i, best_iou = -1, 0.0
            for i, t in enumerate(self.tracks):
                if i in used or t.cls != cls:
                    continue
                iou = _iou(box, t.predict(frame))
                # also allow association by proximity for fast/large motion
                if iou > best_iou:
                    best_iou, best_i = iou, i
            if best_i >= 0 and (best_iou > 0.1 or self._same_class_unique(cls)):
                t = self.tracks[best_i]
                gap = max(1, frame - t.last_frame)
                new_vel = tuple((box[i] - t.bbox[i]) / gap for i in range(4))
                # EMA smoothing of velocity
                a = 0.5
                t.vel = tuple(a * new_vel[i] + (1 - a) * t.vel[i] for i in range(4))
                t.bbox = box
                t.last_frame = frame
                t.conf = conf
                t.hits += 1
                used.add(best_i)
            else:
                self.tracks.append(Track(cls=cls, bbox=box,
                                         vel=(0.0, 0.0, 0.0, 0.0),
                                         last_frame=frame, conf=conf))
        # drop stale tracks
        self.tracks = [t for t in self.tracks if frame - t.last_frame <= MAX_TRACK_GAP]

    def _same_class_unique(self, cls):
        return sum(1 for t in self.tracks if t.cls == cls) == 1

    def report(self, frame: int):
        """Return (class, source_xyxy, conf) for every live track at this frame."""
        out = []
        for t in self.tracks:
            gap = frame - t.last_frame
            box = t.predict(frame) if gap > 0 else t.bbox
            conf = t.conf * (CONF_DECAY ** gap)
            out.append((t.cls, box, max(0.03, conf)))
        return out


# --------------------------------------------------------------------------- #
# Camera policy
# --------------------------------------------------------------------------- #
def _center_bounds(level):
    w, h = SOURCE_REGION_SIZES[level]
    return (w // 2, IMAGE_WIDTH - w // 2, h // 2, IMAGE_HEIGHT - h // 2)


def _step_toward(cur, tgt, limit):
    dx, dy = tgt[0] - cur[0], tgt[1] - cur[1]
    d = math.hypot(dx, dy)
    if d <= limit or d == 0:
        return (int(tgt[0]), int(tgt[1]))
    return (int(cur[0] + dx / d * limit), int(cur[1] + dy / d * limit))


class CameraPolicy:
    """Point the camera to keep the most stale / least seen regions fresh."""

    def __init__(self):
        self.phase = 0

    def choose(self, request, world: WorldModel) -> Optional[RequestedViewDto]:
        cur_level = request.view.resolution_level
        cx, cy = request.view.center_x, request.view.center_y
        frame = request.frame

        # Always operate at L1 (quarter frame): good balance of detail+coverage.
        # From L0 we can only reach L1; from L2 we can reach L1. So target L1.
        target_level = 1
        limit = MAXIMUM_CENTER_DELTA_PIXELS.get(cur_level, 1102.0)
        mnx, mxx, mny, mxy = _center_bounds(target_level)

        if POLICY == 'l0':
            # Stay at the full frame every frame (isolates detector-at-L0 quality).
            return RequestedViewDto(resolution_level=0, center_x=1920, center_y=1080)
        if POLICY == 'hybrid':
            return self._hybrid(request, world)
        if POLICY == 'lr':
            # oscillate left/right at mid height
            targets = [(mnx, 1080), (mxx, 1080)]
            tgt = targets[(frame // 3) % 2]
        elif POLICY == 'l1sweep':
            grid = [(mnx, mny), (mxx, mny), (mxx, mxy), (mnx, mxy)]
            tgt = grid[(frame // 3) % 4]
        else:  # 'stale': aim at the region with the stalest / newest-entering objects
            tgt = self._stale_target(frame, world, (mnx, mxx, mny, mxy), (cx, cy))

        nx, ny = _step_toward((cx, cy), tgt, limit)
        nx = min(max(nx, mnx), mxx)
        ny = min(max(ny, mny), mxy)
        if cur_level == 0:
            # from L0, only legal L1 center reachable within limit; step ok
            pass
        return RequestedViewDto(resolution_level=target_level, center_x=int(nx), center_y=int(ny))

    def _hybrid(self, request, world):
        """Interleave an L0 full-frame refresh (all big objects at once) with L1
        zoom excursions to the stalest region (small objects)."""
        cur_level = request.view.resolution_level
        cx, cy = request.view.center_x, request.view.center_y
        fi = request.frame_index
        P = int(os.environ.get('DF_L0_PERIOD', '4'))
        mnx, mxx, mny, mxy = _center_bounds(1)
        limit = MAXIMUM_CENTER_DELTA_PIXELS.get(cur_level, 1102.0)
        # full-frame refresh slot
        if fi % P == 0:
            if cur_level == 0:
                pass  # already refreshed; fall through to zoom this frame
            elif cur_level == 2:
                tgt = _step_toward((cx, cy), (1920, 1080), limit)
                return RequestedViewDto(resolution_level=1,
                                        center_x=int(min(max(tgt[0], mnx), mxx)),
                                        center_y=int(min(max(tgt[1], mny), mxy)))
            else:
                return RequestedViewDto(resolution_level=0, center_x=1920, center_y=1080)
        tgt = self._stale_target(request.frame, world, (mnx, mxx, mny, mxy), (cx, cy))
        nx, ny = _step_toward((cx, cy), tgt, limit)
        return RequestedViewDto(resolution_level=1,
                                center_x=int(min(max(nx, mnx), mxx)),
                                center_y=int(min(max(ny, mny), mxy)))

    def _stale_target(self, frame, world, bounds, cur):
        mnx, mxx, mny, mxy = bounds
        # candidate centers: a 3x3 grid of L1 centers
        xs = [mnx, (mnx + mxx) // 2, mxx]
        ys = [mny, (mny + mxy) // 2, mxy]
        cands = [(x, y) for y in ys for x in xs]
        w1, h1 = SOURCE_REGION_SIZES[1]

        def score(c):
            rx1, ry1, rx2, ry2 = c[0] - w1 // 2, c[1] - h1 // 2, c[0] + w1 // 2, c[1] + h1 // 2
            s = 0.0
            for t in world.tracks:
                bx = t.predict(frame)
                bcx, bcy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
                if rx1 <= bcx <= rx2 and ry1 <= bcy <= ry2:
                    s += (frame - t.last_frame)      # reward stale tracks
            # bias toward the top band, where new objects enter (drone flies +y)
            s += 0.5 * (1.0 - c[1] / IMAGE_HEIGHT)
            # small penalty for distance (can't reach far anyway)
            s -= 0.0005 * math.hypot(c[0] - cur[0], c[1] - cur[1])
            return s

        return max(cands, key=score)


# --------------------------------------------------------------------------- #
# Per-sequence state
# --------------------------------------------------------------------------- #
_worlds: Dict[str, WorldModel] = defaultdict(WorldModel)
_policies: Dict[str, CameraPolicy] = defaultdict(CameraPolicy)
_state_lock = threading.Lock()


def predict(request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
    # --- diagnostics (DF_DIAG): probe whether the evaluator score responds to
    # prediction content at all. 'empty' => send no boxes; 'onebox' => send one
    # fixed high-confidence box every frame. Remove once the anomaly is resolved.
    _diag = os.environ.get('DF_DIAG')
    if _diag == 'empty':
        return DroneFlybyPredictResponseDto(
            request_id=request.request_id, frame=request.frame,
            annotations=[], requested_view=None)
    if _diag == 'onebox':
        return DroneFlybyPredictResponseDto(
            request_id=request.request_id, frame=request.frame,
            annotations=[DroneFlybyPredictionDto(
                object_id='hangar', bbox=[0.4, 0.4, 0.6, 0.6], confidence=0.9)],
            requested_view=None)

    with _state_lock:
        world = _worlds[request.sequence_id]
        policy = _policies[request.sequence_id]

    try:
        image = decode_view(request.view)
        dets_view = Detector.detect(image)
        # lift to global source pixels
        src_region = request.view.source_region_xyxy
        dets_global = []
        for cls, vbox, conf in dets_view:
            gx1, gy1, gx2, gy2 = view_bbox_to_global(
                vbox, src_region, request.original_width, request.original_height)
            # global normalized -> source px
            sx1 = gx1 * request.original_width
            sy1 = gy1 * request.original_height
            sx2 = gx2 * request.original_width
            sy2 = gy2 * request.original_height
            if sx2 - sx1 < 1 or sy2 - sy1 < 1:
                continue
            dets_global.append((cls, (sx1, sy1, sx2, sy2), conf))

        world.update(request.frame, dets_global)
        reports = world.report(request.frame)

        annotations = []
        for cls, sbox, conf in reports:
            # source px -> global normalized
            gb = (sbox[0] / request.original_width, sbox[1] / request.original_height,
                  sbox[2] / request.original_width, sbox[3] / request.original_height)
            gb = clip_bbox_to_frame(gb)
            if gb is None:
                continue
            annotations.append(DroneFlybyPredictionDto(
                object_id=cls, bbox=list(gb), confidence=round(float(min(1.0, max(0.0, conf))), 4)))
        annotations = annotations[:500]
    except Exception:
        logger.exception('Detector/tracker failed on frame %s', request.frame)
        annotations = []

    try:
        requested_view = policy.choose(request, world)
    except Exception:
        logger.exception('Camera policy failed on frame %s', request.frame)
        requested_view = None

    return DroneFlybyPredictResponseDto(
        request_id=request.request_id,
        frame=request.frame,
        annotations=annotations,
        requested_view=requested_view,
    )
