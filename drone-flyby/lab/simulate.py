"""In-process replay of a scene: same camera rules and same scorer, no HTTP.

``local_evaluator.py`` is the ground truth for correctness, but it needs a live
server and a round trip per frame. This module drives a solver object directly
so an experiment is a function call, which is what makes iterating on the
detector practical.

Also provides ``oracle_camera_frames`` for measuring detector quality on its
own: it tiles the whole frame at a fixed level so the score is not entangled
with the camera policy.
"""

import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtos import (  # noqa: E402
    ALLOWED_RESOLUTION_LEVELS,
    FULL_FRAME_CENTER,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MAXIMUM_CENTER_DELTA_PIXELS,
    SOURCE_REGION_SIZES,
    TRANSMITTED_VIEW_SIZE,
)
from local_evaluator import Camera, CameraRejection, score  # noqa: E402
from utils import center_bounds_for_level, frame_numbers, load_annotations, load_frame  # noqa: E402

DEFAULT_SCENE = 'helsinki'


class FrameCache:
    """Full frames are 24 MB each; keep them around between experiments."""

    def __init__(self, scene: str = DEFAULT_SCENE):
        self.scene = scene
        self._frames: Dict[int, np.ndarray] = {}

    def frame(self, number: int) -> np.ndarray:
        if number not in self._frames:
            self._frames[number] = load_frame(number, self.scene)
        return self._frames[number]


def render_view(frame_image: np.ndarray, level: int, center_x: int, center_y: int):
    """Crop and downsample exactly as the evaluator does, returning a BGR array."""
    width, height = SOURCE_REGION_SIZES[level]
    x1 = center_x - width // 2
    y1 = center_y - height // 2
    view = frame_image[y1:y1 + height, x1:x1 + width]
    if (view.shape[1], view.shape[0]) != TRANSMITTED_VIEW_SIZE:
        view = cv2.resize(view, TRANSMITTED_VIEW_SIZE, interpolation=cv2.INTER_AREA)
    return view, (x1, y1, x1 + width, y1 + height)


class SimpleRequest:
    """Just enough of DroneFlybyPredictRequestDto for a solver to work with."""

    __slots__ = ('frame', 'frame_index', 'image', 'source_region_xyxy',
                 'resolution_level', 'center_x', 'center_y', 'sequence_id',
                 'original_width', 'original_height', 'allowed_levels',
                 'maximum_center_delta')

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def replay(
    solver: Callable[[SimpleRequest], Tuple[List[dict], Optional[Tuple[int, int, int]]]],
    scene: str = DEFAULT_SCENE,
    cache: Optional[FrameCache] = None,
    verbose: bool = False,
) -> Tuple[Dict[int, List[dict]], Dict[str, float]]:
    """Run a solver over a scene.

    ``solver`` receives a SimpleRequest and returns
    ``(annotations, requested_view)`` where annotations are dicts with
    ``object_id``, ``bbox`` in **source pixels**, ``confidence``, and
    ``requested_view`` is ``(level, center_x, center_y)`` or None.
    """
    cache = cache or FrameCache(scene)
    frames = frame_numbers(scene)
    camera = Camera()
    predictions: Dict[int, List[dict]] = {}
    latencies: List[float] = []
    refused = 0

    for index, number in enumerate(frames):
        image = cache.frame(number)
        view, region = render_view(
            image, camera.resolution_level, camera.center_x, camera.center_y
        )
        request = SimpleRequest(
            frame=number,
            frame_index=index,
            image=view,
            source_region_xyxy=region,
            resolution_level=camera.resolution_level,
            center_x=camera.center_x,
            center_y=camera.center_y,
            sequence_id=scene,
            original_width=IMAGE_WIDTH,
            original_height=IMAGE_HEIGHT,
            allowed_levels=list(ALLOWED_RESOLUTION_LEVELS[camera.resolution_level]),
            maximum_center_delta=MAXIMUM_CENTER_DELTA_PIXELS[camera.resolution_level],
        )
        started = time.perf_counter()
        annotations, requested = solver(request)
        latencies.append((time.perf_counter() - started) * 1000.0)
        predictions[number] = annotations

        if verbose:
            print(f'  frame {number:3d} L{camera.resolution_level} '
                  f'({camera.center_x:4d},{camera.center_y:4d}) '
                  f'-> {len(annotations):3d} det, {latencies[-1]:6.1f} ms')

        if requested is not None:
            try:
                camera.apply(int(requested[0]), int(requested[1]), int(requested[2]))
            except CameraRejection as exc:
                refused += 1
                if verbose:
                    print(f'    camera refused: {exc}')

    stats = {
        'mean_ms': float(np.mean(latencies)),
        'max_ms': float(np.max(latencies)),
        'p90_ms': float(np.percentile(latencies, 90)),
        'refused': refused,
    }
    return predictions, stats


def evaluate(predictions, scene: str = DEFAULT_SCENE):
    """COCO mAP@0.50 exactly as local_evaluator computes it."""
    return score(scene, predictions)


def report(name: str, predictions, stats=None, scene: str = DEFAULT_SCENE) -> float:
    map50, per_class = evaluate(predictions, scene)
    print(f'\n=== {name} ===')
    print(f'  mAP@0.50 = {map50:.4f}')
    for cls, ap in sorted(per_class.items(), key=lambda kv: -kv[1]):
        print(f'    {cls:16s} {ap:.4f}')
    if stats:
        print(f'  latency mean {stats["mean_ms"]:.1f} ms / p90 '
              f'{stats["p90_ms"]:.1f} ms / max {stats["max_ms"]:.1f} ms'
              f'   camera refusals {stats["refused"]}')
    return map50


# --------------------------------------------------------------------------- #
# Detector-only evaluation: tile the frame, ignore the camera problem
# --------------------------------------------------------------------------- #

def tile_centers(level: int) -> List[Tuple[int, int]]:
    """Centres that tile the whole frame at a level, as the camera would see it."""
    width, height = SOURCE_REGION_SIZES[level]
    min_x, max_x, min_y, max_y = center_bounds_for_level(level)
    xs = list(range(min_x, max_x + 1, width)) or [min_x]
    ys = list(range(min_y, max_y + 1, height)) or [min_y]
    if xs[-1] != max_x:
        xs.append(max_x)
    if ys[-1] != max_y:
        ys.append(max_y)
    return [(x, y) for y in ys for x in xs]


def oracle_camera_frames(level: int, scene: str = DEFAULT_SCENE,
                         cache: Optional[FrameCache] = None,
                         frames: Optional[Sequence[int]] = None):
    """Yield (frame, view_image, source_region) tiling every frame at a level."""
    cache = cache or FrameCache(scene)
    for number in (frames if frames is not None else frame_numbers(scene)):
        image = cache.frame(number)
        for cx, cy in tile_centers(level):
            view, region = render_view(image, level, cx, cy)
            yield number, view, region
