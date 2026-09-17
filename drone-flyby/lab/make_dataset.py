"""Build a YOLO dataset by simulating the evaluator's imaging chain.

There are 25 frames and 16 object instances in total, which is far too little to
train a detector on directly. The way around it is to treat the supplied scene
as a *sprite library plus a background library* and synthesise views.

Every generated image goes through exactly the chain the evaluator uses:

    pick a source region of size (960*f, 540*f) from a 3840x2160 frame
      -> paste augmented sprites into it at source resolution
      -> cv2.resize to 960x540 with INTER_AREA
      -> that is the training image

with f = 4, 2, 1 for resolution levels 0, 1, 2. Pasting before the downsample
is the point: it reproduces the sub-pixel blur that makes the small classes
hard, instead of pasting crisp sprites onto an already-downsampled image.

Real objects that happen to fall inside the chosen region keep their true
labels, so each image mixes genuine objects in genuine context with pasted
ones.
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtos import OBJECT_CLASSES  # noqa: E402
from utils import frame_numbers, load_annotations, load_frame  # noqa: E402

SCENE = 'helsinki'
LAB = Path(__file__).resolve().parent
OUT = LAB / 'out'
SPRITES = OUT / 'sprites'
DATASET = LAB / 'dataset'

VIEW_W, VIEW_H = 960, 540
CLASS_INDEX = {name: i for i, name in enumerate(OBJECT_CLASSES)}

# Level -> source-pixels-per-transmitted-pixel, and how often to sample it.
LEVEL_FACTOR = {0: 4, 1: 2, 2: 1}
LEVEL_WEIGHTS = {0: 0.15, 1: 0.60, 2: 0.25}

# An object must have at least this fraction of its area inside the canvas to
# be labelled. Objects enter and leave the source frame at its edges, so the
# model has to be able to detect a partially visible one.
MINIMUM_VISIBLE = 0.35
EDGE_PASTE_PROBABILITY = 0.18
# Blend modes drawn per paste; variety matters more than realism.
BLEND_MODES = ('hard', 'feather', 'blur')
# How often a sprite may land on top of one already placed, and how much of the
# earlier object it may cover. Cut-Paste-Learn measures -10.6 mAP for training
# with no occlusion at all.
OCCLUSION_PROBABILITY = 0.25
MAXIMUM_OCCLUDED = 0.35


# --------------------------------------------------------------------------- #
# Recorded views as background material
# --------------------------------------------------------------------------- #
#
# The 25 supplied frames are forest, field and coastline, and contain almost no
# vehicles, boats or built clutter. A detector trained only on them learns
# "compact man-made thing on terrain", because in training that description fit
# nothing but the pasted objects. Put it over a marina and it labels every boat.
#
# Views recorded from a real attempt (see recorder.py) are the cure: they are
# full of cars, boats, greenhouses and rooftop plant, none of which is a target.
# Pasting onto them makes all of that clutter explicit negative space.
#
# These arrive already downsampled - a Level-0 view is the whole frame at a
# quarter scale - so a sprite has to be shrunk by the level factor before it is
# pasted, rather than pasted at source scale and downsampled with the canvas.

def load_recorded_views(directory: Path) -> List[Tuple[Path, int]]:
    """Return (image path, resolution level) for every recorded view."""
    out: List[Tuple[Path, int]] = []
    for sequence in sorted(p for p in directory.iterdir() if p.is_dir()):
        meta_dir, view_dir = sequence / 'meta', sequence / 'views'
        if not meta_dir.is_dir() or not view_dir.is_dir():
            continue
        for meta_path in sorted(meta_dir.glob('*.json')):
            view_path = view_dir / (meta_path.stem + '.png')
            if not view_path.exists():
                continue
            with open(meta_path) as handle:
                level = json.load(handle).get('resolution_level', 0)
            out.append((view_path, int(level)))
    return out


def shrink_sprite(rgba: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a source-scale sprite to how it looks at a resolution level."""
    if factor <= 1:
        return rgba
    h, w = rgba.shape[:2]
    new_w, new_h = max(2, int(round(w / factor))), max(2, int(round(h / factor)))
    return cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)


def make_recorded_sample(views, library, rng, pastes=(2, 8)):
    """One training image built on a recorded view instead of a Helsinki crop."""
    view_path, level = views[rng.randrange(len(views))]
    canvas = cv2.imread(str(view_path))
    if canvas is None:
        return None, None
    canvas = canvas.copy()
    factor = LEVEL_FACTOR[level]

    if rng.random() < 0.6:
        canvas = photometric_only(canvas, rng)

    labels: List[Tuple[str, Tuple[int, int, int, int]]] = []
    occupied: List[Tuple[int, int, int, int]] = []
    wanted = rng.randint(*pastes)
    names = list(library)
    for _ in range(wanted * 3):
        if len(labels) >= wanted:
            break
        name = rng.choice(names)
        rgba = shrink_sprite(augment_sprite(rng.choice(library[name]), rng), factor)
        h, w = rgba.shape[:2]
        if w >= VIEW_W - 8 or h >= VIEW_H - 8 or w < 3 or h < 3:
            continue
        x = rng.randint(0, VIEW_W - w - 1)
        y = rng.randint(0, VIEW_H - h - 1)
        if any(boxes_overlap((x, y, x + w, y + h), box) for box in occupied):
            continue
        full = opaque_area(rgba)
        box = paste(canvas, rgba, x, y, rng)
        if box is None:
            continue
        if full > 0 and (box[2] - box[0]) * (box[3] - box[1]) < MINIMUM_VISIBLE * full:
            continue
        occupied.append(box)
        labels.append((name, box))

    if rng.random() < 0.25:
        noise = np.random.normal(0, rng.uniform(1.5, 4.0), canvas.shape)
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    yolo_labels = []
    for name, (x1, y1, x2, y2) in labels:
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        yolo_labels.append((CLASS_INDEX[name],
                            (x1 + x2) / 2 / VIEW_W, (y1 + y2) / 2 / VIEW_H,
                            (x2 - x1) / VIEW_W, (y2 - y1) / VIEW_H))
    return canvas, yolo_labels


# --------------------------------------------------------------------------- #
# Sprite library
# --------------------------------------------------------------------------- #

def load_sprites() -> Dict[str, List[np.ndarray]]:
    library: Dict[str, List[np.ndarray]] = {}
    for directory in sorted(SPRITES.iterdir()):
        if not directory.is_dir():
            continue
        images = []
        for path in sorted(directory.glob('*.png')):
            sprite = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if sprite is not None and sprite.shape[2] == 4:
                images.append(sprite)
        if images:
            library[directory.name] = images
    return library


def augment_sprite(sprite: np.ndarray, rng: random.Random) -> np.ndarray:
    """Rotate, rescale and recolour one sprite, keeping its alpha."""
    bgr = sprite[..., :3].astype(np.float32)
    alpha = sprite[..., 3].astype(np.float32)

    # Photometric: the eval scene has different light and a different ground
    # albedo, so the model must not key on absolute colour.
    gain = rng.uniform(0.72, 1.32)
    bias = rng.uniform(-22, 22)
    bgr = bgr * gain + bias
    # per-channel tint
    for channel in range(3):
        bgr[..., channel] *= rng.uniform(0.90, 1.10)
    if rng.random() < 0.30:
        hsv = cv2.cvtColor(np.clip(bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + rng.randint(-12, 12)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.6, 1.3), 0, 255)
        bgr = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8),
                           cv2.COLOR_HSV2BGR).astype(np.float32)
    bgr = np.clip(bgr, 0, 255)

    rgba = np.dstack([bgr, alpha]).astype(np.float32)

    # Geometric. The altitude is fixed at 600 m in every capture, so an object's
    # size on the ground is nearly fixed: only modest scale jitter is realistic.
    scale = rng.uniform(0.80, 1.35)
    angle = rng.uniform(0, 360)
    h, w = rgba.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(h * sin + w * cos) + 2
    new_h = int(h * cos + w * sin) + 2
    matrix[0, 2] += new_w / 2 - w / 2
    matrix[1, 2] += new_h / 2 - h / 2
    rgba = cv2.warpAffine(rgba, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR,
                          borderValue=(0, 0, 0, 0))
    if rng.random() < 0.5:
        rgba = rgba[:, ::-1]
    return rgba


def opaque_area(rgba: np.ndarray) -> float:
    """Area of the sprite's opaque part, ignoring the transparent margin that
    rotation adds."""
    return float((rgba[..., 3] > 32).sum())


def paste(canvas: np.ndarray, rgba: np.ndarray, x: int, y: int,
          rng: random.Random) -> Tuple[int, int, int, int]:
    """Alpha-composite a sprite at (x, y) and return its tight box, or None.

    The blend mode is drawn at random. Cut-Paste-Learn (arXiv:1708.01642)
    measures that mixing modes is worth +7.8 mAP over a single one, while the
    most physically realistic mode alone is 7.5 *worse* than none: what the
    detector must not do is learn one particular paste artefact.
    """
    h, w = rgba.shape[:2]
    ch, cw = canvas.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(cw, x + w), min(ch, y + h)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    sprite = rgba[y1 - y:y2 - y, x1 - x:x2 - x]
    alpha = (sprite[..., 3:4] / 255.0)
    if alpha.max() < 0.2:
        return None

    mode = rng.choice(BLEND_MODES)
    if mode == 'feather':
        soft = cv2.GaussianBlur(alpha[..., 0], (0, 0), rng.uniform(0.6, 1.4))
        alpha = soft[..., None]
    region = canvas[y1:y2, x1:x2].astype(np.float32)
    blended = np.clip(sprite[..., :3] * alpha + region * (1 - alpha), 0, 255)
    if mode == 'blur':
        blended = cv2.GaussianBlur(blended, (3, 3), rng.uniform(0.4, 0.9))
    canvas[y1:y2, x1:x2] = blended.astype(np.uint8)

    ys, xs = np.where(sprite[..., 3] > 32)
    if len(xs) == 0:
        return None
    return (x1 + int(xs.min()), y1 + int(ys.min()),
            x1 + int(xs.max()) + 1, y1 + int(ys.max()) + 1)


# --------------------------------------------------------------------------- #
# Backgrounds
# --------------------------------------------------------------------------- #

def augment_background(canvas: np.ndarray, rng: random.Random) -> np.ndarray:
    """Recolour the terrain hard - the evaluation scene is somewhere else.

    Background diversity is the real bottleneck here: 25 heavily overlapping
    frames cover only about 1.7 frames' worth of unique ground, so the same
    trees and rooflines recur constantly. Wide colour jitter plus a flip stops
    the detector from memorising them.
    """
    out = canvas.astype(np.float32)
    out = out * rng.uniform(0.75, 1.25) + rng.uniform(-22, 22)
    hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + rng.randint(-16, 16)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.55, 1.40), 0, 255)
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    # Flips only: a transpose would change the aspect ratio of the region.
    if rng.random() < 0.5:
        out = out[:, ::-1]
    if rng.random() < 0.5:
        out = out[::-1]
    return np.ascontiguousarray(out)


def photometric_only(canvas: np.ndarray, rng: random.Random) -> np.ndarray:
    """Colour jitter without the flip, for canvases holding real labelled
    objects."""
    out = canvas.astype(np.float32)
    out = out * rng.uniform(0.78, 1.22) + rng.uniform(-18, 18)
    hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + rng.randint(-14, 14)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.60, 1.35), 0, 255)
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


# --------------------------------------------------------------------------- #
# Hard negatives
# --------------------------------------------------------------------------- #
#
# The supplied scene is forest, field and water. Almost nothing in it is a
# compact, geometric, man-made object except the sixteen targets, so
# "compact man-made thing on terrain" separates the training data perfectly and
# is what the detector learns. Over a real town it then boxes every boat, car
# and rooftop unit - measured on the competition validation set, 93 false
# positives per frame.
#
# The recorded validation frames contain exactly the missing clutter, and the
# rules permit keeping them, but they are **deliberately not used here**. They
# are the only held-out data available, and training on them would leave no way
# to measure anything. They stay a test set.
#
# So the fix comes from the generator instead. Two things were missing and both
# are free:
#
#   * images containing **no objects at all**. Every generated image had
#     between three and eleven, so the detector was never once shown that the
#     right answer can be "nothing here".
#   * **distractors**: flat-shaded geometric shapes at target-like sizes,
#     pasted through the same imaging chain and deliberately left *unlabelled*.
#     They stand in for the hulls, car roofs, containers and rooftop plant that
#     the supplied scene does not contain.
#
# Distractors are crude next to a real boat. What they teach is the useful part:
# that being a compact geometric bright thing is not sufficient.

DISTRACTOR_COLOURS = (
    (235, 235, 235), (200, 205, 210), (150, 155, 160), (90, 95, 100),
    (45, 50, 55), (60, 80, 140), (140, 90, 60), (70, 110, 80), (25, 30, 35),
)


def make_distractor(rng: random.Random, size: int) -> np.ndarray:
    """A flat-shaded man-made-looking shape, as BGRA, in source pixels."""
    kind = rng.choice(('hull', 'box', 'ell', 'disc'))
    aspect = {'hull': rng.uniform(2.2, 4.5), 'box': rng.uniform(1.0, 2.2),
              'ell': rng.uniform(1.0, 1.8), 'disc': rng.uniform(1.0, 1.3)}[kind]
    w = max(4, int(size * aspect))
    h = max(4, size)
    canvas = np.zeros((h + 4, w + 4, 4), np.uint8)
    body = DISTRACTOR_COLOURS[rng.randrange(len(DISTRACTOR_COLOURS))]
    body = tuple(int(np.clip(c * rng.uniform(0.8, 1.2), 0, 255)) for c in body)

    if kind == 'hull':
        cv2.ellipse(canvas, (w // 2 + 2, h // 2 + 2), (w // 2, h // 2), 0, 0, 360,
                    (*body, 255), -1)
        inner = tuple(int(np.clip(c * rng.uniform(0.45, 0.75), 0, 255)) for c in body)
        cv2.ellipse(canvas, (int(w * 0.55) + 2, h // 2 + 2),
                    (max(1, w // 5), max(1, h // 4)), 0, 0, 360, (*inner, 255), -1)
    elif kind == 'box':
        cv2.rectangle(canvas, (2, 2), (w + 1, h + 1), (*body, 255), -1)
        if rng.random() < 0.6:
            edge = tuple(int(np.clip(c * rng.uniform(0.5, 0.8), 0, 255)) for c in body)
            cv2.rectangle(canvas, (2, 2), (w + 1, h + 1), (*edge, 255),
                          max(1, size // 8))
    elif kind == 'ell':
        cv2.rectangle(canvas, (2, 2), (w + 1, int(h * 0.55) + 1), (*body, 255), -1)
        cv2.rectangle(canvas, (2, 2), (int(w * 0.5) + 1, h + 1), (*body, 255), -1)
    else:
        cv2.circle(canvas, (w // 2 + 2, h // 2 + 2), max(2, min(w, h) // 2),
                   (*body, 255), -1)

    if rng.random() < 0.5:
        centre = (canvas.shape[1] / 2, canvas.shape[0] / 2)
        matrix = cv2.getRotationMatrix2D(centre, rng.uniform(0, 360), 1.0)
        canvas = cv2.warpAffine(canvas, matrix, (canvas.shape[1], canvas.shape[0]),
                                flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    if rng.random() < 0.4:
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    return canvas


def paste_distractors(canvas, rng, occupied, factor, count):
    """Paste unlabelled clutter, avoiding anything that carries a label."""
    region_h, region_w = canvas.shape[:2]
    for _ in range(count * 3):
        if count <= 0:
            break
        # Sized like the targets are, at source scale, so they are genuinely
        # confusable rather than trivially separable.
        size = int(rng.uniform(14, 90) * factor / 2) or 6
        shape = make_distractor(rng, size)
        h, w = shape.shape[:2]
        if w >= region_w - 8 or h >= region_h - 8:
            continue
        x = rng.randint(0, region_w - w - 1)
        y = rng.randint(0, region_h - h - 1)
        if any(boxes_overlap((x, y, x + w, y + h), box, 6) for box in occupied):
            continue
        if paste(canvas, shape.astype(np.float32), x, y, rng) is None:
            continue
        occupied.append((x, y, x + w, y + h))
        count -= 1


# --------------------------------------------------------------------------- #
# One sample
# --------------------------------------------------------------------------- #

def boxes_overlap(a, b, margin: int = 4) -> bool:
    return not (a[2] + margin < b[0] or b[2] + margin < a[0] or
                a[3] + margin < b[1] or b[3] + margin < a[1])


def covered_fraction(box, other) -> float:
    """How much of ``box`` the rectangle ``other`` would hide."""
    ix1, iy1 = max(box[0], other[0]), max(box[1], other[1])
    ix2, iy2 = min(box[2], other[2]), min(box[3], other[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
    return inter / area


def make_sample(frames: Dict[int, np.ndarray], annotations: Dict[int, list],
                library: Dict[str, List[np.ndarray]], rng: random.Random,
                pastes: Tuple[int, int] = (3, 11),
                background_frames: Optional[List[int]] = None,
                empty: bool = False, distractors: Tuple[int, int] = (0, 6)):
    if empty:
        # A Level-0 view is the entire frame and always contains objects, and a
        # Level-1 quarter nearly always does. Only the deeper zooms can show
        # genuinely empty ground, so that is where background images come from.
        level = rng.choices((1, 2), weights=(0.25, 0.75))[0]
    else:
        level = rng.choices(list(LEVEL_WEIGHTS),
                            weights=list(LEVEL_WEIGHTS.values()))[0]
    factor = LEVEL_FACTOR[level]
    region_w, region_h = VIEW_W * factor, VIEW_H * factor

    frame = rng.choice(background_frames if background_frames else list(frames))
    image = frames[frame]

    def region_is_clear(x: int, y: int) -> bool:
        for annotation in annotations[frame]:
            bx1, by1, bx2, by2 = annotation['bbox']
            if not (bx2 <= x or bx1 >= x + region_w or
                    by2 <= y or by1 >= y + region_h):
                return False
        return True

    ox = rng.randint(0, image.shape[1] - region_w)
    oy = rng.randint(0, image.shape[0] - region_h)
    if empty:
        # A random crop nearly always catches one of the annotated objects - a
        # Level-1 crop is a quarter of a frame that holds eight to twelve of
        # them - so empty ground has to be sought out rather than stumbled on.
        for _ in range(40):
            if region_is_clear(ox, oy):
                break
            ox = rng.randint(0, image.shape[1] - region_w)
            oy = rng.randint(0, image.shape[0] - region_h)

    canvas = image[oy:oy + region_h, ox:ox + region_w].copy()

    labels: List[Tuple[str, Tuple[int, int, int, int]]] = []

    # Real objects overlapping the chosen region. Partially visible ones are
    # kept and clipped: an object entering at the top of the source frame is
    # exactly that case, and the evaluator's ground truth is clipped too.
    for annotation in annotations[frame]:
        bx1, by1, bx2, by2 = annotation['bbox']
        cx1, cy1 = max(bx1, ox), max(by1, oy)
        cx2, cy2 = min(bx2, ox + region_w), min(by2, oy + region_h)
        if cx2 - cx1 < 2 or cy2 - cy1 < 2:
            continue
        visible = ((cx2 - cx1) * (cy2 - cy1)) / max(1, (bx2 - bx1) * (by2 - by1))
        if visible < MINIMUM_VISIBLE:
            continue
        labels.append((annotation['object_id'],
                       (cx1 - ox, cy1 - oy, cx2 - ox, cy2 - oy)))

    # A background image must genuinely contain nothing, so a crop that caught
    # a real object cannot be used as one.
    if empty and labels:
        empty = False

    # Colour and flip the terrain, but only when there is nothing real in it to
    # keep aligned - flipping would invalidate the real boxes.
    if not labels and rng.random() < 0.90:
        canvas = augment_background(canvas, rng)
    elif rng.random() < 0.85:
        canvas = photometric_only(canvas, rng)

    # Pasted objects.
    occupied = [box for _, box in labels]
    wanted = 0 if empty else rng.randint(*pastes)
    names = list(library)
    for _ in range(wanted * 3):
        if len(labels) >= wanted + len(occupied):
            break
        name = rng.choice(names)
        rgba = augment_sprite(rng.choice(library[name]), rng)
        h, w = rgba.shape[:2]
        if w >= region_w - 8 or h >= region_h - 8:
            continue
        # Every so often, let the sprite hang off the edge of the canvas, so
        # the model sees partial objects as often as it will at run time.
        if rng.random() < EDGE_PASTE_PROBABILITY:
            x = rng.randint(-w // 2, region_w - w // 2)
            y = rng.randint(-h // 2, region_h - h // 2)
        else:
            x = rng.randint(0, region_w - w - 1)
            y = rng.randint(0, region_h - h - 1)
        candidate = (x, y, x + w, y + h)
        # Usually keep objects apart, but sometimes let one partly cover
        # another: a detector never shown occlusion handles it badly.
        if rng.random() < OCCLUSION_PROBABILITY:
            if any(covered_fraction(box, candidate) > MAXIMUM_OCCLUDED
                   for box in occupied):
                continue
        elif any(boxes_overlap(candidate, box) for box in occupied):
            continue
        full = opaque_area(rgba)
        box = paste(canvas, rgba, x, y, rng)
        if box is None:
            continue
        # A sprite mostly off-canvas is not a usable training example.
        visible = (box[2] - box[0]) * (box[3] - box[1])
        if full > 0 and visible < MINIMUM_VISIBLE * full:
            continue
        occupied.append(box)
        labels.append((name, box))

    # Unlabelled clutter, pasted last so it never lands on a target.
    wanted_distractors = rng.randint(*distractors)
    if empty:
        wanted_distractors = max(wanted_distractors, rng.randint(2, 8))
    if wanted_distractors:
        paste_distractors(canvas, rng, occupied, factor, wanted_distractors)

    view = cv2.resize(canvas, (VIEW_W, VIEW_H), interpolation=cv2.INTER_AREA)

    # Sensor-side effects happen after the downsample, on the transmitted image.
    if rng.random() < 0.25:
        view = cv2.GaussianBlur(view, (3, 3), rng.uniform(0.3, 0.8))
    if rng.random() < 0.30:
        noise = np.random.normal(0, rng.uniform(1.5, 5.0), view.shape)
        view = np.clip(view.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    yolo_labels = []
    for name, (x1, y1, x2, y2) in labels:
        x1, y1, x2, y2 = x1 / factor, y1 / factor, x2 / factor, y2 / factor
        x1, x2 = max(0.0, x1), min(float(VIEW_W), x2)
        y1, y2 = max(0.0, y1), min(float(VIEW_H), y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        yolo_labels.append((CLASS_INDEX[name],
                            (x1 + x2) / 2 / VIEW_W, (y1 + y2) / 2 / VIEW_H,
                            (x2 - x1) / VIEW_W, (y2 - y1) / VIEW_H))
    return view, yolo_labels, level


# --------------------------------------------------------------------------- #

def write_split(split: str, count: int, frames, annotations, library, seed: int,
                pastes, background_frames=None, empty_share: float = 0.0,
                distractors: Tuple[int, int] = (0, 6)):
    rng = random.Random(seed)
    np.random.seed(seed)
    image_dir = DATASET / 'images' / split
    label_dir = DATASET / 'labels' / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    histogram = {name: 0 for name in OBJECT_CLASSES}
    empties = 0
    for i in range(count):
        empty = rng.random() < empty_share
        view, labels, _ = make_sample(frames, annotations, library, rng,
                                      pastes, background_frames, empty,
                                      distractors)
        if not labels:
            empties += 1
        cv2.imwrite(str(image_dir / f'{i:06d}.jpg'), view,
                    [cv2.IMWRITE_JPEG_QUALITY, 96])
        with open(label_dir / f'{i:06d}.txt', 'w') as handle:
            for index, cx, cy, w, h in labels:
                handle.write(f'{index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')
                histogram[OBJECT_CLASSES[index]] += 1
        if (i + 1) % 250 == 0:
            print(f'  {split}: {i + 1}/{count}')
    print(f'  {split}: {empties} images with no objects '
          f'({empties / max(1, count) * 100:.0f}%)')
    return histogram


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=6000)
    parser.add_argument('--val', type=int, default=600)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=str(LAB / 'dataset'))
    parser.add_argument(
        '--background-frames', default='',
        help='restrict background terrain to these frames, e.g. "0-12". '
             'Sprites still come from every frame. Used to build a held-out '
             'terrain test: the drone translates, so late frames show ground '
             'that early frames never covered.')
    parser.add_argument(
        '--empty-share', type=float, default=0.22,
        help='fraction of images containing no objects at all. Without these '
             'the detector is never shown that "nothing here" is a valid '
             'answer, and it invents objects on unfamiliar terrain.')
    parser.add_argument(
        '--distractors', default='0,6',
        help='min,max unlabelled man-made-looking shapes per image. These are '
             'the hard negatives the supplied scene does not contain.')
    arguments = parser.parse_args()

    global DATASET
    DATASET = Path(arguments.out)
    if DATASET.exists():
        shutil.rmtree(DATASET)

    library = load_sprites()
    print(f'sprite library: {len(library)} classes, '
          f'{sum(len(v) for v in library.values())} sprites')
    missing = [n for n in OBJECT_CLASSES if n not in library]
    if missing:
        raise SystemExit(f'no sprites for {missing}')

    numbers = frame_numbers(SCENE)
    print('loading frames ...')
    frames = {f: load_frame(f, SCENE) for f in numbers}
    annotations = {f: load_annotations(f, SCENE) for f in numbers}

    background_frames = None
    if arguments.background_frames:
        first, _, last = arguments.background_frames.partition('-')
        background_frames = [f for f in numbers
                             if int(first) <= f <= int(last or first)]
        print(f'backgrounds restricted to frames {background_frames[0]}..'
              f'{background_frames[-1]} ({len(background_frames)} frames)')

    low, _, high = arguments.distractors.partition(',')
    distractors = (int(low), int(high or low))
    print(f'empty images: {arguments.empty_share:.0%}, '
          f'distractors per image: {distractors[0]}-{distractors[1]}')

    print('writing train split ...')
    train_histogram = write_split('train', arguments.train, frames, annotations,
                                  library, arguments.seed, (3, 11),
                                  background_frames, arguments.empty_share,
                                  distractors)
    print('writing val split ...')
    val_histogram = write_split('val', arguments.val, frames, annotations,
                                library, arguments.seed + 9999, (3, 8),
                                background_frames, arguments.empty_share,
                                distractors)

    yaml_path = DATASET / 'data.yaml'
    with open(yaml_path, 'w') as handle:
        handle.write(f'path: {DATASET.as_posix()}\n')
        handle.write('train: images/train\nval: images/val\n')
        handle.write(f'nc: {len(OBJECT_CLASSES)}\n')
        handle.write('names:\n')
        for i, name in enumerate(OBJECT_CLASSES):
            handle.write(f'  {i}: {name}\n')

    with open(DATASET / 'stats.json', 'w') as handle:
        json.dump({'train': train_histogram, 'val': val_histogram}, handle, indent=2)

    print(f'\nwrote {yaml_path}')
    print(f'{"class":16s} {"train":>7s} {"val":>6s}')
    for name in OBJECT_CLASSES:
        print(f'{name:16s} {train_histogram[name]:7d} {val_histogram[name]:6d}')
    print(f'{"TOTAL":16s} {sum(train_histogram.values()):7d} '
          f'{sum(val_histogram.values()):6d}')


if __name__ == '__main__':
    main()
