"""Build a detector dataset by compositing the Helsinki sprites onto foreign ground.

The detector currently scores 0.94 on Helsinki and 0.153 on the competition
scene. Retrieval showed why: descriptors built on Helsinki carry the ground
with them, so a real helicopter matched `medium_launcher`. Background is a
usable cue and the model learns it.

Here the same 236 masked sprites appear over LoveDA and RESISC45 ground as well
as Helsinki, so background stops predicting class. Every paste randomises
scale, rotation, blending and colour so the seam is not a cue either, and the
photometric range spans what lab/domain_gap.py measured on the real flight
(softer, brighter, less saturated than Helsinki).

    python lab/make_composites.py --images 8000 --name v7
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))

from dtos import OBJECT_CLASSES  # noqa: E402

SPRITES = LAB / 'out' / 'sprites'
BACKGROUNDS = LAB / 'backgrounds'
VIEW = (960, 540)
# Helsinki truth is heavily skewed small: short side p10 5 px, p50 12 px,
# p90 28 px as the camera delivers it at Level 0, roughly double at Level 1.
# Sampling uniformly over the range put the median at 51 px, four times life
# size, so sizes are drawn log-uniformly instead.
SIZE_RANGE = (5, 46)
# lab/domain_gap.py on the recorded flight: saturation 106, brightness 113,
# median local texture 1504. Backgrounds are matched into these ranges rather
# than nudged, so the training distribution covers the real one.
TARGET_SATURATION = (62, 190)
TARGET_VALUE = (78, 158)


def load_sprites():
    sprites = {}
    for folder in sorted(p for p in SPRITES.iterdir() if p.is_dir()):
        items = []
        for path in sorted(folder.glob('*.png')):
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is not None and image.ndim == 3 and image.shape[2] == 4:
                items.append(image)
        if items:
            sprites[folder.name] = items
    missing = set(OBJECT_CLASSES) - set(sprites)
    if missing:
        raise SystemExit(f'no sprites for {sorted(missing)}')
    return sprites


def background_pool():
    pool = []
    for folder in ('loveda', 'resisc45'):
        pool.extend(sorted((BACKGROUNDS / folder).glob('*.jpg')))
    return pool


def helsinki_backgrounds():
    from utils import frame_numbers, load_frame
    tiles = []
    for number in frame_numbers('helsinki')[:6]:
        frame = load_frame(number, 'helsinki')
        for _ in range(12):
            x = random.randint(0, frame.shape[1] - VIEW[0])
            y = random.randint(0, frame.shape[0] - VIEW[1])
            tiles.append(frame[y:y + VIEW[1], x:x + VIEW[0]].copy())
        del frame
    return tiles


def make_canvas(pool, generator):
    """A 960x540 patch of foreign ground assembled at close to native detail.

    RESISC45 tiles are 256x256; stretching one to fill the view would upscale
    3.75x and destroy the texture the detector needs to learn to ignore. A
    mosaic of several keeps detail and adds terrain variety per image.
    """
    tile = int(generator.choice([160, 192, 224, 256]))
    canvas = np.zeros((VIEW[1], VIEW[0], 3), np.uint8)
    for top in range(0, VIEW[1], tile):
        for left in range(0, VIEW[0], tile):
            image = None
            for _ in range(4):
                image = cv2.imread(str(pool[int(generator.integers(len(pool)))]))
                if image is not None:
                    break
            if image is None:
                continue
            short = min(image.shape[:2])
            if short > tile:
                y = int(generator.integers(0, image.shape[0] - tile + 1))
                x = int(generator.integers(0, image.shape[1] - tile + 1))
                patch = image[y:y + tile, x:x + tile]
            else:
                patch = cv2.resize(image, (tile, tile), interpolation=cv2.INTER_LINEAR)
            if generator.random() < 0.5:
                patch = cv2.flip(patch, int(generator.integers(-1, 2)))
            height = min(tile, VIEW[1] - top)
            width = min(tile, VIEW[0] - left)
            canvas[top:top + height, left:left + width] = patch[:height, :width]
    return canvas


def rotate_sprite(sprite, angle, scale):
    height, width = sprite.shape[:2]
    target = max(4, int(round(max(height, width) * scale)))
    sprite = cv2.resize(sprite, (target, target), interpolation=cv2.INTER_AREA)
    matrix = cv2.getRotationMatrix2D((target / 2, target / 2), angle, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    box = int(target * cosine + target * sine)
    matrix[0, 2] += box / 2 - target / 2
    matrix[1, 2] += box / 2 - target / 2
    return cv2.warpAffine(sprite, matrix, (box, box), flags=cv2.INTER_LINEAR,
                          borderValue=(0, 0, 0, 0))


def harmonise(patch, canvas_region, generator):
    """Pull the sprite's colour toward the ground it is landing on."""
    strength = generator.uniform(0.10, 0.45)
    for channel in range(3):
        source = patch[:, :, channel].astype(np.float32)
        target_mean = float(canvas_region[:, :, channel].mean())
        source_mean = float(source.mean()) or 1.0
        shifted = source + strength * (target_mean - source_mean)
        patch[:, :, channel] = np.clip(shifted, 0, 255).astype(np.uint8)
    return patch


def paste(canvas, sprite, generator):
    """Alpha-composite one sprite and return its YOLO box, or None."""
    angle = generator.uniform(0, 360)
    low, high = SIZE_RANGE
    side = float(low * (high / low) ** generator.random())
    native = max(1, max(sprite.shape[:2]))
    # Enlarging a cut-out invents detail the renderer never produced.
    scale = min(side / native, 1.25)
    rotated = rotate_sprite(sprite, angle, scale)
    box = rotated.shape[0]
    if box < 4 or box >= min(VIEW):
        return None
    x = int(generator.integers(0, VIEW[0] - box))
    y = int(generator.integers(0, VIEW[1] - box))

    region = canvas[y:y + box, x:x + box]
    colour = harmonise(rotated[:, :, :3].copy(), region, generator)
    alpha = rotated[:, :, 3].astype(np.float32) / 255.0
    # The box comes from the true sprite extent; feathering only softens the
    # seam, so it must stay sub-pixel relative to the sprite or it widens the
    # visible object past its own label.
    # Thin structures such as rotor blades carry a low GrabCut alpha but are
    # still visible, so the extent threshold sits well below half.
    ys, xs = np.nonzero(alpha > 0.2)
    if len(xs) < 4:
        return None
    if generator.random() < 0.75:
        sigma = min(generator.uniform(0.3, 1.1), max(0.25, box / 50))
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigma)
    if alpha.max() < 0.2:
        return None
    blended = region * (1 - alpha[..., None]) + colour * alpha[..., None]
    canvas[y:y + box, x:x + box] = np.clip(blended, 0, 255).astype(np.uint8)

    x1, x2 = x + int(xs.min()), x + int(xs.max())
    y1, y2 = y + int(ys.min()), y + int(ys.max())
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return ((x1 + x2) / 2 / VIEW[0], (y1 + y2) / 2 / VIEW[1],
            (x2 - x1) / VIEW[0], (y2 - y1) / VIEW[1])


def photometric(canvas, generator):
    """Rescale into the saturation and brightness range the real flight shows."""
    hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV).astype(np.float32)
    for channel, (low, high) in ((1, TARGET_SATURATION), (2, TARGET_VALUE)):
        current = float(hsv[:, :, channel].mean())
        if current > 1.0:
            hsv[:, :, channel] *= generator.uniform(low, high) / current
    canvas = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if generator.random() < 0.20:
        canvas = cv2.GaussianBlur(canvas, (0, 0), generator.uniform(0.3, 0.8))
    if generator.random() < 0.35:
        noise = generator.normal(0, generator.uniform(1.5, 6.0), canvas.shape)
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return canvas


def checks():
    """Pasted geometry must agree with the label written for it."""
    generator = np.random.default_rng(0)
    sprites = load_sprites()
    print(f'{sum(len(v) for v in sprites.values())} sprites across {len(sprites)} classes')
    checked = 0
    for name, items in sprites.items():
        for sprite in items[:2]:
            canvas = np.full((VIEW[1], VIEW[0], 3), 120, np.uint8)
            before = canvas.copy()
            box = paste(canvas, sprite, generator)
            if box is None:
                continue
            cx, cy, width, height = box
            assert 0 <= cx <= 1 and 0 <= cy <= 1, f'{name} centre outside view'
            assert 0 < width <= 1 and 0 < height <= 1, f'{name} degenerate size'
            x1 = int((cx - width / 2) * VIEW[0])
            y1 = int((cy - height / 2) * VIEW[1])
            x2 = int((cx + width / 2) * VIEW[0])
            y2 = int((cy + height / 2) * VIEW[1])
            delta = np.abs(canvas.astype(np.int16) - before.astype(np.int16))
            changed = np.any(delta > 8, axis=2)
            if not changed.any():
                continue
            # One pixel of slack for the feathered seam itself.
            inside = changed[max(0, y1 - 1):y2 + 2, max(0, x1 - 1):x2 + 2].sum()
            assert inside > 0, f'{name} label box contains no pasted pixels'
            share = inside / max(1, changed.sum())
            assert share > 0.97, f'{name} box misses {1 - share:.1%} of the paste'
            checked += 1
    canvas = np.full((VIEW[1], VIEW[0], 3), 120, np.uint8)
    out = photometric(canvas.copy(), generator)
    assert out.shape == canvas.shape and out.dtype == np.uint8
    print(f'PASS: {checked} pastes, labels enclose >97% of visibly altered '
          f'pixels, photometric pass preserves shape')


def build(arguments):
    generator = np.random.default_rng(arguments.seed)
    random.seed(arguments.seed)
    sprites = load_sprites()
    foreign = background_pool()
    if len(foreign) < 50:
        raise SystemExit(f'only {len(foreign)} foreign backgrounds; run '
                         f'lab/fetch_backgrounds.py first')
    local = helsinki_backgrounds()
    print(f'{sum(len(v) for v in sprites.values())} sprites, '
          f'{len(foreign)} foreign backgrounds, {len(local)} Helsinki tiles')

    root = LAB / f'dataset_{arguments.name}'
    counts, placed = Counter(), Counter()
    index = {name: i for i, name in enumerate(OBJECT_CLASSES)}
    for split, total in (('train', arguments.images),
                         ('val', max(1, arguments.images // 10))):
        (root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (root / 'labels' / split).mkdir(parents=True, exist_ok=True)
        for n in range(total):
            use_local = generator.random() < arguments.helsinki_share
            canvas = (local[int(generator.integers(len(local)))].copy() if use_local
                      else make_canvas(foreign, generator))
            if canvas is None:
                continue
            lines = []
            if generator.random() >= arguments.empty_share:
                for _ in range(int(generator.integers(1, arguments.max_objects + 1))):
                    name = OBJECT_CLASSES[int(generator.integers(len(OBJECT_CLASSES)))]
                    choices = sprites[name]
                    sprite = choices[int(generator.integers(len(choices)))]
                    box = paste(canvas, sprite, generator)
                    if box is None:
                        continue
                    lines.append(f'{index[name]} ' + ' '.join(f'{v:.6f}' for v in box))
                    placed[name] += 1
            canvas = photometric(canvas, generator)
            stem = f'{split}_{n:06d}'
            cv2.imwrite(str(root / 'images' / split / f'{stem}.jpg'), canvas,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            (root / 'labels' / split / f'{stem}.txt').write_text(
                '\n'.join(lines), encoding='utf-8')
            counts[split] += 1
            if counts[split] % 500 == 0:
                print(f'  {split}: {counts[split]}/{total}', flush=True)

    yaml = (f'path: {root.resolve().as_posix()}\ntrain: images/train\n'
            f'val: images/val\nnames:\n' +
            ''.join(f'  {i}: {n}\n' for i, n in enumerate(OBJECT_CLASSES)))
    (root / 'data.yaml').write_text(yaml, encoding='utf-8')
    manifest = {'name': arguments.name, 'seed': arguments.seed,
                'images': dict(counts), 'instances': dict(placed),
                'foreign_backgrounds': len(foreign),
                'helsinki_share': arguments.helsinki_share,
                'empty_share': arguments.empty_share,
                'sources': json.loads((BACKGROUNDS / 'SOURCES.json').read_text())
                if (BACKGROUNDS / 'SOURCES.json').is_file() else None}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'\nimages {dict(counts)}')
    print('instances per class:')
    for name in OBJECT_CLASSES:
        print(f'  {name:16s} {placed[name]:6d}')
    print(f'\nwrote {root / "data.yaml"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', default='v7')
    parser.add_argument('--images', type=int, default=8000)
    parser.add_argument('--max-objects', type=int, default=9)
    parser.add_argument('--helsinki-share', type=float, default=0.25,
                        help='fraction of images on Helsinki ground')
    parser.add_argument('--empty-share', type=float, default=0.12,
                        help='fraction with no objects at all')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--check', action='store_true',
                        help='verify paste geometry and exit')
    arguments = parser.parse_args()
    if not 0 <= arguments.helsinki_share <= 1 or not 0 <= arguments.empty_share <= 1:
        parser.error('shares must be between 0 and 1')
    if arguments.check:
        checks()
        return
    checks()
    build(arguments)


if __name__ == '__main__':
    main()
