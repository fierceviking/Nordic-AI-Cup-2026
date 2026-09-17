"""Cut the 16 object models out of the supplied frames as RGBA sprites.

Only bounding boxes are supplied, not masks, so the alpha has to be recovered.
GrabCut initialised from the box does most of the work: the box is a tight fit
around a compositied CGI model, so the border pixels are reliably background and
the centre is reliably foreground.

Every annotated box of every instance is harvested, not just one per class. The
same model seen at 25 positions in the frame gives 25 slightly different
renderings - different ground under it, different sub-pixel sampling - and all
of them are useful as paste material.

Writes ``lab/out/sprites/<class>/<frame>.png`` as 4-channel BGRA, and a
contact sheet at ``lab/out/sprites_preview.png``.
"""

import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import frame_numbers, load_annotations, load_frame  # noqa: E402

SCENE = 'helsinki'
OUT = Path(__file__).resolve().parent / 'out'
SPRITES = OUT / 'sprites'

# GrabCut is initialised on a box grown by this fraction so it has genuine
# background to learn from on all four sides.
CONTEXT = 0.45
# Pixels within this fraction of the annotated box centre are forced foreground.
CORE = 0.30


def extract_sprite(image: np.ndarray, bbox, iterations: int = 5):
    """Return a BGRA sprite for one annotated box, or None if it is unusable."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (int(round(c)) for c in bbox)
    box_w, box_h = x2 - x1, y2 - y1
    if box_w < 8 or box_h < 8:
        return None
    # Skip boxes clipped by the frame edge: they are partial objects.
    if x1 <= 1 or y1 <= 1 or x2 >= width - 1 or y2 >= height - 1:
        return None

    pad_x = int(round(box_w * CONTEXT))
    pad_y = int(round(box_h * CONTEXT))
    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    cx2, cy2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
    patch = image[cy1:cy2, cx1:cx2].copy()
    if patch.size == 0:
        return None

    # Rectangle of the true object inside the padded patch.
    rx1, ry1 = x1 - cx1, y1 - cy1
    rx2, ry2 = x2 - cx1, y2 - cy1

    mask = np.full(patch.shape[:2], cv2.GC_BGD, np.uint8)
    mask[ry1:ry2, rx1:rx2] = cv2.GC_PR_FGD
    core_x = int(box_w * CORE / 2)
    core_y = int(box_h * CORE / 2)
    mcx, mcy = (rx1 + rx2) // 2, (ry1 + ry2) // 2
    mask[max(0, mcy - core_y):mcy + core_y + 1,
         max(0, mcx - core_x):mcx + core_x + 1] = cv2.GC_FGD

    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(patch, mask, None, background_model, foreground_model,
                    iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None

    alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    # Nothing outside the annotated box can belong to the object.
    keep = np.zeros_like(alpha)
    keep[ry1:ry2, rx1:rx2] = 255
    alpha = cv2.bitwise_and(alpha, keep)

    # Keep only the connected component that covers the box centre.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(alpha, 8)
    if n <= 1:
        return None
    label = labels[mcy, mcx]
    if label == 0:
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    alpha = np.where(labels == label, 255, 0).astype(np.uint8)

    area = int(alpha.sum() // 255)
    if area < 0.10 * box_w * box_h or area > 0.98 * box_w * box_h:
        return None

    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # A one-pixel feather stops the paste from having a hard aliased border.
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

    ys, xs = np.where(alpha > 8)
    if len(xs) == 0:
        return None
    tx1, tx2 = int(xs.min()), int(xs.max()) + 1
    ty1, ty2 = int(ys.min()), int(ys.max()) + 1
    sprite = np.dstack([patch[ty1:ty2, tx1:tx2], alpha[ty1:ty2, tx1:tx2]])
    return sprite


def main():
    if SPRITES.exists():
        shutil.rmtree(SPRITES)
    SPRITES.mkdir(parents=True)

    counts = defaultdict(int)
    attempts = defaultdict(int)
    for frame in frame_numbers(SCENE):
        image = load_frame(frame, SCENE)
        for annotation in load_annotations(frame, SCENE):
            name = annotation['object_id']
            attempts[name] += 1
            sprite = extract_sprite(image, annotation['bbox'])
            if sprite is None:
                continue
            directory = SPRITES / name
            directory.mkdir(exist_ok=True)
            cv2.imwrite(str(directory / f'{frame:06d}.png'), sprite)
            counts[name] += 1

    print(f'{"class":16s} {"kept":>5s} {"tried":>6s}')
    for name in sorted(attempts):
        print(f'{name:16s} {counts[name]:5d} {attempts[name]:6d}')
    print(f'\ntotal sprites: {sum(counts.values())}')
    missing = [n for n in attempts if counts[n] == 0]
    if missing:
        print(f'NO SPRITES FOR: {missing}')

    # ------------------------------------------------------- contact sheet
    cell = 160
    names = sorted(counts)
    cols = 4
    rows = (len(names) + cols - 1) // cols
    sheet = np.full((rows * (cell + 20), cols * cell, 3), 40, np.uint8)
    for i, name in enumerate(names):
        path = sorted((SPRITES / name).glob('*.png'))[0]
        sprite = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        bgr, alpha = sprite[..., :3], sprite[..., 3:4].astype(np.float32) / 255.0
        # checkerboard so the matte is visible
        h, w = bgr.shape[:2]
        board = np.indices((h, w)).sum(0) // 8 % 2
        back = np.where(board[..., None] == 0, 90, 150).astype(np.float32)
        composited = (bgr * alpha + back * (1 - alpha)).astype(np.uint8)
        scale = min(cell / w, cell / h)
        composited = cv2.resize(composited, (max(1, int(w * scale)), max(1, int(h * scale))),
                                interpolation=cv2.INTER_NEAREST)
        r, c = divmod(i, cols)
        y0, x0 = r * (cell + 20) + 20, c * cell
        sheet[y0:y0 + composited.shape[0], x0:x0 + composited.shape[1]] = composited
        cv2.putText(sheet, f'{name} ({counts[name]})', (x0 + 2, y0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / 'sprites_preview.png'), sheet)
    print(f'wrote {OUT / "sprites_preview.png"}')


if __name__ == '__main__':
    main()
