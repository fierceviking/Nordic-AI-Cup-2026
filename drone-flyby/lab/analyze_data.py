"""Exploratory analysis of the supplied helsinki scene.

Answers the questions that decide the modelling approach:
  * how big are the objects, in source pixels and at each resolution level?
  * how do they move from frame to frame (is the flight a pure translation)?
  * how many instances are visible per frame, and where?
  * what does each class actually look like?
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import load_annotations, load_frame, frame_numbers, scene_directory  # noqa: E402

OUT = Path(__file__).resolve().parent / 'out'
OUT.mkdir(exist_ok=True)


def pose(frame, scene='helsinki'):
    path = scene_directory(scene) / 'annotations' / f'frame_{frame:06d}.json'
    with open(path) as handle:
        return json.load(handle)['pose']


def main():
    scene = 'helsinki'
    frames = frame_numbers(scene)
    print(f'frames: {frames[0]}..{frames[-1]} ({len(frames)})')

    per_frame = {f: load_annotations(f, scene) for f in frames}

    # ---------------------------------------------------------------- poses
    print('\n=== poses ===')
    prev = None
    for f in frames[:6]:
        p = pose(f, scene)
        d = None if prev is None else (p['x'] - prev['x'], p['y'] - prev['y'], p['z'] - prev['z'])
        print(f'  frame {f:3d}: {p}  delta={d}')
        prev = p

    # ------------------------------------------------------- object sizes
    print('\n=== object sizes (source pixels) ===')
    sizes = defaultdict(list)
    for f in frames:
        for a in per_frame[f]:
            x1, y1, x2, y2 = a['bbox']
            sizes[a['object_id']].append((x2 - x1, y2 - y1))
    rows = []
    for name in sorted(sizes):
        arr = np.array(sizes[name], dtype=float)
        w, h = arr[:, 0].mean(), arr[:, 1].mean()
        rows.append((name, len(arr), w, h, w / 4, h / 4, w / 2, h / 2))
    print(f'{"class":16s} {"n":>3s} {"w":>7s} {"h":>7s} {"L0 w":>6s} {"L0 h":>6s} {"L1 w":>6s} {"L1 h":>6s}')
    for r in sorted(rows, key=lambda r: r[2] * r[3]):
        print(f'{r[0]:16s} {r[1]:3d} {r[2]:7.1f} {r[3]:7.1f} {r[4]:6.1f} {r[5]:6.1f} {r[6]:6.1f} {r[7]:6.1f}')

    # -------------------------------------------------- per-frame counts
    print('\n=== visible instances per frame ===')
    for f in frames:
        names = sorted(a['object_id'] for a in per_frame[f])
        print(f'  {f:3d}: {len(names):2d}  {", ".join(names)}')

    # ------------------------------------------ frame-to-frame motion
    print('\n=== frame-to-frame displacement of matched objects ===')
    deltas = []
    for a_f, b_f in zip(frames, frames[1:]):
        by_name_a = {a['object_id']: a['bbox'] for a in per_frame[a_f]}
        by_name_b = {a['object_id']: a['bbox'] for a in per_frame[b_f]}
        for name in set(by_name_a) & set(by_name_b):
            ax1, ay1, ax2, ay2 = by_name_a[name]
            bx1, by1, bx2, by2 = by_name_b[name]
            acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
            bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
            deltas.append((bcx - acx, bcy - acy))
    d = np.array(deltas)
    print(f'  n={len(d)}  dx mean={d[:, 0].mean():.2f} std={d[:, 0].std():.2f} '
          f'min={d[:, 0].min():.1f} max={d[:, 0].max():.1f}')
    print(f'            dy mean={d[:, 1].mean():.2f} std={d[:, 1].std():.2f} '
          f'min={d[:, 1].min():.1f} max={d[:, 1].max():.1f}')

    # size change across frames for the same object
    print('\n=== per-object track (center + size over frames) ===')
    tracks = defaultdict(list)
    for f in frames:
        for a in per_frame[f]:
            x1, y1, x2, y2 = a['bbox']
            tracks[a['object_id']].append((f, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    for name in sorted(tracks):
        t = tracks[name]
        print(f'  {name:16s} frames {t[0][0]:2d}->{t[-1][0]:2d} ({len(t):2d})  '
              f'c {t[0][1]:7.1f},{t[0][2]:7.1f} -> {t[-1][1]:7.1f},{t[-1][2]:7.1f}  '
              f'size {t[0][3]:5.1f}x{t[0][4]:5.1f} -> {t[-1][3]:5.1f}x{t[-1][4]:5.1f}')

    # ------------------------------------------------------ crop montage
    print('\n=== writing crops ===')
    crops_dir = OUT / 'crops'
    crops_dir.mkdir(exist_ok=True)
    seen = set()
    for f in frames:
        img = None
        for a in per_frame[f]:
            name = a['object_id']
            if name in seen:
                continue
            if img is None:
                img = load_frame(f, scene)
            x1, y1, x2, y2 = a['bbox']
            pad = 12
            crop = img[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
            if crop.size == 0:
                continue
            seen.add(name)
            cv2.imwrite(str(crops_dir / f'{name}_src.png'), crop)
            # what it looks like at L0 (downsample 4x) and L1 (2x)
            for level, factor in ((0, 4), (1, 2)):
                small = cv2.resize(
                    img[max(0, y1 - pad * factor):y2 + pad * factor,
                        max(0, x1 - pad * factor):x2 + pad * factor],
                    None, fx=1 / factor, fy=1 / factor, interpolation=cv2.INTER_AREA)
                if small.size:
                    cv2.imwrite(str(crops_dir / f'{name}_L{level}.png'),
                                cv2.resize(small, None, fx=4, fy=4,
                                           interpolation=cv2.INTER_NEAREST))
    print(f'  wrote {len(seen)} classes to {crops_dir}')

    # ---------------------------------------------- background statistics
    print('\n=== frame statistics ===')
    img = load_frame(frames[0], scene)
    print(f'  shape={img.shape} dtype={img.dtype} mean={img.mean():.1f}')


if __name__ == '__main__':
    main()
