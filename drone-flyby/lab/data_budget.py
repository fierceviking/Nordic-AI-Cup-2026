"""How much genuinely distinct imagery is behind a synthetic dataset?

"6000 training images" invites the reading that there are 6000 independent
samples. There are not. They are random crops of 25 frames that themselves
overlap heavily, with sprites pasted on top. This measures the real budget, so
the number in the dataset summary can be read honestly.
"""

import json
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))

from utils import frame_numbers, load_annotations, scene_directory  # noqa: E402

SCENE = 'helsinki'
W, H = 3840, 2160
VIEW_W, VIEW_H = 960, 540
LEVEL_FACTOR = {0: 4, 1: 2, 2: 1}


def pose(frame):
    path = scene_directory(SCENE) / 'annotations' / f'frame_{frame:06d}.json'
    with open(path) as handle:
        return json.load(handle)['pose']


def main():
    frames = frame_numbers(SCENE)
    print(f'supplied frames: {len(frames)} at {W}x{H} '
          f'= {len(frames) * W * H / 1e6:.0f} Mpix of image data')

    # --------------------------------------------- how much unique ground
    # Objects move ~64 px/frame down the image, so consecutive frames overlap
    # by all but that strip. Measured from the annotations rather than assumed.
    shifts = []
    for a, b in zip(frames, frames[1:]):
        aa = {x['object_id']: x['bbox'] for x in load_annotations(a, SCENE)}
        bb = {x['object_id']: x['bbox'] for x in load_annotations(b, SCENE)}
        for key in set(aa) & set(bb):
            shifts.append(((bb[key][1] + bb[key][3]) / 2) -
                          ((aa[key][1] + aa[key][3]) / 2))
    step = float(np.median(shifts))
    swept = H + step * (len(frames) - 1)
    print(f'\nmedian ground shift per frame: {step:.1f} px')
    print(f'ground swept by the whole flight: {W} x {swept:.0f} px '
          f'= {W * swept / 1e6:.1f} Mpix')
    print(f'  that is {swept / H:.2f} frames worth of UNIQUE terrain, '
          f'not {len(frames)}')
    print(f'  average redundancy: each patch of ground appears in '
          f'{len(frames) * H / swept:.1f} frames')

    # ------------------------------------------- how many distinct crops
    print('\ndistinct crop positions per frame, by resolution level:')
    for level, factor in sorted(LEVEL_FACTOR.items()):
        rw, rh = VIEW_W * factor, VIEW_H * factor
        positions = max(0, W - rw + 1) * max(0, H - rh + 1)
        print(f'  L{level}: region {rw}x{rh} -> {positions:,} offsets'
              + ('  (the whole frame: only ONE possible crop)' if positions == 1 else ''))

    # ----------------------------------------------- what actually varies
    sprites = sum(1 for _ in (LAB / 'out' / 'sprites').rglob('*.png'))
    print(f'\nwhat genuinely varies between generated images:')
    print(f'  sprite instances       {sprites} (16 classes)')
    print(f'  sprite rotation        continuous, 0-360 deg')
    print(f'  sprite scale           0.80-1.35')
    print(f'  paste position         continuous')
    print(f'  colour jitter          continuous')
    print(f'  crop offset            continuous, but of the same {W * swept / 1e6:.0f} Mpix')
    print(f'\nwhat does NOT vary:')
    print(f'  the terrain itself. Every generated image is a recolouring of')
    print(f'  one {W * swept / 1e6:.0f} Mpix strip of Finnish coastline.')


if __name__ == '__main__':
    main()
