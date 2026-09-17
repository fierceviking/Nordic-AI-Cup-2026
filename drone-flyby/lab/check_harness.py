"""Sanity check: feed the ground truth through the tiling harness.

If this does not print recall 1.000 and mAP 1.000 then a low score from a real
detector is the harness, not the detector.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from detector_eval import FrameCache, report_detector, run_detector  # noqa: E402
from utils import frame_numbers, load_annotations  # noqa: E402

SCENE = 'helsinki'


def cheat_detector_factory(level: int):
    """Return GT boxes that fall inside the current tile, in source pixels."""
    by_frame = {f: load_annotations(f, SCENE) for f in frame_numbers(SCENE)}
    # run_detector gives the detector no frame number, so key on the tile
    # content instead: look up by matching region across all frames is wrong,
    # so cheat by tracking call order.
    state = {'index': 0, 'frames': list(frame_numbers(SCENE))}
    from simulate import tile_centers
    tiles = len(tile_centers(level))

    def detector(view, region):
        frame = state['frames'][state['index'] // tiles]
        state['index'] += 1
        x1, y1, x2, y2 = region
        out = []
        for annotation in by_frame[frame]:
            bx1, by1, bx2, by2 = annotation['bbox']
            cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
            if x1 <= cx < x2 and y1 <= cy < y2:
                out.append((annotation['object_id'],
                            [float(bx1), float(by1), float(bx2), float(by2)], 1.0))
        return out

    return detector


if __name__ == '__main__':
    cache = FrameCache()
    for level in (0, 1, 2):
        predictions = run_detector(cheat_detector_factory(level), level=level,
                                   cache=cache)
        report_detector(f'L{level} cheat detector (should be 1.000/1.000)',
                        predictions, show_classes=False)
