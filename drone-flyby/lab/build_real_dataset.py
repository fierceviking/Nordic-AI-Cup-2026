"""Build a real-frame YOLO split from approved flyby proposals, with densification.

v9real trained on the detector's in-view *hits* for each approved object. But a
ground-fixed object is in view for more frames than the detector fired on - the
faint edge frames are exactly the hard positives the model needs. The flyby is
one constant homography, so an approved ground-fixed object's box can be warped
into every frame it occupies, detector hit or not.

Densification is gated on evidence, not on a hardcoded class list: an object is
only warped when its per-sighting reference boxes are *tight* (it really is
ground-fixed). Moving objects - planes, helicopters - have scattered references
and fall back to their detector-hit boxes untouched, so a mislabelled class can
never smear a moving object across the frame.

    python lab/build_real_dataset.py --out dataset_v10real

Offline only; no API, does not touch the deployment.
"""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))
sys.path.insert(0, str(LAB))

from dtos import OBJECT_CLASSES  # noqa: E402
from utils import source_bbox_to_view  # noqa: E402
from propose_candidates import load, to_reference  # noqa: E402

W, H = 3840, 2160
VW, VH = 960, 540
SEQ = '14356d0b32754c4f9484a4bbb2d5b25d'
# Classes that fly rather than sit on the ground. Their sightings do not warp to
# a single reference point through the ground homography, so they are never
# densified - always kept as raw detector hits. The variance guard is a second
# line of defence for everything else.
MOVING = {'jet_plane', 'small_plane', 'medium_plane', 'helicopter',
          'condor', 'spacecraft'}


def in_view(box, margin=2.0):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return margin <= cx <= VW - margin and margin <= cy <= VH - margin


def view_box_from_source(src_box, region):
    nb = source_bbox_to_view(src_box, region)
    return [nb[0] * VW, nb[1] * VH, nb[2] * VW, nb[3] * VH]


def reference_boxes(proposal, homography, first_frame):
    """Each in-view hit mapped back to first-frame source pixels."""
    refs = []
    for entry in proposal['track']:
        if not entry.get('inview', True):
            continue
        bg = [entry['bbox_source'][0] / W, entry['bbox_source'][1] / H,
              entry['bbox_source'][2] / W, entry['bbox_source'][3] / H]
        refs.append(to_reference(bg, entry['frame'], first_frame, homography))
    return np.stack(refs) if refs else None


def detector_boxes(proposal):
    """Fallback: the in-view detector hits as view-pixel boxes, per stem."""
    out = []
    for entry in proposal['track']:
        if not entry.get('inview', True):
            continue
        box = view_box_from_source(entry['bbox_source'], entry['region'])
        if box[2] > box[0] and box[3] > box[1]:
            out.append((entry['stem'], box))
    return out


def densified_boxes(ref_median, byframe, homography, first_frame):
    """Warp one fixed reference box into every frame it is in view."""
    x1, y1, x2, y2 = ref_median
    corners = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], np.float64)
    out = []
    for frame, record in byframe.items():
        matrix = np.linalg.matrix_power(homography, frame - first_frame)
        warped = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
        src = [warped[:, 0].min(), warped[:, 1].min(),
               warped[:, 0].max(), warped[:, 1].max()]
        box = view_box_from_source(src, list(record['source_region_xyxy']))
        if in_view(box) and box[2] > box[0] and box[3] > box[1]:
            out.append((record['stem'], box))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sequence', default=SEQ)
    parser.add_argument('--out', default='dataset_v10real')
    parser.add_argument('--train-frac', type=float, default=0.70)
    parser.add_argument('--tight', type=float, default=0.6,
                        help='densify only if ref-centre spread < tight * object size')
    args = parser.parse_args()

    labels_dir = LAB / 'labels'
    proposals = {p['obj_id']: p for p in
                 json.loads((labels_dir / f'{args.sequence}.proposals.json').read_text())['proposals']}
    review = json.loads((labels_dir / f'{args.sequence}.review.json').read_text())['decisions']
    homography = np.load(LAB / 'out' / f'h_{args.sequence[:8]}.npy')
    records = load(LAB / 'recordings' / args.sequence)
    for rec in records:
        rec.setdefault('stem', Path(str(rec['_view'])).stem)
    first_frame = records[0]['frame']
    byframe = {r['frame']: r for r in records}
    stem_to_frame = {r['stem']: r['frame'] for r in records}

    frames = {}                       # stem -> list[(class, box_viewpx)]
    densified, fallback = [], []
    for key, decision in review.items():
        if decision.get('decision') != 'approve':
            continue
        proposal = proposals[int(key)]
        cls = decision.get('object_id') or proposal['guess']
        if cls not in OBJECT_CLASSES:
            continue
        refs = reference_boxes(proposal, homography, first_frame)
        boxes = None
        if cls not in MOVING and refs is not None and len(refs) >= 3:
            centres = np.column_stack(((refs[:, 0] + refs[:, 2]) / 2,
                                       (refs[:, 1] + refs[:, 3]) / 2))
            spread = float(np.median(np.linalg.norm(centres - np.median(centres, axis=0), axis=1)))
            size = float(np.median((refs[:, 2] - refs[:, 0] + refs[:, 3] - refs[:, 1]) / 2))
            if spread < args.tight * size:
                ref_median = np.median(refs, axis=0)
                boxes = densified_boxes(ref_median, byframe, homography, first_frame)
                densified.append((key, cls, len(boxes)))
        if boxes is None:
            boxes = detector_boxes(proposal)
            fallback.append((key, cls, len(boxes)))
        for stem, box in boxes:
            frames.setdefault(stem, []).append((cls, box))

    print(f'densified {len(densified)} ground-fixed objects, '
          f'{len(fallback)} kept as detector-hit (moving/uncertain)')
    total_boxes = sum(len(v) for v in frames.values())
    print(f'{len(frames)} labeled frames, {total_boxes} boxes')
    print('boxes/class:', dict(Counter(c for v in frames.values() for c, _ in v)))

    # Temporal split: earliest frames train, latest val (mirrors v9real).
    stems = sorted(frames, key=lambda s: stem_to_frame[s])
    cut = int(len(stems) * args.train_frac)
    split = {s: ('rtrain' if i < cut else 'rval') for i, s in enumerate(stems)}

    out_root = LAB / args.out
    for sub in ('images/rtrain', 'images/rval', 'labels/rtrain', 'labels/rval'):
        d = out_root / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    views = LAB / 'recordings' / args.sequence / 'views'
    counts = Counter()
    for stem, boxlist in frames.items():
        part = split[stem]
        shutil.copyfile(views / f'{stem}.png', out_root / 'images' / part / f'{stem}.png')
        lines = []
        for cls, box in boxlist:
            cx = (box[0] + box[2]) / 2 / VW
            cy = (box[1] + box[3]) / 2 / VH
            bw = (box[2] - box[0]) / VW
            bh = (box[3] - box[1]) / VH
            if bw <= 0 or bh <= 0:
                continue
            cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
            bw, bh = min(bw, 1), min(bh, 1)
            lines.append(f'{OBJECT_CLASSES.index(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
        (out_root / 'labels' / part / f'{stem}.txt').write_text('\n'.join(lines) + '\n')
        counts[part] += 1

    yaml = (f'path: {out_root.as_posix()}\n'
            'train:\n'
            f'  - {(LAB / "dataset_v4" / "images" / "train").as_posix()}\n'
            f'  - {(out_root / "images" / "rtrain").as_posix()}\n'
            'val: images/rval\n'
            'nc: 16\n'
            'names:\n'
            + ''.join(f'  {i}: {n}\n' for i, n in enumerate(OBJECT_CLASSES)))
    (out_root / 'data.yaml').write_text(yaml)
    print(f'wrote {out_root}  (rtrain {counts["rtrain"]}, rval {counts["rval"]})')
    print(f'data.yaml -> {out_root / "data.yaml"}')


if __name__ == '__main__':
    main()
