"""Propose object candidates in a recorded sequence, for human verification.

The competition does not publish ground truth, but we are allowed to keep the
frames. Annotating them ourselves turns the recording into a genuine held-out
test set - the only one available.

Doing that by eye over 248 frames is not necessary. Two things make it cheap:

* the flight is one constant homography (measured for this sequence in
  ``recording_motion.py``), so a ground-fixed object traces a predictable path
  and an object annotated once can be propagated to every frame it appears in;
* the detector already proposes candidates. Most are wrong - that is the whole
  problem - but the real objects are among them, and a human glance at a
  zoomed crop separates a CGI model from a moored yacht instantly.

So: cluster detections into ground-fixed tracks through the homography, rank
the tracks by how strongly and consistently they were detected, and render a
contact sheet of each track for verification. A real object appears in many
consecutive frames at the same ground position; a one-off false positive does
not.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))

RECORDINGS = LAB / 'recordings'
W, H = 3840, 2160


def load(sequence: Path):
    records = []
    for meta_path in sorted((sequence / 'meta').glob('*.json')):
        with open(meta_path) as handle:
            meta = json.load(handle)
        meta['_view'] = sequence / 'views' / (meta_path.stem + '.png')
        records.append(meta)
    return sorted(records, key=lambda m: m['frame'])


def to_reference(bbox_global, frame, first_frame, homography):
    """Map a frame-global box back to the coordinates of the first frame.

    Ground-fixed objects then land on the same spot regardless of when they
    were seen, which is what makes clustering possible.
    """
    steps = frame - first_frame
    matrix = np.linalg.matrix_power(np.linalg.inv(homography), steps)
    x1, y1, x2, y2 = (bbox_global[0] * W, bbox_global[1] * H,
                      bbox_global[2] * W, bbox_global[3] * H)
    corners = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], np.float64)
    warped = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
    return np.array([warped[:, 0].min(), warped[:, 1].min(),
                     warped[:, 0].max(), warped[:, 1].max()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--min-conf', type=float, default=0.30)
    parser.add_argument('--min-hits', type=int, default=6,
                        help='frames a track must appear in to be a candidate')
    parser.add_argument('--top', type=int, default=24)
    arguments = parser.parse_args()

    sequence = RECORDINGS / arguments.sequence
    records = load(sequence)
    homography = np.load(LAB / 'out' / f'h_{arguments.sequence[:8]}.npy')
    first_frame = records[0]['frame']
    print(f'{len(records)} frames, using detections >= {arguments.min_conf}')

    # ------------------------------------------------- cluster into tracks
    clusters = []            # each: dict(ref_box, hits=[(frame, conf, cls, bbox)])
    for meta in records:
        for prediction in meta['predictions']:
            if prediction['confidence'] < arguments.min_conf:
                continue
            reference = to_reference(prediction['bbox'], meta['frame'],
                                     first_frame, homography)
            centre = (reference[:2] + reference[2:]) / 2
            size = max(8.0, np.mean(reference[2:] - reference[:2]))
            best, best_distance = None, 1e9
            for cluster in clusters:
                c = (cluster['ref'][:2] + cluster['ref'][2:]) / 2
                distance = float(np.hypot(*(centre - c)))
                if distance < best_distance:
                    best, best_distance = cluster, distance
            if best is not None and best_distance < size * 1.2:
                best['hits'].append((meta['frame'], prediction['confidence'],
                                     prediction['object_id'], prediction['bbox']))
                best['ref'] = 0.8 * best['ref'] + 0.2 * reference
            else:
                clusters.append({'ref': reference,
                                 'hits': [(meta['frame'], prediction['confidence'],
                                           prediction['object_id'],
                                           prediction['bbox'])]})

    tracks = [c for c in clusters if len(c['hits']) >= arguments.min_hits]
    tracks.sort(key=lambda c: -(len(c['hits']) * np.mean([h[1] for h in c['hits']])))
    print(f'{len(clusters)} clusters, {len(tracks)} with >= {arguments.min_hits} hits\n')

    by_frame = {m['frame']: m for m in records}
    out_dir = LAB / 'out' / f'candidates_{arguments.sequence[:8]}'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'{"#":>3s} {"hits":>5s} {"meanconf":>9s} {"maxconf":>8s}  '
          f'{"frames":>12s}  votes')
    for index, track in enumerate(tracks[:arguments.top]):
        hits = track['hits']
        votes = defaultdict(float)
        for _, confidence, name, _ in hits:
            votes[name] += confidence
        ordered = sorted(votes.items(), key=lambda kv: -kv[1])
        frames = [h[0] for h in hits]
        print(f'{index:3d} {len(hits):5d} {np.mean([h[1] for h in hits]):9.3f} '
              f'{max(h[1] for h in hits):8.3f}  {min(frames):4d}-{max(frames):4d}  '
              + ', '.join(f'{n}:{v:.1f}' for n, v in ordered[:3]))

        # contact sheet: the same object across the frames it was seen in
        chosen = hits[::max(1, len(hits) // 8)][:8]
        tiles = []
        for frame, confidence, name, bbox in chosen:
            meta = by_frame[frame]
            image = cv2.imread(str(meta['_view']))
            if image is None:
                continue
            vh, vw = image.shape[:2]
            sx1, sy1, sx2, sy2 = meta['source_region_xyxy']
            cx = ((bbox[0] + bbox[2]) / 2 * W - sx1) / (sx2 - sx1) * vw
            cy = ((bbox[1] + bbox[3]) / 2 * H - sy1) / (sy2 - sy1) * vh
            half = 48
            x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
            crop = image[y1:y1 + 2 * half, x1:x1 + 2 * half]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (192, 192), interpolation=cv2.INTER_NEAREST)
            cv2.putText(crop, f'f{frame} {confidence:.2f}', (3, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(crop)
        if tiles:
            cv2.imwrite(str(out_dir / f'track{index:02d}.jpg'), cv2.hconcat(tiles),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f'\nwrote contact sheets to {out_dir}')
    print('A real object holds the same ground position and the same silhouette '
          'across frames.\nA moored boat does too - so judge by appearance, not '
          'persistence alone.')


if __name__ == '__main__':
    main()
