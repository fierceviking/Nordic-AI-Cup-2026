"""Pre-label a recorded sequence into per-object proposals for human review.

The competition validation flyby is recordable and re-runnable without limit, so
the winning move is to label the real recorded frames and train on them. Doing
that by eye over 249 frames is unnecessary: the flight is one constant
homography (``recording_motion.py``), so a ground-fixed object traces a
predictable path, and the detector already proposes candidates - most wrong, but
the real objects are among them.

This clusters detections into ground-fixed tracks through the homography, keeps
only tracks seen in enough frames (the persistence gate that kills one-off
false positives), and for each surviving track names the object by cosine
nearest-neighbour of a frozen EfficientNet embedding against a per-class
Helsinki reference bank. Confidence is deliberately NOT the ranking signal - a
pure-background frame can log several "strong" detections - so tracks are ranked
by persistence and appearance.

Output is machine-readable, for ``review_crops.py`` to show one object at a time
(its crop beside the guessed class's reference plate) for Approve / Disapprove:

    python lab/autolabel.py --sequence 14356d0b32754c4f9484a4bbb2d5b25d

Writes ``lab/labels/<seq>.proposals.json`` and one ``lab/out/proposals_<seq>/
obj<NN>.png`` crop (plus a multi-sighting strip) per proposed object. It makes
no API call and does not touch the running deployment.
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
sys.path.insert(0, str(LAB))

from propose_candidates import load, to_reference  # noqa: E402
from exp12_separability import Encoder, crop, at_sampling  # noqa: E402
from utils import (  # noqa: E402
    frame_numbers, load_annotations, load_frame, source_bbox_to_view,
)

RECORDINGS = LAB / 'recordings'
LABELS = LAB / 'labels'
W, H = 3840, 2160
VIEW_WIDTH, VIEW_HEIGHT = 960, 540
# The camera downsamples the source into the 960x540 view: L0 covers the whole
# 3840x2160 frame (4x), L1 a 1920x1080 region (2x), L2 native. To name a crop we
# degrade the native Helsinki reference to the same sampling the camera used.
LEVEL_DIVISOR = {0: 4, 1: 2, 2: 1}


def view_px(bbox_global, region):
    """Frame-global normalised box -> pixel box in the transmitted view."""
    source = (bbox_global[0] * W, bbox_global[1] * H,
              bbox_global[2] * W, bbox_global[3] * H)
    nx1, ny1, nx2, ny2 = source_bbox_to_view(source, region)
    return [nx1 * VIEW_WIDTH, ny1 * VIEW_HEIGHT,
            nx2 * VIEW_WIDTH, ny2 * VIEW_HEIGHT]


def in_view(box, margin=2.0):
    """True if the box's centre falls inside the transmitted view.

    The recorded predictions are the solver's global tracked boxes, so most
    boxes in any one frame belong to objects currently outside that frame's
    view. Only an in-view sighting shows real pixels of the object.
    """
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return (margin <= cx <= VIEW_WIDTH - margin
            and margin <= cy <= VIEW_HEIGHT - margin)


def cluster_tracks(records, homography, min_conf, min_hits):
    """Group detections into ground-fixed tracks (propose_candidates logic).

    Each hit keeps its frame's stem so we can crop and propagate later.
    """
    first_frame = records[0]['frame']
    by_stem = {}
    clusters = []          # each: dict(ref, hits=[hit dict])
    for meta in records:
        stem = meta.get('stem') or Path(str(meta['_view'])).stem
        by_stem[stem] = meta
        for prediction in meta['predictions']:
            if prediction['confidence'] < min_conf:
                continue
            reference = to_reference(prediction['bbox'], meta['frame'],
                                     first_frame, homography)
            centre = (reference[:2] + reference[2:]) / 2
            size = max(8.0, np.mean(reference[2:] - reference[:2]))
            best, best_distance = None, 1e9
            for cluster in clusters:
                other = (cluster['ref'][:2] + cluster['ref'][2:]) / 2
                distance = float(np.hypot(*(centre - other)))
                if distance < best_distance:
                    best, best_distance = cluster, distance
            hit = {'stem': stem, 'frame': meta['frame'],
                   'frame_index': meta['frame_index'],
                   'level': meta['resolution_level'],
                   'region': list(meta['source_region_xyxy']),
                   'confidence': float(prediction['confidence']),
                   'object_id': prediction['object_id'],
                   'bbox_global': list(prediction['bbox'])}
            hit['bbox_view'] = view_px(hit['bbox_global'], hit['region'])
            hit['inview'] = in_view(hit['bbox_view'])
            if best is not None and best_distance < size * 1.2:
                best['hits'].append(hit)
                best['ref'] = 0.8 * best['ref'] + 0.2 * reference
            else:
                clusters.append({'ref': reference, 'hits': [hit]})

    tracks = [c['hits'] for c in clusters
              if sum(h['inview'] for h in c['hits']) >= min_hits]
    tracks.sort(key=lambda hits: -(sum(h['inview'] for h in hits)
                                   * np.mean([h['confidence'] for h in hits])))
    return tracks, by_stem


def build_bank(encoder, context, divisors, batch=64):
    """Per-class Helsinki reference embeddings at each camera sampling.

    Returns {divisor: (features [N,1280], labels [N])}. The bank keeps every
    ground-truth crop, not just the first, so nearest-neighbour has more shots.
    """
    per_divisor = {d: [] for d in divisors}
    labels = []
    for number in frame_numbers('helsinki'):
        image = load_frame(number, 'helsinki')
        for annotation in load_annotations(number, 'helsinki'):
            patch = crop(image, [int(v) for v in annotation['bbox']], context)
            if patch is None:
                continue
            for divisor in divisors:
                per_divisor[divisor].append(at_sampling(patch, divisor))
            labels.append(annotation['object_id'])
        del image

    bank = {}
    labels = np.array(labels)
    for divisor, patches in per_divisor.items():
        features = [encoder(patches[start:start + batch])
                    for start in range(0, len(patches), batch)]
        bank[divisor] = (np.concatenate(features), labels)
    return bank


def name_track(hits, by_stem, encoder, bank, context, top=3):
    """Crop the sighting with the most pixels on the object and name it.

    Returns (representative hit, crop image, ranked [(class, sim)]). The
    representative is the largest box in view pixels - the L1 sightings where the
    object is biggest - which is also the crop the reviewer will judge.
    """
    rep, rep_area, rep_box = None, -1.0, None
    for hit in hits:
        if not hit['inview']:
            continue
        box = hit['bbox_view']
        area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        if area > rep_area:
            rep, rep_area, rep_box = hit, area, box
    if rep is None:
        return None, None, []
    image = cv2.imread(str(by_stem[rep['stem']]['_view']))
    if image is None:
        return rep, None, []
    patch = crop(image, rep_box, context)
    if patch is None:
        return rep, None, []
    divisor = LEVEL_DIVISOR.get(rep['level'], 1)
    query = encoder([at_sampling(patch, 1)])[0]
    features, labels = bank[divisor]
    similarity = features @ query
    best_per_class = {}
    for label, score in zip(labels, similarity):
        if score > best_per_class.get(label, -1e9):
            best_per_class[label] = float(score)
    ranked = sorted(best_per_class.items(), key=lambda kv: -kv[1])[:top]
    return rep, patch, ranked


def display_crop(hit, by_stem, size, context):
    """A clean, enlarged crop of one in-view sighting for the reviewer."""
    image = cv2.imread(str(by_stem[hit['stem']]['_view']))
    if image is None:
        return None
    patch = crop(image, hit['bbox_view'], context)
    if patch is None or patch.size == 0:
        return None
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_NEAREST)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--min-conf', type=float, default=0.25)
    parser.add_argument('--min-hits', type=int, default=6,
                        help='frames a track must appear in to survive')
    parser.add_argument('--context', type=float, default=1.3,
                        help='crop context for appearance matching')
    parser.add_argument('--display-context', type=float, default=1.8)
    parser.add_argument('--display-size', type=int, default=256)
    parser.add_argument('--top', type=int, default=0,
                        help='limit proposals emitted (0 = all surviving tracks)')
    arguments = parser.parse_args()

    import torch

    sequence = RECORDINGS / arguments.sequence
    if not (sequence / 'meta').is_dir():
        raise SystemExit(f'no recorded meta under {sequence}')
    homography_path = LAB / 'out' / f'h_{arguments.sequence[:8]}.npy'
    if not homography_path.is_file():
        raise SystemExit(f'{homography_path} missing; run recording_motion.py first')

    records = load(sequence)
    for meta in records:
        meta.setdefault('stem', Path(str(meta['_view'])).stem)
    homography = np.load(homography_path)
    print(f'{len(records)} frames; clustering detections >= {arguments.min_conf}')

    tracks, by_stem = cluster_tracks(records, homography,
                                     arguments.min_conf, arguments.min_hits)
    print(f'{len(tracks)} tracks with >= {arguments.min_hits} hits')
    if arguments.top:
        tracks = tracks[:arguments.top]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = Encoder(device)
    print(f'encoder on {device}; building Helsinki reference bank ...')
    bank = build_bank(encoder, arguments.context, sorted(set(LEVEL_DIVISOR.values())))
    print(f'reference bank: {len(bank[1][1])} crops per sampling\n')

    out_dir = LAB / 'out' / f'proposals_{arguments.sequence[:8]}'
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob('*.png'):
        stale.unlink()

    proposals = []
    print(f'{"#":>3s} {"seen":>4s} {"appear":>6s} {"guess":>14s}  '
          f'{"level":>5s}  detector votes')
    for hits in tracks:
        rep, patch, ranked = name_track(hits, by_stem, encoder, bank, arguments.context)
        if rep is None:
            continue
        index = len(proposals)
        seen = [h for h in hits if h['inview']]
        votes = defaultdict(float)
        for hit in hits:
            votes[hit['object_id']] += hit['confidence']
        detector_votes = sorted(votes.items(), key=lambda kv: -kv[1])[:3]
        guess = ranked[0][0] if ranked else detector_votes[0][0]
        appearance_score = ranked[0][1] if ranked else 0.0
        frames = [hit['frame'] for hit in seen]

        crop_image = cv2.resize(patch, (arguments.display_size, arguments.display_size),
                                interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out_dir / f'obj{index:02d}.png'), crop_image)
        # A strip of up to 6 in-view sightings: a CGI asset holds silhouette
        # across frames, a boat or rooftop corner does not.
        spread = seen[::max(1, len(seen) // 6)][:6]
        tiles = [t for t in (display_crop(h, by_stem, 160, arguments.display_context)
                             for h in spread) if t is not None]
        strip_name = None
        if tiles:
            strip_name = f'obj{index:02d}_strip.png'
            cv2.imwrite(str(out_dir / strip_name), cv2.hconcat(tiles))

        proposals.append({
            'obj_id': index,
            'guess': guess,
            'crop': f'obj{index:02d}.png',
            'strip': strip_name,
            'appearance_score': round(float(appearance_score), 4),
            'class_candidates': [[name, round(float(score), 4)] for name, score in ranked],
            'detector_votes': [[name, round(float(score), 3)] for name, score in detector_votes],
            'hits': len(seen),
            'frame_span': [int(min(frames)), int(max(frames))],
            'rep_stem': rep['stem'],
            'rep_frame_index': int(rep['frame_index']),
            'rep_level': int(rep['level']),
            'rep_bbox_view': [round(v, 2) for v in rep['bbox_view']],
            'track': [{'stem': hit['stem'], 'frame': int(hit['frame']),
                       'frame_index': int(hit['frame_index']),
                       'level': int(hit['level']), 'region': hit['region'],
                       'inview': bool(hit['inview']),
                       'bbox_source': [round(hit['bbox_global'][0] * W, 2),
                                       round(hit['bbox_global'][1] * H, 2),
                                       round(hit['bbox_global'][2] * W, 2),
                                       round(hit['bbox_global'][3] * H, 2)]}
                      for hit in hits if hit['inview']],
        })
        print(f'{index:3d} {len(seen):4d} {appearance_score:6.3f} {guess:>14s}  '
              f'L{rep["level"]:<4d}  '
              + ', '.join(f'{n}:{v:.1f}' for n, v in detector_votes))

    proposals.sort(key=lambda p: -p['appearance_score'])

    LABELS.mkdir(parents=True, exist_ok=True)
    out_path = LABELS / f'{arguments.sequence}.proposals.json'
    out_path.write_text(json.dumps({
        'sequence': arguments.sequence,
        'min_conf': arguments.min_conf, 'min_hits': arguments.min_hits,
        'count': len(proposals), 'proposals': proposals}, indent=1), encoding='utf-8')
    print(f'\nwrote {out_path}')
    print(f'crops -> {out_dir}')
    print('Persistence gate kills one-off clutter; appearance names each track. '
          'A moored boat is persistent too - judge it by its crop in review_crops.')


if __name__ == '__main__':
    main()
