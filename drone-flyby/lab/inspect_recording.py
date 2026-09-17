"""Summarise a recorded attempt and draw what we predicted on it.

The validation sequence is the only held-out data available, and it arrives as
248 views with no ground truth. What can be read off it: whether frames were
dropped, where the camera went, how confident the detector was, and - by
eye - whether the predictions are sane or nonsense.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))

RECORDINGS = LAB / 'recordings'


def load(directory: Path):
    records = []
    for path in sorted((directory / 'meta').glob('*.json')):
        with open(path) as handle:
            meta = json.load(handle)
        meta['_view'] = directory / 'views' / (path.stem + '.png')
        records.append(meta)
    return sorted(records, key=lambda m: m['frame'])


def draw_inference(image, meta, predictions, weights, threshold):
    height, width = image.shape[:2]
    left, top, right, bottom = meta['source_region_xyxy']
    visible = []
    for prediction in sorted(predictions, key=lambda item: -item['confidence']):
        if prediction['confidence'] < threshold:
            continue
        source_left, source_top, source_right, source_bottom = prediction['bbox']
        box = [(source_left - left) * width / (right - left),
               (source_top - top) * height / (bottom - top),
               (source_right - left) * width / (right - left),
               (source_bottom - top) * height / (bottom - top)]
        if box[2] <= 0 or box[3] <= 0 or box[0] >= width or box[1] >= height:
            continue
        box = [int(round(np.clip(value, 0, limit))) for value, limit in
               zip(box, (width - 1, height - 1, width - 1, height - 1))]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        visible.append((prediction, box))
    header = 64
    panel = np.full((header + max(height, 100 + len(visible) * 22), width + 350, 3), 24, np.uint8)
    panel[header:header + height, :width] = image
    cv2.putText(panel, f'{Path(weights).name} | frame {meta["frame"]} | index {meta["frame_index"]} | L{meta["resolution_level"]}',
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(panel, f'Fresh detector predictions; no tracking or ground truth | display >= {threshold:.2f}',
                (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(panel, f'{len(visible)} shown / {len(predictions)} raw',
                (width + 12, header + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (245, 245, 245), 1, cv2.LINE_AA)
    for number, (prediction, box) in enumerate(visible, start=1):
        hue = np.uint8([[[number * 37 % 180, 200, 255]]])
        colour = tuple(int(value) for value in cv2.cvtColor(hue, cv2.COLOR_HSV2BGR)[0, 0])
        start = (box[0], box[1] + header)
        end = (box[2], box[3] + header)
        cv2.rectangle(panel, start, end, colour, 1)
        cv2.putText(panel, str(number), (min(box[0], width - 30), max(header + 12, start[1] - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
        cv2.putText(panel, f'{number:2d} {prediction["object_id"]} {prediction["confidence"]:.2f}',
                    (width + 12, header + 52 + (number - 1) * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, colour, 1, cv2.LINE_AA)
    return panel, len(visible)


def fresh_inference(records, arguments):
    from solution import Detector

    weights = arguments.weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if arguments.indices:
        selected = sorted({min(range(len(records)), key=lambda index:
                               abs(records[index]['frame_index'] - wanted))
                           for wanted in map(int, arguments.indices.split(','))})
    else:
        selected = np.linspace(0, len(records) - 1, min(arguments.sheet, len(records)), dtype=int)
    output = arguments.out or LAB / 'out' / f'inference_{weights.stem}_{arguments.sequence[:8]}'
    output.mkdir(parents=True, exist_ok=True)
    detector = Detector(weights=str(weights), imgsz=960, confidence=0.05)
    manifest = {'weights': str(weights), 'sequence': arguments.sequence,
                'mode': 'fresh detector only; no tracking or ground truth',
                'inference_confidence': 0.05, 'display_confidence': arguments.min_conf,
                'coordinate_system': 'source pixels xyxy', 'frames': []}
    panels = []
    for index in selected:
        meta = records[index]
        image = cv2.imread(str(meta['_view']))
        if image is None:
            raise ValueError(f'Cannot decode {meta["_view"]}')
        predictions = [{'object_id': str(item['object_id']), 'confidence': float(item['confidence']),
                        'bbox': [float(value) for value in item['bbox']]}
                       for item in detector(image, meta['source_region_xyxy'])]
        panel, shown = draw_inference(image, meta, predictions, weights, arguments.min_conf)
        path = output / f'frame_{meta["frame"]:06d}_L{meta["resolution_level"]}.png'
        if not cv2.imwrite(str(path), panel):
            raise OSError(f'Could not write {path}')
        manifest['frames'].append({'frame': meta['frame'], 'frame_index': meta['frame_index'],
                                   'level': meta['resolution_level'], 'source_region_xyxy': meta['source_region_xyxy'],
                                   'input': str(meta['_view']), 'output': str(path),
                                   'displayed': shown, 'predictions': predictions})
        panels.append(cv2.resize(panel, (650, int(round(panel.shape[0] * 650 / panel.shape[1])))))
        print(f'{path.name}: {shown} shown / {len(predictions)} raw detections', flush=True)
    cell_height = max(panel.shape[0] for panel in panels)
    sheet = np.full((((len(panels) + 1) // 2) * cell_height, 1300, 3), 24, np.uint8)
    for index, panel in enumerate(panels):
        top, left = (index // 2) * cell_height, (index % 2) * 650
        sheet[top:top + panel.shape[0], left:left + 650] = panel
    if not cv2.imwrite(str(output / 'overview.jpg'), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError('Could not write overview')
    (output / 'predictions.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Fresh inference saved to {output}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--sheet', type=int, default=12,
                        help='how many frames to draw into a contact sheet')
    parser.add_argument('--weights', type=Path, help='run fresh detector inference instead of drawing recorded answers')
    parser.add_argument('--min-conf', type=float, default=0.30, help='display threshold for fresh predictions')
    parser.add_argument('--indices', default='', help='requested frame indices; selects nearest recorded views')
    parser.add_argument('--out', type=Path, help='output directory for fresh inference')
    arguments = parser.parse_args()
    if arguments.sheet < 1 or not 0 <= arguments.min_conf <= 1:
        parser.error('sheet must be positive and min-conf in [0, 1]')

    directory = RECORDINGS / arguments.sequence
    records = load(directory)
    if not records:
        parser.error(f'No recorded frames in {directory}')
    if arguments.weights:
        fresh_inference(records, arguments)
        return
    print(f'{len(records)} frames recorded')

    frames = [m['frame'] for m in records]
    indices = [m['frame_index'] for m in records]
    print(f'  frame numbers  {min(frames)}..{max(frames)}')
    print(f'  frame indices  {min(indices)}..{max(indices)}')
    missing = sorted(set(range(min(indices), max(indices) + 1)) - set(indices))
    print(f'  skipped frames {len(missing)}'
          + (f' -> {missing[:20]}' if missing else ' (none: we kept up)'))

    levels = Counter(m['resolution_level'] for m in records)
    print(f'  camera levels  ' +
          ', '.join(f'L{k}: {v} ({v / len(records) * 100:.0f}%)'
                    for k, v in sorted(levels.items())))

    counts = [len(m['predictions']) for m in records]
    print(f'  predictions    mean {np.mean(counts):.1f} per frame, '
          f'min {min(counts)}, max {max(counts)}')

    classes = Counter(p['object_id'] for m in records for p in m['predictions'])
    print('\n  predicted class distribution (all frames, includes repeats '
          'of the same tracked object):')
    for name, count in classes.most_common():
        print(f'    {name:16s} {count:6d}')

    confidences = np.array([p['confidence'] for m in records
                            for p in m['predictions']])
    if len(confidences):
        print(f'\n  confidence     median {np.median(confidences):.3f}, '
              f'p90 {np.percentile(confidences, 90):.3f}, '
              f'max {confidences.max():.3f}')

    # ------------------------------------------------------- contact sheet
    step = max(1, len(records) // arguments.sheet)
    chosen = records[::step][:arguments.sheet]
    tiles = []
    for meta in chosen:
        image = cv2.imread(str(meta['_view']))
        if image is None:
            continue
        h, w = image.shape[:2]
        sx1, sy1, sx2, sy2 = meta['source_region_xyxy']
        for prediction in meta['predictions']:
            # Predictions are frame-global; map them back into this view.
            gx1, gy1, gx2, gy2 = prediction['bbox']
            x1 = (gx1 * meta['original_width'] - sx1) / (sx2 - sx1) * w
            y1 = (gy1 * meta['original_height'] - sy1) / (sy2 - sy1) * h
            x2 = (gx2 * meta['original_width'] - sx1) / (sx2 - sx1) * w
            y2 = (gy2 * meta['original_height'] - sy1) / (sy2 - sy1) * h
            if x2 < 0 or y2 < 0 or x1 > w or y1 > h:
                continue
            colour = (0, 255, 255) if prediction['confidence'] > 0.3 else (120, 120, 120)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), colour, 1)
            cv2.putText(image, f'{prediction["object_id"]} {prediction["confidence"]:.2f}',
                        (int(x1), max(9, int(y1) - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, colour, 1, cv2.LINE_AA)
        cv2.putText(image, f'frame {meta["frame"]} L{meta["resolution_level"]}',
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(cv2.resize(image, (640, 360)))

    rows = [cv2.hconcat(tiles[i:i + 3]) for i in range(0, len(tiles) - 2, 3)]
    if rows:
        out = LAB / 'out' / f'recorded_{arguments.sequence[:8]}.jpg'
        cv2.imwrite(str(out), cv2.vconcat(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
