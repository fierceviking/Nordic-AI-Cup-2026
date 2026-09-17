"""Draw hand labels and recorded predictions on a frame, to check the labels.

Ground truth eyeballed off a quarter-resolution strip can easily be wrong by
more than the IoU threshold allows, which would make the whole benchmark
measure the labeller rather than the model. This renders both so the question
can be settled by looking.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))

from score_recording import SOURCE_H, SOURCE_W, truth_for_frame  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', required=True)
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--frames', default='')
    parser.add_argument('--min-conf', type=float, default=0.30)
    arguments = parser.parse_args()

    labels = json.loads(Path(arguments.labels).read_text())
    root = LAB / 'recordings' / arguments.sequence
    metas = sorted((root / 'meta').glob('*.json'))

    wanted = {int(v) for v in arguments.frames.split(',')} if arguments.frames else None
    panels = []
    for path in metas:
        meta = json.loads(path.read_text())
        if meta.get('resolution_level') != 0:
            continue
        index = meta['frame_index']
        if wanted is not None and index not in wanted:
            continue
        truth = truth_for_frame(labels, index)
        if not truth:
            continue

        image_path = root / 'views' / (path.stem + '.png')
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        vh, vw = image.shape[:2]
        fx, fy = vw / SOURCE_W, vh / SOURCE_H

        for prediction in meta.get('predictions', []):
            if float(prediction['confidence']) < arguments.min_conf:
                continue
            x1, y1, x2, y2 = prediction['bbox']
            cv2.rectangle(image, (int(x1 * vw), int(y1 * vh)),
                          (int(x2 * vw), int(y2 * vh)), (0, 140, 255), 1)

        for item in truth:
            x1, y1, x2, y2 = item['bbox']
            cv2.rectangle(image, (int(x1 * fx), int(y1 * fy)),
                          (int(x2 * fx), int(y2 * fy)), (0, 255, 0), 2)
            cv2.putText(image, item['object_id'], (int(x1 * fx), int(y1 * fy) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'frame_index {index}  green=label  orange=ours',
                    (6, vh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(image)
        if len(panels) >= 4:
            break

    if not panels:
        raise SystemExit('no frames with labels found')
    out = LAB / 'out' / f'labelcheck_{arguments.sequence[:8]}.jpg'
    cv2.imwrite(str(out), cv2.vconcat(panels), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f'wrote {out} ({len(panels)} frames)')


if __name__ == '__main__':
    main()
