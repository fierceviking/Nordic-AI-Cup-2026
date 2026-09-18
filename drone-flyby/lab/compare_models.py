"""Compare detectors on the two axes that have pulled against each other.

Helsinki rewards a model that can use Helsinki ground as a cue; the
foreign-background set rewards one that cannot. v7nft scored 0.51 and 0.31,
worse than both of its parents on one axis each, which is only visible when
both are measured together. Neither predicts the competition - four Helsinki
results have now pointed the wrong way - so treat these as regression guards.

    python lab/compare_models.py --models v4s1 v7n v7nft v8mix
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
ROOT = LAB.parent

PIPELINE = """
import sys
from pathlib import Path
import numpy as np
ROOT = Path(r'{root}')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'lab'))
import simulate
from solution import Solver
solver = Solver()
preds, stats = simulate.replay(lambda r: solver(
    r.sequence_id, r.frame, r.image, r.source_region_xyxy, r.resolution_level,
    r.center_x, r.center_y, r.allowed_levels, r.maximum_center_delta))
counts = np.array([len(v) for _, v in sorted(preds.items())])
map50, per_class = simulate.evaluate(preds)
zero = sum(1 for v in per_class.values() if v < 1e-9)
print(f'RESULT map={{map50:.4f}} preds={{counts.mean():.1f}} zero={{zero}} '
      f'ms={{stats[\"mean_ms\"]:.0f}}')
"""

BASE = dict(DRONE_IMGSZ='960', DRONE_CONF='0.05', DRONE_ALTERNATES='2',
            DRONE_ALTERNATE_DAMPING='0.5', DRONE_ASSIGN='0', DRONE_MAX_MISSES='3',
            DRONE_MAX_AGE='45', DRONE_MIN_HITS='1', DRONE_REPORT_MAX='400',
            DRONE_DOMAIN='0', DRONE_CAMERA='loop')


def helsinki_pipeline(weights):
    env = dict(os.environ)
    env.update(BASE)
    env['DRONE_WEIGHTS'] = str(weights)
    out = subprocess.run([sys.executable, '-c', PIPELINE.format(root=ROOT)],
                         env=env, capture_output=True, text=True, cwd=str(ROOT))
    line = [l for l in out.stdout.splitlines() if l.startswith('RESULT')]
    if not line:
        return None
    return dict(x.split('=') for x in line[0].split()[1:])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', nargs='+', default=['v4s1', 'v7n', 'v8mix'])
    parser.add_argument('--foreign', default=str(LAB / 'dataset_v7' / 'data.yaml'))
    parser.add_argument('--out', type=Path, default=LAB / 'out' / 'model_compare.json')
    arguments = parser.parse_args()

    logging.getLogger('ultralytics').setLevel(logging.ERROR)
    from ultralytics import YOLO

    available = []
    for name in arguments.models:
        path = ROOT / 'model' / f'{name}.pt'
        if path.is_file():
            available.append((name, path))
        else:
            print(f'skipping {name}: {path} missing')
    if not available:
        raise SystemExit('no weights found')

    rows = {}
    print(f'\n{"model":9s} | {"HELSINKI pipeline":>32s} | {"FOREIGN backgrounds":>26s}')
    print(f'{"":9s} | {"mAP50":>8s} {"preds/f":>8s} {"zeroAP":>6s} {"ms":>5s} | '
          f'{"mAP50":>8s} {"recall":>8s} {"prec":>6s}')
    print('-' * 74)
    for name, path in available:
        pipeline = helsinki_pipeline(path)
        result = YOLO(str(path)).val(data=arguments.foreign, imgsz=960, batch=16,
                                     workers=0, verbose=False, plots=False)
        rows[name] = {
            'helsinki_map50': float(pipeline['map']) if pipeline else None,
            'helsinki_preds_per_frame': float(pipeline['preds']) if pipeline else None,
            'helsinki_zero_ap_classes': int(pipeline['zero']) if pipeline else None,
            'latency_ms': float(pipeline['ms']) if pipeline else None,
            'foreign_map50': float(result.box.map50),
            'foreign_recall': float(result.box.mr),
            'foreign_precision': float(result.box.mp),
        }
        r = rows[name]
        print(f'{name:9s} | {r["helsinki_map50"]:8.4f} {r["helsinki_preds_per_frame"]:8.1f} '
              f'{r["helsinki_zero_ap_classes"]:6d} {r["latency_ms"]:5.0f} | '
              f'{r["foreign_map50"]:8.4f} {r["foreign_recall"]:8.3f} '
              f'{r["foreign_precision"]:6.3f}')

    print('\nA model beaten on BOTH axes by one other model is strictly dominated:')
    for name, row in rows.items():
        beaten = [other for other, value in rows.items()
                  if other != name
                  and value['helsinki_map50'] >= row['helsinki_map50']
                  and value['foreign_map50'] >= row['foreign_map50']]
        print(f'  {name:9s} {"dominated by " + ", ".join(beaten) if beaten else "on the frontier"}')

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(f'\nwrote {arguments.out}')


if __name__ == '__main__':
    main()
