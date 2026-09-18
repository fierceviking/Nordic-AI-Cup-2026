"""Fetch aerial backgrounds for compositing the Helsinki sprites onto.

The detector has only ever seen Helsinki ground, so background is a usable cue
and it learns it. These supply foreign ground at a comparable scale so the same
object appears over many different surfaces.

  LoveDA    0.3 m/px nadir, urban and rural, 1024x1024
  RESISC45  varied, 256x256, selected scene types that match the competition
            harbour: harbor, ship, industrial_area, storage_tank, ...

Licences are non-commercial/research; recorded in backgrounds/SOURCES.json
because the top five teams must submit training code.
"""

import argparse
import io
import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
BACKGROUNDS = LAB / 'backgrounds'

RESISC_CLASSES = (
    'harbor', 'ship', 'industrial_area', 'commercial_area', 'storage_tank',
    'dense_residential', 'medium_residential', 'parking_lot', 'bridge',
    'freeway', 'runway', 'airport', 'railway_station', 'thermal_power_station',
    'intersection', 'river', 'beach', 'meadow', 'sparse_residential', 'island',
)
SOURCES = {
    'LoveDA': {'repo': 'mrmeepsle/LoveDa', 'licence': 'CC-BY-NC-4.0',
               'gsd_m_per_px': 0.3, 'use': 'background only, no labels used'},
    'RESISC45': {'repo': 'blanchon/RESISC45', 'licence': 'research use only',
                 'note': 'NWPU-RESISC45', 'use': 'background only, no labels used'},
}


def fetch_resisc(limit_per_class):
    from huggingface_hub import snapshot_download
    patterns = [f'data/{name}/*.jpg' for name in RESISC_CLASSES]
    target = BACKGROUNDS / 'resisc45'
    target.mkdir(parents=True, exist_ok=True)
    print(f'RESISC45: {len(RESISC_CLASSES)} scene types -> {target}', flush=True)
    path = snapshot_download('blanchon/RESISC45', repo_type='dataset',
                             allow_patterns=patterns, max_workers=8)
    kept = 0
    for name in RESISC_CLASSES:
        folder = Path(path) / 'data' / name
        if not folder.is_dir():
            print(f'  {name:22s} missing')
            continue
        files = sorted(folder.glob('*.jpg'))[:limit_per_class]
        for source in files:
            link = target / f'{name}_{source.name}'
            if not link.exists():
                link.write_bytes(source.read_bytes())
            kept += 1
        print(f'  {name:22s} {len(files):4d}', flush=True)
    return kept


def fetch_loveda(shards, per_shard):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image

    target = BACKGROUNDS / 'loveda'
    target.mkdir(parents=True, exist_ok=True)
    print(f'LoveDA: {shards} shards -> {target}', flush=True)
    kept = 0
    for index in range(shards):
        name = f'data/train-{index:05d}-of-00009.parquet'
        print(f'  downloading {name}', flush=True)
        path = hf_hub_download('mrmeepsle/LoveDa', name, repo_type='dataset')
        table = pq.read_table(path)
        column = next((c for c in table.column_names if 'image' in c.lower()),
                      table.column_names[0])
        rows = table.column(column).to_pylist()[:per_shard]
        for row in rows:
            payload = row['bytes'] if isinstance(row, dict) else row
            if not payload:
                continue
            out = target / f'loveda_{index:02d}_{kept:05d}.jpg'
            if not out.exists():
                Image.open(io.BytesIO(payload)).convert('RGB').save(out, quality=92)
            kept += 1
        print(f'    shard {index}: {len(rows)} images (total {kept})', flush=True)
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resisc-per-class', type=int, default=300)
    parser.add_argument('--loveda-shards', type=int, default=3)
    parser.add_argument('--loveda-per-shard', type=int, default=700)
    parser.add_argument('--skip-loveda', action='store_true')
    parser.add_argument('--skip-resisc', action='store_true')
    arguments = parser.parse_args()

    BACKGROUNDS.mkdir(parents=True, exist_ok=True)
    counts = {}
    if not arguments.skip_resisc:
        counts['resisc45'] = fetch_resisc(arguments.resisc_per_class)
    if not arguments.skip_loveda:
        counts['loveda'] = fetch_loveda(arguments.loveda_shards,
                                        arguments.loveda_per_shard)
    (BACKGROUNDS / 'SOURCES.json').write_text(
        json.dumps({'sources': SOURCES, 'counts': counts}, indent=2), encoding='utf-8')
    print(f'\nbackgrounds: {counts}')
    print(f'wrote {BACKGROUNDS / "SOURCES.json"}')


if __name__ == '__main__':
    main()
