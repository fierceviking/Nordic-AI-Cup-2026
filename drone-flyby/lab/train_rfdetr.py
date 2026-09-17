"""Fine-tune RF-DETR on the synthetic dataset.

The point of the comparison is the architecture, not the input size, so the
resolution defaults to ~960 rather than the checkpoint's native 576. Our
objects are about 13.5 px across in a 960x540 view; letterboxing that into a
576x576 square would shrink them to roughly 8 px and the result would measure
downsampling rather than RF-DETR. Resolution must be divisible by 56.
"""

import argparse
import sys
from pathlib import Path

# RF-DETR prints its metrics tables with rich, whose box-drawing characters a
# cp1252 Windows console cannot encode - it killed a run after epoch 1.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

LAB = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='COCO dataset root')
    parser.add_argument('--size', default='small',
                        choices=['nano', 'small', 'medium', 'large'])
    parser.add_argument('--resolution', type=int, default=952)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--accum', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--name', default='rfdetr')
    arguments = parser.parse_args()

    if arguments.resolution % 56:
        raise SystemExit(f'resolution must be divisible by 56, '
                         f'got {arguments.resolution}')

    import rfdetr

    classes = {'nano': 'RFDETRNano', 'small': 'RFDETRSmall',
               'medium': 'RFDETRMedium', 'large': 'RFDETRLarge'}
    factory = getattr(rfdetr, classes[arguments.size])

    output = LAB / 'runs' / arguments.name
    output.mkdir(parents=True, exist_ok=True)

    print(f'model      {classes[arguments.size]}')
    print(f'resolution {arguments.resolution}')
    print(f'batch      {arguments.batch} x {arguments.accum} accum '
          f'= {arguments.batch * arguments.accum} effective')
    print(f'output     {output}', flush=True)

    model = factory(resolution=arguments.resolution)
    model.train(
        dataset_dir=str(Path(arguments.data)),
        epochs=arguments.epochs,
        batch_size=arguments.batch,
        grad_accum_steps=arguments.accum,
        lr=arguments.lr,
        output_dir=str(output),
    )
    print(f'\nbest weights: {output / "checkpoint_best_total.pth"}')


if __name__ == '__main__':
    main()
