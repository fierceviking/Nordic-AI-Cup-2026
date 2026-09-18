"""Train a YOLO detector on the synthetic views.

Two stages, because the failure mode being fixed is false positives rather than
missed objects:

  stage 1  positives plus pure-background negatives, full augmentation. Teaches
           the detector that "nothing here" is a valid answer.
  stage 2  the same images minus the backgrounds, lower learning rate, gentler
           augmentation. Spends the remaining capacity on localising and
           classifying what is actually there.

``--stage 2`` switches to the fine-tune schedule.
"""

import argparse
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
DATASET = LAB / 'dataset'
sys.path.insert(0, str(LAB.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolo11s.pt')
    parser.add_argument('--data', default=str(DATASET / 'data.yaml'))
    parser.add_argument('--name', default='v1')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--batch', type=int, default=12)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--save-period', type=int, default=-1)
    parser.add_argument('--stage', type=int, default=1, choices=(1, 2))
    parser.add_argument('--lr0', type=float, default=None,
                        help='override the stage learning rate')
    parser.add_argument('--optimizer', default='auto',
                        choices=('auto', 'SGD', 'Adam', 'AdamW'),
                        help="'auto' silently replaces lr0, so pass an explicit "
                             "optimizer whenever --lr0 is meant to be honoured")
    parser.add_argument('--light-aug', action='store_true',
                        help='flips and mild scale only, the recipe current '
                             'transfer work uses; preserves pretrained features')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--monitor-every', type=int, default=5,
                        help='report detections per frame on the recorded '
                             'competition views every N epochs. Readout only - '
                             'checkpoint selection stays on the real-frame '
                             'validation split. 0 disables.')
    arguments = parser.parse_args()

    from ultralytics import YOLO

    stage_two = arguments.stage == 2
    light = arguments.light_aug
    model = YOLO(arguments.model)
    requested_lr = (arguments.lr0 if arguments.lr0 is not None
                    else (0.0002 if stage_two else 0.01))
    if arguments.lr0 is not None and arguments.optimizer == 'auto':
        parser.error('--lr0 with --optimizer auto is ignored by Ultralytics; '
                     'pass --optimizer AdamW or SGD')

    if arguments.monitor_every > 0:
        sys.path.insert(0, str(LAB))
        import clutter_monitor
        clutter_monitor.attach(
            model, every=arguments.monitor_every,
            out=LAB / 'runs' / arguments.name / 'clutter.jsonl',
            imgsz=arguments.imgsz)

    model.train(
        data=arguments.data,
        epochs=arguments.epochs,
        imgsz=arguments.imgsz,
        batch=arguments.batch,
        device=0,
        workers=arguments.workers,
        project=str(LAB / 'runs'),
        name=arguments.name,
        exist_ok=True,
        resume=arguments.resume,
        patience=arguments.patience,
        save_period=arguments.save_period,
        cache=False,
        pretrained=True,
        optimizer=arguments.optimizer,
        cos_lr=True,
        # Fine-tuning starts from a converged model, so it needs a small,
        # short schedule; restarting at the stage-1 rate would undo stage 1.
        lr0=requested_lr,
        lrf=0.05 if stage_two else 0.01,
        warmup_epochs=0.0 if stage_two else 3.0,
        close_mosaic=3 if stage_two else 10,
        # Geometry: objects are photographed from directly above at a fixed
        # altitude, so orientation is arbitrary but scale is nearly fixed.
        # Ultralytics transforms the labels with the image for all of these.
        degrees=180.0,
        fliplr=0.5,
        flipud=0.5,
        scale=0.10 if light else (0.20 if stage_two else 0.30),
        translate=0.05 if light else (0.10 if stage_two else 0.20),
        shear=0.0 if light else (1.0 if stage_two else 3.0),
        perspective=0.0,
        mosaic=0.0 if light else (0.4 if stage_two else 1.0),
        mixup=0.0 if light else (0.0 if stage_two else 0.10),
        copy_paste=0.0,
        # Colour: the evaluation scene is a different location under different
        # light, so let the model see a wide range of it.
        hsv_h=0.015 if light else (0.020 if stage_two else 0.035),
        hsv_s=0.40 if light else (0.60 if stage_two else 0.85),
        hsv_v=0.30 if light else (0.40 if stage_two else 0.55),
        erasing=0.0,
        val=True,
        plots=True,
        seed=0,
    )
    effective = model.trainer.args
    print(f'optimizer {effective.optimizer}  lr0 requested {requested_lr} '
          f'-> effective {effective.lr0}')
    if effective.optimizer != 'auto' and effective.lr0 != requested_lr:
        print('WARNING: the learning rate was replaced during setup')
    print('best weights:', LAB / 'runs' / arguments.name / 'weights' / 'best.pt')


if __name__ == '__main__':
    main()
