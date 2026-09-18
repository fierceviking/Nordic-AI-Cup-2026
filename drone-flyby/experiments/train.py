import sys, os
from ultralytics import settings
settings.update({'mlflow': False})
from ultralytics import YOLO

# Write runs to a short, non-OneDrive-synced path to avoid the intermittent
# Windows OSError[22] on best.pt writes (antivirus/OneDrive locking the file).
OUT = os.environ.get('DF_RUNDIR', 'C:/df_runs')
BASE = os.environ.get('DF_BASE', 'yolo11m.pt')
NAME = os.environ.get('DF_NAME', 'det_m')

def main():
    model = YOLO(BASE)
    model.train(
        data='dataset/data.yaml',
        epochs=150, imgsz=960, batch=12, device=0, workers=4,
        patience=40, close_mosaic=15,
        degrees=180.0, fliplr=0.5, flipud=0.5, scale=0.5, translate=0.1,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, mosaic=1.0, mixup=0.1,
        project=OUT, name=NAME, exist_ok=True, verbose=False, plots=False,
    )
    print('DONE', os.path.join(OUT, NAME, 'weights', 'best.pt'))

if __name__ == '__main__':
    main()
