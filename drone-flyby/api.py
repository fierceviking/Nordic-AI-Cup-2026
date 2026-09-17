"""The endpoint the evaluation service calls.

Run ``python api.py --experiment verifier-v1`` for the experimental crop
verifier, or ``--experiment v4s1`` for the matched detector-only control.
Without a preset, the baseline honours the existing DRONE_WEIGHTS setting.
``--report-conf 0.25`` filters final annotations without altering track updates.

The URL you submit is used exactly as you give it, path included, so if you
keep the ``/predict`` route below then submit ``http://<your-host>:9053/predict``
rather than just the host.
"""

import datetime
import logging
import os
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException


def configure_experiment():
    import argparse

    names = ('baseline', 'v4s1', 'verifier-v1')
    selected = os.environ.get('DRONE_EXPERIMENT', 'baseline')
    report_conf = float(os.environ.get('DRONE_REPORT_CONF', '0.0'))
    verifier_conf = os.environ.get('DRONE_VERIFIER_CONF') or None
    if verifier_conf is not None:
        verifier_conf = float(verifier_conf)
    if __name__ == '__main__':
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--experiment', choices=names, default=selected)
        parser.add_argument('--report-conf', type=float, default=report_conf,
                            help='minimum final confidence after verification and tracking (default: 0)')
        parser.add_argument('--verifier-conf', type=float, default=verifier_conf,
                            help='override verifier rejection threshold before tracking (default: checkpoint value)')
        arguments = parser.parse_args()
        selected, report_conf = arguments.experiment, arguments.report_conf
        verifier_conf = arguments.verifier_conf
        if not 0.0 <= report_conf <= 1.0:
            parser.error('--report-conf must be finite and between 0 and 1')
        if verifier_conf is not None and (not 0.0 <= verifier_conf <= 1.0 or selected != 'verifier-v1'):
            parser.error('--verifier-conf requires verifier-v1 and a finite value between 0 and 1')
    if selected not in names:
        raise ValueError(f'Unknown DRONE_EXPERIMENT: {selected}')
    if not 0.0 <= report_conf <= 1.0:
        raise ValueError('DRONE_REPORT_CONF must be finite and between 0 and 1')
    if verifier_conf is not None and (not 0.0 <= verifier_conf <= 1.0 or selected != 'verifier-v1'):
        raise ValueError('DRONE_VERIFIER_CONF requires verifier-v1 and a finite value between 0 and 1')
    os.environ['DRONE_EXPERIMENT'] = selected
    os.environ['DRONE_REPORT_CONF'] = str(report_conf)
    if verifier_conf is not None:
        os.environ['DRONE_VERIFIER_CONF'] = str(verifier_conf)
    if selected in ('v4s1', 'verifier-v1'):
        root = Path(__file__).resolve().parent
        os.environ.update(DRONE_WEIGHTS=str(root / 'model' / 'v4s1.pt'),
                          DRONE_IMGSZ='960', DRONE_CONF='0.05', DRONE_ALTERNATES='2',
                          DRONE_ALTERNATE_DAMPING='0.5', DRONE_ASSIGN='0', DRONE_MAX_MISSES='3')


configure_experiment()

from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
from example import model_status, predict
from utils import validate_response

HOST = '0.0.0.0'
PORT = 9053

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
start_time = time.time()


@app.post('/predict', response_model=DroneFlybyPredictResponseDto)
def predict_endpoint(request: DroneFlybyPredictRequestDto):
    """Answer one frame."""
    if not model_status()['solver_loaded']:
        raise HTTPException(status_code=503, detail='Selected experiment failed to load; check /health')
    response = predict(request)

    # Fail here, loudly, rather than having the evaluator silently discard the
    # frame. Every rule this checks is a rule the evaluator also enforces.
    validate_response(response)

    logger.info(
        'frame %s (index %s) L%s at (%s, %s): returned %s detections (report_conf=%.2f)',
        request.frame,
        request.frame_index,
        request.view.resolution_level,
        request.view.center_x,
        request.view.center_y,
        len(response.annotations),
        model_status()['report_conf'],
    )
    return response


@app.get('/api')
def hello():
    return {
        'service': 'drone-flyby-usecase',
        'uptime': '{}'.format(datetime.timedelta(seconds=time.time() - start_time)),
    }


@app.get('/health')
def health():
    """Report loaded experiment identity, verifier state and runtime settings."""
    import solution
    loaded = model_status()
    ready = loaded['solver_loaded'] and loaded['verifier'].get('last_error') is None
    return {
        **loaded,
        'status': 'ready' if ready else 'DEGRADED - selected experiment unavailable or failed',
        'weights': solution.WEIGHTS,
        'weights_exist': Path(solution.WEIGHTS).is_file(),
        'device': solution.DEVICE,
        'imgsz': solution.INFER_IMGSZ,
        'confidence': solution.DETECT_CONF,
        'alternate_classes': solution.ALTERNATE_CLASSES,
        'camera_loop': [list(entry) for entry in solution.CameraPolicy.LOOP],
        'recording_to': os.environ.get('DRONE_RECORD_DIR') or None,
        'uptime': '{}'.format(datetime.timedelta(seconds=time.time() - start_time)),
    }


@app.get('/')
def index():
    return "Your endpoint is running!"


if __name__ == '__main__':
    from example import SOLVER
    import solution

    if SOLVER is None:
        logger.error('=' * 70)
        logger.error('SELECTED EXPERIMENT DID NOT LOAD. /predict will return HTTP 503.')
        logger.error('Check that the weights exist and that torch can see the '
                     'GPU, then restart.')
        logger.error('=' * 70)
    else:
        logger.info('=' * 70)
        logger.info('experiment %s', model_status()['experiment'])
        logger.info('verifier   %s', model_status()['verifier'])
        logger.info('weights   %s', solution.WEIGHTS)
        logger.info('device    %s   imgsz %s   conf %s',
                    solution.DEVICE, solution.INFER_IMGSZ, solution.DETECT_CONF)
        logger.info('report confidence floor %s', model_status()['report_conf'])
        logger.info('alternates %s   camera loop %s positions',
                    solution.ALTERNATE_CLASSES, len(solution.CameraPolicy.LOOP))
        record = os.environ.get('DRONE_RECORD_DIR')
        logger.info('recording %s', record or 'off')
        logger.info('=' * 70)
        logger.info('Detector loaded and warmed up; serving on %s:%s', HOST, PORT)

    uvicorn.run(app, host=HOST, port=PORT)
