"""The endpoint the evaluation service calls.

You should not need to change much in here. Put your model in ``example.py``
and leave the transport alone.

The URL you submit is used exactly as you give it, path included, so if you
keep the ``/predict`` route below then submit ``http://<your-host>:9053/predict``
rather than just the host.
"""

import datetime
import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
from solution import predict, Detector
from utils import validate_response

HOST = '0.0.0.0'
PORT = 9053

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load + warm the detector before the server accepts any frame.

    The evaluator polls ``/`` and only starts the 3 fps clock once it answers.
    Uvicorn does not begin serving until this startup phase finishes, so paying
    the CUDA init + warmup here (on the dedicated inference thread, via
    Detector.get) means the FIRST real frame is already warm. Without this the
    first served frame stalls ~2 s and the realtime clock skips the opening of
    the sequence -- measured as 0.643 -> 0.302.
    """
    try:
        Detector.get()
        logger.info('Detector warmed up; ready to serve.')
    except Exception:
        logger.exception('Detector warmup failed; will retry lazily per frame.')
    yield


app = FastAPI(lifespan=lifespan)
start_time = time.time()


@app.post('/predict', response_model=DroneFlybyPredictResponseDto)
def predict_endpoint(request: DroneFlybyPredictRequestDto):
    """Answer one frame."""
    response = predict(request)

    # Fail here, loudly, rather than having the evaluator silently discard the
    # frame. Every rule this checks is a rule the evaluator also enforces.
    validate_response(response)

    logger.info(
        'frame %s (index %s) L%s at (%s, %s): returned %s detections',
        request.frame,
        request.frame_index,
        request.view.resolution_level,
        request.view.center_x,
        request.view.center_y,
        len(response.annotations),
    )
    return response


@app.get('/api')
def hello():
    return {
        'service': 'drone-flyby-usecase',
        'uptime': '{}'.format(datetime.timedelta(seconds=time.time() - start_time)),
    }


@app.get('/')
def index():
    return "Your endpoint is running!"


if __name__ == '__main__':
    uvicorn.run('api:app', host=HOST, port=PORT)
