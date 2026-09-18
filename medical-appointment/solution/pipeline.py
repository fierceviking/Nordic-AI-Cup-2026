"""The serving pipeline: audio in, answers and spans out.

Transcription happens once per request and is shared by all ten questions.
Whisper and the quote reader run locally through MLX; the supporting neural
evidence candidate uses PyTorch MPS on Apple Silicon.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import List, Optional, Tuple

from . import asr
from .approaches import build

logger = logging.getLogger(__name__)

Span = Tuple[float, float]

APPROACH = os.environ.get('APPROACH', 'quote_judge')
ASR_MODEL = os.environ.get('ASR_MODEL', asr.DEFAULT_MODEL)

# Cache transcripts by content hash. The service sends each conversation once,
# so this buys nothing there; locally it makes a re-run of local_evaluator.py
# cost seconds instead of minutes.
CACHE_TRANSCRIPTS = os.environ.get('CACHE_TRANSCRIPTS', '1') == '1'

_approach = None
_lock = threading.Lock()


def get_approach():
    global _approach
    with _lock:
        if _approach is None:
            logger.info('Building approach %s', APPROACH)
            _approach = build(APPROACH)
        return _approach


def warmup() -> None:
    """Load every model before the first request arrives.

    The evaluator allows 60 seconds per conversation and polls the root URL
    first, so paying for the weights at import time costs nothing and keeps the
    first conversation inside its budget.
    """
    try:
        asr.get_model(ASR_MODEL)
        get_approach()
    except Exception:
        logger.exception('Warmup failed; the first request will pay for it')


def transcribe(audio_bytes: bytes) -> asr.Transcript:
    key = f'serve_{asr.audio_key(audio_bytes)}'

    if CACHE_TRANSCRIPTS:
        cached = asr.load_cached(key, ASR_MODEL)
        if cached is not None:
            return cached

    transcript = asr.transcribe_bytes(audio_bytes, model_name=ASR_MODEL)

    if CACHE_TRANSCRIPTS:
        try:
            asr.save_cached(key, transcript, ASR_MODEL)
        except OSError:
            logger.warning('Could not cache the transcript', exc_info=True)

    return transcript


def answer(
    audio_bytes: bytes, questions: List[str]
) -> List[Tuple[bool, Optional[Span]]]:
    """Answer every question about one conversation. Never raises."""
    started = time.time()

    try:
        transcript = transcribe(audio_bytes)
    except Exception:
        logger.exception('Transcription failed; guessing')
        return [(True, None) for _ in questions]

    transcribed = time.time()

    try:
        predictions = get_approach().predict(transcript, questions)
    except Exception:
        logger.exception('Answering failed; guessing')
        return [(True, None) for _ in questions]

    if len(predictions) != len(questions):
        logger.error(
            'approach returned %d predictions for %d questions',
            len(predictions), len(questions),
        )
        predictions = list(predictions)[: len(questions)]
        predictions += [(True, None)] * (len(questions) - len(predictions))

    logger.info(
        '%.1f s audio: %.1f s transcribing, %.1f s answering %d questions',
        transcript.duration, transcribed - started,
        time.time() - transcribed, len(questions),
    )

    return predictions
