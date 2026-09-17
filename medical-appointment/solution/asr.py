"""Local speech recognition with word-level timestamps.

The span we return is read straight off the ASR timings, so word timings are
part of the answer rather than a convenience. Everything here runs locally.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

TRANSCRIPT_CACHE = Path(__file__).resolve().parent.parent / 'transcripts'

DEFAULT_MODEL = os.environ.get('ASR_MODEL', 'large-v3')
DEFAULT_DEVICE = os.environ.get('ASR_DEVICE', 'auto')
DEFAULT_COMPUTE_TYPE = os.environ.get('ASR_COMPUTE_TYPE', 'float16')

# Spoken medical English the generic model tends to mangle. Fed to Whisper as an
# initial prompt: it biases the decoder without constraining it.
INITIAL_PROMPT = (
    'Consultation between a doctor and a patient. Medication doses in '
    'milligrams, micrograms and millilitres, tablets, vaccines, blood '
    'pressure, cholesterol, blood sugar, antibiotics, penicillin, ibuprofen, '
    'paracetamol, prednisolone, metformin, levothyroxine, follow-up in weeks.'
)


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float = 1.0


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)


@dataclass
class Transcript:
    text: str
    segments: List[Segment]
    duration: float
    model: str = ''

    @property
    def words(self) -> List[Word]:
        out: List[Word] = []
        for segment in self.segments:
            out.extend(segment.words)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'duration': self.duration,
            'model': self.model,
            'segments': [asdict(segment) for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> 'Transcript':
        segments = [
            Segment(
                start=float(segment['start']),
                end=float(segment['end']),
                text=segment['text'],
                words=[Word(**word) for word in segment.get('words', [])],
            )
            for segment in payload['segments']
        ]
        return cls(
            text=payload['text'],
            segments=segments,
            duration=float(payload.get('duration') or 0.0),
            model=payload.get('model', ''),
        )


# --------------------------------------------------------------------------- #
# Model handling
# --------------------------------------------------------------------------- #

_model = None
_model_key: Optional[str] = None
_model_lock = threading.Lock()
_dlls_ready = False


def _ensure_cuda_libraries() -> None:
    """Point CTranslate2 at the CUDA DLLs torch ships, on Windows.

    CTranslate2 wheels do not bundle cuBLAS or cuDNN; without this it fails
    with "Library cublas64_12.dll is not found" even though the GPU is there.
    """
    global _dlls_ready
    if _dlls_ready or os.name != 'nt':
        _dlls_ready = True
        return

    try:
        import torch

        lib = Path(torch.__file__).resolve().parent / 'lib'
        if lib.is_dir():
            os.add_dll_directory(str(lib))
    except Exception:  # pragma: no cover - CPU-only machines
        logger.debug('No torch CUDA libraries to add', exc_info=True)

    _dlls_ready = True


def _resolve_device(device: str) -> str:
    if device != 'auto':
        return device
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return 'cuda'
    except Exception:  # pragma: no cover - probing only
        pass
    return 'cpu'


def get_model(
    model_name: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
):
    """Load (once) and return the faster-whisper model."""
    global _model, _model_key

    _ensure_cuda_libraries()
    resolved_device = _resolve_device(device)
    if resolved_device == 'cpu' and compute_type in ('float16', 'int8_float16'):
        compute_type = 'int8'

    key = f'{model_name}|{resolved_device}|{compute_type}'

    with _model_lock:
        if _model is not None and _model_key == key:
            return _model

        from faster_whisper import WhisperModel

        logger.info('Loading ASR model %s on %s (%s)',
                    model_name, resolved_device, compute_type)
        _model = WhisperModel(
            model_name, device=resolved_device, compute_type=compute_type
        )
        _model_key = key
        return _model


def transcribe_bytes(
    audio_bytes: bytes,
    model_name: str = DEFAULT_MODEL,
    beam_size: int = 5,
    **kwargs: Any,
) -> Transcript:
    """Transcribe raw MP3 bytes into a word-timestamped transcript."""
    handle, path = tempfile.mkstemp(suffix='.mp3')
    try:
        with os.fdopen(handle, 'wb') as f:
            f.write(audio_bytes)
        return transcribe_file(path, model_name=model_name,
                               beam_size=beam_size, **kwargs)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def transcribe_file(
    path: str,
    model_name: str = DEFAULT_MODEL,
    beam_size: int = 5,
    **kwargs: Any,
) -> Transcript:
    model = get_model(model_name)

    segments_iter, info = model.transcribe(
        path,
        language='en',
        beam_size=beam_size,
        word_timestamps=True,
        condition_on_previous_text=False,
        initial_prompt=INITIAL_PROMPT,
        vad_filter=True,
        vad_parameters={'min_silence_duration_ms': 300},
        **kwargs,
    )

    segments: List[Segment] = []
    for segment in segments_iter:
        words = [
            Word(
                start=float(word.start),
                end=float(word.end),
                word=word.word,
                probability=float(getattr(word, 'probability', 1.0) or 1.0),
            )
            for word in (segment.words or [])
            if word.start is not None and word.end is not None
        ]
        segments.append(
            Segment(
                start=float(segment.start),
                end=float(segment.end),
                text=segment.text.strip(),
                words=words,
            )
        )

    text = ' '.join(segment.text for segment in segments).strip()
    return Transcript(
        text=text,
        segments=segments,
        duration=float(info.duration),
        model=model_name,
    )


# --------------------------------------------------------------------------- #
# Cache — experiments re-run hundreds of times, transcription runs once
# --------------------------------------------------------------------------- #

def cache_path(key: str, model_name: str = DEFAULT_MODEL) -> Path:
    safe_model = model_name.replace('/', '_')
    return TRANSCRIPT_CACHE / safe_model / f'{key}.json'


def load_cached(key: str, model_name: str = DEFAULT_MODEL) -> Optional[Transcript]:
    path = cache_path(key, model_name)
    if not path.exists():
        return None
    with open(path, encoding='utf-8') as f:
        return Transcript.from_dict(json.load(f))


def save_cached(key: str, transcript: Transcript,
                model_name: str = DEFAULT_MODEL) -> None:
    path = cache_path(key, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(transcript.to_dict(), f, ensure_ascii=False, indent=1)


def audio_key(audio_bytes: bytes) -> str:
    return hashlib.sha1(audio_bytes).hexdigest()[:16]
