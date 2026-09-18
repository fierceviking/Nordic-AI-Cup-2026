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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

TRANSCRIPT_CACHE = Path(__file__).resolve().parent.parent / 'transcripts'

# Short name -> (backend, hugging face repo). CTranslate2 (faster-whisper) has no
# Metal backend, so on Apple Silicon everything runs through MLX instead.
BACKENDS: Dict[str, tuple] = {
    'parakeet': ('parakeet', 'mlx-community/parakeet-tdt-0.6b-v2'),
    'parakeet-v3': ('parakeet', 'mlx-community/parakeet-tdt-0.6b-v3'),
    'mlx-large-v3-turbo': ('mlx_whisper', 'mlx-community/whisper-large-v3-turbo'),
    'mlx-large-v3': ('mlx_whisper', 'mlx-community/whisper-large-v3-mlx'),
}

DEFAULT_MODEL = os.environ.get('ASR_MODEL', 'mlx-large-v3-turbo')

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

# MLX streams are thread-local: an array created on one thread cannot be
# evaluated on another, which surfaces as
# "RuntimeError: There is no Stream(cpu, 1) in current thread".
# FastAPI runs synchronous endpoints in a threadpool, so the request thread is
# not the thread that loaded the weights. Every MLX call therefore goes through
# this one worker, warmup included.
_MLX_THREAD = ThreadPoolExecutor(max_workers=1, thread_name_prefix='mlx')


def _on_mlx_thread(function, *args, **kwargs):
    return _MLX_THREAD.submit(function, *args, **kwargs).result()


def resolve(model_name: str) -> tuple:
    """Map a short model name onto its (backend, repo) pair."""
    if model_name in BACKENDS:
        return BACKENDS[model_name]
    if '/' in model_name:
        kind = 'parakeet' if 'parakeet' in model_name else 'mlx_whisper'
        return kind, model_name
    raise ValueError(
        f'unknown ASR model {model_name!r}; known: {sorted(BACKENDS)}'
    )


def _get_model(model_name: str = DEFAULT_MODEL):
    """Load (once) and return the ASR model. Must run on the MLX thread.

    Whisper is invoked functionally by ``mlx_whisper.transcribe``, so for that
    backend there is nothing to hold; the repo id is returned instead.
    """
    global _model, _model_key

    kind, repo = resolve(model_name)

    with _model_lock:
        if _model is not None and _model_key == model_name:
            return _model

        logger.info('Loading ASR model %s (%s, %s)', model_name, kind, repo)
        if kind == 'parakeet':
            from parakeet_mlx import from_pretrained

            _model = from_pretrained(repo)
        else:
            _model = repo

        _model_key = model_name
        return _model


def get_model(model_name: str = DEFAULT_MODEL):
    return _on_mlx_thread(_get_model, model_name)


def transcribe_bytes(
    audio_bytes: bytes,
    model_name: str = DEFAULT_MODEL,
    **kwargs: Any,
) -> Transcript:
    """Transcribe raw MP3 bytes into a word-timestamped transcript."""
    handle, path = tempfile.mkstemp(suffix='.mp3')
    try:
        with os.fdopen(handle, 'wb') as f:
            f.write(audio_bytes)
        return transcribe_file(path, model_name=model_name, **kwargs)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def transcribe_file(
    path: str,
    model_name: str = DEFAULT_MODEL,
    **kwargs: Any,
) -> Transcript:
    kind, _ = resolve(model_name)
    worker = (
        _transcribe_parakeet if kind == 'parakeet' else _transcribe_mlx_whisper
    )
    # Never call _on_mlx_thread from inside the worker: one thread, so waiting
    # on itself would deadlock. The workers use _get_model directly.
    return _on_mlx_thread(worker, path, model_name, **kwargs)


# --------------------------------------------------------------------------- #
# Parakeet (TDT). Timestamps come from transducer frame alignment, so they are
# monotonic and do not suffer Whisper's habit of placing punctuation seconds
# after the speech it follows.
# --------------------------------------------------------------------------- #

def _transcribe_parakeet(
    path: str, model_name: str = DEFAULT_MODEL, **kwargs: Any
) -> Transcript:
    model = _get_model(model_name)
    result = model.transcribe(path, **kwargs)

    segments: List[Segment] = []
    for sentence in result.sentences:
        words = _tokens_to_words(sentence.tokens)
        if not words:
            continue
        segments.append(
            Segment(
                start=float(sentence.start),
                end=float(sentence.end),
                text=sentence.text.strip(),
                words=words,
            )
        )

    return Transcript(
        text=result.text.strip(),
        segments=segments,
        duration=_duration(path, segments),
        model=model_name,
    )


def _tokens_to_words(tokens: List[Any]) -> List[Word]:
    """Merge Parakeet's sub-word tokens into whole words.

    A token that starts with a space opens a new word; everything else —
    continuations and trailing punctuation — attaches to the current one. The
    leading space is kept because ``windows._unit`` rebuilds utterance text by
    concatenating ``word.word`` directly.
    """
    words: List[Word] = []

    for token in tokens:
        text = token.text
        if not text:
            continue

        if words and not text.startswith(' '):
            current = words[-1]
            current.word += text
            current.end = float(token.end)
            current.probability = min(
                current.probability, float(token.confidence)
            )
            continue

        words.append(
            Word(
                start=float(token.start),
                end=float(token.end),
                word=text,
                probability=float(token.confidence),
            )
        )

    return [word for word in words if word.word.strip()]


# --------------------------------------------------------------------------- #
# Whisper via MLX. Word timings are DTW over cross-attention, kept as a fallback
# and as the closest thing to the original CUDA transcripts.
# --------------------------------------------------------------------------- #

def _transcribe_mlx_whisper(
    path: str, model_name: str = DEFAULT_MODEL, **kwargs: Any
) -> Transcript:
    import mlx_whisper

    _, repo = resolve(model_name)
    options = {
        'language': 'en',
        'word_timestamps': True,
        'condition_on_previous_text': False,
        'initial_prompt': INITIAL_PROMPT,
    }
    options.update(kwargs)

    result = mlx_whisper.transcribe(path, path_or_hf_repo=repo, **options)

    segments: List[Segment] = []
    for segment in result.get('segments', []):
        words = [
            Word(
                start=float(word['start']),
                end=float(word['end']),
                word=word['word'],
                probability=float(word.get('probability', 1.0) or 1.0),
            )
            for word in segment.get('words', [])
            if word.get('start') is not None and word.get('end') is not None
        ]
        segments.append(
            Segment(
                start=float(segment['start']),
                end=float(segment['end']),
                text=segment['text'].strip(),
                words=words,
            )
        )

    return Transcript(
        text=result.get('text', '').strip(),
        segments=segments,
        duration=_duration(path, segments),
        model=model_name,
    )


def _duration(path: str, segments: List[Segment]) -> float:
    """True media duration, falling back to the last word that was heard."""
    try:
        from utils import audio_duration_seconds

        with open(path, 'rb') as f:
            measured = audio_duration_seconds(f.read())
        if measured:
            return float(measured)
    except Exception:
        logger.debug('Could not measure audio duration', exc_info=True)

    return float(segments[-1].end) if segments else 0.0


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
