"""Answer ten yes/no questions about one recorded consultation.

The pipeline, end to end:

1. ``parakeet-tdt-0.6b-v2`` transcribes the MP3 with **word-level timestamps**
   via MLX on the Apple GPU (~0.018 x real time). Paid once per request and
   shared by all ten questions. It is a transducer, so the timings are frame
   alignments rather than attention DTW, which is why they can be trusted as
   span boundaries.
2. The transcript is cut into utterances at sentence ends and at pauses, and
   the candidate evidence spans are every run of up to three consecutive
   utterances.
3. Each question is rewritten as a declarative statement and matched against
   those candidates: IDF-weighted lexical coverage for *where*, a dense
   bi-encoder to widen the shortlist, and a DeBERTa NLI model for *whether*.
   Entailment is what separates a hard negative ("200 mg daily") from the
   positive it imitates ("100 mg daily"); lexical overlap cannot.
4. The span handed back is the single utterance carrying most of the matched
   words, corrected by three globally fitted scalars. Emitting the utterance
   beats shrinking to the matched words, because the annotation is a phrase and
   an utterance is the closest thing the transcript has to one.

Everything runs locally. See EXPERIMENTS.md for how each piece was chosen and
what it was worth, and RESEARCH.md for the literature behind it.
"""

import logging

from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from solution import pipeline
from utils import decode_audio

logger = logging.getLogger(__name__)

# Load the weights while uvicorn is still starting, rather than inside the
# first request, which has a 60 second budget to keep.
pipeline.warmup()


### CALL YOUR CUSTOM MODEL VIA THIS FUNCTION ###

def predict(request: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    """Answer every question about one conversation.

    ``pipeline.answer`` never raises: an exception here would cost all ten
    questions, so a failure degrades to a guess instead.
    """
    audio_bytes = decode_audio(request.audio_base64)

    logger.info(
        '%s (%.1f MB): %d questions',
        request.audio_filename,
        len(audio_bytes) / 1e6,
        len(request.questions),
    )

    predictions = pipeline.answer(audio_bytes, request.questions)

    return ASRQuestionResponseDto(
        answers=[bool(answer) for answer, _ in predictions],
        evidence_start=[span[0] if span else None for _, span in predictions],
        evidence_end=[span[1] if span else None for _, span in predictions],
    )
