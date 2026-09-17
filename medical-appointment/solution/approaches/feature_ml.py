"""Experiment 3: a small classifier over lexical features.

The classifier only decides yes/no. The span comes from the same retrieval
front end the rule system uses, so the evidence half is unchanged and the two
experiments are comparable on the answer half alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from ..asr import Transcript
from ..features import FeatureExtractor
from ..windows import tighten

MODEL_PATH = Path(__file__).resolve().parents[2] / 'models' / 'feature_ml.joblib'

Span = Tuple[float, float]


class FeatureMLApproach:
    name = 'feature_ml'

    def __init__(
        self,
        model_path: str = str(MODEL_PATH),
        max_units: int = 3,
        length_penalty: float = 0.0,
        span_pad: float = 0.3,
        min_span: float = 2.0,
        max_span: float = 6.0,
        spans_for_no: bool = True,
    ) -> None:
        import joblib

        bundle = joblib.load(model_path)
        self.model = bundle['model']
        self.extractor = FeatureExtractor(
            max_units=max_units, length_penalty=length_penalty
        )
        self.span_pad = span_pad
        self.min_span = min_span
        self.max_span = max_span
        self.spans_for_no = spans_for_no

    def predict(
        self, transcript: Transcript, questions: List[str]
    ) -> List[Tuple[bool, Optional[Span]]]:
        prepared = self.extractor.prepare(transcript)

        rows = []
        spans: List[Optional[Span]] = []
        for question in questions:
            features, best, _ = self.extractor.features_for(
                question, transcript, prepared
            )
            rows.append(features)
            spans.append(
                tighten(best.window, best.matched, pad=self.span_pad,
                        min_duration=self.min_span, max_duration=self.max_span)
                if best else None
            )

        answers = self.model.predict(rows)
        return [
            (bool(answer), span if (answer or self.spans_for_no) else None)
            for answer, span in zip(answers, spans)
        ]
