"""Experiment 0: the supplied baseline, reproduced offline."""

from __future__ import annotations

from typing import List, Optional, Tuple

from ..asr import Transcript


class AlwaysYesApproach:
    name = 'always_yes'

    def predict(
        self, transcript: Transcript, questions: List[str]
    ) -> List[Tuple[bool, Optional[Tuple[float, float]]]]:
        return [(True, None) for _ in questions]
