"""The approaches, in the order they were tried. See EXPERIMENTS.md."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from ..asr import Transcript

Span = Tuple[float, float]
Prediction = Tuple[bool, Optional[Span]]


class Approach(Protocol):
    name: str

    def predict(
        self, transcript: Transcript, questions: List[str]
    ) -> List[Prediction]:
        ...


def _lexical(**kwargs: Any):
    from .lexical import LexicalApproach

    return LexicalApproach(**kwargs)


def _lexical_rules(**kwargs: Any):
    from .lexical import LexicalApproach

    kwargs.setdefault('use_rules', True)
    return LexicalApproach(**kwargs)


def _always_yes(**kwargs: Any):
    from .baseline import AlwaysYesApproach

    return AlwaysYesApproach(**kwargs)


def _feature_ml(**kwargs: Any):
    from .feature_ml import FeatureMLApproach

    return FeatureMLApproach(**kwargs)


def _neural(**kwargs: Any):
    from .neural import NeuralApproach

    return NeuralApproach(**kwargs)


def _llm(**kwargs: Any):
    from .llm import LLMApproach

    return LLMApproach(**kwargs)


REGISTRY: Dict[str, Callable[..., Approach]] = {
    'always_yes': _always_yes,
    'lexical': _lexical,
    'lexical_rules': _lexical_rules,
    'feature_ml': _feature_ml,
    'neural': _neural,
    'llm': _llm,
}


def build(name: str, **kwargs: Any) -> Approach:
    if name not in REGISTRY:
        raise KeyError(f'unknown approach {name!r}; have {sorted(REGISTRY)}')
    return REGISTRY[name](**kwargs)
