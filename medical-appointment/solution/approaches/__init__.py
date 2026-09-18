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


def _quote(**kwargs: Any):
    from .llm import QuoteApproach

    return QuoteApproach(**kwargs)


def _quote_judge(**kwargs: Any):
    kwargs.setdefault('model_name', 'mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit')
    kwargs.setdefault('width_scale', 0.98)
    kwargs.setdefault('start_offset', 0.22)
    kwargs.setdefault('end_offset', 0.02)
    kwargs.setdefault('adjudicate', True)
    return _quote(**kwargs)


REGISTRY: Dict[str, Callable[..., Approach]] = {
    'always_yes': _always_yes,
    'lexical': _lexical,
    'lexical_rules': _lexical_rules,
    'feature_ml': _feature_ml,
    'neural': _neural,
    'llm': _llm,
    'quote': _quote,
    'quote_judge': _quote_judge,
}


def build(name: str, **kwargs: Any) -> Approach:
    if name not in REGISTRY:
        raise KeyError(f'unknown approach {name!r}; have {sorted(REGISTRY)}')
    return REGISTRY[name](**kwargs)
