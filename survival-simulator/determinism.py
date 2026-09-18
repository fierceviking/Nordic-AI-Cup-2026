"""Make the simulator reproducible, for search only.

`Environment._get_local_objects` builds observations out of `set()`s holding
entity objects.  Those hash by `id()`, i.e. memory address, so iteration order
changes between processes: the same config on the same seed measured 1147.5 and
676.8.  Observation order feeds straight into the policy's nearest-marker
matching and tie-breaks, and into which of two touching fruit gets eaten first,
so the run diverges.

Giving every entity a stable creation-ordered hash makes set iteration
deterministic, which turns "run both configs and hope n is large enough" into a
paired comparison on identical worlds.  That is what lets a local search accept
or reject a single neighbour move.

    from determinism import make_deterministic
    make_deterministic()        # call BEFORE creating SimulationCore

This is a HARNESS-ONLY patch.  It changes tie-break order, so anything it finds
must be re-validated on the unpatched simulator before shipping.
"""

import itertools

_counter = itertools.count(1)
_patched = False


def _stable_hash(self):
    # assigned lazily so we do not depend on every subclass calling super().__init__
    sid = getattr(self, "_stable_id", None)
    if sid is None:
        sid = next(_counter)
        object.__setattr__(self, "_stable_id", sid)
    return sid


def _stable_eq(self, other):
    return self is other


def make_deterministic() -> list:
    """Patch entity classes for stable set ordering.  Returns the names patched."""
    global _patched
    from src.elements.agent import Agent
    from src.elements.fruit import Fruit
    from src.elements.tree import Tree
    from src.elements.predator import Predator
    from src.elements.obstacle import Obstacle

    targets = [Agent, Fruit, Tree, Predator, Obstacle]
    for cls in targets:
        cls.__hash__ = _stable_hash
        cls.__eq__ = _stable_eq
    _patched = True
    return [cls.__name__ for cls in targets]


def is_patched() -> bool:
    return _patched
