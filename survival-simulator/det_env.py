"""Make LOCAL runs reproducible by canonicalising the environment's iteration order.

`Environment._get_local_*` all return `set()`, so iteration order follows `id()`
and therefore memory layout. That order reaches physics through obstacle and
edge collision resolution, so two identically-seeded processes drift apart.
Canonicalising the policy's observations alone was not enough: seed 11 h400 went
from spread 26.1 to 0.3, but a residual branch still amplified to 213.6 by
t=3000. `trace_divergence.py` pinned it to a predator observation differing while
every agent state matched, i.e. the leak is environment-side.

Patching these six helpers to return ordered lists gives byte-identical runs.
Physics is unchanged; only the choice among equally-valid orderings is fixed.

MEASURED LIMIT: this does NOT reduce the variance that blocks A/B testing.
Paired differences on a deterministic env came back with sd 289-306, worse than
the unpaired noise floor, because the system is chaotic in the policy and there
is no stable per-seed effect to cancel. Use this for exact replay and debugging,
not as a statistical instrument.

The evaluation server runs the UNPATCHED environment.

    import det_env; det_env.install()          # natural ordering
    det_env.install(salt=3)                    # a different valid ordering
"""

import zlib

from src.elements.environment import Environment

_PATCHED = "unset"


def _key(obj):
    return (round(float(getattr(obj, "x", 0.0)), 9),
            round(float(getattr(obj, "y", 0.0)), 9),
            int(getattr(obj, "agent_id", -1) or -1),
            round(float(getattr(obj, "size", 0.0)), 9),
            type(obj).__name__)


def _edge_key(edge):
    try:
        (sx, sy), (ex, ey) = edge
        return (0, float(sx), float(sy), float(ex), float(ey), "")
    except (TypeError, ValueError):
        return (1, 0.0, 0.0, 0.0, 0.0, repr(edge))


def _salted(salt):
    # crc32, not hash(): str hashing is randomised per process, which is exactly
    # what this module exists to remove.
    def key(obj):
        raw = (f"{salt}|{float(getattr(obj, 'x', 0.0)):.9f}"
               f"|{float(getattr(obj, 'y', 0.0)):.9f}"
               f"|{int(getattr(obj, 'agent_id', -1) or -1)}"
               f"|{type(obj).__name__}")
        return zlib.crc32(raw.encode())
    return key


def _salted_edge(salt):
    def key(edge):
        return zlib.crc32(f"{salt}|{edge!r}".encode())
    return key


def install(salt=None):
    """Idempotent per salt; safe to call in every worker process."""
    global _PATCHED
    if _PATCHED == salt:
        return
    _PATCHED = salt

    key = _key if salt is None else _salted(salt)
    ekey = _edge_key if salt is None else _salted_edge(salt)

    originals = getattr(Environment, "_det_originals", None)
    if originals is None:
        originals = {name: getattr(Environment, name) for name in (
            "_get_local_objects", "_get_local_agents", "_get_local_predators",
            "_get_local_fruits", "_get_local_trees", "_get_local_obstacles",
            "_get_local_edges")}
        Environment._det_originals = originals

    def objects(self, creature, _o=originals["_get_local_objects"]):
        a, f, t, o, p, e = _o(self, creature)
        return (sorted(a, key=key), sorted(f, key=key), sorted(t, key=key),
                sorted(o, key=key), sorted(p, key=key), sorted(e, key=ekey))

    def simple(name, k):
        original = originals[name]

        def wrapper(self, creature, _o=original, _k=k):
            return sorted(_o(self, creature), key=_k)
        return wrapper

    Environment._get_local_objects = objects
    Environment._get_local_agents = simple("_get_local_agents", key)
    Environment._get_local_predators = simple("_get_local_predators", key)
    Environment._get_local_fruits = simple("_get_local_fruits", key)
    Environment._get_local_trees = simple("_get_local_trees", key)
    Environment._get_local_obstacles = simple("_get_local_obstacles", key)
    Environment._get_local_edges = simple("_get_local_edges", ekey)
