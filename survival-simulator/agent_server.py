import asyncio
import hashlib
import json
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Body, Request, Response
from pydantic import BaseModel

from src.utils.controllers.expert_agent_policy import DT, Hive
from src.utils.controllers import expert_agent_policy

try:
    import orjson as _json

    def _loads(raw: bytes):
        return _json.loads(raw)

    def _dumps(obj) -> bytes:
        return _json.dumps(obj)
except ImportError:                          # orjson is a speed-up, not a need
    import json as _json

    def _loads(raw: bytes):
        return _json.loads(raw)

    def _dumps(obj) -> bytes:
        return _json.dumps(obj).encode()

HOST = "0.0.0.0"
PORT = 9052

# Scoring constraints from the README: the run ends once accumulated response
# time hits 600 s, and a full run is 30000 ticks.
BUDGET_S = 600.0
FULL_RUN_TICKS = 30000
IDLE_END_S = 15.0        # no request for this long means the run is over

logger = logging.getLogger("agent_server")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    logger.addHandler(_h)


class RunStats:
    """Per-run timing. Appending two floats per tick costs ~1 us."""

    __slots__ = ("index", "started", "handler", "gaps", "last_end",
                 "sim_time", "score", "agents", "peak_agents", "errors", "reported")

    def __init__(self, index: int) -> None:
        self.index = index
        self.started = time.perf_counter()
        self.handler: List[float] = []
        self.gaps: List[float] = []
        self.last_end = self.started
        self.sim_time = 0.0
        self.score = 0.0
        self.agents = 0
        self.peak_agents = 0
        self.errors = 0
        self.reported = False

    def summary(self) -> str:
        n = len(self.handler)
        if not n:
            return f"run {self.index}: no ticks"
        srv = sorted(self.handler)
        mean = sum(srv) / n
        p95 = srv[min(n - 1, int(n * 0.95))]
        # Gap = time from our response going out to the next request arriving:
        # network both ways plus the simulator's own step.
        gap_mean = sum(self.gaps) / len(self.gaps) if self.gaps else 0.0
        loop = (self.last_end - self.started) / n
        # The scored quantity is request->response as seen by their server; we
        # can only observe our own handler time plus the round trip in the gap.
        est_rtt = max(0.0, gap_mean)
        est_scored = mean + est_rtt
        projected = est_scored * FULL_RUN_TICKS
        budget_tick = BUDGET_S / max(est_scored, 1e-9)
        verdict = ("within budget" if projected <= BUDGET_S else
                   f"OVER BUDGET: cut off near tick {budget_tick:.0f} "
                   f"(score ~{budget_tick / 10:.0f})")
        return (
            f"run {self.index} finished: {n} ticks, sim_time {self.sim_time:.1f}s, "
            f"score {self.score:.1f}, agents {self.agents} (peak {self.peak_agents}), "
            f"errors {self.errors}\n"
            f"    server handler : mean {mean * 1000:.2f} ms | p95 {p95 * 1000:.2f} ms "
            f"| max {srv[-1] * 1000:.1f} ms | total {sum(srv):.1f} s\n"
            f"    gap to next req: mean {gap_mean * 1000:.2f} ms  "
            f"(network round trip + their sim step)\n"
            f"    loop period    : {loop * 1000:.2f} ms/tick over {self.last_end - self.started:.0f} s wall\n"
            f"    est. scored latency ~{est_scored * 1000:.1f} ms/tick -> "
            f"{projected:.0f} s of the {BUDGET_S:.0f} s budget: {verdict}"
        )


_RUN: Optional[RunStats] = None
_RUNS = 0
_FAILURES = 0
_LAST_ERROR = None
_POLICY_REVISION = hashlib.sha256(Path(expert_agent_policy.__file__).read_bytes()).hexdigest()[:12]

def _finish_run(reason: str) -> None:
    global _RUN
    run = _RUN
    if run is None or run.reported:
        return
    run.reported = True
    logger.info("[%s] %s", reason, run.summary())


async def _watchdog() -> None:
    while True:
        await asyncio.sleep(5.0)
        run = _RUN
        if run and not run.reported and time.perf_counter() - run.last_end > IDLE_END_S:
            _finish_run("idle")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    error_log = RotatingFileHandler(Path(__file__).with_name("agent_errors.log"),
                                   maxBytes=1_000_000, backupCount=2, encoding="utf-8")
    error_log.setLevel(logging.ERROR)
    error_log.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(error_log)
    task = asyncio.create_task(_watchdog())
    logger.info("agent server ready (budget %.0f s / %d ticks = %.1f ms per tick)",
                BUDGET_S, FULL_RUN_TICKS, BUDGET_S / FULL_RUN_TICKS * 1000)
    logger.info("policy %s settings=%s", _POLICY_REVISION, _policy_settings())
    try:
        yield
    finally:
        task.cancel()
        _finish_run("shutdown")
        logger.removeHandler(error_log)
        error_log.close()


app = FastAPI(title="Survival Simulator Agent Endpoint", lifespan=lifespan)

# One persistent hive: it carries the world map, per-agent dead reckoning and
# the population controller across ticks.  Hive.act() resets itself when
# sim_time jumps backwards, so the three back-to-back evaluation runs are
# handled without restarting the server.
def _load_program():
    """RULE_PROGRAM may be inline JSON or a path to a rule_search.json.

    Unset means the shipped Hive, so the default deploy path is unchanged.
    """
    raw = os.environ.get("RULE_PROGRAM", "").strip()
    if not raw:
        return None
    data = json.loads(raw) if raw.startswith("[") else json.loads(
        Path(raw).read_text(encoding="utf-8"))
    if isinstance(data, list) and data and isinstance(data[-1], dict):
        data = data[-1].get("final", data[-1]).get("program")
    if not data:
        return None

    from rule_policy import ACTIONS, CONDITIONS
    program = []
    for cond, theta, action in data:
        # Fail at startup rather than mid-evaluation on a typo.
        if cond not in CONDITIONS or action not in ACTIONS:
            raise ValueError(f"unknown rule ({cond}, {action})")
        program.append((str(cond), float(theta), str(action)))
    return program


_PROGRAM = _load_program()
if _PROGRAM:
    from rule_policy import RuleHive
    HIVE = RuleHive(program=_PROGRAM)
    # Without this a searched policy reports the same revision as the Hive.
    _POLICY_REVISION += "+" + hashlib.sha256(
        json.dumps(_PROGRAM, sort_keys=True).encode()).hexdigest()[:8]
else:
    HIVE = Hive()


def _policy_settings():
    cls = type(HIVE)
    return {name: getattr(HIVE, name)
            for name in dir(cls) if name.isupper() and name != "PROGRAM"}


class StepPayload(BaseModel):
    """Tolerant view of StepResponse.

    The /verify sample omits sim_time and n_agents, which the strict DTO
    rejects with 422 before the controller ever runs, so everything is optional
    here and observations stay raw dicts.
    """

    game_status: str = "ok"
    score: float = 0.0
    sim_time: Optional[float] = None
    n_agents: Optional[int] = None
    agent_status: List[Dict[str, Any]] = []


_AGENT_DEFAULTS = {
    "observations": [],
    "energy": 100.0,
    "biome": "forest",
    "age": 0.0,
    "speed": 10.0,
    "sprint_speed": 20.0,
    "hearing_radius": 50.0,
    "vision_angle": math.pi / 3,
    "vision_range": 200.0,
    "max_energy": 500.0,
}


def _normalize(state: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(state)
    for key, default in _AGENT_DEFAULTS.items():
        if out.get(key) is None:
            out[key] = default
    out["agent_id"] = int(out.get("agent_id") or 0)
    return out


def _as_dict(model):
    dump = getattr(model, "model_dump", None)
    return dump() if dump is not None else model.dict()


# Two round trips is the floor for a client that opens a fresh connection per
# tick, so the only remaining lever is being physically near the caller.
# Recording who calls us is how we find out which region to host in.
_CALLERS: Dict[str, dict] = {}


def _note_caller(request: Request) -> None:
    client = request.client
    ip = client.host if client else "?"
    # Through a tunnel the socket peer is the local cloudflared, so prefer the
    # forwarded chain, whose left-most entry is the original caller.
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        ip = fwd.split(",")[0].strip()
    rec = _CALLERS.get(ip)
    now = time.time()
    if rec is None:
        _CALLERS[ip] = {"first_seen": now, "last_seen": now, "hits": 1,
                        "path": request.url.path}
    else:
        rec["last_seen"] = now
        rec["hits"] += 1

@app.post("/predict")
async def predict(request: Request):
    """
    Receives the current simulation state and returns actions for all agents.
    """
    global _RUN, _RUNS, _FAILURES, _LAST_ERROR
    t_in = time.perf_counter()
    _note_caller(request)
    states = []
    run = None
    sim_time = None
    score = 0.0
    stage = "parse"
    try:
        step = _loads(await request.body())
        stage = "normalize"
        states = [_normalize(agent) for agent in (step.get("agent_status") or [])]
        stage = "clock"
        sim_time = step.get("sim_time")
        if sim_time is None:
            sim_time = 0.0 if HIVE.last_time < 0.0 else HIVE.last_time + DT
        sim_time = float(sim_time)
        if not math.isfinite(sim_time):
            raise ValueError("sim_time must be finite")
        stage = "score"
        score = float(step.get("score") or 0.0)
        if not math.isfinite(score):
            raise ValueError("score must be finite")

        stage = "run"
        run = _RUN
        if run is None or sim_time + 1e-9 < run.sim_time:
            _finish_run("reset")
            _RUNS += 1
            run = _RUN = RunStats(_RUNS)
            logger.info("run %d started (policy %s)", _RUNS, _POLICY_REVISION)
        else:
            run.gaps.append(t_in - run.last_end)

        stage = "policy"
        actions = HIVE.act_dicts(states, sim_time)
        stage = "serialize"
        body = _dumps({"actions": actions})
    except Exception as error:
        _FAILURES += 1
        if run is not None:
            run.errors += 1
        _LAST_ERROR = {"stage": stage, "type": type(error).__name__,
                       "message": str(error)[:500], "time_unix": time.time(),
                       "policy_revision": _POLICY_REVISION,
                       "run": run.index if run is not None else None}
        logger.exception("predict failed stage=%s run=%s sim_time=%s policy=%s",
                         stage, _LAST_ERROR["run"], sim_time, _POLICY_REVISION)
        if stage in ("policy", "serialize"):
            HIVE.reset()
        actions = [{"agent_id": s["agent_id"], "move_distance": 0.0,
                    "move_direction": 0.0, "turn_angle": 0.0,
                    "spawn_agent": False} for s in states]
        body = json.dumps({"actions": actions}).encode()

    if run is not None:
        run.last_end = time.perf_counter()
        run.handler.append(run.last_end - t_in)
        run.sim_time = sim_time
        run.score = score
        run.agents = len(states)
        if run.agents > run.peak_agents:
            run.peak_agents = run.agents

    # Must return {"actions": [...]} format
    return Response(content=body, media_type="application/json")

@app.get("/")
def index():
    return {"message": "Agent endpoint running!",
            "controller": type(HIVE).__module__ + "." + type(HIVE).__name__,
            "rule_program": _PROGRAM,
            "policy_revision": _POLICY_REVISION,
            "policy_settings": _policy_settings(),
            "sim_time": HIVE.last_time,
            "debug": HIVE.debug}


@app.get("/callers")
def callers():
    """Source IPs that have called us — geolocate these to pick a host region."""
    return {"callers": [{"ip": ip, **rec} for ip, rec in
                        sorted(_CALLERS.items(), key=lambda kv: -kv[1]["hits"])]}


@app.get("/stats")
def stats():
    """Live view of the current run; also printed automatically when it ends."""
    run = _RUN
    health = {"policy_revision": _POLICY_REVISION, "policy_settings": _policy_settings(),
              "failures_total": _FAILURES, "last_error": _LAST_ERROR}
    if run is None:
        return {**health, "runs": _RUNS, "message": "no run yet"}
    n = len(run.handler)
    return {
        **health,
        "run": run.index,
        "ticks": n,
        "sim_time": run.sim_time,
        "score": run.score,
        "agents": run.agents,
        "peak_agents": run.peak_agents,
        "errors": run.errors,
        "handler_ms_mean": round(sum(run.handler) / n * 1000, 3) if n else None,
        "handler_ms_max": round(max(run.handler) * 1000, 2) if n else None,
        "gap_ms_mean": round(sum(run.gaps) / len(run.gaps) * 1000, 2) if run.gaps else None,
        "budget_used_s": round(sum(run.handler) + sum(run.gaps), 1),
        "budget_s": BUDGET_S,
    }

if __name__ == "__main__":
    import uvicorn
    # access_log writes a line per tick (30000 per run) and costs more than the
    # controller does; timeout_keep_alive keeps a reused connection open.
    uvicorn.run(app, host=HOST, port=PORT, access_log=False,
                log_level="warning", timeout_keep_alive=65)