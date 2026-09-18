import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import det_env

det_env.install(salt=0)

from src.core import SimulationCore
from src.utils.controllers.expert_agent_policy import Hive


def run(weight, seed=11):
    sim = SimulationCore(seed=seed)
    hive = Hive()
    hive.FLEE_TO_FOOD = weight
    hive.reset()
    env = sim.env
    actions = []
    flee_ticks = 0
    while True:
        state = sim.step(actions)
        reqs = hive.act(state["observations"], state["sim_time"])
        flee_ticks += sum(1 for m in hive.minds.values() if m.mode == "flee")
        actions = [(r.agent_id, r) for r in reqs]
        if state["num_agents"] == 0 or env.time > 3000:
            break
    return state["score"], flee_ticks


for w in (0.0, 0.5, 1.0):
    score, flee = run(w)
    print("FLEE_TO_FOOD=%.1f  score %8.1f  flee_ticks %6d" % (w, score, flee))
