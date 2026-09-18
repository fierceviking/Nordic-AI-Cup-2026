import os, sys, json
sys.path.insert(0, ".")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
from src.core import SimulationCore
seed = int(sys.argv[1]); horizon = int(sys.argv[2])
sim = SimulationCore(seed=seed); env = sim.env
actions = []
while True:
    state = sim.step(actions)
    actions = []
    if state["num_agents"] == 0 or env.time > horizon:
        break
print(json.dumps({"score": round(state["score"], 6), "time": round(env.time, 3)}))
