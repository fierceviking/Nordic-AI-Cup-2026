"""End-to-end check of agent_server: drives a real simulation through HTTP.

    python agent_server.py          # in one terminal
    python server_smoketest.py      # in another
"""

import sys
import time

import requests

URL = "http://127.0.0.1:9052"


def main(seed=11, ticks=400):
    from src.core import SimulationCore

    print(requests.get(URL, timeout=10).json())

    sim = SimulationCore(seed=seed)
    actions = []
    t0 = time.time()
    latency = 0.0
    for _ in range(ticks):
        state = sim.step(actions)
        payload = {
            "game_status": "ok",
            "score": state["score"],
            "sim_time": state["sim_time"],
            "n_agents": state["num_agents"],
            "agent_status": state["observations"],
        }
        c0 = time.time()
        resp = requests.post(f"{URL}/predict", json=payload, timeout=10)
        latency += time.time() - c0
        resp.raise_for_status()
        acts = resp.json()["actions"]
        assert len(acts) == state["num_agents"], (len(acts), state["num_agents"])
        ids = {a["agent_id"] for a in acts}
        assert ids == {o["agent_id"] for o in state["observations"]}, "agent id mismatch"
        for a in acts:
            for key in ("move_distance", "move_direction", "turn_angle"):
                assert isinstance(a[key], (int, float)), (key, a)
        actions = [(a["agent_id"], _Req(a)) for a in acts]
        if state["num_agents"] == 0:
            print("all agents died")
            break

    print(f"ok: {ticks} ticks, score={sim.env.score:.1f}, agents={len(sim.env.agents)}, "
          f"mean latency {latency/ticks*1000:.1f} ms, wall {time.time()-t0:.0f}s")
    print(requests.get(URL, timeout=10).json())


class _Req:
    """Duck-types ActionRequest for SimulationCore.step."""

    __slots__ = ("move_distance", "move_direction", "turn_angle", "spawn_agent")

    def __init__(self, d):
        self.move_distance = d["move_distance"]
        self.move_direction = d["move_direction"]
        self.turn_angle = d["turn_angle"]
        self.spawn_agent = d["spawn_agent"]


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
