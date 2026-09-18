"""Replays the /verify sample payload against the local endpoint.

The verification service posts a sample with no sim_time and no n_agents, and
with lowercase observation types.  Run agent_server.py first.
"""

import json
import sys
import unittest
from unittest.mock import patch

import requests

URL = "http://127.0.0.1:9052/predict"

SAMPLE = {
    "game_status": "running",
    "score": 123.4,
    "agent_status": [
        {
            "agent_id": 1,
            "observations": [
                {"type": "tree", "distance": 12.5, "angle": 1.57},
                {"type": "predator", "distance": 30.0, "angle": -1.57, "rel_dir": -2.57},
                {"type": "edge", "coords": [[50.0, 50.0], [100.0, 100.0]]},
            ],
            "energy": 85.0,
            "biome": "forest",
            "age": 5.2,
            "speed": 12.5,
            "sprint_speed": 13.5,
            "hearing_radius": 10.0,
            "vision_angle": 1.57,
            "vision_range": 50.0,
            "max_energy": 500.0,
        }
    ],
}


def main():
    for label, payload in (("verify sample", SAMPLE),
                           ("empty agent list", {"game_status": "ok", "score": 0.0,
                                                 "agent_status": []})):
        r = requests.post(URL, json=payload, timeout=10)
        print(f"{label}: HTTP {r.status_code} -> {json.dumps(r.json())[:300]}")
        r.raise_for_status()
        acts = r.json()["actions"]
        assert len(acts) == len(payload["agent_status"]), acts
        for a in acts:
            assert set(a) == {"agent_id", "move_distance", "move_direction",
                              "turn_angle", "spawn_agent"}, a
    print("OK: endpoint accepts the verification payload")


class HandlerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        import agent_server
        self.server = agent_server
        self.server.HIVE.reset()
        self.server._RUN = None
        self.server._RUNS = 0
        self.server._FAILURES = 0
        self.server._LAST_ERROR = None

    async def predict(self, payload):
        from starlette.requests import Request
        raw = json.dumps(payload).encode()

        async def receive():
            return {"type": "http.request", "body": raw, "more_body": False}

        request = Request({"type": "http", "method": "POST", "path": "/predict",
                           "headers": []}, receive)
        response = await self.server.predict(request)
        self.assertEqual(response.status_code, 200)
        return json.loads(response.body)

    async def test_verify_payload_and_clock_reset(self):
        for payload in (SAMPLE, dict(SAMPLE, sim_time=10.0), dict(SAMPLE, sim_time=0.0)):
            result = await self.predict(payload)
            self.assertEqual([action["agent_id"] for action in result["actions"]], [1])
        self.assertEqual(self.server._FAILURES, 0)
        self.assertEqual(self.server.HIVE.last_time, 0.0)

    async def test_normalization_failure_is_reported(self):
        with self.assertLogs("agent_server", level="ERROR"):
            result = await self.predict({"agent_status": [None]})
        self.assertEqual(result, {"actions": []})
        self.assertEqual(self.server.stats()["last_error"]["stage"], "normalize")
        self.assertEqual(self.server.stats()["failures_total"], 1)

    async def test_bad_clock_or_score_returns_known_agent_fallback(self):
        for field, value, stage in (("sim_time", "invalid", "clock"),
                                    ("sim_time", "nan", "clock"),
                                    ("score", "invalid", "score")):
            with self.subTest(field=field, value=value):
                with self.assertLogs("agent_server", level="ERROR"):
                    result = await self.predict(dict(SAMPLE, **{field: value}))
                self.assertEqual(result["actions"][0]["agent_id"], 1)
                self.assertEqual(result["actions"][0]["move_distance"], 0.0)
                self.assertEqual(self.server.stats()["last_error"]["stage"], stage)
                self.assertEqual(self.server.HIVE.last_time, -1.0)

    async def test_policy_exception_resets_partial_state(self):
        self.server.HIVE.pending_spawns = [99]
        with patch.object(self.server.HIVE, "act_dicts", side_effect=ValueError("injected")):
            with self.assertLogs("agent_server", level="ERROR"):
                result = await self.predict(SAMPLE)
        self.assertFalse(result["actions"][0]["spawn_agent"])
        self.assertEqual(self.server.HIVE.pending_spawns, [])
        self.assertEqual(self.server.stats()["errors"], 1)
        self.assertEqual(self.server.stats()["last_error"]["stage"], "policy")
        await self.predict(dict(SAMPLE, sim_time=1.0))
        self.assertEqual(self.server.stats()["failures_total"], 1)

    async def test_serialization_exception_uses_independent_fallback(self):
        with patch.object(self.server, "_dumps", side_effect=TypeError("injected")):
            with self.assertLogs("agent_server", level="ERROR"):
                result = await self.predict(SAMPLE)
        self.assertEqual(result["actions"][0]["agent_id"], 1)
        self.assertEqual(self.server.stats()["last_error"]["stage"], "serialize")
        self.assertEqual(self.server.stats()["errors"], 1)
        self.assertEqual(self.server.HIVE.last_time, -1.0)

    async def test_health_identifies_loaded_policy(self):
        health = self.server.stats()
        self.assertEqual(len(health["policy_revision"]), 12)
        self.assertTrue(health["policy_settings"]["PRIORITIZE_FOOD"])
        self.assertFalse(health["policy_settings"]["FOOD_AWARE_BREEDING"])


if __name__ == "__main__":
    if "--offline" in sys.argv:
        sys.argv.remove("--offline")
        unittest.main()
    else:
        main()
