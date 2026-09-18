"""Measures per-tick response time against the scoring budget.

The evaluation server allows 600 s of accumulated response time for a 30000
tick run, i.e. 20 ms per tick including network.  Run agent_server.py first.

    python latency_bench.py                         # loopback
    python latency_bench.py https://your-tunnel.example
"""

import statistics
import sys
import time

import requests

BUDGET_S = 600.0
TICKS = 30000


def build_payloads(seed=11, n=300):
    """Replay a real simulation so payload sizes are representative."""
    from src.core import SimulationCore
    from src.utils.controllers.expert_agent_policy import Hive

    sim, hive = SimulationCore(seed=seed), Hive()
    hive.reset()
    actions, out = [], []
    for _ in range(n):
        state = sim.step(actions)
        out.append({"game_status": "ok", "score": state["score"],
                    "sim_time": state["sim_time"], "n_agents": state["num_agents"],
                    "agent_status": state["observations"]})
        reqs = hive.act(state["observations"], state["sim_time"])
        actions = [(r.agent_id, r) for r in reqs]
    return out


def bench(url, payloads, session):
    post = session.post if session else requests.post
    times = []
    for p in payloads:
        t0 = time.perf_counter()
        r = post(f"{url}/predict", json=p, timeout=10)
        times.append(time.perf_counter() - t0)
        r.raise_for_status()
    return times


def report(label, times, payloads):
    mean = statistics.mean(times)
    size = statistics.mean(len(str(p)) for p in payloads) / 1024
    projected = mean * TICKS
    verdict = "OK" if projected < BUDGET_S else f"OVER BUDGET -> run cut at ~{BUDGET_S/mean:.0f} ticks (score ~{BUDGET_S/mean/10:.0f})"
    print(f"{label:22s} mean {mean*1000:7.2f} ms  p95 {sorted(times)[int(len(times)*.95)]*1000:7.2f} ms"
          f"  payload ~{size:.1f} KB  ->  {projected:6.0f} s of {BUDGET_S:.0f} s   {verdict}")


def phases(url, payload, rounds=25):
    """Split one new-connection request into DNS / TCP / TLS / exchange.

    The evaluator calls bare `requests.post()` with no Session, so it pays
    connection setup on EVERY tick.  Knowing how that 100 ms splits decides the
    fix: TLS cost is removed by dropping to http, TCP+TLS together are removed
    only by getting the client to reuse connections, and the exchange leg is
    the floor set by geography plus our handler.
    """
    import json as _json
    import socket
    import ssl
    from urllib.parse import urlparse

    u = urlparse(url)
    secure = u.scheme == "https"
    host = u.hostname
    port = u.port or (443 if secure else 80)
    body = _json.dumps(payload).encode()
    head = (f"POST {u.path or ''}/predict HTTP/1.1\r\nHost: {host}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
            f"Connection: close\r\n\r\n").encode()

    acc = {"dns": [], "tcp": [], "tls": [], "exchange": []}
    for _ in range(rounds):
        t0 = time.perf_counter()
        addr = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)[0][4]
        t1 = time.perf_counter()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect(addr)
        t2 = time.perf_counter()
        if secure:
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
        t3 = time.perf_counter()
        sock.sendall(head + body)
        if not sock.recv(65536):
            raise RuntimeError("empty response")
        t4 = time.perf_counter()
        sock.close()
        acc["dns"].append(t1 - t0)
        acc["tcp"].append(t2 - t1)
        acc["tls"].append(t3 - t2)
        acc["exchange"].append(t4 - t3)

    total = sum(statistics.median(v) for v in acc.values())
    print(f"\nper-tick breakdown for a fresh connection ({rounds} samples):")
    for name in ("dns", "tcp", "tls", "exchange"):
        ms = statistics.median(acc[name]) * 1000.0
        print(f"  {name:9s}{ms:8.2f} ms{ms / (total * 1000.0) * 100:7.0f}%")
    print(f"  {'TOTAL':9s}{total * 1000.0:8.2f} ms")
    setup = sum(statistics.median(acc[k]) for k in ("dns", "tcp", "tls"))
    print(f"\n  connection setup = {setup * 1000.0:.1f} ms of {total * 1000.0:.1f} ms "
          f"({setup / total:.0%}) — paid again every single tick")
    print(f"  score cap at this latency: ~{BUDGET_S / total / 10:.0f}")
    return acc


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:9052"
    payloads = build_payloads()
    bench(url, payloads[:20], None)                      # warm up
    report("new connection/tick", bench(url, payloads, None), payloads)
    with requests.Session() as s:
        report("keep-alive session", bench(url, payloads, s), payloads)
    phases(url, payloads[len(payloads) // 2])


if __name__ == "__main__":
    main()
