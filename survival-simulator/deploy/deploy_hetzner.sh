#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Deploy the agent endpoint to a Hetzner Helsinki (hel1) VPS.
# ---------------------------------------------------------------------------
# WHY hel1 SPECIFICALLY:
#   The evaluator calls us from 46.62.240.126
#   -> rDNS static.126.240.62.46.clients.your-server.de, RIPE DE-HETZNER, country FI
#   -> Hetzner Helsinki. Same-DC RTT is sub-ms.
#   The evaluator's client opens a FRESH CONNECTION EVERY TICK (bare
#   requests.post, no Session), so per-tick cost is ~2 round trips. Cutting the
#   round trip from ~28 ms to ~0.3 ms is the whole game:
#       cloudflared https   257 ms/tick -> score cap ~233
#       cloudflared http     57 ms/tick -> score cap ~1060
#       hel1 same-DC        ~10 ms/tick -> score cap ~6000 (max sim time is 3000,
#                                          so the wire stops binding entirely)
#   AWS eu-north-1 (Stockholm) is ~400 km away, so it leaves ~12 ms on the table.
#
# Usage:
#   ./deploy/deploy_hetzner.sh root@<server-ip>
#   ./deploy/deploy_hetzner.sh root@<server-ip> --restart-only
# ---------------------------------------------------------------------------
set -euo pipefail

TARGET="${1:?usage: $0 user@host [--restart-only]}"
MODE="${2:-full}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# A fresh box has no known_hosts entry, and the fingerprint prompt would block
# the whole unattended run.
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o BatchMode=yes)
ssh()  { command ssh  "${SSH_OPTS[@]}" "$@"; }
scp()  { command scp  "${SSH_OPTS[@]}" "$@"; }

if [ "$MODE" = "--restart-only" ]; then
  ssh "$TARGET" 'systemctl restart agent && sleep 2 && systemctl --no-pager -l status agent | head -12'
  exit 0
fi

echo "=== shipping code -> $TARGET ==="
# Only what the endpoint actually imports. No simulator, no pygame, no numpy.
ssh "$TARGET" 'mkdir -p /opt/agent/src/utils/controllers'
scp -q "$HERE/agent_server.py"                                   "$TARGET:/opt/agent/"
scp -q "$HERE/deploy/requirements-server.txt"                    "$TARGET:/opt/agent/"
scp -q "$HERE/deploy/agent.service"                              "$TARGET:/etc/systemd/system/agent.service"
scp -q "$HERE/src/utils/DTOs.py"                                 "$TARGET:/opt/agent/src/utils/"
scp -q "$HERE/src/utils/controllers/expert_agent_policy.py"      "$TARGET:/opt/agent/src/utils/controllers/"

echo "=== remote setup ==="
ssh "$TARGET" 'bash -s' <<'REMOTE'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

id agent >/dev/null 2>&1 || useradd --system --home /opt/agent --shell /usr/sbin/nologin agent

apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip ufw >/dev/null

# `src` is imported as a package, so every level needs an __init__.py.
touch /opt/agent/src/__init__.py \
      /opt/agent/src/utils/__init__.py \
      /opt/agent/src/utils/controllers/__init__.py

[ -d /opt/agent/.venv ] || python3 -m venv /opt/agent/.venv
/opt/agent/.venv/bin/pip install -q --upgrade pip
/opt/agent/.venv/bin/pip install -q -r /opt/agent/requirements-server.txt

chown -R agent:agent /opt/agent

# Only the endpoint is reachable. SSH stays open or you lock yourself out.
ufw allow 22/tcp  >/dev/null
ufw allow 80/tcp  >/dev/null
ufw --force enable >/dev/null

systemctl daemon-reload
systemctl enable  agent >/dev/null 2>&1
systemctl restart agent
sleep 2
systemctl --no-pager -l status agent | head -12
REMOTE

HOST="${TARGET#*@}"
echo
echo "=== colocation check: RTT from this VPS to the evaluator ==="
# The whole point of hel1 is being next to 46.62.240.126. Sub-ms means same
# facility; >5 ms means the server landed in the wrong location.
ssh "$TARGET" 'ping -c 5 -q 46.62.240.126 2>/dev/null | tail -2 || echo "  ping blocked; not fatal"'
echo
echo "=== smoke test ==="
curl -sS --max-time 10 "http://$HOST/" || echo "  (no response yet)"
echo
echo
echo "submit this URL:  http://$HOST/predict"
echo "check callers:    curl http://$HOST/callers"
echo "bench it:         python latency_bench.py http://$HOST"
echo "tail logs:        ssh $TARGET journalctl -u agent -f"
