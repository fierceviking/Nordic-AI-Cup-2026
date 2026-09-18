#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Nordic AI Cup - survival-simulator  ::  VALIDATION-ONLY helper
# ---------------------------------------------------------------------------
# Validate as many times as you like. It NEVER evaluates: the one-shot
# `evaluate` endpoint is intentionally not wired here, so this script cannot
# spend the single evaluation attempt by accident.
#
# Reads $token and $predict from ./secrets.ps1 (the single source of truth you
# already keep up to date as the cloudflared URL changes). Override with env
# vars NORDIC_API_KEY / SURVIVAL_PREDICT_URL if you prefer.
#
# WHY THIS SCRIPT FORCES http:// :
#   The run ends when accumulated wait time reaches 600 s, so
#       max score ~= (600000 ms / per_tick_ms) / 10
#   and the evaluator's client is a bare `requests.post()` with no Session, so
#   it pays full connection setup on EVERY tick. Measured through a cloudflared
#   quick tunnel, the TLS handshake alone costs ~176 ms of a ~235 ms tick while
#   TCP connect is only ~11 ms. Dropping to http removes it outright:
#       https  257 ms/tick -> score cap ~233
#       http    57 ms/tick -> score cap ~1060
#   Submitting an https URL silently throws away ~830 points.
#
# Usage:
#   ./validate.sh status              # team/use-case status (safe GET)
#   ./validate.sh bench               # measure tick latency + projected cap
#   ./validate.sh verify              # one-step reachability check of your URL
#   ./validate.sh validate            # queue a validation attempt, then watch
#   ./validate.sh poll <uuid>         # re-check a queued attempt
# ---------------------------------------------------------------------------
set -euo pipefail

SLUG="${SURVIVAL_SLUG:-survival-simulator}"
BASE="https://cases.nordicaicup.com/api/v1/usecases/$SLUG"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECRETS="$HERE/secrets.ps1"
PY="${SURVIVAL_PY:-$HERE/../.venv/Scripts/python.exe}"
[ -x "$PY" ] || PY="$(command -v python3 || command -v python)"

# A tick budget above this means the wire, not the policy, is the constraint.
# 60 ms/tick projects to a cap near 1000; the local policy means ~740.
MAX_TICK_MS="${SURVIVAL_MAX_TICK_MS:-70}"

# Pull a PowerShell `$name = "value"` out of secrets.ps1.
ps_var() {
  [ -f "$SECRETS" ] || return 0
  grep -iE "^[[:space:]]*\\\$$1[[:space:]]*=" "$SECRETS" | head -1 \
    | sed -E 's/^[^"]*"([^"]*)".*$/\1/'
}

TOKEN="${NORDIC_API_KEY:-$(ps_var token)}"
URL="${SURVIVAL_PREDICT_URL:-$(ps_var predict)}"

if [ -z "${TOKEN:-}" ]; then
  echo "ERROR: no API token (set \$token in secrets.ps1 or NORDIC_API_KEY)." >&2
  exit 2
fi
if [ -z "${URL:-}" ]; then
  echo "ERROR: no predict URL (set \$predict in secrets.ps1 or SURVIVAL_PREDICT_URL)." >&2
  exit 2
fi

# --- the http:// gate ------------------------------------------------------
# Downgrade unless explicitly overridden. Set SURVIVAL_ALLOW_HTTPS=1 only if the
# submission form rejects the http scheme.
if [ "${SURVIVAL_ALLOW_HTTPS:-0}" != "1" ]; then
  case "$URL" in
    https://*)
      DOWNGRADED="http://${URL#https://}"
      echo "NOTE: rewriting https -> http (worth ~+830 points; see header)."
      echo "      was: $URL"
      echo "      now: $DOWNGRADED"
      echo "      set SURVIVAL_ALLOW_HTTPS=1 to keep https."
      URL="$DOWNGRADED"
      ;;
  esac
fi

# latency_bench.py appends /predict itself, so hand it the bare origin.
BENCH_URL="${URL%/predict}"
BENCH_URL="${BENCH_URL%/}"

pp() { "$PY" -m json.tool 2>/dev/null || cat; }

api_get()  { curl -sS -H "x-token: $TOKEN" "$BASE/$1"; }
api_post() { curl -sS -H "x-token: $TOKEN" -H "Content-Type: application/json" \
                  -X POST -d "{\"url\":\"$URL\"}" "$BASE/$1"; }

get_uuid() {
  "$PY" -c 'import sys,json
d=json.load(sys.stdin)
print(d.get("queued_attempt_uuid") or d.get("uuid") or "")' 2>/dev/null || true
}

# Measure what the evaluator will actually pay per tick and project the cap.
# The quick tunnel drifts a lot (exchange leg measured 37-65 ms minutes apart),
# so sample twice and judge on the better one rather than on one unlucky draw.
# Returns non-zero if the wire is slow enough to cap us below the local policy.
bench() {
  echo "=== tick latency -> $BENCH_URL ==="
  local out best=""
  for attempt in 1 2; do
    out="$("$PY" "$HERE/latency_bench.py" "$BENCH_URL" 2>&1 \
           | tr -d '\000' | grep -av '^pygame\|^Hello from' || true)"
    printf '%s\n' "$out"
    best="$best
$out"
  done
  printf '%s' "$best" | MAX_TICK_MS="$MAX_TICK_MS" BENCH_URL="$BENCH_URL" "$PY" -c '
import os, re, sys
text = sys.stdin.read()
samples = [float(m) for m in re.findall(r"new connection/tick\s+mean\s+([0-9.]+) ms", text)]
if not samples:
    print("\n  could not parse bench output; is agent_server.py running?")
    sys.exit(1)
ms = min(samples)
limit = float(os.environ.get("MAX_TICK_MS", "70"))
url = os.environ.get("BENCH_URL", "")
tunnel = any(s in url for s in ("trycloudflare", "ngrok", "loca.lt", "localhost", "127.0.0.1"))
print("\n  samples  : " + ", ".join(f"{s:.1f}" for s in samples) + " ms")
print(f"  best tick: {ms:.1f} ms  (measured from THIS machine)")
if tunnel:
    # Serving through a tunnel from here means the evaluator pays what we pay.
    print(f"  score cap: ~{600000.0 / ms / 10.0:.0f}   (local policy means ~740)")
    if ms > limit:
        print(f"  VERDICT  : TOO SLOW (> {limit:.0f} ms). The wire caps the score.")
        sys.exit(1)
    print(f"  VERDICT  : OK (<= {limit:.0f} ms).")
else:
    # Direct endpoint: this is OUR round trip, and says nothing about theirs.
    print("  score cap: NOT MEASURABLE FROM HERE.")
    print("             The evaluator sits beside the server in Helsinki, so its")
    print("             round trip is far shorter than ours. This figure is only")
    print("             an upper bound and a reachability check.")
    print("  VERDICT  : reachable. Read real per-tick cost from a validation:")
    print("             wall_seconds / ticks, minus their ~3.8 ms sim step.")
'
}

# Pre-flight before every queue: confirm the key maps to the right team, that
# the service can reach and call the endpoint, and that the wire is fast enough
# to be worth spending a queue slot on.
preflight() {
  # A tunnel from this laptop caps the score at ~233 (https) or ~1060 (http),
  # and dies with the machine. The colocated Helsinki server is the only thing
  # worth spending a queue slot on, so refuse anything else outright.
  case "$BENCH_URL" in
    *trycloudflare.com*|*ngrok*|*loca.lt*|*localhost*|*127.0.0.1*)
      echo "REFUSING TO QUEUE: '$URL' is a tunnel or loopback address." >&2
      echo "  Queue only against the Helsinki server (http://89.167.18.234/predict)." >&2
      echo "  A tunnel run ended at 633 s wall for a 600 s budget and scored 558;" >&2
      echo "  the colocated server finished in 122 s and scored 846." >&2
      echo "  Override with SURVIVAL_ALLOW_TUNNEL=1 only if you really mean it." >&2
      [ "${SURVIVAL_ALLOW_TUNNEL:-0}" = "1" ] || return 1
      echo "  (SURVIVAL_ALLOW_TUNNEL=1 set; continuing against the tunnel)" >&2
      ;;
  esac

  echo "=== status ==="
  local st
  st="$(api_get status)"
  printf '%s' "$st" | "$PY" -c 'import sys,json
d=json.load(sys.stdin)
print("team:", d.get("team_name"), "| validations:", d.get("n_validations"),
      "| evaluations:", d.get("n_evaluations"))' 2>/dev/null || printf '%s' "$st" | pp

  echo "=== verify -> $URL ==="
  local vr
  vr="$(api_post verify)"; printf '%s' "$vr" | pp
  printf '%s' "$vr" | "$PY" -c 'import sys,json
d=json.load(sys.stdin)
ok = bool(d.get("can_connect")) and bool(d.get("can_predict"))
sys.exit(0 if ok else 1)' 2>/dev/null || return 1

  if ! MAX_TICK_MS="$MAX_TICK_MS" bench; then
    # The bench times OUR hop to the endpoint. That equalled the evaluator's
    # cost only while we served through a tunnel from this machine. Once the
    # endpoint is colocated with the evaluator in Helsinki, our reading is an
    # upper bound on theirs and must not block a queue slot.
    case "$BENCH_URL" in
      *trycloudflare.com*|*ngrok*|*localhost*|*127.0.0.1*)
        return 1 ;;
      *)
        echo "  NOTE: endpoint is not a tunnel from this machine, so the figure"
        echo "        above is OUR round trip, not the evaluator's. Continuing."
        ;;
    esac
  fi
}

poll() {
  local uuid="$1" tries="${2:-60}" i=0
  while [ "$i" -lt "$tries" ]; do
    i=$((i+1))
    echo "=== poll $i/$tries (uuid $uuid) ==="
    api_get "validate/queue/$uuid" | pp || true
    local code
    code="$(curl -sS -o /tmp/survival_attempt.json -w '%{http_code}' \
                 -H "x-token: $TOKEN" "$BASE/validate/queue/$uuid/attempt" || echo 000)"
    if [ "$code" = "200" ]; then
      echo "=== attempt finished ==="; pp < /tmp/survival_attempt.json; return 0
    fi
    echo "attempt not ready (HTTP $code); sleeping 15s..."
    sleep 15
  done
  echo "gave up after $tries polls; re-check with: $0 poll $uuid"
}

cmd="${1:-status}"
case "$cmd" in
  status) api_get status | pp ;;
  bench)  MAX_TICK_MS="$MAX_TICK_MS" bench ;;
  verify) echo "verify -> $URL"; api_post verify | pp ;;
  validate)
    if ! preflight; then
      echo "PRE-FLIGHT FAILED. NOT queueing." >&2
      echo "  - can_connect/can_predict must both be true" >&2
      echo "  - tick latency must be <= ${MAX_TICK_MS} ms (override: SURVIVAL_IGNORE_LATENCY=1)" >&2
      exit 3
    fi
    echo "=== queueing validation for -> $URL ==="
    resp="$(api_post validate/queue)"; echo "$resp" | pp
    uuid="$(printf '%s' "$resp" | get_uuid)"
    if [ -n "$uuid" ]; then echo "queued uuid: $uuid"; poll "$uuid";
    else echo "no uuid returned; not polling."; fi ;;
  poll) poll "${2:?usage: $0 poll <uuid>}" ;;
  *) echo "usage: $0 {status|bench|verify|validate|poll <uuid>}" >&2; exit 1 ;;
esac
