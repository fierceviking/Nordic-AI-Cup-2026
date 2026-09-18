#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Nordic AI Cup - drone-flyby  ::  VALIDATION-ONLY helper
# ---------------------------------------------------------------------------
# Validate as many times as you like. It NEVER evaluates: the one-shot
# `evaluate` endpoint is intentionally not wired here, so this script cannot
# spend the single evaluation attempt by accident.
#
# Reads $token and $predict from ./secrets.ps1 (the single source of truth you
# already keep up to date as the cloudflared URL changes). Override with env
# vars NORDIC_API_KEY / DRONE_PREDICT_URL if you prefer.
#
# Usage:
#   ./validate.sh status              # team/use-case status (safe GET)
#   ./validate.sh verify              # one-frame reachability check of your URL
#   ./validate.sh validate            # queue a validation attempt, then watch
#   ./validate.sh poll <uuid>         # re-check a queued attempt
# ---------------------------------------------------------------------------
set -euo pipefail

BASE='https://cases.nordicaicup.com/api/v1/usecases/drone-flyby'
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECRETS="$HERE/secrets.ps1"
PY="${DF_PY:-$HOME/miniconda3/envs/dm/python.exe}"

# Pull a PowerShell `$name = "value"` out of secrets.ps1.
ps_var() {
  [ -f "$SECRETS" ] || return 0
  grep -iE "^[[:space:]]*\\\$$1[[:space:]]*=" "$SECRETS" | head -1 \
    | sed -E 's/^[^"]*"([^"]*)".*$/\1/'
}

TOKEN="${NORDIC_API_KEY:-$(ps_var token)}"
URL="${DRONE_PREDICT_URL:-$(ps_var predict)}"

if [ -z "${TOKEN:-}" ]; then
  echo "ERROR: no API token (set \$token in secrets.ps1 or NORDIC_API_KEY)." >&2
  exit 2
fi

# Pretty-print JSON if python is available, else pass through untouched.
pp() { "$PY" -m json.tool 2>/dev/null || cat; }

# GET/POST helpers. -s silent, -S show errors, never echo the token.
api_get()  { curl -sS -H "x-token: $TOKEN" "$BASE/$1"; }
api_post() { curl -sS -H "x-token: $TOKEN" -H "Content-Type: application/json" \
                  -X POST -d "{\"url\":\"$URL\"}" "$BASE/$1"; }

get_uuid() {
  "$PY" -c 'import sys,json;
d=json.load(sys.stdin)
print(d.get("queued_attempt_uuid") or d.get("uuid") or "")' 2>/dev/null || true
}

# Pre-flight the user requires before every queue: confirm the key maps to the
# right team (7.1 status) and that the eval service can reach + call /predict
# (7.2 verify -> can_connect / can_predict). Aborts the queue if either fails.
preflight() {
  echo "=== 7.1 status ==="
  local st
  st="$(api_get status)"
  printf '%s' "$st" | "$PY" -c 'import sys,json
d=json.load(sys.stdin)
print("team:", d.get("team_name"), "| validations:", d.get("n_validations"),
      "| evaluations:", d.get("n_evaluations"))' 2>/dev/null || printf '%s' "$st" | pp

  echo "=== 7.2 verify -> $URL ==="
  local vr
  vr="$(api_post verify)"; printf '%s' "$vr" | pp
  printf '%s' "$vr" | "$PY" -c 'import sys,json
d=json.load(sys.stdin)
ok = bool(d.get("can_connect")) and bool(d.get("can_predict"))
sys.exit(0 if ok else 1)' 2>/dev/null
}

poll() {
  local uuid="$1" tries="${2:-60}" i=0
  while [ "$i" -lt "$tries" ]; do
    i=$((i+1))
    echo "=== poll $i/$tries (uuid $uuid) ==="
    api_get "validate/queue/$uuid" | pp || true
    local code
    code="$(curl -sS -o /tmp/df_attempt.json -w '%{http_code}' \
                 -H "x-token: $TOKEN" "$BASE/validate/queue/$uuid/attempt" || echo 000)"
    if [ "$code" = "200" ]; then
      echo "=== attempt finished ==="; pp < /tmp/df_attempt.json; return 0
    fi
    echo "attempt not ready (HTTP $code); sleeping 15s..."
    sleep 15
  done
  echo "gave up after $tries polls; re-check with: $0 poll $uuid"
}

cmd="${1:-status}"
case "$cmd" in
  status) api_get status | pp ;;
  verify) echo "verify -> $URL"; api_post verify | pp ;;
  validate)
    if ! preflight; then
      echo "PRE-FLIGHT FAILED: can_connect/can_predict not both true. NOT queueing." >&2
      exit 3
    fi
    echo "=== 7.3 queueing validation for -> $URL ==="
    resp="$(api_post validate/queue)"; echo "$resp" | pp
    uuid="$(printf '%s' "$resp" | get_uuid)"
    if [ -n "$uuid" ]; then echo "queued uuid: $uuid"; poll "$uuid";
    else echo "no uuid returned; not polling."; fi ;;
  poll) poll "${2:?usage: $0 poll <uuid>}" ;;
  *) echo "usage: $0 {status|verify|validate|poll <uuid>}" >&2; exit 1 ;;
esac
