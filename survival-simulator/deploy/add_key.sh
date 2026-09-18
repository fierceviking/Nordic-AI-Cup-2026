#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Grant a teammate SSH access to the agent server.
# ---------------------------------------------------------------------------
# Per-person keys, never a shared private key: that way access can be revoked
# for one person without re-keying everyone.
#
# Ask them for the PUBLIC key only (~/.ssh/id_ed25519.pub). A private key
# should never be sent over chat/email.
#
# Usage:
#   ./deploy/add_key.sh root@89.167.18.234 "ssh-ed25519 AAAA... alice"
#   ./deploy/add_key.sh root@89.167.18.234 --list
#   ./deploy/add_key.sh root@89.167.18.234 --revoke alice
# ---------------------------------------------------------------------------
set -euo pipefail

TARGET="${1:?usage: $0 user@host \"<ssh-public-key>\" | --list | --revoke <comment>}"
ARG="${2:?missing key, --list or --revoke}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)

case "$ARG" in
  --list)
    ssh "${SSH_OPTS[@]}" "$TARGET" 'cat -n ~/.ssh/authorized_keys'
    ;;
  --revoke)
    WHO="${3:?usage: $0 user@host --revoke <comment>}"
    ssh "${SSH_OPTS[@]}" "$TARGET" \
      "grep -v ' ${WHO}\$' ~/.ssh/authorized_keys > ~/.ssh/ak.tmp \
       && mv ~/.ssh/ak.tmp ~/.ssh/authorized_keys \
       && chmod 600 ~/.ssh/authorized_keys \
       && echo 'revoked ${WHO}; remaining:' && cat -n ~/.ssh/authorized_keys"
    ;;
  ssh-*)
    # Appended only if absent, so re-running is harmless.
    ssh "${SSH_OPTS[@]}" "$TARGET" \
      "mkdir -p ~/.ssh && chmod 700 ~/.ssh \
       && grep -qxF '$ARG' ~/.ssh/authorized_keys 2>/dev/null \
       && echo 'key already present' \
       || { echo '$ARG' >> ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys; echo 'key added'; } \
       ; echo 'authorized keys now:' ; cat -n ~/.ssh/authorized_keys"
    ;;
  *)
    echo "ERROR: that does not look like an SSH public key (should start 'ssh-')." >&2
    exit 2
    ;;
esac
