#!/usr/bin/env bash
# One-off check: does the deploy key have an empty passphrase?
if ssh-keygen -y -P "" -f "$HOME/.ssh/id_ed25519" >/dev/null 2>&1; then
  echo "PASSPHRASE: empty - good, ssh/scp will not prompt"
else
  echo "PASSPHRASE: NOT empty - regenerate or use ssh-agent"
fi
