#!/usr/bin/env bash
# Copy the repo to ~/claimsense on the VM (code only; data, models and secrets stay out).
set -euo pipefail
. "$(dirname "$0")/_env.sh"
rsync -az --delete -e "ssh ${SSH_OPTS[*]}" \
  --exclude .git --exclude .venv --exclude .env --exclude 'secrets/' \
  --exclude 'shared/' --exclude '__pycache__/' --exclude '.pytest_cache/' --exclude '*.pem' \
  "$ROOT/" "$L1CC_USER@$L1CC_HOST:~/claimsense/"
echo "synced to $L1CC_HOST:~/claimsense"
