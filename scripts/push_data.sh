#!/usr/bin/env bash
# Copy raw data (and optionally trained models) to the VM's shared folder.
set -euo pipefail
. "$(dirname "$0")/_env.sh"
ssh "${SSH_OPTS[@]}" "$L1CC_USER@$L1CC_HOST" "mkdir -p ~/shared/data/raw ~/shared/models"
rsync -az --progress -e "ssh ${SSH_OPTS[*]}" "$ROOT/shared/data/raw/" "$L1CC_USER@$L1CC_HOST:~/shared/data/raw/"
