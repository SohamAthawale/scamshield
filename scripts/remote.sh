#!/usr/bin/env bash
# Run a command on the L1CC VM: scripts/remote.sh "docker ps"
set -euo pipefail
. "$(dirname "$0")/_env.sh"
exec ssh "${SSH_OPTS[@]}" "$L1CC_USER@$L1CC_HOST" "$@"
