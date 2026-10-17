#!/usr/bin/env bash
# Forward the dashboard (8000) and Jupyter (38888) to this laptop. Nothing is exposed publicly.
set -euo pipefail
. "$(dirname "$0")/_env.sh"
echo "dashboard: http://localhost:8000/ui   jupyter: http://localhost:38888"
exec ssh "${SSH_OPTS[@]}" -N -L 8000:localhost:8000 -L 38888:localhost:38888 "$L1CC_USER@$L1CC_HOST"
