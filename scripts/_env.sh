#!/usr/bin/env bash
# Loads .env from the repo root and sets defaults for the L1CC helpers.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$ROOT/.env" ] && set -a && . "$ROOT/.env" && set +a
L1CC_USER="${L1CC_USER:-linux1}"
L1CC_KEY="${L1CC_KEY:-$HOME/.ssh/claimsense.pem}"
L1CC_KEY="${L1CC_KEY/#\~/$HOME}"
: "${L1CC_HOST:?set L1CC_HOST in .env}"
SSH_OPTS=(-i "$L1CC_KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=15)
