#!/usr/bin/env bash
# Loads .env then runs the bot. Usage: ./run.sh --once
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] || { echo "No .env found. cp .env.example .env and fill it in."; exit 1; }
set -a; source .env; set +a
exec python3 bot.py "$@"
