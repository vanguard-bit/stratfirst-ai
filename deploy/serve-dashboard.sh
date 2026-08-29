#!/usr/bin/env bash
# Bind dashboard to this machine's Tailscale IPv4 (MagicDNS / 100.x).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${NSE_TRADER_DASHBOARD_PORT:-18765}"
HOST="${NSE_TRADER_DASHBOARD_HOST:-}"

if [[ -z "$HOST" ]]; then
  if command -v tailscale >/dev/null 2>&1; then
    HOST="$(tailscale ip -4 2>/dev/null | head -n1 || true)"
  fi
fi
if [[ -z "$HOST" ]]; then
  echo "tailscale IPv4 not found — set NSE_TRADER_DASHBOARD_HOST or start tailscaled" >&2
  exit 1
fi

cd "$ROOT"
exec "$ROOT/.venv/bin/python" main.py dashboard --serve --host "$HOST" --port "$PORT"
