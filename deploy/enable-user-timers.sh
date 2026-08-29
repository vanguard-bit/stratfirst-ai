#!/usr/bin/env bash
# Install user-level systemd timers for nse-trader (paper local sim).
# Units are generated from deploy/systemd templates with the resolved project root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
TEMPLATE_DIR="$ROOT/deploy/systemd"
mkdir -p "$UNIT_DIR" "$ROOT/data/logs" "$ROOT/data/state"

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "Missing $ROOT/.venv/bin/python — create a venv and pip install -r requirements.txt first." >&2
  exit 1
fi

for f in \
  nse-trader-ingest.service \
  nse-trader-ingest.timer \
  nse-trader-paper.service \
  nse-trader-paper.timer \
  nse-trader-eod.service \
  nse-trader-eod.timer \
  nse-trader-fyers-refresh.service \
  nse-trader-fyers-refresh.timer \
  nse-trader-llm.service \
  nse-trader-llm.timer \
  nse-trader-dashboard.service
do
  src="$TEMPLATE_DIR/$f"
  dst="$UNIT_DIR/$f"
  if [[ ! -f "$src" ]]; then
    echo "Missing template: $src" >&2
    exit 1
  fi
  # Replace any prior symlink/mask so we write a real generated unit (never through a symlink into templates).
  rm -f "$dst"
  sed "s|__PROJECT_ROOT__|$ROOT|g" "$src" >"$dst"
done

systemctl --user daemon-reload
systemctl --user unmask \
  nse-trader-ingest.timer \
  nse-trader-paper.timer \
  nse-trader-eod.timer \
  nse-trader-fyers-refresh.timer \
  nse-trader-llm.timer \
  nse-trader-dashboard.service 2>/dev/null || true
systemctl --user enable --now \
  nse-trader-ingest.timer \
  nse-trader-paper.timer \
  nse-trader-eod.timer \
  nse-trader-fyers-refresh.timer \
  nse-trader-llm.timer \
  nse-trader-dashboard.service
systemctl --user status --no-pager \
  nse-trader-ingest.timer \
  nse-trader-paper.timer \
  nse-trader-eod.timer \
  nse-trader-fyers-refresh.timer \
  nse-trader-llm.timer \
  nse-trader-dashboard.service || true
echo
echo "Enabled from $ROOT"
echo "Check: systemctl --user list-timers 'nse-trader*'"
TS_IP="$(tailscale ip -4 2>/dev/null | head -n1 || true)"
echo "Dashboard: http://${TS_IP:-<tailscale-ip>}:18765/dashboard.html"
echo "Logs: $ROOT/data/logs/"
