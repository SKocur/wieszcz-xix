#!/usr/bin/env bash
# Cheap cron probe: the fact sheet alone, no LLM. Escalates to a full
# monitor_pass.sh (Opus) only when the sheet flags something.
set -u
cd "$(dirname "$0")/.."
{
  date -u +'-- probe %Y-%m-%dT%H:%M:%SZ --'
  .venv/bin/python3 scripts/monitor_status.py
} >> "$HOME/monitor.log" 2>&1
if [ $? -ne 0 ]; then
  exec "$(dirname "$0")/monitor_pass.sh"
fi
