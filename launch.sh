#!/usr/bin/env bash
#
# Project launcher — preflight checks, port hygiene, then run something.
#
#   ./launch.sh                    start the dashboard (default)
#   ./launch.sh dashboard          same, explicitly
#   ./launch.sh ask "question"     run quick_question.py
#   ./launch.sh build "a thing"    run quick_build.py
#   ./launch.sh status             what's running right now
#   ./launch.sh stop               free the dashboard port
#
# Environment:
#   DASHBOARD_PORT   port for the dashboard (default 8787)
#   DASHBOARD_HOST   bind address (default 127.0.0.1 — there is no auth here)
#   OLLAMA_URL       Ollama endpoint (default http://localhost:11434)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PORT="${DASHBOARD_PORT:-8787}"
HOST="${DASHBOARD_HOST:-127.0.0.1}"
OLLAMA="${OLLAMA_URL:-http://localhost:11434}"

if [ -t 1 ]; then
  bold=$'\033[1m'; dim=$'\033[2m'; red=$'\033[31m'; grn=$'\033[32m'; ylw=$'\033[33m'; blu=$'\033[34m'; off=$'\033[0m'
else
  bold=""; dim=""; red=""; grn=""; ylw=""; blu=""; off=""
fi
say()  { printf '%s\n' "$*"; }
ok()   { printf '  %s✓%s %s\n' "$grn" "$off" "$*"; }
warn() { printf '  %s!%s %s\n' "$ylw" "$off" "$*"; }
bad()  { printf '  %s✗%s %s\n' "$red" "$off" "$*"; }
die()  { printf '\n%sERROR%s %s\n' "$red" "$off" "$*" >&2; exit 1; }

# ── which PIDs hold $PORT ────────────────────────────────────────────────────
port_pids() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true
  elif command -v ss >/dev/null 2>&1; then
    ss -ltnp 2>/dev/null | awk -v p=":$PORT\$" '$4 ~ p' \
      | grep -oP 'pid=\K[0-9]+' | sort -u || true
  elif command -v fuser >/dev/null 2>&1; then
    fuser "$PORT"/tcp 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]+$' || true
  fi
}

# Only ever kill our own dashboard. Anything else on the port is somebody's
# work — say what it is and stop, rather than guessing it's disposable.
free_port() {
  local pids; pids="$(port_pids)"
  [ -z "$pids" ] && { ok "port $PORT is free"; return 0; }

  local ours=() theirs=()
  for pid in $pids; do
    local cmd; cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
    if printf '%s' "$cmd" | grep -q 'dashboard\.py'; then ours+=("$pid"); else theirs+=("$pid: ${cmd:0:60}"); fi
  done

  if [ ${#theirs[@]} -gt 0 ]; then
    bad "port $PORT is held by something that isn't this dashboard:"
    printf '      %s\n' "${theirs[@]}"
    die "refusing to kill it. Stop it yourself, or set DASHBOARD_PORT to another port."
  fi

  for pid in "${ours[@]}"; do
    # the `uv run` parent needs to go too, or it respawns nothing but lingers
    local parent; parent="$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')"
    if [ -n "$parent" ] && ps -p "$parent" -o args= 2>/dev/null | grep -q 'dashboard\.py'; then
      kill "$parent" 2>/dev/null || true
    fi
    kill "$pid" 2>/dev/null || true
  done

  for _ in 1 2 3 4 5 6 7 8 9 10; do
    [ -z "$(port_pids)" ] && { ok "freed port $PORT"; return 0; }
    sleep 0.3
  done
  for pid in $(port_pids); do kill -9 "$pid" 2>/dev/null || true; done
  sleep 0.5
  [ -z "$(port_pids)" ] && ok "freed port $PORT (forced)" || die "could not free port $PORT"
}

preflight() {
  say "${bold}preflight${off}"
  command -v uv >/dev/null 2>&1 || die "uv not found — https://docs.astral.sh/uv/"
  ok "uv $(uv --version | awk '{print $2}')"

  [ -d .venv ] || { warn "no .venv — running uv sync"; uv sync -q; }
  uv sync -q 2>/dev/null && ok "dependencies in sync" || warn "uv sync reported a problem; continuing"

  if curl -fsS --max-time 3 "$OLLAMA/api/tags" >/dev/null 2>&1; then
    local n; n="$(curl -fsS --max-time 3 "$OLLAMA/api/tags" 2>/dev/null | grep -o '"name":' | wc -l | tr -d ' ' || true)"
    ok "ollama up at $OLLAMA (${n:-0} models)"
  else
    bad "ollama unreachable at $OLLAMA"
    say "      start it with: ${dim}ollama serve${off}"
    die "the local agents cannot run without it."
  fi

  # Only the escalation stage needs Claude; local-only runs are fine without it.
  if [ -n "${ANTHROPIC_API_KEY:-}" ]; then ok "ANTHROPIC_API_KEY set"
  else warn "ANTHROPIC_API_KEY unset — local stages work, Claude escalation will fail"; fi
  say ""
}

cmd_status() {
  say "${bold}status${off}"
  local pids; pids="$(port_pids)"
  if [ -n "$pids" ]; then
    for pid in $pids; do ok "dashboard on $PORT — pid $pid ($(ps -p "$pid" -o etime= | tr -d ' ') uptime)"; done
    say "      ${blu}http://$HOST:$PORT${off}"
  else
    warn "nothing listening on $PORT"
  fi
  if curl -fsS --max-time 3 "$OLLAMA/api/ps" >/dev/null 2>&1; then
    local resident
    resident="$(curl -fsS --max-time 3 "$OLLAMA/api/ps" 2>/dev/null \
      | grep -o '"name":"[^"]*"' | cut -d'"' -f4 | paste -sd', ' - || true)"
    ok "ollama up — resident: ${resident:-nothing loaded}"
  else
    bad "ollama down"
  fi
  local log="${QUICK_AGENTS_HOME:-$HOME/.quick-agents}/runs.jsonl"
  if [ -f "$log" ]; then
    ok "run log: $(grep -c run_started "$log" 2>/dev/null || true) runs recorded"
  else
    warn "no run log yet"
  fi
}

case "${1:-dashboard}" in
  dashboard|"")
    preflight; free_port
    say "${bold}dashboard${off}  ${blu}http://$HOST:$PORT${off}   ${dim}ctrl-c to stop${off}"
    say ""
    exec env DASHBOARD_HOST="$HOST" DASHBOARD_PORT="$PORT" uv run dashboard.py
    ;;
  ask)
    shift; [ $# -gt 0 ] || die "usage: ./launch.sh ask \"your question\""
    preflight; exec uv run quick_question.py "$@" ;;
  build)
    shift; [ $# -gt 0 ] || die "usage: ./launch.sh build \"what to build\""
    preflight; exec uv run quick_build.py "$@" ;;
  stop)   free_port ;;
  status) cmd_status ;;
  -h|--help|help)
    awk 'NR>2 && !/^#/ {exit} NR>2 {sub(/^# ?/,""); print}' "${BASH_SOURCE[0]}" ;;
  *) die "unknown command '${1}' — try: dashboard | ask | build | status | stop" ;;
esac
