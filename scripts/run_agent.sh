#!/usr/bin/env bash
# Supervisor for the unattended overnight run. Launches a headless Claude Code agent that executes HANDOFF.md,
# and relaunches it (resuming the same session) if it exits before results/DONE exists.
# Start detached:  setsid nohup bash scripts/run_agent.sh > logs/agent_supervisor.log 2>&1 < /dev/null &
set -uo pipefail
cd /home/ubuntu/MOE
if [ -s scripts/claude_bin.txt ]; then CLAUDE_BIN="$(cat scripts/claude_bin.txt)"
else CLAUDE_BIN="$(command -v claude || echo "$HOME/.local/bin/claude")"; fi
if [ ! -x "$CLAUDE_BIN" ]; then echo "claude binary not found at '$CLAUDE_BIN'. Install with: curl -fsSL https://claude.ai/install.sh | bash"; exit 1; fi
echo "using claude binary: $CLAUDE_BIN"
SESSION_ID_FILE=logs/agent_session_id.txt
MAX_RUNS=${MAX_RUNS:-20}
export HF_HOME=/opt/dlami/nvme/hf
mkdir -p logs results

FIRST_PROMPT='You are running unattended overnight. Read /home/ubuntu/MOE/HANDOFF.md fully, then /home/ubuntu/MOE/PLAN.md, and execute the whole reproduction end to end exactly as specified there. Never ask questions; make the documented default choice and log it in logs/PROGRESS.md. Keep working until results/DONE exists. Start now.'
RESUME_PROMPT='You were restarted (the previous process exited). Re-read /home/ubuntu/MOE/HANDOFF.md and /home/ubuntu/MOE/logs/PROGRESS.md, inspect what already exists under moetrace/ and results/, then continue from the last completed milestone. Do not redo finished passes whose parquet outputs exist. Keep working until results/DONE exists, then stop.'

run=0
while [ ! -f results/DONE ] && [ "$run" -lt "$MAX_RUNS" ]; do
  run=$((run+1))
  ts=$(date +%Y%m%d_%H%M%S)
  log="logs/agent_run${run}_${ts}.jsonl"
  echo "$(date -u +%FT%TZ) starting run $run -> $log"
  if [ -s "$SESSION_ID_FILE" ]; then
    sid="$(cat "$SESSION_ID_FILE")"
    env -i HOME=/home/ubuntu USER=ubuntu PATH=/home/ubuntu/MOE/.venv/bin:/usr/local/bin:/usr/bin:/bin TERM=dumb LANG=C.UTF-8 HF_HOME=/opt/dlami/nvme/hf \
      "$CLAUDE_BIN" -p "$RESUME_PROMPT" --resume "$sid" --dangerously-skip-permissions \
      --output-format stream-json --verbose > "$log" 2> "${log%.jsonl}.err"
    rc=$?
  else
    sid="$(python3 -c 'import uuid; print(uuid.uuid4())')"
    echo "$sid" > "$SESSION_ID_FILE"
    env -i HOME=/home/ubuntu USER=ubuntu PATH=/home/ubuntu/MOE/.venv/bin:/usr/local/bin:/usr/bin:/bin TERM=dumb LANG=C.UTF-8 HF_HOME=/opt/dlami/nvme/hf \
      "$CLAUDE_BIN" -p "$FIRST_PROMPT" --session-id "$sid" --dangerously-skip-permissions \
      --output-format stream-json --verbose > "$log" 2> "${log%.jsonl}.err"
    rc=$?
  fi
  echo "$(date -u +%FT%TZ) run $run exited rc=$rc"
  # If resume fails because the session cannot be found, fall back to a fresh session next time.
  if grep -qiE "No conversation found|session.*not found|Failed to resume" "${log%.jsonl}.err" 2>/dev/null; then
    echo "$(date -u +%FT%TZ) resume failed; starting a fresh session next run"; rm -f "$SESSION_ID_FILE"
  fi
  [ -f results/DONE ] && break
  sleep 30
done
echo "$(date -u +%FT%TZ) supervisor finished; DONE=$([ -f results/DONE ] && echo yes || echo no) runs=$run"
