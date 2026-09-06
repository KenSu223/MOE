#!/usr/bin/env bash
# Supervisor for an unattended headless Claude Code run.
# Launches an agent that executes HANDOFF.md, and relaunches it (resuming the same session) until results/DONE exists.
#
# Start detached (from a terminal the user owns, not from inside another Claude session):
#   tmux new -d -s moe "bash scripts/run_agent.sh > logs/agent_supervisor.log 2>&1"
# Watch:   tail -f logs/agent_supervisor.log ; cat logs/agent_status.txt ; tail logs/PROGRESS.md
#
# Lessons from 2026-09-03 baked in:
#   * a usage-limit exit ("You've hit your session limit · resets 6:10am (UTC)") is NOT a failure to retry every 30 s:
#     the supervisor now parses the reset time from the result event and sleeps until then (+ margin), and such waits
#     do not count against MAX_RUNS;
#   * every exit reason is classified and written to logs/agent_status.txt so a human (or the next session) sees it;
#   * other errors back off exponentially (30 s, 60 s, ... capped at 30 min).
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
if [ -s scripts/claude_bin.txt ]; then CLAUDE_BIN="$(cat scripts/claude_bin.txt)"
else CLAUDE_BIN="$(command -v claude || echo "$HOME/.local/bin/claude")"; fi
if [ ! -x "$CLAUDE_BIN" ]; then echo "claude binary not found at '$CLAUDE_BIN'. Install with: curl -fsSL https://claude.ai/install.sh | bash"; exit 1; fi
echo "using claude binary: $CLAUDE_BIN"
SESSION_ID_FILE=logs/agent_session_id.txt
STATUS_FILE=logs/agent_status.txt
MAX_RUNS=${MAX_RUNS:-20}            # counts only real agent runs, not usage-limit waits
LIMIT_MARGIN_S=${LIMIT_MARGIN_S:-300}
MAX_LIMIT_WAIT_S=${MAX_LIMIT_WAIT_S:-21600}
export HF_HOME=${HF_HOME:-/opt/dlami/nvme/hf}
mkdir -p logs results

FIRST_PROMPT="You are running unattended. Read $ROOT/HANDOFF.md fully, then $ROOT/PLAN.md, and execute the whole task end to end exactly as specified there. Never ask questions; make the documented default choice and log it in logs/PROGRESS.md. Keep working until results/DONE exists. Start now."
RESUME_PROMPT="You were restarted (the previous process exited). Re-read $ROOT/HANDOFF.md and $ROOT/logs/PROGRESS.md, inspect what already exists under the code and results directories, then continue from the last completed milestone. Do not redo finished passes whose outputs exist. Keep working until results/DONE exists, then stop."

status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$STATUS_FILE"; }

# Print the final 'result' text of a stream-json log (empty if none).
result_text() { python3 - "$1" <<'PY'
import json, sys
txt = ""
for line in open(sys.argv[1], errors="replace"):
    try: d = json.loads(line)
    except Exception: continue
    if d.get("type") == "result": txt = str(d.get("result", ""))
print(txt[:500])
PY
}

# Seconds to sleep for a usage-limit message; parses "resets 6:10am (UTC)" / "resets 11pm (UTC)" / "resets Sep 4, 6:10am (UTC)".
limit_wait_seconds() { python3 - "$1" "$LIMIT_MARGIN_S" "$MAX_LIMIT_WAIT_S" <<'PY'
import re, sys, datetime as dt
msg, margin, cap = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
now = dt.datetime.now(dt.timezone.utc)
m = re.search(r"resets?\s+(.+?)\s*\((UTC|GMT)\)", msg, re.I)
wait = 3600
if m:
    s = m.group(1).strip().replace(" ", "")
    for fmt in ("%I:%M%p", "%I%p", "%b%d,%I:%M%p", "%b%d,%I%p"):
        try:
            t = dt.datetime.strptime(s.upper(), fmt)
            if "%b" in fmt:
                target = t.replace(year=now.year, tzinfo=dt.timezone.utc)
                if target < now - dt.timedelta(days=1): target = target.replace(year=now.year + 1)
            else:
                target = now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
                if target <= now: target += dt.timedelta(days=1)
            wait = int((target - now).total_seconds())
            break
        except ValueError:
            continue
print(max(60, min(wait + margin, cap)))
PY
}

run=0; backoff=30
while [ ! -f results/DONE ] && [ "$run" -lt "$MAX_RUNS" ]; do
  run=$((run+1))
  ts=$(date +%Y%m%d_%H%M%S)
  log="logs/agent_run${run}_${ts}.jsonl"
  status "starting run $run -> $log"
  if [ -s "$SESSION_ID_FILE" ]; then
    sid="$(cat "$SESSION_ID_FILE")"
    env -i HOME="$HOME" USER="$USER" PATH="$ROOT/.venv/bin:/usr/local/bin:/usr/bin:/bin" TERM=dumb LANG=C.UTF-8 HF_HOME="$HF_HOME" \
      "$CLAUDE_BIN" -p "$RESUME_PROMPT" --resume "$sid" --dangerously-skip-permissions \
      --output-format stream-json --verbose > "$log" 2> "${log%.jsonl}.err"
    rc=$?
  else
    sid="$(python3 -c 'import uuid; print(uuid.uuid4())')"
    echo "$sid" > "$SESSION_ID_FILE"
    env -i HOME="$HOME" USER="$USER" PATH="$ROOT/.venv/bin:/usr/local/bin:/usr/bin:/bin" TERM=dumb LANG=C.UTF-8 HF_HOME="$HF_HOME" \
      "$CLAUDE_BIN" -p "$FIRST_PROMPT" --session-id "$sid" --dangerously-skip-permissions \
      --output-format stream-json --verbose > "$log" 2> "${log%.jsonl}.err"
    rc=$?
  fi
  res="$(result_text "$log")"
  status "run $run exited rc=$rc; result: ${res:0:160}"
  [ -f results/DONE ] && break

  if grep -qiE "No conversation found|session.*not found|Failed to resume" "${log%.jsonl}.err" 2>/dev/null; then
    status "resume failed; starting a fresh session next run"; rm -f "$SESSION_ID_FILE"
  fi

  if echo "$res" | grep -qiE "hit your (session|usage) limit|usage limit|rate limit|out of (usage|credits)"; then
    wait_s=$(limit_wait_seconds "$res")
    run=$((run-1))   # a limit wait is not a real attempt
    status "USAGE LIMIT: sleeping ${wait_s}s (until $(date -u -d "+${wait_s} seconds" +%FT%TZ)) before resuming"
    sleep "$wait_s"
    continue
  fi
  status "sleeping ${backoff}s before the next attempt"
  sleep "$backoff"; backoff=$(( backoff * 2 > 1800 ? 1800 : backoff * 2 ))
done
status "supervisor finished; DONE=$([ -f results/DONE ] && echo yes || echo no) runs=$run"
