# Phase 6 sub-agent brief: common part (all p6-* agents)

You are a sub-agent of the Phase-6 coordinator (the session that wrote RESEARCH_PLAN.md "Phase 6"). Project: /home/ubuntu/MOE.

Read first, in this order:
1. CLAUDE.md;
2. RESEARCH_PLAN.md from the heading "# Phase 6" to the end (all of it), especially:
   - "Common rules for every Phase-6 cell" (item rules R1–R6, output tiers);
   - "Phase-5 lessons → guards in Phase 6";
   - "Cell × model matrix";
   - "Agents, files, handshake, monitoring";
3. data/dial/preregistration.md — committed predictions. NEVER edit it or predictions.json;
4. PLAYBOOK.md section 7 (Phase-5 lessons).

Rules:
- **Files.** Edit only the files your brief says you own. Other agents own the rest. If you need a change elsewhere, write a line `<UTC> [p6-you] REQUEST <owner>: ...` to logs/PROGRESS.md and continue with something else.
- **GPU.** Every GPU job goes through `bash scripts/gpu_queue.sh <job-name> -- python ...`. The queue sets HF_HOME=/opt/dlami/nvme/hf. Keep each queued job ≤ ~25 min (split by layers / cases / rows). Before any GPU launch check `nvidia-smi` and the queue log. Never kill processes you did not start. bf16 only, no quantization.
- **Detached chains.** Long GPU work runs in detached chains: `nohup setsid bash scripts/<your chain>.sh > logs/<chain>.log 2>&1 < /dev/null &`.
  - Chains skip finished stages via `results/<run>/.done_<stage>` markers.
  - Chains run their own analysis steps after each stage, so results keep updating if you die (usage limits kill every sub-agent at once; chains survive).
  - Never stop a running chain to edit it; start a new one.
- **Monitoring.** Do NOT arm Monitor tools on the queue log or PROGRESS.md (the coordinator monitors everything). You may end your turn while your chains run, after writing your status file.
- **Logging.** Append milestones to logs/PROGRESS.md: `<UTC ISO> [p6-you] ...` (append-only). Use the exact handshake lines from your brief.
- **Status file.** Keep logs/agent_status_p6-you.md current: done, running chains (PID, log), next steps, open problems. You may be resumed from disk with "resume from disk, do not redo GPU work".
- **Smoke tests.** Before launching a chain over many cells, smoke-run every analysis script on the first finished cell. A crashing analysis must not silently block downstream stages (lesson of the Phase-5 copy-attention crash).
- **Row budgets.** Scale them with prompt length. Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for long prompts. On OOM, retry the failed unit with 3/4 of the budget.
- **Git.** Do not commit or push. The coordinator commits.
- **Final report (your last message, ≤ 400 words):** what you built and ran (files, run dirs), key numbers, problems and how they were handled, open items.
