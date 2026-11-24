#!/bin/bash
# Queue 22: the two experiments of the invented-symbol task.
#   experiment 2   the new targets a+3b and a-2b, the budget of 3,000 steps
#   experiment 1   the target a+2b, the budget of 20,000 steps
# The scripts write their rows into the CSV as they come, so a disconnect does
# not lose the progress; the log of every step is a file of its own.
set -u
cd "${ROOT}" || exit 1
mkdir -p runs_queue22
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export NSEED_DETERMINISTIC=1
export NSEED_DETERMINISTIC_GPU=1
export PYTHONUNBUFFERED=1
PY=/root/venv/bin/python
PROCS="${PROCS:-4}"

echo "START experiment 2 $(date -u +%FT%TZ)  disk=$(df -BG / | tail -1 | awk '{print $4}')"
$PY nseed_backbone/q22_run.py --experiment 2 --procs "$PROCS" \
    --out runs_queue22/q22_targets.csv \
    --check runs_queue22/q22_source_check.csv \
    2>&1 | tee -a runs_queue22/exp2.log
echo "END experiment 2 rc=$? $(date -u +%FT%TZ)  disk=$(df -BG / | tail -1 | awk '{print $4}')"
