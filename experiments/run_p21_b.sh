#!/bin/bash
# Queue 21, step P2: the LayerNorm by batch-size 2x2, the two missing cells.
#
# The cell comes from the environment, because common reads NSEED_PRELN and
# NSEED_BATCH when it is imported. The two cells run one after the other.
set -u
cd "${ROOT}" || exit 1
mkdir -p runs_nseed_queue21
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export NSEED_DETERMINISTIC=1
export NSEED_DETERMINISTIC_GPU=1
export PYTHONUNBUFFERED=1
PY=/root/venv/bin/python
COMMON="--seeds 5 6 7 8 9 --procs 2 --out-dir runs_nseed_queue21 \
        --runs-root runs_nseed_queue21/runs_nseed_q21"

echo "START none/512 $(date -u +%FT%TZ)  disk=$(df -BG / | tail -1 | awk '{print $4}')"
NSEED_BATCH=512 $PY nseed_backbone/p21_lnbatch.py run $COMMON
echo "END none/512 rc=$? $(date -u +%FT%TZ)  disk=$(df -BG / | tail -1 | awk '{print $4}')"

echo "START preln/full $(date -u +%FT%TZ)  disk=$(df -BG / | tail -1 | awk '{print $4}')"
NSEED_PRELN=1 $PY nseed_backbone/p21_lnbatch.py run $COMMON
echo "END preln/full rc=$? $(date -u +%FT%TZ)  disk=$(df -BG / | tail -1 | awk '{print $4}')"

# The 20,000-step transfers of the cell no LayerNorm with the full batch. The
# sources of that cell are on the instance, so nothing is trained again; only
# the long transfers are added to the rows that are reused.
echo "START none/full long $(date -u +%FT%TZ)  disk=$(df -BG / | tail -1 | awk '{print $4}')"
$PY nseed_backbone/p21_lnbatch.py long --seeds 5 6 7 8 9 --ln none --batch full \
    --runs-root "${EXISTING_ROOT:-results/existing}/runs_nseed" --out-dir runs_nseed_queue21
echo "END none/full long rc=$? $(date -u +%FT%TZ)  disk=$(df -BG / | tail -1 | awk '{print $4}')"

echo "ALL DONE $(date -u +%FT%TZ)"
