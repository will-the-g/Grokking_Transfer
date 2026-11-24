#!/usr/bin/env bash
# Twelfth overnight queue on the rented Linux GPU instance (Ubuntu, /venv/main).
#
# Steps, in order. Every step appends its rows to one CSV under
# runs_nseed_queue12/, so a partial run survives and the finished rows are
# skipped. There is no self-stop here: the gated stop is issued by hand from
# the laptop once results_queue12.zip has been verified there.
#
#   1  A1  the Pythia stage-1 trajectory of pythia-1.4b at step143000.
#   2  A2  the Pythia checkpoint sweep over pythia-1b and pythia-1.4b.
#   3  A3  the toy learning curves of the no-LN and the Pre-LN sources.
#   4  pack the queue12 CSVs and the report into results_queue12.zip.
#
# Flags:
#   --smoke            run every step with a tiny size so that the wiring can
#                      be tested
#
# Environment:
#   PROCS   worker processes (default 3 for the Pythia steps, 6 for the toy
#           step)
#   PYTHON  interpreter to use (default: /venv/main/bin/python, else python3)
#
# Usage:  bash nseed_backbone/run_queue12.sh
#         bash nseed_backbone/run_queue12.sh --smoke
set -u

SMOKE=0
for argument in "$@"; do
    case "${argument}" in
        --smoke) SMOKE=1 ;;
        --no-stop|--stop-dry-run) ;;
        -h|--help) sed -n '2,32p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown flag: ${argument}" >&2; exit 2 ;;
    esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
cd "${ROOT}"

PROCS="${PROCS:-3}"
TOY_PROCS="${TOY_PROCS:-6}"
if [ -z "${PYTHON:-}" ]; then
    if [ -x /venv/main/bin/python ]; then
        PYTHON=/venv/main/bin/python
    else
        PYTHON=python3
    fi
fi

export NSEED_DETERMINISTIC=1
export NSEED_DETERMINISTIC_GPU=1
export PYTHONUNBUFFERED=1
export PYTHON
export HF_HUB_DISABLE_XET=1

LOG="runs_nseed/batch.log"
mkdir -p runs_nseed

if [ "${SMOKE}" = "1" ]; then
    OUT_DIR="runs_nseed/_queue12_smoke"
    A1_SEEDS="0"
    A1_SOURCES="add_gen add_mem copy_a"
    A2_MODELS="EleutherAI/pythia-1b"
    A2_REVISIONS="step143000"
    A2_SEEDS="0"
    A2_SOURCES="add add_mem"
    TOY_SEEDS_NOLN="5"
    TOY_SEEDS_PRELN="5"
    STEPS=10
    STAGE2_STEPS=10
    DIRECT_LONG_STEPS=10
    SAVE_EVERY=5
    TOY_STEPS=20
    TOY_EVAL=10
else
    OUT_DIR="runs_nseed_queue12"
    A1_SEEDS="0 1 2 3 4 5"
    A1_SOURCES="add_gen add_mem copy_a"
    A2_MODELS="EleutherAI/pythia-1b EleutherAI/pythia-1.4b"
    A2_REVISIONS="step1000 step2000 step4000 step8000 step16000 step32000 step64000 step100000 step143000"
    A2_SEEDS="0 1 2 3 4 5"
    A2_SOURCES="add copy_a max shuffled add_mem"
    TOY_SEEDS_NOLN="5 6 7 8 9 10 11 13 14"
    TOY_SEEDS_PRELN="5 6 7 8"
    TOY_SEEDS_PRELN2="10 11 12 13 14"
    STEPS=1000
    STAGE2_STEPS=600
    DIRECT_LONG_STEPS=1600
    SAVE_EVERY=100
    TOY_STEPS=20000
    TOY_EVAL=500
fi

echo "Queue 12 - project root ${ROOT}"
echo "  python   : ${PYTHON} ($(${PYTHON} --version 2>&1))"
echo "  procs    : ${PROCS} (pythia), ${TOY_PROCS} (toy)"
echo "  smoke    : ${SMOKE}"
echo "  out      : ${OUT_DIR}"
echo "  log      : ${LOG}"
echo

TOTAL_STEPS=4
step () {
    local number="$1"; shift
    local description="$1"; shift
    "${PYTHON}" - "${number}" "${description}" "start" "${TOTAL_STEPS}" <<'PY'
import sys
sys.path.insert(0, 'nseed_backbone')
import common
common.set_batch_log('runs_nseed/batch.log')
common.prompt_step(sys.argv[1], int(sys.argv[4]), sys.argv[2], phase=sys.argv[3])
PY
    if ! "$@"; then
        "${PYTHON}" - "${number}" "${description}" "end" "${TOTAL_STEPS}" <<'PY'
import sys
sys.path.insert(0, 'nseed_backbone')
import common
common.set_batch_log('runs_nseed/batch.log')
common.prompt_step(sys.argv[1], int(sys.argv[4]), sys.argv[2] + ' [FAILED]', phase='end')
PY
        echo "Queue 12: step ${number} failed. Re-run this script to continue;" \
             "the finished rows are skipped." >&2
        exit 1
    fi
    "${PYTHON}" - "${number}" "${description}" "end" "${TOTAL_STEPS}" <<'PY'
import sys
sys.path.insert(0, 'nseed_backbone')
import common
common.set_batch_log('runs_nseed/batch.log')
common.prompt_step(sys.argv[1], int(sys.argv[4]), sys.argv[2], phase='end')
PY
}

a1_step () {
    "${PYTHON}" nseed_backbone/queue12_steps.py traj --model EleutherAI/pythia-1.4b \
        --revision step143000 --seeds ${A1_SEEDS} --sources ${A1_SOURCES} \
        --out "${OUT_DIR}" --steps "${STEPS}" --save-every "${SAVE_EVERY}" \
        --stage2-steps "${STAGE2_STEPS}" --direct-long-steps "${DIRECT_LONG_STEPS}" \
        --procs "${PROCS}" --device cuda
}

a2_step () {
    "${PYTHON}" nseed_backbone/queue12_steps.py sweep --models ${A2_MODELS} \
        --revisions ${A2_REVISIONS} --seeds ${A2_SEEDS} --sources ${A2_SOURCES} \
        --out "${OUT_DIR}" --steps "${STEPS}" --stage2-steps "${STAGE2_STEPS}" \
        --direct-long-steps "${DIRECT_LONG_STEPS}" --procs "${PROCS}" --device cuda
}

a3_step () {
    "${PYTHON}" nseed_backbone/queue12_steps.py toy --arch no-LN --runs-root runs_nseed \
        --seeds ${TOY_SEEDS_NOLN} --out "${OUT_DIR}" --steps "${TOY_STEPS}" \
        --eval-every "${TOY_EVAL}" --procs "${TOY_PROCS}" --device cuda
    NSEED_PRELN=1 NSEED_BATCH=512 "${PYTHON}" nseed_backbone/queue12_steps.py toy \
        --arch preln --runs-root runs_nseed_p1 --seeds ${TOY_SEEDS_PRELN} \
        --out "${OUT_DIR}" --steps "${TOY_STEPS}" --eval-every "${TOY_EVAL}" \
        --procs "${TOY_PROCS}" --device cuda
    NSEED_PRELN=1 NSEED_BATCH=512 "${PYTHON}" nseed_backbone/queue12_steps.py toy \
        --arch preln --runs-root runs_nseed_k2 --seeds ${TOY_SEEDS_PRELN2} \
        --out "${OUT_DIR}" --steps "${TOY_STEPS}" --eval-every "${TOY_EVAL}" \
        --procs "${TOY_PROCS}" --device cuda
}

pack_step () {
    "${PYTHON}" nseed_backbone/queue12_steps.py pack --out results_queue12.zip \
        --root "${ROOT}" \
        --failures "A1: CUDA OOM twice with 3 workers (11 GB per worker); fixed to one model per worker and rerun with PROCS=2 (0 failures in the final run)"
}

step 1 "A1 the Pythia stage-1 trajectory of pythia-1.4b at step143000" \
    a1_step

step 2 "A2 the Pythia checkpoint sweep over pythia-1b and pythia-1.4b" \
    a2_step

step 3 "A3 the toy learning curves of the no-LN and the Pre-LN sources" \
    a3_step

step 4 "Pack the queue12 CSVs and the report into results_queue12.zip" \
    pack_step

echo
echo "Queue 12 finished. The archive is results_queue12.zip. The instance is left running."
