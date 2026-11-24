"""Run one job per seed across a pool of spawned worker processes.

The workers are single-threaded CUDA processes, one seed each, and every log
line they produce is appended to the batch log by `common.set_batch_log`.

A worker that cannot start, whose job raises, or that skips a seed exits
nonzero and says so in the batch log. `run_parallel` collects the exit codes
and the jobs that reached a terminal state, so the caller can exit nonzero
instead of reporting a successful batch when nothing ran.

Produces: the pool of the worker processes.
"""
from __future__ import annotations

import multiprocessing
import os
import queue as queue_module
import time
import traceback
from pathlib import Path

import common

RESULT_TIMEOUT = 30.0


class ParallelReport:
    """What the pool did: the exit codes, the failures and the unfinished jobs."""

    def __init__(self, exit_codes, failed, missing):
        self.exit_codes = list(exit_codes)
        self.failed = bool(failed)
        self.missing = list(missing)

    @property
    def ok(self):
        return not self.failed and not self.missing

    def summary(self):
        return (f'exit codes {self.exit_codes}, failed={self.failed}, '
                f'unfinished jobs {self.missing}')

    def __repr__(self):
        return f'ParallelReport({self.summary()})'


def _worker_main(job_queue, result_queue, worker, config):
    try:
        common.setup(config.get('threads'), config.get('device'))
        common.set_batch_log(config.get('batch_log'))
    except Exception:  # noqa: BLE001 - a dead worker must not look like a pass
        common.log_line('[worker] could not start:')
        common.log_line(traceback.format_exc())
        result_queue.put({'pid': os.getpid(), 'done': [], 'failed': True,
                          'error': 'the worker could not start'})
        raise SystemExit(1)

    failed = False
    while True:
        try:
            job = job_queue.get(timeout=0.5)
        except queue_module.Empty:
            continue
        if job is None:
            break
        seed = job
        lock_path = None
        acquired = False
        try:
            lock_path = Path(config['runs_dir']) / 'locks' / f'job_seed_{seed}.lock'
            acquired, reason = common.acquire_job_lock(lock_path)
            if not acquired:
                failed = True
                common.log_line(f'JOB | seed={seed} | name=* | skipped: {reason}')
                continue
            worker(seed, config)
        except Exception as exc:  # noqa: BLE001 - one seed must not kill the batch
            failed = True
            common.log_line(f'JOB | seed={seed} | name=* | FAILED '
                            f'{type(exc).__name__}: {exc}')
            common.log_line(traceback.format_exc())
        finally:
            if acquired and lock_path is not None:
                common.release_job_lock(lock_path)
    common.log_line(f'[worker] finished (pid {os.getpid()})')
    result_queue.put({'pid': os.getpid(), 'done': sorted(common.DONE_JOBS),
                      'failed': failed, 'error': None})
    if failed:
        raise SystemExit(1)
    return 0


def _collect(result_queue, expected):
    """Read the worker reports, waiting at most RESULT_TIMEOUT for each one."""
    reports = []
    deadline = time.monotonic() + RESULT_TIMEOUT
    while len(reports) < expected:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            reports.append(result_queue.get(timeout=remaining))
        except queue_module.Empty:
            break
    return reports


def run_parallel(procs, seeds, worker, config):
    """Run `worker(seed, config)` for every seed and report what happened.

    The report is not ok when a worker crashed, when a job raised, or when a
    planned job never reached a terminal state. The callers turn that into a
    nonzero exit code.
    """
    seeds = list(seeds)
    context = multiprocessing.get_context('spawn')
    job_queue = context.Queue()
    result_queue = context.Queue()
    for seed in seeds:
        job_queue.put(seed)
    for _ in range(procs):
        job_queue.put(None)
    workers = []
    for _ in range(procs):
        process = context.Process(target=_worker_main,
                                  args=(job_queue, result_queue, worker, config))
        process.daemon = False
        process.start()
        workers.append(process)
    for process in workers:
        process.join()

    reports = _collect(result_queue, len(workers))
    exit_codes = [process.exitcode for process in workers]
    failed = any(code != 0 for code in exit_codes) or len(reports) != len(workers)
    done = set()
    for report in reports:
        done.update(tuple(item) for item in report['done'])
        failed = failed or bool(report['failed'])
    missing = [(seed, name)
               for seed in sorted(seeds)
               for name in sorted(common.PLANNED_JOBS.get(seed, ()))
               if (seed, name) not in done]
    return ParallelReport(exit_codes, failed, missing)
