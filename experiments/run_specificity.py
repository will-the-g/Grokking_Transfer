"""Specificity: does the grokked addition structure help other source functions?

For each seed, two alternative source tasks are trained with the same success
rule and the same 20,000-step cap as the addition source:

  copy_a    y = a in every context
  selector  y = a in contexts 0 and 2, y = b in contexts 1 and 3

When a source reaches success, a 2,000-step affine transfer is run from its
success checkpoint. A source that does not reach success is reported and its
transfer is skipped.

The seed's existing mask and splits are used.

    python nseed_backbone\\run_specificity.py --seeds 5 6 7 --device cuda

Produces: Figure 2 (the copy and the selector sources).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
HERE = Path(__file__).resolve().parent
SRC = HERE.parent / 'src'
for _path in (str(SRC),):
    if _path not in sys.path:
        sys.path.insert(0, _path)

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402
from parallel import run_parallel  # noqa: E402
from run_components import run_transfer_job  # noqa: E402
from run_seed import SeedRun  # noqa: E402

SOURCE_TASKS = ('copy_a', 'selector')
TRANSFER_STEPS = 2000
EVAL_EVERY = 100


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else common.PARENT / path


def run_specificity(seed, out_dir, device, steps=TRANSFER_STEPS, eval_every=EVAL_EVERY,
                    source_cap=common.SOURCE_CAP, log=common.log_line):
    """Both alternative sources and their affine transfers for one seed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'ckpts').mkdir(parents=True, exist_ok=True)
    summary = common.read_summary(out_dir / 'summary.json')
    summary.setdefault('seed', seed)
    summary.setdefault('device', device)
    summary['runtime'] = common.runtime_metadata(device)
    jobs = summary.setdefault('jobs', {})

    mask = common.mask_for_seed(seed)
    strata, _ = common.split_for_mask(mask)
    target_data = common.build_task('affine', device)

    for task in SOURCE_TASKS:
        job_name = f'source_{task}'
        existing = jobs.get(job_name, {})
        needs_source = not existing.get('done')
        needs_transfer = (existing.get('done') and existing.get('status') == 'grokked'
                          and not existing.get('transfer_done'))
        if not needs_source and not needs_transfer:
            log(f'  [{job_name}] already done, skipping')
            continue
        record = dict(existing)
        source_data = common.build_task(task, device)
        if needs_source:
            run = SeedRun(seed, out_dir, threads=1, resume=False, tiers=(),
                          device=device, source_cap=source_cap, log=log)
            run._ensure_dirs()
            common.job_line(seed, job_name, 0, run.source_cap, 'nan', 'start')
            started = time.time()
            status, grok_step, rows = run.train_source(strata, source_data, run._summary())
            common.job_line(seed, job_name, grok_step or run.source_cap, run.source_cap,
                            'nan', 'source_done')
            record = dict(name=job_name, kind='specificity_source', task=task, seed=seed,
                          source_task=task, status=status, grok_step=grok_step,
                          steps_run=run.source_cap, elapsed_s=time.time() - started,
                          done=True)
            jobs[job_name] = record
            common.write_summary(out_dir / 'summary.json', summary)
        else:
            log(f'  [{job_name}] the source is done but its transfer is not; '
                f'running the transfer only')
        if record.get('status') == 'grokked':
            model = common.make_model(common.CONFIG)
            model.load_state_dict(common.load_checkpoint(
                out_dir / 'ckpts' / 'grokked.pt')['model'])
            model = model.to(device)
            # The stage accuracy has to be measured on the checkpoint that is
            # actually loaded, not taken from another column of the transfer.
            stage = common.evaluate(model, source_data, strata)
            transfer_rows, trained = run_transfer_job(
                f'from_{task}', seed, model, target_data, strata, out_dir,
                steps=steps, eval_every=eval_every, log=log)
            common.write_metrics(out_dir / 'metrics.csv', transfer_rows)
            common.save_model_checkpoint(
                out_dir / 'ckpts' / f'from_{task}_final.pt', trained, steps,
                transfer_rows[-1] if transfer_rows else None,
                extra={'seed': seed, 'task': 'affine', 'init': f'from_{task}',
                       'stage': 'specificity',
                       'global_norm': common.global_norm(trained)})
            final = transfer_rows[-1] if transfer_rows else {}
            record.update(
                transfer_done=True,
                final_dev_mean=common.as_float(final.get('dev_mean')),
                final_dev_worst=common.as_float(final.get('dev_worst')),
                final_train_acc=common.as_float(final.get('train_acc')),
                source_train_acc=stage['train_acc'],
                source_dev_mean=stage['dev_mean'],
                source_dev_worst=stage['dev_worst'],
                stage_global_norm=common.global_norm(model),
                runtime=common.runtime_metadata(device),
                final_ckpt=f'ckpts/from_{task}_final.pt')
            common.job_line(seed, f'from_{task}', steps, steps,
                            f'{final.get("dev_mean", float("nan")):.6f}', 'done')
        else:
            record.update(transfer_done=False, skipped='source did not reach success')
            log(f'  [{job_name}] did not reach success '
                f'({record.get("status")}); transfer skipped')
        jobs[job_name] = record
        common.write_summary(out_dir / 'summary.json', summary)
    summary['finished'] = True
    common.write_summary(out_dir / 'summary.json', summary)
    return summary


def specificity_worker(seed, config):
    """Both alternative sources of one seed (runs inside a worker process)."""
    common.set_batch_log(config.get('batch_log'))
    run_specificity(seed, Path(config['out_root']) / f'seed_{seed}', config['device'],
                    steps=config['steps'], eval_every=config['eval_every'],
                    source_cap=config['source_cap'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--procs', type=int, default=1,
                        help='number of single-thread worker processes')
    parser.add_argument('--out', default='runs_nseed/specificity')
    parser.add_argument('--device', default=None,
                        help='cpu, cuda or auto (default: cuda when available)')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--steps', type=int, default=TRANSFER_STEPS)
    parser.add_argument('--source-cap', type=int, default=common.SOURCE_CAP,
                        help='test-only override of the source step cap')
    parser.add_argument('--eval-every', type=int, default=EVAL_EVERY)
    parser.add_argument('--batch-log', default='runs_nseed/batch.log')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)

    runtime = common.setup(args.threads, args.device)
    device = runtime['device']
    common.set_batch_log(project_path(args.batch_log))
    common.log_line(f'[specificity] device={device} deterministic={runtime["deterministic"]} '
                    f'source_cap={common.SOURCE_CAP} steps={args.steps}')

    out_root = project_path(args.out)
    config = dict(out_root=str(out_root), device=device, steps=args.steps,
                  eval_every=args.eval_every, source_cap=args.source_cap,
                  batch_log=str(common.BATCH_LOG), threads=args.threads,
                  runs_dir=str(common.PARENT / 'runs_nseed'))
    for seed in args.seeds:
        common.planned_line(seed, [f'source_{task}' for task in SOURCE_TASKS])
    report = run_parallel(args.procs, args.seeds, specificity_worker, config)
    if not report.ok:
        common.log_line(f'[specificity] FAILED: {report.summary()}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
