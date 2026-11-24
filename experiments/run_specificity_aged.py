"""Age-matched specificity: does the age of the source decide the transfer?

For every seed the copy_a and selector sources are trained from scratch with
two stage points:

  (i)  age matched - exactly as many steps as that seed's addition source
       needed to grok, so the comparison is against a source of the same age;
  (ii) own success  - until the source itself reaches train >= 0.99 and
       worst-context clean dev >= 0.90 (the spec's success rule, cap 20,000).

A 2,000-step affine transfer is then run from each stage point. The recorded
source step, source train accuracy, source clean-dev accuracy and global norm
are the ones measured at the checkpoint that is actually loaded; nothing looks
at the transfer results while the source is being trained.

Output: runs_nseed/specificity_aged/seed_<S>_<task>/.

    python nseed_backbone\\run_specificity_aged.py --seeds 5 6 7 8 9 10 --device cuda

Produces: Figure 2 (the aged copy and selector sources).
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
SOURCE_CAP = 20000
AGE_CHECKPOINT = 'age_matched.pt'


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else common.PARENT / path


def addition_grok_step(seed):
    """The step at which the seed's addition source grokked, or None."""
    summary = common.read_summary(
        common.PARENT / 'runs_nseed' / f'seed_{seed}' / 'summary.json')
    if summary.get('status') != 'grokked':
        return None
    step = summary.get('grok_step')
    return int(step) if step is not None else None


def run_seed_aged(seed, out_root, device, steps=TRANSFER_STEPS, eval_every=EVAL_EVERY,
                  source_cap=SOURCE_CAP, resume=False, runtime=None,
                  log=common.log_line, tasks=SOURCE_TASKS, init_seed=None):
    """Both alternative sources of one seed, with both stage points."""
    runtime = runtime or common.runtime_metadata(device)
    age_step = addition_grok_step(seed)
    if age_step is None:
        log(f'[specificity_aged] seed {seed}: the addition source never grokked, skipped')
        return None
    log(f'[specificity_aged] seed {seed}: age-matched step {age_step}')

    mask = common.mask_for_seed(seed)
    strata, _ = common.split_for_mask(mask)
    target_data = common.build_task('affine', device)
    summary = {'seed': seed, 'age_step': age_step, 'jobs': {}, 'runtime': runtime,
               'init_seed': int(seed) if init_seed is None else int(init_seed)}

    for task in tasks:
        out_dir = Path(out_root) / f'seed_{seed}_{task}'
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'ckpts').mkdir(parents=True, exist_ok=True)
        source_data = common.build_task(task, device)
        run = SeedRun(seed, out_dir, threads=1, resume=False, tiers=(),
                      device=device, source_cap=source_cap, log=log,
                      init_seed=init_seed)
        run._ensure_dirs()
        job_name = f'source_{task}'
        started = time.time()
        existing = common.read_summary(out_dir / 'summary.json')
        already_done = (existing.get('status') in ('grokked', 'nongrok')
                        and (out_dir / 'ckpts' / AGE_CHECKPOINT).exists())
        if already_done:
            status = existing['status']
            grok_step = existing.get('grok_step')
            log(f'  [{job_name}] already trained (status {status}), reusing the checkpoints')
            common.job_line(seed, job_name, grok_step or source_cap, source_cap, 'nan',
                            'source_done', elapsed=time.time() - started)
        else:
            common.job_line(seed, job_name, 0, source_cap, 'nan', 'start',
                            elapsed=time.time() - started)
            status, grok_step, rows = run.train_source(
                strata, source_data, run._summary(),
                snapshot_steps={age_step: AGE_CHECKPOINT},
                keep_training_to=age_step)
            common.job_line(seed, job_name, grok_step or source_cap, source_cap, 'nan',
                            'source_done', elapsed=time.time() - started)
        record = {'name': job_name, 'kind': 'specificity_aged_source', 'task': task,
                  'seed': seed, 'source_task': task, 'status': status,
                  'grok_step': grok_step, 'age_step': age_step,
                  'elapsed_s': time.time() - started, 'done': True,
                  'runtime': runtime}
        stages = {}
        for label, ckpt in (('aged', AGE_CHECKPOINT),
                            ('grok', 'grokked.pt' if status == 'grokked' else None)):
            if ckpt is None or not (out_dir / 'ckpts' / ckpt).exists():
                continue
            model = common.make_model(common.CONFIG)
            model.load_state_dict(common.load_checkpoint(
                out_dir / 'ckpts' / ckpt)['model'])
            model = model.to(device)
            stage = common.evaluate(model, source_data, strata)
            transfer_name = f'from_{task}_{label}'
            transfer_rows, trained = run_transfer_job(
                transfer_name, seed, model, target_data, strata, out_dir,
                steps=steps, eval_every=eval_every, log=log)
            common.write_metrics(out_dir / 'metrics.csv',
                                 common.merge_rows(
                                     common.read_metrics(out_dir / 'metrics.csv'),
                                     transfer_rows))
            common.save_model_checkpoint(
                out_dir / 'ckpts' / f'{transfer_name}_final.pt', trained, steps,
                transfer_rows[-1] if transfer_rows else None,
                extra={'seed': seed, 'task': 'affine', 'init': transfer_name,
                       'stage': 'specificity_aged',
                       'global_norm': common.global_norm(trained)})
            final = transfer_rows[-1] if transfer_rows else {}
            stages[label] = {
                'checkpoint': ckpt,
                'source_step': age_step if label == 'aged' else grok_step,
                'source_train_acc': stage['train_acc'],
                'source_dev_mean': stage['dev_mean'],
                'source_dev_worst': stage['dev_worst'],
                'source_global_norm': common.global_norm(model),
                'final_dev_mean': common.as_float(final.get('dev_mean')),
                'final_dev_worst': common.as_float(final.get('dev_worst')),
                'final_train_acc': common.as_float(final.get('train_acc')),
                'steps_to_30': next((int(r['step']) for r in
                                     sorted(transfer_rows, key=lambda r: int(r['step']))
                                     if (common.as_float(r['dev_mean']) or 0.0) >= 0.30), None),
                'final_ckpt': f'ckpts/{transfer_name}_final.pt',
                'runtime': runtime,
            }
            common.job_line(seed, transfer_name, steps, steps,
                            f'{final.get("dev_mean", float("nan")):.6f}', 'done',
                            elapsed=time.time() - started)
        record['stages'] = stages
        summary['jobs'][job_name] = record
        merged = common.read_summary(out_dir / 'summary.json')
        merged['jobs'] = summary['jobs']
        merged['age_step'] = age_step
        merged['runtime'] = runtime
        common.write_summary(out_dir / 'summary.json', merged)
    common.write_summary(Path(out_root) / f'seed_{seed}_summary.json', summary)
    return summary


def aged_worker(seed, config):
    """One seed (runs inside a worker process)."""
    common.set_batch_log(config.get('batch_log'))
    init_seed = config.get('init_seed')
    if init_seed is not None:
        init_seed = int(init_seed) + int(seed)
    run_seed_aged(seed, config['out_root'], config['device'],
                  steps=config['steps'], eval_every=config['eval_every'],
                  source_cap=config['source_cap'], resume=bool(config.get('resume')),
                  runtime=common.runtime_metadata(config['device']),
                  log=common.log_line,
                  tasks=tuple(config.get('tasks', SOURCE_TASKS)),
                  init_seed=init_seed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--procs', type=int, default=1)
    parser.add_argument('--out', default='runs_nseed/specificity_aged')
    parser.add_argument('--device', default=None,
                        help='cpu, cuda or auto (default: cuda when available)')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--steps', type=int, default=TRANSFER_STEPS)
    parser.add_argument('--eval-every', type=int, default=EVAL_EVERY)
    parser.add_argument('--source-cap', type=int, default=SOURCE_CAP)
    parser.add_argument('--batch-log', default='runs_nseed/batch.log')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--tasks', nargs='+', default=list(SOURCE_TASKS),
                        choices=list(SOURCE_TASKS),
                        help='which alternative sources to train')
    parser.add_argument('--init-seed-offset', type=int, default=0,
                        help='the initialization seed of every source is the seed plus '
                             'this offset (the training mask always uses the seed)')
    args = parser.parse_args(argv)

    runtime = common.setup(args.threads, args.device)
    common.set_batch_log(project_path(args.batch_log))
    common.log_line(f'[specificity_aged] device={runtime["device"]} '
                    f'deterministic={runtime["deterministic"]} source_cap={args.source_cap}')

    out_root = project_path(args.out)
    config = dict(out_root=str(out_root), device=runtime['device'], steps=args.steps,
                  eval_every=args.eval_every, source_cap=args.source_cap,
                  resume=args.resume, batch_log=str(common.BATCH_LOG),
                  threads=args.threads,
                  tasks=tuple(args.tasks),
                  init_seed=(int(args.init_seed_offset)
                             if int(args.init_seed_offset) else None),
                  runs_dir=str(common.PARENT / 'runs_nseed'))
    for seed in args.seeds:
        common.planned_line(seed, [f'source_{task}' for task in args.tasks])
    report = run_parallel(args.procs, args.seeds, aged_worker, config)
    if not report.ok:
        common.log_line(f'[specificity_aged] FAILED: {report.summary()}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
