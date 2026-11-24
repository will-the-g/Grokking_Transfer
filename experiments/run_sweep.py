"""Dense sweep of the saved source checkpoints.

For every source checkpoint of a seed (`runs_nseed/seed_<S>/ckpts/step_*.pt`)
the standard affine transfer is run: a fresh AdamW, the same masks, 2,000 steps
and a dev evaluation every 100 steps. The source checkpoint itself is
evaluated on the source task, the transfer on the affine task.

No transfer checkpoint is written. The result is one row per (seed,
source_step) in runs_nseed/sweep/sweep_summary.csv. A checkpoint whose result
is already in the table is skipped, so the sweep can be re-run.

    python nseed_backbone\\run_sweep.py --seeds 5 6 7 8 9 10 11 12 13 14 --procs 4 --device cuda

Produces: Figure 1 (the three checkpoint sweeps).
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
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

TRANSFER_STEPS = 2000
EVAL_EVERY = 100
SWEEP_FIELDS = ['seed', 'source_step', 'source_train_acc', 'source_dev_mean',
                'source_global_norm', 'final_dev_mean', 'final_dev_worst',
                'steps_to_30pct']


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else common.PARENT / path


def source_checkpoints(runs_root, seed):
    """The saved source checkpoints of one seed, in step order."""
    ckpt_dir = Path(runs_root) / f'seed_{seed}' / 'ckpts'
    if not ckpt_dir.is_dir():
        return []
    found = []
    for path in ckpt_dir.glob('step_*.pt'):
        try:
            step = int(path.stem.split('_')[1])
        except (IndexError, ValueError):
            continue
        found.append((step, path))
    return sorted(found)


def read_rows(path):
    """The rows of a sweep CSV (empty list when the file is missing)."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open('r', newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def append_row(path, row):
    """Append one row to the shard of a seed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=SWEEP_FIELDS, extrasaction='ignore')
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def merge_shards(out_root, path):
    """Every shard row of the sweep, sorted by seed and source step."""
    rows = []
    for shard in sorted((Path(out_root) / 'parts').glob('seed_*.csv')):
        rows.extend(read_rows(shard))
    rows.sort(key=lambda row: (int(row['seed']), int(row['source_step'])))
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=SWEEP_FIELDS, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    common.atomic_bytes(path, out.getvalue().encode())
    return rows


def steps_to_30(rows):
    """The first step whose dev mean reaches 0.30."""
    for row in sorted(rows, key=lambda r: int(r['step'])):
        if (common.as_float(row.get('dev_mean')) or 0.0) >= 0.30:
            return int(row['step'])
    return None


def sweep_worker(seed, config):
    """Every source checkpoint of one seed (runs inside a worker process)."""
    common.set_batch_log(config.get('batch_log'))
    device = config['device']
    mask = common.mask_for_seed(seed)
    strata, _ = common.split_for_mask(mask)
    source_data = common.build_task('source', device)
    target_data = common.build_task('affine', device)
    shard = Path(config['out_root']) / 'parts' / f'seed_{seed}.csv'
    done = {int(row['source_step']): row for row in read_rows(shard)}
    for step, path in source_checkpoints(config['runs_root'], seed):
        job_name = f'sweep_{step}'
        if step in done:
            common.job_line(seed, job_name, config['steps'], config['steps'],
                            done[step].get('final_dev_mean', 'nan'), 'done')
            continue
        payload = common.load_checkpoint(path)
        model = common.make_model(common.CONFIG)
        model.load_state_dict(payload['model'])
        model = model.to(device)
        evaluation = common.evaluate(model, source_data, strata)
        rows, _trained = run_transfer_job(
            job_name, seed, model, target_data, strata, Path(config['out_root']),
            steps=config['steps'], eval_every=config['eval_every'],
            log=common.log_line)
        final = rows[-1] if rows else {}
        row = dict(seed=seed, source_step=step,
                   source_train_acc=evaluation['train_acc'],
                   source_dev_mean=evaluation['dev_mean'],
                   source_global_norm=common.global_norm(model),
                   final_dev_mean=final.get('dev_mean'),
                   final_dev_worst=final.get('dev_worst'),
                   steps_to_30pct=steps_to_30(rows))
        append_row(shard, row)
        common.job_line(seed, job_name, config['steps'], config['steps'],
                        f'{final.get("dev_mean", float("nan")):.6f}', 'done')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(5, 15)))
    parser.add_argument('--procs', type=int, default=1)
    parser.add_argument('--out', default='runs_nseed/sweep')
    parser.add_argument('--runs-root', default='runs_nseed')
    parser.add_argument('--device', default=None,
                        help='cpu, cuda or auto (default: cuda when available)')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--steps', type=int, default=TRANSFER_STEPS)
    parser.add_argument('--eval-every', type=int, default=EVAL_EVERY)
    parser.add_argument('--batch-log', default='runs_nseed/batch.log')
    args = parser.parse_args(argv)

    runtime = common.setup(args.threads, args.device)
    common.set_batch_log(project_path(args.batch_log))
    common.log_line(f'[sweep] device={runtime["device"]} deterministic={runtime["deterministic"]} '
                    f'steps={args.steps} eval_every={args.eval_every}')

    out_root = project_path(args.out)
    runs_root = project_path(args.runs_root)
    (out_root / 'parts').mkdir(parents=True, exist_ok=True)
    config = dict(out_root=str(out_root), runs_root=str(runs_root),
                  device=runtime['device'], steps=args.steps,
                  eval_every=args.eval_every, batch_log=str(common.BATCH_LOG),
                  threads=args.threads,
                  runs_dir=str(common.PARENT / 'runs_nseed'))
    for seed in args.seeds:
        common.planned_line(seed, [f'sweep_{step}' for step, _ in
                                   source_checkpoints(runs_root, seed)])
    report = run_parallel(args.procs, args.seeds, sweep_worker, config)
    rows = merge_shards(out_root, out_root / 'sweep_summary.csv')
    common.log_line(f'[sweep] wrote {len(rows)} rows to {out_root / "sweep_summary.csv"}')
    if not report.ok:
        common.log_line(f'[sweep] FAILED: {report.summary()}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
