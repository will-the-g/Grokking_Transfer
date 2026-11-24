"""Steps of queue 10: the Pre-LN scratch runs and the Jia random-split probe.

K1  the Pre-LN model (batch 512) trained from scratch on the affine task, with
    the dev mean at 2,000 and 10,000 steps.
K2  the per-stage transfers of the p=59 Pre-LN sources.
K3  the Jia probe with the random split.
K4  the per-stage transfers of the Jia random-split sources.
K5  the cold-recipient factorial with the Jia grokked source as the donor.

    python nseed_backbone\\queue10_steps.py scratch --seeds 5 6 7 8 9
    python nseed_backbone\\queue10_steps.py pack --out results_queue10.zip

Produces: Table 1 (the Pre-LN row of queue 10).
"""
from __future__ import annotations

import argparse
import csv
import io
import statistics
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
HERE = Path(__file__).resolve().parent
SRC = HERE.parent / 'src'
for _path in (str(SRC),):
    if _path not in sys.path:
        sys.path.insert(0, _path)

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402
import torch  # noqa: E402
from queue9_steps import (gpu_name, read_csv, step_cold, step_probe,  # noqa: E402
                          step_stage, version_info, write_csv)

SCRATCH_FIELDS = ['seed', 'dev_mean_2000', 'dev_mean_10000', 'torch_version',
                   'gpu_name']
FOLDERS = ('runs_nseed_k1', 'runs_nseed_k2', 'runs_nseed_k3', 'runs_nseed_k4',
           'runs_nseed_k5')
STEP_TABLES = {'runs_nseed_k1': 'k1_scratch_preln.csv',
               'runs_nseed_k2': 'k2_preln.csv',
               'runs_nseed_k3': 'k3_probe.csv',
               'runs_nseed_k4': 'k4_stage.csv',
               'runs_nseed_k5': 'k5_cold.csv'}


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else common.PARENT / path


def step_scratch(args, device):
    """K1: a random-init model trained on the affine task."""
    version, gpu = version_info()
    eval_at = set(args.eval_at)
    rows = []
    for seed in args.seeds:
        mask = common.mask_for_seed(seed)
        strata, _exposure = common.split_for_mask(mask)
        target_data = common.build_task('affine', device)
        model = common.init_model(seed, device)
        optimizer = common.make_optimizer(model)
        train_rows = common.index_for(strata['train'], device)
        generator = torch.Generator().manual_seed(int(seed) + 7919)
        batches = common.epoch_batches(train_rows, common.BATCH_SIZE, generator)
        batch_index = 0
        values = {}
        step = 0
        started = time.time()
        while step < args.steps:
            if batch_index >= len(batches):
                batches = common.epoch_batches(train_rows, common.BATCH_SIZE,
                                               generator)
                batch_index = 0
            batch = batches[batch_index]
            batch_index += 1
            common.train_step(model, optimizer, target_data, batch)
            step += 1
            if step in eval_at:
                evaluation = common.evaluate(model, target_data, strata)
                values[step] = evaluation['dev_mean']
                common.log_line(f'[queue10] K1 seed {seed} step {step}: dev_mean '
                                f'{evaluation["dev_mean"]:.4f}')
        row = dict(seed=seed, dev_mean_2000=values.get(2000, ''),
                   dev_mean_10000=values.get(10000, ''),
                   torch_version=version, gpu_name=gpu)
        rows.append(row)
        common.log_line(f'[queue10] K1 seed {seed} done in {time.time() - started:.0f} s')
    out_dir = project_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / 'k1_scratch_preln.csv', SCRATCH_FIELDS, rows)
    common.log_line(f'[queue10] K1 wrote {len(rows)} rows to '
                    f'{out_dir / "k1_scratch_preln.csv"}')
    return rows


def step_pack(args):
    """Pack only the new folders and write the report."""
    root = Path(args.root) if args.root else common.PARENT
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    members = []
    tables = {}
    for folder in FOLDERS:
        base = root / folder
        if not base.is_dir():
            continue
        for path in sorted(base.rglob('*.csv')):
            members.append(path)
        table = read_csv(base / STEP_TABLES[folder])
        if table:
            tables[folder] = table
    report = root / 'nseed_backbone' / 'REPORT_job10.md'
    write_report(report, tables)
    if report.exists():
        members.append(report)
    with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in members:
            archive.write(path, path.relative_to(root).as_posix())
    common.log_line(f'[queue10] packed {len(members)} files into {out_path}')
    return out_path


def write_report(path, tables):
    """The K1 means, the per-stage and per-cell means, the probe table."""
    lines = ['# REPORT_job10', '',
             '## K1 scratch Pre-LN (mean over the seeds)',
             '| n_seeds | dev_mean_2000 | dev_mean_10000 |', '|---|---|---|']
    scratch = tables.get('runs_nseed_k1', [])
    if scratch:
        def mean_of(field):
            values = [float(row[field]) for row in scratch
                      if row.get(field) not in (None, '')]
            return statistics.fmean(values) if values else float('nan')
        lines.append(f"| {len(scratch)} | {mean_of('dev_mean_2000'):.4f} "
                     f"| {mean_of('dev_mean_10000'):.4f} |")
    for folder, label in (('runs_nseed_k2', 'K2 p=59 Pre-LN'),
                          ('runs_nseed_k4', 'K4 Jia Pre-LN random split')):
        rows = tables.get(folder, [])
        if not rows:
            continue
        lines += ['', f'## {label} per-stage transfer', '| stage | n_seeds | mean_transfer_dev |',
                  '|---|---|---|']
        for stage in ('memorized', 'transition', 'grokked', 'norm_control'):
            subset = [row for row in rows if row['stage'] == stage]
            if not subset:
                continue
            values = [float(row['transfer_dev_mean']) for row in subset]
            lines.append(f"| {stage} | {len(subset)} | {statistics.fmean(values):.4f} |")
    probe = tables.get('runs_nseed_k3', [])
    if probe:
        lines += ['', '## K3 Jia probe (random split)', '| seed | grokked | grok_step | source_steps | status |',
                  '|---|---|---|---|---|']
        for row in probe:
            lines.append(f"| {row['seed']} | {row['grokked']} | {row['grok_step']} "
                         f"| {row['source_steps']} | {row['status']} |")
    cold = tables.get('runs_nseed_k5', [])
    if cold:
        lines += ['', '## K5 cold-recipient means per cell',
                  '| cell | scaling | n_seeds | mean_final_dev |', '|---|---|---|---|']
        for cell in sorted({row['cell'] for row in cold}):
            for scaling in sorted({row['scaling'] for row in cold if row['cell'] == cell}):
                subset = [row for row in cold
                          if row['cell'] == cell and row['scaling'] == scaling]
                values = [float(row['final_dev_mean']) for row in subset
                          if row.get('final_dev_mean') not in (None, '')]
                if not values:
                    continue
                lines.append(f"| {cell} | {scaling} | {len(subset)} "
                             f"| {statistics.fmean(values):.4f} |")
    Path(path).write_text('\n'.join(lines), encoding='utf-8')
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['scratch', 'stage', 'probe', 'cold', 'pack'])
    parser.add_argument('--name', default='stage')
    parser.add_argument('--runs-root', default='runs_nseed')
    parser.add_argument('--source-csv', default='runs_nseed_k5/embed_ablation.csv')
    parser.add_argument('--seeds', nargs='+', type=int, default=[])
    parser.add_argument('--out', default='runs_nseed_k1')
    parser.add_argument('--csv', default=None)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--eval-at', nargs='+', type=int, default=[2000, 10000])
    parser.add_argument('--procs', type=int, default=1)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default=None)
    parser.add_argument('--root', default=None)
    parser.add_argument('--batch-log', default='runs_nseed/batch.log')
    args = parser.parse_args(argv)

    runtime = common.setup(args.threads, args.device)
    device = runtime['device']
    if args.csv is None:
        args.csv = {'scratch': 'k1_scratch_preln.csv', 'probe': 'k3_probe.csv',
                    'stage': 'stage.csv', 'cold': 'k5_cold.csv'}.get(args.step, 'stage.csv')
    common.set_batch_log(project_path(args.batch_log))
    common.log_line(f'[queue10] step {args.step} device={device} torch {torch.__version__} '
                    f'gpu {gpu_name()}')

    if args.step == 'scratch':
        step_scratch(args, device)
    elif args.step == 'stage':
        step_stage(args)
    elif args.step == 'probe':
        step_probe(args)
    elif args.step == 'cold':
        step_cold(args)
    elif args.step == 'pack':
        step_pack(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
