"""Steps of queue 9: the forced recompute, the Jia probe and the Pre-LN runs.

P0  three p=97 grokked-source affine transfers rerun from the stored
    checkpoints into a fresh output directory, next to the stored value.
J0  the Jia probe (Pre-LN, batch 512): which seeds grok.
P1  the p=59 setting with Pre-LN and batch 512: the per-stage transfers.
J1  the same for the Jia setting, only when the probe grokked.
J2  the cold-recipient factorial with the Jia grokked source as the donor.

    python nseed_backbone\\queue9_steps.py p0 --seeds 5 6 7
    python nseed_backbone\\queue9_steps.py probe --runs-root runs_nseed_j0
    python nseed_backbone\\queue9_steps.py stage --runs-root runs_nseed_p1
    python nseed_backbone\\queue9_steps.py pack --out results_queue9.zip

Produces: Table 1 (the Pre-LN row of queue 9).
"""
from __future__ import annotations

import argparse
import csv
import io
import statistics
import sys
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
from run_components import run_transfer_job  # noqa: E402

TRANSFER_STEPS = 2000
STAGES = ('memorized', 'transition', 'grokked', 'norm_control')
P0_FIELDS = ['seed', 'stage', 'stored_dev_mean', 'new_dev_mean', 'difference',
              'torch_version', 'gpu_name']
PROBE_FIELDS = ['seed', 'grokked', 'grok_step', 'source_steps', 'status',
                'torch_version', 'gpu_name']
STAGE_FIELDS = ['seed', 'stage', 'source_step', 'source_dev_mean',
                'transfer_dev_mean', 'torch_version', 'gpu_name']
COLD_FIELDS = ['seed', 'cell', 'scaling', 'source_dev_mean', 'final_dev_mean',
                'torch_version', 'gpu_name']
FOLDERS = ('runs_nseed_p0', 'runs_nseed_j0', 'runs_nseed_p1', 'runs_nseed_j1',
           'runs_nseed_j2')
STEP_TABLES = {'runs_nseed_p0': 'p0_recompute.csv', 'runs_nseed_j0': 'j0_probe.csv',
               'runs_nseed_p1': 'p1_preln.csv', 'runs_nseed_j1': 'j1_stage.csv',
               'runs_nseed_j2': 'j2_cold.csv'}


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else common.PARENT / path


def write_csv(path, fields, rows):
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    common.atomic_bytes(path, out.getvalue().encode())
    return path


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open('r', newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def gpu_name():
    try:
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    except Exception:  # noqa: BLE001 - the name is only metadata
        return 'unknown'


def version_info():
    return torch.__version__, gpu_name()


def step_p0(args, device):
    """The forced recompute of three p=97 grokked-source transfers."""
    version, gpu = version_info()
    rows = []
    for seed in args.seeds:
        summary = common.read_summary(Path(args.runs_root) / f'seed_{seed}' / 'summary.json')
        stages = summary.get('stages', {}) or {}
        transfers = summary.get('transfers', {}) or {}
        if 'grokked' not in stages:
            common.log_line(f'[queue9] P0 seed {seed}: no grokked stage, skipped')
            continue
        stored = (transfers.get('affine|grokked', {}) or {}).get('final_dev_mean')
        checkpoint = Path(args.runs_root) / f'seed_{seed}' / 'ckpts' / stages['grokked']['ckpt']
        model = common.make_model(common.CONFIG)
        model.load_state_dict(common.load_checkpoint(checkpoint)['model'])
        model = model.to(device)
        model.eval()
        mask = common.mask_for_seed(seed)
        strata, _exposure = common.split_for_mask(mask)
        target_data = common.build_task('affine', device)
        transfer_rows, _trained = run_transfer_job(
            f'p0_grokked_seed{seed}', seed, model, target_data, strata, Path('.'),
            steps=args.steps, eval_every=100, log=common.log_line)
        final = transfer_rows[-1] if transfer_rows else {}
        new = common.as_float(final.get('dev_mean'))
        difference = (new - stored) if (new is not None and stored is not None) else None
        rows.append(dict(seed=seed, stage='grokked', stored_dev_mean=stored,
                         new_dev_mean=new, difference=difference,
                         torch_version=version, gpu_name=gpu))
        common.log_line(f'[queue9] P0 seed {seed}: stored {stored} new {new} '
                        f'difference {difference}')
    out_dir = project_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / 'p0_recompute.csv', P0_FIELDS, rows)
    common.log_line(f'[queue9] P0 wrote {len(rows)} rows to {out_dir / "p0_recompute.csv"}')
    return rows


def step_probe(args):
    """Which seeds grokked."""
    version, gpu = version_info()
    rows = []
    for seed in args.seeds:
        summary = common.read_summary(Path(args.runs_root) / f'seed_{seed}' / 'summary.json')
        status = summary.get('status')
        grok_step = summary.get('grok_step')
        rows.append(dict(seed=seed, grokked='yes' if status == 'grokked' else 'no',
                         grok_step=grok_step if grok_step is not None else '',
                         source_steps=summary.get('source_steps_run', ''),
                         status=status or '', torch_version=version, gpu_name=gpu))
    out_dir = project_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / args.csv, PROBE_FIELDS, rows)
    common.log_line(f'[queue9] J0 wrote {len(rows)} rows to {out_dir / args.csv}')
    return rows


def step_stage(args):
    """The per-stage transfers of a set of runs."""
    version, gpu = version_info()
    rows = []
    for seed in args.seeds:
        summary = common.read_summary(Path(args.runs_root) / f'seed_{seed}' / 'summary.json')
        stages = summary.get('stages', {}) or {}
        transfers = summary.get('transfers', {}) or {}
        if not stages:
            common.log_line(f'[queue9] {args.name} seed {seed}: no stages (no grok), skipped')
            continue
        for stage in STAGES:
            source = 'memorized' if stage == 'norm_control' else stage
            if source not in stages:
                continue
            value = (transfers.get(f'affine|{stage}', {}) or {}).get('final_dev_mean')
            if value is None:
                continue
            rows.append(dict(seed=seed, stage=stage,
                             source_step=(stages.get(source, {}) or {}).get('step', ''),
                             source_dev_mean=(stages.get(source, {}) or {}).get('dev_mean', ''),
                             transfer_dev_mean=value, torch_version=version, gpu_name=gpu))
    out_dir = project_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / args.csv, STAGE_FIELDS, rows)
    common.log_line(f'[queue9] {args.name} wrote {len(rows)} rows to {out_dir / args.csv}')
    return rows


def step_cold(args):
    """The cold-recipient factorial of the Jia runs."""
    version, gpu = version_info()
    rows = []
    for row in read_csv(args.source_csv):
        rows.append(dict(seed=row['seed'], cell=row['cell'], scaling=row['scaling'],
                         source_dev_mean=row.get('source_dev_mean', ''),
                         final_dev_mean=row.get('final_dev_mean_10000',
                                                row.get('final_dev_mean', '')),
                         torch_version=version, gpu_name=gpu))
    out_dir = project_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / 'j2_cold.csv', COLD_FIELDS, rows)
    common.log_line(f'[queue9] J2 wrote {len(rows)} rows to {out_dir / "j2_cold.csv"}')
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
    report = root / 'nseed_backbone' / 'REPORT_job9.md'
    write_report(report, tables)
    if report.exists():
        members.append(report)
    with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in members:
            archive.write(path, path.relative_to(root).as_posix())
    common.log_line(f'[queue9] packed {len(members)} files into {out_path}')
    return out_path


def write_report(path, tables):
    """The P0 differences, the probe table and the per-stage and per-cell means."""
    lines = ['# REPORT_job9', '',
             '## P0 forced recompute (p=97 grokked-source transfers)',
             '| seed | stored | new | difference | torch | gpu |', '|---|---|---|---|---|---|']
    for row in tables.get('runs_nseed_p0', []):
        lines.append(f"| {row['seed']} | {row['stored_dev_mean']} | {row['new_dev_mean']} "
                     f"| {row['difference']} | {row['torch_version']} | {row['gpu_name']} |")
    lines += ['', '## J0 grok probe', '| seed | grokked | grok_step | source_steps | status |',
              '|---|---|---|---|---|']
    for row in tables.get('runs_nseed_j0', []):
        lines.append(f"| {row['seed']} | {row['grokked']} | {row['grok_step']} "
                     f"| {row['source_steps']} | {row['status']} |")
    for folder, label in (('runs_nseed_p1', 'P1 p=59 Pre-LN'),
                          ('runs_nseed_j1', 'J1 Jia Pre-LN')):
        rows = tables.get(folder, [])
        if not rows:
            continue
        lines += ['', f'## {label} per-stage transfer', '| stage | n_seeds | mean_transfer_dev |',
                  '|---|---|---|']
        for stage in STAGES:
            subset = [row for row in rows if row['stage'] == stage]
            if not subset:
                continue
            values = [float(row['transfer_dev_mean']) for row in subset]
            lines.append(f"| {stage} | {len(subset)} | {statistics.fmean(values):.4f} |")
    cold = tables.get('runs_nseed_j2', [])
    if cold:
        lines += ['', '## J2 cold-recipient means per cell',
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
    parser.add_argument('step', choices=['p0', 'probe', 'stage', 'cold', 'pack'])
    parser.add_argument('--name', default='stage')
    parser.add_argument('--runs-root', default='runs_nseed')
    parser.add_argument('--source-csv', default='runs_nseed_j2/embed_ablation.csv')
    parser.add_argument('--seeds', nargs='+', type=int, default=[])
    parser.add_argument('--out', default='runs_nseed_p0')
    parser.add_argument('--csv', default=None)
    parser.add_argument('--steps', type=int, default=TRANSFER_STEPS)
    parser.add_argument('--procs', type=int, default=1)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default=None)
    parser.add_argument('--root', default=None)
    parser.add_argument('--batch-log', default='runs_nseed/batch.log')
    args = parser.parse_args(argv)

    runtime = common.setup(args.threads, args.device)
    device = runtime['device']
    if args.csv is None:
        args.csv = {'p0': 'p0_recompute.csv', 'probe': 'j0_probe.csv',
                     'stage': 'stage.csv', 'cold': 'j2_cold.csv'}.get(args.step, 'stage.csv')
    common.set_batch_log(project_path(args.batch_log))
    common.log_line(f'[queue9] step {args.step} device={device} torch {torch.__version__} '
                    f'gpu {gpu_name()}')

    if args.step == 'p0':
        step_p0(args, device)
    elif args.step == 'probe':
        step_probe(args)
    elif args.step == 'stage':
        step_stage(args)
    elif args.step == 'cold':
        step_cold(args)
    elif args.step == 'pack':
        step_pack(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
