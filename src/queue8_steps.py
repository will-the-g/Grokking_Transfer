"""Steps of queue 8: the G2 reruns and the G1 Jia setting.

G2  the standard 2,000-step affine transfer is rerun from the stored source
    checkpoints (memorized, transition, grokked and the norm-matched
    memorized control) for p=59, p=97 and the two-layer sources; the earlier
    values come from the stored summaries, so every row holds the earlier
    value, the rerun value, their difference, the torch version and the GPU.

G1  the Jia setting (modulus 113, a fifth of the pairs, one context, the
    affine rule of context 1): the per-stage transfers and the cold-recipient
    factorial are read from the runs of the queue and written as
    g1_stage.csv and g1_cold.csv.

    python nseed_backbone\\queue8_steps.py g2 --family p59 --runs-root runs_nseed
    python nseed_backbone\\queue8_steps.py g1_stage --runs-root runs_nseed_jia
    python nseed_backbone\\queue8_steps.py pack --out results_queue8.zip

Produces: the stage checkpoints of queue 8.
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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402
import torch  # noqa: E402
from run_components import run_transfer_job  # noqa: E402

TRANSFER_STEPS = 2000
STAGES = ('memorized', 'transition', 'grokked', 'norm_control')
G2_FIELDS = ['family', 'seed', 'stage', 'earlier_dev_mean', 'rerun_dev_mean',
             'difference', 'torch_version', 'gpu_name']
G1_STAGE_FIELDS = ['seed', 'stage', 'source_step', 'source_dev_mean',
                   'transfer_dev_mean', 'torch_version', 'gpu_name']
G1_COLD_FIELDS = ['seed', 'cell', 'scaling', 'source_dev_mean', 'final_dev_mean',
                   'torch_version', 'gpu_name']
FOLDERS = ('runs_nseed_g2', 'runs_nseed_jia', 'runs_nseed_jia_cold')


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


def stage_checkpoint(runs_root, seed, stage, stages, grokked_norm):
    """The state of one stage, with the norm control rescaled."""
    folder = Path(runs_root) / f'seed_{seed}' / 'ckpts'
    if stage == 'norm_control':
        source = folder / stages['memorized']['ckpt']
        payload = common.load_checkpoint(source)
        model = common.make_model(common.CONFIG)
        model.load_state_dict(payload['model'])
        common.rescale_to_norm(model, grokked_norm)
        return {key: value.detach().cpu() for key, value in model.state_dict().items()}
    path = folder / stages[stage]['ckpt']
    return common.load_checkpoint(path)['model']


def g2_rows(args, device, done=()):
    """The reruns of the stored stages for one family."""
    version, gpu = version_info()
    rows = []
    for seed in args.seeds:
        summary = common.read_summary(Path(args.runs_root) / f'seed_{seed}' / 'summary.json')
        stages = summary.get('stages', {}) or {}
        transfers = summary.get('transfers', {}) or {}
        if not stages:
            common.log_line(f'[queue8] G2 seed {seed}: no stages, skipped')
            continue
        grokked_norm = (stages.get('grokked', {}) or {}).get('global_norm')
        for stage in STAGES:
            if (str(seed), stage) in done:
                common.log_line(f'[queue8] G2 seed {seed} {stage}: already done')
                continue
            if stage != 'norm_control' and stage not in stages:
                continue
            if stage == 'norm_control' and (grokked_norm is None or 'memorized' not in stages):
                continue
            earlier = (transfers.get(f'affine|{stage}', {}) or {}).get('final_dev_mean')
            if earlier is None:
                common.log_line(f'[queue8] G2 seed {seed} {stage}: no earlier value, skipped')
                continue
            state = stage_checkpoint(args.runs_root, seed, stage, stages, grokked_norm)
            model = common.make_model(common.CONFIG)
            model.load_state_dict(state)
            model = model.to(device)
            model.eval()
            mask = common.mask_for_seed(seed)
            strata, _exposure = common.split_for_mask(mask)
            target_data = common.build_task('affine', device)
            transfer_rows, _trained = run_transfer_job(
                f'g2_{stage}_seed{seed}', seed, model, target_data, strata, Path('.'),
                steps=args.steps, eval_every=100, log=common.log_line)
            final = transfer_rows[-1] if transfer_rows else {}
            rerun = common.as_float(final.get('dev_mean'))
            difference = (rerun - earlier) if (rerun is not None and earlier is not None) else None
            rows.append(dict(family=args.family, seed=seed, stage=stage,
                             earlier_dev_mean=earlier, rerun_dev_mean=rerun,
                             difference=difference, torch_version=version, gpu_name=gpu))
            common.log_line(f'[queue8] G2 {args.family} seed {seed} {stage}: earlier '
                            f'{earlier} rerun {rerun} difference {difference}')
    return rows


def step_g2(args, device):
    out_dir = project_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / args.csv
    done = {(row['seed'], row['stage']) for row in read_csv(path)}
    rows = g2_rows(args, device, done)
    merged = read_csv(path) + rows
    merged.sort(key=lambda row: (int(row['seed']), str(row['stage'])))
    write_csv(path, G2_FIELDS, merged)
    common.log_line(f'[queue8] G2 {args.family}: wrote {len(merged)} rows to {path}')
    return merged


def step_g1_stage(args):
    """The per-stage transfers of the Jia runs, read from their summaries."""
    version, gpu = version_info()
    rows = []
    for seed in args.seeds:
        summary = common.read_summary(Path(args.runs_root) / f'seed_{seed}' / 'summary.json')
        stages = summary.get('stages', {}) or {}
        transfers = summary.get('transfers', {}) or {}
        if not stages:
            common.log_line(f'[queue8] G1 seed {seed}: no stages (no grok), skipped')
            continue
        for stage in STAGES:
            if stage == 'norm_control':
                source = 'memorized'
            else:
                source = stage
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
    write_csv(out_dir / 'g1_stage.csv', G1_STAGE_FIELDS, rows)
    common.log_line(f'[queue8] G1 wrote {len(rows)} stage rows to {out_dir / "g1_stage.csv"}')
    return rows


def step_g1_cold(args):
    """The cold-recipient factorial of the Jia runs, read from its CSV."""
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
    write_csv(out_dir / 'g1_cold.csv', G1_COLD_FIELDS, rows)
    common.log_line(f'[queue8] G1 wrote {len(rows)} cold rows to {out_dir / "g1_cold.csv"}')
    return rows


def step_pack(args):
    """Pack only the new folders and write the report."""
    root = Path(args.root) if args.root else common.PARENT
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    members = []
    g2 = []
    for path in sorted((root / 'runs_nseed_g2').glob('*.csv')):
        members.append(path)
        g2.extend(read_csv(path))
    stage_rows = []
    cold_rows = []
    for folder, name in (('runs_nseed_jia', 'g1_stage.csv'),
                         ('runs_nseed_jia_cold', 'g1_cold.csv')):
        path = root / folder / name
        if path.exists():
            members.append(path)
            stage_rows.extend(read_csv(path)) if name.startswith('g1_stage') \
                else cold_rows.extend(read_csv(path))
    report = root / 'nseed_backbone' / 'REPORT_job8.md'
    write_report(report, g2, stage_rows, cold_rows)
    if report.exists():
        members.append(report)
    with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in members:
            archive.write(path, path.relative_to(root).as_posix())
    common.log_line(f'[queue8] packed {len(members)} files into {out_path}')
    return out_path


def write_report(path, g2, stage_rows, cold_rows):
    """The G2 differences and the G1 means, numbers only."""
    lines = ['# REPORT_job8', '',
             '## G2 earlier vs rerun (absolute difference)',
             '| family | stage | n | mean_abs_diff | max_abs_diff |',
             '|---|---|---|---|---|']
    for family in sorted({row['family'] for row in g2}):
        for stage in STAGES:
            subset = [row for row in g2
                      if row['family'] == family and row['stage'] == stage]
            if not subset:
                continue
            values = [abs(float(row['difference'])) for row in subset
                      if row.get('difference') not in (None, '')]
            lines.append(f"| {family} | {stage} | {len(subset)} "
                         f"| {statistics.fmean(values):.4f} | {max(values):.4f} |")
    differences = [abs(float(row['difference'])) for row in g2
                   if row.get('difference') not in (None, '')]
    if differences:
        lines.append(f"| all | all | {len(differences)} | {statistics.fmean(differences):.4f} "
                     f"| {max(differences):.4f} |")
    lines += ['', '## G1 per-stage transfer (mean over the seeds)',
              '| stage | n_seeds | mean_transfer_dev |', '|---|---|---|']
    for stage in STAGES:
        subset = [row for row in stage_rows if row['stage'] == stage]
        if not subset:
            continue
        values = [float(row['transfer_dev_mean']) for row in subset]
        lines.append(f"| {stage} | {len(subset)} | {statistics.fmean(values):.4f} |")
    lines += ['', '## G1 cold-recipient means per cell',
              '| cell | scaling | n_seeds | mean_final_dev |',
              '|---|---|---|---|']
    for cell in sorted({row['cell'] for row in cold_rows}):
        for scaling in sorted({row['scaling'] for row in cold_rows
                               if row['cell'] == cell}):
            subset = [row for row in cold_rows
                      if row['cell'] == cell and row['scaling'] == scaling]
            if not subset:
                continue
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
    parser.add_argument('step', choices=['g2', 'g1_stage', 'g1_cold', 'pack'])
    parser.add_argument('--family', default='p59')
    parser.add_argument('--runs-root', default='runs_nseed')
    parser.add_argument('--source-csv', default='runs_nseed_jia_cold/embed_ablation.csv')
    parser.add_argument('--seeds', nargs='+', type=int, default=[])
    parser.add_argument('--out', default='runs_nseed_g2')
    parser.add_argument('--csv', default='g2_headline.csv')
    parser.add_argument('--steps', type=int, default=TRANSFER_STEPS)
    parser.add_argument('--procs', type=int, default=1)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default=None)
    parser.add_argument('--root', default=None)
    parser.add_argument('--batch-log', default='runs_nseed/batch.log')
    args = parser.parse_args(argv)

    runtime = common.setup(args.threads, args.device)
    device = runtime['device']
    common.set_batch_log(project_path(args.batch_log))
    common.log_line(f'[queue8] step {args.step} device={device} torch {torch.__version__} '
                    f'gpu {gpu_name()}')

    if args.step == 'g2':
        step_g2(args, device)
    elif args.step == 'g1_stage':
        step_g1_stage(args)
    elif args.step == 'g1_cold':
        step_g1_cold(args)
    elif args.step == 'pack':
        step_pack(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
