"""Steps of queue 4: R1, the component reinitialization of the grokked sources.

For every source the components are reinitialized with the fresh donor (the
model built after torch.manual_seed(70000 + seed), the same donor the reset_*
jobs use), alone or together with the key frequencies projected out of the
embedding and the unembedding, and the standard 2,000-step affine transfer is
run afterwards.

    python nseed_backbone\\queue4_steps.py reinit --runs-root runs_nseed_p97 --seeds 5 6 7
    python nseed_backbone\\queue4_steps.py pack --out results_queue4.zip

Produces: Section 5 (the reruns of the stages).
"""
from __future__ import annotations

import argparse
import csv
import io
import multiprocessing
import statistics
import sys
import traceback
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
from queue3_steps import (FIELDS, STEP_FOLDERS, grokked_state_at,  # noqa: E402
                           model_from, project_path, read_csv, transfer,
                           variant_set, write_csv)

TRANSFER_STEPS = 2000
FULL_VARIANTS = ('mlp_reinit', 'attn_reinit', 'attn_mlp_reinit',
                 'E+U_key_removed+mlp_reinit', 'E+U_key_removed+attn_mlp_reinit')
SHORT_VARIANTS = ('mlp_reinit', 'attn_reinit', 'attn_mlp_reinit')
PYTHIA_FOLDER = 'runs_nseed_r2_pythia'


def fresh_state(seed):
    """The fresh donor of the reset_* jobs."""
    from run_factorial import fresh_donor_state
    return fresh_donor_state(seed)


def reinit_state(state, seed, variant):
    """The state with the named components reinitialized."""
    merged = {key: value.clone() for key, value in state.items()}
    fresh = fresh_state(seed)
    if 'mlp' in variant:
        for key, value in fresh.items():
            if key.startswith(('mlp_in.', 'mlp_out.')):
                merged[key] = value.clone()
    if 'attn' in variant:
        for key, value in fresh.items():
            if key.startswith('attn.'):
                merged[key] = value.clone()
    return merged


def reinit_job(job):
    """One (seed, variant) of R1."""
    seed = job['seed']
    variant = job['variant']
    device = job['device']
    state = grokked_state_at(job['runs_root'], seed)
    if variant.startswith('E+U_key_removed'):
        base, count = variant_set(state)['E+U_key_removed']
    else:
        base, count = state, 0
    merged = reinit_state(base, seed, variant)
    model = model_from(merged, device)
    model.eval()
    mask = common.mask_for_seed(seed)
    strata, _exposure = common.split_for_mask(mask)
    source_data = common.build_task('source', device)
    target_data = common.build_task('affine', device)
    with torch.no_grad():
        source_evaluation = common.evaluate(model, source_data, strata)
    result = transfer(model, seed, device, strata, target_data,
                      f'q4_{variant}_seed{seed}', job['steps'])
    row = dict(seed=seed, variant=variant, n_key=count,
               source_dev_mean=source_evaluation['dev_mean'], **result)
    common.log_line(f'[queue4] seed {seed} {variant}: source dev '
                    f'{row["source_dev_mean"]:.4f} -> transfer {row["final_dev_mean"]}')
    return row


def worker_init(device, batch_log, threads):
    common.setup(threads, device)
    common.set_batch_log(batch_log)


def worker(job):
    try:
        return reinit_job(job)
    except Exception as exc:  # noqa: BLE001 - one job must not kill the batch
        common.log_line(f'[queue4] job {job.get("name", "")} FAILED '
                        f'{type(exc).__name__}: {exc}')
        common.log_line(traceback.format_exc())
        return dict(name=job.get('name', ''), error=f'{type(exc).__name__}: {exc}')


def run_pool(jobs, device, procs, batch_log, threads=1):
    """Run the jobs over a pool of worker processes."""
    results = []
    if not jobs:
        return results
    if procs <= 1:
        worker_init(device, batch_log, threads)
        for job in jobs:
            results.append(worker(job))
        return results
    context = multiprocessing.get_context('spawn')
    with context.Pool(processes=procs, initializer=worker_init,
                      initargs=(device, batch_log, threads)) as pool:
        for result in pool.imap_unordered(worker, jobs):
            results.append(result)
    return results


def step_reinit(args, device):
    """R1: the reinitialization variants of one source family."""
    out_dir = project_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'queue4.csv'
    done = {(row['seed'], row['variant']) for row in read_csv(path)}
    jobs = []
    for seed in args.seeds:
        for variant in args.variants:
            if (str(seed), variant) in done:
                common.log_line(f'[queue4] seed {seed} {variant}: already done')
                continue
            jobs.append(dict(name=f'{variant}_seed{seed}', kind='reinit', seed=seed,
                             variant=variant, runs_root=args.runs_root,
                             device=device, steps=args.steps))
    results = run_pool(jobs, device, args.procs, str(common.BATCH_LOG), args.threads)
    rows = [row for row in results if 'error' not in row]
    merged = read_csv(path) + rows
    merged.sort(key=lambda row: (int(row['seed']), str(row['variant'])))
    write_csv(path, FIELDS, merged)
    common.log_line(f'[queue4] wrote {len(merged)} rows to {path}')
    return merged


def step_pack(args):
    """Pack only the new folders and write the report."""
    root = Path(args.root) if args.root else common.PARENT
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    folders = {'r1_p97': 'runs_nseed_r1_p97', 'r1_L2': 'runs_nseed_r1_L2',
               'r1_p59': 'runs_nseed_r1_p59', 'r2_pythia': PYTHIA_FOLDER}
    members = []
    r1_rows = []
    r2_rows = []
    for step, folder in folders.items():
        base = root / folder
        if not base.is_dir():
            continue
        for path in sorted(base.rglob('*')):
            if path.is_file() and path.suffix.lower() in ('.csv', '.json'):
                members.append(path)
        if step.startswith('r1'):
            r1_rows.extend(read_csv(base / 'queue4.csv'))
        elif step.startswith('r2'):
            for csv_path in sorted(base.glob('pythia_transfer*.csv')):
                r2_rows.extend(read_csv(csv_path))
    report = root / 'nseed_backbone' / 'REPORT_job4.md'
    if report.exists():
        members.append(report)
    with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in members:
            archive.write(path, path.relative_to(root).as_posix())
    write_report(report, r1_rows, r2_rows)
    write_csv(root / 'runs_nseed_queue4_summary.csv',
              ['step', 'variant', 'n_seeds', 'source_dev_mean', 'final_dev_mean'],
              summary_rows(r1_rows, r2_rows))
    common.log_line(f'[queue4] packed {len(members)} files into {out_path}')
    return out_path


def summary_rows(r1_rows, r2_rows):
    """The per-variant and per-revision means."""
    rows = []
    for variant in sorted({row['variant'] for row in r1_rows}):
        subset = [row for row in r1_rows if row['variant'] == variant]
        rows.append(dict(step='r1', variant=variant, n_seeds=len(subset),
                         source_dev_mean=statistics.fmean(
                             [float(row['source_dev_mean']) for row in subset]),
                         final_dev_mean=statistics.fmean(
                             [float(row['final_dev_mean']) for row in subset])))
    for revision in sorted({(row.get('model', ''), row['revision']) for row in r2_rows}):
        subset = [row for row in r2_rows
                  if (row.get('model', ''), row['revision']) == revision]
        if not subset:
            continue
        rows.append(dict(step='r2', variant=f"{subset[0].get('model', '')} "
                                             f"{subset[0]['revision']}",
                         n_seeds=len(subset),
                         source_dev_mean=statistics.fmean(
                             [float(row['stage1_heldout_ab']) for row in subset]),
                         final_dev_mean=statistics.fmean(
                             [float(row['stage2_a2b']) for row in subset])))
    return rows


def write_report(path, r1_rows, r2_rows):
    """The per-variant means of R1 and the per-revision means of R2."""
    from pythia_transfer import REVISIONS
    missing = []
    for model_name, revisions in REVISIONS.items():
        present = {row['revision'] for row in r2_rows
                   if row.get('model') == model_name}
        missing.extend(f'{model_name} {revision}' for revision in revisions
                       if revision not in present)
    lines = ['# REPORT_job4', '',
             '## R1 per-variant means (source_dev / final_dev)',
             '| variant | n_seeds | source_dev | final_dev | steps_to_30pct |',
             '|---|---|---|---|---|']
    for variant in sorted({row['variant'] for row in r1_rows}):
        subset = [row for row in r1_rows if row['variant'] == variant]
        steps = [int(row['steps_to_30pct']) for row in subset
                 if row.get('steps_to_30pct') not in (None, '')]
        lines.append(f"| {variant} | {len(subset)} "
                     f"| {statistics.fmean([float(row['source_dev_mean']) for row in subset]):.4f} "
                     f"| {statistics.fmean([float(row['final_dev_mean']) for row in subset]):.4f} "
                     f"| {statistics.fmean(steps) if steps else 'none'} |")
    lines += ['', '## R2 per-revision means over seeds',
              '| model revision | n_seeds | stage1_train | stage1_ab | stage2_a2b '
              '| direct_a2b | s1_copy | s1_max |',
              '|---|---|---|---|---|---|---|---|']
    keys = [('stage1_train_acc', 'stage1_train'), ('stage1_heldout_ab', 'stage1_ab'),
            ('stage2_a2b', 'stage2_a2b'), ('baseline_direct_a2b', 'direct_a2b'),
            ('stage1_copy', 's1_copy'), ('stage1_max', 's1_max')]
    for revision in sorted({(row.get('model', ''), row['revision']) for row in r2_rows}):
        subset = [row for row in r2_rows
                  if (row.get('model', ''), row['revision']) == revision]
        if not subset:
            continue
        cells = [f"{subset[0].get('model', '')} {subset[0]['revision']}", str(len(subset))]
        for field, _label in keys:
            values = [float(row[field]) for row in subset
                      if row.get(field) not in (None, '')]
            cells.append(f'{statistics.fmean(values):.4f}' if values else 'nan')
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')
    lines.append('R2 revisions missing: ' + (', '.join(missing) if missing else 'none'))
    Path(path).write_text('\n'.join(lines), encoding='utf-8')
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['reinit', 'pack'])
    parser.add_argument('--runs-root', default='runs_nseed')
    parser.add_argument('--seeds', nargs='+', type=int, default=[5, 6, 7, 8, 9, 10])
    parser.add_argument('--variants', nargs='+', default=list(FULL_VARIANTS))
    parser.add_argument('--out', default='runs_nseed_r1_p97')
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
    common.log_line(f'[queue4] step {args.step} device={device} '
                    f'runs_root={args.runs_root}')

    if args.step == 'reinit':
        step_reinit(args, device)
    elif args.step == 'pack':
        step_pack(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
