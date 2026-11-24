"""Queue 21, step P2: the LayerNorm by batch-size 2x2.

The 2x2 has the cells no LayerNorm / Pre-LN against the full batch / the batch
of 512, with p = 59, the seeds 5 to 9, the coverage 0.4 and the weight decay
1.0 of the standard small-transformer setup.

Two of the cells exist already and are not run again:

  * no LayerNorm, full batch: the rows of `runs_nseed/seed_5..9` (the metrics
    and the summary of the source runs);
  * Pre-LN, batch 512: the rows of `runs_nseed_p1/seed_5..9` inside
    `results_queue9.zip` (the metrics of the runs) and of
    `runs_nseed_p1/p1_preln.csv` (the summary of the stages).

The two missing cells are run here: no LayerNorm with the batch of 512 and
Pre-LN with the full batch. The cell is taken from the environment, because
`common` reads NSEED_PRELN and NSEED_BATCH when it is imported.

For every run the source is trained to the step 20,000 with the recipe of the
other sources: the memorized, transition and grokked checkpoints are the ones
of the standard stage selection, and the norm control is the memorized
checkpoint rescaled to the norm of the grokked checkpoint.

Every stage is fine-tuned on the affine target for 2,000 steps with the dev
split scored every 100 steps; the memorized source and the norm control are
fine-tuned to the step 20,000 and their dev accuracy at that step is recorded
as well.

Only the dev split is read; the audit split is never touched.

    NSEED_BATCH=512 python nseed_backbone/p21_lnbatch.py run --seeds 5 6 7 8 9
    NSEED_PRELN=1 python nseed_backbone/p21_lnbatch.py run --seeds 5 6 7 8 9
    python nseed_backbone/p21_lnbatch.py assemble

Produces: Table 2 (the LayerNorm by batch-size 2x2).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import multiprocessing
import os
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

import torch  # noqa: E402

import common  # noqa: E402
from queue8_steps import stage_checkpoint  # noqa: E402
from queue9_steps import STAGES  # noqa: E402
from run_components import run_transfer_job  # noqa: E402
from run_seed import SeedRun  # noqa: E402

# a source that never groks has the late memorized checkpoint instead of the
# grokked one, and it takes the place of the memorized one in the long transfer
ALL_STAGES = STAGES + ('late_memorized',)

FIELDS = ['ln', 'batch', 'seed', 'stage', 'transfer_2000', 'transfer_20000',
          'source_dev', 'source_grok_step']
TRANSFER_STEPS = 2000
LONG_TRANSFER_STEPS = 20000
EVAL_EVERY = 100
SOURCE_CAP = 20000
CKPT_EVERY = 250
LONG_STAGES = ('memorized', 'norm_control', 'late_memorized')
GROK_TRAIN_ACC = 0.99
GROK_DEV_ACC = 0.90
GROK_CONSECUTIVE = 3
DISK_FLOOR_GB = 10.0

# the roots of the cells that exist already
EXISTING = {
    # the roots of the two cells that existed already; they are given by
    # --existing-root and --existing-archive, relative to the repository
    ('none', 'full'): 'results/existing/runs_nseed',
    ('preln', '512'): 'results/existing/results_queue9.zip',
}


def cell_labels():
    """The labels of the cell of the environment."""
    ln = 'preln' if common.CONFIG.get('preln') else 'none'
    batch = 'full' if not common.BATCH_SIZE else str(common.BATCH_SIZE)
    return ln, batch


def append_rows(path, fields, rows):
    """Append the rows to a CSV; the header is written only when it is created."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
    return path


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open('r', newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def ensure_header(path, fields):
    """Create the CSV with its header before the workers append to it.

    The workers run in several processes; when each of them wrote the header
    when the file did not exist yet, the file got one header per worker that
    arrived first.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open('w', newline='', encoding='utf-8') as stream:
            csv.DictWriter(stream, fieldnames=fields).writeheader()
    return path


def grok_step_of(source_rows):
    """The grok step of a source, from its evaluation rows.

    The rule is the one of `run_seed`: three evaluations in a row with a train
    accuracy of at least 0.99 and a worst-context dev accuracy of at least
    0.90. The step of the third one is the grok step.
    """
    consecutive = 0
    for row in sorted(source_rows, key=lambda r: int(r['step'])):
        train_acc = common.as_float(row.get('train_acc'))
        dev_worst = common.as_float(row.get('dev_worst'))
        if train_acc is not None and dev_worst is not None and \
                train_acc >= GROK_TRAIN_ACC and dev_worst >= GROK_DEV_ACC:
            consecutive += 1
            if consecutive >= GROK_CONSECUTIVE:
                return int(row['step'])
        else:
            consecutive = 0
    return ''


def free_ckpts(run, stages):
    """Delete the source checkpoints that no stage uses."""
    keep = {record['ckpt'] for record in stages.values() if record.get('ckpt')}
    removed = 0
    for path in sorted(run.ckpts.glob('step_*.pt')):
        if path.name not in keep:
            path.unlink()
            removed += 1
    return removed


# ---------------------------------------------------------------------------
# the missing cells
# ---------------------------------------------------------------------------
def train_source(seed, out_dir, device, source_cap=SOURCE_CAP):
    """The source of one seed, with the standard stage selection."""
    summary = common.read_summary(Path(out_dir) / 'summary.json')
    if summary.get('source_finished'):
        stages = summary.get('stages') or {}
        source_devs = {}
        for row in read_rows(Path(out_dir) / 'metrics.csv'):
            if row.get('stage') == 'source':
                source_devs[int(row['step'])] = row.get('dev_mean')
        return stages, summary.get('grok_step'), \
            summary.get('source_final_dev', ''), source_devs
    run = SeedRun(seed, out_dir, threads=1, resume=False, tiers=(), device=device,
                  source_cap=source_cap, ckpt_every=CKPT_EVERY, log=common.log_line)
    run._ensure_dirs()
    mask = common.mask_for_seed(seed)
    strata, _exposure = common.split_for_mask(mask)
    data = common.build_task('source', device)
    status, grok_step, rows = run.train_source(strata, data, run._summary())
    stages = run.select_stages(rows, grok_step, status)
    summary = run._summary()
    summary['stages'] = stages
    summary['source_finished'] = True
    summary['source_final_dev'] = rows[-1].get('dev_mean') if rows else ''
    run._write_summary(summary)
    common.log_line(f'[q21] source seed {seed}: status {status} grok_step {grok_step} '
                    f'stages {sorted(stages)}')
    removed = free_ckpts(run, stages)
    common.log_line(f'[q21] source seed {seed}: {removed} unused checkpoints deleted')
    source_devs = {int(r['step']): r.get('dev_mean') for r in rows
                   if r.get('stage') == 'source'}
    return stages, grok_step, summary['source_final_dev'], source_devs


def stage_source_dev(stage, stages, source_devs):
    """The dev accuracy of the source checkpoint of a stage.

    It is the dev accuracy that the stage selection measured for that
    checkpoint; the late memorized checkpoint of a source that never grokked
    has none, so the dev accuracy of the source at that step is used.
    """
    record = stages.get(stage) or {}
    value = record.get('dev_mean')
    if value in (None, ''):
        value = source_devs.get(int(record.get('step', 0)))
    if value in (None, ''):
        values = [v for v in source_devs.values() if v not in (None, '')]
        value = values[-1] if values else ''
    return value


def run_seed(job):
    """One seed: the source and the transfers of its four stages."""
    args = job['args']
    device = job['device']
    seed = job['seed']
    ln, batch = cell_labels()
    runs_root = Path(args.runs_root)
    out_dir = runs_root / f'seed_{seed}'
    out_path = Path(args.out)

    stages, grok_step, source_dev, source_devs = train_source(
        seed, out_dir, device, source_cap=args.source_cap)
    if not stages:
        common.log_line(f'[q21] seed {seed}: no stages, skipped')
        return []
    grokked_norm = (stages.get('grokked') or {}).get('global_norm')
    mask = common.mask_for_seed(seed)
    strata, _exposure = common.split_for_mask(mask)
    target_data = common.build_task('affine', device)
    rows = []
    for stage in ALL_STAGES:
        if stage == 'norm_control':
            if grokked_norm is None or 'memorized' not in stages:
                continue
        elif stage not in stages:
            continue
        state = stage_checkpoint(runs_root, seed, stage, stages, grokked_norm)
        model = common.make_model(common.CONFIG)
        model.load_state_dict(state)
        model = model.to(device)
        model.eval()
        if stage == 'norm_control':
            # the memorized checkpoint rescaled to the grokked norm is kept
            common.save_model_checkpoint(
                out_dir / 'ckpts' / 'norm_control.pt', model, 0,
                extra={'seed': seed, 'stage': 'norm_control',
                       'global_norm': grokked_norm})
        steps = args.long_steps if stage in LONG_STAGES else args.steps
        trows, _trained = run_transfer_job(f'q21_{ln}_{batch}_{stage}_seed{seed}',
                                           seed, model, target_data, strata,
                                           Path('.'), steps=steps,
                                           eval_every=EVAL_EVERY, log=common.log_line)
        by_step = {int(r['step']): r for r in trows}
        at_2000 = by_step.get(args.steps, {})
        at_20000 = by_step.get(args.long_steps, {})
        # the dev accuracy of the source checkpoint of the stage, the same
        # value that the rows of the cells that exist already hold
        stage_dev = stage_source_dev(stage, stages, source_devs)
        if stage == 'norm_control':
            stage_dev = stage_source_dev('memorized', stages, source_devs)
        row = dict(ln=ln, batch=batch, seed=seed, stage=stage,
                   transfer_2000=at_2000.get('dev_mean', ''),
                   transfer_20000=at_20000.get('dev_mean', '') if stage in LONG_STAGES
                   else '',
                   source_dev=stage_dev, source_grok_step=grok_step)
        rows.append(row)
        append_rows(out_path, FIELDS, [row])
        common.log_line(f'[q21] seed {seed} {stage}: transfer 2000 '
                        f'{row["transfer_2000"]} 20000 {row["transfer_20000"]}')
        del model
        torch.cuda.empty_cache()
    return rows


def worker_init(device, batch_log, threads):
    common.setup(threads, device)


def worker(job):
    try:
        rows = run_seed(job)
        return dict(job=job, rows=rows, error=None)
    except Exception as exc:  # noqa: BLE001 - one seed must not kill the pool
        common.log_line(f'[q21] seed {job["seed"]} FAILED {type(exc).__name__}: {exc}')
        common.log_line(traceback.format_exc())
        return dict(job=job, rows=[], error=str(exc))


def run_pool(jobs, device, procs, threads=1):
    results = []
    if not jobs:
        return results
    if procs <= 1:
        worker_init(device, common.BATCH_LOG, threads)
        for job in jobs:
            results.append(worker(job))
        return results
    context = multiprocessing.get_context('spawn')
    with context.Pool(processes=procs, initializer=worker_init,
                      initargs=(device, common.BATCH_LOG, threads)) as pool:
        for result in pool.imap_unordered(worker, jobs):
            results.append(result)
    return results


def step_run(args, device):
    """The two missing cells."""
    ln, batch = cell_labels()
    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    ensure_header(out_path, FIELDS)
    done = {(row['seed'], row['stage']) for row in read_rows(out_path)}
    jobs = []
    for seed in args.seeds:
        if any(key[0] == str(seed) for key in done):
            common.log_line(f'[q21] seed {seed}: rows already there, skipped')
            continue
        jobs.append(dict(args=args, device=device, seed=seed))
    print(f'[q21] cell {ln} {batch}: {len(jobs)} seeds to run', flush=True)
    results = run_pool(jobs, device, args.procs, args.threads)
    failed = [result for result in results if result.get('error')]
    for result in failed:
        common.log_line(f'[q21] FAILED seed {result["job"]["seed"]}: {result["error"]}')
    print(f'[q21] cell {ln} {batch}: {len(results) - len(failed)} seeds finished, '
          f'{len(failed)} failed', flush=True)
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# the cells that exist already
# ---------------------------------------------------------------------------
def reuse_none_full(args):
    """The rows of the cell no LayerNorm with the full batch."""
    root = Path(args.existing_root) / 'runs_nseed'
    rows = []
    for seed in args.seeds:
        seed_dir = root / f'seed_{seed}'
        summary = common.read_summary(seed_dir / 'summary.json')
        stages = summary.get('stages') or {}
        grok_step = summary.get('grok_step')
        metrics = read_rows(seed_dir / 'metrics.csv')
        transfers = {}
        for row in metrics:
            if row.get('stage') == 'transfer' and int(row.get('step', 0)) == TRANSFER_STEPS:
                transfers[row.get('init')] = row.get('dev_mean')
        for stage in STAGES:
            if stage == 'norm_control':
                if 'memorized' not in stages:
                    continue
            elif stage not in stages:
                continue
            record = stages.get(stage) or {}
            # the norm control is the memorized checkpoint rescaled, so it keeps
            # the held-out accuracy of the memorized stage
            source_dev = record.get('dev_mean', '')
            if stage == 'norm_control':
                source_dev = (stages.get('memorized') or {}).get('dev_mean', '')
            rows.append(dict(ln='none', batch='full', seed=seed, stage=stage,
                             transfer_2000=transfers.get(stage, ''),
                             transfer_20000='',
                             source_dev=source_dev,
                             source_grok_step=grok_step))
    return rows


def reuse_preln_512(args):
    """The rows of the cell Pre-LN with the batch of 512."""
    archive = Path(args.existing_root) / 'results_queue9.zip'
    rows = []
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
        summary_rows = {}
        if 'runs_nseed_p1/p1_preln.csv' in names:
            text = z.read('runs_nseed_p1/p1_preln.csv').decode('utf-8')
            for row in csv.DictReader(io.StringIO(text)):
                summary_rows[(int(row['seed']), row['stage'])] = row
        for seed in args.seeds:
            member = f'runs_nseed_p1/seed_{seed}/metrics.csv'
            if member not in names:
                common.log_line(f'[q21] seed {seed}: {member} is missing, skipped')
                continue
            text = z.read(member).decode('utf-8')
            metrics = list(csv.DictReader(io.StringIO(text)))
            transfers = {}
            for row in metrics:
                if row.get('stage') == 'transfer' and \
                        int(row.get('step', 0)) == TRANSFER_STEPS:
                    transfers[row.get('init')] = row.get('dev_mean')
            source_rows = [row for row in metrics if row.get('stage') == 'source']
            grok_step = grok_step_of(source_rows)
            last_source = max(source_rows, key=lambda r: int(r['step'])) \
                if source_rows else {}
            records = {stage: record for (seed_, stage), record
                       in summary_rows.items() if seed_ == seed}
            if 'grokked' in records:
                grok_step = int(records['grokked']['source_step'])
            stages = sorted(set(transfers) | set(records))
            for stage in stages:
                record = records.get(stage)
                if record is not None:
                    transfer_2000 = record['transfer_dev_mean']
                    source_dev = record['source_dev_mean']
                else:
                    # a stage that the summary of queue 9 does not hold, for
                    # example the late memorized one of a source that never
                    # grokked: the values come from the metrics of the run
                    transfer_2000 = transfers.get(stage, '')
                    source_dev = last_source.get('dev_mean', '')
                rows.append(dict(ln='preln', batch='512', seed=seed, stage=stage,
                                 transfer_2000=transfer_2000,
                                 transfer_20000='',
                                 source_dev=source_dev,
                                 source_grok_step=grok_step))
    return rows


def step_long(args, device):
    """The 20,000-step transfers of a cell whose sources exist already.

    The sources of the cell no LayerNorm with the full batch are on the
    instance, so the long transfers of their memorized and norm-matched
    checkpoints can be run without training anything again. The cells are not
    rerun: the 2,000-step transfers keep the values of the existing rows and
    only the 20,000-step ones are added.
    """
    ln = args.ln
    batch = args.batch
    runs_root = Path(args.runs_root)
    out_path = Path(args.out)
    ensure_header(out_path, FIELDS)
    done = {(row['ln'], row['batch'], row['seed'], row['stage'])
            for row in read_rows(out_path)}
    mask = common.mask_for_seed(0)
    target_data = common.build_task('affine', device)
    for seed in args.seeds:
        if (args.ln, args.batch, str(seed), 'memorized') in done:
            common.log_line(f'[q21] seed {seed}: the long rows are already there, '
                            'skipped')
            continue
        summary = common.read_summary(runs_root / f'seed_{seed}' / 'summary.json')
        stages = summary.get('stages') or {}
        if not stages:
            common.log_line(f'[q21] seed {seed}: no stages, skipped')
            continue
        grokked_norm = (stages.get('grokked') or {}).get('global_norm')
        mask = common.mask_for_seed(seed)
        strata, _exposure = common.split_for_mask(mask)
        for stage in LONG_STAGES:
            if stage == 'norm_control':
                if grokked_norm is None or 'memorized' not in stages:
                    continue
            elif stage not in stages:
                continue
            state = stage_checkpoint(runs_root, seed, stage, stages, grokked_norm)
            model = common.make_model(common.CONFIG)
            model.load_state_dict(state)
            model = model.to(device)
            model.eval()
            trows, _trained = run_transfer_job(
                f'q21long_{ln}_{batch}_{stage}_seed{seed}', seed, model, target_data,
                strata, Path('.'), steps=args.long_steps, eval_every=EVAL_EVERY,
                log=common.log_line)
            by_step = {int(r['step']): r for r in trows}
            row = dict(ln=ln, batch=batch, seed=seed, stage=stage,
                       transfer_2000=by_step.get(args.steps, {}).get('dev_mean', ''),
                       transfer_20000=by_step.get(args.long_steps, {}).get('dev_mean', ''),
                       source_dev='', source_grok_step=summary.get('grok_step'))
            append_rows(out_path, FIELDS, [row])
            common.log_line(f'[q21] long seed {seed} {stage}: transfer 2000 '
                            f'{row["transfer_2000"]} 20000 {row["transfer_20000"]}')
            del model
            torch.cuda.empty_cache()
    return 0


def step_assemble(args):
    """The final table: the two existing cells and the two new ones."""
    out_path = Path(args.out)
    rows = []
    rows.extend(reuse_none_full(args))
    rows.extend(reuse_preln_512(args))
    for cell in ['none_512', 'preln_full']:
        rows.extend(read_rows(Path(args.out_dir) / f'm21_lnbatch_{cell}.csv'))
    # the long transfers of the cell no LayerNorm with the full batch fill the
    # empty 20,000-step cells of its rows
    longs = read_rows(Path(args.out_dir) / 'm21_lnbatch_none_full_long.csv')
    index = {(row['ln'], row['batch'], str(row['seed']), row['stage']): row
             for row in longs}
    for row in rows:
        if row.get('transfer_20000'):
            continue
        other = index.get((row['ln'], row['batch'], str(row['seed']), row['stage']))
        if other and other.get('transfer_20000'):
            row['transfer_20000'] = other['transfer_20000']
    order = {'none': 0, 'preln': 1}
    batch_order = {'full': 0, '512': 1}
    rows.sort(key=lambda r: (order.get(r['ln'], 9), batch_order.get(r['batch'], 9),
                              int(r['seed']), ALL_STAGES.index(r['stage'])
                              if r['stage'] in ALL_STAGES else 9))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f'[q21] {len(rows)} rows in {out_path}', flush=True)
    counts = {}
    for row in rows:
        key = (row['ln'], row['batch'])
        counts[key] = counts.get(key, 0) + 1
    for key in sorted(counts):
        print(f'[q21]   {key[0]} {key[1]}: {counts[key]} rows', flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['run', 'long', 'assemble'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[5, 6, 7, 8, 9])
    parser.add_argument('--runs-root', default='runs_nseed_queue21/runs_nseed_q21')
    parser.add_argument('--out', default='runs_nseed_queue21/m21_lnbatch.csv')
    parser.add_argument('--out-dir', default='runs_nseed_queue21')
    parser.add_argument('--existing-root', default='results/existing',
                        help='the folder with the runs of the cells that '
                             'existed already')
    parser.add_argument('--procs', type=int, default=2)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--source-cap', type=int, default=SOURCE_CAP)
    parser.add_argument('--steps', type=int, default=TRANSFER_STEPS)
    parser.add_argument('--long-steps', type=int, default=LONG_TRANSFER_STEPS)
    parser.add_argument('--ln', default='none')
    parser.add_argument('--batch', default='full')
    args = parser.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.step == 'run':
        ln, batch = cell_labels()
        args.runs_root = f'{args.runs_root}_{ln}_{batch}'
        args.out = f'{args.out_dir}/m21_lnbatch_{ln}_{batch}.csv'
        common.log_line(f'[q21] cell {ln} {batch}: preln={common.CONFIG.get("preln")} '
                        f'batch={common.BATCH_SIZE} device={device}')
        return step_run(args, device)
    if args.step == 'long':
        args.out = f'{args.out_dir}/m21_lnbatch_{args.ln}_{args.batch}_long.csv'
        common.log_line(f'[q21] long cell {args.ln} {args.batch}: '
                        f'preln={common.CONFIG.get("preln")} '
                        f'batch={common.BATCH_SIZE} device={device}')
        return step_long(args, device)
    return step_assemble(args)


if __name__ == '__main__':
    raise SystemExit(main())
