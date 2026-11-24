"""Steps of queue 3: the key-frequency ablations of the embedding and the unembed.

Q1  p=97, seeds 5-13: the grokked sources with the key frequencies projected
    out of both the embedding and the unembedding, a control with the same
    number of non-key frequencies in both, and the unmodified baseline.
Q2  the same three variants on the two-layer sources (seeds 5-9).
Q3  p=59, seeds 5-10: the E+U-key-removed network with the attention weights,
    the MLP weights or both re-initialized, next to the existing E+U result.
Q4  p=59, seeds 11, 13, 14: the three variants of Q1.

Every variant is followed by the standard 2,000-step affine transfer, and the
row records the source's addition dev accuracy after the edit and the
transfer's final dev_mean.

    python nseed_backbone\\queue3_steps.py variants --runs-root runs_nseed_p97 --seeds 5 6 7
    python nseed_backbone\\queue3_steps.py reinit --seeds 5 6 7 8 9 10
    python nseed_backbone\\queue3_steps.py pack --out results_queue3.zip

Produces: Section 5 (the sweep of the key-frequency removal).
"""
from __future__ import annotations

import argparse
import csv
import io
import multiprocessing
import statistics
import sys
import time
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
import fourier_metrics as fm  # noqa: E402
from queue2_steps import (embed2_variants, embedding_and_unembed,  # noqa: E402
                           token_basis)

TRANSFER_STEPS = 2000
COVER = 0.90
FIELDS = ['seed', 'variant', 'n_key', 'source_dev_mean', 'final_dev_mean',
          'final_dev_worst', 'final_train_acc', 'steps_to_30pct']
STEP_FOLDERS = {'q1_p97': 'runs_nseed_q1_p97', 'q2_L2': 'runs_nseed_q2_L2',
                'q3_reinit': 'runs_nseed_q3_reinit', 'q4_p59': 'runs_nseed_q4_p59'}


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else common.PARENT / path


def write_csv(path, fields, rows):
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=fields, extrasaction='ignore')
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


def grokked_state_at(runs_root, seed):
    """The grokked checkpoint state of one seed inside a runs root."""
    path = Path(runs_root) / f'seed_{seed}' / 'ckpts' / 'grokked.pt'
    if not path.exists():
        raise RuntimeError(f'seed {seed}: no grokked checkpoint in {runs_root}')
    return common.load_checkpoint(path)['model']


def variant_set(state):
    """The three variants of Q1: baseline, keys removed, non-key control."""
    variants, keys, _control = embed2_variants(state)
    embedding = state['token_embed.weight'][:common.P]
    energy, _power = common.embedding_spectrum(embedding)
    all_keys = list(range(1, (common.P - 1) // 2 + 1))
    non_keys = sorted([k for k in all_keys if k not in set(keys)],
                      key=lambda k: -float(energy[k - 1]))
    control = non_keys[:len(keys)]
    return {
        'baseline': (state, len(keys)),
        'E+U_key_removed': (variants['E+U'][0], len(keys)),
        'E+U_nonkey_control': (embedding_and_unembed(state, token_basis(control)),
                               len(control)),
    }


def model_from(state, device):
    model = common.make_model(common.CONFIG)
    model.load_state_dict(state)
    return model.to(device)


def transfer(model, seed, device, strata, target_data, name, steps):
    """The standard affine transfer."""
    from run_components import run_transfer_job
    rows, _trained = run_transfer_job(name, seed, model, target_data, strata,
                                      Path('.'), steps=steps, eval_every=100,
                                      log=common.log_line)
    final = rows[-1] if rows else {}
    step_to_30 = None
    for row in sorted(rows, key=lambda r: int(r['step'])):
        value = common.as_float(row.get('dev_mean'))
        if value is not None and value >= 0.30:
            step_to_30 = int(row['step'])
            break
    return dict(final_dev_mean=common.as_float(final.get('dev_mean')),
                final_dev_worst=common.as_float(final.get('dev_worst')),
                final_train_acc=common.as_float(final.get('train_acc')),
                steps_to_30pct=step_to_30)


def variant_job(job):
    """One (seed, variant) of Q1, Q2 or Q4."""
    seed = job['seed']
    variant = job['variant']
    device = job['device']
    state = grokked_state_at(job['runs_root'], seed)
    edited, count = variant_set(state)[variant]
    model = model_from(edited, device)
    model.eval()
    mask = common.mask_for_seed(seed)
    strata, _exposure = common.split_for_mask(mask)
    source_data = common.build_task('source', device)
    target_data = common.build_task('affine', device)
    with torch.no_grad():
        source_evaluation = common.evaluate(model, source_data, strata)
    result = transfer(model, seed, device, strata, target_data,
                      f'q3_{variant}_seed{seed}', job['steps'])
    row = dict(seed=seed, variant=variant, n_key=count,
               source_dev_mean=source_evaluation['dev_mean'], **result)
    common.log_line(f'[queue3] seed {seed} {variant}: source dev '
                    f'{row["source_dev_mean"]:.4f} -> transfer {row["final_dev_mean"]}')
    return row


def fresh_state(seed):
    """The fresh donor of the reset_* jobs."""
    from run_factorial import fresh_donor_state
    return fresh_donor_state(seed)


def reinit_job(job):
    """One (seed, variant) of Q3."""
    seed = job['seed']
    variant = job['variant']
    device = job['device']
    state = grokked_state_at(job['runs_root'], seed)
    edited, _count = variant_set(state)['E+U_key_removed']
    merged = {key: value.clone() for key, value in edited.items()}
    if variant in ('attn_reinit', 'attn_mlp_reinit'):
        fresh = fresh_state(seed)
        for key, value in fresh.items():
            if key.startswith('attn.'):
                merged[key] = value.clone()
    if variant in ('mlp_reinit', 'attn_mlp_reinit'):
        fresh = fresh_state(seed)
        for key, value in fresh.items():
            if key.startswith(('mlp_in.', 'mlp_out.')):
                merged[key] = value.clone()
    model = model_from(merged, device)
    model.eval()
    mask = common.mask_for_seed(seed)
    strata, _exposure = common.split_for_mask(mask)
    source_data = common.build_task('source', device)
    target_data = common.build_task('affine', device)
    with torch.no_grad():
        source_evaluation = common.evaluate(model, source_data, strata)
    result = transfer(model, seed, device, strata, target_data,
                      f'q3_{variant}_seed{seed}', job['steps'])
    row = dict(seed=seed, variant=variant, n_key=0,
               source_dev_mean=source_evaluation['dev_mean'], **result)
    common.log_line(f'[queue3] seed {seed} {variant}: source dev '
                    f'{row["source_dev_mean"]:.4f} -> transfer {row["final_dev_mean"]}')
    return row


def worker_init(device, batch_log, threads):
    common.setup(threads, device)
    common.set_batch_log(batch_log)


def worker(job):
    try:
        if job['kind'] == 'variant':
            return variant_job(job)
        if job['kind'] == 'reinit':
            return reinit_job(job)
        raise ValueError(f'unknown job kind {job["kind"]!r}')
    except Exception as exc:  # noqa: BLE001 - one job must not kill the batch
        common.log_line(f'[queue3] job {job.get("name", "")} FAILED '
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


def step_variants(args, device):
    """Q1, Q2 and Q4."""
    out_dir = project_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'queue3.csv'
    done = {(row['seed'], row['variant']) for row in read_csv(path)}
    jobs = []
    for seed in args.seeds:
        for variant in ('baseline', 'E+U_key_removed', 'E+U_nonkey_control'):
            if (str(seed), variant) in done:
                common.log_line(f'[queue3] seed {seed} {variant}: already done')
                continue
            jobs.append(dict(name=f'{variant}_seed{seed}', kind='variant', seed=seed,
                             variant=variant, runs_root=args.runs_root,
                             device=device, steps=args.steps))
    results = run_pool(jobs, device, args.procs, str(common.BATCH_LOG), args.threads)
    rows = [row for row in results if 'error' not in row]
    merged = read_csv(path) + rows
    merged.sort(key=lambda row: (int(row['seed']), str(row['variant'])))
    write_csv(path, FIELDS, merged)
    common.log_line(f'[queue3] wrote {len(merged)} rows to {path}')
    return merged


def step_reinit(args, device):
    """Q3: the E+U-key-removed network with re-initialized attention or MLP."""
    out_dir = project_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'queue3.csv'
    done = {(row['seed'], row['variant']) for row in read_csv(path)}
    jobs = []
    for seed in args.seeds:
        for variant in ('attn_reinit', 'mlp_reinit', 'attn_mlp_reinit'):
            if (str(seed), variant) in done:
                continue
            jobs.append(dict(name=f'{variant}_seed{seed}', kind='reinit', seed=seed,
                             variant=variant, runs_root=args.runs_root,
                             device=device, steps=args.steps))
    results = run_pool(jobs, device, args.procs, str(common.BATCH_LOG), args.threads)
    rows = [row for row in results if 'error' not in row]
    reused = [row for row in read_csv(args.reuse_csv)
              if row.get('variant') in ('E+U', 'E+U_key_removed')
              and int(row['seed']) in args.seeds] if args.reuse_csv else []
    merged = reused + read_csv(path) + rows
    merged.sort(key=lambda row: (int(row['seed']), str(row['variant'])))
    write_csv(path, FIELDS, merged)
    common.log_line(f'[queue3] wrote {len(merged)} rows to {path} '
                    f'({len(reused)} copied E+U rows)')
    return merged


def step_pack(args):
    """Pack only the new folders and write the per-step summary."""
    root = Path(args.root) if args.root else common.PARENT
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    members = []
    summary = []
    for step, folder in STEP_FOLDERS.items():
        base = root / folder
        if not base.is_dir():
            continue
        for path in sorted(base.rglob('*')):
            if path.is_file() and path.suffix.lower() in ('.csv', '.json'):
                members.append(path)
        rows = read_csv(base / 'queue3.csv')
        for variant in sorted({row['variant'] for row in rows}):
            subset = [row for row in rows if row['variant'] == variant]
            summary.append(dict(
                step=step, variant=variant, n_seeds=len(subset),
                source_dev_mean=statistics.fmean(
                    [float(row['source_dev_mean']) for row in subset]),
                final_dev_mean=statistics.fmean(
                    [float(row['final_dev_mean']) for row in subset])))
    write_csv(root / 'runs_nseed_queue3_summary.csv',
              ['step', 'variant', 'n_seeds', 'source_dev_mean', 'final_dev_mean'],
              summary)
    report = root / 'nseed_backbone' / 'REPORT_queue3.md'
    if report.exists():
        members.append(report)
    with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in members:
            archive.write(path, path.relative_to(root).as_posix())
    common.log_line(f'[queue3] packed {len(members)} files into {out_path}')
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['variants', 'reinit', 'pack'])
    parser.add_argument('--runs-root', default='runs_nseed')
    parser.add_argument('--seeds', nargs='+', type=int, default=[5, 6, 7, 8, 9, 10])
    parser.add_argument('--out', default='runs_nseed_q1_p97')
    parser.add_argument('--reuse-csv', default=None)
    parser.add_argument('--root', default=None)
    parser.add_argument('--report', default='nseed_backbone/REPORT_queue3.md')
    parser.add_argument('--steps', type=int, default=TRANSFER_STEPS)
    parser.add_argument('--procs', type=int, default=1)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default=None)
    parser.add_argument('--batch-log', default='runs_nseed/batch.log')
    args = parser.parse_args(argv)

    runtime = common.setup(args.threads, args.device)
    device = runtime['device']
    common.set_batch_log(project_path(args.batch_log))
    common.log_line(f'[queue3] step {args.step} device={device} runs_root={args.runs_root}')

    if args.step == 'variants':
        step_variants(args, device)
    elif args.step == 'reinit':
        step_reinit(args, device)
    elif args.step == 'pack':
        step_pack(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
