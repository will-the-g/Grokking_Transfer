"""Steps of the second overnight queue (S1 embed ablation, S2 copy_a norm, S3/S4 fine sweep).

Existing code only: the key frequencies and the frequency projections come from
`fourier_metrics`, the transfers are the standard 2,000-step affine transfers
(`run_components.run_transfer_job`), the sources are retrained with
`run_seed.SeedRun.train_source` and the sweep rows use the columns of
`run_sweep.sweep_summary.csv`.

Every step writes into its own new folder, so nothing existing is overwritten.

    python nseed_backbone\\queue2_steps.py embed_ablation --seeds 5 6 7 8 9 10
    python nseed_backbone\\queue2_steps.py copy_a_lownorm
    python nseed_backbone\\queue2_steps.py fine_sweep --seeds 5 6 7
    python nseed_backbone\\queue2_steps.py pack --out results_queue2.zip

Produces: Figure 1 (the sweep_summary.csv files).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
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
from run_components import run_transfer_job  # noqa: E402
from run_seed import SeedRun  # noqa: E402
from run_sweep import SWEEP_FIELDS, steps_to_30  # noqa: E402

TRANSFER_STEPS = 2000
EVAL_EVERY = 100
COVER = 0.90
HIGH_NORM_SEEDS = (5, 8, 10)
NORM_UP = 60.0
NORM_DOWN = 40.0
TRANSFER_THRESHOLD = 0.08

EMBED_FIELDS = ['seed', 'variant', 'n_key', 'n_control', 'source_dev_mean',
                'final_dev_mean', 'final_dev_worst', 'final_train_acc',
                'steps_to_30pct']
NORM_FIELDS = ['seed', 'init_offset', 'age_step', 'global_norm', 'frac_top5',
               'frac_top10', 'n_keys', 'key_overlap', 'aged_transfer_dev_mean',
               'intervention', 'transfer_dev_mean', 'transfer_dev_worst']
REPORT_LINES = 20


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


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def grokked_state(seed):
    """The seed's grokked checkpoint state and its summary."""
    seed_dir = common.PARENT / 'runs_nseed' / f'seed_{seed}'
    summary = common.read_summary(seed_dir / 'summary.json')
    stages = summary.get('stages', {}) or {}
    if 'grokked' not in stages:
        raise RuntimeError(f'seed {seed}: no grokked stage')
    payload = common.load_checkpoint(seed_dir / 'ckpts' / stages['grokked']['ckpt'])
    return payload['model'], summary


def model_from(state, device):
    model = common.make_model(common.CONFIG)
    model.load_state_dict(state)
    return model.to(device)


def transfer(model, seed, device, strata, target_data, name, steps):
    """The standard affine transfer; returns the final row."""
    rows, _trained = run_transfer_job(name, seed, model, target_data, strata,
                                      Path('.'), steps=steps, eval_every=EVAL_EVERY,
                                      log=common.log_line)
    final = rows[-1] if rows else {}
    return dict(final_dev_mean=common.as_float(final.get('dev_mean')),
                final_dev_worst=common.as_float(final.get('dev_worst')),
                final_train_acc=common.as_float(final.get('train_acc')),
                steps_to_30pct=steps_to_30(rows))


def worker_init(device, batch_log, threads):
    common.setup(threads, device)
    common.set_batch_log(batch_log)


def worker(job):
    try:
        return run_job(job)
    except Exception as exc:  # noqa: BLE001 - one job must not kill the batch
        common.log_line(f'[queue2] job {job.get("name", "")} FAILED '
                        f'{type(exc).__name__}: {exc}')
        common.log_line(traceback.format_exc())
        return dict(name=job.get('name', ''), error=f'{type(exc).__name__}: {exc}')


def run_job(job):
    """Dispatch one job to its step."""
    kind = job['kind']
    if kind == 'embed':
        return embed_job(job)
    if kind == 'embed2':
        return embed2_job(job)
    if kind == 'norm':
        return norm_job(job)
    if kind == 'snapshot':
        return snapshot_job(job)
    if kind == 'fine':
        return fine_job(job)
    raise ValueError(f'unknown job kind {kind!r}')


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


def out_of_time(deadline):
    return deadline is not None and time.time() > deadline


# ---------------------------------------------------------------------------
# S1: embedding ablation
# ---------------------------------------------------------------------------
def with_embedding(state, embedding):
    """The state with new numeric embedding rows."""
    merged = {key: value.clone() for key, value in state.items()}
    merged['token_embed.weight'][:common.P] = embedding.to(
        merged['token_embed.weight'].dtype)
    return merged


def embed_variants(state):
    """The four variants of the numeric embedding and the key frequencies."""
    embedding = state['token_embed.weight'][:common.P]
    keys, _share, _total = fm.top_frequencies(embedding, COVER)
    energy, _power = common.embedding_spectrum(embedding)
    all_keys = list(range(1, (common.P - 1) // 2 + 1))
    non_keys = sorted([k for k in all_keys if k not in set(keys)],
                      key=lambda k: -float(energy[k - 1]))
    control = non_keys[:len(keys)]
    variants = {
        'baseline': (state, 0),
        'key_removed': (with_embedding(state, fm.knockout(embedding, 'key',
                                                            top_k=keys)[0]), len(keys)),
        'control_nonkey_removed': (with_embedding(state,
                                                  fm.project_out(embedding, control)),
                                   len(control)),
        'key_only': (with_embedding(state, fm.knockout(embedding, 'nonkey',
                                                       top_k=keys)[0]), len(keys)),
    }
    return variants, keys, control


def embed_job(job):
    """One (seed, variant) ablation."""
    seed = job['seed']
    variant = job['variant']
    device = job['device']
    state, _summary = grokked_state(seed)
    variants, _keys, _control = embed_variants(state)
    edited, count = variants[variant]
    model = model_from(edited, device)
    model.eval()
    mask = common.mask_for_seed(seed)
    strata, _exposure = common.split_for_mask(mask)
    source_data = common.build_task('source', device)
    target_data = common.build_task('affine', device)
    with torch.no_grad():
        source_evaluation = common.evaluate(model, source_data, strata)
    result = transfer(model, seed, device, strata, target_data,
                      f'embed_{variant}_seed{seed}', job['steps'])
    row = dict(seed=seed, variant=variant, n_key=count,
               n_control=count if variant == 'control_nonkey_removed' else 0,
               source_dev_mean=source_evaluation['dev_mean'], **result)
    common.log_line(f'[queue2] S1 seed {seed} {variant}: source dev '
                    f'{row["source_dev_mean"]:.4f} -> transfer {row["final_dev_mean"]}')
    return row


# ---------------------------------------------------------------------------
# embed ablation 2: the key frequencies out of the unembed and the MLP
# ---------------------------------------------------------------------------
def token_basis(keys):
    """The orthonormal basis of the key frequencies over the token/class axis."""
    rows = []
    for key in keys:
        cos, sin = fm.frequency_basis(common.P, key)
        rows.append(cos)
        rows.append(sin)
    return torch.stack(rows).double()


def coefficient_basis(embedding, keys):
    """The orthonormal basis of the embedding's key-frequency directions.

    The directions are the cosine and sine coefficients of the embedding over
    the token axis, so they span the key-frequency subspace of the d_model
    (residual) space.
    """
    spectrum = torch.fft.fft(embedding.double(), dim=0)
    rows = []
    for key in keys:
        rows.append(spectrum[key].real)
        rows.append(spectrum[key].imag)
    matrix = torch.stack(rows)
    basis, _r = torch.linalg.qr(matrix.T)
    return basis.T.double()


def project_last_axis(weight, basis):
    """Remove the basis components along the last axis of the weight."""
    work = weight.double()
    return work - (work @ basis.T) @ basis


def project_first_axis(weight, basis):
    """Remove the basis components along the first axis of the weight."""
    work = weight.double()
    return work - basis.T @ (basis @ work)


def embed2_variants(state):
    """The four new variants: unembed, embedding plus unembed, plus the MLP."""
    embedding = state['token_embed.weight'][:common.P]
    keys, _share, _total = fm.top_frequencies(embedding, COVER)
    energy, _power = common.embedding_spectrum(embedding)
    all_keys = list(range(1, (common.P - 1) // 2 + 1))
    non_keys = sorted([k for k in all_keys if k not in set(keys)],
                      key=lambda k: -float(energy[k - 1]))
    control = non_keys[:len(keys)]

    token_keys = token_basis(keys)
    token_control = token_basis(control)
    coeff_keys = coefficient_basis(embedding, keys)
    coeff_control = coefficient_basis(embedding, control)

    def build(token_b, coeff_b):
        merged = {key: value.clone() for key, value in state.items()}
        merged['unembed.weight'] = project_first_axis(
            merged['unembed.weight'], token_b).to(merged['unembed.weight'].dtype)
        merged['token_embed.weight'][:common.P] = project_first_axis(
            embedding, token_b).to(merged['token_embed.weight'].dtype)
        merged['mlp_in.weight'] = project_last_axis(
            merged['mlp_in.weight'], coeff_b).to(merged['mlp_in.weight'].dtype)
        merged['mlp_out.weight'] = project_first_axis(
            merged['mlp_out.weight'], coeff_b).to(merged['mlp_out.weight'].dtype)
        return merged

    unembed_only = {key: value.clone() for key, value in state.items()}
    unembed_only['unembed.weight'] = project_first_axis(
        unembed_only['unembed.weight'], token_keys).to(unembed_only['unembed.weight'].dtype)

    variants = {
        'U': (unembed_only, len(keys)),
        'E+U': (embedding_and_unembed(state, token_keys), len(keys)),
        'E+U+MLP': (build(token_keys, coeff_keys), len(keys)),
        'control_nonkey': (build(token_control, coeff_control), len(control)),
    }
    return variants, keys, control


def embedding_and_unembed(state, token_b):
    """The state with the key frequencies out of the embedding rows and the unembed."""
    merged = {key: value.clone() for key, value in state.items()}
    merged['unembed.weight'] = project_first_axis(
        merged['unembed.weight'], token_b).to(merged['unembed.weight'].dtype)
    merged['token_embed.weight'][:common.P] = project_first_axis(
        state['token_embed.weight'][:common.P], token_b).to(
            merged['token_embed.weight'].dtype)
    return merged


def embed2_job(job):
    """One (seed, variant) of the second embedding ablation."""
    seed = job['seed']
    variant = job['variant']
    device = job['device']
    state, _summary = grokked_state(seed)
    variants, _keys, _control = embed2_variants(state)
    edited, count = variants[variant]
    model = model_from(edited, device)
    model.eval()
    mask = common.mask_for_seed(seed)
    strata, _exposure = common.split_for_mask(mask)
    source_data = common.build_task('source', device)
    target_data = common.build_task('affine', device)
    with torch.no_grad():
        source_evaluation = common.evaluate(model, source_data, strata)
    result = transfer(model, seed, device, strata, target_data,
                      f'embed2_{variant}_seed{seed}', job['steps'])
    row = dict(seed=seed, variant=variant, n_key=count,
               n_control=count if variant == 'control_nonkey' else 0,
               source_dev_mean=source_evaluation['dev_mean'], **result)
    common.log_line(f'[queue2] embed2 seed {seed} {variant}: source dev '
                    f'{row["source_dev_mean"]:.4f} -> transfer {row["final_dev_mean"]}')
    return row


def reused_embed_rows(args):
    """The baseline and key_removed rows of the first ablation, reused as they are."""
    rows = []
    sources = [Path(args.reuse_csv)] if args.reuse_csv else []
    zip_path = Path(args.reuse_zip) if args.reuse_zip else None
    if zip_path and zip_path.exists():
        try:
            with zipfile.ZipFile(zip_path) as archive:
                member = 'runs_nseed_embed_ablation/embed_ablation.csv'
                if member in archive.namelist():
                    rows = list(csv.DictReader(
                        io.StringIO(archive.read(member).decode('utf-8'))))
        except Exception as exc:  # noqa: BLE001 - a missing archive must not stop the step
            common.log_line(f'[queue2] could not read {zip_path}: {exc}')
    for path in sources:
        if path.exists():
            rows.extend(read_csv(path))
    return [row for row in rows if row.get('variant') in ('baseline', 'key_removed')]


def step_embed_ablation2(args, device):
    out_dir = project_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = ['U', 'E+U', 'E+U+MLP', 'control_nonkey']
    done = {(row['seed'], row['variant'])
            for row in read_csv(out_dir / 'embed_ablation2.csv')}
    jobs = []
    for seed in args.seeds:
        for variant in variants:
            if (str(seed), variant) in done:
                continue
            jobs.append(dict(name=f'embed2_{variant}_seed{seed}', kind='embed2',
                             seed=seed, variant=variant, device=device,
                             steps=args.steps, deadline=args.deadline))
    results = run_pool(jobs, device, args.procs, str(common.BATCH_LOG), args.threads)
    rows = [row for row in results if 'error' not in row]
    reused = reused_embed_rows(args)
    merged = reused + rows
    merged.sort(key=lambda row: (str(row['variant']), int(row['seed'])))
    write_csv(out_dir / 'embed_ablation2.csv', EMBED_FIELDS, merged)
    common.log_line(f'[queue2] embed2 wrote {len(merged)} rows to '
                    f'{out_dir / "embed_ablation2.csv"}')
    write_embed2_report(project_path(args.report), merged)
    common.log_line(f'[queue2] embed2 wrote the report to {project_path(args.report)}')
    return merged


def step_embed_ablation(args, device):
    out_dir = project_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = ['baseline', 'key_removed', 'control_nonkey_removed', 'key_only']
    done = {(row['seed'], row['variant'])
            for row in read_csv(out_dir / 'embed_ablation.csv')}
    jobs = []
    for seed in args.seeds:
        for variant in variants:
            if (str(seed), variant) in done:
                common.log_line(f'[queue2] S1 seed {seed} {variant}: already done')
                continue
            jobs.append(dict(name=f'embed_{variant}_seed{seed}', kind='embed',
                             seed=seed, variant=variant, device=device,
                             steps=args.steps, deadline=args.deadline))
    results = run_pool(jobs, device, args.procs, str(common.BATCH_LOG), args.threads)
    rows = [row for row in results if 'error' not in row]
    merged = read_csv(out_dir / 'embed_ablation.csv') + rows
    merged.sort(key=lambda row: (int(row['seed']), str(row['variant'])))
    write_csv(out_dir / 'embed_ablation.csv', EMBED_FIELDS, merged)
    common.log_line(f'[queue2] S1 wrote {len(merged)} rows to '
                    f'{out_dir / "embed_ablation.csv"}')
    return merged


# ---------------------------------------------------------------------------
# S2: copy_a aged sources, frequency structure and norm interventions
# ---------------------------------------------------------------------------
def aged_copy_a_sources():
    """Every aged copy_a source: (seed, init offset, folder, stage record)."""
    runs = common.PARENT / 'runs_nseed'
    folders = [('aged', runs / 'specificity_aged')]
    for offset in (100, 200):
        folders.append((offset, runs / f'specificity_aged_init{offset}'))
    found = []
    for offset, base in folders:
        if not base.is_dir():
            continue
        for path in sorted(base.glob('seed_*_copy_a')):
            checkpoint = path / 'ckpts' / 'age_matched.pt'
            if not checkpoint.exists():
                continue
            summary = common.read_summary(path / 'summary.json')
            jobs = summary.get('jobs', {}) or {}
            record = (jobs.get('source_copy_a', {}) or {}).get('stages', {}) or {}
            aged = record.get('aged', {}) or {}
            seed = int(path.name.split('_')[1])
            found.append(dict(seed=seed, init_offset=offset, folder=path,
                              checkpoint=checkpoint, stage=aged))
    return found


def embedding_fractions(embedding):
    """(fraction of the non-DC energy in the top 5 and top 10 frequencies)."""
    energy, _power = common.embedding_spectrum(embedding)
    total = float(energy.sum())
    ordered = torch.sort(energy, descending=True).values
    if total <= 0:
        return 0.0, 0.0
    return float(ordered[:5].sum()) / total, float(ordered[:10].sum()) / total


def norm_job(job):
    """One aged copy_a source: the measurements and the norm intervention."""
    seed = job['seed']
    device = job['device']
    entry = job['entry']
    payload = common.load_checkpoint(entry['checkpoint'])
    state = payload['model']
    embedding = state['token_embed.weight'][:common.P]
    keys, _share, _total = fm.top_frequencies(embedding, COVER)
    fraction5, fraction10 = embedding_fractions(embedding)
    grokked, _summary = grokked_state(seed)
    grokked_keys, _share2, _total2 = fm.top_frequencies(
        grokked['token_embed.weight'][:common.P], COVER)
    overlap = len(set(keys) & set(grokked_keys))
    norm = common.global_norm(model_from(state, device))

    mask = common.mask_for_seed(seed)
    strata, _exposure = common.split_for_mask(mask)
    target_data = common.build_task('affine', device)

    interventions = [('norm%d' % int(job['target']), job['target'])]
    results = {}
    for label, target in interventions:
        model = model_from(state, device)
        common.rescale_to_norm(model, target)
        outcome = transfer(model, seed, device, strata, target_data,
                           f'norm_{label}_seed{seed}', job['steps'])
        results[label] = outcome
        common.log_line(f'[queue2] S2 seed {seed} init {entry["init_offset"]} {label}: '
                        f'transfer dev {outcome["final_dev_mean"]}')

    row = dict(seed=seed, init_offset=entry['init_offset'],
               age_step=(entry['stage'] or {}).get('source_step', ''),
               global_norm=norm, frac_top5=fraction5, frac_top10=fraction10,
               n_keys=len(keys), key_overlap=overlap,
               aged_transfer_dev_mean=(entry['stage'] or {}).get('final_dev_mean', ''),
               intervention='norm%d' % int(job['target']),
               transfer_dev_mean=results['norm%d' % int(job['target'])]['final_dev_mean'],
               transfer_dev_worst=results['norm%d' % int(job['target'])]['final_dev_worst'])
    return row


def step_copy_a_lownorm(args, device):
    out_dir = project_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = aged_copy_a_sources()
    common.log_line(f'[queue2] S2 found {len(candidates)} aged copy_a sources')
    rows = []
    if not candidates:
        write_csv(out_dir / 'copy_a_lownorm.csv', NORM_FIELDS, rows)
        return rows
    norms = []
    for entry in candidates:
        payload = common.load_checkpoint(entry['checkpoint'])
        norms.append(common.global_norm(model_from(payload['model'], device)))
    median_norm = statistics.median(norms) if norms else 0.0

    selected = []
    for entry, norm in zip(candidates, norms):
        aged_dev = (entry['stage'] or {}).get('final_dev_mean')
        high_norm = (entry['seed'] in HIGH_NORM_SEEDS and norm > median_norm)
        keep = (entry['seed'] in (7, 9, 14)
                or (aged_dev is not None and float(aged_dev) > TRANSFER_THRESHOLD)
                or high_norm)
        if keep:
            selected.append((entry, norm, high_norm))
    common.log_line(f'[queue2] S2 selected {len(selected)} sources, '
                    f'median norm {median_norm:.3f}')

    jobs = []
    for entry, norm, high_norm in selected:
        target = NORM_DOWN if high_norm else NORM_UP
        jobs.append(dict(name=f'norm_seed{entry["seed"]}_init{entry["init_offset"]}',
                         kind='norm', seed=entry['seed'], entry=entry, device=device,
                         steps=args.steps, target=target, deadline=args.deadline))
    results = run_pool(jobs, device, args.procs, str(common.BATCH_LOG), args.threads)
    rows = [row for row in results if 'error' not in row]
    merged = read_csv(out_dir / 'copy_a_lownorm.csv') + rows
    merged.sort(key=lambda row: (int(row['seed']), str(row['init_offset'])))
    write_csv(out_dir / 'copy_a_lownorm.csv', NORM_FIELDS, merged)
    common.log_line(f'[queue2] S2 wrote {len(merged)} rows to '
                    f'{out_dir / "copy_a_lownorm.csv"}')
    return merged


# ---------------------------------------------------------------------------
# S3 / S4: fine sweep over snapshots every 100 steps
# ---------------------------------------------------------------------------
def snapshot_job(job):
    """Retrain one source with snapshots every `snap_every` steps."""
    seed = job['seed']
    device = job['device']
    out_dir = Path(job['out']) / f'seed_{seed}'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'ckpts').mkdir(parents=True, exist_ok=True)
    earlier, summary = grokked_state(seed)
    grok_step = summary.get('grok_step')
    last = job['snap_to']
    if job['extend_grok'] and grok_step and job['snap_to'] <= grok_step:
        last = grok_step + 500
    snapshots = {step: f'snap_{step:07d}.pt'
                 for step in range(job['snap_from'], last + 1, job['snap_every'])}
    if last not in snapshots:
        snapshots[last] = f'snap_{last:07d}.pt'
    mask = common.mask_for_seed(seed)
    strata, _exposure = common.split_for_mask(mask)
    source_data = common.build_task('source', device)
    run = SeedRun(seed, out_dir, threads=1, resume=False, tiers=(), device=device,
                  source_cap=last + 1000, log=common.log_line)
    run._ensure_dirs()
    status, new_grok, _rows = run.train_source(strata, source_data, run._summary(),
                                               snapshot_steps=snapshots,
                                               keep_training_to=last)
    common.log_line(f'[queue2] fine sweep seed {seed}: status {status}, grok step '
                    f'{new_grok} (earlier {grok_step}), {len(snapshots)} snapshots')
    return dict(kind='snapshot', seed=seed, name=f'snapshot_seed{seed}',
                grok_step_earlier=grok_step, grok_step_new=new_grok, status=status,
                snapshots=[str(out_dir / 'ckpts' / name) for name in snapshots.values()])


def fine_job(job):
    """One affine transfer from one snapshot."""
    seed = job['seed']
    device = job['device']
    payload = common.load_checkpoint(job['checkpoint'])
    model = model_from(payload['model'], device)
    model.eval()
    mask = common.mask_for_seed(seed)
    strata, _exposure = common.split_for_mask(mask)
    source_data = common.build_task('source', device)
    target_data = common.build_task('affine', device)
    with torch.no_grad():
        evaluation = common.evaluate(model, source_data, strata)
    result = transfer(model, seed, device, strata, target_data,
                      f'fine_{job["source_step"]}_seed{seed}', job['steps'])
    row = dict(seed=seed, source_step=job['source_step'],
               source_train_acc=evaluation['train_acc'],
               source_dev_mean=evaluation['dev_mean'],
               source_global_norm=common.global_norm(model),
               final_dev_mean=result['final_dev_mean'],
               final_dev_worst=result['final_dev_worst'],
               steps_to_30pct=result['steps_to_30pct'])
    common.log_line(f'[queue2] fine sweep seed {seed} step {job["source_step"]}: '
                    f'transfer dev {row["final_dev_mean"]}')
    return row


def step_fine_sweep(args, device):
    out_dir = project_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_jobs = []
    for seed in args.seeds:
        snapshot_jobs.append(dict(name=f'snapshot_seed{seed}', kind='snapshot',
                                   seed=seed, device=device, steps=args.steps,
                                   snap_from=args.snap_from, snap_to=args.snap_to,
                                   snap_every=args.snap_every, out=str(out_dir),
                                   extend_grok=args.extend_grok,
                                   deadline=args.deadline))
    prepared = run_pool(snapshot_jobs, device, min(args.procs, max(1, len(args.seeds))),
                        str(common.BATCH_LOG), args.threads)
    transfer_jobs = []
    done = {(row['seed'], row['source_step'])
            for row in read_csv(out_dir / 'sweep_summary.csv')}
    for record in prepared:
        if record.get('error') or not record.get('snapshots'):
            continue
        for path in record['snapshots']:
            if out_of_time(args.deadline):
                common.log_line('[queue2] fine sweep deadline reached, '
                                'stopping the transfers')
                break
            step = int(Path(path).stem.split('_')[1])
            if (str(record['seed']), str(step)) in done:
                continue
            transfer_jobs.append(dict(name=f'fine_seed{record["seed"]}', kind='fine',
                                      seed=record['seed'], checkpoint=path,
                                      source_step=step,
                                      device=device, steps=args.steps,
                                      deadline=args.deadline))
    results = run_pool(transfer_jobs, device, args.procs, str(common.BATCH_LOG),
                       args.threads)
    rows = [row for row in results if 'error' not in row]
    merged = {(row['seed'], row['source_step']): row
              for row in read_csv(out_dir / 'sweep_summary.csv')}
    for row in rows:
        merged[(row['seed'], row['source_step'])] = row
    table = [merged[key] for key in sorted(merged, key=lambda k: (int(k[0]), int(k[1])))]
    write_csv(out_dir / 'sweep_summary.csv', SWEEP_FIELDS, table)
    common.log_line(f'[queue2] fine sweep wrote {len(table)} rows to '
                    f'{out_dir / "sweep_summary.csv"}')
    meta = {row['seed']: row for row in read_csv(out_dir / 'fine_sweep_meta.csv')}
    for record in prepared:
        if 'error' in record:
            continue
        meta[str(record['seed'])] = dict(
            seed=record['seed'], grok_step_earlier=record.get('grok_step_earlier'),
            grok_step_new=record.get('grok_step_new'),
            snapshots=len(record.get('snapshots', [])))
    write_csv(out_dir / 'fine_sweep_meta.csv',
              ['seed', 'grok_step_earlier', 'grok_step_new', 'snapshots'],
              [meta[key] for key in sorted(meta, key=int)])
    return table


# ---------------------------------------------------------------------------
# packing and the report
# ---------------------------------------------------------------------------
def step_pack(args):
    """Pack only the new folders into the archive and write the report."""
    root = Path(args.root) if args.root else common.PARENT
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    members = []
    for name in args.folders.split(','):
        base = root / name.strip()
        if not base.is_dir():
            continue
        for path in sorted(base.rglob('*')):
            if path.is_file() and path.suffix.lower() in ('.csv', '.json'):
                members.append(path)
    report = root / 'nseed_backbone' / 'REPORT_queue2.md'
    if report.exists():
        members.append(report)
    with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in members:
            archive.write(path, path.relative_to(root).as_posix())
    common.log_line(f'[queue2] packed {len(members)} files into {out_path}')
    return out_path


def mean_of(rows, field, where=None):
    values = [float(row[field]) for row in rows
              if row.get(field) not in (None, '') and (where is None or where(row))]
    return sum(values) / len(values) if values else float('nan')


def write_embed2_report(path, rows):
    """One table of numbers, ten lines at most."""
    lines = ['# REPORT_embed2 (seeds 5-10, p=59, context c=0, standard 2,000-step '
             'affine transfer)',
             '| variant | n_seeds | source_dev_mean | final_dev_mean |',
             '|---|---|---|---|']
    for variant in ('baseline', 'key_removed', 'U', 'E+U', 'E+U+MLP',
                    'control_nonkey'):
        subset = [row for row in rows if row['variant'] == variant]
        if not subset:
            continue
        lines.append(f"| {variant} | {len(subset)} "
                     f"| {mean_of(subset, 'source_dev_mean'):.4f} "
                     f"| {mean_of(subset, 'final_dev_mean'):.4f} |")
    Path(path).write_text('\n'.join(lines), encoding='utf-8')
    return path


def write_report(path, folders=None):
    """Twenty lines of numbers at most: one table per step."""
    names = (folders or 'runs_nseed_embed_ablation,runs_nseed_copy_a_lownorm,'
                      'runs_nseed_fine').split(',')
    embed = read_csv(Path(names[0].strip()) / 'embed_ablation.csv')
    norms = read_csv(Path(names[1].strip()) / 'copy_a_lownorm.csv')
    fine = read_csv(Path(names[2].strip()) / 'sweep_summary.csv')
    meta = read_csv(Path(names[2].strip()) / 'fine_sweep_meta.csv')

    first = ['## S1 embed_ablation (seeds 5-10)',
             '| variant | n_seeds | source_dev_mean | final_dev_mean |',
             '|---|---|---|---|']
    first_rows = []
    for variant in ('baseline', 'key_removed', 'control_nonkey_removed', 'key_only'):
        subset = [row for row in embed if row['variant'] == variant]
        if not subset:
            continue
        first_rows.append(f"{variant} {len(subset)} "
                          f"{mean_of(subset, 'source_dev_mean'):.4f}/"
                          f"{mean_of(subset, 'final_dev_mean'):.4f}")
        first.append(f"| {variant} | {len(subset)} "
                     f"| {mean_of(subset, 'source_dev_mean'):.4f} "
                     f"| {mean_of(subset, 'final_dev_mean'):.4f} |")

    second = ['## S2 copy_a_lownorm',
              'global_norm = L2 norm over all model parameters at the age-matched '
              'checkpoint (common.global_norm): 21-27 for the copy_a sources, '
              '36-49 for the grokked addition sources',
              '| seed | init | norm | frac_top5 | frac_top10 | keys | overlap '
              '| aged_dev | intervention | transfer_dev |',
              '|---|---|---|---|---|---|---|---|---|---|']
    for row in norms:
        second.append(f"| {row['seed']} | {row['init_offset']} "
                      f"| {float(row['global_norm']):.2f} "
                      f"| {float(row['frac_top5']):.4f} | {float(row['frac_top10']):.4f} "
                      f"| {row['n_keys']} | {row['key_overlap']} "
                      f"| {row['aged_transfer_dev_mean']} | {row['intervention']} "
                      f"| {row['transfer_dev_mean']} |")

    third = ['## S3/S4 fine_sweep',
             '| seed | snapshots | grok_earlier | grok_new | mean_final_dev '
             '| max_final_dev |',
             '|---|---|---|---|---|---|']
    third_rows = []
    for seed in sorted({row['seed'] for row in fine}, key=int):
        subset = [row for row in fine if row['seed'] == seed]
        record = next((row for row in meta if row['seed'] == seed), {})
        third_rows.append(f"{seed} {len(subset)} "
                          f"{record.get('grok_step_earlier', '')}/"
                          f"{record.get('grok_step_new', '')} "
                          f"{mean_of(subset, 'final_dev_mean'):.4f}/"
                          f"{max(float(row['final_dev_mean']) for row in subset):.4f}")
        third.append(f"| {seed} | {len(subset)} | {record.get('grok_step_earlier', '')} "
                     f"| {record.get('grok_step_new', '')} "
                     f"| {mean_of(subset, 'final_dev_mean'):.4f} "
                     f"| {max(float(row['final_dev_mean']) for row in subset):.4f} |")

    if len(first) + len(second) + len(third) <= REPORT_LINES:
        lines = first + second + third
    else:
        lines = (['## S1 embed_ablation | ' + ' | '.join(first_rows)] + second
                 + ['## S3/S4 fine_sweep | ' + ' | '.join(third_rows)])
    Path(path).write_text('\n'.join(lines), encoding='utf-8')
    return path


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['embed_ablation', 'embed_ablation2',
                                          'copy_a_lownorm', 'fine_sweep', 'pack'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[5, 6, 7, 8, 9, 10])
    parser.add_argument('--out', default=None)
    parser.add_argument('--steps', type=int, default=TRANSFER_STEPS)
    parser.add_argument('--procs', type=int, default=1)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default=None)
    parser.add_argument('--deadline', type=float, default=None,
                        help='epoch seconds after which no new work starts')
    parser.add_argument('--snap-from', type=int, default=500)
    parser.add_argument('--snap-to', type=int, default=1500)
    parser.add_argument('--snap-every', type=int, default=100)
    parser.add_argument('--extend-grok', action='store_true',
                        help='extend the snapshot range to grok_step + 500')
    parser.add_argument('--report', default='nseed_backbone/REPORT_queue2.md')
    parser.add_argument('--root', default=None)
    parser.add_argument('--reuse-zip', default='results_queue2.zip')
    parser.add_argument('--reuse-csv', default=None)
    parser.add_argument('--folders', default='runs_nseed_embed_ablation,'
                                             'runs_nseed_copy_a_lownorm,runs_nseed_fine')
    parser.add_argument('--batch-log', default='runs_nseed/batch.log')
    args = parser.parse_args(argv)

    runtime = common.setup(args.threads, args.device)
    device = runtime['device']
    common.set_batch_log(project_path(args.batch_log))
    common.log_line(f'[queue2] step {args.step} device={device}')

    if args.step == 'embed_ablation':
        step_embed_ablation(args, device)
    elif args.step == 'embed_ablation2':
        step_embed_ablation2(args, device)
    elif args.step == 'copy_a_lownorm':
        step_copy_a_lownorm(args, device)
    elif args.step == 'fine_sweep':
        step_fine_sweep(args, device)
    elif args.step == 'pack':
        write_report(project_path(args.report), args.folders)
        step_pack(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
