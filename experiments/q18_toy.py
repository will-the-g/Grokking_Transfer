"""Queue 18, toy track: the sparse parity with a memorized stage, the
polynomial with a memorized stage and the stage timing of the add and the sub
sources.

T1  the sparse parity (n = 30 bits, k = 3, N = 2,000, wd 2.0, the seeds 5 to
    12, the cap of 20,000 steps) with a snapshot every 100 steps. The
    memorized stage is the first snapshot with a train accuracy of at least
    0.99 and a dev accuracy of at most 0.6. The stages are transferred to the
    related (two shared bits), the control (a disjoint subset) and the
    identical target, with the target training sizes 100, 300 and 1,000 and
    2,000 transfer steps. The sources are the memorized, the transition, the
    grokked, the norm control and the random init.

T2  the polynomial (x^2 + y^2 mod 59, coverage 0.5, wd 2.0, the seeds 5 to 9)
    with a snapshot every 50 steps up to the grok step. The memorized stage is
    the earliest snapshot with a train accuracy of at least 0.99 and a dev
    accuracy of at most 0.1. The stages are transferred to affine_poly,
    affine_add and affine_mult on the fractions 0.10, 0.20 and 0.40 of the
    target grid.

T3  the add and the sub sources (coverage 0.5, wd 2.0, the seeds 5 to 9) with
    a snapshot every 100 steps. The grok step is the first step with a dev
    accuracy of at least 0.9, and the snapshots at the grok step and 2,000,
    1,500, 1,000, 500 and 250 steps before it are transferred to affine_add and
    affine_sub.

Only the dev splits are scored. The audit split is never read.

    python nseed_backbone/q18_toy.py --step t1 --procs 4

Produces: Table 3 (the sparse parity and the polynomial).
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import random
import sys
import traceback
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
import torch.nn as nn  # noqa: E402

import common  # noqa: E402
from q16_toy import (ParityTransformer, parity_accuracy, parity_batch,  # noqa: E402
                      parity_label, parity_strings, parity_tokens, poly_data,
                      toy_transfer)
from q17_toy import (append_rows, parity_source_snapshots,  # noqa: E402
                      parity_transfer_n, read_rows, subset_pairs,
                      toy_transfer_subset)
from run_seed import SeedRun  # noqa: E402

PARITY_FIELDS = ['seed', 'source_kind', 'target', 'target_n', 'step', 'dev_acc',
                 'source_final_dev', 'source_grok_step']
POLY_FIELDS = ['source_task', 'target_task', 'seed', 'source_kind',
               'target_fraction', 'step', 'dev_acc', 'source_final_dev',
               'source_grok_step']
TIMING_FIELDS = ['source_task', 'seed', 'steps_before_grok', 'source_step',
                 'source_grok_step', 'source_dev', 'train_acc', 'frac_top5_nodc_a',
                 'frac_top5_nodc_b', 'target_task', 'transfer_dev']
PARITY_SEEDS = tuple(range(5, 13))
POLY_SEEDS = tuple(range(5, 10))
TIMING_TASKS = ('add', 'sub')
TARGET_SIZES = (100, 300, 1000)
FRACTIONS = (0.10, 0.20, 0.40)
TIMING_OFFSETS = (2000, 1500, 1000, 500, 250, 0)
COVERAGE = 0.5
WEIGHT_DECAY = 2.0


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else common.PARENT / path


def prepare_common():
    """The toy settings of queue 18: coverage 0.5 and the weight decay 2.0."""
    common.COVERAGE = COVERAGE
    common.COUNT = int(COVERAGE * common.P * common.P)
    common.WEIGHT_DECAY = WEIGHT_DECAY


def done_marker(source_dir):
    """The marker of a finished unit, so a restart does not repeat it."""
    return Path(source_dir).parent / (Path(source_dir).name + '.done')


def unit_finished(source_dir):
    """True when the unit of this source already holds all of its rows."""
    return done_marker(source_dir).exists()


def mark_finished(source_dir):
    """Mark the unit of this source as finished."""
    marker = done_marker(source_dir)
    marker.write_text('done\n', encoding='utf-8')
    return marker


def fourier_top5_nodc(embeddings):
    """The share of the five strongest non-zero frequencies of the energy.

    The mean over the p rows is removed first, so the zero frequency holds no
    energy and the total is the energy of the non-zero frequencies only.
    """
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    spectrum = torch.fft.rfft(centered.double(), dim=0)
    energy = spectrum.real ** 2 + spectrum.imag ** 2
    nonzero = energy[1:].flatten()
    total = float(nonzero.sum())
    if total <= 0 or nonzero.numel() == 0:
        return 0.0
    top = torch.topk(nonzero, min(5, nonzero.numel())).values
    return float(top.sum()) / total


def embedding_concentration_nodc(model):
    """The concentration of the a-token and the b-token embeddings."""
    table = model.token_embed.weight.detach()
    pos = model.pos_embed.detach()
    return (fourier_top5_nodc(table[:model.p] + pos[1]),
            fourier_top5_nodc(table[:model.p] + pos[2]))


# ---------------------------------------------------------------------------
# the toy sources
# ---------------------------------------------------------------------------
def train_toy_source(seed, task, out_dir, device, data=None, eval_every=50,
                     ckpt_every=50, cap=20000):
    """Train one toy source and label its stages.

    The training is the recipe of the a+b sources: a full-batch AdamW with the
    learning rate 1e-3, the betas (0.9, 0.98) and the weight decay 2.0. The
    model is a snapshot every `ckpt_every` steps and is evaluated every
    `eval_every` steps.
    """
    out_dir = Path(out_dir)
    summary_path = out_dir / 'summary.json'
    summary = common.read_summary(summary_path)
    if summary.get('source_finished'):
        return summary
    run = SeedRun(seed, out_dir, threads=1, resume=False, tiers=(), device=device,
                  source_cap=cap, eval_every=eval_every, ckpt_every=ckpt_every,
                  log=common.log_line)
    run._ensure_dirs()
    mask = common.mask_for_seed(seed)
    strata, _exposure = common.split_for_mask(mask, seed)
    if data is None:
        data = common.build_task(task, device)
    status, grok_step, rows = run.train_source(strata, data, run._summary())
    stages = run.select_stages(rows, grok_step, status)
    summary = run._summary()
    summary['stages'] = stages
    summary['source_finished'] = True
    summary['source_task'] = task
    summary['source_final_dev'] = rows[-1].get('dev_mean') if rows else ''
    summary['status'] = status
    summary['grok_step'] = grok_step
    run._write_summary(summary)
    common.log_line(f'[q18] the source {task} seed {seed}: status {status} '
                    f'grok_step {grok_step} final dev '
                    f'{summary["source_final_dev"]}')
    return summary


def source_rows_of(seed_dir):
    """The source rows of the metrics of a source, ordered by the step."""
    rows = common.read_metrics(Path(seed_dir) / 'metrics.csv')
    rows = [row for row in rows if row.get('stage') == 'source']
    rows.sort(key=lambda row: int(row['step']))
    return rows


def stage_states(seed_dir, summary, device, memorized_ceiling=0.1):
    """The states of the stages of a source.

    The memorized stage is the earliest snapshot with a train accuracy of at
    least 0.99 and a dev accuracy of at most `memorized_ceiling`. The
    transition is the snapshot whose dev accuracy is the closest to 0.5 inside
    [0.25, 0.75], the same rule as in the earlier sweeps. The norm control is
    the memorized checkpoint rescaled to the global norm of the grokked one and
    the random init is a fresh model of the same seed.
    """
    seed_dir = Path(seed_dir)
    rows = source_rows_of(seed_dir)
    available = {}
    for path in sorted((seed_dir / 'ckpts').glob('step_*.pt')):
        available[int(path.stem.split('_')[-1])] = path
    stages = {}
    for row in rows:
        train_acc = common.as_float(row.get('train_acc'))
        dev_mean = common.as_float(row.get('dev_mean'))
        step = common.as_int(row.get('step'))
        if train_acc is None or dev_mean is None or step not in available:
            continue
        if train_acc >= 0.99 and dev_mean <= memorized_ceiling:
            stages['memorized'] = (step, available[step])
            break
    candidates = []
    for row in rows:
        dev_mean = common.as_float(row.get('dev_mean'))
        step = common.as_int(row.get('step'))
        if dev_mean is None or step not in available:
            continue
        if 0.25 <= dev_mean <= 0.75:
            candidates.append((abs(dev_mean - 0.5), step))
    if candidates:
        step = min(candidates)[1]
        stages['transition'] = (step, available[step])
    grokked = seed_dir / 'ckpts' / 'grokked.pt'
    if grokked.exists():
        stages['grokked'] = (summary.get('grok_step'), grokked)
    states = {}
    for name, (_step, path) in stages.items():
        states[name] = common.load_checkpoint(path)['model']
    if 'memorized' in states and 'grokked' in states:
        model = common.make_model(common.CONFIG)
        model.load_state_dict(states['memorized'])
        grokked_model = common.make_model(common.CONFIG)
        grokked_model.load_state_dict(states['grokked'])
        common.rescale_to_norm(model, common.global_norm(grokked_model))
        states['norm_control'] = {key: value.detach().cpu() for key, value
                                  in model.state_dict().items()}
        del model, grokked_model
        torch.cuda.empty_cache()
    model = common.init_model(seed_of(seed_dir), device)
    states['random_init'] = {key: value.detach().cpu() for key, value
                              in model.state_dict().items()}
    del model
    torch.cuda.empty_cache()
    return states


def seed_of(seed_dir):
    """The seed of a source directory."""
    name = Path(seed_dir).name
    if name.startswith('seed_'):
        return int(name.split('_')[-1])
    return 0


def parity_transfer_fast(state, n_bits, seed, train_strings, dev_strings, subset,
                         device, steps=2000, eval_every=100):
    """The transfer of a parity stage with the tokens built once.

    The same transfer as `q17_toy.parity_transfer_n`: a fresh AdamW with the
    learning rate 1e-3, the weight decay 1.0 and the betas (0.9, 0.98), the
    batches of 512 strings drawn by the same generator and the dev accuracy
    every `eval_every` steps. The tokens and the labels of the training strings
    are built once instead of at every step, which is the same computation with
    the same numbers.
    """
    model = ParityTransformer(n_bits=n_bits)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    tokens = torch.stack([parity_tokens(bits, device) for bits in train_strings])
    labels = torch.tensor([parity_label(bits, subset) for bits in train_strings],
                          dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0,
                                  betas=(0.9, 0.98))
    generator = torch.Generator(device=device).manual_seed(int(seed + 31))
    rows = []
    step = 0
    while step < steps:
        index = torch.randint(0, len(train_strings), (512,), generator=generator,
                              device=device)
        loss = nn.functional.cross_entropy(model(tokens[index]), labels[index])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % eval_every == 0 or step == steps:
            rows.append(dict(step=step,
                             dev_mean=parity_accuracy(model, dev_strings, subset,
                                                       device)))
    del model
    torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------------------
# STEP T1: the sparse parity with a memorized stage
# ---------------------------------------------------------------------------
def unit_t1(job):
    """The parity sources and the transfers of one seed."""
    args = job['args']
    device = job['device']
    seed = job['seed']
    path = job['path']
    notes = []
    source_dir = Path(args.sources) / 'parity' / 'N2000_wd2.0' / f'seed_{seed}'
    if unit_finished(source_dir):
        return [f'parity seed {seed}: the unit was already finished']
    summary = parity_source_snapshots(seed, 2000, WEIGHT_DECAY, source_dir,
                                      args.cap, device)
    stages = summary.get('stages', {}) or {}
    if 'memorized' not in stages:
        notes.append(f'parity seed {seed}: no memorized snapshot')
    subset = summary['subset']
    n_bits = summary['n_bits']
    dev = parity_strings(5000, n_bits, 30000 + seed)
    new_bits = [index for index in range(n_bits) if index not in subset]
    related = sorted(random.Random(seed + 7).sample(subset[:2] + new_bits[:1], 3))
    control = sorted(random.Random(seed + 11).sample(new_bits, 3))
    targets = {'related': related, 'control': control, 'identical': subset}
    states = {}
    for name in ('memorized', 'transition', 'grokked'):
        if name in stages:
            states[name] = torch.load(source_dir / stages[name]['ckpt'],
                                      map_location='cpu')
    if 'memorized' in states and 'grokked' in states:
        model = ParityTransformer(n_bits=n_bits)
        model.load_state_dict(states['memorized'])
        model = model.to(device)
        grokked_model = ParityTransformer(n_bits=n_bits)
        grokked_model.load_state_dict(states['grokked'])
        grokked_model = grokked_model.to(device)
        common.rescale_to_norm(model, common.global_norm(grokked_model))
        states['norm_control'] = {key: value.detach().cpu() for key, value
                                  in model.state_dict().items()}
        del model, grokked_model
        torch.cuda.empty_cache()
    fresh = ParityTransformer(n_bits=n_bits)
    states['random_init'] = {key: value.detach().cpu() for key, value
                              in fresh.state_dict().items()}
    for kind, state in states.items():
        for target, target_subset in targets.items():
            for target_n in TARGET_SIZES:
                train = parity_strings(target_n, n_bits, 40000 + seed)
                rows = parity_transfer_fast(state, n_bits, seed, train, dev,
                                            target_subset, device, steps=args.steps,
                                            eval_every=args.eval_every)
                final = rows[-1]['dev_mean'] if rows else ''
                append_rows(path, PARITY_FIELDS, [dict(
                    seed=seed, source_kind=kind, target=target, target_n=target_n,
                    step=args.steps, dev_acc=final,
                    source_final_dev=summary.get('final_dev'),
                    source_grok_step=summary.get('grok_step'))])
                common.log_line(f'[q18] t1 parity {kind} {target} n {target_n} '
                                f'seed {seed}: dev {final}')
    mark_finished(source_dir)
    return notes


# ---------------------------------------------------------------------------
# STEP T2: the polynomial with a memorized stage
# ---------------------------------------------------------------------------
def unit_t2(job):
    """The polynomial source and the transfers of one seed."""
    args = job['args']
    device = job['device']
    seed = job['seed']
    path = job['path']
    notes = []
    source_dir = Path(args.sources) / 'poly' / 'cov0.5_wd2.0' / f'seed_{seed}'
    if unit_finished(source_dir):
        return [f'poly seed {seed}: the unit was already finished']
    summary = train_toy_source(seed, 'poly', source_dir, device,
                               data=poly_data(device, 'poly'), eval_every=50,
                               ckpt_every=50, cap=args.cap)
    states = stage_states(source_dir, summary, device, memorized_ceiling=0.1)
    if 'memorized' not in states:
        notes.append(f'poly seed {seed}: no memorized snapshot')
    targets = {'affine_poly': poly_data(device, 'affine_poly'),
               'affine_add': common.build_task('affine_add', device),
               'affine_mult': common.build_task('affine_mult', device)}
    mask = common.mask_for_seed(seed)
    strata, _exposure = common.split_for_mask(mask, seed)
    for kind, state in states.items():
        for target, target_data in targets.items():
            for fraction in FRACTIONS:
                train_rows = subset_pairs(strata, fraction, seed, common.P)
                rows = toy_transfer_subset(state, seed, target_data, strata, device,
                                           f't18_poly_{kind}_{target}_{fraction}'
                                           f'_seed{seed}', train_rows,
                                           steps=args.steps,
                                           eval_every=args.eval_every)
                final = rows[-1]['dev_mean'] if rows else ''
                append_rows(path, POLY_FIELDS, [dict(
                    source_task='poly', target_task=target, seed=seed,
                    source_kind=kind, target_fraction=fraction, step=args.steps,
                    dev_acc=final, source_final_dev=summary.get('source_final_dev'),
                    source_grok_step=summary.get('grok_step'))])
                common.log_line(f'[q18] t2 poly {kind} {target} {fraction} '
                                f'seed {seed}: dev {final}')
    mark_finished(source_dir)
    return notes


# ---------------------------------------------------------------------------
# STEP T3: the stage timing of the add and the sub sources
# ---------------------------------------------------------------------------
def unit_t3(job):
    """One source of the stage timing and its transfers."""
    args = job['args']
    device = job['device']
    seed = job['seed']
    task_name = job['task']
    path = job['path']
    notes = []
    source_dir = Path(args.sources) / task_name / 'cov0.5_wd2.0' / f'seed_{seed}'
    if unit_finished(source_dir):
        return [f'{task_name} seed {seed}: the unit was already finished']
    summary = train_toy_source(seed, task_name, source_dir, device,
                               data=common.build_task(task_name, device),
                               eval_every=100, ckpt_every=100, cap=args.cap)
    rows = source_rows_of(source_dir)
    grok_step = None
    for row in rows:
        dev_mean = common.as_float(row.get('dev_mean'))
        if dev_mean is not None and dev_mean >= 0.9:
            grok_step = common.as_int(row.get('step'))
            break
    if grok_step is None:
        notes.append(f'{task_name} seed {seed}: the source never reached a dev of 0.9')
        return notes
    available = sorted(int(path.stem.split('_')[-1])
                       for path in (source_dir / 'ckpts').glob('step_*.pt'))
    if not available:
        notes.append(f'{task_name} seed {seed}: no snapshot')
        return notes
    targets = {'affine_add': common.build_task('affine_add', device),
               'affine_sub': common.build_task('affine_sub', device)}
    mask = common.mask_for_seed(seed)
    strata, _exposure = common.split_for_mask(mask, seed)
    for offset in TIMING_OFFSETS:
        target_step = grok_step - offset
        if target_step < 0:
            notes.append(f'{task_name} seed {seed}: the step {target_step} is before '
                        f'the start of the source')
            continue
        step = min(available, key=lambda value: (abs(value - target_step), value))
        checkpoint = source_dir / 'ckpts' / f'step_{step:07d}.pt'
        if not checkpoint.exists():
            notes.append(f'{task_name} seed {seed}: the snapshot at the step {step} '
                        f'is missing')
            continue
        state = common.load_checkpoint(checkpoint)['model']
        model = common.make_model(common.CONFIG)
        model.load_state_dict(state)
        model = model.to(device)
        model.eval()
        train_rows = common.index_for(strata['train'], device)
        train_acc = common.accuracy_on(model, targets['affine_add'], train_rows)
        dev_rows = common.context_rows(strata['dev'], 0, device)
        source_dev = common.accuracy_on(model, targets['affine_add'], dev_rows)
        frac_a, frac_b = embedding_concentration_nodc(model)
        del model
        torch.cuda.empty_cache()
        for target_name, target_data in targets.items():
            transfer_rows = toy_transfer(state, seed, target_data, strata, device,
                                         f't18_timing_{task_name}_{offset}_'
                                         f'{target_name}_seed{seed}',
                                         steps=args.steps, eval_every=args.eval_every)
            final = transfer_rows[-1]['dev_mean'] if transfer_rows else ''
            append_rows(path, TIMING_FIELDS, [dict(
                source_task=task_name, seed=seed, steps_before_grok=offset,
                source_step=step, source_grok_step=grok_step,
                source_dev=source_dev, train_acc=train_acc,
                frac_top5_nodc_a=frac_a, frac_top5_nodc_b=frac_b,
                target_task=target_name, transfer_dev=final)])
            common.log_line(f'[q18] t3 {task_name} {offset} {target_name} '
                            f'seed {seed} step {step}: dev {final}')
    mark_finished(source_dir)
    return notes


# ---------------------------------------------------------------------------
# the pool
# ---------------------------------------------------------------------------
UNITS = {'t1': unit_t1, 't2': unit_t2, 't3': unit_t3}


def worker_init(device, batch_log, threads):
    prepare_common()
    common.setup(threads, device)


def worker(job):
    try:
        notes = UNITS[job['step']](job)
        return dict(job=job, notes=notes, error=None)
    except Exception as exc:  # noqa: BLE001 - one unit must not kill the pool
        common.log_line(f'[q18] {job["step"]} seed {job["seed"]} FAILED '
                        f'{type(exc).__name__}: {exc}')
        common.log_line(traceback.format_exc())
        return dict(job=job, notes=[], error=str(exc))


def run_pool(jobs, device, procs, threads=1):
    """Run the jobs over a pool of worker processes."""
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


def ensure_header(path, fields):
    """Create the CSV with its header before the workers append to it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open('w', newline='', encoding='utf-8') as stream:
            csv.DictWriter(stream, fieldnames=fields).writeheader()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', default='t1', choices=sorted(UNITS))
    parser.add_argument('--out', default='runs_nseed_queue18')
    parser.add_argument('--sources', default='runs_nseed_queue18/sources')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--cap', type=int, default=20000)
    parser.add_argument('--procs', type=int, default=4)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.smoke:
        return run_smoke(args, device)
    prepare_common()
    out_dir = project_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.sources).mkdir(parents=True, exist_ok=True)

    if args.step == 't1':
        fields = PARITY_FIELDS
        seeds = list(PARITY_SEEDS)
        units = [dict(step='t1', seed=seed) for seed in seeds]
    elif args.step == 't2':
        fields = POLY_FIELDS
        seeds = list(POLY_SEEDS)
        units = [dict(step='t2', seed=seed) for seed in seeds]
    else:
        fields = TIMING_FIELDS
        seeds = list(POLY_SEEDS)
        units = [dict(step='t3', seed=seed, task=task) for task in TIMING_TASKS
                 for seed in seeds]
    path = out_dir / {'t1': 't18_parity.csv', 't2': 't18_poly.csv',
                      't3': 't18_timing.csv'}[args.step]
    ensure_header(path, fields)

    jobs = [dict(args=args, device=device, path=path, **unit) for unit in units]
    print(f'{len(jobs)} units of {args.step}', flush=True)
    results = run_pool(jobs, device, args.procs, args.threads)
    failed = [result for result in results if result.get('error')]
    notes = []
    for result in results:
        notes.extend(result.get('notes') or [])
    for note in notes:
        common.log_line(f'[q18] {args.step} note: {note}')
    for result in failed:
        job = result['job']
        common.log_line(f'[q18] FAILED {args.step} seed {job["seed"]}: '
                        f'{result["error"]}')
    print(f'{len(results) - len(failed)} units finished, {len(failed)} failed',
          flush=True)
    return 0


def run_smoke(args, device):
    """The parity model, the concentration and one small transfer."""
    prepare_common()
    summary = parity_source_snapshots(5, 2000, WEIGHT_DECAY,
                                      Path(args.sources) / 'smoke' / 'seed_5', 200,
                                      device)
    print(f'the smoke parity stages: {sorted(summary.get("stages") or {})}', flush=True)
    model = common.make_model(common.CONFIG)
    frac_a, frac_b = embedding_concentration_nodc(model)
    print(f'the smoke concentration without the zero frequency: {frac_a:.4f} '
          f'{frac_b:.4f}', flush=True)
    dev = parity_strings(500, summary['n_bits'], 30000 + 5)
    parity_model = ParityTransformer(n_bits=summary['n_bits'])
    state = {key: value.detach().cpu() for key, value
             in parity_model.state_dict().items()}
    rows = parity_transfer_n(state, summary['n_bits'], 5,
                             parity_strings(100, summary['n_bits'], 40000 + 5), dev,
                             summary['subset'], device, steps=20, eval_every=10)
    print(f'the smoke transfer: {rows[-1]["dev_mean"]:.4f}', flush=True)
    strata, _exposure = common.split_for_mask(common.mask_for_seed(5), 5)
    pairs = subset_pairs(strata, 0.10, 5, common.P)
    print(f'the smoke fraction 0.10 holds {len(pairs)} rows', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
