"""Queue 19, toy track: the small sources that memorize and the sub timing.

T1  the sparse parity with small training sets (n = 30 bits, k = 3, wd 2.0,
    the seeds 5 to 12, the training sets of 300 and 600 examples, at most 5,000
    steps, a snapshot every 50 steps). The memorized snapshot is the first
    snapshot with a train accuracy of at least 0.99; it is only kept when its
    dev accuracy is at most 0.6. The norm control is the memorized snapshot
    rescaled to the global parameter norm of the grokked source of queue 18 of
    the same seed. The stages are transferred to the related, the control and
    the identical target, with the target training sizes 100, 300 and 1,000.

T2  the polynomial (x^2 + y^2 mod 59) with the coverages 0.15 and 0.25, the
    seeds 5 to 9, at most 5,000 steps, a snapshot every 50 steps. The
    memorized snapshot is the first snapshot with a train accuracy of at least
    0.99 and it is only kept when its dev accuracy is at most 0.1. The stages
    are transferred to affine_poly, affine_add and affine_mult on the
    fractions 0.10, 0.20 and 0.40.

T3  the sub sources (a - b mod 59, coverage 0.5, wd 2.0, the seeds 5 to 9)
    with the source dev and the source train accuracy computed with the sub
    labels, which is what the rows of queue 18 and of queue 17 did wrong. The
    grok step is the first step whose dev accuracy is at least 0.9; the
    snapshots at the grok step and 2,000, 1,500, 1,000, 500 and 250 steps
    before it are transferred to affine_add and affine_sub. The random init is
    transferred to the same targets.

Only the dev splits are scored. The audit split is never read.

Produces: Table 3 (the memorizing and the norm-matched controls).
"""
from __future__ import annotations

import argparse
import collections
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
from q16_toy import (ParityTransformer, parity_accuracy, parity_label,  # noqa: E402
                      parity_strings, parity_tokens, poly_data, toy_transfer)
from q17_toy import subset_pairs, toy_transfer_subset  # noqa: E402
import q18_toy  # noqa: E402

PARITY_FIELDS = ['seed', 'source_kind', 'source_n', 'target', 'target_n', 'step',
                 'dev_acc', 'source_train_acc', 'source_dev_acc', 'source_norm']
POLY_FIELDS = ['seed', 'source_kind', 'source_coverage', 'target_task',
               'target_fraction', 'step', 'dev_acc', 'source_train_acc',
               'source_dev_acc', 'source_norm']
SUB_FIELDS = ['seed', 'source_kind', 'steps_before_grok', 'source_step',
              'source_grok_step', 'source_dev', 'source_train_acc', 'target_task',
              'transfer_dev']
PARITY_SEEDS = tuple(range(5, 13))
POLY_SEEDS = tuple(range(5, 10))
SUB_SEEDS = tuple(range(5, 10))
PARITY_SIZES = (300, 600)
COVERAGES = (0.15, 0.25)
TARGET_SIZES = (100, 300, 1000)
FRACTIONS = (0.10, 0.20, 0.40)
TIMING_OFFSETS = (2000, 1500, 1000, 500, 250, 0)
WEIGHT_DECAY = 2.0
CAP = 5000
SNAPSHOT_EVERY = 50
MEMORIZED_TRAIN = 0.99
MEMORIZED_DEV_PARITY = 0.6
MEMORIZED_DEV_POLY = 0.1
QUEUE18_SOURCES = 'runs_nseed_queue18/sources'


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else common.PARENT / path


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


def ensure_header(path, fields):
    """Create the CSV with its header before the workers append to it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open('w', newline='', encoding='utf-8') as stream:
            csv.DictWriter(stream, fieldnames=fields).writeheader()


def state_norm(state):
    """The global parameter norm of a state dictionary."""
    total = 0.0
    for value in state.values():
        if not torch.is_floating_point(value):
            continue
        total += float(value.detach().float().pow(2).sum())
    return total ** 0.5


def queue18_grokked_norm(source_dir):
    """The global parameter norm of the grokked source of queue 18.

    The norm is read from the summary of the source and, when the summary does
    not hold it, from the grokked checkpoint itself. None is returned when the
    source or the checkpoint is missing.
    """
    source_dir = Path(source_dir)
    summary_path = source_dir / 'summary.json'
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
    except Exception:  # noqa: BLE001 - a broken summary means no norm
        return None
    stages = summary.get('stages') or {}
    if 'grokked' not in stages:
        return None
    record = stages['grokked'] or {}
    if record.get('global_norm') is not None:
        return float(record['global_norm'])
    name = record.get('ckpt')
    if not name:
        return None
    checkpoint = source_dir / 'ckpts' / name
    if not checkpoint.exists():
        checkpoint = source_dir / name
    if not checkpoint.exists():
        return None
    try:
        payload = torch.load(checkpoint, map_location='cpu')
    except Exception:  # noqa: BLE001 - a broken checkpoint means no norm
        return None
    state = payload.get('model', payload)
    return state_norm(state)


def parity_targets(seed, n_bits, subset):
    """The target subsets of queue 18 for the seed."""
    new_bits = [index for index in range(n_bits) if index not in subset]
    related = sorted(random.Random(seed + 7).sample(subset[:2] + new_bits[:1], 3))
    control = sorted(random.Random(seed + 11).sample(new_bits, 3))
    return {'related': related, 'control': control, 'identical': list(subset)}


def parity_source_small(seed, n_pairs, out_dir, cap, device, n_bits=30, k=3,
                        wd=WEIGHT_DECAY):
    """A parity source with a snapshot every 50 steps.

    The training is the recipe of the earlier sweeps: a full AdamW with the
    learning rate 1e-3, the betas (0.9, 0.98) and the weight decay 2.0, with
    the batches of 512 strings. The tokens and the labels of the training
    strings are built once instead of at every step, which is the same
    computation with the same numbers.

    The summary holds the first snapshot with a train accuracy of at least
    0.99 and whether its dev accuracy is at most the ceiling.
    """
    out_dir = Path(out_dir)
    summary_path = out_dir / 'summary.json'
    summary = common.read_summary(summary_path)
    if summary.get('source_finished'):
        return summary
    out_dir.mkdir(parents=True, exist_ok=True)
    subset = sorted(random.Random(seed).sample(range(n_bits), k))
    train = parity_strings(n_pairs, n_bits, 20000 + seed)
    dev = parity_strings(5000, n_bits, 30000 + seed)
    tokens = torch.stack([parity_tokens(bits, device) for bits in train])
    labels = torch.tensor([parity_label(bits, subset) for bits in train],
                          dtype=torch.long, device=device)
    model = ParityTransformer(n_bits=n_bits).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd,
                                  betas=(0.9, 0.98))
    generator = torch.Generator(device=device).manual_seed(int(1000 + seed))
    history = []
    memorized = None
    step = 0
    while step < cap:
        index = torch.randint(0, len(train), (512,), generator=generator,
                              device=device)
        loss = nn.functional.cross_entropy(model(tokens[index]), labels[index])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % SNAPSHOT_EVERY == 0:
            train_acc = parity_accuracy(model, train[:1000], subset, device)
            dev_acc = parity_accuracy(model, dev, subset, device)
            history.append(dict(step=step, train_acc=train_acc, dev_acc=dev_acc))
            torch.save({key: value.detach().cpu() for key, value
                        in model.state_dict().items()},
                       out_dir / f'step_{step:06d}.pt')
            if memorized is None and train_acc >= MEMORIZED_TRAIN:
                memorized = dict(step=step, train_acc=train_acc, dev_acc=dev_acc)
    payload = dict(source_finished=True, seed=seed, n_pairs=n_pairs, wd=wd,
                   subset=subset, n_bits=n_bits, k=k, steps=step,
                   memorized=memorized, history=history)
    summary_path.write_text(json.dumps(payload, indent=2, default=str),
                            encoding='utf-8')
    common.log_line(f'[q19] the parity source seed {seed} N {n_pairs}: '
                    f'memorized {memorized}')
    del model
    torch.cuda.empty_cache()
    return payload


def memorized_of(source_dir):
    """The first snapshot with a train accuracy of at least 0.99."""
    source_dir = Path(source_dir)
    rows = [row for row in common.read_metrics(source_dir / 'metrics.csv')
            if row.get('stage') == 'source']
    rows.sort(key=lambda row: int(row['step']))
    available = {int(path.stem.split('_')[-1]): path
                 for path in (source_dir / 'ckpts').glob('step_*.pt')}
    for row in rows:
        train_acc = common.as_float(row.get('train_acc'))
        dev_mean = common.as_float(row.get('dev_mean'))
        step = common.as_int(row.get('step'))
        if train_acc is None or step not in available:
            continue
        if train_acc >= MEMORIZED_TRAIN:
            return dict(step=step, train_acc=train_acc, dev_acc=dev_mean,
                        ckpt=available[step])
    return None


# ---------------------------------------------------------------------------
# STEP T1: the small parity sources
# ---------------------------------------------------------------------------
def unit_t1(job):
    """The parity sources of the sizes of one seed and their transfers."""
    args = job['args']
    device = job['device']
    seed = job['seed']
    path = job['path']
    notes = []
    n_bits = 30
    subset = sorted(random.Random(seed).sample(range(n_bits), 3))
    dev = parity_strings(5000, n_bits, 30000 + seed)
    targets = parity_targets(seed, n_bits, subset)
    for n_pairs in PARITY_SIZES:
        source_dir = Path(args.sources) / 'parity' / f'N{n_pairs}' / f'seed_{seed}'
        summary = parity_source_small(seed, n_pairs, source_dir, CAP, device,
                                      n_bits=n_bits)
        memorized = summary.get('memorized')
        if not memorized:
            notes.append(f'parity seed {seed} N {n_pairs}: no snapshot reaches a '
                         f'train accuracy of {MEMORIZED_TRAIN}')
            continue
        if memorized['dev_acc'] > MEMORIZED_DEV_PARITY:
            notes.append(f'parity seed {seed} N {n_pairs}: not memorized, the dev '
                         f'accuracy of the first snapshot with a train accuracy of '
                         f'{MEMORIZED_TRAIN} is {memorized["dev_acc"]:.4f}')
            continue
        state = torch.load(source_dir / f'step_{memorized["step"]:06d}.pt',
                           map_location='cpu')
        norm = state_norm(state)
        kinds = {'memorized': (state, norm)}
        grokked_norm = queue18_grokked_norm(
            Path(QUEUE18_SOURCES) / 'parity' / 'N2000_wd2.0' / f'seed_{seed}')
        if grokked_norm is None:
            notes.append(f'parity seed {seed} N {n_pairs}: the grokked source of '
                         f'queue 18 is missing, the norm control is skipped')
        else:
            model = ParityTransformer(n_bits=n_bits)
            model.load_state_dict(state)
            common.rescale_to_norm(model, grokked_norm)
            kinds['norm_control'] = (
                {key: value.detach().cpu() for key, value
                 in model.state_dict().items()}, grokked_norm)
            del model
        for kind, (kind_state, kind_norm) in kinds.items():
            for target, target_subset in targets.items():
                for target_n in TARGET_SIZES:
                    train = parity_strings(target_n, n_bits, 40000 + seed)
                    rows = q18_toy.parity_transfer_fast(kind_state, n_bits, seed,
                                                        train, dev, target_subset,
                                                        device, steps=args.steps,
                                                        eval_every=args.eval_every)
                    final = rows[-1]['dev_mean'] if rows else ''
                    append_rows(path, PARITY_FIELDS, [dict(
                        seed=seed, source_kind=kind, source_n=n_pairs,
                        target=target, target_n=target_n, step=args.steps,
                        dev_acc=final, source_train_acc=memorized['train_acc'],
                        source_dev_acc=memorized['dev_acc'], source_norm=kind_norm)])
                    common.log_line(f'[q19] t1 parity {kind} N {n_pairs} {target} '
                                    f'n {target_n} seed {seed}: dev {final}')
    return notes


# ---------------------------------------------------------------------------
# STEP T2: the small polynomial sources
# ---------------------------------------------------------------------------
def unit_t2(job):
    """The polynomial sources of the coverages of one seed and their transfers."""
    args = job['args']
    device = job['device']
    seed = job['seed']
    path = job['path']
    notes = []
    target_data = {'affine_poly': poly_data(device, 'affine_poly'),
                   'affine_add': common.build_task('affine_add', device),
                   'affine_mult': common.build_task('affine_mult', device)}
    for coverage in COVERAGES:
        common.COVERAGE = coverage
        common.COUNT = int(coverage * common.P * common.P)
        mask = common.mask_for_seed(seed)
        strata, _exposure = common.split_for_mask(mask, seed)
        source_dir = Path(args.sources) / 'poly' / f'cov{coverage}' / f'seed_{seed}'
        summary = q18_toy.train_toy_source(seed, 'poly', source_dir, device,
                                           data=poly_data(device, 'poly'),
                                           eval_every=SNAPSHOT_EVERY,
                                           ckpt_every=SNAPSHOT_EVERY, cap=CAP)
        memorized = memorized_of(source_dir)
        if not memorized:
            notes.append(f'poly seed {seed} coverage {coverage}: no snapshot reaches '
                         f'a train accuracy of {MEMORIZED_TRAIN}')
            continue
        if memorized['dev_acc'] > MEMORIZED_DEV_POLY:
            notes.append(f'poly seed {seed} coverage {coverage}: not memorized, the '
                         f'dev accuracy of the first snapshot with a train accuracy '
                         f'of {MEMORIZED_TRAIN} is {memorized["dev_acc"]:.4f}')
            continue
        state = common.load_checkpoint(memorized['ckpt'])['model']
        norm = state_norm(state)
        kinds = {'memorized': (state, norm)}
        grokked_norm = queue18_grokked_norm(
            Path(QUEUE18_SOURCES) / 'poly' / 'cov0.5_wd2.0' / f'seed_{seed}')
        if grokked_norm is None:
            notes.append(f'poly seed {seed} coverage {coverage}: the grokked source '
                         f'of queue 18 is missing, the norm control is skipped')
        else:
            model = common.make_model(common.CONFIG)
            model.load_state_dict(state)
            common.rescale_to_norm(model, grokked_norm)
            kinds['norm_control'] = (
                {key: value.detach().cpu() for key, value
                 in model.state_dict().items()}, grokked_norm)
            del model
        for kind, (kind_state, kind_norm) in kinds.items():
            for target, data in target_data.items():
                for fraction in FRACTIONS:
                    train_rows = subset_pairs(strata, fraction, seed, common.P)
                    rows = toy_transfer_subset(kind_state, seed, data, strata, device,
                                               f't19_poly_{kind}_{target}_{fraction}'
                                               f'_seed{seed}', train_rows,
                                               steps=args.steps,
                                               eval_every=args.eval_every)
                    final = rows[-1]['dev_mean'] if rows else ''
                    append_rows(path, POLY_FIELDS, [dict(
                        seed=seed, source_kind=kind, source_coverage=coverage,
                        target_task=target, target_fraction=fraction, step=args.steps,
                        dev_acc=final, source_train_acc=memorized['train_acc'],
                        source_dev_acc=memorized['dev_acc'], source_norm=kind_norm)])
                    common.log_line(f'[q19] t2 poly {kind} {target} {fraction} '
                                    f'seed {seed}: dev {final}')
    return notes


# ---------------------------------------------------------------------------
# STEP T3: the sub timing, evaluated with the sub labels
# ---------------------------------------------------------------------------
def sub_evaluation(model, data, strata, device):
    """The train accuracy and the dev accuracy of a model on the sub labels."""
    train_rows = common.index_for(strata['train'], device)
    dev_rows = common.context_rows(strata['dev'], 0, device)
    train_acc = common.accuracy_on(model, data, train_rows)
    dev_acc = common.accuracy_on(model, data, dev_rows)
    return train_acc, dev_acc


def unit_t3(job):
    """The sub source of one seed: the preparation or the transfers.

    The preparation trains the source when it is missing, finds the grok step
    and checks the snapshot at that step with the same evaluation that the rows
    hold. The transfers run only when every seed passed the check.
    """
    args = job['args']
    device = job['device']
    seed = job['seed']
    path = job['path']
    notes = []
    source_dir = Path(args.sources) / 'sub' / 'cov0.5_wd2.0' / f'seed_{seed}'
    common.COVERAGE = 0.5
    common.COUNT = int(0.5 * common.P * common.P)
    if job['mode'] == 'transfer':
        return transfer_sub(job, notes)
    q18_toy.train_toy_source(seed, 'sub', source_dir, device,
                             data=common.build_task('sub', device),
                             eval_every=100, ckpt_every=100, cap=20000)
    rows = q18_toy.source_rows_of(source_dir)
    grok_step = None
    for row in rows:
        dev_mean = common.as_float(row.get('dev_mean'))
        if dev_mean is not None and dev_mean >= 0.9:
            grok_step = common.as_int(row.get('step'))
            break
    if grok_step is None:
        notes.append(f'sub seed {seed}: the source never reached a dev of 0.9')
        return notes
    available = sorted(int(item.stem.split('_')[-1])
                       for item in (source_dir / 'ckpts').glob('step_*.pt'))
    if not available:
        notes.append(f'sub seed {seed}: no snapshot')
        return notes
    data = common.build_task('sub', device)
    mask = common.mask_for_seed(seed)
    strata, _exposure = common.split_for_mask(mask, seed)
    grok_snapshot = min(available, key=lambda value: (abs(value - grok_step), value))
    checkpoint = source_dir / 'ckpts' / f'step_{grok_snapshot:07d}.pt'
    state = common.load_checkpoint(checkpoint)['model']
    model = common.make_model(common.CONFIG)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    grok_train_acc, grok_dev_acc = sub_evaluation(model, data, strata, device)
    del model
    torch.cuda.empty_cache()
    if grok_dev_acc < 0.9 or grok_train_acc < MEMORIZED_TRAIN:
        notes.append(f'sub seed {seed}: the snapshot at the grok step {grok_step} '
                     f'(the nearest snapshot is {grok_snapshot}) holds a source dev '
                     f'of {grok_dev_acc:.4f} and a source train accuracy of '
                     f'{grok_train_acc:.4f}, so the step is stopped and no transfer '
                     f'is run')
        return notes
    snapshots = []
    for offset in TIMING_OFFSETS:
        target_step = grok_step - offset
        if target_step < 0:
            notes.append(f'sub seed {seed}: the step {target_step} is before the '
                        f'start of the source')
            continue
        step = min(available, key=lambda value: (abs(value - target_step), value))
        checkpoint = source_dir / 'ckpts' / f'step_{step:07d}.pt'
        if not checkpoint.exists():
            notes.append(f'sub seed {seed}: the snapshot at the step {step} is '
                        f'missing')
            continue
        state = common.load_checkpoint(checkpoint)['model']
        model = common.make_model(common.CONFIG)
        model.load_state_dict(state)
        model = model.to(device)
        model.eval()
        train_acc, source_dev = sub_evaluation(model, data, strata, device)
        del model
        torch.cuda.empty_cache()
        snapshots.append(dict(offset=offset, step=step, source_dev=source_dev,
                              source_train_acc=train_acc))
    job['snapshots'] = snapshots
    job['grok_step'] = grok_step
    return notes


def transfer_sub(job, notes):
    """The transfers of the snapshots that the preparation checked."""
    args = job['args']
    device = job['device']
    seed = job['seed']
    path = job['path']
    source_dir = Path(args.sources) / 'sub' / 'cov0.5_wd2.0' / f'seed_{seed}'
    mask = common.mask_for_seed(seed)
    strata, _exposure = common.split_for_mask(mask, seed)
    targets = {'affine_add': common.build_task('affine_add', device),
               'affine_sub': common.build_task('affine_sub', device)}
    grok_step = job['grok_step']
    for snapshot in job['snapshots']:
        step = snapshot['step']
        state = common.load_checkpoint(source_dir / 'ckpts' / f'step_{step:07d}.pt')['model']
        for target_name, target_data in targets.items():
            transfer_rows = toy_transfer(state, seed, target_data, strata, device,
                                         f't19_sub_{snapshot["offset"]}_'
                                         f'{target_name}_seed{seed}',
                                         steps=args.steps, eval_every=args.eval_every)
            final = transfer_rows[-1]['dev_mean'] if transfer_rows else ''
            append_rows(path, SUB_FIELDS, [dict(
                seed=seed, source_kind='sub_snapshot',
                steps_before_grok=snapshot['offset'], source_step=step,
                source_grok_step=grok_step, source_dev=snapshot['source_dev'],
                source_train_acc=snapshot['source_train_acc'],
                target_task=target_name, transfer_dev=final)])
            common.log_line(f'[q19] t3 sub {snapshot["offset"]} {target_name} '
                            f'seed {seed} step {step}: dev {final}')
    data = common.build_task('sub', device)
    fresh = common.init_model(seed, device)
    train_acc, source_dev = sub_evaluation(fresh, data, strata, device)
    state = {key: value.detach().cpu() for key, value in fresh.state_dict().items()}
    del fresh
    torch.cuda.empty_cache()
    for target_name, target_data in targets.items():
        transfer_rows = toy_transfer(state, seed, target_data, strata, device,
                                     f't19_sub_random_{target_name}_seed{seed}',
                                     steps=args.steps, eval_every=args.eval_every)
        final = transfer_rows[-1]['dev_mean'] if transfer_rows else ''
        append_rows(path, SUB_FIELDS, [dict(
            seed=seed, source_kind='random_init', steps_before_grok='', source_step='',
            source_grok_step='', source_dev=source_dev, source_train_acc=train_acc,
            target_task=target_name, transfer_dev=final)])
        common.log_line(f'[q19] t3 sub random {target_name} seed {seed}: dev {final}')
    return notes


UNITS = {'t1': unit_t1, 't2': unit_t2, 't3': unit_t3}
FIELDS = {'t1': PARITY_FIELDS, 't2': POLY_FIELDS, 't3': SUB_FIELDS}
NAMES = {'t1': 't19_parity.csv', 't2': 't19_poly.csv', 't3': 't19_sub.csv'}


def worker_init(device, batch_log, threads):
    common.COVERAGE = 0.5
    common.COUNT = int(0.5 * common.P * common.P)
    common.WEIGHT_DECAY = WEIGHT_DECAY
    common.setup(threads, device)


def worker(job):
    try:
        notes = UNITS[job['step']](job)
        return dict(job=job, notes=notes, error=None)
    except Exception as exc:  # noqa: BLE001 - one unit must not kill the pool
        common.log_line(f'[q19] {job["step"]} seed {job["seed"]} FAILED '
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', default='t1', choices=sorted(UNITS))
    parser.add_argument('--out', default='runs_nseed_queue19')
    parser.add_argument('--sources', default='runs_nseed_queue19/sources')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--procs', type=int, default=4)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.smoke:
        return run_smoke(args, device)
    out_dir = project_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.sources).mkdir(parents=True, exist_ok=True)
    path = out_dir / NAMES[args.step]
    ensure_header(path, FIELDS[args.step])

    seeds = {'t1': PARITY_SEEDS, 't2': POLY_SEEDS, 't3': SUB_SEEDS}[args.step]
    if args.step != 't3':
        units = [dict(step=args.step, seed=seed) for seed in seeds]
        jobs = [dict(args=args, device=device, path=path, **unit) for unit in units]
        print(f'{len(jobs)} units of {args.step}', flush=True)
        results = run_pool(jobs, device, args.procs, args.threads)
        failed = [result for result in results if result.get('error')]
        for result in results:
            for note in result.get('notes') or []:
                common.log_line(f'[q19] {args.step} note: {note}')
        for result in failed:
            job = result['job']
            common.log_line(f'[q19] FAILED {args.step} seed {job["seed"]}: '
                            f'{result["error"]}')
        print(f'{len(results) - len(failed)} units finished, {len(failed)} failed',
              flush=True)
        return 0

    jobs = [dict(args=args, device=device, path=path, step='t3', seed=seed,
                 mode='prepare') for seed in seeds]
    print(f'{len(jobs)} preparations of t3', flush=True)
    results = run_pool(jobs, device, args.procs, args.threads)
    notes = []
    for result in results:
        notes.extend(result.get('notes') or [])
    for note in notes:
        common.log_line(f'[q19] t3 note: {note}')
    stopped = [note for note in notes if 'the step is stopped' in note]
    if stopped:
        for note in stopped:
            common.log_line(f'[q19] t3 is stopped: {note}')
        print('the preparation of t3 failed, no transfer is run', flush=True)
        return 0
    transfer_jobs = []
    for result in results:
        job = result['job']
        if job.get('snapshots'):
            transfer_jobs.append(dict(args=args, device=device, path=path,
                                      step='t3', seed=job['seed'], mode='transfer',
                                      snapshots=job['snapshots'],
                                      grok_step=job['grok_step']))
    print(f'{len(transfer_jobs)} transfers of t3', flush=True)
    results = run_pool(transfer_jobs, device, args.procs, args.threads)
    failed = [result for result in results if result.get('error')]
    for result in failed:
        job = result['job']
        common.log_line(f'[q19] FAILED t3 seed {job["seed"]}: {result["error"]}')
    print(f'{len(results) - len(failed)} transfers finished, {len(failed)} failed',
          flush=True)
    return 0


def run_smoke(args, device):
    """The parity source with a small training set and the sub evaluation."""
    summary = parity_source_small(5, 300, Path(args.sources) / 'smoke' / 'N300', 100,
                                  device)
    print(f'the smoke parity memorized: {summary.get("memorized")}', flush=True)
    mask = common.mask_for_seed(5)
    strata, _exposure = common.split_for_mask(mask, 5)
    model = common.init_model(5, device)
    data = common.build_task('sub', device)
    train_acc, dev_acc = sub_evaluation(model, data, strata, device)
    print(f'the smoke sub evaluation of a fresh model: train {train_acc:.4f} '
          f'dev {dev_acc:.4f}', flush=True)
    norm = queue18_grokked_norm(
        Path(QUEUE18_SOURCES) / 'parity' / 'N2000_wd2.0' / 'seed_5')
    print(f'the smoke norm of the grokked source of queue 18: {norm}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
