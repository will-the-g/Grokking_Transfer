"""Queue 17, toy track: the harder targets, the sparse parity with a memorized
stage and the stage timing of the memorized transfer.

T5  the polynomial and the S5 sources transferred to targets that are trained
    on only a fraction of the target grid, with all the contexts of a kept pair
    together.
T6  the sparse parity with the stages of the memorizing sources and targets of
    three different sizes.
T7  the add and the sub sources transferred from the snapshots around their
    grok step, with the stage metrics at each snapshot.

Only dev splits are scored. The audit split is never read.

Produces: Table 3 (the harder targets).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
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
from q14_steps import (dihedral_table, group_data, group_power,  # noqa: E402
                       permutation_table, source_state, train_source)
from q16_toy import (ParityTransformer, parity_accuracy, parity_batch,  # noqa: E402
                      parity_label, parity_strings, parity_tokens,
                      poly_data, toy_transfer)

T5_FIELDS = ['source_task', 'target_task', 'seed', 'source_kind', 'target_fraction',
             'step', 'dev_acc', 'source_final_dev', 'source_grok_step']
T6_FIELDS = ['setting', 'seed', 'source_kind', 'target', 'target_n', 'step',
             'dev_acc', 'source_final_dev', 'source_grok_step']
T7_FIELDS = ['source_task', 'seed', 'snapshot_frac', 'source_step', 'source_grok_step',
             'source_dev', 'train_acc', 'frac_top5_a', 'frac_top5_b', 'target_task',
             'transfer_dev']
FRACTIONS = (0.05, 0.10, 0.20, 0.40)
PARITY_SETTINGS = [('N2000_wd1.0', 2000, 1.0), ('N2000_wd2.0', 2000, 2.0)]
PARITY_SEEDS = list(range(5, 13))
PARITY_TARGET_SIZES = (100, 300, 1000)
SNAPSHOT_FRACTIONS = (0.25, 0.50, 0.75, 0.90)


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


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open('r', newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def subset_pairs(strata, fraction, seed, p):
    """A subset of the target grid: the pairs are kept whole with their contexts.

    The fraction is over the grid of p * p pairs and every kept pair keeps all
    of its contexts, which is the construction that reproduced the Pre-LN
    baseline in queue14 step A.
    """
    train_ids = [int(value) for value in strata['train']]
    keep = min(len(train_ids), max(1, int(round(fraction * p * p))))
    chosen = set(random.Random(1000 + int(seed)).sample(train_ids, keep))
    rows = [pair_id for pair_id in train_ids if pair_id in chosen]
    return common.index_for(torch.tensor(rows, dtype=torch.long), 'cpu')


def fourier_top5(embeddings):
    """The fraction of the energy in the five strongest Fourier frequencies."""
    spectrum = torch.fft.rfft(embeddings.double(), dim=0)
    energy = spectrum.real ** 2 + spectrum.imag ** 2
    total = float(energy.sum())
    if total <= 0:
        return 0.0
    top = torch.topk(energy.flatten(), min(5, energy.numel())).values
    return float(top.sum()) / total


def embedding_concentration(model):
    """The Fourier concentration of the a-token and the b-token embeddings."""
    table = model.token_embed.weight.detach()
    pos = model.pos_embed.detach()
    return (fourier_top5(table[:model.p] + pos[1]),
            fourier_top5(table[:model.p] + pos[2]))



# ---------------------------------------------------------------------------
# the toy transfer with a subset of the target grid
# ---------------------------------------------------------------------------
def toy_transfer_subset(state, seed, data, strata, device, name, train_rows,
                         steps=2000, eval_every=100):
    """The transfer of the toy protocol on a subset of the target pairs."""
    from run_components import run_transfer_job
    model = common.make_model(common.CONFIG)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    rows, _trained = run_transfer_job(name, seed, model, data, strata, Path('.'),
                                      steps=steps, eval_every=eval_every,
                                      log=common.log_line, train_rows=train_rows)
    return rows


# ---------------------------------------------------------------------------
# STEP T5: the harder targets
# ---------------------------------------------------------------------------
def step_t5(args, device):
    """The polynomial and the S5 sources transferred to a fraction of the grid."""
    out_dir = project_path(args.out)
    path = out_dir / 't5_hard.csv'
    notes = []
    polynomial_targets = {'affine_poly': poly_data(device, 'affine_poly'),
                          'affine_add': common.build_task('affine_add', device),
                          'affine_mult': common.build_task('affine_mult', device)}
    table = permutation_table()
    group_targets = {'power_affine': group_data(table, 120, device, power=True)}
    tasks = {
        'poly': dict(dir=Path('runs_nseed_queue16/sources/poly/cov0.5_wd2.0'),
                     p=59, targets=polynomial_targets, seeds=[5, 6, 7, 8, 9],
                     model_config=dict(common.CONFIG)),
        'S5': dict(dir=Path('runs_nseed_queue16/sources/S5/cov0.6_wd2.0'), p=120,
                   targets=group_targets, seeds=[5, 6, 7],
                   model_config=dict(common.CONFIG, p=120, d_model=256, d_mlp=1024)),
    }
    for task_name, task in tasks.items():
        for seed in task['seeds']:
            seed_dir = task['dir'] / f'seed_{seed}'
            if not (seed_dir / 'summary.json').exists():
                notes.append(f'{task_name} seed {seed}: the source is missing')
                continue
            summary = common.read_summary(seed_dir / 'summary.json')
            stages = summary.get('stages', {}) or {}
            if not stages:
                notes.append(f'{task_name} seed {seed}: no stages')
                continue
            if task_name == 'S5':
                common.P = 120
                common.CONFIG['p'] = 120
                common.CONFIG['d_model'] = 256
                common.CONFIG['d_mlp'] = 1024
                common.COUNT = int(common.COVERAGE * 120 * 120)
            else:
                common.P = 59
                common.CONFIG['p'] = 59
                common.CONFIG['d_model'] = 128
                common.CONFIG['d_mlp'] = 512
                common.COUNT = int(common.COVERAGE * 59 * 59)
            grokked_norm = (stages.get('grokked', {}) or {}).get('global_norm')
            mask = common.mask_for_seed(seed)
            strata, _exposure = common.split_for_mask(mask, seed)
            sources = {}
            for stage in ('memorized', 'transition', 'grokked'):
                if stage in stages:
                    sources[stage] = (stages[stage]['step'],
                                      common.load_checkpoint(seed_dir / 'ckpts' /
                                                             stages[stage]['ckpt'])['model'])
            if 'memorized' in stages and grokked_norm is not None:
                sources['norm_control'] = ('', source_state(seed_dir, 'norm_control',
                                                             stages, grokked_norm))
            model = common.init_model(seed, device)
            sources['random_init'] = ('', {key: value.detach().cpu() for key, value
                                           in model.state_dict().items()})
            for kind, (step, state) in sources.items():
                for target, target_data in task['targets'].items():
                    for fraction in args.fractions:
                        train_rows = subset_pairs(strata, fraction, seed, task['p'])
                        rows = toy_transfer_subset(state, seed, target_data, strata,
                                                   device,
                                                   f't5_{task_name}_{kind}_{target}_'
                                                   f'{fraction}_seed{seed}',
                                                   train_rows, steps=args.steps,
                                                   eval_every=args.eval_every)
                        final = rows[-1]['dev_mean'] if rows else ''
                        append_rows(path, T5_FIELDS, [dict(
                            source_task=task_name, target_task=target, seed=seed,
                            source_kind=kind, target_fraction=fraction,
                            step=args.steps, dev_acc=final,
                            source_final_dev=summary.get('source_final_dev'),
                            source_grok_step=summary.get('grok_step'))])
                        common.log_line(f'[q17] t5 {task_name} {kind} {target} '
                                        f'{fraction} seed {seed}: dev {final}')
    common.P = 59
    common.CONFIG['p'] = 59
    common.CONFIG['d_model'] = 128
    common.CONFIG['d_mlp'] = 512
    common.COUNT = int(common.COVERAGE * common.P * common.P)
    return notes


# ---------------------------------------------------------------------------
# STEP T6: the sparse parity with a memorized stage
# ---------------------------------------------------------------------------
def parity_source_snapshots(seed, n_pairs, wd, out_dir, cap, device, n_bits=30, k=3):
    """A parity source with a snapshot every 100 steps."""
    out_dir = Path(out_dir)
    summary = common.read_summary(out_dir / 'summary.json')
    if summary.get('source_finished'):
        return summary
    subset = sorted(random.Random(seed).sample(range(n_bits), k))
    train = parity_strings(n_pairs, n_bits, 20000 + seed)
    dev = parity_strings(5000, n_bits, 30000 + seed)
    model = ParityTransformer(n_bits=n_bits).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd,
                                  betas=(0.9, 0.98))
    generator = torch.Generator(device=device).manual_seed(int(1000 + seed))
    states = {}
    history = []
    best_dev = 0.0
    out_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    while step < cap:
        index = torch.randint(0, len(train), (512,), generator=generator,
                              device=device)
        tokens = torch.stack([parity_tokens(bits, device) for bits in
                              [train[int(row)] for row in index]])
        labels = torch.tensor([parity_label(train[int(row)], subset) for row in index],
                              dtype=torch.long, device=device)
        loss = nn.functional.cross_entropy(model(tokens), labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % 100 == 0:
            train_acc = parity_accuracy(model, train[:1000], subset, device)
            dev_acc = parity_accuracy(model, dev, subset, device)
            history.append(dict(step=step, train_acc=train_acc, dev_acc=dev_acc))
            state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save(state, out_dir / f'step_{step:06d}.pt')
            if train_acc >= 0.99 and dev_acc <= 0.6 and 'memorized' not in states:
                states['memorized'] = (step, state)
            elif dev_acc >= 0.95 and 'grokked' not in states:
                states['grokked'] = (step, state)
            elif dev_acc < 0.95 and dev_acc > best_dev and 'transition' not in states:
                best_dev = dev_acc
                states['transition'] = (step, state)
            if 'grokked' in states:
                break
    final = history[-1] if history else dict(step=step, train_acc=0.0, dev_acc=0.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(source_finished=True, seed=seed, n_pairs=n_pairs, wd=wd,
                   subset=subset, n_bits=n_bits, k=k,
                   stages={name: dict(step=at, ckpt=f'step_{at:06d}.pt')
                           for name, (at, _state) in states.items()},
                   final_dev=final['dev_acc'], final_train=final['train_acc'],
                   grok_step=states.get('grokked', (None, None))[0], history=history)
    (out_dir / 'summary.json').write_text(json.dumps(payload, indent=2, default=str),
                                          encoding='utf-8')
    common.log_line(f'[q17] the parity source seed {seed} N {n_pairs} wd {wd}: '
                    f'train {final["train_acc"]:.4f} dev {final["dev_acc"]:.4f} '
                    f'stages {sorted(states)}')
    del model
    torch.cuda.empty_cache()
    return payload


def parity_transfer_n(state, n_bits, seed, train_strings, dev_strings, subset,
                       device, steps=2000, eval_every=100):
    """The transfer of a parity stage on a target of a given size."""
    model = ParityTransformer(n_bits=n_bits)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0,
                                  betas=(0.9, 0.98))
    generator = torch.Generator(device=device).manual_seed(int(seed + 31))
    rows = []
    step = 0
    while step < steps:
        index = torch.randint(0, len(train_strings), (512,), generator=generator,
                              device=device)
        tokens = torch.stack([parity_tokens(train_strings[int(row)], device)
                              for row in index])
        labels = torch.tensor([parity_label(train_strings[int(row)], subset)
                               for row in index], dtype=torch.long, device=device)
        loss = nn.functional.cross_entropy(model(tokens), labels)
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



def step_t6(args, device):
    """The parity sources with the stages and the targets of three sizes."""
    out_dir = project_path(args.out)
    path = out_dir / 't6_parity.csv'
    notes = []
    for setting, n_pairs, wd in PARITY_SETTINGS:
        for seed in PARITY_SEEDS:
            source_dir = out_dir / 'sources' / setting / f'seed_{seed}'
            summary = parity_source_snapshots(seed, n_pairs, wd, source_dir,
                                             args.cap, device)
            stages = summary.get('stages', {}) or {}
            if 'memorized' not in stages:
                notes.append(f'{setting} seed {seed}: no memorized snapshot')
            subset = summary['subset']
            n_bits = summary['n_bits']
            dev = parity_strings(5000, n_bits, 30000 + seed)
            new_bits = [index for index in range(n_bits) if index not in subset]
            related = sorted(random.Random(seed + 7).sample(subset[:2] + new_bits[:1],
                                                            3))
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
                    for target_n in PARITY_TARGET_SIZES:
                        train = parity_strings(target_n, n_bits, 40000 + seed)
                        rows = parity_transfer_n(state, n_bits, seed, train, dev,
                                                 target_subset, device,
                                                 steps=args.steps,
                                                 eval_every=args.eval_every)
                        final = rows[-1]['dev_mean'] if rows else ''
                        append_rows(path, T6_FIELDS, [dict(
                            setting=setting, seed=seed, source_kind=kind,
                            target=target, target_n=target_n, step=args.steps,
                            dev_acc=final, source_final_dev=summary.get('final_dev'),
                            source_grok_step=summary.get('grok_step'))])
                        common.log_line(f'[q17] t6 {setting} {kind} {target} '
                                        f'n {target_n} seed {seed}: dev {final}')
    return notes


# ---------------------------------------------------------------------------
# STEP T7: the stage timing of the memorized transfer
# ---------------------------------------------------------------------------
def step_t7(args, device):
    """The transfers from the snapshots around the grok step."""
    out_dir = project_path(args.out)
    path = out_dir / 't7_timing.csv'
    notes = []
    sources = {
        'add': Path('runs_nseed_queue16/sources/add/cov0.5_wd2.0'),
        'sub': Path('runs_nseed_queue14/sources/sub/cov0.5_wd2.0'),
    }
    targets = {'affine_add': common.build_task('affine_add', device),
               'affine_sub': common.build_task('affine_sub', device)}
    for task_name, source_root in sources.items():
        for seed in args.seeds:
            seed_dir = source_root / f'seed_{seed}'
            if not (seed_dir / 'summary.json').exists():
                notes.append(f'{task_name} seed {seed}: the source is missing')
                continue
            summary = common.read_summary(seed_dir / 'summary.json')
            stages = summary.get('stages', {}) or {}
            grok_step = summary.get('grok_step')
            if not grok_step:
                notes.append(f'{task_name} seed {seed}: the source never grokked')
                continue
            mask = common.mask_for_seed(seed)
            strata, _exposure = common.split_for_mask(mask, seed)
            checkpoints = sorted((seed_dir / 'ckpts').glob('step_*.pt'),
                                 key=lambda path: int(path.stem.split('step')[-1]
                                                     .lstrip('_')))
            available = [int(path.stem.split('step')[-1].lstrip('_'))
                          for path in checkpoints]
            selected = []
            for fraction in SNAPSHOT_FRACTIONS:
                target = int(round(grok_step * fraction))
                if not available:
                    continue
                nearest = min(available, key=lambda step: (abs(step - target), step))
                selected.append((fraction, nearest))
            for name in ('memorized', 'grokked'):
                if name in stages:
                    selected.append((name, stages[name]['step']))
            for label, step in selected:
                checkpoint = seed_dir / 'ckpts' / f'step_{step:06d}.pt'
                if not checkpoint.exists():
                    candidates = [path for path in checkpoints
                                  if int(path.stem.split('step')[-1].lstrip('_')) == step]
                    checkpoint = candidates[0] if candidates else checkpoint
                if not checkpoint.exists():
                    notes.append(f'{task_name} seed {seed}: the snapshot at step '
                                f'{step} is missing')
                    continue
                state = common.load_checkpoint(checkpoint)['model']
                model = common.make_model(common.CONFIG)
                model.load_state_dict(state)
                model = model.to(device)
                model.eval()
                train_rows = common.index_for(strata['train'], device)
                train_acc = common.accuracy_on(model, targets['affine_add'],
                                                train_rows)
                dev_rows = common.context_rows(strata['dev'], 0, device)
                dev_acc = common.accuracy_on(model, targets['affine_add'], dev_rows)
                norm = common.global_norm(model)
                frac_a, frac_b = embedding_concentration(model)
                del model
                torch.cuda.empty_cache()
                for target_name, target_data in targets.items():
                    rows = toy_transfer(state, seed, target_data, strata, device,
                                        f't7_{task_name}_{label}_{target_name}_'
                                        f'seed{seed}', steps=args.steps,
                                        eval_every=args.eval_every)
                    final = rows[-1]['dev_mean'] if rows else ''
                    append_rows(path, T7_FIELDS, [dict(
                        source_task=task_name, seed=seed, snapshot_frac=label,
                        source_step=step, source_grok_step=grok_step,
                        source_dev=dev_acc, train_acc=train_acc, frac_top5_a=frac_a,
                        frac_top5_b=frac_b, target_task=target_name,
                        transfer_dev=final)])
                    common.log_line(f'[q17] t7 {task_name} {label} {target_name} '
                                    f'seed {seed} step {step}: dev {final}')
    return notes


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------
def run_smoke(args, device):
    """The fraction subset, the parity model and the Fourier concentration."""
    strata, _exposure = common.split_for_mask(common.mask_for_seed(5), 5)
    for fraction in (0.05, 0.40):
        rows = subset_pairs(strata, fraction, 5, 59)
        print(f'p=59 the fraction {fraction}: {len(rows)} rows of the target grid',
              flush=True)
    model = common.make_model(common.CONFIG)
    model = model.to(device)
    frac_a, frac_b = embedding_concentration(model)
    print(f'the toy model: frac_top5_a {frac_a:.4f} frac_top5_b {frac_b:.4f}', flush=True)
    parity = ParityTransformer(n_bits=30).to(device)
    strings = parity_strings(16, 30, 7)
    subset = sorted(random.Random(7).sample(range(30), 3))
    tokens, labels = parity_batch(strings, subset, device)
    loss = nn.functional.cross_entropy(parity(tokens), labels)
    print(f'the parity model: {tuple(tokens.shape)} tokens, the loss {float(loss):.4f}',
          flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['t5', 't6', 't7'])
    parser.add_argument('--out', default='runs_nseed_queue17')
    parser.add_argument('--seeds', nargs='+', type=int, default=[5, 6, 7, 8, 9])
    parser.add_argument('--fractions', nargs='+', type=float, default=list(FRACTIONS))
    parser.add_argument('--cap', type=int, default=100000)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    common.setup(1, args.device)
    if args.smoke:
        return run_smoke(args, device)
    if args.step == 't5':
        notes = step_t5(args, device)
    elif args.step == 't6':
        notes = step_t6(args, device)
    else:
        notes = step_t7(args, device)
    for note in notes:
        common.log_line(f'[q17] {note}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
