"""Queue 16, toy track: the memorized-subtraction diagnosis, the sparse
parity, the modular polynomial and the groups.

T1  the queue14 subtraction sources and fresh addition sources: the stage
    checkpoints are measured (the train and dev accuracy, the global norm and
    the Fourier concentration of the embeddings) and transferred to affine_add
    and affine_sub.
T2  the sparse parity over 30 bits, a mechanism that is not a Fourier
    feature: the settings are searched for a grokking source and the stages
    are transferred to a related, a control and an identical target.
T3  the modular polynomial x^2 + y^2 mod 59 and its affine targets.
T4  the groups: the dihedral D_59 with a memorized stage and the symmetric S5
    with a wider model.

Only dev splits are scored. The audit split is never read.

Produces: Table 3 (the polynomial and the parity sources).
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import multiprocessing
import random
import sys
import time
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
from model_phase3 import Block  # noqa: E402
from q14_steps import (dihedral_table, group_data, group_power,  # noqa: E402
                       permutation_table, source_state, train_source)

T1_FIELDS = ['source_task', 'seed', 'stage', 'source_step', 'train_acc', 'source_dev',
             'global_norm', 'frac_top5_a', 'frac_top5_b', 'target_task', 'transfer_dev',
             'source_kind']
T2_FIELDS = ['setting', 'seed', 'source_kind', 'target', 'step', 'dev_acc',
             'source_final_dev', 'source_grok_step']
T3_FIELDS = ['source_task', 'target_task', 'seed', 'source_kind', 'step', 'dev_acc',
             'source_final_dev', 'source_grok_step']
T4_FIELDS = ['group', 'config', 'seed', 'source_kind', 'step', 'dev_acc',
             'source_final_dev', 'source_grok_step']
STAGES = ('memorized', 'transition', 'grokked', 'norm_control', 'random_init')


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else common.PARENT / path


def append_rows(path, fields, rows):
    """Append the rows to a CSV; the header is written only when it is created."""
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


# ---------------------------------------------------------------------------
# the toy tasks
# ---------------------------------------------------------------------------
def poly_data(device, kind='poly'):
    """The data of a polynomial task with the usual token layout."""
    p = common.P
    rows = p * p * common.N_CONTEXTS
    tokens = torch.empty(rows, 4, dtype=torch.long)
    labels = torch.empty(rows, dtype=torch.long)
    for context in range(common.N_CONTEXTS):
        for a in range(p):
            for b in range(p):
                position = context * p * p + a * p + b
                tokens[position, 0] = p + context
                tokens[position, 1] = a
                tokens[position, 2] = b
                tokens[position, 3] = p + common.N_CONTEXTS
                if kind == 'poly':
                    labels[position] = (a * a + b * b) % p
                elif kind == 'affine_poly':
                    labels[position] = (a * a + (context + 1) * b * b) % p
                else:
                    raise ValueError(f'unknown polynomial task {kind!r}')
    return dict(x=tokens.to(device), y=labels.to(device), device=device)


def common_data(device, kind):
    """The data of a task of the common vocabulary."""
    return common.build_task(kind, device)


# ---------------------------------------------------------------------------
# the parity task
# ---------------------------------------------------------------------------
class ParityTransformer(nn.Module):
    """The toy transformer with a longer sequence for the parity task.

    The same block as the toy model (d_model 128, 4 heads, d_mlp 512, one
    layer), but the sequence holds the 30 bits and the classification token.
    """

    def __init__(self, n_bits=30, d_model=128, n_heads=4, d_mlp=512, n_classes=2):
        super().__init__()
        self.n_bits = n_bits
        self.vocab = 3
        self.token_embed = nn.Embedding(self.vocab, d_model)
        self.pos_embed = nn.Parameter(torch.empty(n_bits + 1, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp_in = nn.Linear(d_model, d_mlp)
        self.mlp_out = nn.Linear(d_mlp, d_model)
        self.unembed = nn.Linear(d_model, n_classes, bias=False)
        self.register_buffer('causal_mask',
                             torch.triu(torch.ones(n_bits + 1, n_bits + 1,
                                                   dtype=torch.bool), diagonal=1))
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, tokens):
        x = self.token_embed(tokens) + self.pos_embed.unsqueeze(0)
        hidden, _weights = self.attn(x, x, x, attn_mask=self.causal_mask)
        x = x + hidden
        hidden = self.mlp_out(torch.relu(self.mlp_in(x)))
        x = x + hidden
        return self.unembed(x[:, -1, :])


def parity_strings(count, n_bits, seed):
    """`count` random bit strings of the seed."""
    generator = random.Random(seed)
    out = []
    for _ in range(count):
        bits = [generator.randrange(2) for _ in range(n_bits)]
        out.append(bits)
    return out


def parity_label(bits, subset):
    """The parity of the bits of the subset."""
    total = 0
    for index in subset:
        total ^= bits[index]
    return total


def parity_tokens(bits, device):
    """The token sequence of a bit string: the bits and the class token."""
    tokens = torch.tensor(list(bits) + [2], dtype=torch.long, device=device)
    return tokens


def parity_batch(strings, subset, device):
    """The token batch and the labels of a list of bit strings."""
    tokens = torch.stack([parity_tokens(bits, device) for bits in strings])
    labels = torch.tensor([parity_label(bits, subset) for bits in strings],
                          dtype=torch.long, device=device)
    return tokens, labels


def parity_accuracy(model, strings, subset, device, batch_size=256):
    """The fraction of the strings whose parity is predicted."""
    if not strings:
        return float('nan')
    correct = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(strings), batch_size):
            chunk = strings[start:start + batch_size]
            tokens, labels = parity_batch(chunk, subset, device)
            predicted = model(tokens).argmax(dim=-1)
            correct += int((predicted == labels).sum().item())
    return correct / len(strings)


def parity_train(model, strings, subset, device, steps, batch_size, learning_rate,
                 weight_decay, seed=0, eval_every=0, evaluate=None):
    """Full fine-tuning of the parity model; the loss is over the label."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay, betas=(0.9, 0.98))
    generator = random.Random(seed)
    measured = []
    for step in range(1, steps + 1):
        batch = [strings[generator.randrange(len(strings))]
                 for _ in range(batch_size)]
        tokens, labels = parity_batch(batch, subset, device)
        loss = nn.functional.cross_entropy(model(tokens), labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if evaluate is not None and eval_every and (step % eval_every == 0
                                                    or step == steps):
            measured.append((step, evaluate()))
    model.eval()
    return measured


# ---------------------------------------------------------------------------
# the measurements of a stage checkpoint
# ---------------------------------------------------------------------------
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
    a = table[:model.p] + pos[1]
    b = table[:model.p] + pos[2]
    return fourier_top5(a), fourier_top5(b)


def stage_metrics(model, data, strata, device):
    """The train accuracy, the dev accuracy and the global norm of a model."""
    train_rows = common.index_for(strata['train'], device)
    train_acc = common.accuracy_on(model, data, train_rows)
    dev_rows = common.context_rows(strata['dev'], 0, device)
    dev_acc = common.accuracy_on(model, data, dev_rows)
    return train_acc, dev_acc, common.global_norm(model)


def toy_transfer(state, seed, data, strata, device, name, steps=2000, eval_every=100):
    """The standard transfer of the toy protocol on a prepared dataset."""
    from run_components import run_transfer_job
    model = common.make_model(common.CONFIG)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    rows, _trained = run_transfer_job(name, seed, model, data, strata, Path('.'),
                                      steps=steps, eval_every=eval_every,
                                      log=common.log_line)
    return rows


# ---------------------------------------------------------------------------
# the pool
# ---------------------------------------------------------------------------
def worker_init(device, batch_log, threads):
    common.setup(threads, device)
    common.set_batch_log(batch_log)


def worker(job):
    meta = {key: value for key, value in job.items()
            if key not in ('state', 'data', 'strata')}
    try:
        if job['kind'] == 'transfer':
            rows = toy_transfer(job['state'], job['seed'], job['data'], job['strata'],
                                job['device'], job['name'], steps=job['steps'],
                                eval_every=job['eval_every'])
            return dict(job=meta, rows=rows)
        if job['kind'] == 'measure':
            data = common.build_task(job['task'], job['device'])
            metrics = measure_stage(job['state'], data, job['strata'], job['device'])
            transfers = []
            for target in job['targets']:
                target_data = common.build_task(target, job['device'])
                transfer_rows = toy_transfer(job['state'], job['seed'], target_data,
                                             job['strata'], job['device'],
                                             f'{job["name"]}_{target}',
                                             steps=job['steps'],
                                             eval_every=job['eval_every'])
                transfers.append((target,
                                  transfer_rows[-1]['dev_mean']
                                  if transfer_rows else ''))
            return dict(job=meta, result=(metrics, transfers))
        raise ValueError(f'unknown job kind {job["kind"]!r}')
    except Exception as exc:  # noqa: BLE001 - one job must not kill the batch
        common.log_line(f'[q16] job {job.get("name", "")} FAILED '
                        f'{type(exc).__name__}: {exc}')
        common.log_line(traceback.format_exc())
        return dict(job=meta, rows=[], result=None)


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



# ---------------------------------------------------------------------------
# STEP T1: the memorized-subtraction diagnosis
# ---------------------------------------------------------------------------
def stage_checkpoints(seed_dir, stages):
    """The stage checkpoints of a source with their steps."""
    out = []
    for name in ('memorized', 'transition', 'grokked'):
        if name not in stages:
            continue
        out.append((name, stages[name]['step'],
                    Path(seed_dir) / 'ckpts' / stages[name]['ckpt']))
    return out


def measure_stage(state, data, strata, device):
    """The train accuracy, the dev accuracy, the norm and the concentration."""
    model = common.make_model(common.CONFIG)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    train_acc, dev_acc, norm = stage_metrics(model, data, strata, device)
    frac_a, frac_b = embedding_concentration(model)
    del model
    torch.cuda.empty_cache()
    return train_acc, dev_acc, norm, frac_a, frac_b


def step_t1(args, device):
    """The stage checkpoints of the two sources and their transfers."""
    out_dir = project_path(args.out)
    path = out_dir / 't1_diag.csv'
    notes = []
    sources = {
        'sub': (Path('runs_nseed_queue14/sources/sub/cov0.5_wd2.0'), 'sub'),
        'add': (out_dir / 'sources' / 'add' / 'cov0.5_wd2.0', 'add'),
    }
    for source_task, (seed_dir, task) in sources.items():
        for seed in args.seeds:
            if not (seed_dir / f'seed_{seed}' / 'summary.json').exists():
                notes.append(f'{source_task} seed {seed}: the source was trained now')
                train_source(seed, task, seed_dir / f'seed_{seed}', args.cap, device,
                             ckpt_every=250)
    jobs = []
    for source_task, (seed_dir, task) in sources.items():
        data = (common.build_task('sub', device) if task == 'sub'
                else common.build_task('add', device))
        for seed in args.seeds:
            seed_path = seed_dir / f'seed_{seed}'
            summary = common.read_summary(seed_path / 'summary.json')
            stages = summary.get('stages', {}) or {}
            if not stages:
                notes.append(f'{source_task} seed {seed}: no stages')
                continue
            grokked_norm = (stages.get('grokked', {}) or {}).get('global_norm')
            mask = common.mask_for_seed(seed)
            strata, _exposure = common.split_for_mask(mask, seed)
            for stage, step, ckpt in stage_checkpoints(seed_path, stages):
                state = common.load_checkpoint(ckpt)['model']
                jobs.append(dict(kind='measure',
                                 name=f't1_{source_task}_{stage}_seed{seed}',
                                 source_task=source_task, seed=seed, stage=stage,
                                 source_step=step, state=state, task=task,
                                 strata=strata, device=device,
                                 targets=['affine_add', 'affine_sub'],
                                 steps=args.steps,
                                 eval_every=args.eval_every))
            if 'memorized' in stages and grokked_norm is not None:
                state = source_state(seed_path, 'norm_control', stages, grokked_norm)
                jobs.append(dict(kind='measure',
                                 name=f't1_{source_task}_norm_control_seed{seed}',
                                 source_task=source_task, seed=seed,
                                 stage='norm_control', source_step='', state=state,
                                 task=task, strata=strata, device=device,
                                 targets=['affine_add', 'affine_sub'],
                                 steps=args.steps,
                                 eval_every=args.eval_every))
            model = common.init_model(seed, device)
            jobs.append(dict(kind='measure',
                             name=f't1_{source_task}_random_seed{seed}',
                             source_task=source_task, seed=seed, stage='random_init',
                             source_step='',
                             state={key: value.detach().cpu()
                                    for key, value in model.state_dict().items()},
                             task=task, strata=strata, device=device,
                             targets=['affine_add', 'affine_sub'],
                                 steps=args.steps,
                                 eval_every=args.eval_every))
    results = run_pool(jobs, device, args.procs, str(common.BATCH_LOG), args.threads)
    for result in results:
        job = result['job']
        if result.get('result') is None:
            notes.append(f'{job["name"]}: the measurement failed')
            continue
        (train_acc, dev_acc, norm, frac_a, frac_b), transfers = result['result']
        for target, final in transfers:
            append_rows(path, T1_FIELDS, [dict(
                source_task=job['source_task'], seed=job['seed'], stage=job['stage'],
                source_step=job['source_step'], train_acc=train_acc,
                source_dev=dev_acc, global_norm=norm, frac_top5_a=frac_a,
                frac_top5_b=frac_b, target_task=target, transfer_dev=final,
                source_kind=job['stage'])])
            common.log_line(f'[q16] t1 {job["source_task"]} {job["stage"]} {target} '
                            f'seed {job["seed"]}: dev {final}')
    return notes



# ---------------------------------------------------------------------------
# STEP T2: the sparse parity
# ---------------------------------------------------------------------------
def parity_stage(train_acc, dev_acc):
    """The stage of a parity source by its accuracies."""
    if dev_acc >= 0.95:
        return 'grokked'
    if train_acc >= 0.99 and dev_acc <= 0.6:
        return 'memorized'
    return 'transition'


def prepare_parity(strings, subset, device):
    """The tokenised bit strings, tokenized once.

    Tokenizing the batch inside the training loop costs more than the step
    itself, so the strings are tokenized once and the batches are drawn from
    the cached tensors.
    """
    tokens = torch.stack([parity_tokens(bits, device) for bits in strings])
    labels = torch.tensor([parity_label(bits, subset) for bits in strings],
                          dtype=torch.long, device=device)
    return tokens, labels


def parity_source(seed, n_pairs, wd, out_dir, cap, device, n_bits=30, k=3):
    """Train one parity source and keep the stage snapshots."""
    out_dir = Path(out_dir)
    summary = common.read_summary(out_dir / 'summary.json')
    if summary.get('source_finished'):
        return summary
    subset = sorted(random.Random(seed).sample(range(n_bits), k))
    train = parity_strings(n_pairs, n_bits, 20000 + seed)
    dev = parity_strings(5000, n_bits, 30000 + seed)
    model = ParityTransformer(n_bits=n_bits)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd,
                                  betas=(0.9, 0.98))
    generator = torch.Generator(device=device).manual_seed(int(1000 + seed))
    train_tokens, train_labels = prepare_parity(train, subset, device)
    states = {}
    history = []
    best_dev = 0.0
    step = 0
    while step < cap:
        index = torch.randint(0, train_tokens.shape[0], (512,), generator=generator,
                              device=device)
        loss = torch.nn.functional.cross_entropy(model(train_tokens[index]),
                                                 train_labels[index])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % 500 == 0:
            train_acc = parity_accuracy(model, train[:1000], subset, device)
            dev_acc = parity_accuracy(model, dev, subset, device)
            history.append(dict(step=step, train_acc=train_acc, dev_acc=dev_acc))
            stage = parity_stage(train_acc, dev_acc)
            if stage == 'memorized' and 'memorized' not in states:
                states['memorized'] = (step, {key: value.detach().cpu() for key, value
                                              in model.state_dict().items()})
            elif stage == 'transition' and dev_acc > best_dev and 'grokked' not in states:
                best_dev = dev_acc
                states['transition'] = (step, {key: value.detach().cpu() for key,
                                               value in model.state_dict().items()})
            if stage == 'grokked':
                states['grokked'] = (step, {key: value.detach().cpu() for key, value
                                            in model.state_dict().items()})
                break
    final = history[-1] if history else dict(step=step, train_acc=0.0, dev_acc=0.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (at, state) in states.items():
        torch.save(state, out_dir / f'{name}.pt')
    payload = dict(source_finished=True, seed=seed, n_pairs=n_pairs, wd=wd,
                   subset=subset, n_bits=n_bits, k=k,
                   stages={name: dict(step=at, ckpt=f'{name}.pt')
                           for name, (at, _state) in states.items()},
                   final_dev=final['dev_acc'], final_train=final['train_acc'],
                   grok_step=states.get('grokked', (None, None))[0], history=history)
    (out_dir / 'summary.json').write_text(json.dumps(payload, indent=2, default=str),
                                          encoding='utf-8')
    common.log_line(f'[q16] parity source seed {seed} N {n_pairs} wd {wd}: '
                    f'train {final["train_acc"]:.4f} dev {final["dev_acc"]:.4f} '
                    f'stages {sorted(states)}')
    del model
    torch.cuda.empty_cache()
    return payload


def parity_transfer(state, seed, train_strings, dev_strings, subset, device, name,
                    steps=2000, eval_every=100):
    """The transfer of a parity stage on a target subset."""
    model = ParityTransformer(n_bits=len(train_strings[0]))
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0,
                                  betas=(0.9, 0.98))
    generator = torch.Generator(device=device).manual_seed(int(seed + 31))
    train_tokens, train_labels = prepare_parity(train_strings, subset, device)
    rows = []
    step = 0
    while step < steps:
        index = torch.randint(0, train_tokens.shape[0], (512,), generator=generator,
                              device=device)
        loss = torch.nn.functional.cross_entropy(model(train_tokens[index]),
                                                 train_labels[index])
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



def step_t2(args, device):
    """The parity settings search and the transfers of the stages."""
    out_dir = project_path(args.out)
    path = out_dir / 't2_parity.csv'
    notes = []
    attempts = []
    if not args.transfers_only:
     for n_pairs in args.n_values:
      for wd in args.wd_values:
       for seed in args.seeds:
        summary = parity_source(seed, n_pairs, wd,
                                out_dir / 'sources' / f'N{n_pairs}_wd{wd}' /
                                f'seed_{seed}', args.cap, device)
        attempts.append(dict(n_pairs=n_pairs, wd=wd, seed=seed,
                             final_dev=summary.get('final_dev'),
                             final_train=summary.get('final_train'),
                             stages=';'.join(sorted(summary.get('stages', {}))),
                             grok_step=summary.get('grok_step')))
        append_rows(out_dir / 't2_attempts.csv',
                    ['n_pairs', 'wd', 'seed', 'final_dev', 'final_train',
                     'stages', 'grok_step'], [attempts[-1]])
    if args.transfers_only:
        attempts = read_rows(out_dir / 't2_attempts.csv')
    grokked = [row for row in attempts if 'grokked' in row['stages']]
    if not grokked:
        notes.append('no parity setting grokked: only the source rows are written')
        for row in attempts:
            append_rows(path, T2_FIELDS, [dict(
                setting=f'N{row["n_pairs"]}_wd{row["wd"]}', seed=row['seed'],
                source_kind='source', target='', step=args.cap,
                dev_acc=row['final_dev'], source_final_dev=row['final_dev'],
                source_grok_step='')])
        return notes
    best = max(grokked, key=lambda row: row['final_dev'] or 0)
    setting = (best['n_pairs'], best['wd'])
    seeds = [row['seed'] for row in grokked
             if (row['n_pairs'], row['wd']) == setting]
    for seed in seeds:
        seed = int(seed)
        source_dir = (out_dir / 'sources' / f'N{setting[0]}_wd{setting[1]}' /
                      f'seed_{seed}')
        summary = common.read_summary(source_dir / 'summary.json')
        subset = summary['subset']
        n_bits = summary['n_bits']
        dev = parity_strings(5000, n_bits, 30000 + seed)
        new_bits = [index for index in range(n_bits) if index not in subset]
        related = sorted(random.Random(seed + 7).sample(subset[:2] + new_bits[:1], 3))
        control = sorted(random.Random(seed + 11).sample(new_bits, 3))
        targets = {'related': related, 'control': control, 'identical': subset}
        states = {}
        for name in ('memorized', 'transition', 'grokked'):
            if name in summary.get('stages', {}):
                states[name] = (summary['stages'][name]['step'],
                                torch.load(source_dir / summary['stages'][name]['ckpt'],
                                           map_location='cpu'))
        if 'memorized' in states and 'grokked' in states:
            model = ParityTransformer(n_bits=n_bits)
            model.load_state_dict(states['memorized'][1])
            model = model.to(device)
            grokked_model = ParityTransformer(n_bits=n_bits)
            grokked_model.load_state_dict(states['grokked'][1])
            grokked_model = grokked_model.to(device)
            common.rescale_to_norm(model, common.global_norm(grokked_model))
            states['norm_control'] = ('', {key: value.detach().cpu() for key, value
                                           in model.state_dict().items()})
            del model, grokked_model
            torch.cuda.empty_cache()
        fresh = ParityTransformer(n_bits=n_bits)
        states['random_init'] = ('', {key: value.detach().cpu()
                                      for key, value in fresh.state_dict().items()})
        for name, (step, state) in states.items():
            for target, target_subset in targets.items():
                target_train = parity_strings(1000, n_bits, 40000 + seed)
                rows = parity_transfer(state, seed, target_train, dev, target_subset,
                                       device,
                                       f't2_{setting[0]}_{setting[1]}_{name}_{target}_'
                                       f'seed{seed}', steps=args.steps,
                                       eval_every=args.eval_every)
                final = rows[-1]['dev_mean'] if rows else ''
                append_rows(path, T2_FIELDS, [dict(
                    setting=f'N{setting[0]}_wd{setting[1]}', seed=seed,
                    source_kind=name, target=target, step=args.steps, dev_acc=final,
                    source_final_dev=summary.get('final_dev'),
                    source_grok_step=summary.get('grok_step'))])
                common.log_line(f'[q16] t2 {name} {target} seed {seed}: dev {final}')
    return notes



# ---------------------------------------------------------------------------
# STEP T3: the modular polynomial
# ---------------------------------------------------------------------------
def step_t3(args, device):
    """The polynomial sources and their transfers."""
    out_dir = project_path(args.out)
    path = out_dir / 't3_poly.csv'
    notes = []
    source_dir = out_dir / 'sources' / 'poly' / args.config
    data = poly_data(device, 'poly')
    for seed in args.seeds:
        if not (source_dir / f'seed_{seed}' / 'summary.json').exists():
            notes.append(f'poly seed {seed}: the source was trained now')
            train_source(seed, 'poly', source_dir / f'seed_{seed}', args.cap, device,
                         data=data, source_task='poly', ckpt_every=250)
    targets = {'affine_poly': poly_data(device, 'affine_poly'),
               'affine_add': common.build_task('affine_add', device),
               'affine_mult': common.build_task('affine_mult', device)}
    for seed in args.seeds:
        seed_path = source_dir / f'seed_{seed}'
        summary = common.read_summary(seed_path / 'summary.json')
        stages = summary.get('stages', {}) or {}
        if not stages:
            notes.append(f'poly seed {seed}: no stages')
            continue
        grokked_norm = (stages.get('grokked', {}) or {}).get('global_norm')
        mask = common.mask_for_seed(seed)
        strata, _exposure = common.split_for_mask(mask, seed)
        sources = {}
        for stage in ('memorized', 'transition', 'grokked'):
            if stage in stages:
                sources[stage] = (stages[stage]['step'],
                                  common.load_checkpoint(seed_path / 'ckpts' /
                                                         stages[stage]['ckpt'])['model'])
        if 'memorized' in stages and grokked_norm is not None:
            sources['norm_control'] = ('', source_state(seed_path, 'norm_control',
                                                         stages, grokked_norm))
        model = common.init_model(seed, device)
        sources['random_init'] = ('', {key: value.detach().cpu() for key, value
                                       in model.state_dict().items()})
        for kind, (step, state) in sources.items():
            for target, target_data in targets.items():
                transfer_rows = toy_transfer(state, seed, target_data, strata, device,
                                             f't3_{kind}_{target}_seed{seed}',
                                             steps=args.steps,
                                             eval_every=args.eval_every)
                final = transfer_rows[-1]['dev_mean'] if transfer_rows else ''
                append_rows(path, T3_FIELDS, [dict(
                    source_task='poly', target_task=target, seed=seed,
                    source_kind=kind, step=args.steps, dev_acc=final,
                    source_final_dev=summary.get('source_final_dev'),
                    source_grok_step=summary.get('grok_step'))])
                common.log_line(f'[q16] t3 {kind} {target} seed {seed}: dev {final}')
    return notes


# ---------------------------------------------------------------------------
# STEP T4: the groups
# ---------------------------------------------------------------------------
def step_t4(args, device):
    """The group sources and their power-affine transfers."""
    out_dir = project_path(args.out)
    path = out_dir / 't4_groups.csv'
    notes = []
    if args.group == 'D59':
        size, table = 118, dihedral_table(59)
    else:
        size, table = 120, permutation_table()
    common.P = size
    common.CONFIG['p'] = size
    common.COUNT = int(common.COVERAGE * size * size)
    source_dir = out_dir / 'sources' / args.group / args.config
    data = group_data(table, size, device, power=False)
    for seed in args.seeds:
        if not (source_dir / f'seed_{seed}' / 'summary.json').exists():
            notes.append(f'{args.group} seed {seed}: the source was trained now')
            train_source(seed, args.group, source_dir / f'seed_{seed}', args.cap,
                         device, data=data, source_task=args.group,
                         ckpt_every=args.ckpt_every)
    target_data = group_data(table, size, device, power=True)
    for seed in args.seeds:
        seed_path = source_dir / f'seed_{seed}'
        summary = common.read_summary(seed_path / 'summary.json')
        stages = summary.get('stages', {}) or {}
        if not stages:
            notes.append(f'{args.group} seed {seed}: no stages')
            continue
        grokked_norm = (stages.get('grokked', {}) or {}).get('global_norm')
        mask = common.mask_for_seed(seed)
        strata, _exposure = common.split_for_mask(mask, seed)
        sources = {}
        for stage in ('memorized', 'transition', 'grokked'):
            if stage in stages:
                sources[stage] = (stages[stage]['step'],
                                  common.load_checkpoint(seed_path / 'ckpts' /
                                                         stages[stage]['ckpt'])['model'])
        if 'memorized' in stages and grokked_norm is not None:
            sources['norm_control'] = ('', source_state(seed_path, 'norm_control',
                                                         stages, grokked_norm))
        model = common.init_model(seed, device)
        sources['random_init'] = ('', {key: value.detach().cpu() for key, value
                                       in model.state_dict().items()})
        for kind, (step, state) in sources.items():
            transfer_rows = toy_transfer(state, seed, target_data, strata, device,
                                         f't4_{args.group}_{kind}_seed{seed}',
                                         steps=args.steps, eval_every=args.eval_every)
            final = transfer_rows[-1]['dev_mean'] if transfer_rows else ''
            append_rows(path, T4_FIELDS, [dict(
                group=args.group, config=args.config, seed=seed, source_kind=kind,
                step=args.steps, dev_acc=final,
                source_final_dev=summary.get('source_final_dev'),
                source_grok_step=summary.get('grok_step'))])
            common.log_line(f'[q16] t4 {args.group} {kind} seed {seed}: dev {final}')
    common.P = 59
    common.CONFIG['p'] = 59
    common.COUNT = int(common.COVERAGE * common.P * common.P)
    return notes


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------
def raise_file_limit(limit=65535):
    """The worker pool needs more open files than the default limit."""
    try:
        import resource
        _soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(limit, hard), hard))
    except Exception as exc:  # noqa: BLE001 - the limit is only a convenience
        common.log_line(f'[q16] the open file limit was not raised: {exc}')


def run_smoke(args, device):
    """The parity model, the polynomial data and the Fourier concentration."""
    data = poly_data(device, 'poly')
    print(f'the polynomial data has {data["x"].shape[0]} rows and the labels stay '
          f'below {data["y"].max().item() + 1}', flush=True)
    model = common.make_model(common.CONFIG)
    model = model.to(device)
    strata, _exposure = common.split_for_mask(common.mask_for_seed(5), 5)
    train_acc, dev_acc, norm = stage_metrics(model, data, strata, device)
    frac_a, frac_b = embedding_concentration(model)
    print(f'the toy model: train {train_acc:.4f} dev {dev_acc:.4f} norm {norm:.3f} '
          f'frac_top5_a {frac_a:.4f} frac_top5_b {frac_b:.4f}', flush=True)
    parity = ParityTransformer(n_bits=30).to(device)
    strings = parity_strings(64, 30, 7)
    subset = sorted(random.Random(7).sample(range(30), 3))
    tokens, labels = parity_batch(strings, subset, device)
    logits = parity(tokens)
    print(f'the parity model returns {tuple(logits.shape)} logits for 64 strings of '
          f'30 bits', flush=True)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    print(f'the parity loss is {float(loss):.4f}', flush=True)
    started = time.time()
    parity_train(parity, strings, subset, device, steps=5, batch_size=16,
                 learning_rate=1e-3, weight_decay=1.0)
    print(f'five parity steps: {time.time() - started:.2f} s', flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['t1', 't2', 't3', 't4'])
    parser.add_argument('--out', default='runs_nseed_queue16')
    parser.add_argument('--seeds', nargs='+', type=int, default=[5, 6, 7, 8, 9])
    parser.add_argument('--n-values', nargs='+', type=int,
                        default=[2000, 5000, 10000, 20000])
    parser.add_argument('--wd-values', nargs='+', type=float, default=[1.0, 2.0])
    parser.add_argument('--group', default='D59', choices=['D59', 'S5'])
    parser.add_argument('--config', default='cov0.3_wd1.0')
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--d-mlp', type=int, default=512)
    parser.add_argument('--cap', type=int, default=50000)
    parser.add_argument('--ckpt-every', type=int, default=1000)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--procs', type=int, default=4)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--transfers-only', action='store_true')
    args = parser.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    raise_file_limit()
    common.CONFIG['d_model'] = args.d_model
    common.CONFIG['d_mlp'] = args.d_mlp
    if args.step == 't4' and args.group == 'S5':
        common.COVERAGE = 0.6
    if args.smoke:
        return run_smoke(args, device)
    notes = []
    if args.step == 't1':
        notes = step_t1(args, device)
    elif args.step == 't2':
        notes = step_t2(args, device)
    elif args.step == 't3':
        notes = step_t3(args, device)
    elif args.step == 't4':
        notes = step_t4(args, device)
    for note in notes:
        common.log_line(f'[q16] {note}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
