"""Queue 16, step R1: the pilot of the invented-symbol task at a realistic
coverage.

The task is the one of queue15 (the saved seed-0 symbol map, the prompt ends
with the equals sign, the loss is taken over the last token only), but the
coverage is the one the toy sources need (about 40 percent of the pairs) and
the schedule has a warmup.

For pythia-160m the grid is p in {59, 97} times coverage in {0.35, 0.5} times
lr in {3e-4, 1e-3} times weight decay in {1.0, 3.0}, capped at 40,000 steps.
For pythia-410m it is p = 59 times coverage in {0.35, 0.5} times lr 3e-4 times
weight decay in {1.0, 3.0}, capped at 20,000 steps. No checkpoint is written.

If no configuration shows the delayed generalization, the three 160m
configurations with the highest held-out accuracy continue to 120,000 steps
from their state.

Produces: the grid of the Pythia pilot of queue 16.
"""
from __future__ import annotations

import argparse
import json
import random
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import common  # noqa: E402
from p15_common import (MODELS, REVISION, clear_model_cache, load_model,  # noqa: E402
                        read_rows, token_accuracy, write_rows)

FIELDS = ['model', 'p', 'coverage', 'N', 'lr', 'wd', 'step', 'train_acc', 'heldout_acc']
GRID_160M = [('160m', p, coverage, lr, wd)
             for p in (59, 97)
             for coverage in (0.35, 0.5)
             for lr in (3e-4, 1e-3)
             for wd in (1.0, 3.0)]
GRID_410M = [('410m', 59, coverage, 3e-4, wd)
             for coverage in (0.35, 0.5)
             for wd in (1.0, 3.0)]
CAPS = {'160m': 40000, '410m': 20000}
HELDOUT = 1000
TRAIN_SAMPLE = 1000
R1B_STEPS = 120000
R1B_TOP = 3
BETAS = (0.9, 0.98)


def key_of(model, p, coverage, lr, wd):
    return f'{model}|{p}|{coverage}|{lr}|{wd}'


def splits_for(p, heldout=HELDOUT, seed=0):
    """The fixed pair splits of a task of size p x p."""
    pairs = [(a, b) for a in range(p) for b in range(p)]
    random.Random(seed).shuffle(pairs)
    return dict(train_pool=pairs[heldout:], heldout=pairs[:heldout])


def make_examples(symbols, pairs, p, rule='add'):
    """The (prompt, answer) examples of a rule at the size p."""
    prompts = [f'{symbols[str(a)]} {symbols[str(b)]} =' for a, b in pairs]
    if rule == 'add':
        answers = [symbols[str((a + b) % p)] for a, b in pairs]
    elif rule == 'a2b':
        answers = [symbols[str((a + 2 * b) % p)] for a, b in pairs]
    elif rule == 'shuffled':
        answers = [symbols[str((a + b) % p)] for a, b in pairs]
        random.Random(0).shuffle(answers)
    else:
        raise ValueError(f'unknown rule {rule!r}')
    return list(zip(prompts, answers))


def answer_ids(payload, pairs, p, rule='add'):
    """The answer token ids of the pairs."""
    token_ids = payload['token_ids']
    out = []
    for a, b in pairs:
        residue = (a + 2 * b) % p if rule == 'a2b' else (a + b) % p
        out.append(token_ids[str(residue)])
    return out


def prepare_examples(examples, tokenizer, device):
    """The tokenised training set, tokenized once.

    Tokenizing the batch inside the loop costs more than the training step
    itself, so the whole training set is tokenized once and the batches are
    drawn from the cached tensors.
    """
    texts = [text + ' ' + answer for text, answer in examples]
    encoded = tokenizer(texts, return_tensors='pt', padding=True, truncation=True,
                        max_length=64)
    ids = encoded['input_ids']
    labels = ids.clone()
    labels[encoded['attention_mask'] == 0] = -100
    labels[:, :-1] = -100
    return ids.to(device), labels.to(device)


def fine_tune(model, tokenizer, examples, device, steps, batch_size, learning_rate,
              weight_decay, warmup=100, seed=0, eval_every=0, evaluate=None,
              hook=None, hook_every=0):
    """Full fine-tuning with a warmup and the betas (0.9, 0.98).

    With `evaluate` given it is called every `eval_every` steps; the list of
    (step, value) pairs is returned. With `hook` given it is called every
    `hook_every` steps, which lets a caller write a temporary checkpoint and
    work on it while the training state stays in memory.
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay, betas=BETAS)
    ids, labels = prepare_examples(examples, tokenizer, device)
    generator = torch.Generator(device=ids.device).manual_seed(int(seed))
    measured = []
    for step in range(1, steps + 1):
        if warmup and step <= warmup:
            scale = step / warmup
            for group in optimizer.param_groups:
                group['lr'] = learning_rate * scale
        else:
            for group in optimizer.param_groups:
                group['lr'] = learning_rate
        index = torch.randint(0, ids.shape[0], (batch_size,), generator=generator,
                               device=ids.device)
        loss = model(input_ids=ids[index], labels=labels[index]).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if hook is not None and hook_every and (step % hook_every == 0
                                                or step == steps):
            hook(step)
        if evaluate is not None and eval_every and (step % eval_every == 0
                                                    or step == steps):
            measured.append((step, evaluate()))
    model.eval()
    return measured


def criterion(rows):
    """The delayed-generalization criterion on the rows of one configuration."""
    rows = sorted(rows, key=lambda row: int(row['step']))
    result = dict(qualifies=False, train_90_step=None, heldout_80_step=None,
                  lag=None, train_window=None, max_heldout=None)
    result['max_heldout'] = max((float(row['heldout_acc']) for row in rows),
                                default=None)
    train_90 = next((int(row['step']) for row in rows
                     if float(row['train_acc']) >= 0.9), None)
    heldout_80 = next((int(row['step']) for row in rows
                       if float(row['heldout_acc']) >= 0.8), None)
    result['train_90_step'] = train_90
    result['heldout_80_step'] = heldout_80
    if train_90 is None or heldout_80 is None or heldout_80 <= train_90:
        return result
    run = []
    best = 0
    for row in rows:
        step = int(row['step'])
        if step < train_90 or step > heldout_80:
            continue
        if float(row['train_acc']) >= 0.9 and float(row['heldout_acc']) < 0.10:
            run.append(step)
        else:
            best = max(best, run[-1] - run[0] if len(run) > 1 else 0)
            run = []
    best = max(best, run[-1] - run[0] if len(run) > 1 else 0)
    result['train_window'] = best
    result['lag'] = heldout_80 - train_90
    result['qualifies'] = best >= 500
    return result


def run_config(args, device, payload, model_key, p, coverage, lr, wd, steps,
               table, out_path, resume_from=None):
    """One configuration of the grid; its rows are written as they come."""
    model_name = MODELS[model_key]
    model, tokenizer = load_model(model_name, REVISION, device)
    splits = splits_for(p)
    n_pairs = int(round(coverage * p * p))
    train_pairs = splits['train_pool'][:n_pairs]
    heldout_pairs = splits['heldout']
    examples = make_examples(payload['symbols'], train_pairs, p, 'add')
    sample = train_pairs[:TRAIN_SAMPLE]
    prompts_train = [text for text, _answer in make_examples(payload['symbols'],
                                                              sample, p, 'add')]
    ids_train = answer_ids(payload, sample, p, 'add')
    prompts_held = [f'{payload["symbols"][str(a)]} {payload["symbols"][str(b)]} ='
                    for a, b in heldout_pairs]
    ids_held = answer_ids(payload, heldout_pairs, p, 'add')
    key = key_of(model_key, p, coverage, lr, wd)

    def evaluate():
        return (token_accuracy(model, tokenizer, prompts_train, ids_train, device),
                token_accuracy(model, tokenizer, prompts_held, ids_held, device))

    def store(step, values):
        train_acc, heldout_acc = values
        row = dict(model=model_key, p=p, coverage=coverage, N=n_pairs, lr=lr, wd=wd,
                   step=step, train_acc=train_acc, heldout_acc=heldout_acc)
        table[f'{key}|{step}'] = row
        write_rows(out_path, FIELDS, [table[k] for k in sorted(table)])
        common.log_line(f'[q16] {key} step {step}: train {train_acc:.4f} '
                        f'heldout {heldout_acc:.4f}')
        return row

    train_acc, heldout_acc = evaluate()
    store(0, (train_acc, heldout_acc))
    start = 1
    if resume_from:
        start = int(resume_from) + 1
    measured = fine_tune(model, tokenizer, examples, device, steps=steps - start + 1,
                         batch_size=args.batch_size, learning_rate=lr,
                         weight_decay=wd, warmup=args.warmup, seed=args.seed,
                         eval_every=args.eval_every, evaluate=evaluate)
    for step, values in measured:
        store(step, values)
    del model
    torch.cuda.empty_cache()
    return rows_of(table, key)


def rows_of(table, key):
    return [row for row in table.values() if row.get('key') == key
            or row.get('_key') == key] or [row for name, row in table.items()
                                            if name.startswith(key + '|')]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='runs_nseed_queue16/p16_pilot.csv')
    parser.add_argument('--symbols', default='runs_nseed_queue15/symbols_p97.json')
    parser.add_argument('--verdicts', default='runs_nseed_queue16/r1_verdicts.json')
    parser.add_argument('--steps', type=int, default=40000)
    parser.add_argument('--steps-410', type=int, default=20000)
    parser.add_argument('--steps-b', type=int, default=R1B_STEPS)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if hasattr(signal, 'SIGUSR1'):
        import faulthandler
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    if args.smoke:
        return run_smoke(args, device)

    payload = json.loads(Path(args.symbols).read_text(encoding='utf-8'))
    out_path = Path(args.out)
    table = {}
    for row in read_rows(out_path):
        table[f'{row["model"]}|{row["p"]}|{row["coverage"]}|{row["lr"]}|{row["wd"]}|'
              f'{row["step"]}'] = row
    verdicts = []
    for model_key, grid, cap in (('160m', GRID_160M, args.steps),
                                  ('410m', GRID_410M, args.steps_410)):
        for p, coverage, lr, wd in [(g[1], g[2], g[3], g[4]) for g in grid
                                    if g[0] == model_key]:
            key = key_of(model_key, p, coverage, lr, wd)
            existing = [row for name, row in table.items() if name.startswith(key + '|')]
            if existing and int(existing[-1]['step']) >= cap:
                print(f'{key}: already done', flush=True)
            else:
                print(f'{key}: starts', flush=True)
                run_config(args, device, payload, model_key, p, coverage, lr, wd,
                           cap, table, out_path)
            rows = [row for name, row in table.items() if name.startswith(key + '|')]
            rows.sort(key=lambda row: int(row['step']))
            verdict = criterion(rows)
            verdict.update(model=model_key, p=p, coverage=coverage, lr=lr, wd=wd,
                           N=int(round(coverage * p * p)), steps=rows[-1]['step'] if rows else '',
                           final_train=rows[-1]['train_acc'] if rows else '',
                           final_heldout=rows[-1]['heldout_acc'] if rows else '')
            verdicts.append(verdict)
            print(f'{key}: train {verdict["final_train"]} heldout '
                  f'{verdict["final_heldout"]} max heldout {verdict["max_heldout"]} '
                  f'qualifies {verdict["qualifies"]}', flush=True)
        clear_model_cache(MODELS[model_key])

    if not any(row['qualifies'] for row in verdicts):
        print('no configuration qualifies: the three best 160m configurations '
              'continue to the longer budget', flush=True)
        best = sorted([row for row in verdicts if row['model'] == '160m'],
                      key=lambda row: row['max_heldout'] or 0, reverse=True)[:R1B_TOP]
        for verdict in best:
            key = key_of(verdict['model'], verdict['p'], verdict['coverage'],
                         verdict['lr'], verdict['wd'])
            existing = [row for name, row in table.items()
                        if name.startswith(key + '|')]
            done = int(existing[-1]['step']) if existing else 0
            if done >= args.steps_b:
                continue
            print(f'{key}: continues from step {done} to {args.steps_b}', flush=True)
            run_config(args, device, payload, verdict['model'], verdict['p'],
                       verdict['coverage'], verdict['lr'], verdict['wd'], args.steps_b,
                       table, out_path, resume_from=done)
            rows = [row for name, row in table.items() if name.startswith(key + '|')]
            rows.sort(key=lambda row: int(row['step']))
            new = criterion(rows)
            new.update(model=verdict['model'], p=verdict['p'],
                       coverage=verdict['coverage'], lr=verdict['lr'], wd=verdict['wd'],
                       N=verdict['N'], steps=rows[-1]['step'],
                       final_train=rows[-1]['train_acc'],
                       final_heldout=rows[-1]['heldout_acc'], grid='R1b')
            verdicts = [row for row in verdicts if row.get('key') != key]
            verdicts.append(new)
            print(f'{key}: train {new["final_train"]} heldout {new["final_heldout"]} '
                  f'qualifies {new["qualifies"]}', flush=True)
        clear_model_cache(MODELS['160m'])

    for row in verdicts:
        row['key'] = key_of(row['model'], row['p'], row['coverage'], row['lr'],
                            row['wd'])
    Path(args.verdicts).parent.mkdir(parents=True, exist_ok=True)
    Path(args.verdicts).write_text(json.dumps(verdicts, indent=2, default=str),
                                    encoding='utf-8')
    chosen = [row for row in verdicts if row['qualifies']]
    if chosen:
        best = max(chosen, key=lambda row: row['lag'])
        print(f'the chosen configuration: {best["key"]} lag {best["lag"]}', flush=True)
    else:
        print('no configuration satisfies the criterion', flush=True)
    return 0


def run_smoke(args, device):
    """The splits, the examples, one training step and the step time."""
    payload = json.loads(Path(args.symbols).read_text(encoding='utf-8'))
    for p in (59, 97):
        splits = splits_for(p)
        print(f'p={p}: pool {len(splits["train_pool"])} heldout '
              f'{len(splits["heldout"])} N(0.35) {int(round(0.35 * p * p))} '
              f'N(0.5) {int(round(0.5 * p * p))}', flush=True)
        pairs = splits['train_pool'][:64]
        examples = make_examples(payload['symbols'], pairs, p, 'add')
        print(f'  prompt {examples[0][0]!r} answer {examples[0][1]!r}', flush=True)
    model, tokenizer = load_model(MODELS['160m'], REVISION, device)
    pairs = splits_for(59)['train_pool'][:64]
    examples = make_examples(payload['symbols'], pairs, 59, 'add')
    started = time.time()
    fine_tune(model, tokenizer, examples, device, steps=10, batch_size=16,
              learning_rate=3e-4, weight_decay=1.0, warmup=5)
    print(f'ten training steps at batch 16: {time.time() - started:.2f} s', flush=True)
    clear_model_cache(MODELS['160m'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
