"""Queue 17, step P1: the trajectories with the budget-fair and the specificity
transfer, and step P2: the mechanism test inside the 160m model.

P1 runs four stage-1 configurations (A, A_decay, C and D) with the sources gen,
mem and shuffled. At every step of the checkpoint schedule the source metrics
are recorded, one temporary checkpoint is written, the stage-2 transfers are
run from it and the checkpoint is deleted again.

P2 rides on the config A gen trajectories of the seeds 0 to 4: the energy
spectrum of the symbol embeddings and of the unembedding rows is recorded, and
at step 5000 the key frequencies are removed from the model and the accuracy
is measured again.

    python nseed_backbone/p17_traj.py --out runs_nseed_queue17/p17_traj.csv

Produces: the trajectories and the frequency spectrum of queue 17.
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
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import common  # noqa: E402
from p15_common import (MODELS, REVISION, clear_model_cache, load_model,  # noqa: E402
                        token_accuracy)
from p16_pilot import criterion  # noqa: E402
from p17_common import (STAGE2_BATCH, STAGE2_PAIRS, answer_ids,  # noqa: E402
                         direct_baseline, fine_tune, load_checkpoint,
                         make_examples, save_checkpoint, splits_for,
                         stage2_accuracy)

TRAJ_FIELDS = ['model', 'p', 'variant', 'seed', 'source', 'step', 'train_acc',
               'heldout_acc', 'train_loss', 'param_norm', 'stage2_target',
               'stage2_budget', 'stage2_acc', 'gain_over_best_direct']
DIRECT_FIELDS = ['model', 'p', 'seed', 'target', 'budget', 'acc']
FOURIER_FIELDS = ['seed', 'step', 'model_part', 'frac_top5', 'top5_freqs', 'edit',
                  'heldout_acc', 'stage2_acc']
CONFIGS = {
    'A': dict(model='160m', p=59, coverage=0.5, lr=3e-4, wd=1.0, horizon=6000,
              seeds=range(0, 10), sources=('gen', 'mem', 'shuffled'), decay=False),
    'A_decay': dict(model='160m', p=59, coverage=0.5, lr=3e-4, wd=1.0, horizon=6000,
                    seeds=range(0, 5), sources=('gen', 'mem', 'shuffled'),
                    decay=True),
    'C': dict(model='410m', p=59, coverage=0.5, lr=3e-4, wd=1.0, horizon=5500,
              seeds=range(0, 5), sources=('gen', 'mem'), decay=False),
    'D': dict(model='160m', p=97, coverage=0.5, lr=3e-4, wd=1.0, horizon=5500,
              seeds=range(0, 5), sources=('gen', 'mem'), decay=False),
}
SHUFFLED_SEEDS = (0, 1, 2, 3, 4)
MECHANISM_SEEDS = (0, 1, 2, 3, 4)
LONG_BUDGET_STEPS = (3000, 4000, 5000)


def schedule(horizon):
    """The checkpoint schedule S."""
    steps = [0, 500, 1000, 1500, 2000]
    steps += list(range(2500, horizon + 1, 100))
    return sorted({step for step in steps if step <= horizon})


def stage2_plan(variant, seed, source, steps, horizon):
    """The (step, target, budget) transfers of one trajectory."""
    plan = [(step, 'add2', 600) for step in steps]
    if source in ('gen', 'mem'):
        for step in list(LONG_BUDGET_STEPS) + [horizon]:
            if step in steps:
                plan.append((step, 'add2', 3000))
    if variant == 'A' and seed in SHUFFLED_SEEDS and source in ('gen', 'mem'):
        for step in range(2500, 6001, 500):
            if step in steps:
                plan.append((step, 'sub', 600))
                plan.append((step, 'mult', 600))
    return plan


def param_norm(model):
    """The global parameter norm of the model."""
    total = 0.0
    for parameter in model.parameters():
        total += float(parameter.detach().float().pow(2).sum())
    return total ** 0.5


def embedding_tables(model):
    """The input embedding table and the unembedding matrix."""
    embed = None
    unembed = None
    for module in model.modules():
        if isinstance(module, nn.Embedding) and embed is None:
            embed = module.weight
        if isinstance(module, nn.Linear) and unembed is None and \
                module.out_features == model.config.vocab_size:
            unembed = module.weight
    return embed, unembed


def symbol_rows(weight, payload, p):
    """The rows of the p symbol tokens, ordered by the residue index."""
    ids = [payload['token_ids'][str(residue)] for residue in range(p)]
    return weight[ids].detach().float()


def spectrum(embeddings):
    """The energy per frequency along the residue axis and the top-5 fraction."""
    spec = torch.fft.rfft(embeddings, dim=0)
    energy = (spec.real ** 2 + spec.imag ** 2).sum(dim=1)
    total = float(energy.sum())
    values, indices = torch.topk(energy, 5)
    fraction = float(values.sum()) / total if total > 0 else 0.0
    return fraction, sorted(int(index) for index in indices)


def project_out(embeddings, frequencies):
    """The embeddings with the given frequencies removed along the residue axis."""
    spec = torch.fft.rfft(embeddings, dim=0)
    for frequency in frequencies:
        if 0 <= frequency < spec.shape[0]:
            spec[frequency] = 0
    return torch.fft.irfft(spec, n=embeddings.shape[0], dim=0)


def apply_edit(model, payload, p, frequencies):
    """Remove the frequencies from the symbol embeddings and the unembedding."""
    embed, unembed = embedding_tables(model)
    ids = [payload['token_ids'][str(residue)] for residue in range(p)]
    with torch.no_grad():
        rows = symbol_rows(embed, payload, p)
        edited = project_out(rows, frequencies)
        embed[ids] = edited.to(embed.dtype)
        rows = symbol_rows(unembed, payload, p)
        edited = project_out(rows, frequencies)
        unembed[ids] = edited.to(unembed.dtype)
    return model


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


def write_rows(path, fields, rows):
    """Rewrite the whole CSV; the table is the state of the run."""
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(out.getvalue(), encoding='utf-8')
    return path



# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------
def best_direct(direct_table, target, budget, model_key, p, seed):
    """The best direct baseline of the same target with a budget not larger."""
    values = [float(row['acc']) for row in direct_table
              if row['target'] == target and int(row['budget']) <= budget
              and row['model'] == model_key and row['p'] == str(p)
              and row['seed'] == str(seed) and row['acc'] not in ('', None)]
    return max(values) if values else 0.0


def run_combo(args, device, payload, variant, seed, source, direct_table,
              traj_path, direct_path, fourier_path, fourier_table):
    """One trajectory: the stage-1 training with the stage-2 transfers."""
    config = CONFIGS[variant]
    model_name = MODELS[config['model']]
    p = config['p']
    horizon = config['horizon']
    splits = splits_for(p)
    n_pairs = int(round(config['coverage'] * p * p))
    if source == 'mem':
        pairs = splits['train_pool'][:100]
    else:
        pairs = splits['train_pool'][:n_pairs]
    rule = 'shuffled' if source == 'shuffled' else 'add'
    examples = make_examples(payload['symbols'], pairs, p, rule)
    prompts_train = [text for text, _answer in examples]
    ids_train = answer_ids(payload, pairs, p, rule)
    heldout = splits['heldout']
    prompts_held = [f'{payload["symbols"][str(a)]} {payload["symbols"][str(b)]} ='
                    for a, b in heldout]
    ids_held = answer_ids(payload, heldout, p, 'add')

    model, tokenizer = load_model(model_name, REVISION, device)
    steps = schedule(horizon)
    plan = stage2_plan(variant, seed, source, steps, horizon)
    plan_by_step = collections.defaultdict(list)
    for step, target, budget in plan:
        plan_by_step[step].append((target, budget))
    work = Path(args.work)
    rows = []

    def hook(step, losses):
        if step not in steps:
            return
        train_acc = token_accuracy(model, tokenizer, prompts_train, ids_train, device)
        heldout_acc = token_accuracy(model, tokenizer, prompts_held, ids_held, device)
        window = losses[-100:]
        loss = sum(window) / max(1, len(window))
        norm = param_norm(model)
        for target, budget in plan_by_step.get(step, []):
            checkpoint = work / f'{variant}_{source}_seed{seed}_step{step}.pt'
            save_checkpoint(checkpoint, model)
            forked, forked_tokenizer = load_checkpoint(checkpoint, model_name, device)
            accuracy = stage2_accuracy(forked, forked_tokenizer, payload, splits, p,
                                       device, config['lr'], config['wd'], target,
                                       seed, budget)
            del forked, forked_tokenizer
            torch.cuda.empty_cache()
            checkpoint.unlink()
            gain = accuracy - best_direct(direct_table, target, budget,
                                          config['model'], p, seed)
            rows.append(dict(model=config['model'], p=p, variant=variant, seed=seed,
                             source=source, step=step, train_acc=train_acc,
                             heldout_acc=heldout_acc, train_loss=loss,
                             param_norm=norm, stage2_target=target,
                             stage2_budget=budget, stage2_acc=accuracy,
                             gain_over_best_direct=gain))
            append_rows(traj_path, TRAJ_FIELDS, [rows[-1]])
        if variant == 'A' and seed in MECHANISM_SEEDS and source == 'gen':
            record_mechanism(args, device, payload, model, model_name, splits, p,
                             seed, step, config, train_acc, heldout_acc, norm,
                             fourier_path, fourier_table, work)

    fine_tune(model, tokenizer, examples, device, steps=horizon,
              learning_rate=config['lr'], weight_decay=config['wd'],
              horizon=horizon, decay=config['decay'], seed=seed * 7 + len(source),
              hook=hook, hook_every=100)
    del model
    torch.cuda.empty_cache()
    return rows


def record_mechanism(args, device, payload, model, model_name, splits, p, seed, step,
                     config, train_acc, heldout_acc, norm, fourier_path,
                     fourier_table, work):
    """The spectrum rows and, at step 5000, the frequency removal edits."""
    embed, unembed = embedding_tables(model)
    rows = []
    if step % 500 == 0 and step >= 2500:
        fraction, indices = spectrum(symbol_rows(embed, payload, p))
        rows.append(dict(seed=seed, step=step, model_part='input_embedding',
                         frac_top5=fraction, top5_freqs=';'.join(str(i) for i in indices),
                         edit='none', heldout_acc='', stage2_acc=''))
        fraction, indices = spectrum(symbol_rows(unembed, payload, p))
        rows.append(dict(seed=seed, step=step, model_part='unembedding',
                         frac_top5=fraction, top5_freqs=';'.join(str(i) for i in indices),
                         edit='none', heldout_acc='', stage2_acc=''))
    if step == 5000:
        fraction, indices = spectrum(symbol_rows(embed, payload, p))
        edits = [('none', None)] + [('key5', indices)]
        available = [f for f in range(len(symbol_rows(embed, payload, p)) // 2 + 1)
                     if f not in indices]
        for name in ('rand5_a', 'rand5_b', 'rand5_c'):
            draw = sorted(random.Random(hash(name) % 1000 + seed).sample(available, 5))
            edits.append((name, draw))
        checkpoint = work / f'mech_seed{seed}_step{step}.pt'
        save_checkpoint(checkpoint, model)
        for name, frequencies in edits:
            edited, edited_tokenizer = load_checkpoint(checkpoint, model_name, device)
            if frequencies is not None:
                apply_edit(edited, payload, p, frequencies)
            held = token_accuracy(edited, edited_tokenizer, prompts_held_of(payload,
                                  splits['heldout'], p), ids_held_of(payload,
                                  splits['heldout'], p), device)
            accuracy = stage2_accuracy(edited, edited_tokenizer, payload, splits, p,
                                       device, config['lr'], config['wd'], 'add2',
                                       seed, 600)
            del edited, edited_tokenizer
            torch.cuda.empty_cache()
            rows.append(dict(seed=seed, step=step, model_part='edited', frac_top5='',
                             top5_freqs=('' if frequencies is None else
                                         ';'.join(str(i) for i in frequencies)),
                             edit=name, heldout_acc=held, stage2_acc=accuracy))
        checkpoint.unlink()
    if rows:
        for row in rows:
            fourier_table[f'{seed}|{step}|{row["model_part"]}|{row["edit"]}'] = row
        append_rows(fourier_path, FOURIER_FIELDS, rows)


def prompts_held_of(payload, heldout, p):
    return [f'{payload["symbols"][str(a)]} {payload["symbols"][str(b)]} ='
            for a, b in heldout]


def ids_held_of(payload, heldout, p):
    return answer_ids(payload, heldout, p, 'add')


def direct_worker(job):
    """One direct baseline: the stage-2 task from the pretrained model."""
    try:
        config = CONFIGS[job['variant']]
        splits = splits_for(config['p'])
        accuracy = direct_baseline(MODELS[config['model']], job['payload'], splits,
                                   config['p'], job['device'], config['lr'],
                                   config['wd'], job['seed'], job['target'],
                                   job['budget'])
        return dict(job=job, accuracy=accuracy, error=None)
    except Exception as exc:  # noqa: BLE001 - one baseline must not kill the pool
        return dict(job=job, accuracy=None, error=str(exc))


def worker_init(device, batch_log, threads):
    common.setup(threads, device)


def worker(job):
    try:
        rows = run_combo(job['args'], job['device'], job['payload'], job['variant'],
                         job['seed'], job['source'], job['direct_table'],
                         job['traj_path'], job['direct_path'], job['fourier_path'],
                         job['fourier_table'])
        return dict(job=job, rows=rows, error=None)
    except Exception as exc:  # noqa: BLE001 - one trajectory must not kill the pool
        common.log_line(f'[q17] {job["variant"]} seed {job["seed"]} {job["source"]} '
                        f'FAILED {type(exc).__name__}: {exc}')
        common.log_line(traceback.format_exc())
        return dict(job=job, rows=[], error=str(exc))


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



def run_smoke(args, device):
    """The training loop, the checkpoint, the stage 2 and the spectrum."""
    payload = json.loads(Path(args.symbols).read_text(encoding='utf-8'))
    splits = splits_for(59)
    examples = make_examples(payload['symbols'], splits['train_pool'][:200], 59, 'add')
    model, tokenizer = load_model(MODELS['160m'], REVISION, device)

    def hook(step, losses):
        if step != 100:
            return
        checkpoint = Path(args.work) / 'smoke.pt'
        save_checkpoint(checkpoint, model)
        forked, forked_tokenizer = load_checkpoint(checkpoint, MODELS['160m'], device)
        accuracy = stage2_accuracy(forked, forked_tokenizer, payload, splits, 59,
                                   device, 3e-4, 1.0, 'add2', 0, 20)
        checkpoint.unlink()
        print(f'the smoke stage 2 accuracy: {accuracy:.4f}', flush=True)
        embed, unembed = embedding_tables(model)
        fraction, indices = spectrum(symbol_rows(embed, payload, 59))
        print(f'the smoke spectrum: frac_top5 {fraction:.4f} {indices}', flush=True)
        apply_edit(model, payload, 59, indices)
        print('the smoke edit was applied', flush=True)

    fine_tune(model, tokenizer, examples, device, steps=120, learning_rate=3e-4,
              weight_decay=1.0, horizon=6000, hook=hook, hook_every=100)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='runs_nseed_queue17/p17_traj.csv')
    parser.add_argument('--direct', default='runs_nseed_queue17/p17_direct.csv')
    parser.add_argument('--fourier', default='runs_nseed_queue17/m17_fourier.csv')
    parser.add_argument('--symbols', default='runs_nseed_queue15/symbols_p97.json')
    parser.add_argument('--work', default='runs_nseed_queue17/work')
    parser.add_argument('--variants', nargs='+', default=list(CONFIGS))
    parser.add_argument('--procs', type=int, default=6)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.smoke:
        return run_smoke(args, device)
    payload = json.loads(Path(args.symbols).read_text(encoding='utf-8'))
    Path(args.work).mkdir(parents=True, exist_ok=True)

    direct_table = {}
    for row in read_rows(args.direct):
        direct_table[(row['model'], row['p'], row['seed'], row['target'],
                      row['budget'])] = row

    # the direct baselines, one per (configuration, seed, target, budget)
    direct_jobs = []
    for variant in args.variants:
        config = CONFIGS[variant]
        for seed in config['seeds']:
            for target, budget in (('add2', 600), ('add2', 1600), ('add2', 3000),
                                   ('add2', 6000), ('sub', 600), ('mult', 600)):
                key = (config['model'], str(config['p']), str(seed), target,
                       str(budget))
                if key in direct_table:
                    continue
                direct_jobs.append(dict(kind='direct', variant=variant, seed=seed,
                                        target=target, budget=budget, payload=payload,
                                        device=device, args=args))
    if direct_jobs:
        print(f'{len(direct_jobs)} direct baselines', flush=True)

        context = multiprocessing.get_context('spawn')
        with context.Pool(processes=args.procs, initializer=worker_init,
                          initargs=(device, common.BATCH_LOG, args.threads)) as pool:
            for result in pool.imap_unordered(direct_worker, direct_jobs):
                job = result['job']
                key = (job['variant'], job['seed'], job['target'], job['budget'])
                if result['accuracy'] is None:
                    common.log_line(f'[q17] the direct baseline {key} failed: '
                                    f'{result["error"]}')
                    continue
                config = CONFIGS[job['variant']]
                row = dict(model=config['model'], p=config['p'], seed=job['seed'],
                           target=job['target'], budget=job['budget'],
                           acc=result['accuracy'])
                direct_table[(config['model'], str(config['p']), str(job['seed']),
                              job['target'], str(job['budget']))] = row
                append_rows(args.direct, DIRECT_FIELDS, [row])
                common.log_line(f'[q17] direct {config["model"]} p{config["p"]} '
                                f'seed {job["seed"]} {job["target"]} '
                                f'{job["budget"]}: {result["accuracy"]:.4f}')

    # the trajectories
    traj_rows = read_rows(args.out)
    done = collections.defaultdict(set)
    for row in traj_rows:
        done[(row['variant'], row['seed'], row['source'])].add(int(row['step']))
    fourier_table = {}
    for row in read_rows(args.fourier):
        fourier_table[f'{row["seed"]}|{row["step"]}|{row["model_part"]}|'
                      f'{row["edit"]}'] = row
    jobs = []
    for variant in args.variants:
        config = CONFIGS[variant]
        for seed in config['seeds']:
            for source in config['sources']:
                if variant == 'A' and source == 'shuffled' and seed not in SHUFFLED_SEEDS:
                    continue
                steps = schedule(config['horizon'])
                if steps[-1] in done[(variant, str(seed), source)]:
                    print(f'{variant} seed {seed} {source}: already done', flush=True)
                    continue
                jobs.append(dict(args=args, device=device, payload=payload,
                                 variant=variant, seed=seed, source=source,
                                 direct_table=list(direct_table.values()),
                                 traj_path=args.out, direct_path=args.direct,
                                 fourier_path=args.fourier,
                                 fourier_table=fourier_table))
    print(f'{len(jobs)} trajectories', flush=True)
    results = run_pool(jobs, device, args.procs, args.threads)
    failed = [result for result in results if result.get('error')]
    for result in failed:
        job = result['job']
        common.log_line(f'[q17] FAILED {job["variant"]} seed {job["seed"]} '
                        f'{job["source"]}: {result["error"]}')
    print(f'{len(results) - len(failed)} trajectories finished, {len(failed)} failed',
          flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
