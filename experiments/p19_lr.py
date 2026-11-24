"""Queue 19, step P2: the stage-2 learning-rate sweep.

The config A (pythia-160m, p = 59, coverage 0.5, lr 3e-4, wd 1.0) is trained
with the `gen` source and with the `mem` source (100 pairs), the seeds 0 to 4,
to the horizon of 8,000 steps. At the steps 4,000, 5,000, 6,000, 7,000 and
8,000 one temporary checkpoint is written, the stage-2 add2 transfer at the
budget 600 is run with the learning rates 3e-5, 1e-4, 3e-4 and 1e-3 and the
checkpoint is deleted again. The held-out accuracy of the source and the
global parameter norm are recorded at every step.

The direct baselines are the stage-2 tasks trained straight from the pretrained
model, at the budgets 600 and 3,000, with the same four learning rates.

    python nseed_backbone/p19_lr.py --procs 4

Produces: Figure 6 (the stage-2 learning rates).
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing
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

import common  # noqa: E402
from p15_common import MODELS, REVISION, free_disk_gb, load_model, token_accuracy  # noqa: E402
from p17_common import (answer_ids, direct_baseline, fine_tune, load_checkpoint,  # noqa: E402
                         make_examples, save_checkpoint, splits_for,
                         stage2_accuracy)
from p17_traj import append_rows, param_norm, read_rows  # noqa: E402

LR_FIELDS = ['model', 'p', 'seed', 'source', 'step', 'heldout_acc', 'param_norm',
             'stage2_lr', 'stage2_acc']
DIRECT_FIELDS = ['model', 'p', 'seed', 'stage2_lr', 'budget', 'acc']
CONFIG = dict(model='160m', p=59, coverage=0.5, lr=3e-4, wd=1.0, horizon=8000)
SEEDS = tuple(range(0, 5))
SOURCES = ('gen', 'mem')
CHECK_STEPS = (4000, 5000, 6000, 7000, 8000)
STAGE2_LRS = (3e-5, 1e-4, 3e-4, 1e-3)
DIRECT_BUDGETS = (600, 3000)


def run_combo(job):
    """One trajectory: the training with the four learning rates at every step."""
    args = job['args']
    device = job['device']
    payload = job['payload']
    seed = job['seed']
    source = job['source']
    model_name = MODELS[CONFIG['model']]
    p = CONFIG['p']
    splits = splits_for(p)
    if source == 'mem':
        pairs = splits['train_pool'][:100]
    else:
        pairs = splits['train_pool'][:int(round(CONFIG['coverage'] * p * p))]
    examples = make_examples(payload['symbols'], pairs, p, 'add')
    heldout = splits['heldout']
    prompts_held = [f'{payload["symbols"][str(a)]} {payload["symbols"][str(b)]} ='
                    for a, b in heldout]
    ids_held = answer_ids(payload, heldout, p, 'add')

    model, tokenizer = load_model(model_name, REVISION, device)
    work = Path(args.work)
    rows = []

    def hook(step, _losses):
        if step not in CHECK_STEPS:
            return
        heldout_acc = token_accuracy(model, tokenizer, prompts_held, ids_held, device)
        norm = param_norm(model)
        checkpoint = work / f'lr_{source}_seed{seed}_step{step}.pt'
        if free_disk_gb('/') < 4.0:
            raise RuntimeError('the free disk fell below 4 GB, the step is stopped')
        save_checkpoint(checkpoint, model)
        try:
            for learning_rate in STAGE2_LRS:
                forked, forked_tokenizer = load_checkpoint(checkpoint, model_name,
                                                          device)
                accuracy = stage2_accuracy(forked, forked_tokenizer, payload, splits,
                                           p, device, learning_rate, CONFIG['wd'],
                                           'add2', seed, 600)
                del forked, forked_tokenizer
                torch.cuda.empty_cache()
                row = dict(model=CONFIG['model'], p=p, seed=seed, source=source,
                           step=step, heldout_acc=heldout_acc, param_norm=norm,
                           stage2_lr=learning_rate, stage2_acc=accuracy)
                rows.append(row)
                append_rows(job['out_path'], LR_FIELDS, [row])
        finally:
            checkpoint.unlink(missing_ok=True)

    fine_tune(model, tokenizer, examples, device, steps=CONFIG['horizon'],
              learning_rate=CONFIG['lr'], weight_decay=CONFIG['wd'],
              horizon=CONFIG['horizon'], decay=False, seed=seed * 7 + len(source),
              hook=hook, hook_every=100)
    del model
    torch.cuda.empty_cache()
    return rows


def worker_init(device, batch_log, threads):
    common.setup(threads, device)


def worker(job):
    try:
        rows = run_combo(job)
        return dict(job=job, rows=rows, error=None)
    except Exception as exc:  # noqa: BLE001 - one trajectory must not kill the pool
        common.log_line(f'[q19] lr seed {job["seed"]} {job["source"]} FAILED '
                        f'{type(exc).__name__}: {exc}')
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


def direct_worker(job):
    """One direct baseline: the stage-2 task from the pretrained model."""
    try:
        config = job['config']
        splits = splits_for(config['p'])
        accuracy = direct_baseline(MODELS[config['model']], job['payload'], splits,
                                   config['p'], job['device'], job['stage2_lr'],
                                   config['wd'], job['seed'], 'add2', job['budget'])
        return dict(job=job, accuracy=accuracy, error=None)
    except Exception as exc:  # noqa: BLE001 - one baseline must not kill the pool
        return dict(job=job, accuracy=None, error=str(exc))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='runs_nseed_queue19/p19_lr.csv')
    parser.add_argument('--direct', default='runs_nseed_queue19/p19_direct_lr.csv')
    parser.add_argument('--symbols', default='runs_nseed_queue19/symbols_p97.json')
    parser.add_argument('--work', default='runs_nseed_queue19/work')
    parser.add_argument('--procs', type=int, default=4)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.smoke:
        return run_smoke(args, device)
    payload = json.loads(Path(args.symbols).read_text(encoding='utf-8'))
    Path(args.work).mkdir(parents=True, exist_ok=True)

    # the direct baselines
    direct_table = {}
    for row in read_rows(args.direct):
        direct_table[(row['seed'], row['stage2_lr'], row['budget'])] = row
    direct_jobs = []
    for seed in SEEDS:
        for learning_rate in STAGE2_LRS:
            for budget in DIRECT_BUDGETS:
                key = (str(seed), str(learning_rate), str(budget))
                if key in direct_table:
                    continue
                direct_jobs.append(dict(config=CONFIG, seed=seed, stage2_lr=learning_rate,
                                        budget=budget, payload=payload, device=device))
    if direct_jobs:
        print(f'{len(direct_jobs)} direct baselines', flush=True)
        context = multiprocessing.get_context('spawn')
        with context.Pool(processes=args.procs, initializer=worker_init,
                          initargs=(device, common.BATCH_LOG, args.threads)) as pool:
            for result in pool.imap_unordered(direct_worker, direct_jobs):
                job = result['job']
                if result['accuracy'] is None:
                    common.log_line(f'[q19] the direct baseline seed {job["seed"]} '
                                    f'{job["stage2_lr"]} {job["budget"]} failed: '
                                    f'{result["error"]}')
                    continue
                row = dict(model=CONFIG['model'], p=CONFIG['p'], seed=job['seed'],
                           stage2_lr=job['stage2_lr'], budget=job['budget'],
                           acc=result['accuracy'])
                direct_table[(str(job['seed']), str(job['stage2_lr']),
                              str(job['budget']))] = row
                append_rows(args.direct, DIRECT_FIELDS, [row])
                common.log_line(f'[q19] direct seed {job["seed"]} '
                                f'{job["stage2_lr"]} {job["budget"]}: '
                                f'{result["accuracy"]:.4f}')

    # the trajectories
    done = collections.defaultdict(set)
    for row in read_rows(args.out):
        done[(row['seed'], row['source'])].add(int(row['step']))
    jobs = []
    for seed in SEEDS:
        for source in SOURCES:
            if str(CONFIG['horizon']) in done[(str(seed), source)]:
                print(f'seed {seed} {source}: already done', flush=True)
                continue
            jobs.append(dict(args=args, device=device, payload=payload, seed=seed,
                             source=source, out_path=args.out))
    print(f'{len(jobs)} trajectories', flush=True)
    results = run_pool(jobs, device, args.procs, args.threads)
    failed = [result for result in results if result.get('error')]
    for result in failed:
        job = result['job']
        common.log_line(f'[q19] FAILED lr seed {job["seed"]} {job["source"]}: '
                        f'{result["error"]}')
    print(f'{len(results) - len(failed)} trajectories finished, {len(failed)} failed',
          flush=True)
    return 0


def run_smoke(args, device):
    """The stage-2 transfer with a learning rate of its own."""
    payload = json.loads(Path(args.symbols).read_text(encoding='utf-8'))
    splits = splits_for(59)
    examples = make_examples(payload['symbols'], splits['train_pool'][:200], 59, 'add')
    model, tokenizer = load_model(MODELS['160m'], REVISION, device)
    checkpoint = Path(args.work) / 'smoke.pt'
    save_checkpoint(checkpoint, model)
    forked, forked_tokenizer = load_checkpoint(checkpoint, MODELS['160m'], device)
    accuracy = stage2_accuracy(forked, forked_tokenizer, payload, splits, 59, device,
                               3e-5, 1.0, 'add2', 0, 20)
    checkpoint.unlink(missing_ok=True)
    print(f'the smoke stage 2 accuracy at the learning rate 3e-5: {accuracy:.4f}',
          flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
