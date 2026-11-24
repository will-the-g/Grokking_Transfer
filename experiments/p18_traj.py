"""Queue 18, step P1: the scale-up of the trajectories.

Two configurations are trained from scratch and the stage-2 transfers are run
from a temporary checkpoint at every step of the checkpoint schedule S:

    C   pythia-410m, p = 59,  coverage 0.5 (N = 1,740), lr 3e-4, wd 1.0,
        seeds 0 to 4, horizon 5,500
    D   pythia-160m, p = 97,  coverage 0.5 (N = 4,704), lr 3e-4, wd 1.0,
        seeds 0 to 4, horizon 5,500

The sources are `gen` (the N generated pairs), `mem` (the first 100 pairs of the
same pool, the same number of steps) and, for the seeds 0 to 2, `shuffled` (the
answers of the N pairs are shuffled, so nothing can be learned).

The stage-2 transfers of a `gen` or a `mem` trajectory are add2 at the budget
600 at every step of S, add2 at the budget 3,000 at the steps 3,000, 4,000,
5,000 and the horizon, and sub and mult at the budget 600 at every 500 steps
from 2,500. A `shuffled` trajectory gets the base transfer, add2 at the budget
600 at every step of S, and nothing else.

The direct baselines are the stage-2 tasks trained straight from the pretrained
model: add2 at the budgets 600, 1,600, 3,000 and 6,000 and sub and mult at the
budget 600. The baselines of the 160m p = 97 configuration are the ones of
queue 17: the protocol, the splits and the seed are the same, so they are
copied instead of being measured again.

    python nseed_backbone/p18_traj.py --variants D --procs 4

Produces: Figure 4 (the Pythia trajectories).
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
from p15_common import MODELS, REVISION, clear_model_cache, load_model  # noqa: E402
from p15_common import free_disk_gb, token_accuracy  # noqa: E402
from p17_common import (answer_ids, direct_baseline, fine_tune, load_checkpoint,  # noqa: E402
                         make_examples, save_checkpoint, splits_for,
                         stage2_accuracy)
from p17_traj import (append_rows, best_direct, param_norm, read_rows,  # noqa: E402
                      schedule)
from p18_common import spectrum_nodc  # noqa: E402,F401

TRAJ_FIELDS = ['model', 'p', 'variant', 'seed', 'source', 'step', 'train_acc',
               'heldout_acc', 'train_loss', 'param_norm', 'stage2_target',
               'stage2_budget', 'stage2_acc', 'gain_over_best_direct']
DIRECT_FIELDS = ['model', 'p', 'seed', 'target', 'budget', 'acc']
CONFIGS = {
    'C': dict(model='410m', p=59, coverage=0.5, lr=3e-4, wd=1.0, horizon=5500,
              seeds=range(0, 5), sources=('gen', 'mem', 'shuffled')),
    'D': dict(model='160m', p=97, coverage=0.5, lr=3e-4, wd=1.0, horizon=5500,
              seeds=range(0, 5), sources=('gen', 'mem', 'shuffled')),
}
SHUFFLED_SEEDS = (0, 1, 2)
LONG_BUDGET_STEPS = (3000, 4000, 5000)


def stage2_plan(variant, seed, source, steps, horizon):
    """The (step, target, budget) transfers of one trajectory."""
    plan = [(step, 'add2', 600) for step in steps]
    if source not in ('gen', 'mem'):
        return plan
    for step in list(LONG_BUDGET_STEPS) + [horizon]:
        if step in steps:
            plan.append((step, 'add2', 3000))
    for step in range(2500, horizon + 1, 500):
        if step in steps:
            plan.append((step, 'sub', 600))
            plan.append((step, 'mult', 600))
    return plan


def pairs_for(splits, p, config, source):
    """The training pairs of a source."""
    n_pairs = int(round(config['coverage'] * p * p))
    if source == 'mem':
        return splits['train_pool'][:100]
    return splits['train_pool'][:n_pairs]


def run_combo(job):
    """One trajectory: the stage-1 training with the stage-2 transfers."""
    args = job['args']
    device = job['device']
    payload = job['payload']
    variant = job['variant']
    seed = job['seed']
    source = job['source']
    config = CONFIGS[variant]
    model_name = MODELS[config['model']]
    p = config['p']
    horizon = config['horizon']
    splits = splits_for(p)
    pairs = pairs_for(splits, p, config, source)
    rule = 'shuffled' if source == 'shuffled' else 'add'
    examples = make_examples(payload['symbols'], pairs, p, rule)
    prompts_train = [text for text, _answer in examples]
    ids_train = answer_ids(payload, pairs, p, 'add')
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
        loss = sum(window) / len(window) if window else ''
        norm = param_norm(model)
        for target, budget in plan_by_step.get(step, []):
            checkpoint = work / f'{variant}_{source}_seed{seed}_step{step}.pt'
            if free_disk_gb('/') < 4.0:
                raise RuntimeError('the free disk fell below 4 GB, the step is stopped')
            try:
                save_checkpoint(checkpoint, model)
                forked, forked_tokenizer = load_checkpoint(checkpoint, model_name, device)
                accuracy = stage2_accuracy(forked, forked_tokenizer, payload, splits, p,
                                           device, config['lr'], config['wd'], target,
                                           seed, budget)
                del forked, forked_tokenizer
                torch.cuda.empty_cache()
            finally:
                checkpoint.unlink(missing_ok=True)
            gain = accuracy - best_direct(job['direct_table'], target, budget,
                                          config['model'], p, seed)
            row = dict(model=config['model'], p=p, variant=variant, seed=seed,
                       source=source, step=step, train_acc=train_acc,
                       heldout_acc=heldout_acc, train_loss=loss,
                       param_norm=norm, stage2_target=target,
                       stage2_budget=budget, stage2_acc=accuracy,
                       gain_over_best_direct=gain)
            rows.append(row)
            append_rows(job['traj_path'], TRAJ_FIELDS, [row])

    # the step 0 of the schedule is the pretrained model
    hook(0, [])
    fine_tune(model, tokenizer, examples, device, steps=horizon,
              learning_rate=config['lr'], weight_decay=config['wd'],
              horizon=horizon, decay=False, seed=seed * 7 + len(source),
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
        common.log_line(f'[q18] {job["variant"]} seed {job["seed"]} '
                        f'{job["source"]} FAILED {type(exc).__name__}: {exc}')
        common.log_line(traceback.format_exc())
        return dict(job=job, rows=[], error=str(exc))


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


def seed_direct_rows(path, variants):
    """The direct baselines of queue 17 that belong to the configurations here."""
    rows = []
    for row in read_rows(path):
        model = row.get('model')
        p = row.get('p')
        for variant in variants:
            config = CONFIGS[variant]
            if model == config['model'] and str(p) == str(config['p']):
                rows.append(dict(model=model, p=p, seed=row['seed'],
                                 target=row['target'], budget=row['budget'],
                                 acc=row['acc']))
                break
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='runs_nseed_queue18/p18_traj.csv')
    parser.add_argument('--direct', default='runs_nseed_queue18/p18_direct.csv')
    parser.add_argument('--direct-seed',
                        default='runs_nseed_queue18/p17_direct_seed.csv')
    parser.add_argument('--symbols', default='runs_nseed_queue18/symbols_p97.json')
    parser.add_argument('--work', default='runs_nseed_queue18/work')
    parser.add_argument('--variants', nargs='+', default=list(CONFIGS))
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

    direct_table = {}
    for row in read_rows(args.direct):
        direct_table[(row['model'], row['p'], row['seed'], row['target'],
                      row['budget'])] = row

    # the baselines of queue 17 that belong to these configurations
    if Path(args.direct_seed).exists():
        seeded = seed_direct_rows(args.direct_seed, args.variants)
        fresh = [row for row in seeded
                 if (row['model'], row['p'], row['seed'], row['target'],
                     row['budget']) not in direct_table]
        if fresh:
            append_rows(args.direct, DIRECT_FIELDS, fresh)
            for row in fresh:
                direct_table[(row['model'], row['p'], row['seed'], row['target'],
                              row['budget'])] = row
            common.log_line(f'[q18] {len(fresh)} direct baselines of queue 17 reused')

    # the direct baselines that are missing
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
                if result['accuracy'] is None:
                    common.log_line(f'[q18] the direct baseline {job["variant"]} '
                                    f'{job["seed"]} {job["target"]} '
                                    f'{job["budget"]} failed: {result["error"]}')
                    continue
                config = CONFIGS[job['variant']]
                row = dict(model=config['model'], p=config['p'], seed=job['seed'],
                           target=job['target'], budget=job['budget'],
                           acc=result['accuracy'])
                direct_table[(config['model'], str(config['p']), str(job['seed']),
                              job['target'], str(job['budget']))] = row
                append_rows(args.direct, DIRECT_FIELDS, [row])
                common.log_line(f'[q18] direct {config["model"]} p{config["p"]} '
                                f'seed {job["seed"]} {job["target"]} '
                                f'{job["budget"]}: {result["accuracy"]:.4f}')

    # the trajectories
    done = collections.defaultdict(set)
    for row in read_rows(args.out):
        done[(row['variant'], row['seed'], row['source'])].add(int(row['step']))
    jobs = []
    for variant in args.variants:
        config = CONFIGS[variant]
        for seed in config['seeds']:
            for source in config['sources']:
                if source == 'shuffled' and seed not in SHUFFLED_SEEDS:
                    continue
                steps = schedule(config['horizon'])
                if steps[-1] in done[(variant, str(seed), source)]:
                    print(f'{variant} seed {seed} {source}: already done', flush=True)
                    continue
                jobs.append(dict(args=args, device=device, payload=payload,
                                 variant=variant, seed=seed, source=source,
                                 direct_table=list(direct_table.values()),
                                 traj_path=args.out, direct_path=args.direct))
    print(f'{len(jobs)} trajectories', flush=True)
    results = run_pool(jobs, device, args.procs, args.threads)
    failed = [result for result in results if result.get('error')]
    for result in failed:
        job = result['job']
        common.log_line(f'[q18] FAILED {job["variant"]} seed {job["seed"]} '
                        f'{job["source"]}: {result["error"]}')
    print(f'{len(results) - len(failed)} trajectories finished, {len(failed)} failed',
          flush=True)
    return 0


def run_smoke(args, device):
    """One trajectory of 120 steps with the checkpoint and the stage 2."""
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
        checkpoint.unlink(missing_ok=True)
        print(f'the smoke stage 2 accuracy: {accuracy:.4f}', flush=True)

    fine_tune(model, tokenizer, examples, device, steps=120, learning_rate=3e-4,
              weight_decay=1.0, horizon=6000, hook=hook, hook_every=100)
    plan = stage2_plan('C', 0, 'gen', schedule(5500), 5500)
    print(f'the smoke plan holds {len(plan)} transfers', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
