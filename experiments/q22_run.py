"""Queue 22: the long-budget memorizers and the new target tasks.

The setup is the one of the config A of queue 17 and of queue 19: pythia-160m,
p = 59, coverage 0.5 (the `gen` source with the N = 1,740 generated pairs and
the `mem` source with the first 100 pairs of the same pool), lr 3e-4, wd 1.0,
the seeds 0 to 4, the warmup of 100 steps, the betas (0.9, 0.98) and the batch
128 of the stage-1 training.

Both experiments use the source checkpoints of the step 5,000:

    experiment 1   the `mem` source, the direct baseline from the pretrained
                   model and the `gen` source are transferred to the target
                   a+2b for 20,000 steps, and the held-out accuracy of the
                   transfer is measured every 500 steps;
    experiment 2   the `gen` source, the `mem` source and the direct baseline
                   are transferred to the new targets a+3b and a-2b for 3,000
                   steps, with the same evaluation every 500 steps.

The stage-2 protocol is the one of the existing runs: the first 500 pairs of
the train pool, the first 500 pairs of the held-out split for the score, the
batch 16, the warmup 0, the learning rate 3e-4 and the weight decay 1.0, and
the seed of the trajectory.

The sources are trained in the memory of the worker and are not written to the
disk: the checkpoints of the existing runs were temporary files of the work
directory that were deleted after every trajectory, and ten checkpoints of
pythia-160m would need 6.5 GB of the disk of the instance.

    python nseed_backbone/q22_run.py --experiment 2 --procs 4
    python nseed_backbone/q22_run.py --experiment 1 --procs 4

Produces: the long budget and the new targets of queue 22.
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
from p15_common import MODELS, REVISION, load_model, token_accuracy  # noqa: E402
from p17_common import fine_tune, reset_model, splits_for  # noqa: E402
from p17_traj import append_rows, param_norm, read_rows  # noqa: E402
from p22_targets import (BASE_TARGET, NEW_TARGETS, answer_ids_rule,  # noqa: E402
                          cpu_state, make_examples_rule, stage2_curve)

CONFIG = dict(model='160m', p=59, coverage=0.5, lr=3e-4, wd=1.0, horizon=5000)
SEEDS = (0, 1, 2, 3, 4)
SOURCES = ('gen', 'mem')
MEM_PAIRS = 100
CURVE_FIELDS = ['seed', 'condition', 'target', 'step', 'heldout_acc']
CHECK_FIELDS = ['seed', 'source', 'step', 'heldout_acc', 'param_norm']
EVAL_EVERY = 500
EXP1_STEPS = 20000
EXP2_STEPS = 3000
# the conditions of both experiments: the two sources and the direct baseline
CONDITIONS = {'gen': 'grokked', 'mem': 'memorizing', 'direct': 'direct'}


def source_pairs(splits, source):
    """The training pairs of a source: the N generated pairs or the first 100."""
    if source == 'mem':
        return splits['train_pool'][:MEM_PAIRS]
    return splits['train_pool'][:int(round(CONFIG['coverage'] * CONFIG['p'] ** 2))]


def train_source(seed, source, payload, device):
    """The source of one seed, trained to the step 5,000.

    The weights of the step 5,000 are kept on the CPU, so that every transfer
    of the source starts from the same weights.
    """
    p = CONFIG['p']
    splits = splits_for(p)
    pairs = source_pairs(splits, source)
    examples = make_examples_rule(payload['symbols'], pairs, p, 'add')
    heldout = splits['heldout']
    prompts_held = [f'{payload["symbols"][str(a)]} {payload["symbols"][str(b)]} ='
                    for a, b in heldout]
    ids_held = answer_ids_rule(payload, heldout, p, 'add')
    model, tokenizer = load_model(MODELS[CONFIG['model']], REVISION, device)
    state = {}

    def hook(step, _losses):
        if step != CONFIG['horizon']:
            return
        state['heldout_acc'] = token_accuracy(model, tokenizer, prompts_held,
                                              ids_held, device)
        state['param_norm'] = param_norm(model)
        state['weights'] = cpu_state(model)
        common.log_line(f'[q22] seed {seed} {source}: the source of the step '
                        f'{step} has a held-out accuracy of '
                        f'{state["heldout_acc"]:.4f} and a parameter norm of '
                        f'{state["param_norm"]:.4f}')

    fine_tune(model, tokenizer, examples, device, steps=CONFIG['horizon'],
              learning_rate=CONFIG['lr'], weight_decay=CONFIG['wd'],
              horizon=CONFIG['horizon'], decay=False, seed=seed * 7 + len(source),
              hook=hook, hook_every=100)
    return model, tokenizer, state


def run_curve(model, tokenizer, payload, splits, device, seed, condition, target,
              steps, out_path):
    """One stage-2 transfer; its curve is written as it comes."""
    curve = stage2_curve(model, tokenizer, payload, splits, CONFIG['p'], device,
                         CONFIG['lr'], CONFIG['wd'], target, seed, steps,
                         eval_every=EVAL_EVERY)
    rows = [dict(seed=seed, condition=condition, target=target, step=step,
                 heldout_acc=accuracy) for step, accuracy in curve]
    append_rows(out_path, CURVE_FIELDS, rows)
    for row in rows:
        common.log_line(f'[q22] seed {seed} {condition} {target} step '
                        f'{row["step"]}: {row["heldout_acc"]:.4f}')
    return rows


def run_seed(job):
    """One seed: its sources and their transfers."""
    args = job['args']
    device = job['device']
    payload = job['payload']
    seed = job['seed']
    p = CONFIG['p']
    splits = splits_for(p)
    targets = ([BASE_TARGET] if args.experiment == 1 else list(NEW_TARGETS))
    steps = EXP1_STEPS if args.experiment == 1 else EXP2_STEPS
    out_path = Path(args.out)
    check_path = Path(args.check)

    for source in SOURCES:
        model, tokenizer, state = train_source(seed, source, payload, device)
        if 'weights' not in state:
            common.log_line(f'[q22] seed {seed} {source}: the source has no step '
                            f'{CONFIG["horizon"]}, skipped')
            continue
        append_rows(check_path, CHECK_FIELDS,
                    [dict(seed=seed, source=source, step=CONFIG['horizon'],
                          heldout_acc=state['heldout_acc'],
                          param_norm=state['param_norm'])])
        for target in targets:
            model.load_state_dict(state['weights'])
            run_curve(model, tokenizer, payload, splits, device, seed,
                      CONDITIONS[source], target, steps, out_path)
        del model, tokenizer, state
        torch.cuda.empty_cache()

    # the direct baseline: the target trained straight from the pretrained model
    for target in targets:
        model, tokenizer = reset_model(MODELS[CONFIG['model']], device)
        run_curve(model, tokenizer, payload, splits, device, seed,
                  CONDITIONS['direct'], target, steps, out_path)
    del model, tokenizer
    torch.cuda.empty_cache()
    return True


def worker_init(device, batch_log, threads):
    common.setup(threads, device)


def worker(job):
    try:
        run_seed(job)
        return dict(job=job, error=None)
    except Exception as exc:  # noqa: BLE001 - one seed must not kill the pool
        common.log_line(f'[q22] seed {job["seed"]} FAILED {type(exc).__name__}: {exc}')
        common.log_line(traceback.format_exc())
        return dict(job=job, error=str(exc))


def run_pool(jobs, device, procs, threads=1):
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
    parser.add_argument('--experiment', type=int, choices=[1, 2], required=True)
    parser.add_argument('--out', default='runs_queue22/q22_curves.csv')
    parser.add_argument('--check', default='runs_queue22/q22_source_check.csv')
    parser.add_argument('--symbols', default='runs_nseed_queue20/symbols_p97.json')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    parser.add_argument('--procs', type=int, default=4)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    payload = json.loads(Path(args.symbols).read_text(encoding='utf-8'))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    done = {(row['seed'], row['condition'], row['target'])
            for row in read_rows(args.out)}
    jobs = []
    for seed in args.seeds:
        if any(key[0] == str(seed) for key in done):
            print(f'seed {seed}: already done', flush=True)
            continue
        jobs.append(dict(args=args, device=device, payload=payload, seed=seed))
    print(f'[q22] experiment {args.experiment}: {len(jobs)} seeds', flush=True)
    results = run_pool(jobs, device, args.procs, args.threads)
    failed = [result for result in results if result.get('error')]
    for result in failed:
        common.log_line(f'[q22] FAILED seed {result["job"]["seed"]}: '
                        f'{result["error"]}')
    print(f'[q22] experiment {args.experiment}: {len(results) - len(failed)} seeds '
          f'finished, {len(failed)} failed', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
