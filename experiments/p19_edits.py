"""Queue 19, step P1: the energy-matched controls of the frequency edits.

The config A (pythia-160m, p = 59, coverage 0.5, lr 3e-4, wd 1.0) is trained
with the `gen` source for the seeds 0 to 9 and the config C (pythia-410m,
p = 59, coverage 0.5, lr 3e-4, wd 1.0) for the seeds 0 to 4, both to the step
5,000. At the steps 4,000 and 5,000 one temporary checkpoint is written, the
edits are measured on it and it is deleted again.

The edits of a step are the three of queue 18 (`none`, `key_both`, `key_in`,
`key_out`), the removal of the frequencies of the rank 6 to 10 of the input
spectrum from both parts, and three random sets of five frequencies whose
removed share of the energy is close to the share of the key set in the input
embeddings and in the unembedding at the same time. Every set is measured
three times: removed from both parts, from the input embeddings only and from
the unembedding only.

After every edit the held-out accuracy of the source is measured once and the
stage-2 add2 transfer at the budget 600 is run with the learning rates 3e-4
and 1e-4.

    python nseed_backbone/p19_edits.py --variants A --procs 4

Produces: Figure 5 (the frequency edits of queue 19).
"""
from __future__ import annotations

import argparse
import collections
import csv
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
from p17_common import (answer_ids, fine_tune, load_checkpoint, make_examples,  # noqa: E402
                         save_checkpoint, splits_for, stage2_accuracy)
from p17_traj import append_rows, embedding_tables, read_rows, symbol_rows  # noqa: E402
from p18_common import apply_edit  # noqa: E402
from p19_common import edit_plan, spectrum_energy  # noqa: E402

EDIT_FIELDS = ['model', 'p', 'seed', 'source', 'step', 'edit', 'tolerance',
               'edit_energy_in', 'edit_energy_out', 'heldout_acc', 'stage2_lr',
               'stage2_acc']
CONFIGS = {
    'A': dict(model='160m', p=59, coverage=0.5, lr=3e-4, wd=1.0, horizon=5000,
              seeds=range(0, 10)),
    'C': dict(model='410m', p=59, coverage=0.5, lr=3e-4, wd=1.0, horizon=5000,
              seeds=range(0, 5)),
}
EDIT_STEPS = (4000, 5000)
STAGE2_LRS = (3e-4, 1e-4)
SOURCE = 'gen'


def run_combo(job):
    """One trajectory: the training with the edits at the two steps."""
    args = job['args']
    device = job['device']
    payload = job['payload']
    variant = job['variant']
    seed = job['seed']
    config = CONFIGS[variant]
    model_name = MODELS[config['model']]
    p = config['p']
    splits = splits_for(p)
    pairs = splits['train_pool'][:int(round(config['coverage'] * p * p))]
    examples = make_examples(payload['symbols'], pairs, p, 'add')
    heldout = splits['heldout']
    prompts_held = [f'{payload["symbols"][str(a)]} {payload["symbols"][str(b)]} ='
                    for a, b in heldout]
    ids_held = answer_ids(payload, heldout, p, 'add')

    model, tokenizer = load_model(model_name, REVISION, device)
    work = Path(args.work)
    rows = []

    def hook(step, _losses):
        if step not in EDIT_STEPS:
            return
        embed, unembed = embedding_tables(model)
        energy_in = spectrum_energy(symbol_rows(embed, payload, p))
        energy_out = spectrum_energy(symbol_rows(unembed, payload, p))
        plan = edit_plan(energy_in, energy_out, p, seed, step)
        checkpoint = work / f'edits_{variant}_seed{seed}_step{step}.pt'
        if free_disk_gb('/') < 4.0:
            raise RuntimeError('the free disk fell below 4 GB, the step is stopped')
        save_checkpoint(checkpoint, model)
        try:
            for name, target, frequencies, tolerance, fraction_in, fraction_out in plan:
                held = None
                for index, learning_rate in enumerate(STAGE2_LRS):
                    edited, edited_tokenizer = load_checkpoint(checkpoint, model_name,
                                                              device)
                    if frequencies:
                        apply_edit(edited, payload, p, frequencies, target)
                    if index == 0:
                        held = token_accuracy(edited, edited_tokenizer, prompts_held,
                                              ids_held, device)
                    accuracy = stage2_accuracy(edited, edited_tokenizer, payload,
                                               splits, p, device, learning_rate,
                                               config['wd'], 'add2', seed, 600)
                    del edited, edited_tokenizer
                    torch.cuda.empty_cache()
                    row = dict(model=config['model'], p=p, seed=seed, source=SOURCE,
                               step=step, edit=name, tolerance=tolerance,
                               edit_energy_in=fraction_in, edit_energy_out=fraction_out,
                               heldout_acc=held, stage2_lr=learning_rate,
                               stage2_acc=accuracy)
                    rows.append(row)
                    append_rows(job['out_path'], EDIT_FIELDS, [row])
        finally:
            checkpoint.unlink(missing_ok=True)

    fine_tune(model, tokenizer, examples, device, steps=config['horizon'],
              learning_rate=config['lr'], weight_decay=config['wd'],
              horizon=config['horizon'], decay=False, seed=seed * 7 + len(SOURCE),
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
        common.log_line(f'[q19] {job["variant"]} seed {job["seed"]} FAILED '
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='runs_nseed_queue19/m19_edits.csv')
    parser.add_argument('--symbols', default='runs_nseed_queue19/symbols_p97.json')
    parser.add_argument('--work', default='runs_nseed_queue19/work')
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

    done = set()
    for row in read_rows(args.out):
        done.add((row['model'], row['seed'], row['source'], row['step']))

    jobs = []
    for variant in args.variants:
        config = CONFIGS[variant]
        for seed in config['seeds']:
            if (config['model'], str(seed), SOURCE, str(config['horizon'])) in done:
                print(f'{variant} seed {seed}: already done', flush=True)
                continue
            jobs.append(dict(args=args, device=device, payload=payload,
                             variant=variant, seed=seed, out_path=args.out))
    print(f'{len(jobs)} trajectories', flush=True)
    results = run_pool(jobs, device, args.procs, args.threads)
    failed = [result for result in results if result.get('error')]
    for result in failed:
        job = result['job']
        common.log_line(f'[q19] FAILED {job["variant"]} seed {job["seed"]}: '
                        f'{result["error"]}')
    print(f'{len(results) - len(failed)} trajectories finished, {len(failed)} failed',
          flush=True)
    return 0


def run_smoke(args, device):
    """The spectrum, the edit plan and one edit on a model of 120 steps."""
    payload = json.loads(Path(args.symbols).read_text(encoding='utf-8'))
    splits = splits_for(59)
    examples = make_examples(payload['symbols'], splits['train_pool'][:200], 59, 'add')
    model, tokenizer = load_model(MODELS['160m'], REVISION, device)
    embed, unembed = embedding_tables(model)
    energy_in = spectrum_energy(symbol_rows(embed, payload, 59))
    energy_out = spectrum_energy(symbol_rows(unembed, payload, 59))
    plan = edit_plan(energy_in, energy_out, 59, 0, 4000)
    print(f'the smoke plan holds {len(plan)} edits: {[item[0] for item in plan]}',
          flush=True)
    for name, target, frequencies, tolerance, fraction_in, fraction_out in plan:
        print(f'  {name}: the target {target}, the tolerance {tolerance!r}, '
              f'the energy {fraction_in:.4f} and {fraction_out:.4f}, '
              f'the frequencies {frequencies}', flush=True)
    checkpoint = Path(args.work) / 'smoke.pt'
    save_checkpoint(checkpoint, model)
    edited, edited_tokenizer = load_checkpoint(checkpoint, MODELS['160m'], device)
    apply_edit(edited, payload, 59, plan[1][2], plan[1][1])
    held = token_accuracy(edited, edited_tokenizer,
                          [f'{payload["symbols"][str(a)]} '
                           f'{payload["symbols"][str(b)]} ='
                           for a, b in splits['heldout'][:100]],
                          answer_ids(payload, splits['heldout'][:100], 59, 'add'),
                          device)
    print(f'the smoke edited held-out accuracy: {held:.4f}', flush=True)
    checkpoint.unlink(missing_ok=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
