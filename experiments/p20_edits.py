"""Queue 20, step P1: the dose-matched frequency edits.

The config A (pythia-160m, p = 59, coverage 0.5, lr 3e-4, wd 1.0) is trained
with the `gen` source for the seeds 0 to 9, to the step 5,000. At the steps
4,000 and 5,000 one temporary checkpoint is written, the edits are measured on
it and it is deleted again.

A frequency is removed in part: the factor f removes the share f * f of the
energy of that frequency. The edits of a step are the full removal of the key
set, the same removal with the factors 0.5, 0.7071 and 0.8660, which remove the
25, 50 and 75 percent of the key energy, the removal of the key set with the
factors that match the energy of the ranks 6 to 10 in each part, and the full
removal of the ranks 6 to 10 and of the ranks 6 to 15.

After every edit the held-out accuracy of the source is measured once and the
stage-2 add2 transfer at the budget 600 is run with the learning rates 3e-4
and 1e-4.

    python nseed_backbone/p20_edits.py --procs 4

Produces: Figure 5 (the dose-matched edits of queue 20).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
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
from p19_common import energy_fraction, ranked_frequencies, spectrum_energy  # noqa: E402
from p20_common import removed_fraction, scale_frequencies  # noqa: E402

EDIT_FIELDS = ['model', 'p', 'seed', 'source', 'step', 'edit', 'target_energy_in',
               'target_energy_out', 'edit_energy_in', 'edit_energy_out', 'heldout_acc',
               'stage2_lr', 'stage2_acc']
CONFIG = dict(model='160m', p=59, coverage=0.5, lr=3e-4, wd=1.0, horizon=5000)
SEEDS = tuple(range(0, 10))
EDIT_STEPS = (4000, 5000)
STAGE2_LRS = (3e-4, 1e-4)
SOURCE = 'gen'
DOSE_FACTORS = (0.5, 0.7071, 0.8660, 1.0)
DOSE_NAMES = ('key_e25_both', 'key_e50_both', 'key_e75_both', 'key_e100_both')


def apply_dose_edit(model, payload, p, factors_in, factors_out):
    """Apply the edit to the symbol token rows of the two parts.

    The factors are the ones of the input embeddings and the ones of the
    unembedding; an empty dict leaves the part alone.
    """
    embed, unembed = embedding_tables(model)
    ids = [payload['token_ids'][str(residue)] for residue in range(p)]
    with torch.no_grad():
        if factors_in:
            rows = symbol_rows(embed, payload, p)
            embed[ids] = scale_frequencies(rows, factors_in).to(embed.dtype)
        if factors_out:
            rows = symbol_rows(unembed, payload, p)
            unembed[ids] = scale_frequencies(rows, factors_out).to(unembed.dtype)
    return model


def edit_plan(key, rank6_10, rank6_15, energy_in, energy_out, e_in, e_out):
    """The edits of one step.

    Every entry holds the name, the factors of the input embeddings, the
    factors of the unembedding, the target removed energy in the two parts and
    the factors that match the ranks 6 to 10.
    """
    f_in = math.sqrt(e_in / energy_in) if energy_in > 0 else 1.0
    f_out = math.sqrt(e_out / energy_out) if energy_out > 0 else 1.0
    capped_in = f_in > 1.0
    capped_out = f_out > 1.0
    f_in = min(1.0, f_in)
    f_out = min(1.0, f_out)
    plan = [('none', {}, {}, '', '')]
    for name, factor in zip(DOSE_NAMES, DOSE_FACTORS):
        plan.append((name, {k: factor for k in key}, {k: factor for k in key}, '', ''))
    plan.append(('key_match_both', {k: f_in for k in key}, {k: f_out for k in key},
                 e_in, e_out))
    plan.append(('key_match_in', {k: f_in for k in key}, {}, e_in, 0.0))
    plan.append(('key_match_out', {}, {k: f_out for k in key}, 0.0, e_out))
    plan.append(('rank6_10_both', {k: 1.0 for k in rank6_10},
                 {k: 1.0 for k in rank6_10}, '', ''))
    plan.append(('rank6_15_both', {k: 1.0 for k in rank6_15},
                 {k: 1.0 for k in rank6_15}, '', ''))
    factors = dict(f_in=f_in, f_out=f_out, capped_in=capped_in, capped_out=capped_out)
    return plan, factors


def run_combo(job):
    """One trajectory: the training with the edits at the two steps."""
    args = job['args']
    device = job['device']
    payload = job['payload']
    seed = job['seed']
    p = CONFIG['p']
    model_name = MODELS[CONFIG['model']]
    splits = splits_for(p)
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
        if step not in EDIT_STEPS:
            return
        embed, unembed = embedding_tables(model)
        energy_in = spectrum_energy(symbol_rows(embed, payload, p))
        energy_out = spectrum_energy(symbol_rows(unembed, payload, p))
        key = ranked_frequencies(energy_in, 5)
        rank6_10 = ranked_frequencies(energy_in, 5, skip=5)
        rank6_15 = ranked_frequencies(energy_in, 10, skip=5)
        e_key_in = energy_fraction(energy_in, key)
        e_key_out = energy_fraction(energy_out, key)
        checkpoint = work / f'edits_{seed}_step{step}.pt'
        if free_disk_gb('/') < 4.0:
            raise RuntimeError('the free disk fell below 4 GB, the step is stopped')
        save_checkpoint(checkpoint, model)
        try:
            full_in = {k: 1.0 for k in rank6_10}
            full_out = {k: 1.0 for k in rank6_10}
            probe, _probe_tokenizer = load_checkpoint(checkpoint, model_name, device)
            apply_dose_edit(probe, payload, p, full_in, full_out)
            embed, unembed = embedding_tables(probe)
            e_in = removed_fraction(energy_in, symbol_rows(embed, payload, p))
            e_out = removed_fraction(energy_out, symbol_rows(unembed, payload, p))
            del probe, _probe_tokenizer
            torch.cuda.empty_cache()
            plan, factors = edit_plan(key, rank6_10, rank6_15, e_key_in, e_key_out,
                                      e_in, e_out)
            common.log_line(f'[q20] seed {seed} step {step}: the key holds '
                            f'{e_key_in:.4f} in and {e_key_out:.4f} out, the ranks 6 '
                            f'to 10 remove {e_in:.4f} and {e_out:.4f}, the factors '
                            f'are {factors["f_in"]:.4f} and {factors["f_out"]:.4f}, '
                            f'the caps {factors["capped_in"]} and '
                            f'{factors["capped_out"]}')
            for name, factors_in, factors_out, target_in, target_out in plan:
                realized_in = realized_out = ''
                held = None
                for index, learning_rate in enumerate(STAGE2_LRS):
                    edited, edited_tokenizer = load_checkpoint(checkpoint, model_name,
                                                              device)
                    apply_dose_edit(edited, payload, p, factors_in, factors_out)
                    if index == 0:
                        embed, unembed = embedding_tables(edited)
                        realized_in = removed_fraction(
                            energy_in, symbol_rows(embed, payload, p))
                        realized_out = removed_fraction(
                            energy_out, symbol_rows(unembed, payload, p))
                        held = token_accuracy(edited, edited_tokenizer, prompts_held,
                                              ids_held, device)
                    accuracy = stage2_accuracy(edited, edited_tokenizer, payload,
                                               splits, p, device, learning_rate,
                                               CONFIG['wd'], 'add2', seed, 600)
                    del edited, edited_tokenizer
                    torch.cuda.empty_cache()
                    row = dict(model=CONFIG['model'], p=p, seed=seed, source=SOURCE,
                               step=step, edit=name, target_energy_in=target_in,
                               target_energy_out=target_out,
                               edit_energy_in=realized_in, edit_energy_out=realized_out,
                               heldout_acc=held, stage2_lr=learning_rate,
                               stage2_acc=accuracy)
                    rows.append(row)
                    append_rows(job['out_path'], EDIT_FIELDS, [row])
        finally:
            checkpoint.unlink(missing_ok=True)

    fine_tune(model, tokenizer, examples, device, steps=CONFIG['horizon'],
              learning_rate=CONFIG['lr'], weight_decay=CONFIG['wd'],
              horizon=CONFIG['horizon'], decay=False, seed=seed * 7 + len(SOURCE),
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
        common.log_line(f'[q20] seed {job["seed"]} FAILED {type(exc).__name__}: {exc}')
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
    parser.add_argument('--out', default='runs_nseed_queue20/m20_edits.csv')
    parser.add_argument('--symbols', default='runs_nseed_queue20/symbols_p97.json')
    parser.add_argument('--work', default='runs_nseed_queue20/work')
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
        done.add((row['seed'], row['source'], row['step']))

    jobs = []
    for seed in SEEDS:
        if (str(seed), SOURCE, str(CONFIG['horizon'])) in done:
            print(f'seed {seed}: already done', flush=True)
            continue
        jobs.append(dict(args=args, device=device, payload=payload, seed=seed,
                         out_path=args.out))
    print(f'{len(jobs)} trajectories', flush=True)
    results = run_pool(jobs, device, args.procs, args.threads)
    failed = [result for result in results if result.get('error')]
    for result in failed:
        job = result['job']
        common.log_line(f'[q20] FAILED seed {job["seed"]}: {result["error"]}')
    print(f'{len(results) - len(failed)} trajectories finished, {len(failed)} failed',
          flush=True)
    return 0


def run_smoke(args, device):
    """The spectrum, the edit plan and the partial removal on random rows."""
    rows = torch.randn(59, 16, dtype=torch.float64)
    energy = spectrum_energy(rows)
    key = ranked_frequencies(energy, 5)
    print(f'the smoke key: {key}', flush=True)
    before = float(energy[1:].sum())
    for factor in DOSE_FACTORS:
        edited = scale_frequencies(rows, {k: factor for k in key})
        after = float(spectrum_energy(edited).sum())
        removed = (before - after) / before
        print(f'  the factor {factor}: the removed share is {removed:.4f}, the '
              f'energy of the key is {energy_fraction(energy, key):.4f} and the '
              f'expected share is {factor ** 2 * energy_fraction(energy, key):.4f}',
              flush=True)
    payload = json.loads(Path(args.symbols).read_text(encoding='utf-8'))
    splits = splits_for(59)
    examples = make_examples(payload['symbols'], splits['train_pool'][:200], 59, 'add')
    model, tokenizer = load_model(MODELS['160m'], REVISION, device)
    embed, unembed = embedding_tables(model)
    energy_in = spectrum_energy(symbol_rows(embed, payload, 59))
    energy_out = spectrum_energy(symbol_rows(unembed, payload, 59))
    key = ranked_frequencies(energy_in, 5)
    plan, factors = edit_plan(key, ranked_frequencies(energy_in, 5, skip=5),
                              ranked_frequencies(energy_in, 10, skip=5),
                              energy_fraction(energy_in, key),
                              energy_fraction(energy_out, key),
                              energy_fraction(energy_in, ranked_frequencies(energy_in, 5, skip=5)),
                              energy_fraction(energy_out, ranked_frequencies(energy_in, 5, skip=5)))
    print(f'the smoke plan holds {len(plan)} edits: {[item[0] for item in plan]}',
          flush=True)
    print(f'the smoke factors: {factors}', flush=True)
    checkpoint = Path(args.work) / 'smoke.pt'
    save_checkpoint(checkpoint, model)
    edited, edited_tokenizer = load_checkpoint(checkpoint, MODELS['160m'], device)
    apply_dose_edit(edited, payload, 59, plan[5][1], plan[5][2])
    embed, unembed = embedding_tables(edited)
    print(f'the smoke realized energy: {removed_fraction(energy_in, symbol_rows(embed, payload, 59)):.4f} '
          f'in and {removed_fraction(energy_out, symbol_rows(unembed, payload, 59)):.4f} out',
          flush=True)
    checkpoint.unlink(missing_ok=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
