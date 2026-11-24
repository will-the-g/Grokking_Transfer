"""Queue 21, step P1: the energy-matched non-key deletions.

The config A of queue 20 (pythia-160m, p = 59, coverage 0.5, lr 3e-4, wd 1.0)
is trained with the `gen` source for the seeds 0 to 9 to the step 5,000. At the
step 5,000 the spectrum of the symbol rows is measured and the key set (the
top 5 frequencies) is taken out of it.

On top of the edits of queue 20 this script adds three new deletions: the top
15, 20 and 25 NON-key frequencies, that is the ranks 6 to 20, 6 to 25 and 6 to
30 of the spectrum, exactly the procedure of `rank6_10_both` (the ranks 6 to
10) and `rank6_15_both` (the ranks 6 to 15) of queue 20. Every frequency is
removed in full, in the input embeddings and in the unembedding at the same
time.

For every new edit the share of the non-zero energy that the edit removes is
measured in the two parts, the held-out accuracy of the source is measured
once and the stage-2 transfer to the target a+2b is run with the learning rate
3e-4 of queue 20 and the budget of 600 steps.

Only the dev split is read; the audit split is never touched.

The existing edits of queue 20 are not recomputed: the rows of this script go
into their own file, `m21_energy.csv`, with the same fields as `m20_edits.csv`.

    python nseed_backbone/p21_energy.py --procs 4

Produces: Figure 5 (the non-key deletions of queue 21).
"""
from __future__ import annotations

import argparse
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
from p15_common import (MODELS, REVISION, free_disk_gb, load_model,  # noqa: E402
                        token_accuracy)
from p17_common import (answer_ids, fine_tune, make_examples,  # noqa: E402
                         splits_for, stage2_accuracy)
from p17_traj import append_rows, embedding_tables, read_rows, symbol_rows  # noqa: E402
from p19_common import ranked_frequencies, spectrum_energy  # noqa: E402
from p20_common import removed_fraction  # noqa: E402
from p20_edits import EDIT_FIELDS, apply_dose_edit  # noqa: E402

CONFIG = dict(model='160m', p=59, coverage=0.5, lr=3e-4, wd=1.0, horizon=5000)
SEEDS = tuple(range(0, 10))
SOURCE = 'gen'
STAGE2_LR = 3e-4
STAGE2_STEPS = 600
KEY_SIZE = 5
DISK_FLOOR_GB = 10.0

# the new edits: the count of non-key frequencies and the name of the edit.
# The names follow the ones of queue 20: the name holds the last rank, so the
# top 15 non-key frequencies are the ranks 6 to 20.
NEW_EDITS = ((15, 'rank6_20_both'), (20, 'rank6_25_both'), (25, 'rank6_30_both'))
CHECK_FIELDS = ['seed', 'step', 'heldout_acc']


def run_seed(job):
    """One trajectory: the training to the step 5,000 and the three new edits."""
    args = job['args']
    device = job['device']
    payload = job['payload']
    seed = job['seed']
    p = CONFIG['p']
    horizon = args.horizon
    model_name = MODELS[CONFIG['model']]
    splits = splits_for(p)
    pairs = splits['train_pool'][:int(round(CONFIG['coverage'] * p * p))]
    examples = make_examples(payload['symbols'], pairs, p, 'add')
    heldout = splits['heldout']
    prompts_held = [f'{payload["symbols"][str(a)]} {payload["symbols"][str(b)]} ='
                    for a, b in heldout]
    ids_held = answer_ids(payload, heldout, p, 'add')

    model, tokenizer = load_model(model_name, REVISION, device)
    rows = []

    def hook(step, _losses):
        if step != horizon:
            return
        if free_disk_gb('/') < DISK_FLOOR_GB:
            raise RuntimeError(f'the free disk fell below {DISK_FLOOR_GB} GB, '
                               'the step is stopped')
        # the pristine weights of the step 5,000, so that every edit starts from
        # the same source. They are kept on the CPU: the queue 20 checkpoints
        # were written to the disk and deleted again, and the disk is short.
        pristine = {key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()}
        embed, unembed = embedding_tables(model)
        energy_in = spectrum_energy(symbol_rows(embed, payload, p))
        energy_out = spectrum_energy(symbol_rows(unembed, payload, p))
        key = ranked_frequencies(energy_in, KEY_SIZE)
        # the held-out accuracy of the unedited source: the check that the
        # source of this run is the source of queue 20
        held_source = token_accuracy(model, tokenizer, prompts_held, ids_held, device)
        append_rows(job['check_path'], CHECK_FIELDS,
                    [dict(seed=seed, step=step, heldout_acc=held_source)])
        common.log_line(f'[q21] seed {seed} step {step}: the unedited source has a '
                        f'held-out accuracy of {held_source}')
        for count, name in NEW_EDITS:
            frequencies = ranked_frequencies(energy_in, count, skip=KEY_SIZE)
            model.load_state_dict(pristine)
            factors = {frequency: 1.0 for frequency in frequencies}
            apply_dose_edit(model, payload, p, factors, factors)
            embed, unembed = embedding_tables(model)
            e_in = removed_fraction(energy_in, symbol_rows(embed, payload, p))
            e_out = removed_fraction(energy_out, symbol_rows(unembed, payload, p))
            held = token_accuracy(model, tokenizer, prompts_held, ids_held, device)
            accuracy = stage2_accuracy(model, tokenizer, payload, splits, p, device,
                                       STAGE2_LR, CONFIG['wd'], 'add2', seed,
                                       STAGE2_STEPS)
            row = dict(model=CONFIG['model'], p=p, seed=seed, source=SOURCE,
                       step=step, edit=name, target_energy_in='', target_energy_out='',
                       edit_energy_in=e_in, edit_energy_out=e_out,
                       heldout_acc=held, stage2_lr=STAGE2_LR, stage2_acc=accuracy)
            rows.append(row)
            append_rows(job['out_path'], EDIT_FIELDS, [row])
            common.log_line(f'[q21] seed {seed} {name}: the ranks 6 to '
                            f'{KEY_SIZE + count} remove {e_in:.4f} in and '
                            f'{e_out:.4f} out, the held-out is {held}, the transfer '
                            f'is {accuracy}')
        del pristine
        torch.cuda.empty_cache()

    fine_tune(model, tokenizer, examples, device, steps=horizon,
              learning_rate=CONFIG['lr'], weight_decay=CONFIG['wd'],
              horizon=horizon, decay=False, seed=seed * 7 + len(SOURCE),
              hook=hook, hook_every=100)
    del model
    torch.cuda.empty_cache()
    return rows


def worker_init(device, batch_log, threads):
    common.setup(threads, device)


def worker(job):
    try:
        rows = run_seed(job)
        return dict(job=job, rows=rows, error=None)
    except Exception as exc:  # noqa: BLE001 - one trajectory must not kill the pool
        common.log_line(f'[q21] seed {job["seed"]} FAILED {type(exc).__name__}: {exc}')
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
    parser.add_argument('--out', default='runs_nseed_queue21/m21_energy.csv')
    parser.add_argument('--check', default='runs_nseed_queue21/p21_source_check.csv')
    parser.add_argument('--symbols', default='runs_nseed_queue20/symbols_p97.json')
    parser.add_argument('--procs', type=int, default=4)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    parser.add_argument('--horizon', type=int, default=CONFIG['horizon'],
                        help='the step of the source at which the edits are measured')
    args = parser.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    payload = json.loads(Path(args.symbols).read_text(encoding='utf-8'))

    done = set()
    for row in read_rows(args.out):
        done.add((row['seed'], row['source'], row['step']))

    jobs = []
    for seed in args.seeds:
        if (str(seed), SOURCE, str(args.horizon)) in done:
            print(f'seed {seed}: already done', flush=True)
            continue
        jobs.append(dict(args=args, device=device, payload=payload, seed=seed,
                         out_path=args.out, check_path=args.check))
    print(f'{len(jobs)} trajectories', flush=True)
    results = run_pool(jobs, device, args.procs, args.threads)
    failed = [result for result in results if result.get('error')]
    for result in failed:
        job = result['job']
        common.log_line(f'[q21] FAILED seed {job["seed"]}: {result["error"]}')
    print(f'{len(results) - len(failed)} trajectories finished, {len(failed)} failed',
          flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
