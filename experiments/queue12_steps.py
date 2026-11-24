"""Steps of queue 12: the Pythia trajectories, the checkpoint sweep and the toy
learning curves.

A1  the stage-1 trajectory of pythia-1.4b at step143000: three sources (a
    generalising a+b, a memorising a+b and the unrelated copy-a), an adapter
    saved every 100 steps, and the stage-2 transfer from every adapter.
A2  the same stage-1/stage-2 protocol over the pythia-1b and pythia-1.4b
    revisions, with the stage-1 sources add, copy_a, max, shuffled and add_mem.
A3  the toy learning curves: 20,000-step affine transfers from the stored
    stage checkpoints of the no-LN and the Pre-LN sources.

One job covers one (model, revision, seed) and all of its sources, so the
direct_short and direct_long baselines are computed once per seed. Every row is
appended to its step CSV as soon as it is finished, so a partial run survives;
the rows already in the CSV are skipped.

    python nseed_backbone\\queue12_steps.py traj --model EleutherAI/pythia-1.4b
    python nseed_backbone\\queue12_steps.py sweep --models EleutherAI/pythia-1b
    python nseed_backbone\\queue12_steps.py toy --arch no-LN --runs-root runs_nseed
    python nseed_backbone\\queue12_steps.py pack --out results_queue12.zip

Produces: Figure 3 (the LoRA checkpoint sweep).
"""
from __future__ import annotations

import argparse
import csv
import io
import multiprocessing
import os
import random
import sys
import traceback
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
HERE = Path(__file__).resolve().parent
SRC = HERE.parent / 'src'
for _path in (str(SRC),):
    if _path not in sys.path:
        sys.path.insert(0, _path)

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402
import torch  # noqa: E402
from peft import (LoraConfig, PeftModel, get_peft_model,  # noqa: E402
                  get_peft_model_state_dict, set_peft_model_state_dict)
from pythia_transfer import (BATCH_SIZE, LEARNING_RATE, LORA_ALPHA,  # noqa: E402
                             LORA_RANK, TARGET_MODULES, batched_correct,
                             build_examples, build_shuffled_examples, clear_model_cache,
                             fine_tune, load_model, problems, read_rows)

TRAJ_FIELDS = ['model', 'seed', 'source', 'step', 'train_acc', 'heldout_ab_acc',
               'stage2_acc', 'gain_over_best_direct']
SWEEP_FIELDS = ['model', 'revision', 'seed', 'source', 'train_acc',
                'heldout_ab_acc', 'stage2_acc', 'gain_over_best_direct']
TOY_FIELDS = ['arch', 'seed', 'source_kind', 'step', 'dev_acc']
SOURCES = {'add': ('sum', 4000), 'add_gen': ('sum', 4000), 'copy_a': ('copy', 4000),
           'max': ('max', 4000), 'shuffled': ('sum', 4000), 'add_mem': ('sum', 100)}
STAGE2_EXAMPLES = 500
STAGE2_STEPS = 600
DIRECT_LONG_STEPS = 1600
STAGE1_STEPS = 1000
TRAIN_ACC_SAMPLES = 500
HELD_OUT = 500


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


def split_pairs(seed, n_train):
    """The train pairs and the three disjoint held-out sets."""
    pool = problems(n_train + 3 * HELD_OUT, seed)
    return (pool[:n_train], pool[n_train:n_train + HELD_OUT],
            pool[n_train + HELD_OUT:n_train + 2 * HELD_OUT],
            pool[n_train + 2 * HELD_OUT:n_train + 3 * HELD_OUT])


def examples_for(pairs, source, seed):
    """The training texts and answers of one source."""
    if source == 'shuffled':
        return build_shuffled_examples(pairs, seed)
    mode = SOURCES[source][0]
    return build_examples(pairs, mode)


def adapter_config():
    return LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
                       bias='none', task_type='CAUSAL_LM',
                       target_modules=TARGET_MODULES)


def train_stage1(model, tokenizer, examples, device, steps=STAGE1_STEPS,
                 batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, save_every=0,
                 save_dir=None, tag='adapter'):
    """Train the LoRA and save the adapter every `save_every` steps."""
    model = get_peft_model(model, adapter_config())
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=learning_rate)
    generator = random.Random(0)
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        torch.save(get_peft_model_state_dict(model),
                   Path(save_dir) / f'{tag}_step0.pt')
    for step in range(1, steps + 1):
        batch = [examples[generator.randrange(len(examples))] for _ in range(batch_size)]
        texts = [text for text, _answer in batch]
        answers = [answer for _text, answer in batch]
        full = [text + answer for text, answer in batch]
        encoded = tokenizer(full, return_tensors='pt', padding=True,
                            truncation=True, max_length=64).to(device)
        prompt_lengths = [len(tokenizer(text)['input_ids']) for text in texts]
        labels = encoded['input_ids'].clone()
        labels[encoded['attention_mask'] == 0] = -100
        for row, length in enumerate(prompt_lengths):
            labels[row, :length] = -100
        loss = model(**encoded, labels=labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if save_dir is not None and save_every and step % save_every == 0:
            torch.save(get_peft_model_state_dict(model),
                       Path(save_dir) / f'{tag}_step{step}.pt')
    model.eval()
    return model


def adapter_model(base_model, path):
    """A fresh LoRA carrying the saved adapter weights."""
    model = get_peft_model(base_model, adapter_config())
    set_peft_model_state_dict(model, torch.load(path, map_location='cpu'))
    return model


def example_accuracy(model, tokenizer, examples, device):
    """The accuracy on prepared (prompt, answer) examples."""
    if not examples:
        return float('nan')
    prompts = [text for text, _answer in examples]
    answers = [answer for _text, answer in examples]
    return batched_correct(model, tokenizer, prompts, answers, device) / len(examples)


def accuracy_on_pairs(model, tokenizer, pairs, mode, device):
    """The accuracy on pairs with the true labels of one mode."""
    if not pairs:
        return float('nan')
    labels = {'sum': lambda a, b: a + b, 'a2b': lambda a, b: a + 2 * b,
              'copy': lambda a, b: a, 'max': lambda a, b: max(a, b)}
    prompts = [f'{a}+{b}=' for a, b in pairs]
    answers = [str(labels[mode](a, b)) for a, b in pairs]
    return batched_correct(model, tokenizer, prompts, answers, device) / len(pairs)


# ---------------------------------------------------------------------------
# A1 and A2: the Pythia protocol
# ---------------------------------------------------------------------------
def pythia_job(job):
    """One (model, revision, seed) with every source and the shared baselines."""
    model_name = job['model']
    revision = job['revision']
    seed = int(job['seed'])
    device = job['device']
    rows = []
    a2b_examples = None
    heldout_a2b = None

    for source in job['sources']:
        mode, n_train = SOURCES[source]
        train_pairs, heldout_ab, heldout_a2b, stage2_pairs = split_pairs(seed, n_train)
        if a2b_examples is None:
            a2b_examples = build_examples(stage2_pairs, 'a2b')

        model, tokenizer = load_model(model_name, revision, device)
        stage1 = train_stage1(model, tokenizer, examples_for(train_pairs, source, seed),
                              device, steps=job['steps'],
                              save_every=job['save_every'], save_dir=job['save_dir'],
                              tag=source)
        train_sample = examples_for(train_pairs, source, seed)[:TRAIN_ACC_SAMPLES]
        if job['save_dir'] is not None:
            adapters = sorted(Path(job['save_dir']).glob(f'{source}_step*.pt'),
                              key=lambda path: int(path.stem.split('step')[-1]))
        else:
            adapters = [None]
        for adapter in adapters:
            step = 0 if adapter is None else int(adapter.stem.split('step')[-1])
            if adapter is None:
                stage1_model = stage1
            else:
                base, _tok = load_model(model_name, revision, device)
                stage1_model = adapter_model(base, adapter)
            train_acc = example_accuracy(stage1_model, tokenizer, train_sample, device)
            heldout = accuracy_on_pairs(stage1_model, tokenizer, heldout_ab, 'sum', device)
            if adapter is None:
                stage2_model = stage1_model
            else:
                del stage1_model
                stage1_model = None
                torch.cuda.empty_cache()
                base, _tok = load_model(model_name, revision, device)
                stage2_model = adapter_model(base, adapter)
            stage2, _t = fine_tune(stage2_model, tokenizer, a2b_examples, device,
                                   steps=job['stage2_steps'], batch_size=BATCH_SIZE)
            stage2_acc = accuracy_on_pairs(stage2, tokenizer, heldout_a2b, 'a2b', device)
            del stage2
            if stage1_model is not None:
                del stage1_model
            torch.cuda.empty_cache()
            rows.append(dict(model=model_name, revision=revision, seed=seed,
                            source=source, step=step, train_acc=train_acc,
                            heldout_ab_acc=heldout, stage2_acc=stage2_acc,
                            gain_over_best_direct=''))
            common.log_line(f'[queue12] {model_name} {revision} seed {seed} {source} '
                            f'step {step}: train {train_acc:.4f} ab {heldout:.4f} '
                            f'stage2 {stage2_acc:.4f}')
        del stage1
        torch.cuda.empty_cache()

    model, tokenizer = load_model(model_name, revision, device)
    short, _t = fine_tune(model, tokenizer, a2b_examples, device, steps=job['stage2_steps'],
                          batch_size=BATCH_SIZE, lora_seed=2000 + seed)
    short_acc = accuracy_on_pairs(short, tokenizer, heldout_a2b, 'a2b', device)
    del short
    torch.cuda.empty_cache()
    model, tokenizer = load_model(model_name, revision, device)
    long_model, _t = fine_tune(model, tokenizer, a2b_examples, device,
                              steps=job['direct_long_steps'], batch_size=BATCH_SIZE,
                              lora_seed=3000 + seed)
    long_acc = accuracy_on_pairs(long_model, tokenizer, heldout_a2b, 'a2b', device)
    del long_model
    torch.cuda.empty_cache()
    best = max(short_acc, long_acc)
    common.log_line(f'[queue12] {model_name} {revision} seed {seed} baselines: '
                    f'short {short_acc:.4f} long {long_acc:.4f}')
    for row in rows:
        row['gain_over_best_direct'] = row['stage2_acc'] - best
    return rows


# ---------------------------------------------------------------------------
# A3: the toy learning curves
# ---------------------------------------------------------------------------
def toy_job(job):
    """One (arch, seed, source_kind) toy run of 20,000 steps."""
    from queue8_steps import stage_checkpoint
    from run_components import run_transfer_job
    device = job['device']
    seed = int(job['seed'])
    kind = job['source_kind']
    runs_root = job['runs_root']
    summary = common.read_summary(Path(runs_root) / f'seed_{seed}' / 'summary.json')
    stages = summary.get('stages', {}) or {}
    if kind == 'random_init':
        model = common.init_model(seed, device)
    else:
        grokked_norm = (stages.get('grokked', {}) or {}).get('global_norm')
        state = stage_checkpoint(runs_root, seed, kind, stages, grokked_norm)
        model = common.make_model(common.CONFIG)
        model.load_state_dict(state)
        model = model.to(device)
    model.eval()
    mask = common.mask_for_seed(seed)
    strata, _exposure = common.split_for_mask(mask)
    target_data = common.build_task('affine', device)
    rows, _trained = run_transfer_job(f'toy_{job["arch"]}_{kind}_seed{seed}', seed,
                                      model, target_data, strata, Path('.'),
                                      steps=job['steps'], eval_every=job['eval_every'],
                                      log=common.log_line)
    return [dict(arch=job['arch'], seed=seed, source_kind=kind, step=int(row['step']),
                 dev_acc=common.as_float(row.get('dev_mean'))) for row in rows]


# ---------------------------------------------------------------------------
# the pool
# ---------------------------------------------------------------------------
def worker_init(device, batch_log, threads):
    common.setup(threads, device)
    common.set_batch_log(batch_log)
    os.environ['HF_HUB_DISABLE_XET'] = '1'


def worker(job):
    try:
        if job['kind'] == 'toy':
            return toy_job(job)
        return pythia_job(job)
    except Exception as exc:  # noqa: BLE001 - one job must not kill the batch
        common.log_line(f'[queue12] job {job.get("name", "")} FAILED '
                        f'{type(exc).__name__}: {exc}')
        common.log_line(traceback.format_exc())
        return [dict(name=job.get('name', ''), error=f'{type(exc).__name__}: {exc}')]


def run_pool(jobs, device, procs, batch_log, threads=1):
    """Run the jobs over a pool of worker processes."""
    results = []
    if not jobs:
        return results
    if procs <= 1:
        worker_init(device, batch_log, threads)
        for job in jobs:
            results.extend(worker(job))
        return results
    context = multiprocessing.get_context('spawn')
    with context.Pool(processes=procs, initializer=worker_init,
                      initargs=(device, batch_log, threads)) as pool:
        for result in pool.imap_unordered(worker, jobs):
            results.extend(result)
    return results


# ---------------------------------------------------------------------------
# the steps
# ---------------------------------------------------------------------------
def step_traj(args, device):
    """A1: the stage-1 trajectory of one model."""
    out_dir = project_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'pythia_traj.csv'
    done = {(row['model'], row['seed'], row['source'], row['step'])
            for row in read_rows(path)}
    jobs = []
    for seed in args.seeds:
        if all((args.model, str(seed), source, '0') in done for source in args.sources):
            common.log_line(f'[queue12] {args.model} seed {seed}: already done')
            continue
        jobs.append(dict(kind='pythia', name=f'{args.model}_seed{seed}',
                         model=args.model, revision=args.revision, seed=seed,
                         sources=list(args.sources), device=device, steps=args.steps,
                         save_every=args.save_every,
                         stage2_steps=args.stage2_steps,
                         direct_long_steps=args.direct_long_steps,
                         save_dir=str(out_dir / 'adapters' / f'{args.model}_seed{seed}')))
    results = run_pool(jobs, device, args.procs, str(common.BATCH_LOG), args.threads)
    rows = [row for row in results if 'error' not in row]
    for row in rows:
        key = (row['model'], row['seed'], row['source'], str(row['step']))
        if key in done:
            continue
        append_rows(path, TRAJ_FIELDS, [row])
        done.add(key)
    clear_model_cache(args.model)
    return rows


def step_sweep(args, device):
    """A2: the checkpoint sweep."""
    out_dir = project_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'pythia_ckpt_sweep.csv'
    done = {(row['model'], row['revision'], row['seed'], row['source'])
            for row in read_rows(path)}
    jobs = []
    rows = []
    for model_name in args.models:
        for revision in args.revisions:
            group = []
            for seed in args.seeds:
                if all((model_name, revision, str(seed), source) in done
                       for source in args.sources):
                    continue
                group.append(dict(kind='pythia', name=f'{model_name}_{revision}_{seed}',
                                  model=model_name, revision=revision, seed=seed,
                                  sources=list(args.sources), device=device,
                                  steps=args.steps, save_every=0, save_dir=None,
                                  stage2_steps=args.stage2_steps,
                                  direct_long_steps=args.direct_long_steps))
            if not group:
                continue
            results = run_pool(group, device, args.procs, str(common.BATCH_LOG),
                               args.threads)
            for row in results:
                if 'error' in row:
                    continue
                key = (row['model'], row['revision'], row['seed'], row['source'])
                if key in done:
                    continue
                append_rows(path, SWEEP_FIELDS, [row])
                done.add(key)
                rows.append(row)
            clear_model_cache(model_name)
    return rows


def step_toy(args, device):
    """A3: the toy learning curves."""
    out_dir = project_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'toy_learning_curves.csv'
    done = {(row['arch'], row['seed'], row['source_kind'], row['step'])
            for row in read_rows(path)}
    jobs = []
    finished = {(row['arch'], row['seed'], row['source_kind']) for row in read_rows(path)}
    for seed in args.seeds:
        for kind in args.kinds:
            if (args.arch, str(seed), kind) in finished:
                common.log_line(f'[queue12] {args.arch} seed {seed} {kind}: already done')
                continue
            jobs.append(dict(kind='toy', name=f'{args.arch}_{seed}_{kind}',
                             arch=args.arch, seed=seed, source_kind=kind,
                             runs_root=args.runs_root, device=device,
                             steps=args.steps, eval_every=args.eval_every))
    results = run_pool(jobs, device, args.procs, str(common.BATCH_LOG), args.threads)
    rows = [row for row in results if 'error' not in row]
    for row in rows:
        key = (row['arch'], row['seed'], row['source_kind'], str(row['step']))
        if key in done:
            continue
        append_rows(path, TOY_FIELDS, [row])
        done.add(key)
    return rows


def step_pack(args):
    """Pack only the queue12 CSVs and the report."""
    root = Path(args.root) if args.root else common.PARENT
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = root / 'runs_nseed_queue12'
    members = []
    report_rows = []
    for name in ('pythia_traj.csv', 'pythia_ckpt_sweep.csv', 'toy_learning_curves.csv'):
        path = base / name
        if path.exists():
            members.append(path)
            report_rows.append((name, len(read_rows(path))))
    report = root / 'nseed_backbone' / 'REPORT_queue12.md'
    write_report(report, report_rows, args.failures)
    members.append(report)
    with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in members:
            archive.write(path, path.relative_to(root).as_posix())
    common.log_line(f'[queue12] packed {len(members)} files into {out_path}')
    return out_path


def write_report(path, rows, failures):
    """The files written, the row counts and the failures; nothing else."""
    lines = ['# REPORT_queue12', '',
             '| file | rows |', '|---|---|']
    for name, count in rows:
        lines.append(f'| {name} | {count} |')
    lines.append('')
    lines.append('| failure |')
    lines.append('|---|')
    for name in failures or ('none',):
        lines.append(f'| {name} |')
    Path(path).write_text('\n'.join(lines), encoding='utf-8')
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['traj', 'sweep', 'toy', 'pack'])
    parser.add_argument('--model', default='EleutherAI/pythia-1.4b')
    parser.add_argument('--models', nargs='+', default=['EleutherAI/pythia-1b'])
    parser.add_argument('--revision', default='step143000')
    parser.add_argument('--revisions', nargs='+', default=[])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument('--sources', nargs='+',
                        default=['add', 'copy_a', 'max', 'shuffled', 'add_mem'])
    parser.add_argument('--arch', default='no-LN')
    parser.add_argument('--runs-root', default='runs_nseed')
    parser.add_argument('--kinds', nargs='+',
                        default=['memorized', 'transition', 'grokked', 'norm_control',
                                 'random_init'])
    parser.add_argument('--out', default='runs_nseed_queue12')
    parser.add_argument('--steps', type=int, default=STAGE1_STEPS)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--stage2-steps', type=int, default=STAGE2_STEPS)
    parser.add_argument('--direct-long-steps', type=int, default=DIRECT_LONG_STEPS)
    parser.add_argument('--procs', type=int, default=3)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--root', default=None)
    parser.add_argument('--failures', nargs='*', default=[])
    parser.add_argument('--batch-log', default='runs_nseed/batch.log')
    args = parser.parse_args(argv)

    runtime = common.setup(args.threads, args.device)
    device = runtime['device']
    common.set_batch_log(project_path(args.batch_log))
    common.log_line(f'[queue12] step {args.step} device={device}')

    if args.step == 'traj':
        step_traj(args, device)
    elif args.step == 'sweep':
        step_sweep(args, device)
    elif args.step == 'toy':
        step_toy(args, device)
    elif args.step == 'pack':
        step_pack(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
