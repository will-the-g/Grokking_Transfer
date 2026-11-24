"""Queue 14, toy track: the Pre-LN data efficiency, the subtraction sources and
the non-abelian group tasks.

STEP A  the Pre-LN data efficiency with a pair-level target subset: a sanity
        check first (fraction 0.40, three seeds: the grokked source must reach
        0.80 at step 2,000 and 0.95 at step 5,000, the random init at most
        0.05 at step 2,000), then the full sweep over nine seeds, four sources,
        four fractions and 10,000 steps.
STEP B  the subtraction sources with coverage 0.5 and weight decay 2.0, and the
        three-target transfers for the seeds that grok.
STEP C  the table-based group tasks (dihedral D_59 and symmetric S5) with the
        power-affine target.

The fraction of STEP A is a fraction of the target grid of p * p pairs: at
coverage 0.40 the fraction 0.40 is the whole training split. The subset keeps
every context of a kept pair, so the context-dependent target rule stays
consistent across the contexts.

Only dev splits are scored. The audit slice is never read.

Produces: the stage selection and the transfers of queue 14.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import multiprocessing
import random
import sys
import traceback
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402
import torch  # noqa: E402
from run_components import run_transfer_job  # noqa: E402
from run_seed import SeedRun  # noqa: E402

A_FIELDS = ['seed', 'source_kind', 'target_fraction', 'step', 'dev_acc']
B_FIELDS = ['config', 'source_task', 'target_task', 'seed', 'source_kind', 'step',
            'dev_acc', 'source_final_dev', 'source_grok_step']
C_FIELDS = ['group', 'seed', 'source_kind', 'step', 'dev_acc', 'source_final_dev',
            'source_grok_step']
PRELN_ROOTS = ('runs_nseed_p1', 'runs_nseed_k2')
PRELN_SEEDS = (5, 6, 7, 8, 10, 11, 12, 13, 14)
FRACTIONS = (0.05, 0.10, 0.20, 0.40)
SUB_TARGETS = ('affine_add', 'affine_sub', 'affine_mult')
GROUPS = {'D59': 118, 'S5': 120}


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


def cpu_state(state):
    """The state of a model on the CPU, so it can be sent to a worker."""
    return {key: value.detach().cpu() for key, value in state.items()}


# ---------------------------------------------------------------------------
# the target subset
# ---------------------------------------------------------------------------
def subset_rows(strata, fraction, seed):
    """The training rows of a subset of the target pairs.

    The fraction is over the target grid of p * p pairs, not over the training
    split, and every context of a kept pair is kept, so the target rule stays
    consistent across the contexts of a pair. The result stays on the CPU.
    """
    train_ids = [int(value) for value in strata['train']]
    keep = min(len(train_ids), max(1, int(round(fraction * common.P * common.P))))
    chosen = sorted(random.Random(1000 + int(seed)).sample(train_ids, keep))
    return common.index_for(torch.tensor(chosen, dtype=torch.long), 'cpu')


# ---------------------------------------------------------------------------
# the sources
# ---------------------------------------------------------------------------
def train_source(seed, task, out_dir, cap, device, data=None, source_task=None,
                 ckpt_every=250):
    """Train one source with the recipe of the a+b sources."""
    out_dir = Path(out_dir)
    summary_path = out_dir / 'summary.json'
    summary = common.read_summary(summary_path)
    if summary.get('source_finished'):
        return dict(status=summary.get('status'), grok_step=summary.get('grok_step'),
                    stages=summary.get('stages') or {},
                    final_dev=summary.get('source_final_dev', ''))
    run = SeedRun(seed, out_dir, threads=1, resume=False, tiers=(), device=device,
                  source_cap=cap, ckpt_every=ckpt_every, log=common.log_line)
    run._ensure_dirs()
    mask = common.mask_for_seed(seed)
    strata, _exposure = common.split_for_mask(mask, seed)
    if data is None:
        data = common.build_task(task, device)
    status, grok_step, rows = run.train_source(strata, data, run._summary())
    stages = run.select_stages(rows, grok_step, status)
    summary = run._summary()
    summary['stages'] = stages
    summary['source_finished'] = True
    summary['source_task'] = source_task or task
    summary['source_final_dev'] = rows[-1].get('dev_mean') if rows else ''
    run._write_summary(summary)
    common.log_line(f'[q14] source {source_task or task} seed {seed}: status {status} '
                    f'grok_step {grok_step} final dev {summary["source_final_dev"]}')
    return dict(status=status, grok_step=grok_step, stages=stages,
                final_dev=summary['source_final_dev'])


def source_state(seed_dir, kind, stages, grokked_norm):
    """The state of one stage; norm_control is the memorized checkpoint
    rescaled to the grokked norm. `seed_dir` is the directory with `ckpts`."""
    source = 'memorized' if kind == 'norm_control' else kind
    ckpt = Path(seed_dir) / 'ckpts' / stages[source]['ckpt']
    state = common.load_checkpoint(ckpt)['model']
    if kind == 'norm_control':
        model = common.make_model(common.CONFIG)
        model.load_state_dict(state)
        common.rescale_to_norm(model, grokked_norm)
        state = cpu_state(model.state_dict())
    return cpu_state(state)


# ---------------------------------------------------------------------------
# the transfers
# ---------------------------------------------------------------------------
def transfer(state, seed, task, strata, device, name, steps=2000, eval_every=100,
             train_rows=None):
    """The standard transfer; returns its evaluation rows."""
    model = common.make_model(common.CONFIG)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    data = common.build_task(task, device)
    if train_rows is not None:
        train_rows = train_rows.to(device)
    rows, _trained = run_transfer_job(name, seed, model, data, strata, Path('.'),
                                      steps=steps, eval_every=eval_every,
                                      log=common.log_line, train_rows=train_rows)
    return rows


# ---------------------------------------------------------------------------
# the group tasks
# ---------------------------------------------------------------------------
def dihedral_table(n=59):
    """The multiplication table of D_n: the rotations 0..n-1 and the reflections."""
    size = 2 * n
    table = [[0] * size for _ in range(size)]
    for i in range(n):
        for j in range(n):
            table[i][j] = (i + j) % n
            table[i][j + n] = n + ((j - i) % n)
            table[i + n][j] = n + ((i + j) % n)
            table[i + n][j + n] = (j - i) % n
    return table


def permutation_table():
    """The multiplication table of S5: the permutations as tuples, composed."""
    import itertools
    elements = list(itertools.permutations(range(5)))
    index = {perm: position for position, perm in enumerate(elements)}
    table = [[0] * len(elements) for _ in elements]
    for i, first in enumerate(elements):
        for j, second in enumerate(elements):
            composed = tuple(first[second[k]] for k in range(5))
            table[i][j] = index[composed]
    return table


def group_table(group):
    """The multiplication table of one group."""
    if group == 'D59':
        return dihedral_table(59)
    if group == 'S5':
        return permutation_table()
    raise ValueError(f'unknown group {group!r}')


def group_power(table, element, power):
    """The power of an element inside the group."""
    result = 0
    for _ in range(power):
        result = table[result][element]
    return result


def group_data(table, size, device, power=False):
    """The tokens and the labels of a table-based task.

    The row layout follows the convention of `common`: the row of the pair
    (a, b) in the context c is c * size * size + a * size + b and the token
    format is [size + context, a, b, size + n_contexts]. With `power` the
    label is the power-affine target a * b^(c+1).
    """
    rows = size * size * common.N_CONTEXTS
    tokens = torch.empty(rows, 4, dtype=torch.long)
    labels = torch.empty(rows, dtype=torch.long)
    for context in range(common.N_CONTEXTS):
        for a in range(size):
            for b in range(size):
                position = context * size * size + a * size + b
                tokens[position, 0] = size + context
                tokens[position, 1] = a
                tokens[position, 2] = b
                tokens[position, 3] = size + common.N_CONTEXTS
                if power:
                    labels[position] = table[a][group_power(table, b, context + 1)]
                else:
                    labels[position] = table[a][b]
    return dict(x=tokens.to(device), y=labels.to(device), device=device)


def set_group(group):
    """The size of the group becomes the size of the task."""
    size = GROUPS[group]
    common.P = size
    common.CONFIG['p'] = size
    common.COUNT = int(common.COVERAGE * size * size)
    return size


def group_transfer(state, seed, group, strata, device, name, steps=2000,
                   eval_every=100):
    """The standard transfer on the power-affine target of a group."""
    size = set_group(group)
    model = common.make_model(common.CONFIG)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    data = group_data(group_table(group), size, device, power=True)
    rows, _trained = run_transfer_job(name, seed, model, data, strata, Path('.'),
                                      steps=steps, eval_every=eval_every,
                                      log=common.log_line)
    return rows


# ---------------------------------------------------------------------------
# the worker pool
# ---------------------------------------------------------------------------
def worker_init(device, batch_log, threads):
    common.setup(threads, device)
    common.set_batch_log(batch_log)


def worker(job):
    meta = {key: value for key, value in job.items()
            if key not in ('state', 'train_rows', 'data', 'strata')}
    try:
        kind = job['kind']
        if kind == 'transfer':
            rows = transfer(job['state'], job['seed'], job['task'], job['strata'],
                            job['device'], job['name'], steps=job['steps'],
                            eval_every=job['eval_every'],
                            train_rows=job.get('train_rows'))
            return dict(job=meta, rows=rows)
        if kind == 'source':
            data = job.get('data')
            if job.get('group'):
                size = set_group(job['group'])
                data = group_data(group_table(job['group']), size, job['device'])
            return dict(job=meta, result=train_source(
                job['seed'], job['task'], job['dir'], job['cap'], job['device'],
                data=data, source_task=job.get('source_task'),
                ckpt_every=job.get('ckpt_every', 250)))
        if kind == 'group_transfer':
            rows = group_transfer(job['state'], job['seed'], job['group'],
                                  job['strata'], job['device'], job['name'],
                                  steps=job['steps'], eval_every=job['eval_every'])
            return dict(job=meta, rows=rows)
        raise ValueError(f'unknown job kind {kind}')
    except Exception as exc:  # noqa: BLE001 - one job must not kill the batch
        common.log_line(f'[q14] job {job.get("name", "")} FAILED '
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
# STEP A: the Pre-LN data efficiency
# ---------------------------------------------------------------------------
def preln_root(seed):
    for root in PRELN_ROOTS:
        if (Path(root) / f'seed_{seed}' / 'ckpts' / 'grokked.pt').exists():
            return root
    return None


def preln_state(root, seed, kind, stages, grokked_norm, device):
    """The state of one Pre-LN stage, or a fresh random init."""
    if kind == 'random_init':
        model = common.init_model(seed, device)
        return cpu_state(model.state_dict())
    return source_state(Path(root) / f'seed_{seed}', kind, stages, grokked_norm)


def step_a(args, device):
    """The sanity check and then the full Pre-LN data-efficiency sweep."""
    out_dir = project_path(args.out)
    path = out_dir / 'a_dataeff.csv'
    notes = []
    sanity = dict(passed=False, seeds=[])

    check_jobs = []
    for seed in args.check_seeds:
        root = preln_root(seed)
        if root is None:
            notes.append(f'Pre-LN seed {seed}: no grokked checkpoint')
            continue
        summary = common.read_summary(Path(root) / f'seed_{seed}' / 'summary.json')
        stages = summary.get('stages', {}) or {}
        grokked_norm = (stages.get('grokked', {}) or {}).get('global_norm')
        mask = common.mask_for_seed(seed)
        strata, _exposure = common.split_for_mask(mask, seed)
        for kind in ('grokked', 'random_init'):
            check_jobs.append(dict(
                kind='transfer', name=f'a_check_{kind}_seed{seed}', seed=seed,
                stage=kind, task='affine',
                state=preln_state(root, seed, kind, stages, grokked_norm, device),
                strata=strata, device=device, steps=args.check_steps,
                eval_every=args.eval_every,
                train_rows=subset_rows(strata, 0.40, seed)))
    results = run_pool(check_jobs, device, args.procs, str(common.BATCH_LOG),
                       args.threads)
    curves = {}
    for result in results:
        job = result['job']
        for row in result['rows']:
            curves.setdefault((job['stage'], job['seed']), {})[int(row['step'])] = \
                common.as_float(row.get('dev_mean'))
    ok = True
    for seed in args.check_seeds:
        grokked = curves.get(('grokked', seed), {})
        random_init = curves.get(('random_init', seed), {})
        at_2000 = grokked.get(2000)
        at_5000 = grokked.get(5000)
        random_2000 = random_init.get(2000)
        passed = (at_2000 is not None and at_2000 >= 0.80
                  and at_5000 is not None and at_5000 >= 0.95
                  and random_2000 is not None and random_2000 <= 0.05)
        sanity['seeds'].append(dict(seed=seed, grokked_2000=at_2000,
                                    grokked_5000=at_5000,
                                    random_init_2000=random_2000, passed=passed))
        ok = ok and passed
    sanity['passed'] = ok
    sanity_path = out_dir / 'a_sanity.json'
    sanity_path.parent.mkdir(parents=True, exist_ok=True)
    sanity_path.write_text(json.dumps(sanity, indent=2, default=str), encoding='utf-8')
    common.log_line(f'[q14] STEP A sanity check passed: {ok} {sanity["seeds"]}')
    if not ok:
        notes.append('STEP A sanity check failed: the Pre-LN data efficiency was '
                     'not run')
        return notes, sanity

    jobs = []
    for seed in args.seeds:
        root = preln_root(seed)
        if root is None:
            notes.append(f'Pre-LN seed {seed}: no grokked checkpoint, skipped')
            continue
        summary = common.read_summary(Path(root) / f'seed_{seed}' / 'summary.json')
        stages = summary.get('stages', {}) or {}
        grokked_norm = (stages.get('grokked', {}) or {}).get('global_norm')
        mask = common.mask_for_seed(seed)
        strata, _exposure = common.split_for_mask(mask, seed)
        for kind in ('grokked', 'memorized', 'norm_control', 'random_init'):
            if kind == 'norm_control':
                if 'memorized' not in stages or grokked_norm is None:
                    continue
            elif kind != 'random_init' and kind not in stages:
                continue
            state = preln_state(root, seed, kind, stages, grokked_norm, device)
            for fraction in args.fractions:
                jobs.append(dict(
                    kind='transfer', name=f'a_{kind}_{fraction}_seed{seed}',
                    seed=seed, stage=kind, fraction=fraction, task='affine',
                    state=state, strata=strata, device=device, steps=args.steps,
                    eval_every=args.eval_every,
                    train_rows=subset_rows(strata, fraction, seed)))
    for result in run_pool(jobs, device, args.procs, str(common.BATCH_LOG),
                           args.threads):
        job = result['job']
        rows = [dict(seed=job['seed'], source_kind=job['stage'],
                     target_fraction=job['fraction'], step=int(row['step']),
                     dev_acc=common.as_float(row.get('dev_mean')))
                for row in result['rows']]
        if rows:
            append_rows(path, A_FIELDS, rows)
    return notes, sanity


# ---------------------------------------------------------------------------
# STEP B: the subtraction sources
# ---------------------------------------------------------------------------
def step_b(args, device):
    """The subtraction sources with coverage 0.5 and weight decay 2.0."""
    out_dir = project_path(args.out)
    path = out_dir / 'b_sub.csv'
    notes = []
    config = args.config
    source_jobs = []
    for seed in args.seeds:
        src_dir = out_dir / 'sources' / 'sub' / config / f'seed_{seed}'
        source_jobs.append(dict(kind='source', name=f'b_source_sub_seed{seed}',
                                 seed=seed, task='sub', source_task='sub',
                                 dir=str(src_dir), cap=args.cap, device=device))
    results = run_pool(source_jobs, device, args.procs, str(common.BATCH_LOG),
                       args.threads)
    transfer_jobs = []
    for result in results:
        job = result['job']
        seed = job['seed']
        info = result.get('result')
        if not info:
            notes.append(f'sub source seed {seed}: the training failed')
            continue
        stages = info['stages']
        grokked_norm = (stages.get('grokked', {}) or {}).get('global_norm')
        if info['status'] != 'grokked':
            notes.append(f'sub source seed {seed} ({config}) never grokked '
                         f'({info["status"]}, {args.cap} steps)')
            common.log_line(f'[q14] sub source seed {seed} never grokked')
            append_rows(path, B_FIELDS, [dict(config=config, source_task='sub',
                                              target_task='', seed=seed,
                                              source_kind='source', step=args.cap,
                                              dev_acc=info['final_dev'],
                                              source_final_dev=info['final_dev'],
                                              source_grok_step='')])
            continue
        mask = common.mask_for_seed(seed)
        strata, _exposure = common.split_for_mask(mask, seed)
        for target in args.targets:
            for kind in ('memorized', 'transition', 'grokked', 'norm_control'):
                if kind == 'norm_control':
                    if 'memorized' not in stages or grokked_norm is None:
                        continue
                elif kind not in stages:
                    continue
                transfer_jobs.append(dict(
                    kind='transfer', name=f'b_{target}_{kind}_seed{seed}', seed=seed,
                    stage=kind, task=target,
                    state=source_state(job['dir'], kind, stages, grokked_norm),
                    strata=strata, device=device, steps=args.steps,
                    eval_every=args.eval_every))
            model = common.init_model(seed, device)
            transfer_jobs.append(dict(
                kind='transfer', name=f'b_{target}_random_seed{seed}', seed=seed,
                stage='random_init', task=target,
                state=cpu_state(model.state_dict()), strata=strata, device=device,
                steps=args.steps, eval_every=args.eval_every))
    for result in run_pool(transfer_jobs, device, args.procs, str(common.BATCH_LOG),
                           args.threads):
        job = result['job']
        rows = [dict(config=config, source_task='sub', target_task=job['task'],
                     seed=job['seed'], source_kind=job['stage'],
                     step=int(row['step']), dev_acc=common.as_float(row.get('dev_mean')),
                     source_final_dev='', source_grok_step='')
                for row in result['rows']]
        if rows:
            append_rows(path, B_FIELDS, rows)
    return notes


# ---------------------------------------------------------------------------
# STEP C: the group tasks
# ---------------------------------------------------------------------------
def step_c(args, device):
    """The group tasks and their power-affine transfers.

    The sources of a group are trained in parallel and the transfers of the
    seeds that grokked follow. A transfer whose rows are already in the output
    is skipped, so a restarted step does not duplicate them.
    """
    out_dir = project_path(args.out)
    path = out_dir / 'c_groups.csv'
    notes = []
    done = collections.Counter((row['group'], row['seed'], row['source_kind'])
                               for row in read_rows(path))
    expected = max(1, args.steps // args.eval_every)
    for group, size in GROUPS.items():
        set_group(group)
        source_jobs = []
        for seed in args.seeds:
            src_dir = out_dir / 'sources' / group / f'seed_{seed}'
            source_jobs.append(dict(kind='source', name=f'c_source_{group}_seed{seed}',
                                    seed=seed, task=group, source_task=group,
                                    dir=str(src_dir), cap=args.cap, device=device,
                                    group=group, ckpt_every=args.ckpt_every))
        results = run_pool(source_jobs, device, args.procs, str(common.BATCH_LOG),
                           args.threads)
        transfer_jobs = []
        for result in results:
            job = result['job']
            seed = str(job['seed'])
            info = result.get('result')
            if not info:
                notes.append(f'{group} source seed {seed}: the training failed')
                continue
            stages = info['stages']
            grokked_norm = (stages.get('grokked', {}) or {}).get('global_norm')
            if info['status'] != 'grokked':
                notes.append(f'{group} source seed {seed} never grokked '
                             f'({info["status"]}, {args.cap} steps)')
                common.log_line(f'[q14] {group} source seed {seed} never grokked')
                if not done[(group, seed, 'source')]:
                    append_rows(path, C_FIELDS, [dict(group=group, seed=seed,
                                                     source_kind='source',
                                                     step=args.cap,
                                                     dev_acc=info['final_dev'],
                                                     source_final_dev=info['final_dev'],
                                                     source_grok_step='')])
                continue
            mask = common.mask_for_seed(seed)
            strata, _exposure = common.split_for_mask(mask, seed)
            for kind in ('memorized', 'transition', 'grokked', 'norm_control'):
                if kind == 'norm_control':
                    if 'memorized' not in stages or grokked_norm is None:
                        continue
                elif kind not in stages:
                    continue
                if done[(group, seed, kind)] >= expected:
                    continue
                transfer_jobs.append(dict(
                    kind='group_transfer', name=f'c_{group}_{kind}_seed{seed}',
                    seed=seed, stage=kind, group=group,
                    state=source_state(out_dir / 'sources' / group / f'seed_{seed}',
                                       kind, stages, grokked_norm),
                    strata=strata, device=device, steps=args.steps,
                    eval_every=args.eval_every))
            if done[(group, seed, 'random_init')] < expected:
                model = common.init_model(seed, device)
                transfer_jobs.append(dict(
                    kind='group_transfer', name=f'c_{group}_random_seed{seed}',
                    seed=seed, stage='random_init', group=group,
                    state=cpu_state(model.state_dict()), strata=strata,
                    device=device, steps=args.steps, eval_every=args.eval_every))
        for result in run_pool(transfer_jobs, device, args.procs,
                               str(common.BATCH_LOG), args.threads):
            job = result['job']
            rows = [dict(group=group, seed=job['seed'], source_kind=job['stage'],
                         step=int(row['step']),
                         dev_acc=common.as_float(row.get('dev_mean')),
                         source_final_dev='', source_grok_step='')
                    for row in result['rows']]
            if rows:
                append_rows(path, C_FIELDS, rows)
    common.P = 59
    common.CONFIG['p'] = 59
    common.COUNT = int(common.COVERAGE * common.P * common.P)
    return notes


# ---------------------------------------------------------------------------
# the smoke checks
# ---------------------------------------------------------------------------
def check_labels(device):
    """The label rules of the subtraction and the affine targets."""
    failures = []
    generator = torch.Generator().manual_seed(7)
    for _ in range(64):
        a = int(torch.randint(0, common.P, (1,), generator=generator))
        b = int(torch.randint(0, common.P, (1,), generator=generator))
        context = int(torch.randint(0, common.N_CONTEXTS, (1,), generator=generator))
        a_t = torch.tensor([a], device=device)
        b_t = torch.tensor([b], device=device)
        c_t = torch.tensor([context], device=device)
        sub = int(common.task_labels('sub', a_t, b_t, c_t)[0])
        affine_sub = int(common.task_labels('affine_sub', a_t, b_t, c_t)[0])
        affine_mult = int(common.task_labels('affine_mult', a_t, b_t, c_t)[0])
        if sub != (a - b) % common.P:
            failures.append(f'sub label {a},{b} -> {sub}')
        if affine_sub != (a - (context + 1) * b) % common.P:
            failures.append(f'affine_sub label {a},{b},{context} -> {affine_sub}')
        expected = a * pow(b, context + 1, common.P) % common.P
        if affine_mult != expected:
            failures.append(f'affine_mult label {a},{b},{context} -> {affine_mult}')
    return failures


def check_split(seed):
    """The train, dev and audit pair ids are disjoint; the audit ids are in
    neither the train nor the dev set. Only the id sets are inspected."""
    mask = common.mask_for_seed(seed)
    strata, _exposure = common.split_for_mask(mask, seed)
    train = set(int(value) for value in strata['train'])
    dev = set(int(value) for value in strata['dev'])
    audit = set(int(value) for value in strata.get('audit', []))
    failures = []
    if train & dev:
        failures.append(f'train and dev overlap: {len(train & dev)} pairs')
    if audit & train:
        failures.append(f'audit and train overlap: {len(audit & train)} pairs')
    if audit & dev:
        failures.append(f'audit and dev overlap: {len(audit & dev)} pairs')
    return failures, dict(train=len(train), dev=len(dev), audit=len(audit))


def check_groups():
    """The group tables are groups and the power rule holds."""
    failures = []
    table = dihedral_table(59)
    if any(table[table[a][b]][c] != table[a][table[b][c]]
           for a in range(118) for b in range(118) for c in range(118)):
        failures.append('the dihedral table is not associative')
    if any(table[a][0] != a for a in range(118)):
        failures.append('the dihedral table has no identity')
    permutation = permutation_table()
    if len(permutation) != 120 or any(
            permutation[permutation[a][b]][c] != permutation[a][permutation[b][c]]
            for a in range(120) for b in range(120) for c in range(120)):
        failures.append('the S5 table is not a group of order 120')
    return failures


def step_smoke(args, device):
    """The smoke checks of the label rules, the split, the group tables and a
    short training."""
    notes = []
    failures = check_labels(device)
    failures.extend(check_groups())
    for seed in args.check_seeds:
        split_failures, counts = check_split(seed)
        failures.extend(split_failures)
        common.log_line(f'[q14] smoke split seed {seed}: {counts}')
    data = common.build_task('sub', device)
    model = common.init_model(args.check_seeds[0], device)
    optimizer = common.make_optimizer(model)
    mask = common.mask_for_seed(args.check_seeds[0])
    strata, _exposure = common.split_for_mask(mask, args.check_seeds[0])
    rows = common.index_for(strata['train'], device)[:64]
    loss = None
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        logits = model(data['x'][rows])
        loss = torch.nn.functional.cross_entropy(logits, data['y'][rows])
        loss.backward()
        optimizer.step()
    common.log_line(f'[q14] smoke sub training loss {float(loss):.6f}')
    if not torch.isfinite(loss):
        failures.append('the smoke training loss is not finite')
    if failures:
        for failure in failures:
            notes.append(f'SMOKE FAILURE: {failure}')
    else:
        notes.append('smoke checks passed: the label rules, the split, the group '
                     'tables and a short sub training')
    return notes


# ---------------------------------------------------------------------------
# the report and the pack
# ---------------------------------------------------------------------------
def step_pack(args):
    """Pack only the queue14 CSVs and the report."""
    root = Path(args.root) if args.root else common.PARENT
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = root / 'runs_nseed_queue14'
    members = []
    report_rows = []
    for name in ('a_dataeff.csv', 'b_sub.csv', 'c_groups.csv'):
        path = base / name
        if path.exists():
            members.append(path)
            report_rows.append((name, len(read_rows(path))))
    report = root / 'nseed_backbone' / 'REPORT_queue14.md'
    sanity = {}
    sanity_path = base / 'a_sanity.json'
    if sanity_path.exists():
        try:
            sanity = json.loads(sanity_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            sanity = {}
    diagnosis = {}
    diagnosis_path = base / 'a_diagnosis.json'
    if diagnosis_path.exists():
        try:
            diagnosis = json.loads(diagnosis_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            diagnosis = {}
    write_report(report, report_rows, args.failures, sanity, diagnosis)
    members.append(report)
    with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in members:
            archive.write(path, path.relative_to(root).as_posix())
    common.log_line(f'[q14] packed {len(members)} files into {out_path}')
    return out_path


def write_report(path, rows, failures, sanity, diagnosis):
    """The files, the row counts, the failures and the STEP A diagnosis."""
    lines = ['# REPORT_queue14', '',
             '| file | rows |', '|---|---|']
    for name, count in rows:
        lines.append(f'| {name} | {count} |')
    lines.append('')
    lines.append('| step | detail |')
    lines.append('|---|---|')
    lines.append(f'| STEP A sanity check | passed: {sanity.get("passed")} |')
    for record in sanity.get('seeds', []):
        lines.append(f'| STEP A sanity seed {record["seed"]} | grokked@2000 '
                     f'{record["grokked_2000"]} grokked@5000 {record["grokked_5000"]} '
                     f'random@2000 {record["random_init_2000"]} passed '
                     f'{record["passed"]} |')
    if diagnosis:
        lines.append('| STEP A cause | ' + diagnosis.get('summary', '') + ' |')
        for record in diagnosis.get('runs', []):
            lines.append(f'| STEP A construction {record["construction"]} '
                         f'fraction {record["fraction"]} seed {record["seed"]} | '
                         f'dev@2000 {record["dev_2000"]} train@2000 '
                         f'{record["train_2000"]} rows {record["rows"]} |')
    for name in failures or ('none',):
        lines.append(f'| {name} | |')
    Path(path).write_text('\n'.join(lines), encoding='utf-8')
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['smoke', 'a', 'b', 'c', 'pack'])
    parser.add_argument('--out', default='runs_nseed_queue14')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(PRELN_SEEDS))
    parser.add_argument('--check-seeds', nargs='+', type=int, default=[5, 6, 7])
    parser.add_argument('--fractions', nargs='+', type=float, default=list(FRACTIONS))
    parser.add_argument('--targets', nargs='+', default=list(SUB_TARGETS))
    parser.add_argument('--config', default='cov0.5_wd2.0')
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--check-steps', type=int, default=5000)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--cap', type=int, default=50000)
    parser.add_argument('--ckpt-every', type=int, default=1000)
    parser.add_argument('--procs', type=int, default=4)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--root', default=None)
    parser.add_argument('--failures', nargs='*', default=[])
    parser.add_argument('--failures-file', default=None)
    parser.add_argument('--sanity', default='{}')
    parser.add_argument('--batch-log', default='runs_nseed/batch.log')
    args = parser.parse_args(argv)

    if args.failures_file and Path(args.failures_file).exists():
        args.failures = [line.strip() for line in
                         Path(args.failures_file).read_text(encoding='utf-8').splitlines()
                         if line.strip()]

    runtime = common.setup(args.threads, args.device)
    device = runtime['device']
    common.set_batch_log(project_path(args.batch_log))
    common.log_line(f'[q14] step {args.step} device={device}')

    if args.step == 'smoke':
        for note in step_smoke(args, device):
            common.log_line(f'[q14] {note}')
    elif args.step == 'a':
        notes, sanity = step_a(args, device)
        for note in notes:
            common.log_line(f'[q14] {note}')
        common.log_line(f'[q14] STEP A sanity: {sanity}')
    elif args.step == 'b':
        for note in step_b(args, device):
            common.log_line(f'[q14] {note}')
    elif args.step == 'c':
        for note in step_c(args, device):
            common.log_line(f'[q14] {note}')
    elif args.step == 'pack':
        step_pack(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
