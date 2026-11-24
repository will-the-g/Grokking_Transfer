"""Factorial component study: which grokked components matter?

Every job builds a hybrid model, records the SOURCE-task accuracy (train and
clean dev) at step 0, and then runs a 2,000-step affine transfer with all
parameters trainable and a fresh AdamW, evaluated every 100 steps exactly like
a Tier A transfer. The final checkpoint of every job is saved.

Job families (seeds 5-10):

  factorial   For every subset S of {numeric_embed, attn, mlp, unembed} (16
              subsets, empty and full included): recipient = the seed's
              memorized checkpoint with the components in S copied from its
              grokked checkpoint. `other_embed` always stays memorized.
  obstacle    Recipient = the memorized checkpoint, donor = a fresh model from
              `core.make_model` after `torch.manual_seed(70000 + seed)`; the
              named components are copied from the fresh donor.
  embedding   Recipient = the grokked checkpoint.
              embed_random_matched  the numeric embedding is replaced by the
                                    fresh donor's, rescaled so that its
                                    Frobenius norm equals the grokked one
              embed_rowperm         the rows of the grokked numeric embedding
                                    are permuted with torch.randperm(p,
                                    generator seeded 4242)

Component names: numeric_embed = token_embed.weight[:p]; other_embed =
token_embed.weight[p:] and pos_embed; attn = all parameters of model.attn;
mlp = mlp_in and mlp_out; unembed = unembed.weight.

    python nseed_backbone\\run_factorial.py --seeds 5 6 7 8 9 10 --procs 3 --device cuda

Produces: Table 4 (the factorial reproduction).
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
HERE = Path(__file__).resolve().parent
SRC = HERE.parent / 'src'
for _path in (str(SRC),):
    if _path not in sys.path:
        sys.path.insert(0, _path)

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402
from parallel import run_parallel  # noqa: E402
from run_components import run_transfer_job  # noqa: E402

FACTORIAL_COMPONENTS = ('numeric_embed', 'attn', 'mlp', 'unembed')
OBSTACLE_JOBS = {
    'mem_reset_attn': ('attn',),
    'mem_reset_attn_mlp': ('attn', 'mlp'),
    'mem_reset_attn_unembed': ('attn', 'unembed'),
    'mem_reset_mlp_unembed': ('mlp', 'unembed'),
    'mem_reset_attn_mlp_unembed': ('attn', 'mlp', 'unembed'),
}
EMBED_JOBS = ('embed_random_matched', 'embed_rowperm')
ROWPERM_SEED = 4242
DONOR_SEED_BASE = 70000
TRANSFER_STEPS = 2000
EVAL_EVERY = 100


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else common.PARENT / path


def factorial_cells():
    """The sixteen subsets of the factorial components."""
    cells = []
    for mask in range(16):
        subset = tuple(name for index, name in enumerate(FACTORIAL_COMPONENTS)
                       if mask >> index & 1)
        cells.append(subset)
    return cells


def cell_name(subset):
    return 'fact_' + ('empty' if not subset else '+'.join(sorted(subset)))


def cell_key(subset):
    """The name used to select a cell: 'empty' or the sorted components."""
    return 'empty' if not subset else '+'.join(sorted(subset))


def parse_cells(specs):
    """The set of cell keys named on the command line.

    'empty' is the cell with no components (its job name is fact_empty); the
    other names are the factorial components, optionally joined by '+'.
    """
    wanted = set()
    for spec in specs or ():
        parts = [part for part in str(spec).split('+') if part]
        if 'empty' in parts:
            others = [part for part in parts if part != 'empty']
            if others:
                raise ValueError(f"'empty' cannot be combined with {others} "
                                 f'in --cells entry {spec!r}')
            wanted.add('empty')
            continue
        unknown = [part for part in parts if part not in FACTORIAL_COMPONENTS]
        if unknown:
            raise ValueError(f'unknown component(s) {unknown} in --cells entry {spec!r}')
        wanted.add('empty' if not parts else '+'.join(sorted(parts)))
    return wanted


def component_norms(state):
    """The Frobenius norm of each named component of a state dict."""
    norms = {}
    entries = {}
    for name in ('numeric_embed', 'other_embed', 'attn', 'mlp', 'unembed'):
        entries[name] = common.component_entries(state, name)
    for name, items in entries.items():
        total = 0.0
        for key, rows in items:
            value = state[key]
            piece = value if rows is None else value[rows]
            total += float(piece.detach().double().pow(2).sum())
        norms[name] = total ** 0.5
    return norms


def fresh_donor_state(seed):
    torch.manual_seed(DONOR_SEED_BASE + int(seed))
    model = common.make_model(common.CONFIG)
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def build_job_model(job, kind, subset, memorized, grokked, donor, device):
    """The starting model of one job."""
    if kind == 'factorial':
        merged = common.build_hybrid(memorized, grokked, subset)
    elif kind == 'obstacle':
        merged = common.build_hybrid(memorized, donor, subset)
    elif kind == 'embed_random_matched':
        merged = common.build_hybrid(grokked, donor, ['numeric_embed'])
        donor_rows = donor['token_embed.weight'][:common.P].double()
        target = grokked['token_embed.weight'][:common.P].double().norm()
        current = donor_rows.norm()
        scale = float(target / current) if float(current) > 0 else 0.0
        merged['token_embed.weight'][:common.P] = (donor_rows * scale).to(
            merged['token_embed.weight'].dtype)
    elif kind == 'embed_rowperm':
        merged = {key: value.clone() for key, value in grokked.items()}
        generator = torch.Generator().manual_seed(ROWPERM_SEED)
        permutation = torch.randperm(common.P, generator=generator)
        merged['token_embed.weight'][:common.P] = (
            grokked['token_embed.weight'][:common.P][permutation].clone())
    else:
        raise ValueError(f'Unknown job kind {kind!r}')
    model = common.make_model(common.CONFIG)
    model.load_state_dict(merged)
    return model.to(device)


def job_plan(families=('factorial', 'obstacle', 'embedding'), cells=None):
    """(name, kind, subset) for every job of one seed.

    `cells` restricts the factorial family to the named cells (see
    `parse_cells`); the obstacle and embedding families are unaffected.
    """
    plan = []
    if 'factorial' in families:
        for subset in factorial_cells():
            if cells and cell_key(subset) not in cells:
                continue
            plan.append((cell_name(subset), 'factorial', subset))
    if 'obstacle' in families:
        for name, subset in OBSTACLE_JOBS.items():
            plan.append((name, 'obstacle', subset))
    if 'embedding' in families:
        for name in EMBED_JOBS:
            plan.append((name, name, ()))
    return plan


def run_seed_jobs(seed, out_dir, device, steps=TRANSFER_STEPS, eval_every=EVAL_EVERY,
                  families=('factorial', 'obstacle', 'embedding'), cells=None,
                  resume=False, runtime=None, log=common.log_line):
    """All jobs for one seed."""
    seed_dir = common.PARENT / 'runs_nseed' / f'seed_{seed}'
    summary_path = common.read_summary(seed_dir / 'summary.json')
    stages = summary_path.get('stages', {}) or {}
    if 'grokked' not in stages or 'memorized' not in stages:
        log(f'[factorial] seed {seed}: no grokked/memorized stage, skipped')
        return None
    ckpt_dir = seed_dir / 'ckpts'
    grokked = common.load_checkpoint(ckpt_dir / stages['grokked']['ckpt'])['model']
    memorized = common.load_checkpoint(ckpt_dir / stages['memorized']['ckpt'])['model']
    donor = fresh_donor_state(seed)

    mask = common.mask_for_seed(seed)
    strata, _ = common.split_for_mask(mask)
    source_data = common.build_task('source', device)
    target_data = common.build_task('affine', device)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'ckpts').mkdir(parents=True, exist_ok=True)
    summary = common.read_summary(out_dir / 'summary.json')
    summary.setdefault('seed', seed)
    summary.setdefault('device', device)
    summary['runtime'] = runtime or common.runtime_metadata(device)
    jobs = summary.setdefault('jobs', {})

    norms = {'fresh_donor': component_norms(donor), 'memorized': component_norms(memorized),
             'grokked': component_norms(grokked)}
    summary['component_norms'] = norms
    common.write_summary(out_dir / 'summary.json', summary)

    for name, kind, subset in job_plan(families, cells):
        if resume and jobs.get(name, {}).get('done'):
            log(f'  [{name}] already done, skipping')
            common.skipped_job_line(seed, name, steps, jobs[name])
            continue
        model = build_job_model(name, kind, subset, memorized, grokked, donor, device)
        evaluation = common.evaluate(model, source_data, strata)
        job_started = time.time()
        common.job_line(seed, name, 0, steps, f'{evaluation["dev_mean"]:.6f}', 'start',
                        elapsed=0.0)
        log(f'  [{name}] step 0 source: train {evaluation["train_acc"]:.4f} '
            f'dev {evaluation["dev_mean"]:.4f}')

        rows, trained = run_transfer_job(name, seed, model, target_data, strata, out_dir,
                                         steps=steps, eval_every=eval_every, log=log)
        common.write_metrics(out_dir / 'metrics.csv', rows)
        common.save_model_checkpoint(out_dir / 'ckpts' / f'{name}_final.pt', trained,
                                     steps, rows[-1] if rows else None,
                                     extra={'seed': seed, 'task': 'affine', 'init': name,
                                            'stage': 'factorial',
                                            'global_norm': common.global_norm(trained)})
        final = rows[-1] if rows else {}
        jobs[name] = dict(
            name=name, kind=kind, task='affine', seed=seed, done=True, steps=steps,
            components=list(subset),
            final_dev_mean=common.as_float(final.get('dev_mean')),
            final_dev_worst=common.as_float(final.get('dev_worst')),
            final_train_acc=common.as_float(final.get('train_acc')),
            best_dev_mean=max((common.as_float(r['dev_mean']) or 0.0) for r in rows) if rows else None,
            source_train_acc=evaluation['train_acc'],
            source_dev_mean=evaluation['dev_mean'],
            source_dev_worst=evaluation['dev_worst'],
            global_norm_init=common.global_norm(model),
            runtime=runtime or common.runtime_metadata(device),
            final_ckpt=f'ckpts/{name}_final.pt')
        common.write_summary(out_dir / 'summary.json', summary)
        common.job_line(seed, name, steps, steps,
                        f'{final.get("dev_mean", float("nan")):.6f}', 'done',
                        elapsed=time.time() - job_started)
    summary['finished'] = True
    common.write_summary(out_dir / 'summary.json', summary)
    return summary


def factorial_worker(seed, config):
    """All jobs of one seed (runs inside a worker process)."""
    common.set_batch_log(config.get('batch_log'))
    run_seed_jobs(seed, Path(config['out_root']) / f'seed_{seed}', config['device'],
                  steps=config['steps'], eval_every=config['eval_every'],
                  families=tuple(config.get('families', ('factorial', 'obstacle', 'embedding'))),
                  cells=set(config['cells']) if config.get('cells') else None,
                  resume=bool(config.get('resume')),
                  runtime=common.runtime_metadata(config['device']))


def write_norm_table(runs_dir):
    """The component-norm table of every seed."""
    rows = []
    for seed_dir in sorted(runs_dir.glob('seed_*')):
        summary = common.read_summary(seed_dir / 'summary.json')
        norms = summary.get('component_norms') or {}
        for source in ('fresh_donor', 'memorized', 'grokked'):
            for component, value in (norms.get(source) or {}).items():
                rows.append(dict(seed=summary.get('seed', seed_dir.name.split('_')[-1]),
                                 source=source, component=component, norm=value))
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=['seed', 'source', 'component', 'norm'])
    writer.writeheader()
    writer.writerows(rows)
    path = runs_dir / 'component_norms.csv'
    common.atomic_bytes(path, out.getvalue().encode())
    return rows


def compare_reproduction(runs_dir):
    """Compare the factorial cells that equal Batch 3's keep_* jobs."""
    mapping = {'fact_attn': 'keep_attn', 'fact_mlp': 'keep_mlp',
               'fact_unembed': 'keep_unembed', 'fact_attn+mlp': 'keep_attn_mlp',
               'fact_mlp+unembed': 'keep_mlp_unembed',
               'fact_attn+unembed': 'keep_attn_unembed'}
    fields = ['train_acc', 'dev_acc_c0', 'dev_acc_c1', 'dev_acc_c2', 'dev_acc_c3',
              'dev_mean', 'dev_worst']
    rows = []
    for seed in range(5, 11):
        old = {r['init']: r for r in common.read_metrics(
            runs_dir / 'components' / f'seed_{seed}' / 'metrics.csv')
            if r['stage'] == 'transfer'}
        new = {r['init']: r for r in common.read_metrics(
            runs_dir / 'factorial' / f'seed_{seed}' / 'metrics.csv')
            if r['stage'] == 'transfer'}
        for cell, keep in mapping.items():
            old_rows = sorted((r for r in common.read_metrics(
                runs_dir / 'components' / f'seed_{seed}' / 'metrics.csv')
                if r['stage'] == 'transfer' and r['init'] == keep), key=lambda r: int(r['step']))
            new_rows = sorted((r for r in common.read_metrics(
                runs_dir / 'factorial' / f'seed_{seed}' / 'metrics.csv')
                if r['stage'] == 'transfer' and r['init'] == cell), key=lambda r: int(r['step']))
            if not old_rows or not new_rows:
                rows.append(dict(seed=seed, cell=cell, keep=keep, status='missing',
                                 rows_old=len(old_rows), rows_new=len(new_rows)))
                continue
            mismatches = 0
            for a, b in zip(old_rows, new_rows):
                if a['step'] != b['step']:
                    mismatches += 1
                    continue
                mismatches += sum(1 for f in fields if a[f] != b[f])
            rows.append(dict(seed=seed, cell=cell, keep=keep,
                             status='identical' if mismatches == 0 else 'differs',
                             rows_old=len(old_rows), rows_new=len(new_rows),
                             mismatched_fields=mismatches,
                             old_final=old_rows[-1]['dev_mean'],
                             new_final=new_rows[-1]['dev_mean']))
    out = io.StringIO(newline='')
    fields_out = ['seed', 'cell', 'keep', 'status', 'rows_old', 'rows_new',
                   'mismatched_fields', 'old_final', 'new_final']
    writer = csv.DictWriter(out, fieldnames=fields_out, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    path = runs_dir / 'factorial_reproduction.csv'
    common.atomic_bytes(path, out.getvalue().encode())
    return rows


def worker_config(runtime, out_root, steps, eval_every, resume, families, cells,
                  threads=1):
    """The configuration that the worker processes receive.

    `cells` is the parsed cell filter; it stays None when no filter was given.
    """
    return dict(out_root=str(out_root), device=runtime['device'], steps=steps,
                eval_every=eval_every, resume=resume,
                batch_log=str(common.BATCH_LOG), threads=threads,
                families=tuple(families),
                cells=set(parse_cells(cells)) if cells else None,
                runs_dir=str(common.PARENT / 'runs_nseed'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--procs', type=int, default=1)
    parser.add_argument('--out', default='runs_nseed/factorial')
    parser.add_argument('--device', default=None,
                        help='cpu, cuda or auto (default: cuda when available)')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--steps', type=int, default=TRANSFER_STEPS)
    parser.add_argument('--eval-every', type=int, default=EVAL_EVERY)
    parser.add_argument('--batch-log', default='runs_nseed/batch.log')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--jobs', nargs='+', default=['factorial', 'obstacle', 'embedding'],
                        choices=['factorial', 'obstacle', 'embedding'],
                        help='which job families to run')
    parser.add_argument('--cells', nargs='*', default=None,
                        help='restrict the factorial family to these cells, e.g. '
                             'empty attn mlp numeric_embed unembed attn+mlp attn+unembed '
                             'attn+numeric_embed attn+mlp+numeric_embed+unembed')
    args = parser.parse_args(argv)

    runtime = common.setup(args.threads, args.device)
    device = runtime['device']
    common.set_batch_log(project_path(args.batch_log))
    common.log_line(f'[factorial] device={device} deterministic={runtime["deterministic"]} '
                    f'steps={args.steps} eval_every={args.eval_every}')

    out_root = project_path(args.out)
    config = worker_config(runtime, out_root, args.steps, args.eval_every, args.resume,
                           args.jobs, args.cells, args.threads)
    for seed in args.seeds:
        common.planned_line(seed, [name for name, _, _ in
                                   job_plan(tuple(args.jobs), parse_cells(args.cells))])
    report = run_parallel(args.procs, args.seeds, factorial_worker, config)
    if not report.ok:
        common.log_line(f'[factorial] FAILED: {report.summary()}')
        return 1

    runs_dir = common.PARENT / 'runs_nseed'
    norms = write_norm_table(out_root)
    common.log_line(f'[factorial] wrote {len(norms)} component-norm rows to {out_root / "component_norms.csv"}')
    rows = compare_reproduction(runs_dir)
    identical = sum(1 for r in rows if r['status'] == 'identical')
    common.log_line(f'[factorial] reproduction check: {identical}/{len(rows)} factorial cells '
                    f'match the Batch 3 keep_* rows exactly')
    for row in rows:
        if row['status'] != 'identical':
            common.log_line(f'[factorial]   seed {row["seed"]} {row["cell"]} vs {row["keep"]}: '
                            f'{row["status"]} {row.get("mismatched_fields", "")}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
