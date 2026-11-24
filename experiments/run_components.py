"""Component study: which parts of a grokked network carry the reusable structure?

Every job builds a hybrid model, records the SOURCE-task accuracy (train and
clean dev) at step 0, and then runs a 2,000-step affine transfer with all
parameters trainable and a fresh AdamW, evaluated every 100 steps exactly like
a Tier A transfer. The final checkpoint of every job is saved.

Necessity (recipient = the seed's grokked checkpoint, donor = a fresh model
from `core.make_model` after `torch.manual_seed(70000 + seed)`): the named
component is reset to the donor's values.

    reset_numeric_embed  reset_other_embed  reset_attn  reset_mlp  reset_unembed

Sufficiency (recipient = the seed's memorized checkpoint, donor = its grokked
checkpoint): the named components are taken from the donor.

    keep_attn  keep_mlp  keep_unembed  keep_attn_mlp  keep_mlp_unembed
    keep_attn_unembed

Component names: numeric_embed = token_embed.weight[:p]; other_embed =
token_embed.weight[p:] and pos_embed; attn = all parameters of model.attn;
mlp = mlp_in and mlp_out; unembed = unembed.weight.

    python nseed_backbone\\run_components.py --seeds 5 6 7 8 9 10 --device cuda

Produces: Table 4 (the component hybrids).
"""
from __future__ import annotations

import argparse
import json
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

RESET_JOBS = tuple(f'reset_{name}' for name in common.COMPONENTS)
KEEP_JOBS = ('keep_attn', 'keep_mlp', 'keep_unembed', 'keep_attn_mlp',
             'keep_mlp_unembed', 'keep_attn_unembed')

TRANSFER_STEPS = 2000
EVAL_EVERY = 100
DONOR_SEED_BASE = 70000


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else common.PARENT / path


def components_of(job_name):
    """The component list a job copies from the donor."""
    if job_name.startswith('reset_'):
        return [job_name[len('reset_'):]]
    if job_name.startswith('keep_'):
        return job_name[len('keep_'):].split('_')
    raise ValueError(f'Unknown job {job_name!r}')


def fresh_donor_state(seed):
    """A fresh model state from `core.make_model` after the prescribed seed."""
    torch.manual_seed(DONOR_SEED_BASE + int(seed))
    model = common.make_model(common.CONFIG)
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def source_accuracy(model, source_data, strata):
    evaluation = common.evaluate(model, source_data, strata)
    return evaluation


def run_transfer_job(job_name, seed, model, target_data, strata, out_dir,
                     steps=TRANSFER_STEPS, eval_every=EVAL_EVERY, log=common.log_line,
                     train_rows=None):
    """One affine transfer with all parameters trainable and a fresh AdamW."""
    optimizer = common.make_optimizer(model)
    if train_rows is None:
        train_rows = common.index_for(strata['train'], model.token_embed.weight.device)
    rows = []
    step_times = []
    started = time.time()
    step = 0
    generator = torch.Generator().manual_seed(int(seed) + 7919)
    batches = common.epoch_batches(train_rows, common.BATCH_SIZE, generator)
    batch_index = 0
    while step < steps:
        if batch_index >= len(batches):
            batches = common.epoch_batches(train_rows, common.BATCH_SIZE,
                                           generator)
            batch_index = 0
        batch = batches[batch_index]
        batch_index += 1
        tick = time.time()
        optimizer.zero_grad(set_to_none=True)
        logits = model(target_data['x'][batch])
        loss = torch.nn.functional.cross_entropy(logits, target_data['y'][batch])
        loss.backward()
        optimizer.step()
        step_times.append(time.time() - tick)
        step += 1
        if step % eval_every == 0 or step == steps:
            evaluation = common.evaluate(model, target_data, strata)
            window = step_times[-eval_every:]
            row = common.make_row(seed, 'transfer', 'affine', job_name, step, evaluation,
                                  stage_clean_acc=evaluation['dev_mean'],
                                  norm=common.global_norm(model),
                                  sec_per_step=sum(window) / len(window),
                                  elapsed=time.time() - started)
            rows = common.merge_rows(rows, [row])
            common.job_line(seed, job_name, step, steps, f'{evaluation["dev_mean"]:.6f}', 'run')
    return rows, model


def run_seed_jobs(seed, out_dir, device, steps=TRANSFER_STEPS, eval_every=EVAL_EVERY,
                  resume=False, runtime=None, log=common.log_line):
    """All eleven component jobs for one seed."""
    seed_dir = common.PARENT / 'runs_nseed' / f'seed_{seed}'
    summary_path = common.read_summary(seed_dir / 'summary.json')
    stages = summary_path.get('stages', {}) or {}
    if 'grokked' not in stages or 'memorized' not in stages:
        log(f'[components] seed {seed}: no grokked/memorized stage, skipped')
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

    for job_name in list(RESET_JOBS) + list(KEEP_JOBS):
        if resume and jobs.get(job_name, {}).get('done'):
            log(f'  [{job_name}] already done, skipping')
            common.skipped_job_line(seed, job_name, steps, jobs[job_name])
            continue
        if job_name.startswith('reset_'):
            recipient, job_donor = grokked, donor
        else:
            recipient, job_donor = memorized, grokked
        merged = common.build_hybrid(recipient, job_donor, components_of(job_name))
        model = common.make_model(common.CONFIG)
        model.load_state_dict(merged)
        model = model.to(device)

        evaluation = source_accuracy(model, source_data, strata)
        job_started = time.time()
        common.job_line(seed, job_name, 0, steps, f'{evaluation["dev_mean"]:.6f}', 'start',
                        elapsed=0.0)
        log(f'  [{job_name}] step 0 source: train {evaluation["train_acc"]:.4f} '
            f'dev {evaluation["dev_mean"]:.4f}')

        rows, trained = run_transfer_job(job_name, seed, model, target_data, strata,
                                          out_dir, steps=steps, eval_every=eval_every,
                                          log=log)
        common.write_metrics(out_dir / 'metrics.csv', rows)
        common.save_model_checkpoint(out_dir / 'ckpts' / f'{job_name}_final.pt', trained,
                                     steps, rows[-1] if rows else None,
                                     extra={'seed': seed, 'task': 'affine', 'init': job_name,
                                            'stage': 'components',
                                            'global_norm': common.global_norm(trained)})
        final = rows[-1] if rows else {}
        jobs[job_name] = dict(
            name=job_name, kind='component', task='affine', seed=seed, done=True,
            steps=steps, components=components_of(job_name),
            recipient='grokked' if job_name.startswith('reset_') else 'memorized',
            donor='fresh' if job_name.startswith('reset_') else 'grokked',
            final_dev_mean=common.as_float(final.get('dev_mean')),
            final_dev_worst=common.as_float(final.get('dev_worst')),
            final_train_acc=common.as_float(final.get('train_acc')),
            best_dev_mean=max((common.as_float(r['dev_mean']) or 0.0) for r in rows) if rows else None,
            source_train_acc=evaluation['train_acc'],
            source_dev_mean=evaluation['dev_mean'],
            source_dev_worst=evaluation['dev_worst'],
            global_norm_init=common.global_norm(model),
            runtime=runtime or common.runtime_metadata(device),
            final_ckpt=f'ckpts/{job_name}_final.pt')
        common.write_summary(out_dir / 'summary.json', summary)
        common.job_line(seed, job_name, steps, steps,
                        f'{final.get("dev_mean", float("nan")):.6f}', 'done',
                        elapsed=time.time() - job_started)
    summary['finished'] = True
    common.write_summary(out_dir / 'summary.json', summary)
    return summary


def component_worker(seed, config):
    """All component jobs of one seed (runs inside a worker process)."""
    common.set_batch_log(config.get('batch_log'))
    run_seed_jobs(seed, Path(config['out_root']) / f'seed_{seed}', config['device'],
                  steps=config['steps'], eval_every=config['eval_every'],
                  resume=config.get('resume', False),
                  runtime=common.runtime_metadata(config['device']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--procs', type=int, default=1,
                        help='number of single-thread worker processes')
    parser.add_argument('--out', default='runs_nseed/components')
    parser.add_argument('--device', default=None,
                        help='cpu, cuda or auto (default: cuda when available)')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--steps', type=int, default=TRANSFER_STEPS)
    parser.add_argument('--eval-every', type=int, default=EVAL_EVERY)
    parser.add_argument('--batch-log', default='runs_nseed/batch.log')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)

    runtime = common.setup(args.threads, args.device)
    device = runtime['device']
    common.set_batch_log(project_path(args.batch_log))
    common.log_line(f'[components] device={device} deterministic={runtime["deterministic"]} '
                    f'steps={args.steps} eval_every={args.eval_every}')

    out_root = project_path(args.out)
    config = dict(out_root=str(out_root), device=device, steps=args.steps,
                  eval_every=args.eval_every, resume=args.resume,
                  batch_log=str(common.BATCH_LOG), threads=args.threads,
                  runs_dir=str(common.PARENT / 'runs_nseed'))
    for seed in args.seeds:
        common.planned_line(seed, list(RESET_JOBS) + list(KEEP_JOBS))
    report = run_parallel(args.procs, args.seeds, component_worker, config)
    if not report.ok:
        common.log_line(f'[components] FAILED: {report.summary()}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
