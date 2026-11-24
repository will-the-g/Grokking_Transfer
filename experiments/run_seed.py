"""Per-seed driver for the multi-seed grokking-backbone experiment.

Spec: nseed_backbone/SPEC.md section 3 (phase 1) and section 6 (outputs).

    python nseed_backbone\\run_seed.py --seed 5 --out runs_nseed/seed_5

The script trains the modular-addition source, selects the stage checkpoints
(memorized / transition / grokked / late) and runs the tiered transfers. It is
resumable: the source state lives in `latest.pt`, every transfer keeps its own
state file, and `metrics.csv` is rewritten atomically after each evaluation.

The same three units are used by `run_batch.py`, which schedules them as jobs:
`source_job` trains the source, `stage_job` selects the stages from the saved
evaluation rows, and `transfer_job` runs one transfer from a stage checkpoint
that is read from disk.

The audit split is never read here.

Produces: Table 1 (p=59, one layer), Table 4 and Section 5.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
import torch  # noqa: E402  (common binds the MSVC runtime first)
from common import (GROK_CONSECUTIVE, GROK_DEV_ACC, GROK_TRAIN_ACC,  # noqa: E402
                    LATE_OFFSET, MEMORIZED_DEV_CEILING,
                    MEMORIZED_FRACTION_OF_GROK, RESUMABLE_CKPT_EVERY,
                    SOURCE_CAP, SOURCE_CKPT_EVERY, SOURCE_EVAL_EVERY,
                    TRANSITION_HI, TRANSITION_LO, TRANSITION_TARGET,
                    TRANSFER_EVAL_EVERY, TRANSFER_STEPS)


def project_path(path):
    """Resolve a possibly relative path against the project root."""
    path = Path(path)
    return path if path.is_absolute() else common.PARENT / path


def load_stage_model(state_dict, device='cpu'):
    """Build a model on `device` and load a CPU state dict into it."""
    model = common.make_model(common.CONFIG)
    model.load_state_dict(state_dict)
    return model.to(device)


def load_checkpoint_model(path, device='cpu'):
    payload = common.load_checkpoint(path)
    return load_stage_model(payload['model'], device), payload


def planned_transfers(status, stages, tiers):
    """The (task, init, tier) transfers implied by the stages and the tiers.

    Tier A: affine from {memorized, transition, grokked} plus the norm control.
    Tier B: scaled from {memorized, transition, grokked, norm_control}.
    Tier C: affine from the late stage.
    A nongrok seed runs affine from late_memorized only.
    """
    plan = []
    if 'A' in tiers:
        for init in ('memorized', 'transition', 'grokked'):
            if init in stages:
                plan.append(('affine', init, 'A'))
        if 'memorized' in stages and 'grokked' in stages:
            plan.append(('affine', 'norm_control', 'A'))
        if status == 'nongrok' and 'late_memorized' in stages:
            plan.append(('affine', 'late_memorized', 'A'))
    if 'B' in tiers:
        for init in ('memorized', 'transition', 'grokked'):
            if init in stages:
                plan.append(('scaled', init, 'B'))
        if 'memorized' in stages and 'grokked' in stages:
            plan.append(('scaled', 'norm_control', 'B'))
    if 'C' in tiers:
        if 'late' in stages:
            plan.append(('affine', 'late', 'C'))
    return plan


def run_transfer(seed, init_name, task, model, source_data, target_data, strata,
                 out_dir=None, steps=TRANSFER_STEPS, eval_every=TRANSFER_EVAL_EVERY,
                 trainable=None, ckpt_prefix='transfer', resume=False,
                 state_path=None, post_step=None, train_rows=None, log=print):
    """Train `task` starting from `model` for `steps` steps.

    A fresh AdamW is created for every transfer (spec section 1). The source
    task's clean-dev accuracy of the initialization is measured once and stored
    on every row as `stage_clean_acc`. `post_step` is called after every
    optimizer step (used by the phase-2 freeze study to hold frozen entries
    at their source values).

    Returns (rows, model, info).
    """
    optimizer = common.make_optimizer(model, trainable)
    if train_rows is None:
        train_rows = common.index_for(strata['train'])
    stage_clean = common.evaluate(model, source_data, strata)['dev_mean']

    rows = []
    step = 0
    elapsed = 0.0
    init_norm = common.global_norm(model)
    if resume and state_path is not None and Path(state_path).exists():
        payload, model, optimizer = common.restore_resumable(state_path, model, optimizer)
        step = int(payload.get('step', 0))
        elapsed = float(payload.get('elapsed', 0.0))
        rows = list(payload.get('metrics_rows', []))
        log(f'    resumed transfer {task}/{init_name} at step {step}')

    step_times = []
    started = time.time()
    generator = torch.Generator().manual_seed(int(seed) + 7919)
    batches = common.epoch_batches(train_rows, common.BATCH_SIZE, generator)
    batch_index = 0
    try:
        while step < steps:
            if batch_index >= len(batches):
                batches = common.epoch_batches(train_rows, common.BATCH_SIZE,
                                              generator)
                batch_index = 0
            batch = batches[batch_index]
            batch_index += 1
            tick = time.time()
            loss = common.train_step(model, optimizer, target_data, batch)
            if post_step is not None:
                post_step()
            step_times.append(time.time() - tick)
            step += 1
            if step % eval_every == 0 or step == steps:
                evaluation = common.evaluate(model, target_data, strata)
                window = step_times[-eval_every:]
                row = common.make_row(
                    seed, 'transfer', task, init_name, step, evaluation,
                    stage_clean_acc=stage_clean, norm=common.global_norm(model),
                    sec_per_step=sum(window) / len(window),
                    elapsed=elapsed + (time.time() - started))
                rows = common.merge_rows(rows, [row])
                if state_path is not None and (step % common.TRANSFER_CKPT_EVERY == 0
                                               or step == steps):
                    common.save_resumable(state_path, model, optimizer, step, rows,
                                          extra={'elapsed': elapsed + (time.time() - started),
                                                 'task': task, 'init': init_name})
                log(f'    [{task}/{init_name}] step {step}: dev_mean '
                    f'{evaluation["dev_mean"]:.4f} dev_worst {evaluation["dev_worst"]:.4f} '
                    f'loss {loss:.4f}')
    except KeyboardInterrupt:
        if state_path is not None:
            common.save_resumable(state_path, model, optimizer, step, rows,
                                  extra={'elapsed': elapsed + (time.time() - started),
                                         'task': task, 'init': init_name})
        log(f'    [{task}/{init_name}] interrupted at step {step}; state saved')
        raise
    info = dict(init_global_norm=init_norm, stage_clean_acc=stage_clean)
    return rows, model, info


class SeedRun:
    """One seed: source training, stage selection and transfers."""

    def __init__(self, seed, out_dir, threads=None, resume=False, tiers=('A', 'B', 'C'),
                 log=common.log_line, source_cap=SOURCE_CAP, eval_every=SOURCE_EVAL_EVERY,
                 ckpt_every=SOURCE_CKPT_EVERY, resumable_every=RESUMABLE_CKPT_EVERY,
                 transfer_steps=TRANSFER_STEPS, transfer_eval_every=TRANSFER_EVAL_EVERY,
                 late_offset=LATE_OFFSET, max_train_pairs=None, device=None,
                 init_seed=None):
        self.seed = int(seed)
        self.init_seed = self.seed if init_seed is None else int(init_seed)
        self.out = Path(out_dir)
        self.ckpts = self.out / 'ckpts'
        self.metrics_path = self.out / 'metrics.csv'
        self.summary_path = self.out / 'summary.json'
        self.norms_path = self.out / 'norms.csv'
        self.resume = bool(resume)
        self.tiers = tuple(tiers)
        self.log = log
        self.source_cap = int(source_cap)
        self.eval_every = int(eval_every)
        self.ckpt_every = int(ckpt_every)
        self.resumable_every = int(resumable_every)
        self.transfer_steps = int(transfer_steps)
        self.transfer_eval_every = int(transfer_eval_every)
        self.late_offset = int(late_offset)
        self.max_train_pairs = max_train_pairs
        self.runtime = common.setup(threads, device)
        self.device = self.runtime['device']

    # -- helpers ----------------------------------------------------------
    def _rows(self):
        return common.read_metrics(self.metrics_path)

    def _write_rows(self, rows):
        common.write_metrics(self.metrics_path, rows)

    def _train_rows(self, strata):
        """Training rows; `max_train_pairs` truncates the pair list for smoke tests."""
        ids = strata['train']
        if self.max_train_pairs is not None:
            ids = ids[:int(self.max_train_pairs)]
        return common.index_for(ids, self.device)

    def _write_summary(self, summary):
        common.write_summary(self.summary_path, summary)

    def _summary(self):
        summary = common.read_summary(self.summary_path)
        summary.setdefault('seed', self.seed)
        summary.setdefault('mask_seed', self.seed)
        summary.setdefault('config', common.CONFIG)
        summary.setdefault('coverage', common.COVERAGE)
        summary.setdefault('split_seed', common.SPLIT_SEED)
        summary.setdefault('runtime', self.runtime)
        summary['deterministic'] = common.determinism_enabled()
        summary['device'] = self.device
        return summary

    def _split(self):
        mask = common.mask_for_seed(self.seed)
        strata, exposure = common.split_for_mask(mask)
        return strata, exposure

    def _ensure_dirs(self):
        self.out.mkdir(parents=True, exist_ok=True)
        self.ckpts.mkdir(parents=True, exist_ok=True)

    # -- phase 1a: source -------------------------------------------------
    def train_source(self, strata, source_data, summary, snapshot_steps=None,
                     keep_training_to=None):
        """Train the modular-addition source until it groks or hits the cap.

        `snapshot_steps` maps a step number to a checkpoint file name; the model
        is saved at that step, which is how the age-matched stage point of the
        specificity study is captured.

        `keep_training_to` is a step the source trains to even when it groks
        earlier, so that a snapshot step beyond the grok step is reached. The
        grokked checkpoint is still written at the grok step.
        """
        snapshot_steps = dict(snapshot_steps or {})
        target = int(keep_training_to) if keep_training_to is not None else 0
        rows = self._rows()
        model = common.init_model(self.init_seed, self.device)
        optimizer = common.make_optimizer(model)
        step = 0
        consecutive = 0
        elapsed = 0.0
        latest = self.out / 'latest.pt'

        if self.resume and latest.exists():
            payload, model, optimizer = common.restore_resumable(latest, model, optimizer)
            step = int(payload.get('step', 0))
            consecutive = int(payload.get('consecutive', 0))
            elapsed = float(payload.get('elapsed', 0.0))
            self.log(f'  resumed source at step {step}')

        train_rows = self._train_rows(strata)
        step_times = []
        started = time.time()
        grok_step = summary.get('grok_step')
        status = summary.get('status')

        generator = torch.Generator().manual_seed(self.seed + 7919)
        batches = common.epoch_batches(train_rows, common.BATCH_SIZE, generator)
        batch_index = 0
        try:
            while step < self.source_cap and (status != 'grokked' or step < target):
                if batch_index >= len(batches):
                    batches = common.epoch_batches(train_rows, common.BATCH_SIZE,
                                                  generator)
                    batch_index = 0
                batch = batches[batch_index]
                batch_index += 1
                tick = time.time()
                loss = common.train_step(model, optimizer, source_data, batch)
                step_times.append(time.time() - tick)
                step += 1
                if step % self.eval_every == 0:
                    evaluation = common.evaluate(model, source_data, strata)
                    window = step_times[-self.eval_every:]
                    row = common.make_row(
                        self.seed, 'source', 'source', 'scratch', step, evaluation,
                        stage_clean_acc=evaluation['dev_mean'],
                        norm=common.global_norm(model),
                        sec_per_step=sum(window) / len(window),
                        elapsed=elapsed + (time.time() - started))
                    rows = common.merge_rows(rows, [row])
                    self._write_rows(rows)
                    if (evaluation['train_acc'] >= GROK_TRAIN_ACC
                            and evaluation['dev_worst'] >= GROK_DEV_ACC):
                        consecutive += 1
                    else:
                        consecutive = 0
                    self.log(f'  [source] step {step}: train {evaluation["train_acc"]:.4f} '
                             f'dev_mean {evaluation["dev_mean"]:.4f} '
                             f'dev_worst {evaluation["dev_worst"]:.4f} loss {loss:.4f}')
                    if consecutive >= GROK_CONSECUTIVE and status != 'grokked':
                        status = 'grokked'
                        grok_step = step
                        common.save_model_checkpoint(
                            self.ckpts / 'grokked.pt', model, step, row,
                            extra={'norms': common.per_module_norms(model),
                                   'global_norm': common.global_norm(model),
                                   'seed': self.seed, 'stage': 'grokked'})
                        self._append_norm('grokked', step, model)
                        self.log(f'  [source] GROKKED at step {step}')
                        consecutive = 0
                        if step >= target:
                            break
                if step in snapshot_steps:
                    common.save_model_checkpoint(
                        self.ckpts / snapshot_steps[step], model, step,
                        extra={'norms': common.per_module_norms(model),
                               'global_norm': common.global_norm(model),
                               'seed': self.seed, 'stage': 'source_snapshot'})
                    self.log(f'  [source] snapshot {snapshot_steps[step]} at step {step}')
                if step % self.ckpt_every == 0:
                    common.save_model_checkpoint(
                        self.ckpts / f'step_{step:07d}.pt', model, step,
                        extra={'norms': common.per_module_norms(model),
                               'global_norm': common.global_norm(model),
                               'seed': self.seed, 'stage': 'source'})
                    self._append_norm(f'step_{step:07d}', step, model)
                if step % self.resumable_every == 0:
                    common.save_resumable(latest, model, optimizer, step, rows,
                                          extra={'elapsed': elapsed + (time.time() - started),
                                                 'consecutive': consecutive,
                                                 'seed': self.seed,
                                                 'status': status})
        except KeyboardInterrupt:
            common.save_resumable(latest, model, optimizer, step, rows,
                                  extra={'elapsed': elapsed + (time.time() - started),
                                         'consecutive': consecutive,
                                         'seed': self.seed,
                                         'status': status})
            self.log(f'  [source] interrupted at step {step}; state saved to {latest}')
            raise
        elapsed += time.time() - started

        if status not in ('grokked', 'nongrok'):
            status = 'nongrok'
            common.save_model_checkpoint(
                self.ckpts / 'late_memorized.pt', model, step,
                extra={'norms': common.per_module_norms(model),
                       'global_norm': common.global_norm(model),
                       'seed': self.seed, 'stage': 'late_memorized'})
            self._append_norm('late_memorized', step, model)
            self.log(f'  [source] cap reached at step {step} without grokking -> nongrok')

        common.save_resumable(latest, model, optimizer, step, rows,
                              extra={'elapsed': elapsed, 'consecutive': consecutive,
                                     'seed': self.seed, 'status': status,
                                     'grok_step': grok_step})
        summary['status'] = status
        summary['grok_step'] = grok_step
        summary['source_steps_run'] = step
        summary['source_elapsed_s'] = elapsed
        self._write_summary(summary)
        return status, grok_step, rows

    # -- norms ------------------------------------------------------------
    def _append_norm(self, name, step, model):
        payload = dict(seed=self.seed, name=name, step=int(step),
                       global_norm=common.global_norm(model),
                       module_norms=json.dumps(common.per_module_norms(model)))
        existing = [r for r in _read_norms(self.norms_path)
                    if not (r['name'] == name and int(r['step']) == int(step))]
        existing.append(payload)
        common.write_metrics(self.norms_path, existing,
                              fields=['seed', 'name', 'step', 'global_norm', 'module_norms'])

    # -- phase 1b: stage selection ---------------------------------------
    def select_stages(self, rows, grok_step, status):
        """Label checkpoints from the saved evaluation rows only."""
        source_rows = [r for r in rows if r['stage'] == 'source']
        source_rows.sort(key=lambda r: int(r['step']))
        available = {}
        for path in sorted(self.ckpts.glob('step_*.pt')):
            available[int(path.stem.split('_')[-1])] = path.name
        stages = {}

        def stage_record(name, row, ckpt_name, step):
            row = row or {}
            return dict(name=name, step=int(step), ckpt=ckpt_name,
                        train_acc=common.as_float(row.get('train_acc')),
                        dev_mean=common.as_float(row.get('dev_mean')),
                        dev_worst=common.as_float(row.get('dev_worst')),
                        global_norm=common.as_float(row.get('global_norm')))

        if status == 'grokked' and (self.ckpts / 'grokked.pt').exists():
            payload = common.load_checkpoint(self.ckpts / 'grokked.pt')
            row = payload.get('metrics') or {}
            record = stage_record('grokked', row, 'grokked.pt', payload.get('step', grok_step))
            record['global_norm'] = payload.get('global_norm', record['global_norm'])
            stages['grokked'] = record

        # memorized: at/after the first memorization, dev < 0.03, nearest 40% of grok
        first_memorized = None
        for row in source_rows:
            train_acc = common.as_float(row['train_acc'])
            if train_acc is not None and train_acc >= GROK_TRAIN_ACC:
                first_memorized = common.as_int(row['step'])
                break
        if first_memorized is not None and grok_step:
            target = MEMORIZED_FRACTION_OF_GROK * grok_step
            candidates = [r for r in source_rows
                          if common.as_int(r['step']) >= first_memorized
                          and (common.as_float(r['dev_worst']) or 0.0) < MEMORIZED_DEV_CEILING
                          and common.as_int(r['step']) in available]
            if candidates:
                best = min(candidates, key=lambda r: abs(common.as_int(r['step']) - target))
                stages['memorized'] = stage_record(
                    'memorized', best, available[common.as_int(best['step'])],
                    common.as_int(best['step']))
            else:
                self.log('  [stages] no memorized checkpoint found')
        else:
            self.log('  [stages] no memorization point found')

        # transition: mean clean-dev closest to 0.5 inside [0.25, 0.75]
        candidates = []
        for row in source_rows:
            dev_mean = common.as_float(row['dev_mean'])
            if dev_mean is None or not (TRANSITION_LO <= dev_mean <= TRANSITION_HI):
                continue
            if common.as_int(row['step']) not in available:
                continue
            candidates.append(row)
        if candidates:
            best = min(candidates,
                       key=lambda r: abs((common.as_float(r['dev_mean']) or 0.0) - TRANSITION_TARGET))
            stages['transition'] = stage_record(
                'transition', best, available[common.as_int(best['step'])],
                common.as_int(best['step']))
        else:
            self.log('  [stages] transition checkpoint missing')

        if status == 'nongrok' and (self.ckpts / 'late_memorized.pt').exists():
            payload = common.load_checkpoint(self.ckpts / 'late_memorized.pt')
            row = payload.get('metrics') or {}
            stages['late_memorized'] = stage_record(
                'late_memorized', row, 'late_memorized.pt', payload.get('step', 0))
            stages['late_memorized']['global_norm'] = payload.get(
                'global_norm', stages['late_memorized']['global_norm'])
        return stages

    # -- tier C: late stage ----------------------------------------------
    def build_late_stage(self, source_data, strata, summary):
        """Continue the source for LATE_OFFSET steps past grok (tier C only)."""
        grok_path = self.ckpts / 'grokked.pt'
        if not grok_path.exists():
            return None
        payload = common.load_checkpoint(grok_path)
        model = load_stage_model(payload['model'], self.device)
        optimizer = common.make_optimizer(model)
        train_rows = self._train_rows(strata)
        step_times = []
        started = time.time()
        for step in range(1, self.late_offset + 1):
            tick = time.time()
            common.train_step(model, optimizer, source_data, train_rows)
            step_times.append(time.time() - tick)
            if step % self.eval_every == 0:
                evaluation = common.evaluate(model, source_data, strata)
                self.log(f'  [late] step {step}: train {evaluation["train_acc"]:.4f} '
                         f'dev_mean {evaluation["dev_mean"]:.4f}')
        evaluation = common.evaluate(model, source_data, strata)
        row = common.make_row(self.seed, 'source', 'source', 'late', self.late_offset,
                              evaluation,
                              stage_clean_acc=evaluation['dev_mean'],
                              norm=common.global_norm(model),
                              sec_per_step=sum(step_times) / len(step_times),
                              elapsed=time.time() - started)
        common.save_model_checkpoint(
            self.ckpts / 'late.pt', model, self.late_offset, row,
            extra={'norms': common.per_module_norms(model),
                   'global_norm': common.global_norm(model),
                   'seed': self.seed, 'stage': 'late'})
        self._append_norm('late', self.late_offset, model)
        return dict(name='late', step=self.late_offset, ckpt='late.pt',
                    train_acc=evaluation['train_acc'], dev_mean=evaluation['dev_mean'],
                    dev_worst=evaluation['dev_worst'],
                    global_norm=common.global_norm(model))

    # -- jobs -------------------------------------------------------------
    def source_job(self):
        self._ensure_dirs()
        """Train the source of this seed and select its stages."""
        summary = self._summary()
        strata, exposure = self._split()
        summary['split_sizes'] = {key: len(value) for key, value in strata.items()}
        source_data = common.build_task('source', self.device)
        status, grok_step, rows = self.train_source(strata, source_data, summary)
        stages = self.select_stages(rows, grok_step, status)
        summary['stages'] = stages
        summary['source_finished'] = True
        self._write_summary(summary)
        self.log(f'seed {self.seed}: status {status}; stages {sorted(stages)}')
        return dict(status=status, grok_step=grok_step, stages=stages)

    def stage_job(self):
        self._ensure_dirs()
        """Select the stages from the saved evaluation rows; no training."""
        summary = self._summary()
        rows = self._rows()
        status = summary.get('status')
        grok_step = summary.get('grok_step')
        stages = self.select_stages(rows, grok_step, status)
        summary['stages'] = stages
        self._write_summary(summary)
        return dict(status=status, grok_step=grok_step, stages=stages)

    def transfer_job(self, task, init_name):
        self._ensure_dirs()
        """Run one transfer from a stage checkpoint read from disk."""
        summary = self._summary()
        stages = summary.get('stages', {}) or {}
        rows = self._rows()
        strata, exposure = self._split()
        source_data = common.build_task('source', self.device)
        transfers = summary.setdefault('transfers', {})

        key = f'{task}|{init_name}'
        if transfers.get(key, {}).get('done'):
            self.log(f'  [transfer] {key} already done, skipping')
            return transfers[key]

        ckpt_name, target_norm = self._stage_checkpoint(stages, init_name)
        if ckpt_name is None:
            raise RuntimeError(f'seed {self.seed}: no checkpoint for stage {init_name!r}')
        model, _ = load_checkpoint_model(self.ckpts / ckpt_name, self.device)
        if init_name == 'norm_control':
            scale = common.rescale_to_norm(model, target_norm)
            self.log(f'  [norm_control] scale {scale:.6f} -> global norm '
                     f'{common.global_norm(model):.6f} (target {target_norm:.6f})')

        target_data = common.build_task(task, self.device)
        state_path = self.ckpts / f'transfer_{task}_{init_name}_state.pt'
        final_path = self.ckpts / f'transfer_{task}_{init_name}_final.pt'
        transfer_rows, trained, info = run_transfer(
            self.seed, init_name, task, model, source_data, target_data, strata,
            out_dir=self.out, steps=self.transfer_steps,
            eval_every=self.transfer_eval_every,
            train_rows=self._train_rows(strata),
            resume=self.resume, state_path=state_path, log=self.log)
        rows = common.merge_rows(rows, transfer_rows)
        self._write_rows(rows)
        common.save_model_checkpoint(
            final_path, trained, self.transfer_steps,
            transfer_rows[-1] if transfer_rows else None,
            extra={'seed': self.seed, 'task': task, 'init': init_name,
                   'stage': 'transfer', 'global_norm': common.global_norm(trained)})
        final_row = transfer_rows[-1] if transfer_rows else {}
        transfers[key] = dict(
            task=task, init=init_name, done=True, steps=self.transfer_steps,
            final_dev_mean=common.as_float(final_row.get('dev_mean')),
            final_dev_worst=common.as_float(final_row.get('dev_worst')),
            final_train_acc=common.as_float(final_row.get('train_acc')),
            stage_clean_acc=info['stage_clean_acc'],
            global_norm_init=info['init_global_norm'],
            steps_to_30=_steps_to_threshold(transfer_rows, 0.30),
            final_ckpt=str(final_path.relative_to(self.out)).replace('\\', '/'))
        summary['transfers'] = transfers
        self._write_summary(summary)
        return transfers[key]

    def late_job(self):
        self._ensure_dirs()
        """Build the late stage (tier C) and register it in the summary."""
        summary = self._summary()
        strata, exposure = self._split()
        source_data = common.build_task('source', self.device)
        stages = summary.setdefault('stages', {})
        if 'late' in stages:
            return stages
        late = self.build_late_stage(source_data, strata, summary)
        if late:
            stages['late'] = late
            summary['stages'] = stages
            self._write_summary(summary)
        return stages

    def _stage_checkpoint(self, stages, init_name):
        """The checkpoint file and the target norm for a stage name."""
        if init_name == 'norm_control':
            memorized = stages.get('memorized')
            grokked = stages.get('grokked')
            if not memorized or not grokked:
                return None, None
            return memorized['ckpt'], grokked['global_norm']
        stage = stages.get(init_name)
        if not stage:
            return None, None
        return stage['ckpt'], None

    # -- driver -----------------------------------------------------------
    def run(self):
        """Source, stages and every planned transfer, in tier order."""
        self.out.mkdir(parents=True, exist_ok=True)
        self.ckpts.mkdir(parents=True, exist_ok=True)
        summary = self._summary()
        strata, exposure = self._split()
        summary['split_sizes'] = {key: len(value) for key, value in strata.items()}
        self.log(f'seed {self.seed}: split {summary["split_sizes"]}')
        source_data = common.build_task('source', self.device)
        status, grok_step, rows = self.train_source(strata, source_data, summary)
        stages = self.select_stages(rows, grok_step, status)
        summary['stages'] = stages
        self._write_summary(summary)
        self.log(f'seed {self.seed}: status {status}; stages {sorted(stages)}')

        for tier in ('A', 'B', 'C'):
            if tier not in self.tiers:
                continue
            if tier == 'C' and 'late' not in stages:
                late = self.build_late_stage(source_data, strata, summary)
                if late:
                    stages['late'] = late
                    summary['stages'] = stages
                    self._write_summary(summary)
            for task, init_name, job_tier in planned_transfers(status, stages, (tier,)):
                self.transfer_job(task, init_name)
            summary = self._summary()
            if tier not in summary.setdefault('tiers_done', []):
                summary['tiers_done'].append(tier)
                self._write_summary(summary)
        summary['finished'] = True
        self._write_summary(summary)
        return summary


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _read_norms(path):
    import csv
    path = Path(path)
    if not path.exists():
        return []
    with path.open('r', newline='') as stream:
        return list(csv.DictReader(stream))


def _steps_to_threshold(rows, threshold):
    """First transfer step whose dev_mean reaches `threshold` (None if never)."""
    for row in sorted(rows, key=lambda r: int(r['step'])):
        value = common.as_float(row.get('dev_mean'))
        if value is not None and value >= threshold:
            return int(row['step'])
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', default=None,
                        help='output folder (default runs_nseed/seed_<seed>)')
    parser.add_argument('--threads', type=int, default=None)
    parser.add_argument('--device', default=None,
                        help='cpu, cuda or auto (default: cuda when available)')
    parser.add_argument('--tiers', nargs='+', default=['A', 'B', 'C'],
                        choices=['A', 'B', 'C'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--source-cap', type=int, default=SOURCE_CAP,
                        help='test-only override of the 20000-step source cap')
    parser.add_argument('--eval-every', type=int, default=SOURCE_EVAL_EVERY)
    parser.add_argument('--ckpt-every', type=int, default=SOURCE_CKPT_EVERY)
    parser.add_argument('--resumable-every', type=int, default=RESUMABLE_CKPT_EVERY)
    parser.add_argument('--transfer-steps', type=int, default=TRANSFER_STEPS)
    parser.add_argument('--transfer-eval-every', type=int, default=TRANSFER_EVAL_EVERY)
    parser.add_argument('--late-offset', type=int, default=LATE_OFFSET)
    parser.add_argument('--max-train-pairs', type=int, default=None,
                        help='test-only cap on the number of training pairs per context')
    parser.add_argument('--init-seed', type=int, default=None,
                        help='initialization seed of the source (default: the seed itself; '
                             'the training mask always uses the seed)')
    args = parser.parse_args(argv)

    out = project_path(args.out) if args.out else common.PARENT / 'runs_nseed' / f'seed_{args.seed}'
    run = SeedRun(args.seed, out, threads=args.threads, resume=args.resume,
                  tiers=tuple(args.tiers), device=args.device, source_cap=args.source_cap,
                  eval_every=args.eval_every, ckpt_every=args.ckpt_every,
                  resumable_every=args.resumable_every,
                  transfer_steps=args.transfer_steps,
                  transfer_eval_every=args.transfer_eval_every,
                  late_offset=args.late_offset,
                  max_train_pairs=args.max_train_pairs,
                  init_seed=args.init_seed)
    run.run()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
