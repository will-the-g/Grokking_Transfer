"""Shared task, provenance, and checkpoint utilities for the first A/B pilot.

Produces: the shared task, the provenance and the checkpoint utilities.
"""
import csv
import hashlib
import io
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from model_phase3 import ContextualTransformer

HERE = Path(__file__).resolve().parent
FORMAT = 2


def symmetry_split(parent_mask, p, seed):
    """Primary holdouts exclude either ordering in ANY source context.

    Dev/audit assignment is by unordered input orbit, even for asymmetric
    downstream rules. This is a conservative exposure control, not a claim
    that difference/revision labels are invariant under swapping inputs.
    """
    ids = torch.arange(p * p)
    transpose = (ids % p) * p + ids // p
    exposed = parent_mask.any(0)
    clean = ~(exposed | exposed[transpose])
    orbit = torch.minimum(ids, transpose)
    clean_orbits = torch.unique(orbit[clean], sorted=True)
    if len(clean_orbits) < 2:
        raise ValueError('Fewer than two globally symmetry-clean input orbits.')
    g = torch.Generator().manual_seed(seed)
    shuffled = clean_orbits[torch.randperm(len(clean_orbits), generator=g)]
    dev_mask = clean & torch.isin(orbit, shuffled[:len(shuffled) // 2])
    audit_mask = clean & ~dev_mask
    original_holdout = ~parent_mask[0]
    strata = dict(train=torch.where(parent_mask[0])[0],
                  dev=torch.where(dev_mask)[0], audit=torch.where(audit_mask)[0],
                  exposed=torch.where(original_holdout & ~clean)[0])
    return strata, dict(pair_ids=ids, transpose_ids=transpose, orbit_ids=orbit,
                       global_exact_exposed=exposed, global_clean=clean)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def code_identity():
    return {p.name: digest(p) for p in sorted(HERE.glob('*.py'))
            if not p.name.startswith('test_')}


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path, payload):
    atomic_bytes(path, (json.dumps(payload, indent=2, allow_nan=False) + '\n').encode())


def save_checkpoint(path, payload):
    # Serialize to a file, not RAM, so large optimizer states do not double RAM.
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_checkpoint(path):
    # Full local experiment checkpoints contain Python/NumPy RNG state.
    # Only open checkpoints whose producer you trust.
    return torch.load(path, map_location='cpu', weights_only=False)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])


def model_config(cfg):
    config = {k: int(cfg[k]) for k in ['p', 'n_contexts', 'd_model', 'n_heads', 'd_mlp']}
    config['n_layers'] = int(cfg.get('n_layers', 1))
    config['preln'] = int(cfg.get('preln', 0))
    return config


def make_model(cfg):
    return ContextualTransformer(**model_config(cfg))


def validate_parent(state):
    cfg = state['config']
    spec = state.get('task_spec', {})
    coeffs = spec.get('coefficients', state.get('coefficients'))
    shifts = spec.get('shifts', state.get('shifts', [0] * cfg['n_contexts']))
    if coeffs is None and state.get('condition') == 'shared':
        coeffs = [[1, 1]] * cfg['n_contexts']
    if (coeffs is None or len(coeffs) != cfg['n_contexts']
            or any(list(c) != [1, 1] for c in coeffs)
            or len(shifts) != cfg['n_contexts'] or any(int(s) != 0 for s in shifts)):
        raise ValueError('Pilot requires identical addition in every parent context. '
                         'Shifted/scaled/affine parents need a separate design.')
    mask = state['pair_train_mask'].cpu().bool()
    if tuple(mask.shape) != (cfg['n_contexts'], cfg['p'] ** 2):
        raise ValueError('Parent pair_train_mask has the wrong shape.')
    if cfg['n_contexts'] < 2:
        raise ValueError('Pilot needs at least two existing contexts.')
    if cfg['p'] < 5:
        raise ValueError('Use p >= 5 for meaningful revision subsets.')
    return cfg, mask


def build_data(cfg, parent_mask, job, settings):
    p = cfg['p']
    pairs = torch.cartesian_prod(torch.arange(p), torch.arange(p))
    a, b = pairs.T
    source_y = (a + b) % p
    kind = job['task']
    if kind == 'addition_shift':
        target_y = (source_y + settings['shift']) % p
        context = 1
    elif kind == 'addition_difference':
        target_y = (a - b) % p
        context = 1
    elif kind in ('revision', 'addition_local'):
        target_y = (source_y + settings['revision_delta'] *
                    (a < settings['revision_cutoff'])) % p
        context = 0 if kind == 'revision' else 1
    elif kind == 'source_control':
        target_y = source_y.clone()
        context = 0
    else:
        raise ValueError(f'Unknown task {kind}')

    def tokens(c):
        return torch.stack([torch.full_like(a, p + c), a, b,
                            torch.full_like(a, p + cfg['n_contexts'])], dim=1)
    x = torch.cat([tokens(0), tokens(context)])
    y = torch.cat([source_y, target_y])
    n = p * p
    strata, exposure = symmetry_split(parent_mask, p, settings['split_seed'])
    source_train = strata['train']
    if min(len(strata[s]) for s in ('train', 'dev', 'audit')) == 0:
        raise ValueError('Empty source training, development, or audit split.')
    changed = source_y != target_y
    groups = {}
    for split, ids in strata.items():
        if not len(ids):
            continue
        groups[f'source/{split}/all'] = ids
        groups[f'target/{split}/all'] = ids + n
        if kind in ('revision', 'addition_local'):
            for name, mask in [('changed', changed), ('unchanged', ~changed)]:
                subset = ids[mask[ids]]
                if not len(subset) and split != 'exposed':
                    raise ValueError(f'Empty {split} {name} revision subset.')
                if not len(subset):
                    continue
                groups[f'source/{split}/{name}'] = subset
                groups[f'target/{split}/{name}'] = subset + n
    replay_train = source_train[~changed[source_train]] if kind == 'revision' else source_train
    return dict(x=x, y=y, groups=groups, source_train=source_train, replay_train=replay_train,
                target_train=source_train + n, changed=changed, pairs=pairs,
                target_context=context, strata=strata, exposure=exposure)


@torch.no_grad()
def audit_source(state, split_seed=1701, batch_size=512):
    """Audit development inputs only; never inspect the adaptation audit half."""
    cfg, mask = validate_parent(state)
    strata, exposure = symmetry_split(mask, cfg['p'], split_seed)
    model = make_model(cfg)
    model.load_state_dict(state['model'])
    model.eval()
    p = cfg['p']
    pairs = torch.cartesian_prod(torch.arange(p), torch.arange(p))
    labels = pairs.sum(1) % p
    result = dict(step=int(state['step']), clean_dev_count=len(strata['dev']),
                  clean_audit_count=len(strata['audit']), exposed_count=len(strata['exposed']),
                  clean_dev_context_accuracy=[], train_context_accuracy=[])
    for context in range(cfg['n_contexts']):
        x = torch.stack([torch.full_like(labels, p + context), pairs[:, 0], pairs[:, 1],
                         torch.full_like(labels, p + cfg['n_contexts'])], dim=1)
        for key, ids in [('clean_dev', strata['dev']),
                         ('train', torch.where(mask[context])[0])]:
            if not len(ids):
                raise ValueError(f'Empty {key} for context {context}')
            correct = 0
            for chunk in ids.split(batch_size):
                logits = model(x[chunk])
                if not torch.isfinite(logits).all():
                    raise FloatingPointError('Non-finite source audit logits.')
                correct += int((logits.argmax(-1) == labels[chunk]).sum())
            result[key + '_context_accuracy'].append(correct / len(ids))
    result['clean_dev_min_accuracy'] = min(result['clean_dev_context_accuracy'])
    result['train_min_accuracy'] = min(result['train_context_accuracy'])
    result['stage'] = ('generalized' if result['clean_dev_min_accuracy'] >= .90 else
                       'memorizing' if result['train_min_accuracy'] >= .99
                       and result['clean_dev_min_accuracy'] <= .10 else 'intermediate')
    return result


@torch.no_grad()
def evaluate(model, data, split, batch_size):
    model.eval()
    rows = []
    for key, ids in data['groups'].items():
        task, current_split, subset = key.split('/')
        if current_split != split:
            continue
        loss, correct, margins = 0., 0, 0.
        for chunk in ids.split(batch_size):
            logits = model(data['x'][chunk])
            target = data['y'][chunk]
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Non-finite evaluation logits.')
            loss += float(torch.nn.functional.cross_entropy(logits, target, reduction='sum'))
            correct += int((logits.argmax(-1) == target).sum())
            true = logits.gather(1, target[:, None]).squeeze(1)
            other = logits.clone()
            other.scatter_(1, target[:, None], float('-inf'))
            margins += float((true - other.max(1).values).sum())
        rows.append(dict(task=task, split=split, subset=subset, n=len(ids),
                         loss=loss / len(ids), accuracy=correct / len(ids),
                         margin=margins / len(ids)))
    return rows


def rows_bytes(rows, fields):
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode()


def export_metrics(path, rows):
    if rows:
        atomic_bytes(path, rows_bytes(rows, list(rows[0])))


class RunLock:
    """Exclusive run ownership. A crash lock requires explicit review/removal."""
    def __init__(self, folder):
        self.path = Path(folder) / 'RUNNING.lock'
    def __enter__(self):
        with self.path.open('x') as stream:
            stream.write(json.dumps(dict(pid=os.getpid(), started=time.time())))
        return self
    def __exit__(self, *args):
        self.path.unlink(missing_ok=True)


def make_optimizer(model, settings):
    return torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'],
                             weight_decay=settings['weight_decay'],
                             betas=tuple(settings['betas']))
