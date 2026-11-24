"""Shared utilities for the multi-seed grokking-backbone experiment.

Everything scientific lives here: data construction, labels, evaluation,
checkpointing and resume. The per-seed driver (`run_seed.py`), the scratch
baselines (`run_scratch.py`), the offline Fourier metrics
(`fourier_metrics.py`), the phase-2 freeze study (`run_freeze.py`) and the
analysis (`analyze.py`) all import from this module.

The parent project supplies `core.make_model`, `core.model_config`,
`core.symmetry_split` and `core.save_checkpoint` / `core.load_checkpoint`.
Nothing in the parent project is modified.

Spec: nseed_backbone/SPEC.md (sections 1, 2, 3, 6).

Produces: the shared settings and helpers of every experiment.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path


def _bind_msvc_runtime():
    """Bind the newest MSVC runtime before torch is imported (Windows only).

    This project runs from a venv created from an Anaconda base. The base
    directory ships an old msvcp140.dll (14.27) and python311.dll is loaded from
    there, so the old runtime is bound into the process. torch's c10.dll is
    built against a newer MSVC and faults inside the old runtime while it
    initializes ("DLL initialization routine failed"). Preloading the System32
    copies keeps the newer runtime bound.
    """
    if sys.platform != 'win32':
        return
    import ctypes
    system_root = os.environ.get('SystemRoot') or os.path.join(os.sep, 'Windows')
    for name in ('msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll'):
        try:
            ctypes.WinDLL(os.path.join(system_root, 'System32', name))
        except OSError:
            pass


_bind_msvc_runtime()

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
for _path in (str(PARENT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import core  # noqa: E402  (parent project module)
from core import (atomic_bytes, atomic_json, load_checkpoint, make_model,  # noqa: E402
                  model_config, restore_rng, rng_state, save_checkpoint,
                  seed_all, symmetry_split)

# ---------------------------------------------------------------------------
# Fixed settings (spec section 1). These are never tuned.
# ---------------------------------------------------------------------------
P = int(os.environ.get('NSEED_P', 59))
N_LAYERS = int(os.environ.get('NSEED_LAYERS', 1))
N_CONTEXTS = 4
D_MODEL = 128
N_HEADS = 4
D_MLP = 512
CONFIG = dict(p=P, n_contexts=N_CONTEXTS, d_model=D_MODEL, n_heads=N_HEADS, d_mlp=D_MLP,
              n_layers=N_LAYERS)

COVERAGE = float(os.environ.get('NSEED_COVERAGE', 0.4))
COUNT = int(COVERAGE * P * P)  # 1392 training pairs

# The Jia setting: modulus 113, a fifth of the pairs for training, one context
# only and the affine target rule of context 1 (a + 2b).
JIA = os.environ.get('NSEED_JIA', '').strip() == '1'
if JIA:
    P = 113
    N_CONTEXTS = 1
    COVERAGE = 0.2
    CONFIG = dict(p=P, n_contexts=N_CONTEXTS, d_model=D_MODEL, n_heads=N_HEADS,
                  d_mlp=D_MLP, n_layers=N_LAYERS)
    COUNT = int(COVERAGE * P * P)
AFFINE_MULT = 2 if JIA else int(os.environ.get('NSEED_AFFINE_MULT', 0))

# Pre-LN transformer (LayerNorm before attention and before the MLP, plus a
# final LayerNorm before the unembedding) and minibatch training (batch size
# N, the rows of an epoch sampled without replacement). Both default to the
# previous behaviour.
PRELN = 1 if os.environ.get('NSEED_PRELN', '').strip() == '1' else 0
BATCH_SIZE = int(os.environ.get('NSEED_BATCH', 0))
SPLIT = os.environ.get('NSEED_SPLIT', '').strip()
SPLIT_TASK = os.environ.get('NSEED_SPLIT_TASK', '').strip()
CONFIG['preln'] = PRELN
SPLIT_SEED = 1701

LEARNING_RATE = float(os.environ.get('NSEED_LR', 0.001))
WEIGHT_DECAY = float(os.environ.get('NSEED_WD', 1.0))
BETAS = (0.9, 0.98)

SOURCE_CAP = 20000
TRANSFER_STEPS = 2000
SOURCE_EVAL_EVERY = 250
SOURCE_CKPT_EVERY = 250
RESUMABLE_CKPT_EVERY = 500
TRANSFER_EVAL_EVERY = 100
TRANSFER_CKPT_EVERY = 500

GROK_TRAIN_ACC = 0.99
GROK_DEV_ACC = 0.90
GROK_CONSECUTIVE = 3

MEMORIZED_DEV_CEILING = 0.03
TRANSITION_LO = 0.25
TRANSITION_HI = 0.75
TRANSITION_TARGET = 0.5
MEMORIZED_FRACTION_OF_GROK = 0.4
LATE_OFFSET = 2000

INPUT_PERM_SEED = 555
EVAL_BATCH = 4096

TASKS = ('source', 'affine', 'scaled', 'input_perm', 'copy_a', 'selector')

BATCH_LOG = None

# The parent process declares the plan here and the worker processes report the
# finished jobs back, so `parallel.run_parallel` can tell a complete batch from
# one whose jobs never ran.
PLANNED_JOBS = {}
DONE_JOBS = set()
TERMINAL_STATES = ('done', 'source_done')


def set_batch_log(path):
    """Send every log line to the batch log as well as the console."""
    global BATCH_LOG
    BATCH_LOG = Path(path) if path else None
    if BATCH_LOG is not None:
        BATCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    return BATCH_LOG


def timestamp():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def log_line(message):
    """Print a timestamped line and append it to the batch log."""
    line = f'{timestamp()} {message}'
    print(line, flush=True)
    if BATCH_LOG is not None:
        with open(BATCH_LOG, 'a', encoding='utf-8') as stream:
            stream.write(line + '\n')
            stream.flush()
    return line


def prompt_step(number, total, description, phase='start'):
    """Announce a numbered step of the prompt in the batch log.

    `phase='end'` appends a marker so the log shows where a step finished.
    """
    text = f'PROMPT STEP {number}/{total}: {description}'
    if phase == 'end':
        text += ' [end]'
    return log_line(text)


def planned_line(seed, names):
    """Declare the jobs of one seed so that status.py can show pending work."""
    PLANNED_JOBS.setdefault(seed, set()).update(names)
    return log_line(f'PLANNED | seed={seed} | jobs=' + ','.join(names))


def job_line(seed, name, step, total, dev_mean, state='run', elapsed=None):
    """One progress line per job: timestamp, seed, job, step/total, dev accuracy."""
    seconds = 'n/a' if elapsed is None else f'{float(elapsed):.1f}'
    if state in TERMINAL_STATES:
        DONE_JOBS.add((seed, name))
    return log_line(f'JOB | seed={seed} | name={name} | step={step}/{total} | '
                    f'dev_mean={dev_mean} | {state} | elapsed={seconds}')


def skipped_job_line(seed, name, steps, record):
    """The job line of a job that was already done and is skipped on resume."""
    value = as_float(record.get('final_dev_mean'))
    return job_line(seed, name, steps, steps,
                    'nan' if value is None else f'{value:.6f}', 'done', elapsed=None)

METRIC_FIELDS = [
    'seed', 'stage', 'task', 'init', 'step',
    'train_acc', 'dev_acc_c0', 'dev_acc_c1', 'dev_acc_c2', 'dev_acc_c3',
    'dev_mean', 'dev_worst', 'stage_clean_acc', 'global_norm',
    'sec_per_step', 'elapsed_s',
]


# ---------------------------------------------------------------------------
# Runtime setup
# ---------------------------------------------------------------------------
DETERMINISTIC_ENV = 'NSEED_DETERMINISTIC'
DETERMINISTIC_GPU_ENV = 'NSEED_DETERMINISTIC_GPU'
DEVICE_ENV = 'NSEED_DEVICE'


def determinism_enabled():
    """True when NSEED_DETERMINISTIC=1 (deterministic algorithms on).

    Default is OFF: the amendment makes determinism optional so that the
    overnight run can use the faster non-deterministic CPU kernels. The flag is
    recorded in every run's metadata. On the GPU this only takes effect when
    NSEED_DETERMINISTIC_GPU=1 is also set.
    """
    return os.environ.get(DETERMINISTIC_ENV, '').strip() == '1'


def gpu_determinism_enabled():
    """True when NSEED_DETERMINISTIC_GPU=1 (deterministic algorithms on the GPU).

    Off by default: deterministic algorithms are never enabled on the GPU
    unless this flag is set explicitly. Measured cost on the RTX 3060 Laptop:
    45.4 -> 40.3 steps/s, and two identical runs then agree bit for bit.
    The flag is independent of NSEED_DETERMINISTIC, which controls the CPU.
    """
    return os.environ.get(DETERMINISTIC_GPU_ENV, '').strip() == '1'


def cuda_available():
    return bool(torch.cuda.is_available())


def resolve_device(device=None):
    """Resolve the device name: 'cuda' when available, else 'cpu'."""
    if device is None:
        device = os.environ.get(DEVICE_ENV, '').strip() or 'auto'
    device = str(device).lower()
    if device == 'auto':
        return 'cuda' if cuda_available() else 'cpu'
    if device == 'cuda':
        if not cuda_available():
            raise RuntimeError('CUDA was requested but torch.cuda.is_available() is False.')
        return 'cuda'
    if device == 'cpu':
        return 'cpu'
    raise ValueError(f'Unknown device {device!r}; use cpu, cuda or auto.')


def runtime_metadata(device=None):
    """The environment fingerprint recorded with every job.

    Contains the GPU name, the torch version and the host name, so a result
    folder says which machine produced it.
    """
    import platform
    import socket
    name = None
    if cuda_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001 - a broken driver must not stop a run
            name = None
    return dict(device=str(device) if device is not None else None,
                gpu_name=name,
                torch_version=torch.__version__,
                cuda_version=torch.version.cuda,
                hostname=socket.gethostname(),
                python_version=platform.python_version(),
                platform=platform.platform())


def setup(threads=None, device=None):
    """Set the thread count, resolve the device and apply the determinism flag.

    Returns the run metadata: {'threads', 'deterministic', 'device', ...}.
    """
    if threads is not None:
        torch.set_num_threads(int(threads))
    device = resolve_device(device)
    deterministic = ((determinism_enabled() and device == 'cpu')
                     or (gpu_determinism_enabled() and device == 'cuda'))
    torch.use_deterministic_algorithms(deterministic)
    metadata = runtime_metadata(device)
    metadata.update(threads=torch.get_num_threads(), deterministic=deterministic,
                     cuda_available=cuda_available(),
                     gpu_deterministic=gpu_determinism_enabled())
    return metadata


def rng_state_extended():
    """RNG state including the CUDA generators."""
    state = dict(core.rng_state())
    if cuda_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_extended(state):
    """Restore python/numpy/torch RNG state, including the CUDA generators."""
    core.restore_rng(state)
    cuda_state = state.get('cuda')
    if cuda_state is not None and cuda_available():
        torch.cuda.set_rng_state_all(cuda_state)


# ---------------------------------------------------------------------------
# Masks, splits, model construction
# ---------------------------------------------------------------------------
def mask_for_seed(seed):
    """Boolean pair mask of shape (n_contexts, p*p); the same pairs in every context.

    Spec section 1: `g = torch.Generator().manual_seed(100000 + S)` and
    `selected = torch.randperm(p*p, generator=g)[:int(0.4*p*p)]`.
    """
    generator = torch.Generator().manual_seed(100000 + int(seed))
    selected = torch.randperm(P * P, generator=generator)[:COUNT]
    mask = torch.zeros((N_CONTEXTS, P * P), dtype=torch.bool)
    mask[:, selected] = True
    return mask


def random_split(mask):
    """Train on the mask's pairs, dev on every remaining pair.

    The mask holds a random fraction of all the pairs (the Jia preset's 20 %),
    and the swapped pairs of the training pairs are not excluded from the dev
    split. The audit split is empty and never read.
    """
    ids = torch.arange(P * P)
    train = ids[mask[0]]
    dev = ids[~mask[0]]
    empty = ids[:0]
    return ({'train': train, 'dev': dev, 'audit': empty, 'exposed': empty},
            dict(pair_ids=ids, transpose_ids=ids, orbit_ids=ids,
                 global_exact_exposed=ids[:0], global_clean=ids))


def sub_split(mask, seed):
    """A pair-level random split with the sizes of the symmetry-clean split.

    The non-commutative tasks cannot use the orbit split, so the training pairs
    stay the mask's pairs and the remaining pairs are cut at random into the
    same dev, exposed and audit sizes. The audit slice is never used.
    """
    strata, exposure = symmetry_split(mask, P, SPLIT_SEED)
    ids = torch.arange(P * P)
    train = ids[mask[0]]
    rest = ids[~mask[0]]
    generator = torch.Generator().manual_seed(SPLIT_SEED + int(seed))
    order = rest[torch.randperm(len(rest), generator=generator)]
    dev_size = len(strata['dev'])
    exposed_size = len(strata['exposed'])
    dev = order[:dev_size]
    exposed = order[dev_size:dev_size + exposed_size]
    audit = order[dev_size + exposed_size:]
    return ({'train': train, 'dev': dev, 'audit': audit, 'exposed': exposed},
            dict(pair_ids=ids, transpose_ids=exposure['transpose_ids'],
                 orbit_ids=exposure['orbit_ids'],
                 global_exact_exposed=exposure['global_exact_exposed'],
                 global_clean=exposure['global_clean']))


def split_for_mask(mask, seed=None):
    """The training and dev split of a seed.

    The default is `core.symmetry_split(mask, p, 1701)`; with
    NSEED_SPLIT=random the training set is the mask's pairs and the dev set is
    every remaining pair, and with NSEED_SPLIT_TASK=sub the non-commutative
    tasks get a pair-level random split of the same sizes.
    """
    if SPLIT == 'random':
        return random_split(mask)
    if SPLIT_TASK == 'sub':
        return sub_split(mask, 0 if seed is None else seed)
    return symmetry_split(mask, P, SPLIT_SEED)


def init_model(seed, device='cpu'):
    """Fresh model for `seed`, seeded immediately before construction.

    The model is always built on the CPU (so the init does not depend on the
    device) and then moved to `device`.
    """
    seed_all(int(seed))
    model = make_model(CONFIG)
    return model.to(device)


def input_permutation():
    """Fixed random relabeling of the numeric tokens (spec section 2)."""
    generator = torch.Generator().manual_seed(INPUT_PERM_SEED)
    return torch.randperm(P, generator=generator)


# ---------------------------------------------------------------------------
# Data and labels
# ---------------------------------------------------------------------------
def task_labels(task, a, b, context, permutation=None):
    """Labels for one context. `a`, `b`, `context` are integer tensors."""
    if task in ('source', 'add'):
        return (a + b) % P
    if task == 'sub':
        return (a - b) % P
    if task == 'mult':
        return (a * b) % P
    if task in ('affine', 'affine_add'):
        multiplier = AFFINE_MULT if AFFINE_MULT else (context + 1)
        return (a + multiplier * b) % P
    if task == 'affine_sub':
        return (a - (context + 1) * b) % P
    if task == 'affine_mult':
        return (a * torch.pow(b, context + 1)) % P
    if task == 'scaled':
        return ((context + 1) * (a + b)) % P
    if task == 'copy_a':
        return a % P
    if task == 'selector':
        return (a if context in (0, 2) else b) % P
    if task == 'input_perm':
        if permutation is None:
            raise ValueError('input_perm needs the fixed input permutation.')
        return (permutation[a] + permutation[b]) % P
    raise ValueError(f'Unknown task {task!r}')


def build_task(task, device='cpu', permutation=None):
    """Full dataset for one task, ordered context-major over cartesian_prod(p, p).

    Row index of pair `i` in context `c` is `c * p * p + i`.
    Token format is `[p + context, a, b, p + n_contexts]`.
    The inputs and labels live on `device`; the pair ids stay on the CPU.
    """
    pairs = torch.cartesian_prod(torch.arange(P), torch.arange(P))
    a, b = pairs[:, 0], pairs[:, 1]
    if task == 'input_perm' and permutation is None:
        permutation = input_permutation()
    xs, ys, cs, ids = [], [], [], []
    for context in range(N_CONTEXTS):
        xs.append(torch.stack([
            torch.full_like(a, P + context),
            a,
            b,
            torch.full_like(a, P + N_CONTEXTS),
        ], dim=1))
        ys.append(task_labels(task, a, b, context, permutation))
        cs.append(torch.full_like(a, context))
        ids.append(torch.arange(P * P))
    return dict(task=task, x=torch.cat(xs).to(device), y=torch.cat(ys).to(device),
                context=torch.cat(cs), pair_ids=torch.cat(ids), pairs=pairs,
                device=str(device))


def context_rows(pair_ids, context, device='cpu'):
    """Row indices of `pair_ids` inside one context, on `device`."""
    ids = torch.as_tensor(pair_ids, dtype=torch.long, device=device)
    return int(context) * (P * P) + ids


def index_for(pair_ids, device='cpu'):
    """Row indices of `pair_ids` in every context, context-major, on `device`."""
    ids = torch.as_tensor(pair_ids, dtype=torch.long, device=device)
    blocks = [context_rows(ids, c, device) for c in range(N_CONTEXTS)]
    return torch.cat(blocks)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def accuracy_on(model, data, rows, batch_size=EVAL_BATCH):
    if len(rows) == 0:
        return float('nan')
    correct = 0
    for chunk in rows.split(batch_size):
        logits = model(data['x'][chunk])
        correct += int((logits.argmax(-1) == data['y'][chunk]).sum())
    return correct / len(rows)


@torch.no_grad()
def evaluate(model, data, strata):
    """Train accuracy plus per-context clean-dev accuracy, mean and worst context.

    The split ids stay on the CPU; the index tensors are moved to the device of
    the data.
    """
    device = data.get('device', 'cpu')
    train_rows = index_for(strata['train'], device)
    result = {'train_acc': accuracy_on(model, data, train_rows)}
    per_context = []
    for context in range(N_CONTEXTS):
        rows = context_rows(strata['dev'], context, device)
        per_context.append(accuracy_on(model, data, rows))
    result.update({f'dev_acc_c{c}': per_context[c] for c in range(N_CONTEXTS)})
    result['dev_mean'] = sum(per_context) / len(per_context)
    result['dev_worst'] = min(per_context)
    return result


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------
def global_norm(model):
    """L2 norm over every parameter of the model."""
    total = 0.0
    for parameter in model.parameters():
        total += float(parameter.detach().pow(2).sum())
    return math.sqrt(total)


def per_module_norms(model):
    """L2 norm per top-level module (token_embed, pos_embed, attn, mlp_in, ...)."""
    groups = {}
    for name, parameter in model.named_parameters():
        top = name.split('.')[0]
        groups[top] = groups.get(top, 0.0) + float(parameter.detach().pow(2).sum())
    return {key: math.sqrt(value) for key, value in sorted(groups.items())}


def rescale_to_norm(model, target_norm):
    """Multiply every parameter by one scalar so the global L2 norm matches.

    Returns the scale factor that was applied.
    """
    current = global_norm(model)
    if current <= 0:
        raise ValueError('Cannot rescale a model with zero parameter norm.')
    scale = float(target_norm) / current
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(scale)
    return scale


# ---------------------------------------------------------------------------
# Optimizer and training step
# ---------------------------------------------------------------------------
def make_optimizer(model, parameters=None):
    """Fresh AdamW with the fixed hyperparameters (spec section 1)."""
    params = list(model.parameters()) if parameters is None else list(parameters)
    return torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                             betas=BETAS)


def train_step(model, optimizer, data, rows):
    """One full-batch gradient step. Returns the loss value."""
    optimizer.zero_grad(set_to_none=True)
    logits = model(data['x'][rows])
    loss = torch.nn.functional.cross_entropy(logits, data['y'][rows])
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def epoch_batches(train_rows, batch_size, generator):
    """The row batches of one epoch: a permutation of the rows, cut in batches.

    The rows of an epoch are sampled without replacement; with no batch size (or
    a batch as large as the training set) the whole set is one batch, which is
    the full-batch training of before.
    """
    count = len(train_rows)
    if not batch_size or batch_size <= 0 or batch_size >= count:
        return [train_rows]
    order = train_rows[torch.randperm(count, generator=generator)]
    return [order[start:start + batch_size]
            for start in range(0, count, batch_size)]


# ---------------------------------------------------------------------------
# Metrics rows and CSV
# ---------------------------------------------------------------------------
def make_row(seed, stage, task, init, step, evaluation, stage_clean_acc=None,
             norm=None, sec_per_step=None, elapsed=None):
    """Build one metrics row with exactly the METRIC_FIELDS keys."""
    row = {field: '' for field in METRIC_FIELDS}
    row['seed'] = int(seed)
    row['stage'] = stage
    row['task'] = task
    row['init'] = init
    row['step'] = int(step)
    row['train_acc'] = evaluation['train_acc']
    for c in range(N_CONTEXTS):
        row[f'dev_acc_c{c}'] = evaluation[f'dev_acc_c{c}']
    row['dev_mean'] = evaluation['dev_mean']
    row['dev_worst'] = evaluation['dev_worst']
    row['stage_clean_acc'] = '' if stage_clean_acc is None else stage_clean_acc
    row['global_norm'] = '' if norm is None else norm
    row['sec_per_step'] = '' if sec_per_step is None else sec_per_step
    row['elapsed_s'] = '' if elapsed is None else elapsed
    return row


def rows_bytes(rows, fields=METRIC_FIELDS):
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=list(fields))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode()


def write_metrics(path, rows, fields=METRIC_FIELDS):
    """Atomically rewrite the metrics CSV."""
    if rows:
        atomic_bytes(path, rows_bytes(rows, fields))
    else:
        atomic_bytes(path, rows_bytes([], fields))


def read_metrics(path):
    """Read a metrics CSV into a list of dicts (empty list if missing)."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open('r', newline='') as stream:
        reader = csv.DictReader(stream)
        rows = []
        for raw in reader:
            row = {field: raw.get(field, '') for field in METRIC_FIELDS}
            rows.append(row)
    return rows


def merge_rows(existing, new):
    """Union of metric rows keyed by (stage, task, init, step); later wins.

    The step is compared as a string because rows that come back from the CSV
    carry string steps while freshly built rows carry integers.
    """
    merged = {}
    order = []
    for row in list(existing) + list(new):
        key = (row['stage'], row['task'], row['init'], str(row['step']))
        if key not in merged:
            order.append(key)
        merged[key] = row
    return [merged[key] for key in order]


def as_float(value):
    if value is None or value == '':
        return None
    return float(value)


def as_int(value):
    if value is None or value == '':
        return None
    return int(float(value))


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def cpu_state_dict(model):
    """State dict with every tensor on the CPU, so a checkpoint loads anywhere."""
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def save_model_checkpoint(path, model, step, row=None, extra=None):
    """Model-only checkpoint: state_dict + step + metrics row (no optimizer)."""
    payload = {'model': cpu_state_dict(model), 'step': int(step)}
    if row is not None:
        payload['metrics'] = row
    if extra:
        payload.update(extra)
    save_checkpoint(path, payload)
    return path


# ---------------------------------------------------------------------------
# Model components (component study)
# ---------------------------------------------------------------------------
COMPONENTS = ('numeric_embed', 'other_embed', 'attn', 'mlp', 'unembed')


def component_entries(state, name):
    """The (state-dict key, row slice) entries that make up one component.

    A row slice of None means the whole tensor belongs to the component.
    `numeric_embed` and `other_embed` share the token-embedding tensor, so they
    are described by disjoint row ranges of it.
    """
    if name == 'numeric_embed':
        return [('token_embed.weight', slice(0, P))]
    if name == 'other_embed':
        return [('token_embed.weight', slice(P, None)), ('pos_embed', None)]
    if name == 'attn':
        return [(key, None) for key in state if key.startswith('attn.')]
    if name == 'mlp':
        return [(key, None) for key in state
                if key.startswith(('mlp_in.', 'mlp_out.'))]
    if name == 'unembed':
        return [('unembed.weight', None)]
    raise ValueError(f'Unknown component {name!r}')


def build_hybrid(recipient_state, donor_state, components):
    """The recipient's state dict with the named components taken from the donor."""
    merged = {key: value.clone() for key, value in recipient_state.items()}
    for name in components:
        for key, rows in component_entries(recipient_state, name):
            if key not in donor_state:
                raise KeyError(f'donor has no {key}')
            if rows is None:
                merged[key] = donor_state[key].clone()
            else:
                merged[key][rows] = donor_state[key][rows].clone()
    return merged


def cpu_optimizer_state(optimizer):
    """Optimizer state with every tensor on the CPU."""
    state = optimizer.state_dict()
    return {'state': {key: {field: (value.detach().cpu()
                                    if isinstance(value, torch.Tensor) else value)
                          for field, value in entry.items()}
                      for key, entry in state['state'].items()},
            'param_groups': state['param_groups']}


def save_resumable(path, model, optimizer, step, rows, extra=None):
    """Full resumable state: model, optimizer, RNG state and the metrics rows."""
    payload = {'model': cpu_state_dict(model),
               'optimizer': cpu_optimizer_state(optimizer),
               'rng': rng_state_extended(),
               'step': int(step),
               'metrics_rows': list(rows)}
    if extra:
        payload.update(extra)
    save_checkpoint(path, payload)
    return path


def restore_resumable(path, model, optimizer=None):
    """Restore model/optimizer/RNG from a resumable checkpoint.

    The model weights are stored on the CPU and are copied onto the model's
    current device. The CUDA RNG state is restored as well.

    Returns (payload, model, optimizer).
    """
    payload = load_checkpoint(path)
    model.load_state_dict(payload['model'])
    if optimizer is not None and payload.get('optimizer') is not None:
        optimizer.load_state_dict(payload['optimizer'])
    if payload.get('rng') is not None:
        restore_rng_extended(payload['rng'])
    return payload, model, optimizer


def write_summary(path, payload):
    atomic_json(path, payload)


def read_summary(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8') as stream:
        return json.load(stream)


# ---------------------------------------------------------------------------
# Embedding Fourier metrics (spec section 3d)
# ---------------------------------------------------------------------------
def embedding_spectrum(embedding):
    """Energy per frequency for the numeric token embedding.

    `embedding` has shape (p, d). Returns (energy_k for k = 0..(p-1)//2,
    total_energy_including_k0).
    """
    if embedding.dim() != 2 or embedding.shape[0] != P:
        raise ValueError(f'Expected a (p, d) embedding, got {tuple(embedding.shape)}')
    spectrum = torch.fft.fft(embedding.double(), dim=0)
    power = (spectrum.real ** 2 + spectrum.imag ** 2).sum(dim=1)  # (p,)
    half = (P - 1) // 2
    ks = torch.arange(1, half + 1)
    energy = power[ks] + power[P - ks]  # |F_k|^2 + |F_{p-k}|^2 for k = 1..half
    return energy, power


def fourier_metrics(embedding):
    """top5_share, freq_participation and nonconst_share for one embedding."""
    energy, power = embedding_spectrum(embedding)
    total = float(energy.sum())
    if total <= 0:
        return dict(top5_share=float('nan'), freq_participation=float('nan'),
                    nonconst_share=float('nan'), peak_k=-1)
    ordered = torch.sort(energy, descending=True).values
    top5 = float(ordered[:5].sum()) / total
    participation = (total ** 2) / float((energy ** 2).sum())
    nonconst = total / float(power.sum())
    peak_k = int(torch.argmax(energy).item()) + 1
    return dict(top5_share=top5, freq_participation=participation,
                nonconst_share=nonconst, peak_k=peak_k)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


class Tee:
    """Duplicate stdout into a log file (parent process of the batch runner)."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('a', encoding='utf-8', buffering=1)
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.stdout.flush()
        self.stream.write(data)
        self.stream.flush()

    def flush(self):
        self.stdout.flush()
        self.stream.flush()

    def close(self):
        self.stream.close()


def _process_identity(pid):
    """(host, start time) of a process, or None when the process is gone.

    The start time distinguishes a live process from a recycled process id, so
    a lock left behind by an interrupted run is recognised as stale.
    """
    import socket
    host = socket.gethostname()
    if pid is None:
        return None
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return None
        try:
            creation = wintypes.FILETIME()
            exit_time = wintypes.FILETIME()
            kernel_time = wintypes.FILETIME()
            user_time = wintypes.FILETIME()
            kernel = ctypes.windll.kernel32
            kernel.GetProcessTimes.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_void_p]
            # GetProcessTimes writes to all four time pointers, so none of them
            # may be NULL.
            kernel.GetProcessTimes(handle, ctypes.byref(creation),
                                  ctypes.byref(exit_time), ctypes.byref(kernel_time),
                                  ctypes.byref(user_time))
            stamp = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
            return host, stamp
        except OSError:
            return None
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        with open(f'/proc/{int(pid)}/stat', 'r', encoding='utf-8') as stream:
            fields = stream.read().rsplit(') ', 1)[1].split()
        return host, int(fields[19])
    except (OSError, IndexError, ValueError):
        return None


def _lock_is_stale(record):
    """True when a lock record no longer describes a live owner."""
    import socket
    if not isinstance(record, dict):
        return True
    if record.get('host') and record['host'] != socket.gethostname():
        return True
    identity = _process_identity(record.get('pid'))
    if identity is None:
        return True
    recorded = record.get('identity')
    if recorded is None:
        return True
    if recorded[1] != identity[1]:
        return True
    return False


def acquire_job_lock(path):
    """Take a job lock. Returns (acquired, reason); a stale lock is reclaimed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            holder = json.loads(path.read_text(encoding='utf-8'))
        except Exception:  # noqa: BLE001 - an unreadable lock counts as stale
            holder = {}
        if not _lock_is_stale(holder):
            return False, f'locked by pid {holder.get("pid")}'
        path.unlink(missing_ok=True)
    try:
        handle = path.open('x')
    except FileExistsError:
        return False, 'locked by another worker'
    handle.write(json.dumps({'pid': os.getpid(), 'host': _process_identity(os.getpid())[0],
                             'identity': _process_identity(os.getpid()),
                             'started': time.time()}))
    handle.close()
    return True, 'acquired'


def release_job_lock(path):
    Path(path).unlink(missing_ok=True)


def _pid_alive(pid):
    """True when the process id is still running (Windows-safe)."""
    if pid is None:
        return False
    pid = int(pid)
    if sys.platform == 'win32':
        import ctypes
        query_limited_information = 0x1000
        still_active = 259
        handle = ctypes.windll.kernel32.OpenProcess(query_limited_information, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == still_active
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_seed_lock(folder):
    """Exclusive per-seed lock. Returns the lock path, raises if already held."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'seed.lock'
    try:
        handle = path.open('x')
    except FileExistsError as exc:
        raise RuntimeError(f'Seed lock already held: {path}') from exc
    handle.write(json.dumps({'pid': os.getpid(), 'started': time.time()}))
    handle.close()
    return path


def release_seed_lock(path):
    Path(path).unlink(missing_ok=True)
