"""Shared helpers of queue 17: the stage-1 training with a checkpoint hook,
the stage-2 transfer and the direct baselines.

The stage-1 training follows queue16: full fine-tuning, batch 128, the betas
(0.9, 0.98), 100 warmup steps and then a constant learning rate, with the
learning rate optionally decaying linearly to zero over the horizon. The
training set is tokenized once and the batches are drawn from the cached
tensors, because tokenizing the batch inside the loop costs more than the
training step itself.

Produces: the stage-1 training, the stage-2 transfer and the direct baseline.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import common  # noqa: E402
from p15_common import (MODELS, REVISION, clear_model_cache, load_model,  # noqa: E402
                        read_rows, token_accuracy, write_rows)

BETAS = (0.9, 0.98)
WARMUP = 100
BATCH = 128
STAGE2_BATCH = 16
STAGE2_PAIRS = 500
STAGE2_HELDOUT = 500


def splits_for(p, heldout=1000, seed=0):
    """The fixed pair splits of a task of size p x p."""
    pairs = [(a, b) for a in range(p) for b in range(p)]
    random.Random(seed).shuffle(pairs)
    return dict(train_pool=pairs[heldout:], heldout=pairs[:heldout])


def make_examples(symbols, pairs, p, rule='add'):
    """The (prompt, answer) examples of a rule: add, a2b, sub, mult, shuffled."""
    prompts = [f'{symbols[str(a)]} {symbols[str(b)]} =' for a, b in pairs]
    if rule == 'add':
        answers = [symbols[str((a + b) % p)] for a, b in pairs]
    elif rule in ('a2b', 'add2'):
        answers = [symbols[str((a + 2 * b) % p)] for a, b in pairs]
    elif rule == 'sub':
        answers = [symbols[str((a - b) % p)] for a, b in pairs]
    elif rule == 'mult':
        answers = [symbols[str((a * b) % p)] for a, b in pairs]
    elif rule == 'shuffled':
        answers = [symbols[str((a + b) % p)] for a, b in pairs]
        random.Random(0).shuffle(answers)
    else:
        raise ValueError(f'unknown rule {rule!r}')
    return list(zip(prompts, answers))


def answer_ids(payload, pairs, p, rule='add'):
    """The answer token ids of the pairs."""
    token_ids = payload['token_ids']
    out = []
    for a, b in pairs:
        if rule in ('a2b', 'add2'):
            residue = (a + 2 * b) % p
        elif rule == 'sub':
            residue = (a - b) % p
        elif rule == 'mult':
            residue = (a * b) % p
        else:
            residue = (a + b) % p
        out.append(token_ids[str(residue)])
    return out


def prepare_examples(examples, tokenizer, device):
    """The tokenised training set, tokenized once."""
    texts = [text + ' ' + answer for text, answer in examples]
    encoded = tokenizer(texts, return_tensors='pt', padding=True, truncation=True,
                        max_length=64)
    ids = encoded['input_ids']
    labels = ids.clone()
    labels[encoded['attention_mask'] == 0] = -100
    labels[:, :-1] = -100
    return ids.to(device), labels.to(device)


def fine_tune(model, tokenizer, examples, device, steps, learning_rate, weight_decay,
              horizon=None, decay=False, warmup=WARMUP, batch_size=BATCH, seed=0,
              hook=None, hook_every=0):
    """Full fine-tuning with the losses and a checkpoint hook.

    `hook(step, loss)` is called every `hook_every` steps. With `decay` the
    learning rate falls linearly to zero over `horizon` steps after the warmup.
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay, betas=BETAS)
    ids, labels = prepare_examples(examples, tokenizer, device)
    generator = torch.Generator(device=ids.device).manual_seed(int(seed))
    losses = []
    for step in range(1, steps + 1):
        if step <= warmup:
            scale = step / warmup
        elif decay and horizon:
            scale = max(0.0, (horizon - step) / max(1, horizon - warmup))
        else:
            scale = 1.0
        for group in optimizer.param_groups:
            group['lr'] = learning_rate * scale
        index = torch.randint(0, ids.shape[0], (batch_size,), generator=generator,
                              device=ids.device)
        loss = model(input_ids=ids[index], labels=labels[index]).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss))
        if hook is not None and hook_every and (step % hook_every == 0
                                                or step == steps):
            hook(step, losses)
    model.eval()
    return losses


def save_checkpoint(path, model):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


_MODEL_CACHE = {}


def get_model(model_name, device):
    """The model, the tokenizer and the pretrained weights of one worker.

    The stage-2 transfers load many checkpoints; building the model from the
    Hugging Face cache every time costs more than the transfer itself.
    """
    key = (model_name, str(device))
    if key not in _MODEL_CACHE:
        model, tokenizer = load_model(model_name, REVISION, device)
        initial = {name: value.detach().clone()
                   for name, value in model.state_dict().items()}
        _MODEL_CACHE[key] = (model, tokenizer, initial)
    return _MODEL_CACHE[key]


def reset_model(model_name, device):
    """The cached model with the pretrained weights."""
    get_model(model_name, device)
    model, tokenizer, initial = _MODEL_CACHE[(model_name, str(device))]
    model.load_state_dict(initial)
    model.to(device)
    model.eval()
    return model, tokenizer


def load_checkpoint(path, model_name, device):
    """A model with the weights of the checkpoint."""
    model, tokenizer, _initial = get_model(model_name, device)
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, tokenizer


def stage2_accuracy(model, tokenizer, payload, splits, p, device, learning_rate,
                    weight_decay, rule, seed, steps, batch_size=STAGE2_BATCH):
    """The held-out accuracy of the target after the stage-2 training."""
    pairs = splits['train_pool'][:STAGE2_PAIRS]
    heldout = splits['heldout'][:STAGE2_HELDOUT]
    examples = make_examples(payload['symbols'], pairs, p, rule)
    prompts = [f'{payload["symbols"][str(a)]} {payload["symbols"][str(b)]} ='
               for a, b in heldout]
    ids = answer_ids(payload, heldout, p, rule)
    fine_tune(model, tokenizer, examples, device, steps=steps,
              learning_rate=learning_rate, weight_decay=weight_decay, warmup=0,
              batch_size=batch_size, seed=seed)
    return token_accuracy(model, tokenizer, prompts, ids, device)


def direct_baseline(model_name, payload, splits, p, device, learning_rate,
                    weight_decay, seed, rule, steps):
    """The stage-2 task trained straight from the pretrained model."""
    model, tokenizer = reset_model(model_name, device)
    value = stage2_accuracy(model, tokenizer, payload, splits, p, device,
                             learning_rate, weight_decay, rule, seed, steps)
    del model
    torch.cuda.empty_cache()
    return value
