"""Shared helpers of queue 15: the model, the full fine-tune, the token
accuracy and the resume-safe CSV.

The conventions of `pythia_transfer.py` are reused: the pad token falls back
to the end-of-text token, the padding is on the left, the fine-tune loss is
taken over the answer tokens only, the downloaded weights of a model are
dropped from the Hugging Face cache after the model is done and the output
CSV is rewritten from the full table after every finished configuration, so a
restart resumes instead of duplicating.

Queue 15 fine-tunes the whole model, so the weights are loaded in float32.

Produces: the loader of the Pythia models and the token accuracy.
"""
from __future__ import annotations

import csv
import io
import random
import shutil
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {'160m': 'EleutherAI/pythia-160m', '410m': 'EleutherAI/pythia-410m'}
REVISION = 'step143000'
EVAL_BATCH = 128


def load_model(model_name, revision, device):
    """The checkpoint at `revision`, in float32 on the device."""
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision,
                                                 dtype=torch.float32)
    model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token is None:
        for candidate in (tokenizer.eos_token, tokenizer.bos_token, tokenizer.unk_token):
            if candidate is not None:
                tokenizer.pad_token = candidate
                break
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(0)
    tokenizer.padding_side = 'left'
    return model, tokenizer


def clear_model_cache(model_name):
    """Drop the downloaded weights of one model from the Hugging Face cache."""
    from huggingface_hub import constants as hf_constants
    hub = Path(hf_constants.HF_HUB_CACHE)
    try:
        for folder in (hub / ('models--' + model_name.replace('/', '--')),
                       hub / 'blobs', hub / '.locks'):
            if folder.is_dir():
                shutil.rmtree(folder, ignore_errors=True)
                print(f'cleared {folder}', flush=True)
    except Exception as exc:  # noqa: BLE001 - the cache is only a disk saving
        print(f'could not clear the cache of {model_name}: {exc}', flush=True)


@torch.no_grad()
def token_accuracy(model, tokenizer, prompts, answer_ids, device, batch_size=EVAL_BATCH):
    """The fraction of prompts whose next token is the answer token."""
    if not prompts:
        return float('nan')
    correct = 0
    for start in range(0, len(prompts), batch_size):
        chunk_prompts = prompts[start:start + batch_size]
        chunk_answers = answer_ids[start:start + batch_size]
        encoded = tokenizer(chunk_prompts, return_tensors='pt', padding=True).to(device)
        logits = model(**encoded).logits[:, -1, :]
        predicted = logits.argmax(dim=-1)
        expected = torch.tensor(chunk_answers, dtype=torch.long, device=device)
        correct += int((predicted == expected).sum().item())
    return correct / len(prompts)


def fine_tune(model, tokenizer, examples, device, steps, batch_size, learning_rate,
              weight_decay, seed=0, eval_every=0, evaluate=None, hook=None,
              hook_every=0):
    """Full fine-tuning on the answer tokens; returns the seconds per step.

    With `evaluate` given, it is called every `eval_every` steps and the list
    of (step, value) pairs is returned as well. With `hook` given, it is called
    every `hook_every` steps, which lets a caller write a temporary checkpoint
    and work on it while the training state stays in memory.
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)
    generator = random.Random(seed)
    total = 0.0
    measured = []
    for step in range(1, steps + 1):
        batch = [examples[generator.randrange(len(examples))] for _ in range(batch_size)]
        texts = [text for text, _answer in batch]
        answers = [answer for _text, answer in batch]
        full = [text + ' ' + answer for text, answer in batch]
        encoded = tokenizer(full, return_tensors='pt', padding=True,
                            truncation=True, max_length=64).to(device)
        labels = encoded['input_ids'].clone()
        labels[encoded['attention_mask'] == 0] = -100
        labels[:, :-1] = -100
        tick = time.time()
        loss = model(**encoded, labels=labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        total += time.time() - tick
        if hook is not None and hook_every and (step % hook_every == 0
                                                or step == steps):
            hook(step)
        if evaluate is not None and eval_every and (step % eval_every == 0
                                                    or step == steps):
            measured.append((step, evaluate()))
    model.eval()
    fine_tune.last_seconds_per_step = total / max(1, steps)
    return measured


def write_rows(path, fields, rows):
    """Rewrite the whole CSV; the table is the state of the run."""
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(out.getvalue(), encoding='utf-8')
    return path


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open('r', newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def free_disk_gb(path='/'):
    """The free space of the filesystem that holds `path`, in gigabytes."""
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)
