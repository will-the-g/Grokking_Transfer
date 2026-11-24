"""Shared helpers of queue 18: the mean-removed Fourier spectrum of the symbol
tokens, the frequency-removal edits and the fine-tune with a late weight decay
change.

The conventions of queues 15 to 17 are reused unchanged: the model is loaded
from the Hugging Face cache in float32, the stage-1 training is a full
fine-tune with the betas (0.9, 0.98), 100 warmup steps and then a constant
learning rate, the stage-2 transfer is a full fine-tune with a fresh AdamW on
500 pairs and the answer tokens only.

The spectrum of queue 18 differs from the one of queue 17 in one way: the mean
over the p symbol tokens is removed before the transform, and the zero frequency
is left out of the total energy, so `frac_top5` is the share of the non-zero
frequencies that the five strongest frequencies hold.

Produces: the spectrum of the queue 18 sources.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from p15_common import (MODELS, REVISION, clear_model_cache, load_model,  # noqa: E402
                         token_accuracy)
from p17_common import (BATCH, BETAS, STAGE2_BATCH, STAGE2_HELDOUT,  # noqa: E402
                         STAGE2_PAIRS, WARMUP, answer_ids, direct_baseline,
                         fine_tune, get_model, load_checkpoint, make_examples,
                         prepare_examples, reset_model, save_checkpoint,
                         splits_for, stage2_accuracy)
from p17_traj import (append_rows, best_direct, embedding_tables,  # noqa: E402
                      param_norm, project_out, read_rows, schedule, symbol_rows)


def spectrum_nodc(rows):
    """The energy per frequency of the mean-removed symbol rows.

    `rows` holds the p symbol tokens ordered by the residue index. The mean over
    the p tokens is removed first, the real transform is taken along the residue
    axis and the energy of every frequency is summed over the hidden dimensions.
    The zero frequency is left out of the total, so the returned fraction is the
    share of the five strongest non-zero frequencies.
    """
    centered = rows - rows.mean(dim=0, keepdim=True)
    spec = torch.fft.rfft(centered, dim=0)
    energy = (spec.real ** 2 + spec.imag ** 2).sum(dim=1)
    nonzero = energy[1:]
    total = float(nonzero.sum())
    if total <= 0 or nonzero.numel() == 0:
        return 0.0, []
    count = min(5, int(nonzero.numel()))
    values, indices = torch.topk(nonzero, count)
    fraction = float(values.sum()) / total
    frequencies = sorted(int(index) + 1 for index in indices)
    return fraction, frequencies


def remove_frequencies(weight, ids, rows, frequencies):
    """Remove the frequencies from the given rows of a weight matrix.

    Removing a frequency means projecting out the cosine and the sine component
    along the residue axis in every hidden dimension; the mean is left alone.
    """
    with torch.no_grad():
        edited = project_out(rows, frequencies)
        weight[ids] = edited.to(weight.dtype)
    return weight


def apply_edit(model, payload, p, frequencies, target):
    """Remove the frequencies from the symbol tokens of the model.

    `target` is 'input_embedding', 'unembedding' or 'both'.
    """
    embed, unembed = embedding_tables(model)
    ids = [payload['token_ids'][str(residue)] for residue in range(p)]
    if target in ('input_embedding', 'both'):
        remove_frequencies(embed, ids, symbol_rows(embed, payload, p), frequencies)
    if target in ('unembedding', 'both'):
        remove_frequencies(unembed, ids, symbol_rows(unembed, payload, p), frequencies)
    return model


def fine_tune_wd(model, tokenizer, examples, device, steps, learning_rate,
                 weight_decay, wd_zero_from=0, horizon=None, decay=False,
                 warmup=WARMUP, batch_size=BATCH, seed=0, hook=None, hook_every=0):
    """The stage-1 training of `p17_common.fine_tune` with a late weight decay.

    With `wd_zero_from` above zero the weight decay of the optimizer is set to
    zero from that step on; the optimizer is rebuilt so that the change takes
    effect at once. The loss of every step is returned.
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay, betas=BETAS)
    ids, labels = prepare_examples(examples, tokenizer, device)
    generator = torch.Generator(device=ids.device).manual_seed(int(seed))
    losses = []
    wd_off = False
    for step in range(1, steps + 1):
        if wd_zero_from and step == int(wd_zero_from):
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                          weight_decay=0.0, betas=BETAS)
            wd_off = True
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


def top_frequencies(rows, count=5):
    """The `count` strongest frequencies of the mean-removed rows."""
    _fraction, frequencies = spectrum_nodc(rows)
    return frequencies[:count]
