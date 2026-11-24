"""Queue 22: the new target rules and the stage-2 curve of the invented-symbol task.

The setup is the one of the config A of queue 17 and queue 19 (pythia-160m,
p = 59, coverage 0.5, lr 3e-4, wd 1.0, the `gen` source with the N generated
pairs and the `mem` source with the first 100 pairs of the same pool).

The rules of the stage 2 are the ones of `p17_common` with two new ones: the
a+3b and the a-2b targets, modulo the same prime. The held-out split of the
transfer is the one of the existing runs and it is not changed: the first 500
pairs of `splits['heldout']`, the same split of the pairs that the stage-2
training never sees, scored with the labels of the target.

`stage2_curve` is `p17_common.stage2_accuracy` with the held-out accuracy
measured every `eval_every` steps instead of only at the end. The protocol is
the same: the first 500 pairs of the train pool, the batch 16, the warmup 0,
the learning rate and the weight decay of the caller, and the seed of the
trajectory.

    python nseed_backbone/q22_run.py --help

Produces: the new target rules and the stage-2 curve of queue 22.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from p15_common import token_accuracy  # noqa: E402
from p17_common import (STAGE2_BATCH, STAGE2_HELDOUT, STAGE2_PAIRS,  # noqa: E402
                         fine_tune, splits_for)

# the rules of the stage 2: the name and the residue of the answer
RULES = {
    'add': lambda a, b, p: (a + b) % p,
    'add2': lambda a, b, p: (a + 2 * b) % p,
    'add3': lambda a, b, p: (a + 3 * b) % p,
    'sub': lambda a, b, p: (a - b) % p,
    'sub2': lambda a, b, p: (a - 2 * b) % p,
    'mult': lambda a, b, p: (a * b) % p,
}
# the names of the new targets
NEW_TARGETS = ('add3', 'sub2')
# the target of the existing runs
BASE_TARGET = 'add2'


def make_examples_rule(symbols, pairs, p, rule='add'):
    """The (prompt, answer) examples of a rule at the size p.

    The prompt ends with the equals sign and the answer follows it after a
    space, as in `p17_common.make_examples`.
    """
    if rule not in RULES:
        raise ValueError(f'unknown rule {rule!r}')
    function = RULES[rule]
    prompts = [f'{symbols[str(a)]} {symbols[str(b)]} =' for a, b in pairs]
    answers = [symbols[str(function(a, b, p))] for a, b in pairs]
    return list(zip(prompts, answers))


def answer_ids_rule(payload, pairs, p, rule='add'):
    """The answer token ids of the pairs."""
    if rule not in RULES:
        raise ValueError(f'unknown rule {rule!r}')
    function = RULES[rule]
    token_ids = payload['token_ids']
    return [token_ids[str(function(a, b, p))] for a, b in pairs]


def stage2_pairs(splits):
    """The training pairs and the held-out pairs of the stage 2.

    The held-out split is the one of the existing runs: the first 500 pairs of
    `splits['heldout']`, which the stage-1 training never sees.
    """
    pairs = splits['train_pool'][:STAGE2_PAIRS]
    heldout = splits['heldout'][:STAGE2_HELDOUT]
    return pairs, heldout


def stage2_curve(model, tokenizer, payload, splits, p, device, learning_rate,
                 weight_decay, rule, seed, steps, eval_every=500,
                 batch_size=STAGE2_BATCH):
    """The stage-2 transfer; returns the (step, held-out accuracy) curve.

    The protocol is the one of `p17_common.stage2_accuracy`: the first 500
    pairs of the train pool, the batch 16, the warmup 0 and the seed of the
    trajectory. The held-out accuracy of the first 500 pairs of the held-out
    split is measured every `eval_every` steps.
    """
    pairs, heldout = stage2_pairs(splits)
    examples = make_examples_rule(payload['symbols'], pairs, p, rule)
    prompts = [f'{payload["symbols"][str(a)]} {payload["symbols"][str(b)]} ='
               for a, b in heldout]
    ids = answer_ids_rule(payload, heldout, p, rule)
    curve = []

    def hook(step, _losses):
        model.eval()
        accuracy = token_accuracy(model, tokenizer, prompts, ids, device)
        model.train()
        curve.append((step, accuracy))

    fine_tune(model, tokenizer, examples, device, steps=steps,
              learning_rate=learning_rate, weight_decay=weight_decay, warmup=0,
              batch_size=batch_size, seed=seed, hook=hook, hook_every=eval_every)
    model.eval()
    return curve


def cpu_state(model):
    """The state of a model on the CPU, so it can be restored later."""
    return {key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()}
