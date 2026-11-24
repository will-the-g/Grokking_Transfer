"""Shared helpers of queue 19: the energy of the symbol spectrum and the
energy-matched sets of frequencies.

The spectrum is the one of queue 18: the mean over the p symbol tokens is
removed, the real transform is taken along the residue axis and the energy of
every frequency is summed over the hidden dimensions. The zero frequency is
left out of the total, so a fraction is always a share of the non-zero energy.

The energy-matched sets of queue 19 are five frequencies of the same non-key,
non-zero pool whose removed share of the energy is close to the share of the
key set, in the input embeddings and in the unembedding at the same time.

Produces: the spectrum and the energy-matched sets of queue 19.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from p15_common import free_disk_gb  # noqa: E402,F401
from p18_common import apply_edit, remove_frequencies, spectrum_nodc  # noqa: E402,F401

DRAWS = 5000
TOLERANCES = (0.20, 0.35)


def spectrum_energy(rows):
    """The energy of every frequency of the mean-removed symbol rows.

    The index 0 is the zero frequency, the index k is the frequency k.
    """
    centered = rows - rows.mean(dim=0, keepdim=True)
    spec = torch.fft.rfft(centered, dim=0)
    return (spec.real ** 2 + spec.imag ** 2).sum(dim=1)


def energy_fraction(energy, frequencies):
    """The share of the non-zero energy that the given frequencies hold."""
    nonzero = energy[1:]
    total = float(nonzero.sum())
    if total <= 0:
        return 0.0
    removed = 0.0
    for frequency in frequencies:
        if 1 <= int(frequency) < int(energy.shape[0]):
            removed += float(energy[int(frequency)])
    return removed / total


def ranked_frequencies(energy, count=5, skip=0):
    """The `count` strongest frequencies of the spectrum, after the first `skip`."""
    order = torch.argsort(energy[1:], descending=True) + 1
    return sorted(int(value) for value in order[skip:skip + count])


def matched_sets(energy_in, energy_out, key, p, seed, step):
    """The energy-matched random sets of five frequencies.

    Up to `DRAWS` random sets of five frequencies of the non-key, non-zero
    pool are drawn; a set is accepted when its removed share of the energy is
    within the tolerance of the share of the key set, in the input embeddings
    and in the unembedding at the same time. The first three accepted sets are
    returned. When fewer than three are found the tolerance is relaxed and the
    tolerance that was used is returned with them.
    """
    available = [k for k in range(1, p // 2 + 1) if k not in key]
    if len(available) < 5:
        return [], 'none', 0
    key_in = energy_fraction(energy_in, key)
    key_out = energy_fraction(energy_out, key)
    generator = random.Random(f'q19|{seed}|{step}|em')
    found = []
    tolerance = f'{int(TOLERANCES[-1] * 100)}%'
    for limit in TOLERANCES:
        found = []
        seen = set()
        for _ in range(DRAWS):
            draw = tuple(sorted(generator.sample(available, 5)))
            if draw in seen:
                continue
            seen.add(draw)
            fraction_in = energy_fraction(energy_in, draw)
            fraction_out = energy_fraction(energy_out, draw)
            if abs(fraction_in - key_in) <= limit * key_in and \
                    abs(fraction_out - key_out) <= limit * key_out:
                found.append(list(draw))
                if len(found) == 3:
                    break
        tolerance = f'{int(limit * 100)}%'
        if len(found) == 3:
            return found, tolerance, 3
    return found, tolerance, len(found)


def edit_plan(energy_in, energy_out, p, seed, step):
    """The edits of one step.

    Every entry holds the name, the part of the model that is edited, the
    frequencies that are removed, the tolerance that was used to find the
    energy-matched sets and the share of the energy that the edit removes in
    the input embeddings and in the unembedding.
    """
    key = ranked_frequencies(energy_in, 5)
    rank6_10 = ranked_frequencies(energy_in, 5, skip=5)
    key_in = energy_fraction(energy_in, key)
    key_out = energy_fraction(energy_out, key)
    plan = [('none', 'none', [], '', 0.0, 0.0),
            ('key_both', 'both', key, '', key_in, key_out),
            ('key_in', 'input_embedding', key, '', key_in, 0.0),
            ('key_out', 'unembedding', key, '', 0.0, key_out),
            ('rank6_10_both', 'both', rank6_10, '',
             energy_fraction(energy_in, rank6_10),
             energy_fraction(energy_out, rank6_10))]
    sets, tolerance, found = matched_sets(energy_in, energy_out, key, p, seed, step)
    for index in range(found):
        draw = sets[index]
        fraction_in = energy_fraction(energy_in, draw)
        fraction_out = energy_fraction(energy_out, draw)
        name = 'abcdefg'[index]
        plan.append((f'em_both_{name}', 'both', draw, tolerance,
                     fraction_in, fraction_out))
        plan.append((f'em_in_{name}', 'input_embedding', draw, tolerance,
                     fraction_in, 0.0))
        plan.append((f'em_out_{name}', 'unembedding', draw, tolerance,
                     0.0, fraction_out))
    return plan
