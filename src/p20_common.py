"""Shared helpers of queue 20: the partial removal of a frequency.

The spectrum is the one of queue 19: the mean over the p symbol tokens is
removed, the real transform is taken along the residue axis and the energy of
every frequency is summed over the hidden dimensions. The zero frequency is
left out of the total.

A frequency is removed in part: a factor f removes the share f * f of the
energy of that frequency, so the component of the frequency is scaled by
sqrt(1 - f * f) and the mean is left alone. A factor of 1 is the full removal
of queue 18 and of queue 19.

Produces: the partial removal of a frequency of queue 20.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from p19_common import (energy_fraction, matched_sets, ranked_frequencies,  # noqa: E402,F401
                        spectrum_energy)


def scale_frequencies(rows, factors):
    """The rows with the given frequencies scaled to keep a share of their energy.

    `factors` holds the frequency and the factor f of every frequency that is
    changed; the component of the frequency is multiplied by sqrt(1 - f * f), so
    the energy that the change removes is f * f times the energy of the
    frequency. The zero frequency is not touched unless it is in `factors`, so
    the mean of the rows stays the same.
    """
    spec = torch.fft.rfft(rows, dim=0)
    for frequency, factor in factors.items():
        index = int(frequency)
        if 0 <= index < spec.shape[0]:
            scale = math.sqrt(max(0.0, 1.0 - float(factor) ** 2))
            spec[index] = spec[index] * scale
    return torch.fft.irfft(spec, n=int(rows.shape[0]), dim=0)


def total_energy(energy):
    """The energy of the non-zero frequencies."""
    return float(energy[1:].sum())


def removed_fraction(energy_before, rows_after):
    """The share of the non-zero energy that the edit removed.

    The energy is measured on the edited rows, so the value is the one of the
    weights that were actually changed.
    """
    before = total_energy(energy_before)
    after = total_energy(spectrum_energy(rows_after))
    if before <= 0:
        return 0.0
    return (before - after) / before
