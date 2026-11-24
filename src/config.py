"""The paths of the repository.

Every script reads its inputs and writes its outputs through these paths, so
that a copy of the repository is enough to reproduce the figures and the tables.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "figures"
FIGS_OUT_DIR = REPO_ROOT / "figs_out"
SOURCE_DIR = REPO_ROOT / "src"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def results_path(*parts):
    """A path inside results/."""
    return RESULTS_DIR.joinpath(*parts)


def data_path(*parts):
    """A path inside data/."""
    return DATA_DIR.joinpath(*parts)
