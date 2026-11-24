# Transfer Follows Generalization: Grokking Makes a Network a Useful Starting Point

**William Gatlin**

**The paper**: <https://zenodo.org/records/24489609>

The code and the results of the paper. The question is when a network that has
grokked a task can be reused for another one. The answer of the paper is that
the reuse follows the generalization of the source: a network that has grokked
the modular addition of invented symbols and that generalizes on held-out pairs
transfers to a related target, and a network that has only memorized its
training pairs does not, whatever the budget of the transfer.

The repository holds the shared code (`src/`), one entry script per item of
the paper (`experiments/`), the scripts that draw the figures and build the
tables (`figures/`), the results of every item (`results/`), the data of the
task (`data/`) and the tests (`tests/`).

The checkpoints of the runs are not in the repository: every number of the
paper comes from the CSV and JSON files of `results/`, and the weights can be
regenerated from the seeds of the scripts.

## Install

```
python -m venv .venv
. .venv/bin/activate            # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

The runs of the paper were made with Python 3.12.16, torch 2.15.0+cu132,
CUDA 13.2, transformers and peft. The environment that the archive of the runs
records is in `environment.txt`; the versions of the local environment are in
`results/reports/` and in the `runtime` block of the summaries of the runs.

The hardware of the runs was one NVIDIA GeForce RTX 5090 (32 GB). The
experiments of the paper need that card: the Pythia-160m and Pythia-410m
stages are trained on it, and the small-transformer experiments run on it too.
The figures, the tables and the smoke test run on any machine.

## The items of the paper

| item | the command that produces it | the input files | the command that draws it | the runtime of the run |
|---|---|---|---|---|
| Table 1 (the transfer by the stage of the source) | `experiments/run_seed.py` (p=59), the `runs_nseed_p97` and `runs_nseed_L2` summaries, `experiments/queue9_steps.py` and `queue10_steps.py` (Pre-LN) | `results/summary_runs_nseed.csv`, `results/table1/runs_nseed_p97/seed_*/summary.json`, `results/table1/runs_nseed_L2/seed_*/summary.json`, `results/table1/p1_preln.csv`, `results/table1/k2_preln.csv` | `python figures/make_tables.py` | 2 to 5 hours on the RTX 5090 for the ten seeds of each family |
| Table 2 (the LayerNorm by batch-size 2x2) | `experiments/p21_lnbatch.py` | `results/table2/m21_lnbatch.csv` | `python figures/make_tables.py` | 1 hour on the RTX 5090 for the two new cells |
| Table 3 (the task matrix) | `experiments/q16_toy.py`, `q17_toy.py`, `q18_toy.py`, `q19_toy.py` | `results/table3/b1_matrix.csv`, `t5_hard.csv`, `t18_poly.csv`, `t19_poly.csv`, `t18_parity.csv`, `t19_parity.csv` | `python figures/make_tables.py` | 1 to 3 hours on the RTX 5090 |
| Table 4 (the component factorial) | `experiments/run_factorial.py` (the 16 cells of the factorial, 9 seeds) and `experiments/run_components.py` (the 11 component hybrids, 6 seeds) | `results/table4/factorial/seed_*/summary.json` (the 9 seeds), `results/summary_runs_nseed.csv` (the run kind `components`), `results/table4/components/seed_*/summary.json` | `python figures/make_tables.py` | 3 hours on the RTX 5090 for the 9 seeds of the factorial |
| Figure 1 (the transfer against the held-out accuracy) | `experiments/run_sweep.py`, `experiments/queue2_steps.py` | `results/fig1/sweep_summary_p59.csv`, `sweep_summary_p97.csv`, `sweep_summary_two_layers.csv` | `python figures/make_fig1.py` | 3 hours on the RTX 5090 for the three sweeps |
| Figure 2 (the specificity) | `experiments/run_specificity.py`, `experiments/run_specificity_aged.py` | `results/summary_runs_nseed.csv`, `results/specificity/*/seed_*/summary.json` | `python figures/make_figs3.py` | 1 hour on the RTX 5090 |
| Figure 3 (the LoRA checkpoint sweep) | `experiments/queue12_steps.py`, `experiments/pythia_transfer.py` | `results/fig3/pythia_ckpt_sweep.csv` | `python figures/make_figs3.py` | 2 hours on the RTX 5090 |
| Figure 4 (the Pythia trajectories) | `experiments/p18_traj.py` | `results/fig4/p18_traj.csv`, `results/fig4/p18_direct.csv` | `python figures/make_figs2.py` | 8 hours on the RTX 5090 for the 410M configuration |
| Figure 5 (the frequency edits) | `experiments/p19_edits.py`, `p20_edits.py`, `p21_energy.py` | `results/fig5/m19_edits.csv`, `m20_edits.csv`, `m21_energy.csv`, `m18_fourier.csv`, `m17_fourier_seed.csv` | `python figures/make_figs2.py` | 2 hours on the RTX 5090 |
| Figure 6 (the stage-2 learning rate) | `experiments/p19_lr.py` | `results/fig6/p19_lr.csv`, `results/fig6/p19_direct_lr.csv` | `python figures/make_figs2.py` | 2 hours on the RTX 5090 |
| Section 3 (the timing) | `experiments/run_sweep.py`, `experiments/q19_toy.py` | `results/section3/runs_nseed_fine/seed_*/metrics.csv`, `results/section3/t19_sub.csv` | - | 1 hour on the RTX 5090 |
| Section 5 (the key-frequency removal) | `experiments/queue3_steps.py`, `queue4_steps.py` | `results/section5/queue3_summary.csv`, `queue4_summary.csv`, `embed_ablation.csv` | - | 2 hours on the RTX 5090 |
| the long budget and the new targets (queue 22) | `experiments/q22_run.py` | `results/queue22/q22_curves.csv`, `results/queue22/q22_targets.csv` | - | 3 hours on the RTX 5090 |

The figures are written to `figs_out/` as a PNG and a PDF; the tables are
written to `figs_out/table1_data.csv` and `figs_out/fig2_data.csv`.

## The layout

```
src/          the shared code: the settings, the model, the stage-1 training,
              the stage-2 transfer, the direct baseline and the helpers
experiments/  one entry script per item of the paper
figures/      the scripts that draw the six figures and build the tables
results/      the CSV and JSON files of every item, with results/README.md
data/         the symbol map of the invented symbols
tests/        the unit tests of the model and the smoke test
```

## The tests

```
python tests/smoke_test.py       # five steps of the small transformer on the CPU
python -m pytest tests/test_nseed.py -q
```

The smoke test needs nothing but torch: it builds the small transformer, the
data and the split, takes five steps and writes nothing outside a temporary
folder. It never downloads a model.

## The paper

**Transfer Follows Generalization: Grokking Makes a Network a Useful Starting
Point** — William Gatlin — <https://zenodo.org/records/24489609>

## The license

MIT, see `LICENSE`.
