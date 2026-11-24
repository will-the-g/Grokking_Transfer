# CLEANUP_REPORT.md

The public release of the project was built in `consolidation_reuse/release/`
from the laptop copy only: the mirror `archive_final/`, the project folder
`consolidation_reuse/` and `consolidation_reuse/archive_instance/`. Nothing was
deleted, moved or renamed, and the instance was not started.

**The release**: 208 files, 3.41 MB, no checkpoint, no absolute path, no name of
a tool of the instance.

---

## 1. What was kept, and why

### The results (`release/results/`, 135 files)

One folder per item of the paper, so that a reader can find the file of a
number without knowing the queue that produced it. The file that is read by a
figure or by a table is marked as such in `results/README.md`.

| the item | the files | the source in the archive |
|---|---|---|
| Table 1 | `summary_runs_nseed.csv` (the fuller copy of `runs_nseed/summary.csv`, 452 rows), `table1/runs_nseed_p97/seed_*/summary.json` (9), `table1/runs_nseed_L2/seed_*/summary.json` (5), `table1/p1_preln.csv` (queue 9, 17 rows), `table1/k2_preln.csv` (queue 10, 20 rows) | `archive_instance/runs_nseed/`, `archive_instance/runs_nseed_p97/`, `archive_instance/runs_nseed_L2/`, `archive_instance/runs_nseed_p1/`, `archive_instance/runs_nseed_k2/` |
| Table 2 | `table2/m21_lnbatch.csv` (74 rows) | `archive_final/runs_nseed_queue21/` |
| Table 3 | `table3/b1_matrix.csv` (the 3,060-row version, not the packed 300-row subset), `t5_hard.csv` (236), `t18_poly.csv` (135), `t19_poly.csv` (180), `t18_parity.csv` (207), `t19_parity.csv` (54) | `archive_instance/runs_nseed_queue13/`, `archive_instance/runs_nseed_queue17/`, `archive_final/runs_nseed_queue18/`, `archive_final/runs_nseed_queue19/` |
| Table 4 | `table4/factorial/seed_*/summary.json` and `metrics.csv` (the 9 seeds of the factorial, 23 cells), `summary_runs_nseed.csv` (the run kind `components`, 6 seeds), `table4/components/seed_*/summary.json` and `metrics.csv` | `archive_instance/runs_nseed/factorial/`, `archive_instance/runs_nseed/components/` |
| Figure 1 | `fig1/sweep_summary_p59.csv` (639 rows), `sweep_summary_p97.csv` (262), `sweep_summary_two_layers.csv` (60) | `archive_instance/runs_nseed_fine/`, `archive_instance/runs_nseed_p97/sweep/`, `archive_instance/runs_nseed_L2/sweep/` |
| Figure 2 | `summary_runs_nseed.csv`, `specificity/{specificity,specificity_aged,specificity_aged_init100,specificity_aged_init200}/*/summary.json` (33 files) | `archive_instance/runs_nseed/` |
| Figure 3 | `fig3/pythia_ckpt_sweep.csv` (the 540-row version, not the packed 30-row subset) | `archive_final/runs_nseed_queue12/` |
| Figure 4 | `fig4/p18_traj.csv` (1,296 rows), `p18_direct.csv` (60) | `archive_final/runs_nseed_queue18/` |
| Figure 5 | `fig5/m19_edits.csv` (300), `m20_edits.csv` (400), `m21_energy.csv` (30), `m18_fourier.csv` (720), `m17_fourier_seed.csv` (67) | `archive_final/runs_nseed_queue19/`, `.../queue20/`, `.../queue21/`, `.../queue18/` |
| Figure 6 | `fig6/p19_lr.csv` (200), `p19_direct_lr.csv` (40) | `archive_final/runs_nseed_queue19/` |
| Section 3 | `section3/runs_nseed_fine/seed_*/metrics.csv` and `summary.json` (the 6 seeds), `section3/t19_sub.csv` (56) | `archive_instance/runs_nseed_fine/`, `archive_final/runs_nseed_queue19/` |
| Section 5 | `section5/queue3_summary.csv`, `queue4_summary.csv`, `embed_ablation.csv` | `archive_final/` |
| queue 22 | `queue22/q22_curves.csv` (600), `q22_targets.csv` (180), `q22_source_check.csv` (20) | `consolidation_reuse/results_queue22/` |
| the reports | `results/reports/` (10 reports of the queues 9, 10, 12, 13, 17, 18, 19, 20, 21 and 22), with the absolute paths and the names of the tools of the instance replaced | `archive_final/` and `archive_final/nseed_backbone/` |

Where two copies of a file existed, the fuller one was taken: the 452-row
`runs_nseed/summary.csv` and not the 218-row one, the 3,060-row
`b1_matrix.csv` and not the 300-row packed subset, the 540-row
`pythia_ckpt_sweep.csv` and not the 30-row packed subset.

### The code (`release/src/` 15 modules, `release/experiments/` 24 entry scripts and 8 runners, `release/figures/` 4 scripts, `release/tests/` 2 files)

* `src/` holds the shared code: the settings and the helpers (`common.py`), the
  model (`core.py`, `model_phase3.py`), the Pythia loader (`p15_common.py`), the
  stage-1 training, the stage-2 transfer and the direct baseline
  (`p17_common.py`), the trajectory and the spectrum helpers (`p17_traj.py`,
  `p18_common.py`, `p19_common.py`, `p20_common.py`), the target rules of queue
  22 (`p22_targets.py`), the worker pool (`parallel.py`) and the modules that
  the entries import (`p16_pilot.py`, `fourier_metrics.py`, `q14_steps.py`,
  `queue8_steps.py`).
* `experiments/` holds one entry script per item of the paper, with the line
  that says which item it produces at the top of its docstring, and the runner
  scripts of the queues that the paper uses (9, 10, 12, 13, 21 and 22).
* `figures/` holds the four attached scripts, repointed at `results/` and
  `figs_out/`.
* `tests/` holds the unit tests of the model (`test_nseed.py`) and the smoke
  test.

### The rest

`data/symbols_p97.json` (the map of the invented symbols), `README.md`,
`LICENSE` (MIT, William Gatlin), `requirements.txt`, `environment.txt` (the
environment that the archive of the runs records) and `results/README.md`.

---

## 2. What was left out, and why

| what | the files | the size | why |
|---|---|---|---|
| the model checkpoints (`.pt`) | 12,079 | 17,190 MB | the numbers of the tables and of the figures come from the CSV and JSON files; the weights are regenerated by the scripts from the seeds |
| the local virtual environment | 32,638 | 3,462 MB | the third-party packages, rebuilt by pip |
| the byte code caches | 10,874 | 243 MB | rebuilt on demand |
| the backup archives | 2 | 143 MB | the backups of the project before a cleanup |
| the smoke runs | 149 | 88 MB | the wiring tests, superseded by the full runs |
| the partial laptop copies | 116 | 79 MB | the partial copy of a run, superseded by the full run |
| the run logs | 64 | 26 MB | the record of the runs, not an input of the paper |
| the result archives of the earlier queues | 39 | 26 MB | the loose CSVs of the release replace them |
| **the total left out** | **55,961** | **21,259 MB** | |

The following were left out on purpose, following the plan: `reference/`,
`diag_properties.csv`, `cold_factorial/`, `scratch/`, `pythia_traj.csv`,
`toy_learning_curves.csv`, `benchmark*.json`, `pack_*.py`, `monitor_*.sh`,
`status.py`, `fourier_metrics.csv`, `body_metrics.csv`, `attention_stats.csv`
and `p18_collapse.csv`. Figure 6 and Appendix A read `p19_lr.csv`, not
`p18_collapse.csv`, so the late-decline study stays out; it is still in the
archive if it is ever needed.

---

## 3. What is still UNMAPPED

* the **LoRA sweep of the 2.8B model** (`runs_nseed_r2c/pythia_transfer_r2c.csv`
  and `runs_nseed_r2d/pythia_transfer_r2d.csv`): no figure or table of the
  release reads them, and the figure of the checkpoint sweep (Figure 3) uses the
  1B and the 1.4B models of queue 12. The files stay in the archive.
* the **mechanism analysis of the grokked model** (`fourier_metrics.csv`,
  `body_metrics.csv`, `attention_stats.csv`): they fit a mechanism section, but
  no item of the list of the paper reads them.
* the **late decline of the transfer** (`p18_collapse.csv`): left out with the
  rest of the plan; no number of the list is read from it.
* the **python reproduction of the older figures** (`reference/`): the input of
  a comparison, not an item.
* the **9-seed version of the component factorial is not a mystery**: it is
  `archive_instance/runs_nseed/factorial/`, with the seeds 5, 6, 7, 8, 9, 10,
  11, 13 and 14 and the 23 cells in the `jobs` key of `summary.json`. The run
  kind `components` of `summary.csv` has only the 6 seeds 5 to 10 and the 11
  component hybrids; the paper's Table 4 comes from the factorial, and all
  its 10 cells match (see the verification below).

---

## 4. Every path and every secret that was found, and what was done

| where | what | what was done |
|---|---|---|
| `experiments/p21_lnbatch.py` | `/workspace/laptop_copy/runs_nseed`, `/workspace/consolidation_reuse/results_queue9.zip`, `--workspace /workspace` | replaced by `--existing-root`, which defaults to `results/existing` |
| `src/common.py` | the literal `C:\Windows` (the fallback of the system root) | replaced by `os.environ.get('SystemRoot') or os.path.join(os.sep, 'Windows')`; the behaviour on Windows is the same |
| `experiments/run_p21_a.sh`, `run_p21_b.sh`, `run_q22_e1.sh`, `run_q22_e2.sh` | `HF_HOME=/workspace/.hf_home`, `--runs-root /workspace/laptop_copy/runs_nseed` | `HF_HOME` now defaults to the home of the user; the root of the existing cells is `${EXISTING_ROOT:-results/existing}` |
| `results/reports/REPORT_queue22.md` | `/workspace/consolidation_reuse`, `/workspace/.hf_home` | replaced by `<repo>` and `<instance>` |
| `results/reports/REPORT_queue20.md` | `vastai`, `ssh` | replaced by "the instance command line tool" and "the remote shell" |
| `figures/make_fig1.py`, `make_figs2.py`, `make_figs3.py`, `make_tables.py` | `/tmp/claude-0/-home-claude/.../scratchpad/`, `/home/claude/figs_out/`, `/home/claude/q18/`, `/home/claude/q19/`, `/home/claude/q20/` | replaced by `results/` and `figs_out/`, relative to the root of the repository |
| every report of the release | `/home/claude/`, `/tmp/claude`, `C:\Users\...`, `wgatl` | replaced by `<scratch>`, `<home>` and `<user>` |

The runner scripts of the queues that were kept (`run_queue9.sh`,
`run_queue10.sh`, `run_queue12.sh`, `run_queue13.sh`) had no instance command
and no absolute path. The scripts that had the stop command of the instance
(`run_queue2.sh`, `run_queue3.sh`, `run_queue4.sh`, `run_queue8.sh`,
`run_overnight.sh`) were **not** kept, so no command of the instance remains in
the release.

The final grep of the release for `wgatl`, `C:\`, `/workspace`, `/home/claude`,
`/tmp/claude`, `vastai`, `CONTAINER_` and `ssh` returns **0 hits** over the 46
python files, the 8 shell scripts, the 10 reports, the 135 result files and
the documentation.

---

## 5. The verification, and every mismatch

| the check | the result |
|---|---|
| `python -m py_compile` on every .py | 46 files, 0 failures |
| the grep of the sensitive strings | 0 hits |
| `python tests/smoke_test.py` | 5 steps of the small transformer on the CPU: the loss goes from 4.1980 to 3.9284, the checkpoint round-trip works, nothing is written outside a temporary folder and no model is downloaded |
| the six figures | `figs_out/` holds fig1_sweep, fig2_specificity, fig3_pythia_checkpoints, fig4_pythia_trajectories, fig5_frequency_edits and fig6_stage2_lr, each as a PNG and a PDF |
| the tables | `figs_out/table1_data.csv` and `figs_out/fig2_data.csv` are written; Table 1 and Table 3 are printed |

**The numbers, recomputed from `release/results/`:**

| the item | the release | the paper | |
|---|---|---|---|
| Table 1, p=59 one layer | memorized 0.019, norm-matched 0.031, transition 0.525, grokked 0.805 (9 seeds) | 0.019, 0.031, 0.525, 0.805 (9) | the same |
| Table 1, p=97 | 0.021, 0.036, 0.937, 0.964 (9) | 0.021, 0.036, 0.937, 0.964 (9) | the same |
| Table 1, two layers | 0.024, 0.034, 0.667, 0.700 (5) | 0.024, 0.034, 0.667, 0.700 (5) | the same |
| Table 1, Pre-LN | 0.300, 0.355, 0.739, 0.885 (9, 9, 10, 9) | 0.300, 0.355, 0.739, 0.885 (9-10) | the same |
| Table 2 | none/full grokked 0.791, none/full memorized@2k 0.020, none/512 norm-matched@20k 0.303, Pre-LN/full grokked 0.936, Pre-LN/full memorized@20k 1.000, Pre-LN/full norm-matched@20k 0.497, Pre-LN/512 grokked 0.904 | the same | the same |
| Table 3 | addition->addition 0.817, mult->mult 0.736, S5 0.998, polynomial 0.423, parity 0.999 | 0.817, 0.736, 0.998, 0.423, 0.999 | the same |
| Table 4 | none 0.018, unembedding 0.029, numeric embedding 0.208, MLP 0.240, attention 0.460, numeric embedding + unembedding 0.499, attention + MLP 0.766, attention + MLP + unembedding 0.825, all four 0.856, embedding rows permuted 0.054 (9 seeds) | the same (9 seeds) | the same |
| Figure 1 | 639 / 262 / 60 points, the rank correlations 0.951, 0.889, 0.930 | 639 / 262 / 60, 0.95, 0.89, 0.93 | the same |
| queue 22 | grokked a+2b at 3,000 steps 0.7156, a+3b 0.7808, a-2b 0.6628 | 0.7156, 0.7808, 0.6628 | the same |

**There is no mismatch between the release and the paper.** Two points need a
word of explanation:

* the Table 4 of the paper says 9 seeds, and the 9-seed factorial does exist,
  in `archive_instance/runs_nseed/factorial/`. The run kind `components` of
  `summary.csv` has only 6 seeds (5 to 10) and the 11 component hybrids; it is
  a different study and it is not the source of the table. Every cell of the
  paper's Table 4 matches the factorial of 9 seeds to three decimals.
* the Pre-LN row of Table 1 has 9 seeds for the memorized, the norm-matched and
  the grokked stage and 10 for the transition stage; the paper says "9-10", so
  the two agree.

**The changes that were made to the attached scripts, and the fixes of the
build:**

1. `make_fig1.py`, `make_figs2.py`, `make_figs3.py` and `make_tables.py` were
   repointed at `results/` and `figs_out/`; the numbers, the labels and the
   computations are untouched.
2. `make_tables.py` and `make_figs3.py` now read **both** `p1_preln.csv` and
   `k2_preln.csv` for the Pre-LN row of Table 1, as the corrected mapping asks;
   the attached scripts read only the file of queue 9, although the code called
   the row "combined".
3. `make_tables.py` reads the transfers of the specificity runs from the `jobs`
   key as well as from the `transfers` key: the specificity summaries hold their
   transfers under `jobs`, so the block was empty with the attached script. The
   figure of the specificity (Figure 2) reads the values from `summary.csv` and
   was not touched.
4. Three result files (`fig1/sweep_summary_p59.csv`,
   `fig1/sweep_summary_p97.csv` and `fig1/sweep_summary_two_layers.csv`) were
   copied from the wrong source by a fallback of the build; they were put back
   and checked by sha256 against the archive (639, 262 and 60 rows).
5. A parenthesis was missing in the repointed `make_tables.py`, and the paths
   of the `Path` objects were mixed with strings in two places; they were fixed
   and the script now runs.

---

## 6. The size and the file count

| the part | the files | the size |
|---|---|---|
| the code (`src/`, `experiments/`, `figures/`, `tests/`) | 54 | 0.767 MB |
| the results (`results/`) | 135 | 1.436 MB |
| the documentation (`README.md`, `LICENSE`, `requirements.txt`, `environment.txt`, `results/README.md`, the reports) | 18 | 1.206 MB |
| the data (`data/symbols_p97.json`) | 1 | 0.004 MB |
| **the total** | **208** | **3.414 MB** |

The target was under 100 MB and nothing that `results/` needs was left out.
The archive `consolidation_reuse/release.zip` holds the 208 files: 1,523,437
bytes, sha256 `5a89ac8582e1ca5e9aaa48b9ad382ad520b37405e3ef0125a8fdd80f156bcf7d`,
its integrity checked and every member compared with the file on the disk.
