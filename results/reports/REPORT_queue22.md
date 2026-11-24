# REPORT_queue22

Queue 22 on the GPU instance (an RTX 5090, torch 2.15.0+cu132, the interpreter
`/root/venv/bin/python`, the Hugging Face cache in `<instance>`): the
long-budget memorizers and the direct baselines, and two new target tasks of
the invented-symbol task of pythia-160m.

## The setup that was reused

Everything is the setup of the config A of queue 17 and of the step P2 of queue
19, found in `nseed_backbone/p19_lr.py`, `nseed_backbone/p17_common.py` and
`nseed_backbone/p15_common.py`:

| item | the value |
|---|---|
| the model | `EleutherAI/pythia-160m` at the revision `step143000`, in float32 |
| the symbols | the saved map `runs_nseed_queue20/symbols_p97.json` (the same file as the one of queue 18 and queue 19, sha256 `450f76c5ab436194`) |
| the task | the modular addition over the invented symbols at p = 59 |
| the source `gen` | the first 1,740 pairs of the train pool (the coverage 0.5) |
| the source `mem` | the first 100 pairs of the same pool |
| the optimizer | AdamW, the betas (0.9, 0.98), the warmup of 100 steps, the batch 128, the constant learning rate 3e-4, the weight decay 1.0, `decay=False` |
| the seed of the source | `seed * 7 + len(source)`, as in the scripts of queue 19 |
| the seeds | 0 to 4, the seeds of the existing 160m runs that have the `mem` source (`p19_lr.csv` holds 2 sources * 5 seeds * 5 steps * 4 learning rates) |
| the stage-2 protocol | the first 500 pairs of the train pool for the training, the batch 16, the warmup 0, the learning rate 3e-4, the weight decay 1.0, the seed of the trajectory |
| the held-out split of the transfer | the first 500 pairs of `splits['heldout']`, the same split of the pairs as the existing runs, unchanged for the new targets |

The held-out split is the one of `p17_common.splits_for(59)`: the 3,481 pairs
of the task are shuffled once with the seed 0, the first 1,000 are the held-out
split and the remaining 2,481 are the train pool. The stage 2 trains on the
first 500 pairs of the train pool and is scored on the first 500 pairs of the
held-out split, with the labels of the target. The two sets do not overlap.
The new targets use that split and that protocol unchanged; only the rule of
the answer is new.

The rules of the stage 2 are the ones of `p17_common` with two new ones: the
a+3b and the a-2b targets, modulo the same prime 59. They are in
`nseed_backbone/p22_targets.py`, which the existing file is not touched by.

## The exact commands that were run

```
cd <repo>
export HF_HOME=<instance>
export NSEED_DETERMINISTIC=1
export NSEED_DETERMINISTIC_GPU=1
export PYTHONUNBUFFERED=1

# experiment 2, the short one, first
bash run_q22_e2.sh
#   /root/venv/bin/python nseed_backbone/q22_run.py --experiment 2 --procs 4 \
#       --out runs_queue22/q22_targets.csv \
#       --check runs_queue22/q22_source_check.csv

# experiment 1, in the background
bash run_q22_e1.sh
#   /root/venv/bin/python nseed_backbone/q22_run.py --experiment 1 --procs 4 \
#       --out runs_queue22/q22_curves.csv \
#       --check runs_queue22/q22_source_check.csv

# the tables and the archive
/root/venv/bin/python nseed_backbone/q22_report.py --out runs_queue22/q22_tables.md
/root/venv/bin/python nseed_backbone/q22_pack.py --out results_queue22.zip
```

The two scripts are `run_q22_e2.sh` and `run_q22_e1.sh`; they write their rows
into the CSV as they come and their log into a file of its own, so a disconnect
does not lose the progress.

## The runtime

| step | the start | the end | the runtime |
|---|---|---|---|
| experiment 2 | 2026-11-23T05:42:44Z | 2026-11-23T06:35:22Z | 52 min 38 s |
| experiment 1 | 2026-11-23T06:40:59Z | 2026-11-23T08:37:19Z | 1 h 56 min 20 s |
| the queue | 2026-11-23T05:42:44Z | 2026-11-23T08:37:19Z | 2 h 54 min 35 s |

Four workers ran at the same time, one per seed; the card was at 94 to 97
percent and the free disk stayed at 12 GB during the whole queue.

## The rows of every file

| file | the rows | the expected rows | the status |
|---|---|---|---|
| q22_targets.csv | 180 | 5 seeds * 3 conditions * 2 targets * 6 evaluations | complete |
| q22_curves.csv | 600 | 5 seeds * 3 conditions * 40 evaluations | complete |
| q22_source_check.csv | 20 | 10 sources * 2 passes | complete |
| q22_tables.md | the tables of this report | | complete |

Nothing failed: the two passes report `5 seeds finished, 0 failed`.

## Experiment 1: the long budget on a+2b

The three conditions are transferred to the target a+2b for 20,000 steps and
the held-out accuracy of the transfer is measured every 500 steps: the
memorizing source (the `mem` source of 100 pairs), the direct baseline (the
pretrained model, no source) and the grokked source (the `gen` source of the
step 5,000).

| condition | seed | the first step over 0.9 | the best point | the accuracy at 3,000 | at 6,000 | at 8,000 | at 20,000 |
|---|---|---|---|---|---|---|---|
| grokked | 0 | 2000 | 0.9320 at 2000 | 0.7660 | 0.6720 | 0.5700 | 0.6820 |
| grokked | 1 | 2500 | 0.9840 at 2500 | 0.5460 | 0.5380 | 0.5020 | 0.4660 |
| grokked | 2 | 3000 | 0.9680 at 6000 | 0.9540 | 0.9680 | 0.9480 | 0.4000 |
| grokked | 3 | 3000 | 0.9540 at 3500 | 0.9280 | 0.6540 | 0.5560 | 0.6240 |
| grokked | 4 | 19500 | 0.9440 at 19500 | 0.3840 | 0.2760 | 0.2000 | 0.6660 |
| memorizing | 0 | never | 0.0120 at 4500 | 0.0000 | 0.0100 | 0.0080 | 0.0020 |
| memorizing | 1 | never | 0.0120 at 1000 | 0.0100 | 0.0060 | 0.0060 | 0.0060 |
| memorizing | 2 | never | 0.0140 at 15000 | 0.0080 | 0.0100 | 0.0060 | 0.0100 |
| memorizing | 3 | never | 0.0120 at 8500 | 0.0020 | 0.0040 | 0.0080 | 0.0020 |
| memorizing | 4 | never | 0.0140 at 2500 | 0.0140 | 0.0020 | 0.0060 | 0.0000 |
| direct | 0 | never | 0.0080 at 10500 | 0.0040 | 0.0000 | 0.0020 | 0.0020 |
| direct | 1 | never | 0.0080 at 500 | 0.0000 | 0.0020 | 0.0060 | 0.0020 |
| direct | 2 | never | 0.0080 at 14000 | 0.0000 | 0.0060 | 0.0000 | 0.0060 |
| direct | 3 | never | 0.0140 at 12500 | 0.0000 | 0.0020 | 0.0000 | 0.0060 |
| direct | 4 | never | 0.0120 at 17500 | 0.0040 | 0.0040 | 0.0040 | 0.0020 |

The means over the seeds:

| condition | the seeds | the first step over 0.9 (the mean of the seeds that reach it) | the seeds that never reach it | the best point (the mean) | the accuracy at 3,000 | at 6,000 | at 8,000 | at 20,000 |
|---|---|---|---|---|---|---|---|---|
| grokked | 5 | 6,000 | 0 | 0.9564 | 0.7156 | 0.6216 | 0.5552 | 0.5676 |
| memorizing | 5 | never | 5 | 0.0128 | 0.0068 | 0.0064 | 0.0068 | 0.0040 |
| direct | 5 | never | 5 | 0.0100 | 0.0016 | 0.0028 | 0.0024 | 0.0036 |

The three findings:

* **The long budget does not rescue the memorizing source.** The memorizing
  source never reaches 0.9, and its best point of the whole curve of 20,000
  steps is 0.0128 on the average against 0.0100 for the direct baseline. The
  accuracy at the step 20,000 is 0.0040, the same as at the step 3,000
  (0.0068): the budget of 20,000 steps, six times and two thirds the budget of
  3,000 steps of the existing runs, changes nothing for it.
* **The direct baseline is at chance for the whole budget.** It never reaches
  0.9 and its best point is 0.0100.
* **The grokked source transfers, and it does not need the long budget.** It
  passes 0.9 at the steps 2,000, 2,500, 3,000, 3,000 and 19,500 (the mean of
  the seeds that reach it is 6,000; the last one is a late spike of an
  oscillating curve whose value at the step 3,000 is 0.3840). The curves are
  not monotone: the accuracy of the grokked source is 0.7156 at the step 3,000
  and 0.5676 at the step 20,000, and the best point of the curve is 0.9564.
  The extra 17,000 steps buy nothing and cost accuracy.

## Experiment 2: the new target tasks

The final held-out accuracy of the transfer of the budget of 3,000 steps, as
the mean over the seeds with the smallest and the largest value, in the layout
of the existing table of the targets: the condition and the target in the rows,
the mean, the smallest and the largest in the columns. The row of the target
a+2b is read from the curve of experiment 1 at the step 3,000, so that the
three targets are read at the same budget of 3,000 steps.

| target | condition | n | the mean | the smallest | the largest |
|---|---|---|---|---|---|
| a+2b | grokked | 5 | 0.7156 | 0.3840 | 0.9540 |
| a+2b | memorizing | 5 | 0.0068 | 0.0000 | 0.0140 |
| a+2b | direct | 5 | 0.0016 | 0.0000 | 0.0040 |
| a+3b | grokked | 5 | 0.7808 | 0.6260 | 0.9380 |
| a+3b | memorizing | 5 | 0.0048 | 0.0040 | 0.0060 |
| a+3b | direct | 5 | 0.0024 | 0.0000 | 0.0040 |
| a-2b | grokked | 5 | 0.6628 | 0.3860 | 0.9440 |
| a-2b | memorizing | 5 | 0.0036 | 0.0020 | 0.0060 |
| a-2b | direct | 5 | 0.0016 | 0.0000 | 0.0040 |

The finding: **the grokked structure of the step-5,000 source carries to
target rules that it was never trained on.** The a+3b target transfers with a
mean of 0.7808 and the a-2b target with a mean of 0.6628, against 0.7156 for
the a+2b target that the existing runs used, while the memorizing source and
the direct baseline are at chance on the three targets (0.0016 to 0.0068). The
transfer of the a+2b target is therefore not a special property of the a+2b
rule: the structure that the source grokked is usable for the other rules of
the same group.

The seed 4 of the grokked source is the one that transfers the worst on the
a+2b target (0.3840) and the best on the a+3b target (0.9380); the curves of
the transfers are not monotone, so the value at the end of the budget is one
sample of an oscillating curve and the per-seed rows are in
`runs_queue22/q22_tables.md`.

## The check of the sources

The step-5,000 checkpoints of the existing runs are not on the instance: the
scripts of queue 19 and queue 20 wrote them to the work directory and deleted
them at the end of every trajectory, and no checkpoint of pythia-160m is left
anywhere on the instance (every checkpoint of the instance is smaller than
3 MB, the size of the small transformer). The sources were therefore trained
again with the same code path, and they were checked against the rows of the
existing runs: the held-out accuracy and the global parameter norm of the step
5,000 are the ones of `p19_lr.csv`, and the held-out accuracy is also the one
of the `none` rows of `m20_edits.csv`.

| seed | source | the held-out of this run | the existing rows | the difference | the parameter norm of this run | the existing norm | the difference |
|---|---|---|---|---|---|---|---|
| 0 | gen | 0.9610 | 0.9610 | +0.000000 | 421.3116 | 421.3116 | +0.000000 |
| 1 | gen | 0.9930 | 0.9930 | +0.000000 | 423.9987 | 423.9987 | +0.000000 |
| 2 | gen | 0.9990 | 0.9990 | +0.000000 | 419.2485 | 419.2485 | +0.000000 |
| 3 | gen | 0.9740 | 0.9740 | +0.000000 | 420.1474 | 420.1474 | +0.000000 |
| 4 | gen | 0.9830 | 0.9830 | +0.000000 | 418.2551 | 418.2551 | +0.000000 |
| 0 | mem | 0.0040 | 0.0040 | +0.000000 | 419.1876 | 419.1876 | +0.000000 |
| 1 | mem | 0.0060 | 0.0060 | +0.000000 | 419.1324 | 419.1324 | +0.000000 |
| 2 | mem | 0.0060 | 0.0060 | +0.000000 | 415.1717 | 415.1717 | +0.000000 |
| 3 | mem | 0.0070 | 0.0070 | +0.000000 | 408.1652 | 408.1652 | +0.000000 |
| 4 | mem | 0.0030 | 0.0030 | +0.000000 | 415.3798 | 415.3798 | +0.000000 |

The mean absolute difference of the held-out accuracy is **0.000000** and the
largest one is **0.000000**; the mean absolute difference of the parameter
norm is **0.000000** and the largest one is **0.000000**. The sources of this
queue are the sources of the existing runs, to the last digit.

## The disk events

The free disk was 12 GB at the start of the queue and 12 GB at the end of it.
Nothing was deleted, nothing was moved, and no file outside `runs_queue22/`,
the four new scripts of `nseed_backbone/`, the two runners and this report was
written.

| event | detail | size |
|---|---|---|
| FREE DISK | 12 GB at the start and 12 GB at the end of the queue | 0 bytes |
| CREATED | `runs_queue22/` with the CSVs, the tables and the logs | 168 KB |
| CREATED | `nseed_backbone/p22_targets.py`, `q22_run.py`, `q22_report.py` and `q22_pack.py` | 26.7 KB |
| KEPT | the Hugging Face cache, which the Pythia steps need | 2.2 GB |

## The failures, the skipped steps and the deviations

| time (UTC) | event |
|---|---|
| 2026-11-23T05:42:44Z | Experiment 2 starts with 4 workers: the config A of queue 17 and queue 19, the seeds 0 to 4, the `gen` and the `mem` sources, the targets a+3b and a-2b, the budget of 3,000 steps, the evaluation every 500 steps. |
| 2026-11-23T05:52:59Z | The `gen` sources of the seeds 0 to 3 are trained: their held-out accuracies and their parameter norms are the ones of the existing runs, to the last digit. |
| 2026-11-23T06:35:22Z | Experiment 2 is finished: 5 seeds, 0 failed, 180 rows. |
| 2026-11-23T06:40:59Z | Experiment 1 starts with 4 workers: the same sources, the target a+2b, the budget of 20,000 steps, the evaluation every 500 steps. |
| 2026-11-23T08:37:19Z | Experiment 1 is finished: 5 seeds, 0 failed, 600 rows. |

Nothing failed and no step was skipped. The deviations are the following:

1. **The sources were trained again.** The step-5,000 checkpoints of the
   existing runs do not exist any more (they were temporary files of the work
   directory that every script deleted after its trajectory). The sources were
   trained again with the same code path, and the check above shows that they
   are the same weights.
2. **The sources are kept in the memory of the worker and are not written to
   the disk.** The existing scripts wrote one temporary checkpoint per worker
   and deleted it after every transfer; ten checkpoints of pythia-160m would
   need 6.5 GB of the 12 GB that were free, so the weights of the step 5,000
   are held on the CPU of the worker and restored before every transfer.
3. **The sources are trained to the step 5,000 and not to the horizon of 8,000
   of queue 19.** The learning rate is constant after the warmup and the
   training is seeded, so the weights of the step 5,000 are the same; the
   check of the parameter norm against `p19_lr.csv` confirms it.
4. **The stage 2 is measured every 500 steps and not only at its end.** The
   protocol, the splits, the hyper-parameters and the seed are the ones of the
   existing runs; only the frequency of the measurement is new.
5. **The two experiments are two passes, so the sources are trained twice.**
   The order of the task asks for experiment 2 first and for experiment 1 in
   the background, and the sources cannot be written to the disk between the
   two passes. The two passes agree on every source to the last digit.
6. **The target a*b+a does not exist in the Pythia track.** The targets of the
   existing runs are a+2b, a-b and a*b (`p17_common.make_examples`), and the
   polynomial target a*b+a belongs to the toy track of queue 16 and queue 17.
   Nothing was therefore skipped: the two targets that were asked for, the
   a+3b and the a-2b, are the new ones.
7. **The row of the target a+2b of the table of experiment 2 is read from the
   curve of experiment 1 at the step 3,000.** The existing a+2b rows of
   `p19_lr.csv` are the stage 2 of the budget of 600 steps, which is a
   different budget, so they are not the numbers of this table. Every number
   of the table is read at the budget of 3,000 steps.

## The files of the queue

| file | what it is |
|---|---|
| `nseed_backbone/p22_targets.py` | the new target rules and the stage-2 curve |
| `nseed_backbone/q22_run.py` | the driver of the two experiments |
| `nseed_backbone/q22_report.py` | the tables of this report |
| `nseed_backbone/q22_pack.py` | the archive of the results |
| `run_q22_e2.sh`, `run_q22_e1.sh` | the two runners |
| `runs_queue22/q22_curves.csv` | the 600 rows of the curves of experiment 1 |
| `runs_queue22/q22_targets.csv` | the 180 rows of the curves of experiment 2 |
| `runs_queue22/q22_source_check.csv` | the 20 checks of the sources |
| `runs_queue22/q22_tables.md` | the tables with every seed |
| `runs_queue22/exp1.log`, `runs_queue22/exp2.log` | the logs of the two steps |
| `results_queue22.zip` | the archive of the run directory and of this report |
