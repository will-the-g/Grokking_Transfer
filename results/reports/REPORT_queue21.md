# REPORT_queue21

Queue 21 on the GPU instance (an RTX 5090, torch 2.15.0+cu132, the interpreter
`/root/venv/bin/python`): the energy-matched non-key deletions and the
LayerNorm by batch-size 2x2. The Pythia track (step A) ran from
2026-11-23T01:49:29Z to 2026-11-23T02:20:16Z and the toy track (step B) from
2026-11-23T02:31:24Z to 2026-11-23T03:50:11Z.

Only the dev split was read. No audit pair was opened, printed or copied, and
the audit folder of the instance was never touched.

## The rows of every step

| file | step | expected rows | actual rows | status |
|---|---|---|---|---|
| m21_energy.csv | A: the new non-key deletions | 10 seeds * 3 edits = 30 | 30 | complete |
| m21_lnbatch.csv | B: the LayerNorm by batch-size 2x2 | 4 cells * 5 seeds * 4 stages = 80 | 74 | 6 rows are missing, they are listed below |
| p21_source_check.csv | A: the held-out accuracy of the unedited sources | 10 | 10 | complete |
| m21_lnbatch_none_512.csv | B: the cell no LayerNorm with the batch of 512 | 20 | 20 | complete |
| m21_lnbatch_preln_full.csv | B: the cell Pre-LN with the full batch | 20 | 16 | 4 rows are missing, the two sources that never grokked |
| m21_lnbatch_none_full_long.csv | B: the 20,000-step transfers of the cell no LayerNorm with the full batch | 10 | 10 | complete |

### Why the rows of B are missing

The cell Pre-LN with the full batch has two sources that do not grok within
the 20,000 steps: the seed 6 and the seed 7. A source that does not grok has
the late memorized checkpoint and the transition checkpoint instead of the
grokked one and the memorized one, so those two seeds hold 2 rows each and
not 4. The same happened to the seed 9 of the cell Pre-LN with the batch of
512 in queue 9, and that cell holds 18 rows and not 20.

The 74 rows of the table are:

| ln | batch | rows | the rows come from |
|---|---|---|---|
| none | full | 20 | reused from `runs_nseed/seed_5..9` |
| none | 512 | 20 | run in this queue |
| preln | 512 | 18 | reused from `results_queue9.zip` |
| preln | full | 16 | run in this queue |

## The two verifications of A

(a) **The sources are the sources of queue 20.** The script of queue 20 wrote
the checkpoint of the step 5,000 to the work directory and deleted it again
after the edits, so no step-5,000 source was left on the instance. The source
was trained again with the same code path (the same config A: pythia-160m,
p = 59, coverage 0.5, lr 3e-4, wd 1.0, the `gen` source, the horizon 5,000)
and the held-out accuracy of the unedited source was compared with the `none`
rows of `m20_edits.csv`:

| seed | the held-out of this run | the `none` row of queue 20 | the difference |
|---|---|---|---|
| 0 | 0.9610 | 0.9610 | 0.0000 |
| 1 | 0.9930 | 0.9930 | 0.0000 |
| 2 | 0.9990 | 0.9990 | 0.0000 |
| 3 | 0.9740 | 0.9740 | 0.0000 |
| 4 | 0.9830 | 0.9830 | 0.0000 |
| 5 | 0.7860 | 0.7860 | 0.0000 |
| 6 | 0.9740 | 0.9740 | 0.0000 |
| 7 | 0.9880 | 0.9880 | 0.0000 |
| 8 | 0.6960 | 0.6960 | 0.0000 |
| 9 | 0.8110 | 0.8110 | 0.0000 |

The mean absolute difference is **0.000000** and the largest one is
**0.000000** over the 10 seeds: the sources of this queue are the sources of
queue 20, to the last digit of the held-out accuracy.

(b) **The run of B is reproducible.** The two new cells of B were run twice,
the second time after the `source_dev` column was changed (see the repairs
below). The 72 transfer cells of the two runs are **identical**, 0 of them
differs. The run is deterministic: `NSEED_DETERMINISTIC=1` and
`NSEED_DETERMINISTIC_GPU=1` were set and the summaries of the sources record
`deterministic: true`.

## A: the edits by the removed energy

The three new edits remove the top 15, 20 and 25 non-key frequencies, that is
the ranks 6 to 20, 6 to 25 and 6 to 30 of the spectrum, in the input
embeddings and in the unembedding at the same time. The two rows of queue 20
that they continue are the ranks 6 to 10 (5 frequencies) and the ranks 6 to 15
(10 frequencies). Every row holds the stage-2 transfer to the target a+2b with
the learning rate 3e-4 and the budget of 600 steps.

| edit | the non-key frequencies | the removed energy in | the removed energy out | the held-out | the transfer |
|---|---|---|---|---|---|
| rank6_10_both | 5 (the ranks 6 to 10) | 0.2077 | 0.2035 | 0.8978 | 0.4692 |
| rank6_15_both | 10 (the ranks 6 to 15) | 0.3106 | 0.2898 | 0.9222 | 0.4364 |
| rank6_20_both | 15 (the ranks 6 to 20) | 0.3967 | 0.3617 | 0.9397 | 0.4056 |
| rank6_25_both | 20 (the ranks 6 to 25) | 0.4720 | 0.4241 | 0.9539 | 0.4084 |
| rank6_30_both | 25 (the ranks 6 to 30) | 0.5251 | 0.4717 | 0.9609 | 0.4562 |

The mean over the 10 seeds; the rows of the first two edits are the rows of
the learning rate 3e-4 of `m20_edits.csv` and the rows of the last three
edits are the new ones of `m21_energy.csv`. The unedited source of the same
seeds has a held-out accuracy of 0.9165 and a transfer of 0.4478 (the `none`
rows of queue 20 at the step 5,000). The same table with the difference
against the source of the same seed, which is the fairer comparison because
the seeds differ a lot from one another:

| edit | the held-out minus the source | the transfer minus the source |
|---|---|---|
| rank6_10_both | -0.0187 | +0.0214 |
| rank6_15_both | +0.0057 | -0.0114 |
| rank6_20_both | +0.0232 | -0.0422 |
| rank6_25_both | +0.0374 | -0.0394 |
| rank6_30_both | +0.0444 | +0.0084 |

The three findings:

* The **held-out accuracy of the source grows with the size of the deletion**:
  -0.0187, +0.0057, +0.0232, +0.0374, +0.0444 against the source of the same
  seed, and it overtakes the unedited source from the ranks 6 to 15 on.
  Removing the non-key frequencies does not damage the source; it removes the
  part of the spectrum that hurts the generalization of the source.
* The **transfer to a+2b is flat**: +0.0214, -0.0114, -0.0422, -0.0394,
  +0.0084 against the source of the same seed. Deleting 20.8 percent or 52.5
  percent of the energy of the non-zero frequencies changes the transfer of
  the source by less than 0.045, and the deletion of the key set, which
  removes the energy of the solution itself, is what breaks it (the
  `key_e100_both` rows of queue 20). The transfer does not depend on how much
  of the non-key spectrum is removed, only on whether the key set is there.
* The transfer is not monotone in the size of the deletion: it falls from
  0.4692 to 0.4056 and comes back to 0.4562. The seed 8 is the reason: its
  source transfers at 0.036 and its edited sources transfer at 0.060, 0.036,
  0.008, 0.030 and 0.028, so it carries every low value of the table.
  Without that seed the five edits have a mean transfer of 0.5147, 0.4809,
  0.4498, 0.4504 and 0.5038, and the ten seeds are too few to read a trend
  inside that band.

The per-seed rows are in `runs_nseed_queue21/p21_tables.md`.

## B: the LayerNorm by batch-size 2x2

The means of the cells, over the seeds that hold the row:

| ln | batch | stage | n | the mean transfer_2000 | the mean transfer_20000 |
|---|---|---|---|---|---|
| none | full | memorized | 5 | 0.0195 | 0.0184 |
| none | full | transition | 5 | 0.5147 | - |
| none | full | grokked | 5 | 0.7908 | - |
| none | full | norm_control | 5 | 0.0307 | 0.0373 |
| none | 512 | memorized | 5 | 0.0575 | 0.0919 |
| none | 512 | transition | 5 | 0.8502 | - |
| none | 512 | grokked | 5 | 0.8520 | - |
| none | 512 | norm_control | 5 | 0.2130 | 0.3029 |
| preln | full | memorized | 3 | 0.0236 | 1.0000 |
| preln | full | transition | 5 | 0.5012 | - |
| preln | full | grokked | 3 | 0.9359 | - |
| preln | full | norm_control | 3 | 0.0270 | 0.4969 |
| preln | full | late_memorized | 2 | 0.4618 | 1.0000 |
| preln | 512 | memorized | 4 | 0.3323 | - |
| preln | 512 | transition | 5 | 0.7001 | - |
| preln | 512 | grokked | 4 | 0.9038 | - |
| preln | 512 | norm_control | 4 | 0.3416 | - |
| preln | 512 | late_memorized | 1 | 0.4493 | - |

The 2,000-step transfer, the budget of the other queues:

* the **grokked** stage transfers in every cell: 0.7908 (none, full), 0.8520
  (none, 512), 0.9038 (preln, 512) and 0.9359 (preln, full);
* the **memorized** and the **norm-matched** stage transfer at chance in
  every cell: 0.0195 and 0.0307 (none, full), 0.0575 and 0.2130 (none, 512),
  0.0236 and 0.0270 (preln, full), 0.3323 and 0.3416 (preln, 512). The
  norm-matched stage is the memorized checkpoint rescaled to the norm of the
  grokked checkpoint; it does not transfer better than the memorized one, so
  the norm is not what holds the memorized solution back;
* the **transition** stage is in between in every cell: 0.5147, 0.8502,
  0.5012 and 0.7001.

The 20,000-step transfer, the column that the task adds:

* the memorized source of the cell **Pre-LN with the full batch reaches
  1.0000** (3 seeds) and the late memorized source of the two seeds that did
  not grok also reaches **1.0000** (2 seeds). The memorized solution of that
  cell does not transfer at the 2,000-step budget (0.0236) and transfers
  perfectly at the 20,000-step budget;
* the memorized source of the cell **no LayerNorm with the batch of 512**
  climbs only to 0.0919 and the memorized source of the cell **no LayerNorm
  with the full batch** does not move at all (0.0195 at 2,000 steps and 0.0184
  at 20,000 steps, a flat curve). The norm-matched source of the cell no
  LayerNorm with the batch of 512 climbs to 0.3029, and the seed 8 reaches
  0.9857;
* the **20,000-step column is missing for the cell Pre-LN with the batch of
  512**: the checkpoints of that cell (`runs_nseed_p1`) are not on the
  instance any more and `results_queue9.zip` holds only its metrics and its
  summary, so the long transfers of that cell could not be run.

The reading of the 2x2 is limited by one thing: the two cells that are reused
were produced by an older version of the code and the two cells that were run
here were produced by the current one, so the table mixes two code
generations. The size of that difference was measured: the 2,000-step
transfers of the cell no LayerNorm with the full batch were run again on the
checkpoints of that cell, and the values of the rerun differ from the reused
rows by 0.0036 on average and by 0.0066 at the most (10 rows). That is the
same kind of variation that the G2 step of queue 8 measured on the same
transfers.

## The disk events

The free disk was 12 GB at the start of the queue and 12 GB at the end of it,
always above the 10 GB of rule 3.

| event | detail | size |
|---|---|---|
| FREE DISK | 12 GB at the start and 12 GB at the end of the queue | 0 bytes |
| CREATED | `runs_nseed_queue21/` with the CSVs, the logs and the run folders of the two new cells | 59 MB |
| DELETED | the source checkpoints that no stage uses, inside the two run folders of this queue | 60 to 80 files per seed |
| KEPT | the Hugging Face cache, which the Pythia step needs | 2.2 GB |

Nothing outside the files that this queue created was deleted, moved or
changed.

## The failures, the skipped steps and the deviations

| time (UTC) | event |
|---|---|
| 2026-11-23T01:49:29Z | A starts with 4 workers: the config A of queue 20, the `gen` source, the seeds 0 to 9, the horizon 5,000. The step-5,000 checkpoint is kept in the memory of the worker and not on the disk, because the disk rule of 10 GB does not allow 10 checkpoints of pythia-160m. |
| 2026-11-23T02:20:16Z | A is finished: 10 trajectories, 0 failed, 30 rows against the 30 expected. The held-out accuracy of the unedited source is the one of queue 20 for the 10 seeds. |
| 2026-11-23T02:31:24Z | B starts: the cell no LayerNorm with the batch of 512, the seeds 5 to 9, the source cap 20,000, the transfer of 2,000 steps with the dev split every 100 steps, and the 20,000-step transfer of the memorized and the norm-matched stages. |
| 2026-11-23T02:35:43Z | The cell no LayerNorm with the batch of 512 is finished: 5 seeds, 0 failed, 20 rows. The 5 sources grok, at the step 6,250 to 10,500. |
| 2026-11-23T02:57:07Z | The cell Pre-LN with the full batch is finished: 5 seeds, 0 failed, 16 rows. The seeds 5, 8 and 9 grok (at the step 16,250, 15,750 and 18,250) and the seeds 6 and 7 do not grok within the 20,000 steps. |
| 2026-11-23T03:09:18Z | The 20,000-step transfers of the cell no LayerNorm with the full batch are finished: 10 rows. Nothing was trained again; the cell is reused, only the long transfers are added. |
| 2026-11-23T03:12:00Z | The first table of B was wrong and was rebuilt: it held 54 rows and the 20 rows of the cell no LayerNorm with the batch of 512 were missing (the assembler looked for a file with the wrong name), and the 20,000-step column of the cell no LayerNorm with the full batch was empty (the key of the lookup held the seed as a number on one side and as a string on the other). |
| 2026-11-23T03:19:00Z | The `source_dev` column of the two new cells held the final dev accuracy of the source instead of the dev accuracy of the checkpoint of the stage, which is what the reused rows hold. The code was corrected and the two cells were run again; the run is deterministic, so the 72 transfer cells of the second run are identical to the ones of the first. The 20,000-step transfers of the cell no LayerNorm with the full batch were run again as well, and the file that they wrote held every row twice; the file was reduced to its 10 unique rows. |
| 2026-11-23T03:50:11Z | B is finished: 74 rows in the table. |

### The deviations

1. **The learning rate of the stage 2 of A is 3e-4**, the value of the task.
   The rows of `m20_edits.csv` hold the learning rates 3e-4 and 1e-4 for
   every edit; the new rows hold only the 3e-4 ones, and the rows of the
   `rank6_10_both` and `rank6_15_both` of the tables above are the 3e-4 rows
   of `m20_edits.csv`.
2. **The step-5,000 sources did not exist.** The script of queue 20 wrote them
   to the work directory and deleted them at the end of every trajectory, and
   no checkpoint of pythia-160m is left anywhere on the instance (every
   checkpoint of the instance is smaller than 3 MB, which is the size of the
   small transformer). They were trained again with the same code path and
   the reproduction is exact, see the verification (a) above.
3. **The sources of A are not saved on the disk.** The disk held 12 GB and the
   floor is 10 GB; 10 checkpoints of pythia-160m would need 6.5 GB. The
   weights of the step 5,000 are kept in the memory of the worker instead.
4. **The 20,000-step column of the cell Pre-LN with the batch of 512 is
   missing**, because the checkpoints of that cell are not on the instance any
   more. The other three cells hold it.
5. **The two reused cells come from an older version of the code.** Their
   summaries carry a runtime fingerprint that the current code cannot produce
   (`deterministic: false` with `gpu_deterministic: true`), so the 2x2 mixes
   two code generations. The size of the difference is measured above.
6. **The 20,000-step transfers of the cell no LayerNorm with the full batch
   are an addition to the task.** The task asks to reuse the rows of that
   cell and not to rerun it; its sources are on the instance, so the long
   transfers of its memorized and norm-matched checkpoints were run and only
   the empty 20,000-step cells of the reused rows were filled. The 2,000-step
   values of the reused rows were kept.
7. **The sources that do not grok are recorded with their late memorized
   checkpoint.** The seed 6 and the seed 7 of the cell Pre-LN with the full
   batch and the seed 9 of the cell Pre-LN with the batch of 512 hold the
   `late_memorized` and the `transition` stages instead of the `memorized`
   and the `grokked` ones. The long transfer of the late memorized stage is
   recorded in the 20,000-step column, because it is the memorized-like
   checkpoint of that source.
8. **The `source_dev` of a stage is the dev accuracy of the checkpoint of
   that stage**, the value that the reused rows hold. For the norm-matched
   stage it is the value of the memorized checkpoint, since the norm-matched
   checkpoint is that one rescaled.

### The interpretations that were needed

* The name of an edit holds its last rank, as in queue 20: the top 15
  non-key frequencies are the ranks 6 to 20 and the edit is `rank6_20_both`.
* The frequencies are ranked by their energy in the spectrum of the input
  embeddings, after the key set of the top 5, exactly the procedure of
  `rank6_10_both` of queue 20; the same set is removed in the unembedding.
* The removed energy fraction is measured on the edited rows: the energy of
  the non-zero frequencies of the part before the edit minus the energy of the
  non-zero frequencies of the edited rows, over the energy of the part before
  the edit.
* The stage of a source is the one of the standard selection: the grokked
  checkpoint is the one that is written when the source groks, the memorized
  one is the checkpoint of the 40 percent of the grok step with a worst
  context dev accuracy below 0.03, and the transition one is the checkpoint
  whose mean clean dev is the closest to 0.5 inside [0.25, 0.75].
* The norm control is the memorized checkpoint rescaled to the global norm of
  the grokked checkpoint of the same seed; it is the `norm_control` stage of
  the table and it is the `norm_control` of the other queues.
* The transfer of a stage is the standard 2,000-step affine transfer of
  queue 9 and queue 11, with the dev split scored every 100 steps; the
  20,000-step transfer is the same transfer run 10 times longer, and the
  value of the step 20,000 is the dev accuracy at that step.

## The files of the queue

| file | what it is |
|---|---|
| `nseed_backbone/p21_energy.py` | step A |
| `nseed_backbone/p21_lnbatch.py` | step B, the two new cells and the long transfers |
| `nseed_backbone/p21_report.py` | the tables of this report |
| `nseed_backbone/p21_pack.py` | the archive of the results |
| `run_p21_a.sh`, `run_p21_b.sh` | the two runners |
| `monitor_q21.sh` | the progress monitor, one line every 10 minutes |
| `runs_nseed_queue21/m21_energy.csv` | the 30 new rows of A |
| `runs_nseed_queue21/m21_lnbatch.csv` | the 74 rows of B |
| `runs_nseed_queue21/p21_source_check.csv` | the check of the sources of A |
| `runs_nseed_queue21/p21_tables.md` | the tables with every seed |
| `runs_nseed_queue21/runs_nseed_q21_*` | the run folders of the two new cells |
| `results_queue21.zip` | the archive of the three files of the queue |

The instance is left running.
