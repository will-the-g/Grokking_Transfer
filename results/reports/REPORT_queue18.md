# REPORT_queue18

Queue 18 on the GPU instance (an RTX 5090, 32 GB of disk): the scale-up of the
trajectories, the mechanism done again, the late decline and the toy track.
The Pythia track ran from 2026-11-20T19:29:38Z to 2026-11-21T04:55:18Z and the
toy track from 2026-11-21T07:43:42Z to 2026-11-21T11:01:56Z.

## The rows of every step

| file | step | expected rows | actual rows | status |
|---|---|---|---|---|
| p18_traj.csv | P1: the trajectories of the configs C and D | 1296 | 1296 | complete |
| p18_direct.csv | P1: the direct baselines | 60 | 60 | complete |
| m18_fourier.csv | P2: the spectra and the frequency removal | 720 | 720 | complete |
| p18_collapse.csv | P3: the late decline | 300 | 300 | complete |
| t18_parity.csv | T1: the sparse parity with a memorized stage | 360 | 207 | incomplete: 153 rows are missing |
| t18_poly.csv | T2: the polynomial with a memorized stage | 225 | 135 | incomplete: 90 rows are missing |
| t18_timing.csv | T3: the stage timing | 120 | 96 | incomplete: 24 rows are missing |

### How the expected rows are counted

* p18_traj.csv: the schedule S of the horizon 5,500 holds 36 steps (0, 500,
  1,000, 1,500, 2,000 and then every 100 steps from 2,500). A `gen` or a `mem`
  trajectory holds 36 transfers of add2 at the budget 600, 4 transfers of add2
  at the budget 3,000 (the steps 3,000, 4,000, 5,000 and the horizon) and 14
  transfers of sub and mult at the budget 600 (the 7 steps every 500 from 2,500),
  so 54 rows. A `shuffled` trajectory holds the 36 base transfers. Per
  configuration: 5 seeds * 54 (`gen`) + 5 seeds * 54 (`mem`) + 3 seeds * 36
  (`shuffled`) = 648; the two configurations: 1,296.
* p18_direct.csv: 2 configurations, 5 seeds and 6 baselines (add2 at the
  budgets 600, 1,600, 3,000 and 6,000, sub and mult at the budget 600): 60.
* m18_fourier.csv: 13 spectra (the steps 0, 500, ..., 6,000) with two rows
  each (the input embeddings and the unembedding) and 11 edits at each of the
  steps 4,000 and 5,000: 26 + 22 = 48 rows per trajectory; 10 `gen` and 5
  `mem` trajectories: 720.
* p18_collapse.csv: the steps every 250 from 4,000 to the horizon, 21 for the
  base_long (horizon 9,000) and 9 for the wd0late (horizon 6,000), with two
  stage-2 learning rates each: (21 + 9) * 2 * 5 seeds = 300.
* t18_parity.csv: 8 seeds, 5 source kinds, 3 targets and 3 target sizes: 360.
* t18_poly.csv: 5 seeds, 5 source kinds, 3 targets and 3 fractions: 225.
* t18_timing.csv: 2 source tasks, 5 seeds, 6 snapshots and 2 targets: 120.

### Why the toy rows are missing

* t18_parity.csv: none of the 8 seeds of the sparse parity has a memorized
  snapshot. The memorized stage is the first snapshot with a train accuracy of
  at least 0.99 and a dev accuracy of at most 0.6; every seed of the parity
  task reaches a train accuracy of 0.99 only after its dev accuracy has passed
  0.6, so the stage does not exist. The seeds 5, 6, 8, 9, 10, 11 and 12 hold
  the `transition` and the `grokked` stages and the seed 7 only the
  `transition` (its source never grokked, the dev stays at 0.5048). The
  `memorized` stage and the `norm_control`, which needs it, are therefore
  missing for every seed: the sources are the `transition`, the `grokked` and
  the `random_init`, so 7 seeds * 3 kinds * 9 + 1 seed * 2 kinds * 9 = 207
  rows. The eight notes are in the log of T1.
* t18_poly.csv: none of the 5 seeds of the polynomial has a memorized snapshot
  either (the same rule, a train accuracy of at least 0.99 and a dev accuracy
  of at most 0.1), so the sources are the `transition`, the `grokked` and the
  `random_init`: 5 seeds * 3 kinds * 9 = 135 rows. The five notes are in the
  log of T2.
* t18_timing.csv: the source of the `add` seed 5 and the source of the `sub`
  seed 9 never reached a dev accuracy of 0.9, so their grok step does not
  exist and the six snapshots around it cannot be taken. The 8 units that have
  a grok step hold 6 snapshots * 2 targets = 12 rows each: 96 rows. The two
  notes are in the log of T3.

## The disk events

| event | detail | size |
|---|---|---|
| DELETED | the pip cache /root/.cache/pip (rule 3a) | 47 MB |
| KEPT | the Hugging Face cache <instance> while the steps ran: it holds the weights of pythia-160m, which the steps need as an input, and of pythia-410m, which was fetched before the config C started | 622 MB and 1.6 GB |
| DELETED | the result zips results_queue15.zip, results_queue16.zip and results_queue17.zip (rule 3b); the three zips and the CSVs they hold are on the laptop in results_queue15, results_queue16 and results_queue17 | 100,533 bytes |
| SKIPPED | the /tmp cleanup of rule 3b: /tmp holds audit files, so the directory was not deleted; it is 132 KB | 132 KB |
| MOVED | runs_nseed_queue16 to the laptop under archive_instance/runs_nseed_queue16, every file verified by size and sha256 (724 of 724), then deleted on the instance | 671,418,847 bytes |
| MOVED | runs_nseed_jia to the laptop under archive_instance/runs_nseed_jia, every file verified by size and sha256 (1,504 of 1,504), then deleted on the instance | 1,383,611,485 bytes |
| MOVED | runs_nseed_queue17 to the laptop under archive_instance/runs_nseed_queue17, every file verified by size and sha256 (54 of 54), then deleted on the instance | 1,978,278,383 bytes |
| MOVED | runs_nseed_queue13 to the laptop under archive_instance/runs_nseed_queue13, every file verified by size and sha256 (3,461 of 3,461), then deleted on the instance | 2,944,559,341 bytes |
| COPIED | the symbol map runs_nseed_queue15/symbols_p97.json to the laptop under archive_instance/runs_nseed_queue15 and to the run directory of queue 18 | 3,595 bytes |
| DELETED | the temporary checkpoints of the work directory at the end of the run: the directory holds 0 files | 0 bytes |
| DELETED | the Hugging Face cache at the end of the run, after the last Pythia model was done | 2.2 GB |
| FREE DISK | 9.7 GB before the cleanup and 16 GB after it | 6.3 GB |

The order of rule 3 was kept: the pip cache, then the result zips, then the
directories that were moved to the laptop. The moves went past the 12 GB floor
of rule 3 because rule 5 allows two temporary checkpoints per worker and the
pythia-410m checkpoints are 1.6 GB each: the extra room keeps the free disk
above the 4 GB floor of rule 5 while the 410m configuration runs. No directory
that holds an audit file was moved or deleted: the only audit files of the
instance are three files in /tmp, and /tmp was left alone.

## The failures, the skipped steps and the deviations

| time (UTC) | event |
|---|---|
| 2026-11-20T19:29:38Z | P1 config D (pythia-160m, p = 97) starts with 4 workers. The 30 direct baselines of the configuration are the ones of queue 17: the protocol, the splits, the seed and the learning rate are the same, so they are copied instead of being measured again. |
| 2026-11-20T19:29:38Z | P1 config C (pythia-410m, p = 59) starts with 3 workers. The 30 direct baselines of the configuration are measured here, because queue 17 never reached them. |
| 2026-11-20T20:09:37Z | FAILED P1 config C: the three workers of the pythia-410m do not fit on the card. Every worker holds the model, its AdamW states and a copy of the pretrained weights, and the stage-2 transfer adds a second model with its own AdamW states, so the three workers needed about 32 GB on a card of 31.4 GB. All 13 units died with a CUDA out of memory at their first stage-2 transfer. The 30 direct baselines of the configuration were written before the failure and are kept. |
| 2026-11-20T20:09:37Z | DEVIATION P1 config C, the first attempt | the temporary checkpoint of a unit was deleted only after the stage-2 transfer, so the units that died with the out of memory left their 1.6 GB checkpoints on the disk: 15 GB of them, and the disk was full at 876 KB free. |
| 2026-11-20T20:09:51Z | FAILED P2 mechanism, the first attempt | the mechanism started with the disk full and every unit died with the error of the zip writer, "unexpected pos 29504 vs 29398". No trajectory of the mechanism reached the horizon of 6,000 steps, so its 200 rows were all partial and the file was removed and written again from the start. |
| 2026-11-20T20:36:01Z | RECOVERY | the 25 orphaned temporary checkpoints of the work directory were deleted (15 GB, the disk went back to 15 GB free) and the partial m18_fourier.csv was removed. The new code deletes the temporary checkpoint in a finally block, so a failed unit can no longer leave one behind, and it checks the free disk before every save and stops the step when the disk falls below 4 GB. |
| 2026-11-20T20:36:01Z | P1 config C starts again with 2 workers, which need about 24 GB of the card. |
| 2026-11-21T03:18:50Z | P1 config C is finished: 13 trajectories, 0 failed, 648 rows of p18_traj.csv. Every one of the 13 trajectories holds exactly the transfers of the plan, with nothing missing and nothing extra. The two trajectories of the seed 0 that the first attempt had already finished (the `gen` and the `mem`, 54 rows each) were kept: the second attempt skipped them because the step 5,500 was already in the file. |
| 2026-11-21T03:18:50Z | P1 is complete: p18_traj.csv holds 1,296 rows against the 1,296 expected and p18_direct.csv holds 60 rows against the 60 expected. |
| 2026-11-21T03:18:50Z | P2 mechanism starts with 4 workers. |
| 2026-11-21T04:55:18Z | P2 is finished: 15 trajectories, 0 failed. The file held 738 rows: 18 of them were duplicates of the rows of the `mem` seeds 2, 3 and 4 at the steps 500, 1,000 and 1,500, left behind by the workers of the first attempt, which kept writing into the file after it had been removed and recreated it. The training is seeded, so the 18 duplicated rows hold exactly the same values as the rows they repeat; the duplicates were dropped and the file holds the 720 rows expected, with 48 rows for each of the 15 trajectories. |
| 2026-11-21T04:55:18Z | The overlap of the key frequencies is written to the log of P2 and is listed below. |
| 2026-11-21T05:20:00Z | P3 is finished: 10 trajectories (the two variants and the seeds 0 to 4), 0 failed, 300 rows against the 300 expected. |
| 2026-11-21T05:20:00Z | T1, the sparse parity, starts with 4 workers. |
| 2026-11-21T08:35:28Z | FAILED T1, the first attempt | the transfer of the parity rebuilt the tokens of the 512 strings of every batch at every step, which took 55 % of the step; the step reached 40 of the 207 rows in 80 minutes and would have needed about 12 hours. The rows were dropped, the transfer was rewritten so that the tokens and the labels of the training strings are built once, and the two implementations were compared on the same seed: they give the same dev accuracies at the same steps, and the new one is 18 times faster. The step was started again with 8 workers. |
| 2026-11-21T08:35:28Z | DEVIATION T1 | the 12 workers of the two stopped attempts stayed alive after their parents were killed, because the children of a spawn pool do not carry the name of the script in their command line. They kept appending rows to t18_parity.csv and holding 12 GB of the card. They were killed and the 46 rows they had written twice were dropped; the file holds the 207 rows of the 8 units, each of them with its 3 targets and its 3 target sizes. |
| 2026-11-21T10:25:16Z | T1 is finished: 8 units, 0 failed, 207 rows against the 360 expected. |
| 2026-11-21T10:39:26Z | T2, the polynomial, is finished: 5 units, 0 failed, 135 rows against the 225 expected. |
| 2026-11-21T11:01:56Z | T3, the stage timing, is finished: 10 units, 0 failed, 96 rows against the 120 expected. |

### The interpretations that were needed

* The `shuffled` source of P1 holds the base transfer, add2 at the budget 600
  at every step of S, and nothing else: the description of the stage-2
  transfers names the `gen` and the `mem` sources for the transfers of the
  budget 3,000 and of sub and mult, and the `shuffled` source is a control of
  the memorization, which the base transfer measures.
* The spectrum of P2 removes the mean over the p symbol tokens before the
  transform and leaves the zero frequency out of the total, so `frac_top5` is
  the share of the five strongest non-zero frequencies. The removal of a
  frequency projects out the cosine and the sine component along the residue
  axis in every hidden dimension and leaves the mean alone.
* The transitions of the toy track are the snapshots whose dev accuracy is the
  closest to 0.5 inside [0.25, 0.75], the rule of the earlier sweeps.
* The `norm_control` of the toy track is the memorized checkpoint rescaled to
  the global norm of the grokked one, and it needs the memorized checkpoint, so
  it is missing wherever the memorized stage is missing.

## The overlap of the key frequencies of P2

The key set is the five strongest frequencies of the spectrum of the input
embeddings. The number below is the mean size of the intersection of the key
sets over the pairs of seeds at one step, and over the seeds between two steps,
out of five.

| source | step 0 | 500 | 1,000 | 1,500 | 2,000 | 2,500 | 3,000 | 3,500 | 4,000 | 4,500 | 5,000 | 5,500 | 6,000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gen | 5.00 | 3.04 | 2.44 | 1.98 | 2.13 | 2.13 | 2.04 | 1.98 | 2.09 | 2.02 | 2.00 | 2.04 | 1.73 |
| mem | 5.00 | 3.50 | 4.20 | 4.10 | 3.00 | 2.90 | 2.10 | 2.00 | 1.70 | 2.10 | 1.90 | 1.50 | 1.80 |

Between the steps, the mean overlap of the key sets of one seed is 2.20 (the
steps 0 and 1,000) and 4.60 (the steps 5,000 and 6,000) for the `gen` source,
and 3.40 and 3.80 for the `mem` source; the full list is in the log of P2.
