# REPORT_queue19

Queue 19 on the GPU instance (an RTX 5090): the energy-matched controls of the
frequency edits, the stage-2 learning-rate sweep and the toy track. The Pythia
track ran from 2026-11-21T21:05:26Z to 2026-11-22T00:42:11Z and the toy track
from 2026-11-22T00:42:11Z to 2026-11-22T01:29:57Z.

## The rows of every step

| file | step | expected rows | actual rows | status |
|---|---|---|---|---|
| m19_edits.csv | P1: the energy-matched frequency edits | 840 | 300 | incomplete: 540 rows are missing |
| p19_lr.csv | P2: the stage-2 learning-rate sweep | 200 | 200 | complete |
| p19_direct_lr.csv | P2: the direct baselines of the sweep | 40 | 40 | complete |
| t19_parity.csv | T1: the small parity sources | 288 | 54 | incomplete: 234 rows are missing |
| t19_poly.csv | T2: the small polynomial sources | 180 | 180 | complete |
| t19_sub.csv | T3: the sub timing, evaluated with the sub labels | 70 | 56 | incomplete: 14 rows are missing |

### How the expected rows are counted

* m19_edits.csv: (10 seeds of the config A + 5 seeds of the config C) * 2
  steps (4,000 and 5,000) * 14 edits * 2 stage-2 learning rates = 840.
* p19_lr.csv: 2 sources (`gen` and `mem`) * 5 seeds * 5 steps (4,000, 5,000,
  6,000, 7,000 and 8,000) * 4 stage-2 learning rates = 200.
* p19_direct_lr.csv: 5 seeds * 4 stage-2 learning rates * 2 budgets (600 and
  3,000) = 40.
* t19_parity.csv: 8 seeds * 2 training sets (300 and 600) * 3 targets * 3
  target sizes = 144 per source kind; the `memorized` and the `norm_control`
  kinds: 288.
* t19_poly.csv: 5 seeds * 2 coverages (0.15 and 0.25) * 3 targets * 3
  fractions = 90 per source kind; the `memorized` and the `norm_control`
  kinds: 180.
* t19_sub.csv: 5 seeds * 6 snapshots * 2 targets = 60, plus the random init
  of the 5 seeds and the 2 targets: 70.

### Why the rows of P1 are missing

The three energy-matched sets of every trajectory and step could not be built:
the number of sets found is 0 for the 30 trajectories and steps. The key set
holds between 0.29 and 0.69 of the non-zero energy of the input embeddings and
between 0.35 and 0.74 of the energy of the unembedding, while the strongest set
of five frequencies that does not hold the key, the frequencies of the rank 6
to 10, holds only 0.06 to 0.27 of the energy in the input embeddings and 0.06
to 0.27 in the unembedding. The window of the tolerance of 35 % around the key
is therefore not reachable: in the 30 cases the unembedding condition cannot be
reached by any set of five non-key frequencies, and in the 25 cases where the
input condition is reachable the unembedding condition is not. The 5,000 draws
of the 20 % tolerance and the 5,000 of the 35 % tolerance found no set that
satisfies both conditions, so the `em_both`, `em_in` and `em_out` edits do not
exist and the 9 edits that were measured hold 300 rows against the 840
expected.

### Why the rows of T1 are missing

Only 3 of the 16 sources of the parity have a memorized snapshot: the seeds 8,
11 and 12 with the training set of 300 examples. The other 13 sources have no
memorized snapshot and their transfers were not run:

| source | the reason |
|---|---|
| the seeds 5, 6, 7 and 9 with 300 and with 600 examples, the seeds 8, 11 and 12 with 600 examples | no snapshot reaches a train accuracy of 0.99 within the 5,000 steps |
| the seed 10 with 600 examples | not memorized: the first snapshot with a train accuracy of 0.99 already has a dev accuracy of 0.9998 |

The three sources that were accepted hold 2 kinds * 3 targets * 3 target
sizes = 18 rows each: 54 rows against the 288 expected.

### Why the rows of T3 are missing

The sub source of the seed 9 never reached a dev accuracy of 0.9, so it has no
grok step and its 6 snapshots and its 2 random init rows do not exist: 14
rows are missing. The 4 sources that grokked hold 6 snapshots * 2 targets and
2 random init rows each: 56 rows against the 70 expected. The check of the
snapshot at the grok step was made before any transfer: the 4 sources that
passed hold a source dev of 0.904 to 0.916 and a source train accuracy of 1.0
at the grok step, so the step was not stopped.

## The disk events

The free disk was 13 GB at the start of the queue, above the 10 GB of rule 3,
so nothing had to be moved then. The Pythia steps needed the room for the
temporary checkpoints of the workers and for the weights of pythia-410m, and
the free disk fell to 7.9 GB during the config C of P1; the directories below
were then moved to the laptop under archive_instance/, one at a time and the
smallest first, and every file was checked by size and by sha256 on both sides
before the directory was deleted on the instance.

| event | detail | size |
|---|---|---|
| FREE DISK | 13 GB at the start of the queue, 7.9 GB at the lowest point of P1 and 14 GB at the end of the queue | 0 bytes |
| MOVED | runs_nseed_L2 to the laptop under archive_instance/runs_nseed_L2, 131 files verified by size and sha256, then deleted on the instance | 265,239,224 bytes |
| MOVED | runs_nseed_k2 to the laptop under archive_instance/runs_nseed_k2, 314 files verified, then deleted on the instance | 301,909,508 bytes |
| MOVED | runs_nseed_p1 to the laptop under archive_instance/runs_nseed_p1, 321 files verified, then deleted on the instance | 304,451,948 bytes |
| MOVED | runs_nseed_p97 to the laptop under archive_instance/runs_nseed_p97, 392 files verified, then deleted on the instance | 398,976,749 bytes |
| MOVED | runs_nseed_fine to the laptop under archive_instance/runs_nseed_fine, 935 files verified, then deleted on the instance | 797,838,553 bytes |
| MOVED | runs_nseed_queue14 to the laptop under archive_instance/runs_nseed_queue14, 1,133 files verified, then deleted on the instance | 993,616,367 bytes |
| MOVED | runs_nseed to the laptop under archive_instance/runs_nseed, 2,867 files verified, then deleted on the instance | 2,446,078,029 bytes |
| KEPT | the Hugging Face cache: the free disk was 14 GB at the end of the queue, above the 12 GB of rule 2 | 2.2 GB |
| DELETED | the temporary checkpoints of the work directory: the directory holds 0 files | 0 bytes |

The copy of runs_nseed_queue14 that the laptop held from an earlier queue held
449 of the 1,133 files; it was checked against the manifest of the instance and
the missing files were transferred, so the copy on the laptop is the whole
directory. No directory that holds an audit file was moved or deleted, and no
audit file was opened, printed or copied.

## The failures, the skipped steps and the deviations

| time (UTC) | event |
|---|---|
| 2026-11-21T21:05:26Z | P1 starts with the config A (pythia-160m, p = 59, the seeds 0 to 9, the `gen` source, the horizon 5,000) with 4 workers. |
| 2026-11-21T22:02:06Z | P1 config A is finished: 10 trajectories, 0 failed, 200 rows. The 9 edits of every trajectory and step are there: `none`, `key_both`, `key_in`, `key_out` and `rank6_10_both`; the three energy-matched sets, and the six rows of the input and the unembedding only that go with them, are missing. |
| 2026-11-21T22:02:06Z | P1 config C (pythia-410m, p = 59, the seeds 0 to 4) starts with 2 workers, the limit of the task. |
| 2026-11-21T23:12:22Z | P1 config C is finished: 5 trajectories, 0 failed, 100 rows. P1 holds 300 rows against the 840 expected; the reason is the one of the section above. |
| 2026-11-21T23:12:22Z | P2 starts: the config A with the `gen` and the `mem` source, the seeds 0 to 4, the horizon 8,000, and the direct baselines of the four learning rates. The `gen` trajectories were not taken from the base_long of queue 18: that step wrote its checkpoints to the work directory and deleted them, so it left no checkpoint and the sources were trained again. |
| 2026-11-22T00:42:11Z | P2 is finished: 10 trajectories, 0 failed, 200 rows of p19_lr.csv against the 200 expected and 40 rows of p19_direct_lr.csv against the 40 expected. |
| 2026-11-22T00:42:11Z | T1 starts: the parity with the training sets of 300 and 600 examples, the seeds 5 to 12, at most 5,000 steps, a snapshot every 50 steps. |
| 2026-11-22T01:01:42Z | T1 is finished: 8 units, 0 failed, 54 rows against the 288 expected; the 13 sources without a memorized snapshot are listed above. The norm control was built for the three accepted sources, and the norm of the grokked source of queue 18 of the same seed was found for every one of them. |
| 2026-11-22T01:01:42Z | T2 starts: the polynomial with the coverages 0.15 and 0.25, the seeds 5 to 9, at most 5,000 steps, a snapshot every 50 steps. |
| 2026-11-22T01:14:39Z | T2 is finished: 5 units, 0 failed, 180 rows against the 180 expected. Every source reached a train accuracy of 0.99 with a dev accuracy of at most 0.1, so every seed and coverage has both the `memorized` and the `norm_control` kind. |
| 2026-11-22T01:14:39Z | T3 starts: the sub sources with the source dev and the source train accuracy computed with the sub labels. |
| 2026-11-22T01:21:04Z | T3 note: the sub source of the seed 9 never reached a dev accuracy of 0.9. The check of the snapshot at the grok step passed for the 4 other seeds, so the transfers were run for them. |
| 2026-11-22T01:29:57Z | T3 is finished: 10 units, 0 failed, 56 rows against the 70 expected; the 14 missing rows are the ones of the seed 9. |

### The interpretations that were needed

* The energy of a frequency is the energy of the mean-removed symbol rows in
  that frequency, summed over the hidden dimensions, and the fraction of an
  edit is the share of the non-zero energy that the removed frequencies hold.
  The key set is the five strongest frequencies of the input spectrum and the
  rank 6 to 10 are the next five of the same spectrum.
* The `rank6_10_both` edit removes the frequencies of the rank 6 to 10 of the
  input spectrum from the input embeddings and from the unembedding, and its
  removed energy is measured in both parts.
* The `em_in` and the `em_out` edits remove the set of their `em_both` partner
  from one part only, so the energy that they remove in the other part is
  zero.
* The memorized snapshot of the toy sources is the first snapshot with a train
  accuracy of at least 0.99, and the source is only kept when the dev accuracy
  of that snapshot is at most 0.6 (the parity) or at most 0.1 (the polynomial).
* The norm control of the toy sources is the memorized snapshot rescaled to
  the global parameter norm of the grokked source of queue 18 of the same seed,
  read from the sources of queue 18 under runs_nseed_queue18/sources.
* The source dev and the source train accuracy of T3 are computed with the sub
  labels on the sub dev split, which is the correction of the rows of queue 18
  and of queue 17.
