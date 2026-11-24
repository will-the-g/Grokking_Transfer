# REPORT_queue20

Queue 20 on the GPU instance (an RTX 5090): the dose-matched frequency edits
and the parity memorizing sources with a weight-decay ladder. The Pythia track
ran from 2026-11-22T03:37:12Z to 2026-11-22T05:05:15Z and the toy track from
2026-11-22T05:05:15Z to 2026-11-22T05:43:43Z.

## The rows of every step

| file | step | expected rows | actual rows | status |
|---|---|---|---|---|
| m20_edits.csv | P1: the dose-matched frequency edits | 400 | 400 | complete |
| t20_parity.csv | T1: the parity sources with the weight-decay ladder | at most 180 | 144 | incomplete: 36 rows are missing |

### How the expected rows are counted

* m20_edits.csv: 10 seeds * 2 steps (4,000 and 5,000) * 10 edits * 2 stage-2
  learning rates = 400.
* t20_parity.csv: 8 seeds * 2 training sets (300 and 600) * 3 targets * 2
  target sizes = 96 for the `memorized` kind; the `norm_control` kind needs
  the grokked source of queue 18, which the seed 7 does not have, so it holds
  7 seeds * 2 * 3 * 2 = 84 rows; the two kinds: at most 180.

### Why the rows of T1 are missing

Three of the 16 sources of the parity have no memorized snapshot at any of the
two weight decays, so their transfers were not run; they hold 12 rows each,
36 rows in total:

| source | the weight decay 0.5 | the weight decay 0.1 |
|---|---|---|
| the seed 5 with 600 examples | the first snapshot with a train accuracy of 0.99 has a dev accuracy of 0.9944 | the dev accuracy of that snapshot is 0.7392 |
| the seed 8 with 600 examples | no snapshot reaches a train accuracy of 0.99 | the dev accuracy of that snapshot is 0.6316 |
| the seed 12 with 600 examples | the dev accuracy of that snapshot is 0.9932 | the dev accuracy of that snapshot is 1.0000 |

The 13 sources that were accepted hold 2 kinds * 3 targets * 2 target sizes =
12 rows each, except the two sources of the seed 7, which hold only the
`memorized` kind because the grokked source of queue 18 of that seed does not
exist: 13 * 12 - 2 * 6 = 144 rows against the 180 expected.

The weight decay that was used per source: the weight decay of 0.5 was enough
for the seeds 6, 8, 10, 11 and 12 with 300 examples and for the seeds 6 and
11 with 600 examples; the weight decay of 0.1 was needed for the seed 5 with
300 examples and for the seeds 7, 9 and 10 with 600 examples.

## The two verifications of P1

(a) The realized removed energy against the target of the `key_match` edits:
240 values were checked, the ones of the input embeddings and of the
unembedding of the 3 `key_match` edits of the 10 seeds and the 2 steps and the
2 learning rates, and **0 of them are outside of 0.02**; the share inside
0.02 is 100 percent.

(b) The `none` rows against the `none` rows of the m19_edits.csv of queue 19
(the same model, seed, step and learning rate): the 40 rows of queue 20 were
compared with the 40 matching rows of queue 19. The mean absolute difference
of the held-out accuracy is **0.000000** and the largest difference is
0.000000; the mean absolute difference of the stage-2 accuracy is **0.000000**
and the largest difference is 0.000000; **0 of the 20** seed and step pairs
have a held-out accuracy that differs by more than 0.02. The mean held-out
accuracy of the `none` rows is 0.720050 in both files.

## The disk events

The free disk was 14 GB at the start of the queue, above the 10 GB of rule 3,
so no directory was moved and nothing was deleted; the disk stayed between 11
and 14 GB during the queue.

| event | detail | size |
|---|---|---|
| FREE DISK | 14 GB at the start of the queue and 12 GB at the end of it | 0 bytes |
| DELETED | the temporary checkpoints of the work directory: the directory holds 0 files | 0 bytes |
| KEPT | the Hugging Face cache, which the Pythia step needs and which rule 2 of this queue does not allow to be deleted | 2.2 GB |

No directory that holds an audit file was moved or deleted, and no audit file
was opened, printed or copied.

## The failures, the skipped steps and the deviations

| time (UTC) | event |
|---|---|
| 2026-11-22T02:59:00Z | The first action of the queue failed: the the remote shell to the instance was refused. The instance was stopped, which is the state the gated stop of queue 19 left it in. The the instance command line tool command line and the API key are only on the instance, and the key was not found on the laptop, so the instance could not be started from here; the queue was written and checked on the laptop while the instance was down. The instance was started again by the user, and the queue started from it. |
| 2026-11-22T03:37:12Z | P1 starts with 4 workers: the config A (pythia-160m, p = 59, coverage 0.5, lr 3e-4, wd 1.0), the `gen` source, the seeds 0 to 9, the horizon 5,000. |
| 2026-11-22T05:05:15Z | P1 is finished: 10 trajectories, 0 failed, 400 rows against the 400 expected. The factor of the key set was never capped, in every trajectory and step the ranks 6 to 10 hold less energy than the key set. |
| 2026-11-22T05:05:15Z | T1 starts: the parity with the training sets of 300 and 600 examples, the seeds 5 to 12, the weight decay of 0.5 and then the weight decay of 0.1, at most 5,000 steps, a snapshot every 50 steps. |
| 2026-11-22T05:43:43Z | T1 is finished: 8 units, 0 failed, 144 rows against the 180 expected; the 3 sources that could not be memorized and the missing norm control of the seed 7 are listed above. |

### The repair of the file of P1

The first version of the code reset the held-out accuracy and the realized
energy at the start of every learning rate of an edit, so the second row of
each edit held them empty. The values are measured once per edit, so the
second row of each of the 200 edits was given the values of the first row,
which is the same measurement; the 600 cells were filled and no cell is empty
now. The code was corrected so that the values are written in both rows, and
the corrected file is on the instance. No value was changed by the repair.

### The interpretations that were needed

* The factor f removes the share f * f of the energy of the frequency, so the
  component of the frequency is scaled by sqrt(1 - f * f) and the mean is left
  alone; a factor of 1 is the full removal of queue 18 and of queue 19. The
  ladder of the task uses the factors 0.5, 0.7071, 0.8660 and 1.0, which are
  the roots of the energy shares of 25, 50, 75 and 100 percent.
* The removed energy fraction of an edit is measured on the edited rows: the
  energy of the non-zero frequencies of the part before the edit minus the
  energy of the non-zero frequencies of the edited rows, over the energy of the
  part before the edit.
* The factor that matches the ranks 6 to 10 is the root of the ratio of the
  removed energy of the ranks 6 to 10 to the energy of the key set, in each
  part; it is capped at 1 and the capping is written to the log, and it was
  never needed.
* The `key_match_in` edit removes the key set from the input embeddings only
  and the `key_match_out` edit from the unembedding only; the target energy of
  the part that is not edited is 0.
* A source of the toy track is trained with the weight decay of 0.5 and, when
  it has no accepted memorized snapshot, again with the weight decay of 0.1;
  the weight decay that was used is recorded in the file.
* The target subsets, the dev strings and the training strings of the
  transfers are the ones of queue 18 for the same seed, and the target sizes
  are 100 and 300.
