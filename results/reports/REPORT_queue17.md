# REPORT_queue17

| file | step | expected rows | actual rows | status |
|---|---|---|---|---|
| p17_traj.csv | P1: the trajectories with the stage-2 transfers | 2720 | 1585 | incomplete: 1135 rows are missing |
| p17_direct.csv | P1: the direct baselines | 150 | 120 | incomplete: 30 rows are missing |
| m17_fourier.csv | P2: the spectra and the frequency removal | 105 | 67 | incomplete: 38 rows are missing |
| t5_hard.csv | T5: the harder targets | 360 | 236 | incomplete: 124 rows are missing |
| t6_parity.csv | T6: the parity with the stages | 720 | 2 | incomplete: 718 rows are missing |
| t7_timing.csv | T7: the stage timing | 120 | 100 | incomplete: 20 rows are missing |

| disk event | detail | size |
|---|---|---|
| DELETED | the pip cache /root/.cache/pip (rule 3a) | 2.88 GB |
| DELETED | the Hugging Face cache <instance> (rule 3a) | 0.006 GB |
| FREE DISK | 9.4 GB before the cleanup, 13 GB after the pip and the Hugging Face cache; the 12 GB floor was reached at rule 3a, so no zip and no directory was touched | 13 GB |
| CHECKED | the zips results_queue15.zip and results_queue16.zip are on the laptop at the sizes 23366 and 25340 bytes and were kept; the zips results_queue12.zip, results_queue13.zip and results_queue14.zip are already gone from the instance | 0 bytes |
| CHECKED | no audit file exists anywhere on the instance (the count is 0), so no directory was skipped for the audit rule | 0 bytes |

| failure, skipped step or deviation |
|---|
| [2026-11-20T00:45:55Z] queue17 starts on 2026-11-20T00:45:55Z |
| [2026-11-20T00:45:55Z] the free disk is 12G |
| [2026-11-20T00:45:55Z] START p1_160m |
| [2026-11-20T00:45:55Z] START t5 |
| [2026-11-20T00:45:56Z] FAILED p1_160m (exit 1) |
| [2026-11-20T00:45:56Z] START p1_410m |
| [2026-11-20T00:45:58Z] FAILED p1_410m (exit 1) |
| [2026-11-20T00:48:28Z] FAILED p1_160m and p1_410m, the first attempt | the worker of the direct baselines was defined inside the main function, so the spawn context could not pickle it; the worker is at the module level now and the Pythia track runs outside the launcher while the toy track stays in it |
| [2026-11-20T00:53:28Z] DEVIATION p1 | the stage-2 transfers and the direct baselines build the model once per worker and reuse it, because loading the weights from the Hugging Face cache for every one of the transfers cost more than the transfer itself; the pretrained weights are restored before every direct baseline |
| [2026-11-20T01:16:51Z] FAILED p1_160m, the second attempt | the trajectory module used the token accuracy without importing it, so all 50 trajectories of the 160m configurations stopped at the first checkpoint; the import is there now and the track was started again |
| [2026-11-20T01:22:23Z] FAILED p1_160m, the second attempt | the trajectory module used the token accuracy without importing it, so all 50 trajectories of the 160m configurations stopped at the first checkpoint; the import is there now and the track was started again |
| [2026-11-20T01:28:24Z] FAILED p1_160m, the third attempt | the best direct baseline was read from the csv as a string, so the gain could not be computed, and six workers with the cached models ran out of memory; the baseline is a float now and the track runs with four workers |
| [2026-11-20T01:44:38Z] FAILED p1_160m, the fourth attempt | the function that restores the pretrained weights read the model cache before it was filled, so 111 of the 120 direct baselines failed and the gains would have been wrong; the cache is filled first now, the partial trajectory file was moved to /tmp and the track was started again |
| [2026-11-20T01:50:21Z] FAILED p1_160m, the fourth attempt | the function that restores the pretrained weights read the model cache before it was filled, so 111 of the 120 direct baselines failed and the gains would have been wrong; the cache is filled first now, the partial trajectory file was moved to /tmp and the track was started again |
| [2026-11-20T02:41:46Z] DONE t5 |
| [2026-11-20T02:41:46Z] START t6 |
| [2026-11-20T02:49:32Z] FAILED t6 (exit 1) |
| [2026-11-20T02:49:32Z] START t7 |
| [2026-11-20T02:49:34Z] FAILED t7 (exit 1) |
| [2026-11-20T02:49:34Z] both tracks are finished |
| [2026-11-20T02:49:34Z] START report |
| [2026-11-20T02:49:35Z] FAILED report (exit 1) |
| [2026-11-20T02:49:35Z] queue17 finished on 2026-11-20T02:49:35Z |
| [2026-11-20T02:49:35Z] the instance stays up: the gated stop is issued from the laptop |
| [2026-11-20T02:59:53Z] FAILED t6 and t7, the first attempt | the parity source did not create its directory before the first snapshot and the stage timing read the step numbers with a leading underscore, so both steps stopped at the start; the directory is created now and the step numbers are parsed without the underscore |
| [2026-11-20T04:25:58Z] DEVIATION p2 | the mechanism rows were written with a whole file rewrite from the table of the worker that produced them, so the workers overwrote each other and the file kept only the last worker's rows; the rows are appended now and the track was started again |
| [2026-11-20T04:54:33Z] DEVIATION t7 | the stage timing runs beside the parity step instead of after it, because it needs no training and the parity step is slow while the Pythia track holds the GPU |
| [2026-11-20T06:43:04Z] DEVIATION t6 | the parity step was paused so that the Pythia track, which has the priority, gets the whole GPU; it is resumed after the Pythia track is finished |
| P1 INCOMPLETE | the trajectories hold 1585 rows against the expected 2320: 29 of the 60 combinations are complete, the variants covered are {'A': 1320, 'A_decay': 265} and the sources {'mem': 650, 'gen': 647, 'shuffled': 288}; the 410m configuration C never started because the session ended first |
| P1 direct baselines INCOMPLETE | 120 rows against the expected 150: the baselines of add2 at the budgets 600, 1600, 3000 and 6000 steps and of sub and mult at 600 steps were measured for the 160m configurations A, A_decay and D; the 30 baselines of the 410m configuration C are missing because the session ended before that configuration started |
| P2 INCOMPLETE | the mechanism holds 67 rows against the expected 105: the spectra and the frequency removal of the seeds 2, 3 and 4 are complete; the seed 0 was written before the whole-file rewrite of the mechanism rows was replaced by an append and the seed 1 was interrupted before its later steps |
| T5 INCOMPLETE | the harder targets hold 236 rows against the expected 360: the polynomial sources of queue16 have no memorized stage, so those rows are missing, and the S5 transition stage is missing for one seed |
| T6 INCOMPLETE | the parity holds 2 rows against the expected 720: the step was stopped by the priority of the Pythia track after the source of the first seed was trained |
| T7 INCOMPLETE | the stage timing holds 100 rows against the expected 120: 20 of the snapshots at the rounded steps do not exist, because the sources only keep a checkpoint every 250 steps |

| note |
|---|
| [2026-11-20T00:45:55Z] queue17 starts on 2026-11-20T00:45:55Z |
| [2026-11-20T00:45:55Z] the free disk is 12G |
| [2026-11-20T00:45:55Z] START p1_160m |
| [2026-11-20T00:45:55Z] START t5 |
| [2026-11-20T00:45:56Z] FAILED p1_160m (exit 1) |
| [2026-11-20T00:45:56Z] START p1_410m |
| [2026-11-20T00:45:58Z] FAILED p1_410m (exit 1) |
| [2026-11-20T00:48:28Z] FAILED p1_160m and p1_410m, the first attempt | the worker of the direct baselines was defined inside the main function, so the spawn context could not pickle it; the worker is at the module level now and the Pythia track runs outside the launcher while the toy track stays in it |
| [2026-11-20T00:53:28Z] DEVIATION p1 | the stage-2 transfers and the direct baselines build the model once per worker and reuse it, because loading the weights from the Hugging Face cache for every one of the transfers cost more than the transfer itself; the pretrained weights are restored before every direct baseline |
| [2026-11-20T01:16:51Z] FAILED p1_160m, the second attempt | the trajectory module used the token accuracy without importing it, so all 50 trajectories of the 160m configurations stopped at the first checkpoint; the import is there now and the track was started again |
| [2026-11-20T01:22:23Z] FAILED p1_160m, the second attempt | the trajectory module used the token accuracy without importing it, so all 50 trajectories of the 160m configurations stopped at the first checkpoint; the import is there now and the track was started again |
| [2026-11-20T01:28:24Z] FAILED p1_160m, the third attempt | the best direct baseline was read from the csv as a string, so the gain could not be computed, and six workers with the cached models ran out of memory; the baseline is a float now and the track runs with four workers |
| [2026-11-20T01:44:38Z] FAILED p1_160m, the fourth attempt | the function that restores the pretrained weights read the model cache before it was filled, so 111 of the 120 direct baselines failed and the gains would have been wrong; the cache is filled first now, the partial trajectory file was moved to /tmp and the track was started again |
| [2026-11-20T01:50:21Z] FAILED p1_160m, the fourth attempt | the function that restores the pretrained weights read the model cache before it was filled, so 111 of the 120 direct baselines failed and the gains would have been wrong; the cache is filled first now, the partial trajectory file was moved to /tmp and the track was started again |
| [2026-11-20T02:41:46Z] DONE t5 |
| [2026-11-20T02:41:46Z] START t6 |
| [2026-11-20T02:49:32Z] FAILED t6 (exit 1) |
| [2026-11-20T02:49:32Z] START t7 |
| [2026-11-20T02:49:34Z] FAILED t7 (exit 1) |
| [2026-11-20T02:49:34Z] both tracks are finished |
| [2026-11-20T02:49:34Z] START report |
| [2026-11-20T02:49:35Z] FAILED report (exit 1) |
| [2026-11-20T02:49:35Z] queue17 finished on 2026-11-20T02:49:35Z |
| [2026-11-20T02:49:35Z] the instance stays up: the gated stop is issued from the laptop |
| [2026-11-20T02:59:53Z] FAILED t6 and t7, the first attempt | the parity source did not create its directory before the first snapshot and the stage timing read the step numbers with a leading underscore, so both steps stopped at the start; the directory is created now and the step numbers are parsed without the underscore |
| [2026-11-20T04:25:58Z] DEVIATION p2 | the mechanism rows were written with a whole file rewrite from the table of the worker that produced them, so the workers overwrote each other and the file kept only the last worker's rows; the rows are appended now and the track was started again |
| [2026-11-20T04:54:33Z] DEVIATION t7 | the stage timing runs beside the parity step instead of after it, because it needs no training and the parity step is slow while the Pythia track holds the GPU |
| [2026-11-20T06:43:04Z] DEVIATION t6 | the parity step was paused so that the Pythia track, which has the priority, gets the whole GPU; it is resumed after the Pythia track is finished |
| P1 INCOMPLETE | the trajectories hold 1585 rows against the expected 2320: 29 of the 60 combinations are complete, the variants covered are {'A': 1320, 'A_decay': 265} and the sources {'mem': 650, 'gen': 647, 'shuffled': 288}; the 410m configuration C never started because the session ended first |
| P1 direct baselines INCOMPLETE | 120 rows against the expected 150: the baselines of add2 at the budgets 600, 1600, 3000 and 6000 steps and of sub and mult at 600 steps were measured for the 160m configurations A, A_decay and D; the 30 baselines of the 410m configuration C are missing because the session ended before that configuration started |
| P2 INCOMPLETE | the mechanism holds 67 rows against the expected 105: the spectra and the frequency removal of the seeds 2, 3 and 4 are complete; the seed 0 was written before the whole-file rewrite of the mechanism rows was replaced by an append and the seed 1 was interrupted before its later steps |
| T5 INCOMPLETE | the harder targets hold 236 rows against the expected 360: the polynomial sources of queue16 have no memorized stage, so those rows are missing, and the S5 transition stage is missing for one seed |
| T6 INCOMPLETE | the parity holds 2 rows against the expected 720: the step was stopped by the priority of the Pythia track after the source of the first seed was trained |
| T7 INCOMPLETE | the stage timing holds 100 rows against the expected 120: 20 of the snapshots at the rounded steps do not exist, because the sources only keep a checkpoint every 250 steps |