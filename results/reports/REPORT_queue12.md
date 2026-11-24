# REPORT_queue12

| file | rows |
|---|---|
| pythia_traj.csv | 198 |
| pythia_ckpt_sweep.csv | 540 |
| toy_learning_curves.csv | 3600 |

| failure |
|---|
| A1: CUDA OOM twice with 3 workers (11 GB per worker); fixed to one model per worker and rerun with PROCS=2 (0 failures in the final run) |