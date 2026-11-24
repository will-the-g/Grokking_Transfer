# REPORT_queue13

| file | rows |
|---|---|
| b1_matrix.csv | 3060 |
| b2_hparams.csv | 2160 |
| b3_dataeff.csv | 2880 |

| failure or deviation |
|---|
| B1: the sub sources (seeds 5-9) never grokked within the 50,000-step cap, so their stage transfers are missing and only the random-init transfers ran; B2: with coverage=0.3 the sources of seeds 6, 7 and 8 never grokked within the 50,000-step cap, so their stage transfers are missing; first B1 attempt: the pool returns results unordered and they were paired with the job list, so the rows were mis-attributed (the B1 table was deleted and the step re-run); first B3 attempt: CUDA tensors could not be sent from the worker processes (the row subset is now built inside the worker and the step was re-run); first launch: the cap deadline was not set in the script, so all three steps were skipped (the script was fixed and re-launched); the queued B3 invocation skipped seven of the nine Pre-LN seeds (they were run explicitly and all nine are in the table) |