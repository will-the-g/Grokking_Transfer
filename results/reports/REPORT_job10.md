# REPORT_job10

## K1 scratch Pre-LN (mean over the seeds)
| n_seeds | dev_mean_2000 | dev_mean_10000 |
|---|---|---|
| 5 | 0.0073 | 0.0639 |

## K2 p=59 Pre-LN per-stage transfer
| stage | n_seeds | mean_transfer_dev |
|---|---|---|
| memorized | 5 | 0.2744 |
| transition | 5 | 0.7784 |
| grokked | 5 | 0.8698 |
| norm_control | 5 | 0.3651 |

## K3 Jia probe (random split)
| seed | grokked | grok_step | source_steps | status |
|---|---|---|---|---|
| 5 | no |  | 40000 | nongrok |
| 6 | no |  | 40000 | nongrok |
| 7 | no |  | 40000 | nongrok |