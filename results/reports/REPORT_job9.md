# REPORT_job9

## P0 forced recompute (p=97 grokked-source transfers)
| seed | stored | new | difference | torch | gpu |
|---|---|---|---|---|---|
| 5 | 0.9916765755053507 | 0.9916765755053507 | 0.0 | 2.15.0+cu132 | NVIDIA GeForce RTX 5090 |
| 6 | 0.9725806451612904 | 0.9725806451612904 | 0.0 | 2.15.0+cu132 | NVIDIA GeForce RTX 5090 |
| 7 | 0.9454518430439952 | 0.9454518430439952 | 0.0 | 2.15.0+cu132 | NVIDIA GeForce RTX 5090 |

## J0 grok probe
| seed | grokked | grok_step | source_steps | status |
|---|---|---|---|---|
| 5 | no |  | 40000 | nongrok |
| 6 | no |  | 40000 | nongrok |
| 7 | no |  | 40000 | nongrok |

## P1 p=59 Pre-LN per-stage transfer
| stage | n_seeds | mean_transfer_dev |
|---|---|---|
| memorized | 4 | 0.3323 |
| transition | 5 | 0.7001 |
| grokked | 4 | 0.9038 |
| norm_control | 4 | 0.3416 |