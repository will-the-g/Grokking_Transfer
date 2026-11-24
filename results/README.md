# The results of the release

Every file is the result of one item of the paper. The checkpoints are
not here: the numbers of the tables and of the figures come from the
CSV and JSON files below, and the weights can be regenerated from the
seeds of the scripts in `experiments/`.

## `fig1/sweep_summary_p59.csv`

* **read by**: Figure 1 (the left panel), the sweep of the p=59 checkpoints
* **the columns**: seed, source_step, source_train_acc, source_dev_mean, source_global_norm, final_dev_mean, final_dev_worst, steps_to_30pct
* **present**: yes

## `fig1/sweep_summary_p97.csv`

* **read by**: Figure 1 (the middle panel), the sweep of the p=97 checkpoints
* **the columns**: the same columns as the p=59 sweep
* **present**: yes

## `fig1/sweep_summary_two_layers.csv`

* **read by**: Figure 1 (the right panel), the sweep of the two-layer checkpoints
* **the columns**: the same columns as the p=59 sweep
* **present**: yes

## `fig2/specificity/*/summary.json`

* **read by**: Figure 2 (the copy and the selector sources)
* **the columns**: the key `transfers` with the final_dev_mean
* **present**: NO

## `fig2/specificity_aged/*/summary.json`

* **read by**: Figure 2 (the aged copy and selector sources)
* **the columns**: the same key
* **present**: NO

## `fig2/specificity_aged_init100/*/summary.json`

* **read by**: Figure 2 (the aged sources, 100 steps)
* **the columns**: the same key
* **present**: NO

## `fig2/specificity_aged_init200/*/summary.json`

* **read by**: Figure 2 (the aged sources, 200 steps)
* **the columns**: the same key
* **present**: NO

## `fig3/pythia_ckpt_sweep.csv`

* **read by**: Figure 3 (the LoRA checkpoint sweep)
* **the columns**: model, revision, seed, source, train_acc, heldout_ab_acc, stage2_acc, gain_over_best_direct
* **present**: yes

## `fig4/p18_direct.csv`

* **read by**: Figure 4 (the direct baselines)
* **the columns**: model, p, seed, target, budget, acc
* **present**: yes

## `fig4/p18_traj.csv`

* **read by**: Figure 4 (the Pythia trajectories)
* **the columns**: model, p, variant, seed, source, step, train_acc, heldout_acc, train_loss, param_norm, stage2_target, stage2_budget, stage2_acc, gain_over_best_direct
* **present**: yes

## `fig5/m17_fourier_seed.csv`

* **read by**: Figure 5 (the spectrum of the seeds)
* **the columns**: the columns of the queue 19 files
* **present**: yes

## `fig5/m18_fourier.csv`

* **read by**: Figure 5 (the spectrum of the sources)
* **the columns**: the columns of the queue 19 files
* **present**: yes

## `fig5/m19_edits.csv`

* **read by**: Figure 5 (the frequency edits of queue 19)
* **the columns**: model, p, seed, source, step, edit, tolerance, edit_energy_in, edit_energy_out, heldout_acc, stage2_lr, stage2_acc
* **present**: yes

## `fig5/m20_edits.csv`

* **read by**: Figure 5 (the dose-matched edits of queue 20)
* **the columns**: model, p, seed, source, step, edit, target_energy_in, target_energy_out, edit_energy_in, edit_energy_out, heldout_acc, stage2_lr, stage2_acc
* **present**: yes

## `fig5/m21_energy.csv`

* **read by**: Figure 5 (the non-key deletions of queue 21)
* **the columns**: the columns of m20_edits
* **present**: yes

## `fig6/p19_direct_lr.csv`

* **read by**: Figure 6 (the direct baselines of the sweep)
* **the columns**: model, p, seed, stage2_lr, budget, acc
* **present**: yes

## `fig6/p19_lr.csv`

* **read by**: Figure 6 (the stage-2 learning rates)
* **the columns**: model, p, seed, source, step, heldout_acc, param_norm, stage2_lr, stage2_acc
* **present**: yes

## `jia/j0_probe.csv`

* **read by**: the first attempt at the regime of Jia (2026) at p=113
* **the columns**: seed, grokked, grok_step, source_steps, status, torch_version, gpu_name
* **present**: yes

## `jia/jia_cold_g1.csv`

* **read by**: the third attempt at the regime of Jia (2026), the cold start
* **the columns**: seed, cell, scaling, source_dev_mean, final_dev_mean, torch_version, gpu_name
* **present**: yes

## `jia/jia_cold_summary.csv`

* **read by**: the third attempt at the regime of Jia (2026), the cold start
* **the columns**: seed, cell, scaling, n_steps, final_dev_mean_2000, final_dev_mean_10000, steps_to_30pct, steps_to_90pct
* **present**: yes

## `jia/k3_probe.csv`

* **read by**: the second attempt at the regime of Jia (2026) at p=113
* **the columns**: the same columns
* **present**: yes

## `section3/runs_nseed_fine/seed_*/metrics.csv`

* **read by**: Section 3 (the timing of the transfer)
* **the columns**: the metrics of every evaluation of the source and of the transfer
* **present**: NO

## `section3/runs_nseed_fine/seed_*/summary.json`

* **read by**: Section 3 (the timing of the transfer)
* **the columns**: the summary of the seed
* **present**: NO

## `section3/t19_sub.csv`

* **read by**: Section 3 (the timing of the subtraction source)
* **the columns**: seed, source_kind, source_n, target, target_n, step, dev_acc, source_train_acc, source_dev_acc, source_norm
* **present**: yes

## `section5/embed_ablation.csv`

* **read by**: Section 5 (the ablation of the embedding)
* **the columns**: seed, variant, n_key, n_control, source_dev_mean, final_dev_mean, final_dev_worst, final_train_acc, steps_to_30pct
* **present**: yes

## `section5/queue3_summary.csv`

* **read by**: Section 5 (the sweep of the key-frequency removal)
* **the columns**: step, variant, n_seeds, source_dev_mean, final_dev_mean
* **present**: yes

## `section5/queue4_summary.csv`

* **read by**: Section 5 (the reruns of the stages)
* **the columns**: the same columns
* **present**: yes

## `summary_runs_nseed.csv`

* **read by**: Table 1 (the p=59 row) and Figure 2
* **the columns**: run_kind, seed, mask_seed, stage, task, init, status, grok_step, final_dev_mean, final_dev_worst, final_train_acc, best_dev_mean, steps_to_30pct, stage_clean_acc, global_norm, top5_share, freq_participation, nonconst_share, peak_k, n_evals
* **present**: yes

## `table1/k2_preln.csv`

* **read by**: Table 1 (the Pre-LN row), the runs of queue 10
* **the columns**: the same columns as p1_preln.csv
* **present**: yes

## `table1/p1_preln.csv`

* **read by**: Table 1 (the Pre-LN row), the runs of queue 9
* **the columns**: seed, stage, source_step, source_dev_mean, transfer_dev_mean, torch_version, gpu_name
* **present**: yes

## `table1/runs_nseed_L2/seed_*/summary.json`

* **read by**: Table 1 (the two-layer row), the key `transfers`
* **the columns**: the same keys as the p=97 summaries
* **present**: NO

## `table1/runs_nseed_p97/seed_*/summary.json`

* **read by**: Table 1 (the p=97 row), the key `transfers`
* **the columns**: affine|memorized, affine|norm_control, affine|transition, affine|grokked, each with final_dev_mean
* **present**: NO

## `table2/m21_lnbatch.csv`

* **read by**: Table 2 (the LayerNorm by batch-size 2x2)
* **the columns**: ln, batch, seed, stage, transfer_2000, transfer_20000, source_dev, source_grok_step
* **present**: yes

## `table3/b1_matrix.csv`

* **read by**: Table 3 (the task matrix of queue 13)
* **the columns**: source_task, target_task, seed, source_kind, step, dev_acc, source_final_dev, source_grok_step
* **present**: yes

## `table3/t18_parity.csv`

* **read by**: Table 3 (the sparse-parity row of queue 18)
* **the columns**: seed, source_kind, target, target_n, step, dev_acc, source_final_dev, source_grok_step
* **present**: yes

## `table3/t18_poly.csv`

* **read by**: Table 3 (the polynomial row of queue 18)
* **the columns**: source_task, target_task, seed, source_kind, target_fraction, step, dev_acc, source_final_dev, source_grok_step
* **present**: yes

## `table3/t19_parity.csv`

* **read by**: Table 3 (the memorizing parity of queue 19)
* **the columns**: seed, source_kind, source_n, target, target_n, step, dev_acc, source_train_acc, source_dev_acc, source_norm
* **present**: yes

## `table3/t19_poly.csv`

* **read by**: Table 3 (the memorizing and the norm-matched controls of queue 19)
* **the columns**: seed, source_kind, source_coverage, target_task, target_fraction, step, dev_acc, source_train_acc, source_dev_acc, source_norm
* **present**: yes

## `table3/t5_hard.csv`

* **read by**: Table 3 (the S5 and the polynomial targets of queue 17)
* **the columns**: source_task, target_task, seed, source_kind, target_fraction, step, dev_acc, source_final_dev, source_grok_step
* **present**: yes

## `table4/components/seed_*/metrics.csv`

* **read by**: Table 4 (the component hybrids)
* **the columns**: the metrics of every hybrid
* **present**: NO

## `table4/components/seed_*/summary.json`

* **read by**: Table 4 (the component hybrids)
* **the columns**: the key `jobs` with the final_dev_mean of the eleven hybrids
* **present**: NO

## `queue22/q22_curves.csv`

* **read by**: the long-budget test of queue 22 (the 20,000-step transfer of the
  grokked, the memorizing and the direct sources to a+2b)
* **the columns**: seed, condition, target, step, heldout_acc
* **present**: yes

## `queue22/q22_targets.csv`

* **read by**: the new targets of queue 22 (a+3b and a-2b)
* **the columns**: seed, condition, target, step, heldout_acc
* **present**: yes

## `queue22/q22_source_check.csv`

* **read by**: the check that the sources of queue 22 are the sources of the
  earlier runs
* **the columns**: seed, source, step, heldout_acc, param_norm
* **present**: yes

## `table4/factorial/seed_*/summary.json`

* **read by**: Table 4 (the component factorial), the key `jobs` with the
  `final_dev_mean` of every cell of the factorial
* **the columns**: the 23 cells (`fact_empty`, `fact_attn`, `fact_mlp`,
  `fact_unembed`, `fact_numeric_embed`, their combinations, `embed_rowperm`,
  `embed_random_matched` and the `mem_reset_*` cells) with their
  `final_dev_mean`
* **present**: yes, for the 9 seeds 5, 6, 7, 8, 9, 10, 11, 13 and 14

## `table4/factorial/seed_*/metrics.csv`

* **read by**: Table 4 (the component factorial), the evaluation of every cell
* **the columns**: seed, stage, task, init, step, train_acc, dev_acc_c0..c3,
  dev_mean, dev_worst, stage_clean_acc, global_norm, sec_per_step, elapsed_s
* **present**: yes, for the 9 seeds

## The reports

`reports/` holds the report of every queue that the paper uses, with
the absolute paths of the instance replaced by `<repo>`, `<instance>`
and `<scratch>`.
