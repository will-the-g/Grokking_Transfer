# The tables of queue 22

## Experiment 1: the long budget on a+2b

The step at which the held-out accuracy of the transfer reaches 0.9, and the accuracy at the steps 3,000, 6,000, 8,000 and 20,000. One row per seed and condition. The best point of the curve is given as well, because the curves of the transfers are not monotone.

| condition | seed | the first step over 0.9 | the best point | the accuracy at 3,000 | at 6,000 | at 8,000 | at 20,000 |
|---|---|---|---|---|---|---|---|
| grokked | 0 | 2000 | 0.9320 at 2000 | 0.7660 | 0.6720 | 0.5700 | 0.6820 |
| grokked | 1 | 2500 | 0.9840 at 2500 | 0.5460 | 0.5380 | 0.5020 | 0.4660 |
| grokked | 2 | 3000 | 0.9680 at 6000 | 0.9540 | 0.9680 | 0.9480 | 0.4000 |
| grokked | 3 | 3000 | 0.9540 at 3500 | 0.9280 | 0.6540 | 0.5560 | 0.6240 |
| grokked | 4 | 19500 | 0.9440 at 19500 | 0.3840 | 0.2760 | 0.2000 | 0.6660 |
| memorizing | 0 | never | 0.0120 at 4500 | 0.0000 | 0.0100 | 0.0080 | 0.0020 |
| memorizing | 1 | never | 0.0120 at 1000 | 0.0100 | 0.0060 | 0.0060 | 0.0060 |
| memorizing | 2 | never | 0.0140 at 15000 | 0.0080 | 0.0100 | 0.0060 | 0.0100 |
| memorizing | 3 | never | 0.0120 at 8500 | 0.0020 | 0.0040 | 0.0080 | 0.0020 |
| memorizing | 4 | never | 0.0140 at 2500 | 0.0140 | 0.0020 | 0.0060 | 0.0000 |
| direct | 0 | never | 0.0080 at 10500 | 0.0040 | 0.0000 | 0.0020 | 0.0020 |
| direct | 1 | never | 0.0080 at 500 | 0.0000 | 0.0020 | 0.0060 | 0.0020 |
| direct | 2 | never | 0.0080 at 14000 | 0.0000 | 0.0060 | 0.0000 | 0.0060 |
| direct | 3 | never | 0.0140 at 12500 | 0.0000 | 0.0020 | 0.0000 | 0.0060 |
| direct | 4 | never | 0.0120 at 17500 | 0.0040 | 0.0040 | 0.0040 | 0.0020 |

The means over the seeds, the same layout:

| condition | n | the first step over 0.9 (the mean of the seeds that reach it) | the seeds that never reach it | the best accuracy (the mean) | the accuracy at 3,000 | at 6,000 | at 8,000 | at 20,000 |
|---|---|---|---|---|---|---|---|---|
| grokked | 5 | 6000.0000 | 0 | 0.9564 | 0.7156 | 0.6216 | 0.5552 | 0.5676 |
| memorizing | 5 | never | 5 | 0.0128 | 0.0068 | 0.0064 | 0.0068 | 0.0040 |
| direct | 5 | never | 5 | 0.0100 | 0.0016 | 0.0028 | 0.0024 | 0.0036 |

## Experiment 2: the new target tasks

The final held-out accuracy of the transfer of the budget of 3,000 steps, as the mean over the seeds with the smallest and the largest value. The row of the target a+2b is the value of the curve of experiment 1 at the step 3,000, so that the three targets are read at the same budget.

| target | condition | n | the mean | the smallest | the largest |
|---|---|---|---|---|---|
| a+2b | grokked | 5 | 0.7156 | 0.3840 | 0.9540 |
| a+2b | memorizing | 5 | 0.0068 | 0.0000 | 0.0140 |
| a+2b | direct | 5 | 0.0016 | 0.0000 | 0.0040 |
| a+3b | grokked | 5 | 0.7808 | 0.6260 | 0.9380 |
| a+3b | memorizing | 5 | 0.0048 | 0.0040 | 0.0060 |
| a+3b | direct | 5 | 0.0024 | 0.0000 | 0.0040 |
| a-2b | grokked | 5 | 0.6628 | 0.3860 | 0.9440 |
| a-2b | memorizing | 5 | 0.0036 | 0.0020 | 0.0060 |
| a-2b | direct | 5 | 0.0016 | 0.0000 | 0.0040 |

The same table with every seed:

| target | condition | seed | the final held-out accuracy |
|---|---|---|---|
| a+2b | grokked | 0 | 0.7660 |
| a+2b | grokked | 1 | 0.5460 |
| a+2b | grokked | 2 | 0.9540 |
| a+2b | grokked | 3 | 0.9280 |
| a+2b | grokked | 4 | 0.3840 |
| a+2b | memorizing | 0 | 0.0000 |
| a+2b | memorizing | 1 | 0.0100 |
| a+2b | memorizing | 2 | 0.0080 |
| a+2b | memorizing | 3 | 0.0020 |
| a+2b | memorizing | 4 | 0.0140 |
| a+2b | direct | 0 | 0.0040 |
| a+2b | direct | 1 | 0.0000 |
| a+2b | direct | 2 | 0.0000 |
| a+2b | direct | 3 | 0.0000 |
| a+2b | direct | 4 | 0.0040 |
| a+3b | grokked | 0 | 0.7320 |
| a+3b | grokked | 1 | 0.8120 |
| a+3b | grokked | 2 | 0.7960 |
| a+3b | grokked | 3 | 0.6260 |
| a+3b | grokked | 4 | 0.9380 |
| a+3b | memorizing | 0 | 0.0060 |
| a+3b | memorizing | 1 | 0.0060 |
| a+3b | memorizing | 2 | 0.0040 |
| a+3b | memorizing | 3 | 0.0040 |
| a+3b | memorizing | 4 | 0.0040 |
| a+3b | direct | 0 | 0.0040 |
| a+3b | direct | 1 | 0.0020 |
| a+3b | direct | 2 | 0.0000 |
| a+3b | direct | 3 | 0.0020 |
| a+3b | direct | 4 | 0.0040 |
| a-2b | grokked | 0 | 0.3860 |
| a-2b | grokked | 1 | 0.7020 |
| a-2b | grokked | 2 | 0.9440 |
| a-2b | grokked | 3 | 0.8760 |
| a-2b | grokked | 4 | 0.4060 |
| a-2b | memorizing | 0 | 0.0040 |
| a-2b | memorizing | 1 | 0.0020 |
| a-2b | memorizing | 2 | 0.0060 |
| a-2b | memorizing | 3 | 0.0020 |
| a-2b | memorizing | 4 | 0.0040 |
| a-2b | direct | 0 | 0.0000 |
| a-2b | direct | 1 | 0.0020 |
| a-2b | direct | 2 | 0.0020 |
| a-2b | direct | 3 | 0.0000 |
| a-2b | direct | 4 | 0.0040 |

## The check of the sources

The sources of the step 5,000 were trained again with the same code path; their held-out accuracy and their parameter norm are compared with the rows of the existing runs.

| seed | source | the held-out of this run | the existing rows | the difference | the parameter norm of this run | the existing norm | the difference |
|---|---|---|---|---|---|---|---|
| 0 | gen | 0.9610 | 0.9610 | +0.000000 | 421.3116 | 421.3116 | +0.000000 |
| 3 | gen | 0.9740 | 0.9740 | +0.000000 | 420.1474 | 420.1474 | +0.000000 |
| 1 | gen | 0.9930 | 0.9930 | +0.000000 | 423.9987 | 423.9987 | +0.000000 |
| 2 | gen | 0.9990 | 0.9990 | +0.000000 | 419.2485 | 419.2485 | +0.000000 |
| 3 | mem | 0.0070 | 0.0070 | +0.000000 | 408.1652 | 408.1652 | +0.000000 |
| 1 | mem | 0.0060 | 0.0060 | +0.000000 | 419.1324 | 419.1324 | +0.000000 |
| 2 | mem | 0.0060 | 0.0060 | +0.000000 | 415.1717 | 415.1717 | +0.000000 |
| 0 | mem | 0.0040 | 0.0040 | +0.000000 | 419.1876 | 419.1876 | +0.000000 |
| 4 | gen | 0.9830 | 0.9830 | +0.000000 | 418.2551 | 418.2551 | +0.000000 |
| 4 | mem | 0.0030 | 0.0030 | +0.000000 | 415.3798 | 415.3798 | +0.000000 |
| 1 | gen | 0.9930 | 0.9930 | +0.000000 | 423.9987 | 423.9987 | +0.000000 |
| 0 | gen | 0.9610 | 0.9610 | +0.000000 | 421.3116 | 421.3116 | +0.000000 |
| 2 | gen | 0.9990 | 0.9990 | +0.000000 | 419.2485 | 419.2485 | +0.000000 |
| 3 | gen | 0.9740 | 0.9740 | +0.000000 | 420.1474 | 420.1474 | +0.000000 |
| 1 | mem | 0.0060 | 0.0060 | +0.000000 | 419.1324 | 419.1324 | +0.000000 |
| 2 | mem | 0.0060 | 0.0060 | +0.000000 | 415.1717 | 415.1717 | +0.000000 |
| 0 | mem | 0.0040 | 0.0040 | +0.000000 | 419.1876 | 419.1876 | +0.000000 |
| 3 | mem | 0.0070 | 0.0070 | +0.000000 | 408.1652 | 408.1652 | +0.000000 |
| 4 | gen | 0.9830 | 0.9830 | +0.000000 | 418.2551 | 418.2551 | +0.000000 |
| 4 | mem | 0.0030 | 0.0030 | +0.000000 | 415.3798 | 415.3798 | +0.000000 |

The mean absolute difference of the held-out accuracy is **0.000000** and the largest one is **0.000000**; the mean absolute difference of the parameter norm is **0.000000** and the largest one is **0.000000**, over 20 sources.

The same held-out accuracies against the `none` rows of the step 5,000 of `m20_edits.csv` (the `gen` sources):

| seed | the held-out of this run | the `none` row of queue 20 | the difference |
|---|---|---|---|
| 0 | 0.961 | 0.961 | +0.000000 |
| 3 | 0.974 | 0.974 | +0.000000 |
| 1 | 0.993 | 0.993 | +0.000000 |
| 2 | 0.999 | 0.999 | +0.000000 |
| 4 | 0.983 | 0.983 | +0.000000 |
| 1 | 0.993 | 0.993 | +0.000000 |
| 0 | 0.961 | 0.961 | +0.000000 |
| 2 | 0.999 | 0.999 | +0.000000 |
| 3 | 0.974 | 0.974 | +0.000000 |
| 4 | 0.983 | 0.983 | +0.000000 |

