# Phase 9 Testing Utility Significance Audit

## Objective

Test whether the matched-budget difference between clipping-fork alarms and absolute-delta ranking is statistically supported, and measure how each method allocates alarms across mutation families.

## Paired Result

At `425` alarms, the paired correctness table has `114` fork-only correct rows and `124` delta-only correct rows. The exact McNemar p-value is `0.559727`.

The cluster-bootstrap 95% interval for delta-minus-fork precision is `[0.0, 0.03142857142857147]`; for recall it is `[0.0, 0.0014322916666666685]`. The bootstrap conditions on the 15-mutation catalog and resamples aligned case clusters.

The small aggregate advantage previously assigned to delta ranking is not statistically distinguishable from zero under either paired token analysis or clustered resampling.

## Mutation-Family Coverage

Fork alarms identify at least one row in `11/15` mutation kinds; delta top-budget alarms identify `1/15`.

The largest mutation family accounts for `84.762%` of fork true positives and `100.000%` of delta-ranking true positives. In this catalog, every delta top-budget true positive comes from the catastrophic `rmsnorm_no_upcast` mutation.

This diversity endpoint favors fork alarms, but the 15 operators are artificial and not independent historical bugs. It supports a test-budget allocation claim, not a universal detection-superiority claim.

## Ground-Truth Limitation

The current labels ask whether a row came from an artificial mutation. They do not say whether that mutation changes short-horizon training behavior. The subsequent matched trajectory experiment in `reports/phase10_zero_fork_mutation_trajectories.md` shows that all four initial-zero-clipping-fork mutations create a nonzero first update and delayed clipping forks by step 2 or 3. They are not training-equivalent. Other discrete events, especially sampling, remain outside this test.

## Artifacts

- `results/phase9_testing_utility_significance.json`
- `scripts/phase9_testing_utility_significance.py`
- `results/phase10_zero_fork_mutation_trajectories.json`
