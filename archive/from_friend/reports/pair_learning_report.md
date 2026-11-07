# Pairwise SI learning

Inspired by the pairwise property-difference idea in [DeepDelta](https://pmc.ncbi.nlm.nih.gov/articles/PMC10605784/). This is a from-scratch fingerprint/descriptor implementation, not downloaded DeepDelta weights or a reproduction of its benchmarks.

Official 320 training labels only. Pairs use 8 nearest structural neighbors plus 2 random partners per training molecule, both directions, excluding identical canonical structures. These are similarity pairs, not certified matched-molecular-pair fragment decompositions.

Every pair endpoint and every inference anchor belongs to the current training fold. Descriptors are scaled on that fold. Validation and test targets are never used as anchors. The number of pairs is not the number of independent experimental observations.

Seven configurations: anchor-only ablation; Ridge log/raw; pair Tanimoto-kernel log/raw; antisymmetric context-gated neural log/raw. Fixed comparisons: standalone, reference90/pair10, CPU70/pair30. Same canonical grouped 5 folds, seeds 42/43; every row included.

| mode | variant | mean_MSE | reference_MSE | mean_MAE | without_max_DIAGNOSTIC |
|---|---|---|---|---|---|
| ridge_log | standalone | 3,999,426.65 | 4,074,346.07 | 275.54 | 565,138.72 |
| ridge_log | add10 | 4,040,092.58 | 4,074,346.07 | 225.50 | 209,929.29 |
| ridge_log | replace30 | 4,051,136.59 | 4,074,346.07 | 235.88 | 229,284.75 |
| neural_log | add10 | 4,063,581.02 | 4,074,346.07 | 229.82 | 224,433.27 |
| kernel_log | add10 | 4,086,181.78 | 4,074,346.07 | 219.74 | 197,765.15 |
| anchor_mean | add10 | 4,090,474.57 | 4,074,346.07 | 230.97 | 207,085.01 |
| kernel_raw | add10 | 4,092,944.82 | 4,074,346.07 | 238.73 | 209,885.48 |
| neural_raw | add10 | 4,101,688.13 | 4,074,346.07 | 241.79 | 224,209.58 |
| ridge_raw | add10 | 4,129,459.30 | 4,074,346.07 | 284.04 | 256,470.81 |
| kernel_log | replace30 | 4,192,330.04 | 4,074,346.07 | 217.35 | 194,580.31 |
| neural_log | replace30 | 4,192,922.65 | 4,074,346.07 | 249.26 | 344,365.90 |
| kernel_log | standalone | 4,214,250.08 | 4,074,346.07 | 212.97 | 209,441.39 |
| anchor_mean | replace30 | 4,221,053.29 | 4,074,346.07 | 254.54 | 238,637.53 |
| kernel_raw | replace30 | 4,241,968.79 | 4,074,346.07 | 280.51 | 260,605.34 |
| neural_raw | replace30 | 4,328,822.38 | 4,074,346.07 | 287.15 | 364,597.36 |
| ridge_raw | replace30 | 4,535,763.95 | 4,074,346.07 | 419.64 | 585,528.39 |
| anchor_mean | standalone | 4,679,628.85 | 4,074,346.07 | 341.47 | 727,021.91 |
| kernel_raw | standalone | 4,922,261.71 | 4,074,346.07 | 431.91 | 973,679.59 |
| neural_log | standalone | 5,344,991.89 | 4,074,346.07 | 320.43 | 1,829,963.30 |
| neural_raw | standalone | 6,084,803.76 | 4,074,346.07 | 454.10 | 2,195,703.48 |
| ridge_raw | standalone | 8,267,600.64 | 4,074,346.07 | 908.07 | 4,429,362.91 |

Fixed reference90 + pair-ridge10 scored mean OOF MSE 4,040,092.58 versus 4,074,346.07 for reference, a 0.84% decrease. The error excluding the maximum target increased; the small apparent benefit concentrates in the largest observation.

## Conditional nested follow-up

| seed | MSE | reference_MSE | extreme_prediction | selected_weights |
|---|---|---|---|---|
| 42 | 4,163,135.58 | 4,101,813.50 | 780.78 | [0.3, 0.3, 0.3, 0.1, 0.0] |
| 43 | 4,047,317.26 | 4,046,878.64 | 914.46 | [0.0, 0.3, 0.3, 0.0, 0.0] |

Within each outer training fold, a three-fold inner OOF check selected pair weight among 0, 0.1 and 0.3. Mean outer MSE was 4,105,226.42; the gain did not survive. Both extreme-target folds selected weight 0.

The family was selected after exploratory screening, so this follow-up is conditional rather than an independent estimate of the entire seven-family search. No Kaggle test inference was used to choose a configuration.

Raw pair neural networks and raw Ridge produced large errors on some folds; no difficult fold was removed. Log-delta Ridge was more stable than the larger pair network.

Conclusion: preserve top1. The pair method provides a mildly different predictor, but this experiment does not justify spending a Kaggle submission. All checkpoints, pair endpoints, anchors and OOF predictions are saved.

Detailed files: pair_model_comparison.csv, pair_summary.csv, pair_nested_comparison.csv and pair_extreme_anchors.csv in reports/. Confidence intervals bootstrap original canonical molecule groups; repeated seeds are averaged per original row.

Commands: python -m src.pair_research; python -m src.pair_nested; python -m src.pair_report.
