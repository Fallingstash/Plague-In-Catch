# Handling large SI without editing targets

Eight fixed GCN training schemes and four calibration schemes were measured on identical canonical grouped folds, seeds 42/43. Every target and every fold is retained. No competition test labels or predictions are used to choose parameters.

The control exactly reproduces the saved reference350 OOF predictions. Every alternative replaces the GCN component while retaining the fixed CPU70/GNN30 recipe, or calibrates the complete reference using inner OOF.

| model | mean_MSE | mean_reference_MSE | mean_MAE | extreme_prediction_mean | without_extreme_DIAGNOSTIC |
|---|---|---|---|---|---|
| control_log | 4,074,346.07 | 4,074,346.07 | 220.81 | 847.62 | 198,760.99 |
| log_raw1 | 4,081,921.77 | 4,074,346.07 | 220.52 | 805.09 | 196,971.76 |
| log_raw10 | 4,089,759.25 | 4,074,346.07 | 221.72 | 776.06 | 198,416.29 |
| isotonic_gcn | 4,094,750.89 | 4,074,346.07 | 239.36 | 724.47 | 191,770.20 |
| affine_gcn | 4,096,738.20 | 4,074,346.07 | 252.38 | 856.75 | 223,186.56 |
| weighted_log | 4,114,672.43 | 4,074,346.07 | 221.30 | 640.00 | 193,243.84 |
| affine_reference | 4,131,021.24 | 4,074,346.07 | 338.06 | 1,005.92 | 290,370.57 |
| mixture | 4,183,150.18 | 4,074,346.07 | 221.93 | 308.38 | 187,940.28 |
| root_mse | 4,194,051.55 | 4,074,346.07 | 245.87 | 507.19 | 243,323.06 |
| isotonic_reference | 4,253,448.32 | 4,074,346.07 | 323.04 | 836.22 | 375,241.42 |
| dual_raw | 4,268,194.99 | 4,074,346.07 | 251.42 | 221.09 | 253,655.05 |
| raw_mse | 4,308,725.38 | 4,074,346.07 | 252.97 | 265.59 | 304,309.09 |

The maximum target is row 14: SI=36066.667. With the seed-42 reference prediction 780.777, its single-row contribution is 3,890,918.86 to the 320-row MSE. The squared error is divided by 320, not ignored.

When that row is held out, the fold training target maximum is only 5,000. The model has no training example with SI near 36,000. It can extrapolate, but no loss function can guarantee the unknown magnitude from this sample alone.

Raw MSE, square-root target regression, train-only target-dependent weighting, stronger mixed log/raw losses, a two-output raw/log network and a gated high/low expert model were all trained. Gradients are clipped and nonfinite losses/gradients cause an explicit failure; no hard fold or observation is skipped.

Affine and isotonic calibrators are trained using three-fold inner OOF predictions from the outer training rows. They never see the outer validation target during fitting. Isotonic endpoints come from the training data; there is no manually imposed SI cutoff.

No tested alternative beats the reference on both seeds. Some reduce error on the largest target for one split but worsen the remainder or fail on the other split. They are saved as experiments and not promoted.

MSE_without_extreme is diagnostic only. The acceptance decision uses the complete MSE; exclusion of the extreme target is never substituted for the competition metric.

Recommendation: retain confirmed top1. No new submission is generated for these rejected methods.

Commands: python -m src.tail_research; python -m src.tail_calibration; python -m src.tail_report. Calibration uses the cached 350-tree nested reference checkpoints created by python main.py evaluate --nested.
