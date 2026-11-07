# VACV QSAR

## Confirmed best submissions

- `versions/top1/submission.csv`: Kaggle MSE **886,520.98398**, CPU + one-layer GCN at 70/30.
- `versions/top2/submission.csv`: Kaggle MSE **936,601.51958**, CPU + one-layer GCN at 85/15. Reconstructed from saved components; historical score is user-reported.

## Current candidate

`versions/test/submission.csv`: CPU + one-layer GCN at 65/35. The 52% candidate scored **1,159,246.64900** on Kaggle despite better local OOF, so we are stepping back toward the confirmed 30% result. At 35%, mean paired OOF MSE is **4,057,964** (30%: 4,075,411) and maximum prediction is **15,347.85**. Candidate Kaggle score is pending.

## Validation and sweep

`test.csv` has no target labels, so its true MSE cannot be computed locally. Candidate selection uses paired OOF predictions on `train.csv`, identical canonical grouped folds and seeds 42/43, with every fold included. Full 0%-100%, one-percentage-point sweep is in `reports/gcn_weight_sweep_1pct.csv`; actual reported submission outcomes are in `reports/submission_leaderboard.csv`.

The sweep's local MSE improved as GCN weight increased, but the 52% public result disproved that trend as a reliable Kaggle predictor. Treat OOF as a filter, not a guarantee. Preserve `top1` and `top2` unchanged.

## Rules

- Fit/CV use only `train.csv` and `SI VACV`.
- `test.csv` is used only for inference; no labels are used.
- `CC50_uM`, `IC50_uM_VACV`, and external activity labels are excluded.
- The GCN is our own PyTorch implementation trained from scratch; it combines atom/bond graphs, Morgan fingerprints, MACCS keys, and RDKit descriptors.
