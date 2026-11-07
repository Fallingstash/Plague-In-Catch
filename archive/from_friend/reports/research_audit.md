# Reproducible train-only research

Official train: 320 rows, 312 canonical groups, 83 zeros; SHA256 `d9a876a67b5623d9cdb6592fc53d31ffaa7c535b09968c3a538138740fe246e8`.

Targets remain unchanged. 6 canonical groups contain inconsistent repeated SI measurements; all rows are retained and assigned as groups.

No Git repository was present. Initial files were snapshotted and protected top1/top2 hashes have remained unchanged.

The official SHA256-pinned loader rejects external or test data as training input. Feature generation takes only SMILES; forbidden concentrations are excluded before parsing.

Repeated seeds are scored separately. The headline mean is the mean of seed MSE values, never 640 independent molecules. All folds and all 320 targets are included.

The largest target contributes about 95% of baseline OOF SSE. Removing it is diagnostic only; it is never removed from model selection.

Nested CV uses 5 outer grouped folds and 3 inner grouped folds. Family/regularization/static weights/similarity gates/regularized stacking weights are selected on inner OOF. Full-data Optuna results are not fed into this outer assessment.

The legacy 250-tree baseline is retained for reproduction. New reference350 runs and separate nested tree350 runs use 350 trees consistently with final training. Old and corrected protocols are never pooled as one algorithm.

The historical baseline was previously developed on this same train; no repeated CV can turn it into a wholly untouched independent dataset.

The old test-max threshold sweep is retained as historical data and excluded from all new selection. No arbitrary SI ceiling is fitted or chosen from competition test predictions.

GCN graph depth 2 also introduced a residual connection, so the previous comparison was not a pure depth-only ablation. Its MAE improved but MSE worsened.

Compact D-MPNN, bond-aware GCN, mixed log/raw GCN loss, three-initialization GCN mean and residual MLP were actually trained on both canonical seeds. Scores are exploratory unless nested-assessed.

CatBoost/Optuna/pytest were installed. Optuna uses SQLite, a finite 8-trial run, median pruning, three grouped tuning folds and a 180-second timeout. Its tuning score is not reported as independent validation.

Scientific export contains 59 real ChEMBL candidate pairs, 0 primary-source-validated pairs, and training_allowed=False. It cannot be used as competition training data.

Historical second-best 2-layer checkpoint and submission were recovered exactly from fixed architecture/seed and recorded per-row predictions. Its original CSV SHA256 was reproduced. The existing top2 folder is an older preserved third-best; accurate ranking is in versions/ranking.json.

All checked submission/model pairs reproduce to floating point tolerance. Main CLI now defaults to top1 rather than the stale 5% reference bundle.

Optuna-selected settings are not asserted to improve Kaggle. No Kaggle upload was performed.

Primary implementation references: [KernelRidge](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html), [CatBoost parameters](https://catboost.ai/docs/en/references/training-parameters/common), [Optuna persistent studies](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html).
