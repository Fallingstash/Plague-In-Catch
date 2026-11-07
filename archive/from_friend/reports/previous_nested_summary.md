# Nested selection of a small molecular ensemble

The model compares fixed configurations of a CPU log neural net (38 epochs), ExtraTrees (150 trees), and log1p Tanimoto kernel ridge (alpha 0.1). Inputs are exclusively RDKit molecular features derived from SMILES. For every one of ten outer validation folds (5 folds for each of two random seeds 42/43), nine predefined mixtures are evaluated on three inner canonical GroupKFold splits of the outer training data; the mixture with lowest inner OOF MSE is trained on all outer training data and evaluated on the outer fold. No outer validation targets are used to select its weights. Candidate families and settings were informed by earlier exploration of the same dataset, so this is not a completely untouched final test. The folds contain every molecule, and the training/validation canonical groups are disjoint.

| model | seed 42 MSE | seed 43 MSE |
|---|---:|---:|
| Nested-selected mixture | 4 189 848 | 4 191 229 |
| Log neural network | 4 222 608 | 4 196 817 |
| ExtraTrees | 4 329 209 | 4 220 223 |
| Log Tanimoto kernel ridge | 4 233 501 | 4 241 247 |

The previously recorded CPU ensemble MSE 4 190 501 is optimistic because weights were fitted on the same OOF predictions; it cannot be compared as an equally independent evaluation. This experiment does not establish a new Kaggle record or guarantee improvement.

The final full-data inner selection chose weights [neural=0, ExtraTrees=1, kernel=0]. Therefore `submission.csv` is currently a tree-only prediction, not a neural ensemble. Full-data choice can differ from the selections in outer CV. The Kaggle score of this candidate is unknown. The existing record submission and model are preserved.

`nested_folds.csv` has fold metrics and selected weights; `oof.csv` contains predictions per molecule and per seed; `model.joblib` stores fitted components, weights, preprocessing and train SHA-256. `nested_predict.py` loads it and predicts CSVs. Save/reload reproduces test predictions to 2.3e-13. The sample schema, order, 597 rows, finite and nonnegative outputs were checked. Treat joblib as executable content and load only trusted files.
