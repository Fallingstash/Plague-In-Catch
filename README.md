# VACV Selectivity Index Prediction

Predict the selectivity index (SI) against vaccinia virus from molecular SMILES.
The competition metric is MSE on raw SI values.

## Pipeline

The final workflow is a two-stage GCN:

1. A graph classifier estimates whether SI is positive.
2. A separate regressor predicts `log1p(SI)` for positive-SI molecules.
3. An isotonic calibrator, fitted on out-of-fold classifier probabilities,
   calibrates the test probabilities. The final estimate is
   `calibrated_probability * expm1(regressor_log1p_SI)`.

The GCN uses one message-passing layer (64 hidden units), mean pooling, and
Morgan, MACCS, and RDKit descriptor features. Five seeds (42-46) and five
stratified group folds are used. Folds group identical canonical SMILES.
Only SMILES-derived features are used; assay measurements and hidden test
columns are not model inputs.

The teammate's isotonic implementation calibrates continuous SI regression
outputs to SI targets. This pipeline instead calibrates the activity
probability, as required by the final two-stage prediction formula; it does
not claim to reproduce the teammate's fitted regression calibrator.

## Results and prior experiments

The reproduced teammate GCN baseline reached about 3.832M raw-SI OOF MSE.
Earlier LGBM (about 14M), GIN, multi-task, and Figshare augmentation
experiments did not improve on that baseline. Figshare augmentation and the
ChEMBL multi-task experiment showed negative transfer and were excluded from
the final model. Previous scripts and artifacts are retained under `archive/`.
The current five-seed two-stage run has classifier ROC-AUC about 0.912 and
diagnostic calibrated-blend OOF MSE about 4.181M, so it does not beat the
3.832M baseline. The test prediction maximum is about 25,330, above the
requested ~9,000 target. It is reported as-is; predictions are not artificially
clipped. Detailed prior experiment results are in `archive/`.

## Data

Place the required inputs in `data/inputs/`:

- `train_v2.csv` — 320 labeled molecules with `SMILES` and `SI VACV`.
- `test_clean.csv` — 597 molecules with `index` and `SMILES`.

Additional collected ChEMBL, Figshare, and article data are retained under
`data/`; they are not used in the final model because prior validation did not
show reliable improvement.

## Run

Install dependencies from `requirements.txt`, then run:

```powershell
python src/pipeline.py
python experiments/validation.py
```

For the existing Windows virtual environment, use
`.\\.venv\\Scripts\\python.exe` in place of `python`.

The pipeline writes `submission/submission_final.csv`, fitted models, OOF
arrays, and a JSON report under `experiments/results/`. Predictions are
validated for row count, finite values, and nonnegativity. The observed test
maximum is reported, not artificially clipped to a target value.
