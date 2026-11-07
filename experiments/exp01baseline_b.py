import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"
PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 5,
    "min_child_samples": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
    "early_stopping_rounds": 50,
    "eval_metric": "rmse",
}


def load_features():
    descriptor_path = ROOT / "featurestraindesc.csv"
    fingerprint_path = ROOT / "featurestrainfp.npz"
    meta_path = ROOT / "featurestrainmeta.csv"
    raw_target_path = ROOT / "train_clean.csv"
    for path in (descriptor_path, fingerprint_path, meta_path, raw_target_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")

    descriptors = pd.read_csv(descriptor_path)
    with np.load(fingerprint_path) as data:
        fingerprints = data["fingerprints"]
    meta = pd.read_csv(meta_path)
    train = pd.read_csv(raw_target_path)

    if descriptors.shape != (318, 217) or fingerprints.shape != (318, 2048):
        raise ValueError(
            f"Unexpected feature shapes: {descriptors.shape}, {fingerprints.shape}"
        )
    if len(meta) != len(train) or len(descriptors) != len(train):
        raise ValueError("Feature and target row counts do not match")
    if not np.array_equal(meta["SMILES"].to_numpy(), train["SMILES"].to_numpy()):
        raise ValueError("Feature metadata and training target rows are misaligned")

    features = np.concatenate(
        [descriptors.to_numpy(dtype=np.float32), fingerprints.astype(np.float32)],
        axis=1,
    )
    target_log = meta["log10SI"].to_numpy(dtype=float)
    target_raw = train["SI VACV"].to_numpy(dtype=float)
    if features.shape != (318, 2265):
        raise ValueError(f"Unexpected combined feature shape: {features.shape}")
    if not np.isfinite(target_log).all() or not np.isfinite(target_raw).all():
        raise ValueError("Targets contain missing or non-finite values")
    return features, target_log, target_raw


def load_folds(row_count):
    folds_path = RESULTS_DIR / "folds.json"
    if not folds_path.is_file():
        raise FileNotFoundError(f"Required saved folds are missing: {folds_path}")
    data = json.loads(folds_path.read_text(encoding="utf-8"))
    folds = data["folds"]
    validation_indices = []
    expected = set(range(row_count))
    for fold in folds:
        train_indices = set(fold["train_indices"])
        valid_indices = set(fold["validation_indices"])
        if train_indices & valid_indices or train_indices | valid_indices != expected:
            raise ValueError(f"Fold {fold.get('fold')} does not partition all rows")
        validation_indices.extend(fold["validation_indices"])
    if sorted(validation_indices) != list(range(row_count)):
        raise ValueError("Saved validation folds must assign every row exactly once")
    return folds


def main():
    features, target_log, target_raw = load_features()
    folds = load_folds(len(target_log))
    oof_log = np.full(len(target_log), np.nan, dtype=float)
    fold_mses = []
    best_iterations = []

    for fold in folds:
        fold_id = fold["fold"]
        train_indices = np.asarray(fold["train_indices"], dtype=int)
        validation_indices = np.asarray(fold["validation_indices"], dtype=int)
        model = lgb.LGBMRegressor(
            **{key: value for key, value in PARAMS.items() if key not in ("early_stopping_rounds", "eval_metric")}
        )
        model.fit(
            features[train_indices],
            target_log[train_indices],
            eval_set=[(features[validation_indices], target_log[validation_indices])],
            eval_metric=PARAMS["eval_metric"],
            callbacks=[lgb.early_stopping(PARAMS["early_stopping_rounds"], verbose=False)],
        )
        oof_log[validation_indices] = model.predict(
            features[validation_indices],
            num_iteration=model.best_iteration_,
        )
        best_iterations.append(int(model.best_iteration_ or PARAMS["n_estimators"]))
        fold_prediction_si = np.power(
            10.0, np.clip(oof_log[validation_indices], -2.0, 5.0)
        )
        fold_mses.append(
            float(mean_squared_error(target_raw[validation_indices], fold_prediction_si))
        )
        print(
            f"Fold {fold_id}: best_iteration={best_iterations[-1]}, "
            f"MSE_raw={fold_mses[-1]:.6g}"
        )

    if not np.isfinite(oof_log).all():
        raise RuntimeError("OOF prediction array contains missing or non-finite values")
    oof_log = np.clip(oof_log, -2.0, 5.0)
    prediction_si = np.power(10.0, oof_log)
    mse_raw = float(mean_squared_error(target_raw, prediction_si))
    rmse_log = float(np.sqrt(mean_squared_error(target_log, oof_log)))
    r2_log = float(r2_score(target_log, oof_log))
    mae_raw = float(mean_absolute_error(target_raw, prediction_si))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(RESULTS_DIR / "exp01oof_log10.npy", oof_log)
    result = {
        "model": "lgbmregressoron_log10",
        "params": PARAMS,
        "cvmetric": mse_raw,
        "metricname": "MSEraw",
        "rmse_log": rmse_log,
        "r2_log": r2_log,
        "mae_raw": mae_raw,
        "fold_metrics": fold_mses,
        "best_iterations": best_iterations,
        "notes": "baseline B; out-of-fold predictions on saved folds; log predictions clipped to [-2, 5] before exponentiation and scoring.",
    }
    result_path = RESULTS_DIR / "exp01baseline_b.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"MSE_raw: {mse_raw:.6g}")
    print(f"RMSE_log: {rmse_log:.6g}")
    print(f"R2_log: {r2_log:.6g}")
    print(f"MAE_raw: {mae_raw:.6g}")
    print(f"Best iterations: {best_iterations}")
    print(f"Saved result: {result_path}")


if __name__ == "__main__":
    main()
