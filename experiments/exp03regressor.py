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
    train_path = ROOT / "train_clean.csv"
    for path in (descriptor_path, fingerprint_path, meta_path, train_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")

    descriptors = pd.read_csv(descriptor_path)
    with np.load(fingerprint_path) as data:
        fingerprints = data["fingerprints"]
    meta = pd.read_csv(meta_path)
    train = pd.read_csv(train_path)
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
    active = meta["isactive"].to_numpy(dtype=int) == 1
    target_log = meta["log10SI"].to_numpy(dtype=float)
    target_raw = train["SI VACV"].to_numpy(dtype=float)
    if features.shape != (318, 2265) or active.sum() != 237:
        raise ValueError(f"Unexpected feature shape or active count: {active.sum()}")
    if np.isinf(features).any() or not np.isfinite(target_log[active]).all():
        raise ValueError("Features contain infinite values or active targets are non-finite")
    return features, active, target_log, target_raw


def load_folds(row_count):
    folds_path = RESULTS_DIR / "folds.json"
    if not folds_path.is_file():
        raise FileNotFoundError(f"Required saved folds are missing: {folds_path}")
    folds = json.loads(folds_path.read_text(encoding="utf-8"))["folds"]
    seen_validation = []
    expected = set(range(row_count))
    for fold in folds:
        train_indices = set(fold["train_indices"])
        validation_indices = set(fold["validation_indices"])
        if train_indices & validation_indices:
            raise ValueError(f"Fold {fold.get('fold')} has train/validation overlap")
        if train_indices | validation_indices != expected:
            raise ValueError(f"Fold {fold.get('fold')} does not partition all rows")
        seen_validation.extend(fold["validation_indices"])
    if sorted(seen_validation) != list(range(row_count)):
        raise ValueError("Saved folds must validate every row exactly once")
    return folds


def main():
    features, active, target_log, target_raw = load_features()
    folds = load_folds(len(active))
    active_indices = np.flatnonzero(active)
    oof_log = np.full(len(active), np.nan, dtype=float)
    fold_metrics = []
    best_iterations = []

    for fold in folds:
        fold_id = fold["fold"]
        train_mask = active & np.isin(np.arange(len(active)), fold["train_indices"])
        validation_mask = np.isin(np.arange(len(active)), fold["validation_indices"])
        train_indices = np.flatnonzero(train_mask)
        validation_indices = np.flatnonzero(validation_mask)
        validation_active_indices = validation_indices[active[validation_indices]]
        if not len(train_indices) or not len(validation_active_indices):
            raise ValueError(f"Fold {fold_id} has no active train or validation rows")

        model = lgb.LGBMRegressor(
            **{
                key: value
                for key, value in PARAMS.items()
                if key not in ("early_stopping_rounds", "eval_metric")
            }
        )
        model.fit(
            features[train_indices],
            target_log[train_indices],
            eval_set=[
                (
                    features[validation_active_indices],
                    target_log[validation_active_indices],
                )
            ],
            eval_metric=PARAMS["eval_metric"],
            callbacks=[
                lgb.early_stopping(PARAMS["early_stopping_rounds"], verbose=False)
            ],
        )
        oof_log[validation_indices] = model.predict(
            features[validation_indices],
            num_iteration=model.best_iteration_,
        )
        best_iteration = int(model.best_iteration_ or PARAMS["n_estimators"])
        best_iterations.append(best_iteration)

        active_fold_log = oof_log[validation_active_indices]
        active_fold_si = np.power(10.0, np.clip(active_fold_log, -2.0, 5.0))
        fold_rmse = float(
            np.sqrt(
                mean_squared_error(
                    target_log[validation_active_indices], active_fold_log
                )
            )
        )
        fold_metrics.append(fold_rmse)
        print(
            f"Fold {fold_id}: active_train={len(train_indices)}, "
            f"active_validation={len(validation_active_indices)}, "
            f"best_iteration={best_iteration}, RMSE_log={fold_rmse:.6f}"
        )

    if not np.isfinite(oof_log).all():
        raise RuntimeError("OOF log prediction array contains missing or non-finite values")
    active_log = oof_log[active_indices]
    active_raw = target_raw[active_indices]
    active_prediction_si = np.power(10.0, np.clip(active_log, -2.0, 5.0))
    rmse_log = float(np.sqrt(mean_squared_error(target_log[active_indices], active_log)))
    mse_raw = float(mean_squared_error(active_raw, active_prediction_si))
    mae_raw = float(mean_absolute_error(active_raw, active_prediction_si))
    r2_log = float(r2_score(target_log[active_indices], active_log))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(RESULTS_DIR / "exp03oof_log10.npy", oof_log)
    result = {
        "model": "lgbmregressoron_active",
        "params": PARAMS,
        "cvmetric": rmse_log,
        "metricname": "RMSElog10",
        "mseraw": mse_raw,
        "maeraw": mae_raw,
        "r2_log": r2_log,
        "foldmetrics": fold_metrics,
        "bestiterations": best_iterations,
        "notes": "C stage 2; active-only training and metrics; OOF predictions for all rows on saved folds.",
    }
    result_path = RESULTS_DIR / "exp03regressor.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Active RMSE_log10: {rmse_log:.6f}")
    print(f"Active R2_log: {r2_log:.6f}")
    print(f"Active MSE_raw: {mse_raw:.6g}")
    print(f"Active MAE_raw: {mae_raw:.6g}")
    print(f"Best iterations: {best_iterations}")
    print(f"Saved result: {result_path}")


if __name__ == "__main__":
    main()
