import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"
PARAMS = {
    "objective": "regression",
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
}
EARLY_STOPPING_ROUNDS = 50


def main():
    descriptor_path = ROOT / "featurestraindesc.csv"
    fingerprint_path = ROOT / "featurestrainfp.npz"
    train_path = ROOT / "train_clean.csv"
    folds_path = RESULTS_DIR / "folds.json"
    for path in (descriptor_path, fingerprint_path, train_path, folds_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")

    descriptors = pd.read_csv(descriptor_path)
    with np.load(fingerprint_path) as data:
        fingerprints = data["fingerprints"]
    train = pd.read_csv(train_path)
    folds = json.loads(folds_path.read_text(encoding="utf-8"))["folds"]
    if descriptors.shape != (318, 217) or fingerprints.shape != (318, 2048):
        raise ValueError(
            f"Expected 318x217 descriptors and 318x2048 fingerprints, "
            f"got {descriptors.shape} and {fingerprints.shape}"
        )
    if len(train) != 318:
        raise ValueError(f"Expected 318 original training rows, got {len(train)}")

    features = np.concatenate(
        [descriptors.to_numpy(dtype=np.float32), fingerprints.astype(np.float32)],
        axis=1,
    )
    target = pd.to_numeric(train["SI VACV"], errors="coerce").to_numpy(dtype=float)
    if features.shape != (318, 2265) or not np.isfinite(target).all():
        raise ValueError("Unexpected feature shape or invalid raw SI target")

    expected = set(range(len(train)))
    validation_indices = []
    for fold in folds:
        train_indices = set(fold["train_indices"])
        valid_indices = set(fold["validation_indices"])
        if train_indices & valid_indices or train_indices | valid_indices != expected:
            raise ValueError(f"Saved fold {fold.get('fold')} does not partition the rows")
        validation_indices.extend(fold["validation_indices"])
    if sorted(validation_indices) != list(range(len(train))):
        raise ValueError("Saved folds must validate every row exactly once")

    oof_prediction = np.full(len(train), np.nan, dtype=float)
    fold_mses = []
    best_iterations = []
    for fold in folds:
        train_indices = np.asarray(fold["train_indices"], dtype=int)
        valid_indices = np.asarray(fold["validation_indices"], dtype=int)
        model = lgb.LGBMRegressor(**PARAMS)
        model.fit(
            features[train_indices],
            target[train_indices],
            eval_set=[(features[valid_indices], target[valid_indices])],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        oof_prediction[valid_indices] = model.predict(
            features[valid_indices], num_iteration=model.best_iteration_
        )
        best_iteration = int(model.best_iteration_ or PARAMS["n_estimators"])
        best_iterations.append(best_iteration)
        fold_mse = float(mean_squared_error(target[valid_indices], oof_prediction[valid_indices]))
        fold_mses.append(fold_mse)
        print(
            f"Fold {fold['fold']}: MSE_raw={fold_mse:.6g}, "
            f"best_iteration={best_iteration}"
        )

    if not np.isfinite(oof_prediction).all():
        raise RuntimeError("Raw-target OOF predictions contain missing/non-finite values")
    mse_raw = float(mean_squared_error(target, oof_prediction))
    result = {
        "model": "lgbmregressor_raw_SI",
        "params": PARAMS
        | {
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "eval_metric": "rmse",
            "target": "SI VACV (raw)",
        },
        "cvmetric": mse_raw,
        "metricname": "MSEraw",
        "foldmetrics": fold_mses,
        "best_iterations": best_iterations,
        "notes": "Original 318 rows and saved folds; direct raw-SI regression; no IC50/CC50 features.",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(RESULTS_DIR / "exp08oof_raw_si.npy", oof_prediction)
    result_path = RESULTS_DIR / "exp08raw_target.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"MSE_raw (exp08): {mse_raw:.6g}")
    baseline_path = RESULTS_DIR / "exp01baseline_b.json"
    baseline_mse = float(
        json.loads(baseline_path.read_text(encoding="utf-8"))["cvmetric"]
    )
    print(f"MSE_raw (exp01 log target): {baseline_mse:.6g}")
    print(f"exp08 / exp01: {mse_raw / baseline_mse:.6g}")
    print(f"Best iterations: {best_iterations}")
    print(f"Saved result: {result_path}")


if __name__ == "__main__":
    main()
