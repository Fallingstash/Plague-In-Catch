import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"
CATEGORY_COLUMNS = ["Culture", "strain"]
CLASSIFIER_PARAMS = {
    "n_estimators": 500,
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
    "class_weight": "balanced",
}
REGRESSOR_PARAMS = {
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


def load_inputs():
    paths = [
        ROOT / "featurestraindesc.csv",
        ROOT / "featurestrainfp.npz",
        ROOT / "featurestrainmeta.csv",
        ROOT / "train_clean.csv",
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")

    descriptors = pd.read_csv(paths[0])
    with np.load(paths[1]) as data:
        fingerprints = data["fingerprints"]
    meta = pd.read_csv(paths[2])
    train = pd.read_csv(paths[3])
    if descriptors.shape != (318, 217) or fingerprints.shape != (318, 2048):
        raise ValueError(f"Unexpected feature shapes: {descriptors.shape}, {fingerprints.shape}")
    if len(meta) != len(train) or len(descriptors) != len(train):
        raise ValueError("Feature, metadata, and training row counts do not match")
    if not np.array_equal(meta["SMILES"].to_numpy(), train["SMILES"].to_numpy()):
        raise ValueError("Feature metadata and training rows are misaligned")

    base_features = np.concatenate(
        [descriptors.to_numpy(dtype=np.float32), fingerprints.astype(np.float32)],
        axis=1,
    )
    categories = train[CATEGORY_COLUMNS].reset_index(drop=True)
    labels = meta["isactive"].to_numpy(dtype=int)
    target_log = meta["log10SI"].to_numpy(dtype=float)
    target_raw = train["SI VACV"].to_numpy(dtype=float)
    if base_features.shape != (318, 2265) or labels.sum() != 237:
        raise ValueError("Unexpected base feature shape or active-row count")
    if np.isinf(base_features).any() or not np.isfinite(target_log[labels == 1]).all():
        raise ValueError("Features contain infinity or active targets are non-finite")
    return base_features, categories, labels, target_log, target_raw


def load_folds(row_count):
    path = RESULTS_DIR / "folds.json"
    if not path.is_file():
        raise FileNotFoundError(f"Required saved folds are missing: {path}")
    folds = json.loads(path.read_text(encoding="utf-8"))["folds"]
    expected = set(range(row_count))
    validation_indices = []
    for fold in folds:
        train_indices = set(fold["train_indices"])
        valid_indices = set(fold["validation_indices"])
        if train_indices & valid_indices or train_indices | valid_indices != expected:
            raise ValueError(f"Fold {fold.get('fold')} does not partition all rows")
        validation_indices.extend(fold["validation_indices"])
    if sorted(validation_indices) != list(range(row_count)):
        raise ValueError("Saved folds must validate every row exactly once")
    return folds


def encode_categories(train_categories, valid_categories):
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_encoded = encoder.fit_transform(train_categories)
    valid_encoded = encoder.transform(valid_categories)
    return train_encoded.astype(np.float32), valid_encoded.astype(np.float32), encoder


def main():
    base_features, categories, labels, target_log, target_raw = load_inputs()
    folds = load_folds(len(labels))
    oof_probabilities = np.full(len(labels), np.nan, dtype=float)
    oof_log = np.full(len(labels), np.nan, dtype=float)
    classifier_fold_auc = []
    regressor_fold_rmse = []
    classifier_best_iterations = []
    regressor_best_iterations = []

    for fold in folds:
        fold_id = fold["fold"]
        train_indices = np.asarray(fold["train_indices"], dtype=int)
        valid_indices = np.asarray(fold["validation_indices"], dtype=int)
        train_categories, valid_categories, _ = encode_categories(
            categories.iloc[train_indices], categories.iloc[valid_indices]
        )
        x_train = np.concatenate([base_features[train_indices], train_categories], axis=1)
        x_valid = np.concatenate([base_features[valid_indices], valid_categories], axis=1)

        classifier = lgb.LGBMClassifier(**CLASSIFIER_PARAMS)
        classifier.fit(
            x_train,
            labels[train_indices],
            eval_set=[(x_valid, labels[valid_indices])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        oof_probabilities[valid_indices] = classifier.predict_proba(
            x_valid, num_iteration=classifier.best_iteration_
        )[:, 1]
        classifier_best = int(
            classifier.best_iteration_ or CLASSIFIER_PARAMS["n_estimators"]
        )
        classifier_best_iterations.append(classifier_best)
        fold_auc = float(roc_auc_score(labels[valid_indices], oof_probabilities[valid_indices]))
        classifier_fold_auc.append(fold_auc)

        active_train_indices = train_indices[labels[train_indices] == 1]
        active_valid_indices = valid_indices[labels[valid_indices] == 1]
        valid_category_positions = np.flatnonzero(
            np.isin(valid_indices, active_valid_indices)
        )
        if not len(active_train_indices) or not len(active_valid_indices):
            raise ValueError(f"Fold {fold_id} has no active train or validation rows")
        x_reg_train = x_train[labels[train_indices] == 1]
        x_reg_valid_active = x_valid[valid_category_positions]

        regressor = lgb.LGBMRegressor(**REGRESSOR_PARAMS)
        regressor.fit(
            x_reg_train,
            target_log[active_train_indices],
            eval_set=[
                (x_reg_valid_active, target_log[active_valid_indices])
            ],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        oof_log[valid_indices] = regressor.predict(
            x_valid, num_iteration=regressor.best_iteration_
        )
        regressor_best = int(regressor.best_iteration_ or REGRESSOR_PARAMS["n_estimators"])
        regressor_best_iterations.append(regressor_best)
        fold_rmse = float(
            np.sqrt(
                mean_squared_error(
                    target_log[active_valid_indices], oof_log[active_valid_indices]
                )
            )
        )
        regressor_fold_rmse.append(fold_rmse)
        print(
            f"Fold {fold_id}: classifier_auc={fold_auc:.6f}, "
            f"classifier_best={classifier_best}, active_rmse={fold_rmse:.6f}, "
            f"regressor_best={regressor_best}, features={x_train.shape[1]}"
        )

    if not np.isfinite(oof_probabilities).all() or not np.isfinite(oof_log).all():
        raise RuntimeError("OOF predictions contain missing or non-finite values")

    active = labels == 1
    predicted_active_si = np.power(10.0, np.clip(oof_log, -2.0, 5.0))
    blended_si = (1.0 - oof_probabilities) * 10000.0 + oof_probabilities * predicted_active_si
    mse_raw = float(mean_squared_error(target_raw, blended_si))
    roc_auc = float(roc_auc_score(labels, oof_probabilities))
    rmse_log = float(np.sqrt(mean_squared_error(target_log[active], oof_log[active])))
    exp05_path = RESULTS_DIR / "exp05combined_soft.json"
    if not exp05_path.is_file():
        raise FileNotFoundError(f"Required exp05 result is missing: {exp05_path}")
    mse_base_exp05 = float(
        json.loads(exp05_path.read_text(encoding="utf-8"))["cvmetric"]
    )

    np.save(RESULTS_DIR / "exp06oof_probs.npy", oof_probabilities)
    np.save(RESULTS_DIR / "exp06oof_log10.npy", oof_log)
    result = {
        "model": "combinedsoftwithstrain",
        "cvmetric": mse_raw,
        "metricname": "MSE_raw",
        "msebaseexp05": mse_base_exp05,
        "deltavsexp05": mse_raw - mse_base_exp05,
        "rocauc": roc_auc,
        "rmselog": rmse_log,
        "classifier_fold_auc": classifier_fold_auc,
        "regressor_fold_rmse_log": regressor_fold_rmse,
        "classifier_best_iterations": classifier_best_iterations,
        "regressor_best_iterations": regressor_best_iterations,
        "onehot_feature_count_per_fold": int(x_train.shape[1] - base_features.shape[1]),
        "notes": "soft blend with fold-local one-hot encoding of Culture and strain; delta is exp06 MSE minus exp05 MSE.",
    }
    result_path = RESULTS_DIR / "exp06combinedwithstrain.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"MSE_raw: {mse_raw:.6g}")
    print(f"exp05 MSE_raw: {mse_base_exp05:.6g}")
    print(f"Delta vs exp05 (exp06 - exp05): {result['deltavsexp05']:.6g}")
    print(f"ROC-AUC: {roc_auc:.6f}")
    print(f"Active RMSE_log: {rmse_log:.6f}")
    print(f"Saved result: {result_path}")


if __name__ == "__main__":
    main()
