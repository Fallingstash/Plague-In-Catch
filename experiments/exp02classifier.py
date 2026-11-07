import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"
PARAMS = {
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
    "early_stopping_rounds": 50,
    "eval_metric": "auc",
}


def load_features():
    descriptor_path = ROOT / "featurestraindesc.csv"
    fingerprint_path = ROOT / "featurestrainfp.npz"
    meta_path = ROOT / "featurestrainmeta.csv"
    for path in (descriptor_path, fingerprint_path, meta_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")

    descriptors = pd.read_csv(descriptor_path)
    with np.load(fingerprint_path) as data:
        fingerprints = data["fingerprints"]
    meta = pd.read_csv(meta_path)
    if descriptors.shape != (318, 217) or fingerprints.shape != (318, 2048):
        raise ValueError(
            f"Unexpected feature shapes: {descriptors.shape}, {fingerprints.shape}"
        )
    if len(meta) != len(descriptors):
        raise ValueError("Feature metadata and descriptor row counts do not match")
    features = np.concatenate(
        [descriptors.to_numpy(dtype=np.float32), fingerprints.astype(np.float32)],
        axis=1,
    )
    target = meta["isactive"].to_numpy(dtype=int)
    if features.shape != (318, 2265) or set(np.unique(target)) != {0, 1}:
        raise ValueError("Unexpected feature shape or classifier labels")
    if np.isinf(features).any():
        raise ValueError("Features contain infinite values")
    return features, target, meta


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
    features, target, _ = load_features()
    folds = load_folds(len(target))
    oof_probabilities = np.full(len(target), np.nan, dtype=float)
    fold_metrics = []
    best_iterations = []

    for fold in folds:
        fold_id = fold["fold"]
        train_indices = np.asarray(fold["train_indices"], dtype=int)
        validation_indices = np.asarray(fold["validation_indices"], dtype=int)
        model = lgb.LGBMClassifier(
            **{
                key: value
                for key, value in PARAMS.items()
                if key not in ("early_stopping_rounds", "eval_metric")
            }
        )
        model.fit(
            features[train_indices],
            target[train_indices],
            eval_set=[(features[validation_indices], target[validation_indices])],
            eval_metric=PARAMS["eval_metric"],
            callbacks=[
                lgb.early_stopping(PARAMS["early_stopping_rounds"], verbose=False)
            ],
        )
        oof_probabilities[validation_indices] = model.predict_proba(
            features[validation_indices],
            num_iteration=model.best_iteration_,
        )[:, 1]
        best_iteration = int(model.best_iteration_ or PARAMS["n_estimators"])
        best_iterations.append(best_iteration)
        fold_auc = float(
            roc_auc_score(target[validation_indices], oof_probabilities[validation_indices])
        )
        fold_metrics.append(fold_auc)
        print(f"Fold {fold_id}: best_iteration={best_iteration}, ROC-AUC={fold_auc:.6f}")

    if not np.isfinite(oof_probabilities).all():
        raise RuntimeError("OOF probability array contains missing or non-finite values")
    predicted_labels = (oof_probabilities >= 0.5).astype(int)
    roc_auc = float(roc_auc_score(target, oof_probabilities))
    pr_auc = float(average_precision_score(target, oof_probabilities))
    accuracy = float(accuracy_score(target, predicted_labels))
    confusion = confusion_matrix(target, predicted_labels, labels=[0, 1]).tolist()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(RESULTS_DIR / "exp02oof_probs.npy", oof_probabilities)
    result = {
        "model": "lgbmclassifier",
        "params": PARAMS,
        "cvmetric": roc_auc,
        "metricname": "ROCAUC",
        "pr_auc": pr_auc,
        "accuracy": accuracy,
        "confusion_matrix": confusion,
        "foldmetrics": fold_metrics,
        "bestiterations": best_iterations,
        "notes": "C stage 1; out-of-fold probabilities on saved folds.",
    }
    result_path = RESULTS_DIR / "exp02classifier.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"ROC-AUC: {roc_auc:.6f}")
    print(f"PR-AUC: {pr_auc:.6f}")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Confusion matrix [[tn, fp], [fn, tp]]: {confusion}")
    print(f"Best iterations: {best_iterations}")
    print(f"Saved result: {result_path}")


if __name__ == "__main__":
    main()
