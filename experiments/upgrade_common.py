import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"


def load_folds(row_count):
    path = RESULTS_DIR / "folds.json"
    if not path.is_file():
        raise FileNotFoundError(f"Original folds file is missing: {path}")
    folds = json.loads(path.read_text(encoding="utf-8"))["folds"]
    expected = set(range(row_count))
    validation = []
    for fold in folds:
        train_indices = set(fold["train_indices"])
        valid_indices = set(fold["validation_indices"])
        if train_indices & valid_indices or train_indices | valid_indices != expected:
            raise ValueError(f"Fold {fold.get('fold')} does not partition all rows")
        validation.extend(fold["validation_indices"])
    if sorted(validation) != list(range(row_count)):
        raise ValueError("Saved folds must validate every row exactly once")
    return folds


def load_training_data():
    train = pd.read_csv(ROOT / "train_clean.csv")
    if len(train) != 318:
        raise ValueError(f"Expected 318 training rows, got {len(train)}")
    target = pd.to_numeric(train["SI VACV"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError("Training SI values must be finite and positive")
    return train, target


def load_oof(path_name):
    path = RESULTS_DIR / path_name
    if not path.is_file():
        raise FileNotFoundError(f"Required OOF prediction is missing: {path}")
    prediction = np.load(path)
    if prediction.shape != (318,) or not np.isfinite(prediction).all():
        raise ValueError(f"Expected 318 finite predictions in {path}")
    return prediction


def load_lgbm_features():
    descriptors_train = pd.read_csv(ROOT / "featurestraindesc.csv")
    descriptors_test = pd.read_csv(ROOT / "featurestestdesc.csv")
    with np.load(ROOT / "featurestrainfp.npz") as data:
        fingerprints_train = data["fingerprints"]
    with np.load(ROOT / "featurestestfp.npz") as data:
        fingerprints_test = data["fingerprints"]
    features_train = np.concatenate(
        [
            descriptors_train.to_numpy(dtype=np.float32),
            fingerprints_train.astype(np.float32),
        ],
        axis=1,
    )
    features_test = np.concatenate(
        [
            descriptors_test.to_numpy(dtype=np.float32),
            fingerprints_test.astype(np.float32),
        ],
        axis=1,
    )
    if features_train.shape != (318, 2265) or features_test.shape != (597, 2265):
        raise ValueError(
            f"Unexpected feature shapes: {features_train.shape}, {features_test.shape}"
        )
    if np.isinf(features_train).any() or np.isinf(features_test).any():
        raise ValueError("Feature matrices contain infinite values")
    return features_train, features_test
