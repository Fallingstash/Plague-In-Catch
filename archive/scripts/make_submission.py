import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "experiments" / "results"
CATEGORY_COLUMNS = ["Culture", "strain"]
CLASSIFIER_PARAMS = {
    "n_estimators": 200,
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
    "n_estimators": 200,
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


def load_result(name):
    path = RESULTS_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Required experiment result is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_data():
    train = pd.read_csv(ROOT / "train_clean.csv")
    test = pd.read_csv(ROOT / "test_clean.csv")
    train_desc = pd.read_csv(ROOT / "featurestraindesc.csv")
    test_desc = pd.read_csv(ROOT / "featurestestdesc.csv")
    with np.load(ROOT / "featurestrainfp.npz") as data:
        train_fp = data["fingerprints"]
    with np.load(ROOT / "featurestestfp.npz") as data:
        test_fp = data["fingerprints"]

    if len(train) != 318 or len(test) != 597:
        raise ValueError(f"Unexpected dataset sizes: train={len(train)}, test={len(test)}")
    if train_desc.shape != (318, 217) or test_desc.shape != (597, 217):
        raise ValueError(f"Unexpected descriptor shapes: {train_desc.shape}, {test_desc.shape}")
    if train_fp.shape != (318, 2048) or test_fp.shape != (597, 2048):
        raise ValueError(f"Unexpected fingerprint shapes: {train_fp.shape}, {test_fp.shape}")
    if list(train_desc.columns) != list(test_desc.columns):
        raise ValueError("Train and test descriptor columns are not aligned")

    x_train = np.concatenate(
        [train_desc.to_numpy(dtype=np.float32), train_fp.astype(np.float32)], axis=1
    )
    x_test = np.concatenate(
        [test_desc.to_numpy(dtype=np.float32), test_fp.astype(np.float32)], axis=1
    )
    return train, test, x_train, x_test


def main():
    train, test, base_train, base_test = load_data()
    exp01 = load_result("exp01baseline_b.json")["cvmetric"]
    exp04 = load_result("exp04combined_hard.json")["cvmetric"]
    exp05 = load_result("exp05combined_soft.json")["cvmetric"]
    exp06_path = RESULTS_DIR / "exp06combinedwithstrain.json"
    exp06 = load_result(exp06_path.name)["cvmetric"] if exp06_path.is_file() else None
    candidate_scores = {
        "exp01": float(exp01),
        "exp04": float(exp04),
        "exp05": float(exp05),
    }
    if exp06 is not None and float(exp06) < float(exp05):
        selected_model = "exp06"
        USE_STRAIN_LOCAL = True
        expected_cv_mse = float(exp06)
    else:
        selected_model = "exp05"
        USE_STRAIN_LOCAL = False
        expected_cv_mse = float(exp05)
    if selected_model == "exp06":
        candidate_scores["exp06"] = float(exp06)
    if selected_model not in ("exp05", "exp06"):
        raise RuntimeError("The chosen final model must be a soft-blend strategy")

    if USE_STRAIN_LOCAL:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        train_categories = train[CATEGORY_COLUMNS]
        test_categories = test[CATEGORY_COLUMNS]
        train_encoded = encoder.fit_transform(train_categories).astype(np.float32)
        test_encoded = encoder.transform(test_categories).astype(np.float32)
        x_train = np.concatenate([base_train, train_encoded], axis=1)
        x_test = np.concatenate([base_test, test_encoded], axis=1)
    else:
        x_train, x_test = base_train, base_test

    labels = train["is_active"].to_numpy(dtype=int)
    target_log = train["log10_SI"].to_numpy(dtype=float)
    active = labels == 1
    if active.sum() != 237:
        raise ValueError(f"Expected 237 active rows, found {active.sum()}")

    classifier = lgb.LGBMClassifier(**CLASSIFIER_PARAMS)
    classifier.fit(x_train, labels)
    regressor = lgb.LGBMRegressor(**REGRESSOR_PARAMS)
    regressor.fit(x_train[active], target_log[active])

    probability_test = classifier.predict_proba(x_test)[:, 1]
    regression_test = regressor.predict(x_test)
    predicted_active_si = np.power(10.0, np.clip(regression_test, -2.0, 5.0))
    prediction = (1.0 - probability_test) * 10000.0 + probability_test * predicted_active_si

    submission = pd.DataFrame(
        {"index": test["index"].to_numpy(), "SI VACV": prediction}
    ).sort_values("index", kind="stable").reset_index(drop=True)
    if len(submission) != 597:
        raise ValueError(f"Submission has {len(submission)} rows, expected 597")
    if submission["SI VACV"].isna().any() or not np.isfinite(submission["SI VACV"]).all():
        raise ValueError("Submission contains NaN or non-finite predictions")
    if not (submission["SI VACV"] > 0).all():
        raise ValueError("All SI VACV predictions must be positive")
    if not np.array_equal(submission["index"].to_numpy(), np.arange(597)):
        raise ValueError("Submission index must be sorted from 0 to 596")

    output_path = ROOT / "submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"Selected model: {selected_model}")
    print(f"CV score candidates (MSE_raw): {candidate_scores}")
    print(f"Submission head:\n{submission.head().to_string(index=False)}")
    print(
        "Prediction stats: "
        f"min={submission['SI VACV'].min():.6g}, "
        f"max={submission['SI VACV'].max():.6g}, "
        f"median={submission['SI VACV'].median():.6g}, "
        f"mean={submission['SI VACV'].mean():.6g}"
    )
    print(f"Submission size: {output_path.stat().st_size} bytes")
    print(f"Expected Kaggle MSE proxy (selected OOF CV): {expected_cv_mse:.6g}")
    print("Submission validation: 597 rows, finite positive SI, index 0..596")
    print("Ready for Kaggle upload: yes (not uploaded)")


if __name__ == "__main__":
    main()
