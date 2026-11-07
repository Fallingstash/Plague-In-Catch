import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"


def load_prediction(*paths):
    for path in paths:
        if path.is_file():
            return np.load(path)
    raise FileNotFoundError("None of the expected OOF files exists: " + ", ".join(map(str, paths)))


def main():
    probabilities = load_prediction(
        RESULTS_DIR / "exp02oofprobs.npy",
        RESULTS_DIR / "exp02oof_probs.npy",
    )
    raw_regression = load_prediction(
        RESULTS_DIR / "exp08oofrawsi.npy",
        RESULTS_DIR / "exp08oof_raw_si.npy",
    )
    train_path = ROOT / "train_clean.csv"
    folds_path = RESULTS_DIR / "folds.json"
    for path in (train_path, folds_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required calibration input is missing: {path}")
    train = pd.read_csv(train_path)
    folds = json.loads(folds_path.read_text(encoding="utf-8"))["folds"]
    if probabilities.shape != (318,) or raw_regression.shape != (318,) or len(train) != 318:
        raise ValueError("Expected 318 rows in both OOF arrays and train_clean.csv")
    if not np.isfinite(probabilities).all() or not np.isfinite(raw_regression).all():
        raise ValueError("OOF inputs contain missing or non-finite values")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Classifier probabilities must be in [0, 1]")

    labels = train["is_active"].to_numpy(dtype=int)
    target_raw = pd.to_numeric(train["SI VACV"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(target_raw).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Raw target or activity labels are invalid")

    expected = set(range(len(train)))
    validation_indices = []
    for fold in folds:
        fold_train = set(fold["train_indices"])
        fold_valid = set(fold["validation_indices"])
        if fold_train & fold_valid or fold_train | fold_valid != expected:
            raise ValueError(f"Fold {fold.get('fold')} does not partition all rows")
        validation_indices.extend(fold["validation_indices"])
    if sorted(validation_indices) != list(range(len(train))):
        raise ValueError("Saved folds must validate every row exactly once")

    calibrated_probability = np.full(len(train), np.nan, dtype=float)
    calibrated_regression = np.full(len(train), np.nan, dtype=float)
    fold_mses = []
    classifier_calibrators = []
    regression_calibrators = []

    for fold in folds:
        train_indices = np.asarray(fold["train_indices"], dtype=int)
        valid_indices = np.asarray(fold["validation_indices"], dtype=int)
        classifier = LogisticRegression(random_state=42, max_iter=1000)
        classifier.fit(probabilities[train_indices, None], labels[train_indices])
        calibrated_probability[valid_indices] = classifier.predict_proba(
            probabilities[valid_indices, None]
        )[:, 1]

        active_train_indices = train_indices[labels[train_indices] == 1]
        if len(active_train_indices) < 2:
            raise ValueError(f"Fold {fold['fold']} has too few active samples to calibrate regression")
        reg_calibrator = LinearRegression()
        reg_calibrator.fit(
            raw_regression[active_train_indices, None],
            target_raw[active_train_indices],
        )
        calibrated_regression[valid_indices] = reg_calibrator.predict(
            raw_regression[valid_indices, None]
        )
        fold_prediction = (
            (1.0 - calibrated_probability[valid_indices]) * 10000.0
            + calibrated_probability[valid_indices] * calibrated_regression[valid_indices]
        )
        fold_mses.append(
            float(mean_squared_error(target_raw[valid_indices], fold_prediction))
        )
        classifier_calibrators.append(
            {
                "fold": int(fold["fold"]),
                "intercept": float(classifier.intercept_[0]),
                "coefficient": float(classifier.coef_[0, 0]),
            }
        )
        regression_calibrators.append(
            {
                "fold": int(fold["fold"]),
                "intercept": float(reg_calibrator.intercept_),
                "coefficient": float(reg_calibrator.coef_[0]),
            }
        )
        print(
            f"Fold {fold['fold']}: calibrated blend MSE_raw="
            f"{fold_mses[-1]:.6g}; Platt coefficient={classifier.coef_[0, 0]:.6g}; "
            f"linear slope={reg_calibrator.coef_[0]:.6g}"
        )

    if not np.isfinite(calibrated_probability).all() or not np.isfinite(
        calibrated_regression
    ).all():
        raise RuntimeError("Calibrated OOF arrays contain missing/non-finite values")
    soft_prediction = (
        (1.0 - calibrated_probability) * 10000.0
        + calibrated_probability * calibrated_regression
    )
    calibrated_mse = float(mean_squared_error(target_raw, soft_prediction))
    uncalibrated_prediction = (1.0 - probabilities) * 10000.0 + probabilities * raw_regression
    uncalibrated_mse = float(mean_squared_error(target_raw, uncalibrated_prediction))
    exp09_path = RESULTS_DIR / "exp09combined_raw.json"
    exp09_mse = (
        float(json.loads(exp09_path.read_text(encoding="utf-8"))["mse_soft"])
        if exp09_path.is_file()
        else uncalibrated_mse
    )

    np.save(RESULTS_DIR / "exp10oof_calibrated_probs.npy", calibrated_probability)
    np.save(RESULTS_DIR / "exp10oof_calibrated_raw_si.npy", calibrated_regression)
    np.save(RESULTS_DIR / "exp10oof_calibrated_blend.npy", soft_prediction)
    result = {
        "model": "platt_and_linear_calibrated_raw_blend",
        "cvmetric": calibrated_mse,
        "metricname": "MSE_raw",
        "mse_calibrated": calibrated_mse,
        "mse_uncalibrated_recomputed": uncalibrated_mse,
        "mse_exp09_soft": exp09_mse,
        "improvement_vs_uncalibrated": exp09_mse - calibrated_mse,
        "foldmetrics": fold_mses,
        "platt_calibrators": classifier_calibrators,
        "linear_calibrators": regression_calibrators,
        "notes": "Fold-local logistic calibration of classifier OOF probabilities and fold-local linear calibration of raw-SI OOF regression; predictions for each validation fold are calibrated using only other-fold OOF predictions.",
    }
    result_path = RESULTS_DIR / "exp10calibratedrawblend.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"MSE calibrated soft blend: {calibrated_mse:.6g}")
    print(f"MSE uncalibrated soft blend: {uncalibrated_mse:.6g}")
    print(f"MSE exp09 soft: {exp09_mse:.6g}")
    print(f"Calibration improvement (positive is better): {exp09_mse - calibrated_mse:.6g}")
    print(f"Saved result: {result_path}")


if __name__ == "__main__":
    main()
