import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.upgrade_common import RESULTS_DIR, load_folds, load_oof, load_training_data


def selected_lgbm_oof():
    tuning_path = RESULTS_DIR / "exp31tuning.json"
    tuned_path = RESULTS_DIR / "exp31oof_raw_si.npy"
    if tuning_path.is_file() and tuned_path.is_file():
        report = json.loads(tuning_path.read_text(encoding="utf-8"))
        base_mse = float(report["baseline_mse_raw"])
        tuned_mse = float(report["best_mse_raw"])
        if tuned_mse < base_mse:
            print(f"Using tuned LGBM OOF (MSE {tuned_mse:.6g} < {base_mse:.6g})")
            return np.load(tuned_path), True
    return load_oof("exp08oof_raw_si.npy"), False


def calibrated_blend_from_lgbm(raw_lgbm, folds, probabilities, target, labels):
    calibration_probability = load_oof("exp10oof_calibrated_probs.npy")
    calibrated_raw = np.full(len(target), np.nan, dtype=float)
    for fold in folds:
        train_indices = np.asarray(fold["train_indices"], dtype=int)
        valid_indices = np.asarray(fold["validation_indices"], dtype=int)
        active_train = train_indices[labels[train_indices] == 1]
        calibrator = LinearRegression().fit(
            raw_lgbm[active_train, None], target[active_train]
        )
        calibrated_raw[valid_indices] = calibrator.predict(raw_lgbm[valid_indices, None])
    if not np.isfinite(calibrated_raw).all():
        raise RuntimeError("Cross-fitted regression calibration produced invalid predictions")
    return (1.0 - calibration_probability) * 10000.0 + calibration_probability * calibrated_raw


def main():
    train, target = load_training_data()
    folds = load_folds(len(target))
    gnn = load_oof("exp14oofrawsi.npy")
    lgbm, tuned = selected_lgbm_oof()
    probabilities = load_oof("exp02oof_probs.npy")
    labels = train["is_active"].to_numpy(dtype=int)
    if tuned:
        exp10 = calibrated_blend_from_lgbm(
            lgbm, folds, probabilities, target, labels
        )
    else:
        exp10 = load_oof("exp10oof_calibrated_blend.npy")

    features = np.column_stack([gnn, lgbm, exp10, probabilities])
    if features.shape != (318, 4) or not np.isfinite(features).all():
        raise ValueError("Meta feature matrix must be finite with shape (318, 4)")

    oof = np.full(len(target), np.nan, dtype=float)
    fold_mses = []
    fold_coefficients = []
    for fold in folds:
        train_indices = np.asarray(fold["train_indices"], dtype=int)
        valid_indices = np.asarray(fold["validation_indices"], dtype=int)
        meta = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        meta.fit(features[train_indices], target[train_indices])
        oof[valid_indices] = meta.predict(features[valid_indices])
        mse = float(mean_squared_error(target[valid_indices], oof[valid_indices]))
        fold_mses.append(mse)
        fold_coefficients.append(
            {
                "fold": int(fold["fold"]),
                "standardized_coefficients": meta.named_steps["ridge"].coef_.tolist(),
                "intercept": float(meta.named_steps["ridge"].intercept_),
                "mse_raw": mse,
            }
        )
        print(f"fold={fold['fold']} stacking MSE_raw={mse:.6g}", flush=True)

    if not np.isfinite(oof).all():
        raise RuntimeError("Stacking OOF predictions are incomplete or non-finite")
    mse_stacking = float(mean_squared_error(target, oof))
    full_meta = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    full_meta.fit(features, target)
    joblib.dump(full_meta, RESULTS_DIR / "exp30_meta_model.joblib")
    np.save(RESULTS_DIR / "exp30oof_stacking.npy", oof)

    crossfit_path = RESULTS_DIR / "exp29crossfit.json"
    crossfit_mse = (
        float(json.loads(crossfit_path.read_text(encoding="utf-8"))["cvmetric"])
        if crossfit_path.is_file()
        else None
    )
    comparison = {
        "stacking": mse_stacking,
        "cross_fitted_optimized_ensemble": crossfit_mse,
        "weighted_gnn_lgbm_reference": float(
            mean_squared_error(target, 0.568825363320826 * gnn + 0.43117463667917394 * load_oof("exp08oof_raw_si.npy"))
        ),
    }
    report = {
        "model": "cross_fitted_ridge_stacking",
        "features": ["GIN_raw_SI", "LGBM_raw_SI", "exp10_calibrated_blend", "P_active"],
        "feature_shape": list(features.shape),
        "meta_model": {"type": "StandardScaler + Ridge", "alpha": 1.0},
        "selected_lgbm_oof": "exp31 tuned" if tuned else "exp08 baseline",
        "cvmetric": mse_stacking,
        "metricname": "MSE_raw",
        "comparison_mse": comparison,
        "foldmetrics": fold_mses,
        "fold_coefficients": fold_coefficients,
        "notes": "Meta-model trained only on non-validation rows of the saved OOF feature matrix in each fold.",
    }
    report_path = RESULTS_DIR / "exp30stacking.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"MSE comparison: {json.dumps(comparison, indent=2)}")
    print(f"Saved meta-model: {RESULTS_DIR / 'exp30_meta_model.joblib'}")
    print(f"Saved result: {report_path}")


if __name__ == "__main__":
    main()
