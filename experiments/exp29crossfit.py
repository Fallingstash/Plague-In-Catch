import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.upgrade_common import RESULTS_DIR, load_folds, load_oof, load_training_data


def optimize_weights(predictions, target):
    matrix = np.column_stack(predictions)
    scale = max(float(np.max(np.abs(target))), 1.0)
    normalized_matrix = matrix / scale
    normalized_target = target / scale

    def objective(weights):
        residual = normalized_target - normalized_matrix @ weights
        return float(np.mean(residual**2))

    def gradient(weights):
        residual = normalized_target - normalized_matrix @ weights
        return -2.0 * normalized_matrix.T @ residual / len(target)

    starts = [
        np.full(matrix.shape[1], 1.0 / matrix.shape[1]),
        np.asarray([0.5688, 0.4312, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
    ]
    valid_results = []
    for start in starts:
        result = minimize(
            objective,
            x0=start,
            jac=gradient,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * matrix.shape[1],
            constraints=[{"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}],
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        weights = np.asarray(result.x, dtype=float)
        feasible = (
            np.isfinite(weights).all()
            and (weights >= -1e-7).all()
            and (weights <= 1.0 + 1e-7).all()
            and abs(float(weights.sum()) - 1.0) <= 1e-6
        )
        if feasible:
            weights = np.clip(weights, 0.0, 1.0)
            weights /= weights.sum()
            valid_results.append((objective(weights), weights))
    if not valid_results:
        raise RuntimeError("Simplex weight optimization failed to find a feasible solution")
    valid_results.sort(key=lambda item: item[0])
    return valid_results[0][1]


def main():
    _, target = load_training_data()
    folds = load_folds(len(target))
    gnn = load_oof("exp14oofrawsi.npy")
    lgbm = load_oof("exp08oof_raw_si.npy")
    exp10 = load_oof("exp10oof_calibrated_blend.npy")
    matrix = np.column_stack([gnn, lgbm, exp10])
    fixed_two_model = np.asarray([0.5688, 0.4312, 0.0])
    equal_two_model = np.asarray([0.5, 0.5, 0.0])
    optimized_oof = np.full(len(target), np.nan, dtype=float)
    fixed_prediction = matrix @ fixed_two_model
    equal_prediction = matrix @ equal_two_model
    fold_weights = []
    fold_metrics = []

    for fold in folds:
        train_indices = np.asarray(fold["train_indices"], dtype=int)
        valid_indices = np.asarray(fold["validation_indices"], dtype=int)
        weights = optimize_weights(matrix[train_indices].T, target[train_indices])
        optimized_oof[valid_indices] = matrix[valid_indices] @ weights
        mse = float(mean_squared_error(target[valid_indices], optimized_oof[valid_indices]))
        fold_weights.append(
            {"fold": int(fold["fold"]), "weights": weights.tolist(), "mse_raw": mse}
        )
        fold_metrics.append(mse)
        print(
            f"fold={fold['fold']} weights={weights.round(5).tolist()} "
            f"MSE_raw={mse:.6g}",
            flush=True,
        )

    full_weights = optimize_weights(matrix.T, target)
    candidates = {
        "equal_50_50_gnn_lgbm": equal_prediction,
        "fixed_0.5688_0.4312_gnn_lgbm": fixed_prediction,
        "cross_fitted_optimized_3_model": optimized_oof,
    }
    comparison = {
        name: float(mean_squared_error(target, prediction))
        for name, prediction in candidates.items()
    }
    np.save(RESULTS_DIR / "exp29oof_crossfit_blend.npy", optimized_oof)
    report = {
        "model": "cross_fitted_simplex_ensemble",
        "feature_models": ["GIN", "LGBM_raw", "exp10_calibrated_blend"],
        "constraints": "nonnegative weights summing to one",
        "cvmetric": comparison["cross_fitted_optimized_3_model"],
        "metricname": "MSE_raw",
        "comparison_mse": comparison,
        "full_data_optimal_weights": full_weights.tolist(),
        "fold_weights": fold_weights,
        "foldmetrics": fold_metrics,
        "notes": (
            "For each validation fold, weights are optimized using only the other folds' "
            "OOF predictions and labels. Equal/fixed baselines use GIN and raw LightGBM."
        ),
    }
    report_path = RESULTS_DIR / "exp29crossfit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Comparison MSE: {json.dumps(comparison, indent=2)}")
    print(f"Saved result: {report_path}")


if __name__ == "__main__":
    main()
