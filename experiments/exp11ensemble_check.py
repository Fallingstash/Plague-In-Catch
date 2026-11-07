import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"


def load_prediction(*paths):
    for path in paths:
        if path.is_file():
            return np.load(path)
    raise FileNotFoundError("None of the expected prediction files exists: " + ", ".join(map(str, paths)))


def main():
    gnn_prediction = load_prediction(RESULTS_DIR / "exp14oofrawsi.npy")
    lgbm_prediction = load_prediction(
        RESULTS_DIR / "exp08oofrawsi.npy",
        RESULTS_DIR / "exp08oof_raw_si.npy",
    )
    target = pd.read_csv(ROOT / "train_clean.csv")["SI VACV"].to_numpy(dtype=float)
    if gnn_prediction.shape != (318,) or lgbm_prediction.shape != (318,):
        raise ValueError(
            f"Expected both 318-row OOF arrays, got "
            f"{gnn_prediction.shape} and {lgbm_prediction.shape}"
        )
    if not np.isfinite(gnn_prediction).all() or not np.isfinite(lgbm_prediction).all():
        raise ValueError("Base model predictions contain missing/non-finite values")

    gnn_errors = target - gnn_prediction
    lgbm_errors = target - lgbm_prediction
    error_correlation = float(np.corrcoef(gnn_errors, lgbm_errors)[0, 1])
    ensemble_prediction = 0.5 * gnn_prediction + 0.5 * lgbm_prediction
    ensemble_mse = float(mean_squared_error(target, ensemble_prediction))
    gnn_mse = float(mean_squared_error(target, gnn_prediction))
    lgbm_mse = float(mean_squared_error(target, lgbm_prediction))

    calibrated_blend_path = RESULTS_DIR / "exp10oof_calibrated_blend.npy"
    if calibrated_blend_path.is_file():
        optional_blend = np.load(calibrated_blend_path)
    else:
        optional_blend = load_prediction(
            RESULTS_DIR / "exp05oof_blend.npy",
            RESULTS_DIR / "exp05oof_soft.npy",
        ) if (
            (RESULTS_DIR / "exp05oof_blend.npy").is_file()
            or (RESULTS_DIR / "exp05oof_soft.npy").is_file()
        ) else None
    optional_blend_mse = None
    if optional_blend is not None:
        if optional_blend.shape != (318,) or not np.isfinite(optional_blend).all():
            raise ValueError("Optional blend OOF array must contain 318 finite predictions")
        optional_blend_mse = float(mean_squared_error(target, optional_blend))

    np.save(RESULTS_DIR / "exp11oof_ensemble.npy", ensemble_prediction)
    result = {
        "model": "equal_weight_gnn_lgbm_raw_SI",
        "cvmetric": ensemble_mse,
        "metricname": "MSEraw",
        "error_correlation_pearson": error_correlation,
        "mse_gnn": gnn_mse,
        "mse_lgbm": lgbm_mse,
        "mse_ensemble": ensemble_mse,
        "mse_optional_blend": optional_blend_mse,
        "notes": "Equal-weight arithmetic mean of OOF raw-SI predictions.",
    }
    result_path = RESULTS_DIR / "exp11ensemble_check.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Pearson correlation of residuals (GIN, LightGBM): {error_correlation:.6f}")
    print(f"MSE GIN: {gnn_mse:.6g}")
    print(f"MSE LightGBM: {lgbm_mse:.6g}")
    print(f"MSE 50/50 ensemble: {ensemble_mse:.6g}")
    print(f"Optional blend MSE: {optional_blend_mse}")
    print(
        "Ensembling heuristic: "
        + ("correlation < 0.7; ensemble may help." if error_correlation < 0.7 else "correlation >= 0.7; errors remain strongly correlated.")
    )
    print(f"Saved result: {result_path}")


if __name__ == "__main__":
    main()
