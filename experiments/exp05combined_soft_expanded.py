import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from expanded_common import RESULTS_DIR, load_expanded_data, write_result


def main():
    data = load_expanded_data()
    probability_path = RESULTS_DIR / "exp02oof_probs_expanded.npy"
    regression_path = RESULTS_DIR / "exp03oof_log10_expanded.npy"
    for path in (probability_path, regression_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required expanded OOF predictions are missing: {path}")
    probabilities = np.load(probability_path)[: len(data["train"])]
    predictions_log = np.load(regression_path)
    if probabilities.shape != (349,) or predictions_log.shape != (349,):
        raise ValueError(
            f"Expected expanded blend arrays of shape (349,), "
            f"got {probabilities.shape} and {predictions_log.shape}"
        )
    if not np.isfinite(probabilities).all() or not np.isfinite(predictions_log).all():
        raise ValueError("Expanded OOF predictions contain missing or non-finite values")

    pred_active_si = np.power(10.0, np.clip(predictions_log, -2.0, 5.0))
    blended = (1.0 - probabilities) * 10000.0 + probabilities * pred_active_si
    full_mse = float(mean_squared_error(data["target_raw"], blended))
    original_mse = float(mean_squared_error(data["target_raw"][:318], blended[:318]))
    previous_mse = float(
        json.loads(
            (RESULTS_DIR / "exp05combined_soft.json").read_text(encoding="utf-8")
        )["cvmetric"]
    )
    result = {
        "model": "combinedsoft_expanded",
        "cvmetric": full_mse,
        "metricname": "MSE_raw_349",
        "mse_original_318": original_mse,
        "mse_previous_exp05_318": previous_mse,
        "deltavsexp05_original_318": original_mse - previous_mse,
        "notes": "Soft OOF blend; 349-row MSE includes Figshare additions; original-row MSE is supplied for comparable assessment.",
    }
    path = write_result("exp05combined_soft_expanded.json", result)
    print(f"MSE_raw all 349: {full_mse:.6g}")
    print(f"MSE_raw original 318 rows: {original_mse:.6g}")
    print(f"Previous exp05 MSE 318 rows: {previous_mse:.6g}")
    print(f"Delta on original rows: {result['deltavsexp05_original_318']:.6g}")
    print(f"Saved result: {path}")


if __name__ == "__main__":
    main()
