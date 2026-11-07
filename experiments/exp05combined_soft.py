import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"


def main():
    probabilities = np.load(RESULTS_DIR / "exp02oof_probs.npy")
    predictions_log = np.load(RESULTS_DIR / "exp03oof_log10.npy")
    target = pd.read_csv(ROOT / "train_clean.csv")["SI VACV"].to_numpy(dtype=float)

    if probabilities.shape != (318,) or predictions_log.shape != (318,):
        raise ValueError(
            f"Expected two OOF arrays of shape (318,), got "
            f"{probabilities.shape} and {predictions_log.shape}"
        )
    if not np.isfinite(probabilities).all() or not np.isfinite(predictions_log).all():
        raise ValueError("OOF arrays contain missing or non-finite values")
    if not np.isfinite(target).all():
        raise ValueError("SI VACV target contains missing or non-finite values")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("Classifier probabilities must be in [0, 1]")

    predicted_active_si = np.power(10.0, np.clip(predictions_log, -2.0, 5.0))
    prediction = (1.0 - probabilities) * 10000.0 + probabilities * predicted_active_si
    mse_raw = float(mean_squared_error(target, prediction))
    result = {
        "model": "combinedsoft",
        "cvmetric": mse_raw,
        "metricname": "MSE_raw",
        "notes": "soft blend: (1-p)10000 + p10r",
    }
    result_path = RESULTS_DIR / "exp05combined_soft.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"MSE_raw: {mse_raw:.6g}")
    print(f"Saved result: {result_path}")


if __name__ == "__main__":
    main()
