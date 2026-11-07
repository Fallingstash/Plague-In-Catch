import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"


def load_first_existing(*paths):
    for path in paths:
        if path.is_file():
            return np.load(path)
    raise FileNotFoundError("None of the expected OOF files exists: " + ", ".join(map(str, paths)))


def main():
    probabilities = load_first_existing(
        RESULTS_DIR / "exp02oofprobs.npy",
        RESULTS_DIR / "exp02oof_probs.npy",
    )
    raw_predictions = load_first_existing(
        RESULTS_DIR / "exp08oofrawsi.npy",
        RESULTS_DIR / "exp08oof_raw_si.npy",
    )
    target_path = ROOT / "train_clean.csv"
    if not target_path.is_file():
        raise FileNotFoundError(f"Original training data is missing: {target_path}")
    target = pd.read_csv(target_path)["SI VACV"].to_numpy(dtype=float)

    expected_shape = (318,)
    if probabilities.shape != expected_shape or raw_predictions.shape != expected_shape:
        raise ValueError(
            f"Expected probability and raw-prediction arrays with shape {expected_shape}; "
            f"got {probabilities.shape} and {raw_predictions.shape}"
        )
    if not np.isfinite(probabilities).all() or not np.isfinite(raw_predictions).all():
        raise ValueError("OOF inputs contain missing or non-finite values")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Classifier probabilities must be in [0, 1]")
    if not np.isfinite(target).all():
        raise ValueError("SI VACV contains missing or non-finite values")

    soft_predictions = (1.0 - probabilities) * 10000.0 + probabilities * raw_predictions
    hard_predictions = np.where(probabilities >= 0.5, raw_predictions, 10000.0)
    mse_soft = float(mean_squared_error(target, soft_predictions))
    mse_hard = float(mean_squared_error(target, hard_predictions))
    previous_mse = float(
        json.loads(
            (RESULTS_DIR / "exp05combined_soft.json").read_text(encoding="utf-8")
        )["cvmetric"]
    )

    np.save(RESULTS_DIR / "exp09oof_combined.npy", soft_predictions)
    result = {
        "model": "combined_raw_target",
        "cvmetric": mse_soft,
        "metricname": "MSE_raw_soft",
        "mse_hard": mse_hard,
        "mse_soft": mse_soft,
        "mse_exp05_soft_log_target": previous_mse,
        "improvement_vs_exp05": previous_mse - mse_soft,
        "notes": "Combines existing OOF classifier probabilities and raw-SI OOF regression predictions; no model retraining.",
    }
    result_path = RESULTS_DIR / "exp09combined_raw.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"MSE hard (raw target): {mse_hard:.6g}")
    print(f"MSE soft (raw target): {mse_soft:.6g}")
    print(f"MSE exp05 soft (log target): {previous_mse:.6g}")
    print(f"Improvement vs exp05: {previous_mse - mse_soft:.6g}")
    print(f"Saved result: {result_path}")


if __name__ == "__main__":
    main()
