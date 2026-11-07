import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"
SEED = 42
N_SPLITS = 5


def main():
    train_path = ROOT / "train_clean.csv"
    if not train_path.is_file():
        raise FileNotFoundError(f"Expected {train_path}")

    train = pd.read_csv(train_path)
    target = train["SI VACV"].to_numpy(dtype=float)
    if not np.isfinite(target).all():
        raise ValueError("SI VACV contains missing or non-finite target values")

    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    predictions = np.full(len(train), np.nan, dtype=float)
    fold_ids = np.full(len(train), -1, dtype=np.int8)
    folds = []
    rng = np.random.default_rng(SEED)
    fold_metrics = []

    for fold_id, (train_indices, validation_indices) in enumerate(splitter.split(train)):
        predictions[validation_indices] = 10 ** rng.uniform(
            low=-1.0,
            high=4.5,
            size=len(validation_indices),
        )
        fold_ids[validation_indices] = fold_id
        folds.append(
            {
                "fold": fold_id,
                "train_indices": train_indices.tolist(),
                "validation_indices": validation_indices.tolist(),
            }
        )
        fold_metrics.append(
            float(mean_squared_error(target[validation_indices], predictions[validation_indices]))
        )

    if np.isnan(predictions).any() or (fold_ids < 0).any():
        raise RuntimeError("Cross-validation did not produce one prediction per row")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "folds.json").write_text(
        json.dumps({"n_splits": N_SPLITS, "random_state": SEED, "folds": folds}, indent=2),
        encoding="utf-8",
    )
    result = {
        "model": "Log-uniform random baseline",
        "params": {
            "distribution": "10**Uniform(-1.0, 4.5)",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": SEED,
            "prediction_seed": SEED,
        },
        "cvmetric": float(mean_squared_error(target, predictions)),
        "foldmetrics": fold_metrics,
        "notes": "Out-of-fold predictions scored by MSE on raw SI VACV.",
    }
    result_path = RESULTS_DIR / "exp00random.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"OOF MSE (raw SI): {result['cvmetric']:.6g}")
    print(f"Fold MSEs: {[round(value, 6) for value in fold_metrics]}")
    print(f"Saved result: {result_path}")


if __name__ == "__main__":
    main()
