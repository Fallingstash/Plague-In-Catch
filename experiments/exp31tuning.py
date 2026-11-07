import json
import random
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.upgrade_common import (
    RESULTS_DIR,
    load_folds,
    load_lgbm_features,
    load_training_data,
)


TRIALS = 30
SEED = 31


def candidate_params(rng, trial):
    learning_rate = float(10 ** rng.uniform(np.log10(0.01), np.log10(0.1)))
    max_depth = rng.choice([3, 4, 5, 6, 7, 8, -1])
    num_leaves = int(rng.choice([7, 15, 23, 31, 47, 63]))
    return {
        "objective": "regression",
        "n_estimators": int(rng.choice([300, 600, 1000, 1500])),
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "max_depth": int(max_depth),
        "min_child_samples": int(rng.choice([5, 10, 15, 20, 30])),
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": SEED + trial,
        "verbose": -1,
        "n_jobs": -1,
    }


def evaluate(params, features, target, folds, keep_oof=False):
    oof = np.full(len(target), np.nan, dtype=float)
    best_iterations = []
    for fold in folds:
        train_indices = np.asarray(fold["train_indices"], dtype=int)
        valid_indices = np.asarray(fold["validation_indices"], dtype=int)
        model = lgb.LGBMRegressor(**params)
        model.fit(
            features[train_indices],
            target[train_indices],
            eval_set=[(features[valid_indices], target[valid_indices])],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        best_iteration = int(model.best_iteration_ or params["n_estimators"])
        best_iterations.append(best_iteration)
        oof[valid_indices] = model.predict(
            features[valid_indices], num_iteration=best_iteration
        )
    if not np.isfinite(oof).all():
        raise RuntimeError("Tuning candidate produced incomplete OOF predictions")
    return float(mean_squared_error(target, oof)), oof, best_iterations


def main():
    _, target = load_training_data()
    folds = load_folds(len(target))
    features, _ = load_lgbm_features()
    rng = random.Random(SEED)
    baseline_params = {
        "objective": "regression",
        "n_estimators": 1000,
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
    baseline_mse, _, baseline_iterations = evaluate(
        baseline_params, features, target, folds
    )
    candidates = [baseline_params]
    for trial in range(1, TRIALS + 1):
        candidates.append(candidate_params(rng, trial))

    trial_reports = []
    best_mse = baseline_mse
    best_params = baseline_params
    best_oof = None
    best_iterations = baseline_iterations
    for trial, params in enumerate(candidates):
        mse, oof, iterations = evaluate(
            params, features, target, folds, keep_oof=True
        )
        trial_reports.append(
            {
                "trial": trial,
                "mse_raw": mse,
                "params": params,
                "best_iterations": iterations,
            }
        )
        if mse < best_mse:
            best_mse = mse
            best_params = params
            best_oof = oof.copy()
            best_iterations = iterations
        print(
            f"trial={trial:02d}/{TRIALS} MSE_raw={mse:.6g} "
            f"best={best_mse:.6g} iterations={iterations}",
            flush=True,
        )

    if best_oof is None:
        best_mse, best_oof, best_iterations = evaluate(
            baseline_params, features, target, folds, keep_oof=True
        )
    improved = best_mse < baseline_mse
    np.save(RESULTS_DIR / "exp31oof_raw_si.npy", best_oof)
    report = {
        "model": "random_search_lgbm_raw_SI",
        "cvmetric": best_mse,
        "metricname": "MSE_raw",
        "baseline_mse_raw": baseline_mse,
        "best_mse_raw": best_mse,
        "improved_over_baseline": improved,
        "best_params": best_params,
        "best_iterations": best_iterations,
        "trials_requested": TRIALS,
        "candidate_runs_including_baseline": len(trial_reports),
        "random_seed": SEED,
        "foldmetrics": [
            float(
                mean_squared_error(
                    target[np.asarray(fold["validation_indices"], dtype=int)],
                    best_oof[np.asarray(fold["validation_indices"], dtype=int)],
                )
            )
            for fold in folds
        ],
        "trials": trial_reports,
        "notes": "30 random hyperparameter trials evaluated on the saved five folds; baseline included as trial 0. The best OOF is the full row-wise CV prediction from the best candidate.",
    }
    report_path = RESULTS_DIR / "exp31tuning.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Best params: {json.dumps(best_params, indent=2)}")
    print(f"Baseline MSE={baseline_mse:.6g}; best MSE={best_mse:.6g}; improved={improved}")
    print(f"Saved result: {report_path}")


if __name__ == "__main__":
    main()
