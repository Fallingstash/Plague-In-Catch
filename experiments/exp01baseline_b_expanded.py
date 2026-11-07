import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

from expanded_common import (
    EARLY_STOPPING_ROUNDS,
    REGRESSOR_PARAMS,
    RESULTS_DIR,
    fold_records,
    load_expanded_data,
    load_fold_artifact,
    purge_canonical_duplicates,
    write_result,
)


def main():
    data = load_expanded_data()
    folds = fold_records(load_fold_artifact(), "training_349_folds")
    oof_log = np.full(len(data["target_log"]), np.nan, dtype=float)
    best_iterations = []
    fold_metrics = []
    purged_rows = []
    for fold in folds:
        train_indices = np.asarray(fold["train_indices"], dtype=int)
        valid_indices = np.asarray(fold["validation_indices"], dtype=int)
        train_indices, purged_count = purge_canonical_duplicates(
            train_indices, valid_indices, data["canonical_base"]
        )
        purged_rows.append(purged_count)
        model = lgb.LGBMRegressor(**REGRESSOR_PARAMS)
        model.fit(
            data["x_base"][train_indices],
            data["target_log"][train_indices],
            eval_set=[
                (data["x_base"][valid_indices], data["target_log"][valid_indices])
            ],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        oof_log[valid_indices] = model.predict(
            data["x_base"][valid_indices], num_iteration=model.best_iteration_
        )
        best = int(model.best_iteration_ or REGRESSOR_PARAMS["n_estimators"])
        best_iterations.append(best)
        pred_fold = np.power(10.0, np.clip(oof_log[valid_indices], -2.0, 5.0))
        fold_metrics.append(
            float(
                mean_squared_error(
                    data["target_raw"][valid_indices], pred_fold
                )
            )
        )
        print(f"Fold {fold['fold']}: MSE_raw={fold_metrics[-1]:.6g}, best={best}")

    if not np.isfinite(oof_log).all():
        raise RuntimeError("Expanded baseline OOF contains missing or non-finite values")
    oof_log = np.clip(oof_log, -2.0, 5.0)
    pred_raw = np.power(10.0, oof_log)
    mse_full = float(mean_squared_error(data["target_raw"], pred_raw))
    mse_original = float(mean_squared_error(data["target_raw"][:318], pred_raw[:318]))
    rmse_log = float(np.sqrt(mean_squared_error(data["target_log"], oof_log)))
    r2_log = float(r2_score(data["target_log"], oof_log))
    mae_raw = float(mean_absolute_error(data["target_raw"], pred_raw))

    np.save(RESULTS_DIR / "exp01oof_log10_expanded.npy", oof_log)
    path = write_result(
        "exp01baseline_b_expanded.json",
        {
            "model": "lgbmregressoron_log10_expanded",
            "params": REGRESSOR_PARAMS
            | {"early_stopping_rounds": EARLY_STOPPING_ROUNDS, "eval_metric": "rmse"},
            "cvmetric": mse_full,
            "metricname": "MSEraw",
            "mse_original_318": mse_original,
            "rmse_log": rmse_log,
            "r2_log": r2_log,
            "mae_raw": mae_raw,
            "fold_metrics": fold_metrics,
            "best_iterations": best_iterations,
            "canonical_duplicate_rows_purged_per_fold": purged_rows,
            "notes": "349-row OOF; original Kaggle validation folds retained; duplicate canonical structures are removed from each fold's training data; no IC50/CC50 features.",
        },
    )
    print(f"MSE_raw (all 349): {mse_full:.6g}")
    print(f"MSE_raw (original 318 rows): {mse_original:.6g}")
    print(f"RMSE_log: {rmse_log:.6f}; R2_log: {r2_log:.6f}")
    print(f"Saved result: {path}")


if __name__ == "__main__":
    main()
