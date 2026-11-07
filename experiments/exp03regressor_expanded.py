import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    active = data["active_base"]
    active_indices = np.flatnonzero(active)
    oof_log = np.full(len(active), np.nan, dtype=float)
    fold_metrics = []
    best_iterations = []
    purged_rows = []
    for fold in folds:
        train_indices = np.asarray(fold["train_indices"], dtype=int)
        valid_indices = np.asarray(fold["validation_indices"], dtype=int)
        train_indices, purged_count = purge_canonical_duplicates(
            train_indices, valid_indices, data["canonical_base"]
        )
        purged_rows.append(purged_count)
        active_train = train_indices[active[train_indices]]
        active_valid = valid_indices[active[valid_indices]]
        model = lgb.LGBMRegressor(**REGRESSOR_PARAMS)
        model.fit(
            data["x_base"][active_train],
            data["target_log"][active_train],
            eval_set=[
                (data["x_base"][active_valid], data["target_log"][active_valid])
            ],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        oof_log[valid_indices] = model.predict(
            data["x_base"][valid_indices], num_iteration=model.best_iteration_
        )
        best = int(model.best_iteration_ or REGRESSOR_PARAMS["n_estimators"])
        best_iterations.append(best)
        fold_metrics.append(
            float(
                np.sqrt(
                    mean_squared_error(
                        data["target_log"][active_valid],
                        oof_log[active_valid],
                    )
                )
            )
        )
        print(
            f"Fold {fold['fold']}: active train={len(active_train)}, "
            f"active valid={len(active_valid)}, RMSE_log={fold_metrics[-1]:.6f}, "
            f"best={best}"
        )

    if not np.isfinite(oof_log).all():
        raise RuntimeError("Expanded regressor OOF contains missing or non-finite values")
    active_log = oof_log[active]
    active_raw = data["target_raw"][active]
    pred_raw = np.power(10.0, np.clip(active_log, -2.0, 5.0))
    rmse_log = float(np.sqrt(mean_squared_error(data["target_log"][active], active_log)))
    mse_raw = float(mean_squared_error(active_raw, pred_raw))
    mae_raw = float(mean_absolute_error(active_raw, pred_raw))
    r2_log = float(r2_score(data["target_log"][active], active_log))
    original_active = np.arange(318)[data["active_base"][:318]]
    original_active_prediction = np.power(
        10.0, np.clip(oof_log[original_active], -2.0, 5.0)
    )
    original_active_raw = data["target_raw"][original_active]
    rmse_original = float(
        np.sqrt(
            mean_squared_error(
                data["target_log"][original_active],
                oof_log[original_active],
            )
        )
    )
    mse_raw_original = float(
        mean_squared_error(original_active_raw, original_active_prediction)
    )
    mae_raw_original = float(
        mean_absolute_error(original_active_raw, original_active_prediction)
    )

    np.save(RESULTS_DIR / "exp03oof_log10_expanded.npy", oof_log)
    path = write_result(
        "exp03regressor_expanded.json",
        {
            "model": "lgbmregressoron_active_expanded",
            "params": REGRESSOR_PARAMS
            | {"early_stopping_rounds": EARLY_STOPPING_ROUNDS, "eval_metric": "rmse"},
            "cvmetric": rmse_log,
            "metricname": "RMSElog10",
            "mseraw": mse_raw,
            "maeraw": mae_raw,
            "r2_log": r2_log,
            "rmse_log_original_237": rmse_original,
            "mseraw_original_237": mse_raw_original,
            "maeraw_original_237": mae_raw_original,
            "foldmetrics": fold_metrics,
            "bestiterations": best_iterations,
            "canonical_duplicate_rows_purged_per_fold": purged_rows,
            "notes": "268 active rows; primary metrics include Figshare rows; canonical-SMILES duplicates are purged from each fold's training data; original 237-row RMSE is provided for comparison.",
        },
    )
    print(f"Active rows: {len(active_indices)}")
    print(f"RMSE_log all 268 active: {rmse_log:.6f}")
    print(f"RMSE_log original 237 active: {rmse_original:.6f}")
    print(f"R2_log: {r2_log:.6f}; MSE_raw: {mse_raw:.6g}; MAE_raw: {mae_raw:.6g}")
    print(f"Saved result: {path}")


if __name__ == "__main__":
    main()
