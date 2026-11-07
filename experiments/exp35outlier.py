import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"
PARAMS = {
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


def load_inputs():
    train = pd.read_csv(ROOT / "train_clean_v2.csv")
    target = pd.to_numeric(train["SI VACV"], errors="coerce").to_numpy(dtype=float)
    descriptors = pd.read_csv(ROOT / "featurestraindesc_v2.csv").to_numpy(dtype=np.float32)
    with np.load(ROOT / "featurestrainfp_v2.npz") as data:
        fingerprints = data["fingerprints"].astype(np.float32)
    features = np.concatenate([descriptors, fingerprints], axis=1)
    folds = json.loads((RESULTS_DIR / "folds_v2.json").read_text(encoding="utf-8"))["folds"]
    if len(train) != 320 or features.shape != (len(train), 2265):
        raise ValueError(f"Unexpected v2 train/features shape: {len(train)}, {features.shape}")
    if not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("SI target must be finite and nonnegative")
    return train, target, features, folds


def get_outlier(train, target):
    match = train["Title"].astype(str).str.strip().eq("8e-as-555-15-16").to_numpy()
    if int(match.sum()) != 1:
        raise ValueError(f"Expected one 8e-as-555-15-16 row, found {int(match.sum())}")
    index = int(np.flatnonzero(match)[0])
    row = train.iloc[index]
    ic50 = float(row["IC50_uM_VACV"])
    cc50 = float(row["CC50_uM"])
    si = float(row["SI VACV"])
    ratio = cc50 / ic50
    if not np.isclose(ratio, si, rtol=1e-5, atol=1e-4):
        raise ValueError(f"Outlier SI is inconsistent: CC50/IC50={ratio}, SI={si}")
    if not np.isclose(target[index], si):
        raise ValueError("Outlier target does not match SI VACV")
    return index, {
        "index": index,
        "title": str(row["Title"]),
        "IC50_uM_VACV": ic50,
        "CC50_uM": cc50,
        "SI_VACV": si,
        "CC50_div_IC50": ratio,
        "consistent": True,
    }


def run_cv(features, target, folds, label):
    oof = np.full(len(target), np.nan, dtype=float)
    fold_mses = []
    best_iterations = []
    for fold in folds:
        train_indices = np.asarray(fold["train_indices"], dtype=int)
        valid_indices = np.asarray(fold["validation_indices"], dtype=int)
        model = lgb.LGBMRegressor(**PARAMS)
        model.fit(
            features[train_indices],
            target[train_indices],
            eval_set=[(features[valid_indices], target[valid_indices])],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        iteration = int(model.best_iteration_ or PARAMS["n_estimators"])
        oof[valid_indices] = model.predict(features[valid_indices], num_iteration=iteration)
        mse = float(mean_squared_error(target[valid_indices], oof[valid_indices]))
        fold_mses.append(mse)
        best_iterations.append(iteration)
        print(
            f"{label} fold={fold['fold']} MSE_raw={mse:.6g}, trees={iteration}",
            flush=True,
        )
    if not np.isfinite(oof).all():
        raise RuntimeError(f"{label} OOF predictions are incomplete")
    return oof, fold_mses, best_iterations


def score(prediction, target, keep):
    return float(mean_squared_error(target[keep], prediction[keep]))


def main():
    train, target, features, folds = load_inputs()
    outlier_index, outlier = get_outlier(train, target)
    log_target = np.log1p(target)
    raw_oof, raw_fold_mses, raw_iterations = run_cv(
        features, target, folds, "raw-SI"
    )
    log_oof, log_fold_mses, log_iterations = run_cv(
        features, log_target, folds, "log1p-SI"
    )

    log_base_prediction = np.expm1(np.clip(log_oof, 0.0, 30.0))
    residuals = log_target - log_oof
    global_smearing = float(np.mean(np.exp(np.clip(residuals, -50.0, 50.0))))
    global_smearing_oof = log_base_prediction * global_smearing

    local_smearing = np.full(len(target), np.nan, dtype=float)
    fold_smear_factors = []
    for fold in folds:
        train_indices = np.asarray(fold["train_indices"], dtype=int)
        valid_indices = np.asarray(fold["validation_indices"], dtype=int)
        factor = float(
            np.mean(
                np.exp(
                    np.clip(residuals[train_indices], -50.0, 50.0)
                )
            )
        )
        local_smearing[valid_indices] = log_base_prediction[valid_indices] * factor
        fold_smear_factors.append({"fold": int(fold["fold"]), "smearing_factor": factor})
    if not np.isfinite(local_smearing).all():
        raise RuntimeError("Cross-fitted smearing predictions are incomplete")

    fold_raw_mses = {
        "raw_si": [],
        "log1p_backtransform": [],
        "log1p_global_smearing": [],
        "log1p_fold_cross_fitted_smearing": [],
    }
    for fold in folds:
        valid_indices = np.asarray(fold["validation_indices"], dtype=int)
        fold_raw_mses["raw_si"].append(
            score(raw_oof, target, np.isin(np.arange(len(target)), valid_indices))
        )
        fold_raw_mses["log1p_backtransform"].append(
            score(log_base_prediction, target, np.isin(np.arange(len(target)), valid_indices))
        )
        fold_raw_mses["log1p_global_smearing"].append(
            score(global_smearing_oof, target, np.isin(np.arange(len(target)), valid_indices))
        )
        fold_raw_mses["log1p_fold_cross_fitted_smearing"].append(
            score(local_smearing, target, np.isin(np.arange(len(target)), valid_indices))
        )

    keep_all = np.ones(len(target), dtype=bool)
    keep_without = keep_all.copy()
    keep_without[outlier_index] = False
    scores = {
        "raw_si": {
            "mse_all": score(raw_oof, target, keep_all),
            "mse_without_outlier": score(raw_oof, target, keep_without),
        },
        "log1p_backtransform": {
            "mse_all": score(log_base_prediction, target, keep_all),
            "mse_without_outlier": score(log_base_prediction, target, keep_without),
        },
        "log1p_global_smearing": {
            "mse_all": score(global_smearing_oof, target, keep_all),
            "mse_without_outlier": score(global_smearing_oof, target, keep_without),
        },
        "log1p_fold_cross_fitted_smearing": {
            "mse_all": score(local_smearing, target, keep_all),
            "mse_without_outlier": score(local_smearing, target, keep_without),
        },
    }

    np.save(RESULTS_DIR / "exp35oof_raw_si.npy", raw_oof)
    np.save(RESULTS_DIR / "exp35oof_log1p_si.npy", log_base_prediction)
    np.save(RESULTS_DIR / "exp35oof_log1p_smearing_global.npy", global_smearing_oof)
    np.save(RESULTS_DIR / "exp35oof_log1p_smearing_crossfit.npy", local_smearing)
    result = {
        "model": "lgbm_raw_vs_log1p_vs_smearing",
        "params": PARAMS,
        "outlier": outlier,
        "outlier_fold": next(
            int(fold["fold"])
            for fold in folds
            if outlier_index in fold["validation_indices"]
        ),
        "global_smearing_factor": global_smearing,
        "fold_smearing_factors": fold_smear_factors,
        "cv_mse_raw": scores,
        "foldmetrics_raw_si": fold_raw_mses,
        "foldmetrics_model_target": {
            "raw_si": raw_fold_mses,
            "log1p": log_fold_mses,
            "best_iterations_raw": raw_iterations,
            "best_iterations_log1p": log_iterations,
        },
        "selected_target": min(
            scores,
            key=lambda name: scores[name]["mse_all"],
        ),
        "notes": (
            "Global smearing follows the requested mean(exp(OOF residual)) formula and "
            "is optimistic because the factor uses all OOF labels. Fold-cross-fitted "
            "smearing estimates each validation factor using residuals from other folds."
        ),
    }
    report_path = RESULTS_DIR / "exp35outlier.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Outlier: {json.dumps(outlier)}")
    print(f"Global smearing factor: {global_smearing:.8g}")
    print(f"MSE comparison: {json.dumps(scores, indent=2)}")
    print(f"Selected target: {result['selected_target']}")
    print(f"Saved result: {report_path}")


if __name__ == "__main__":
    main()
