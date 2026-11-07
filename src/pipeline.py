import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from data_prep import DATA, ROOT, load_test, load_train, validate_submission
from models import featurize_smiles, fit_model, graph, predict, transform_features


SEEDS = (42, 43, 44, 45, 46)
N_FOLDS = 5
EPOCHS = 80
RESULTS = ROOT / "experiments" / "results"


def make_folds(target, groups, seed):
    strata = np.digitize(target, bins=[0.001, 10.0, 100.0, 1000.0])
    splitter = StratifiedGroupKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=seed,
    )
    return list(splitter.split(np.zeros(len(target)), strata, groups))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def predict_probability(model, graphs, features, indices):
    logits = predict(model, graphs, features, indices)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def run_oof(train, target, activity, raw, graphs):
    group_ids = raw["canonical"]
    seed_oof_probabilities = []
    seed_oof_logsi = []
    seed_mses = {}
    all_fold_metrics = []
    active_indices = np.flatnonzero(activity == 1)

    for seed in SEEDS:
        prob_oof = np.full(len(train), np.nan, dtype=np.float64)
        logsi_oof = np.full(len(train), np.nan, dtype=np.float64)
        for fold_index, (train_indices, valid_indices) in enumerate(
            make_folds(target, group_ids, seed),
            start=1,
        ):
            train_indices = np.asarray(train_indices, dtype=int)
            valid_indices = np.asarray(valid_indices, dtype=int)
            if set(group_ids[train_indices]) & set(group_ids[valid_indices]):
                raise ValueError("Canonical SMILES cross a GroupKFold boundary")

            classifier_features, _ = transform_features(raw, train_indices)
            classifier = fit_model(
                graphs,
                classifier_features,
                activity,
                train_indices,
                seed * 100 + fold_index,
                "classifier",
                EPOCHS,
            )
            prob_oof[valid_indices] = predict_probability(
                classifier,
                graphs,
                classifier_features,
                valid_indices,
            )

            regressor_train = train_indices[activity[train_indices] == 1]
            regressor_features, _ = transform_features(raw, regressor_train)
            regressor = fit_model(
                graphs,
                regressor_features,
                np.log1p(target),
                regressor_train,
                seed * 1000 + fold_index,
                "regressor",
                EPOCHS,
            )
            logsi_oof[valid_indices] = predict(
                regressor,
                graphs,
                regressor_features,
                valid_indices,
            )
            fold_regression_mse = float(
                mean_squared_error(
                    target[valid_indices[activity[valid_indices] == 1]],
                    np.expm1(
                        np.clip(
                            logsi_oof[valid_indices][activity[valid_indices] == 1],
                            0.0,
                            14.0,
                        )
                    ),
                )
            )
            all_fold_metrics.append(
                {
                    "seed": seed,
                    "fold": fold_index,
                    "classifier_train_rows": int(len(train_indices)),
                    "regressor_train_rows": int(len(regressor_train)),
                    "validation_rows": int(len(valid_indices)),
                    "active_validation_rows": int(
                        np.sum(activity[valid_indices] == 1)
                    ),
                    "regressor_raw_si_mse_active": fold_regression_mse,
                }
            )
            print(
                f"OOF seed={seed} fold={fold_index} "
                f"valid={len(valid_indices)} active_train={len(regressor_train)} "
                f"active_MSE={fold_regression_mse:.6g}",
                flush=True,
            )

        if not np.isfinite(prob_oof).all() or not np.isfinite(logsi_oof).all():
            raise RuntimeError(f"Incomplete OOF predictions for seed {seed}")
        seed_oof_probabilities.append(prob_oof)
        seed_oof_logsi.append(logsi_oof)
        seed_mses[str(seed)] = float(
            mean_squared_error(
                target,
                prob_oof * np.expm1(np.clip(logsi_oof, 0.0, 14.0)),
            )
        )

    average_probability = np.mean(seed_oof_probabilities, axis=0)
    average_logsi = np.mean(seed_oof_logsi, axis=0)
    raw_si_oof = np.expm1(np.clip(average_logsi, 0.0, 14.0))
    active_regression_mse = float(
        mean_squared_error(
            target[active_indices],
            raw_si_oof[active_indices],
        )
    )
    calibrator_oof = np.full(len(train), np.nan, dtype=np.float64)
    for train_indices, valid_indices in make_folds(target, group_ids, SEEDS[0]):
        calibrator = IsotonicRegression(
            increasing=True,
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
        ).fit(
            average_probability[train_indices],
            activity[train_indices],
        )
        calibrator_oof[valid_indices] = calibrator.predict(
            average_probability[valid_indices]
        )
    if not np.isfinite(calibrator_oof).all():
        raise RuntimeError("Cross-fitted isotonic probabilities are incomplete")
    calibrated_oof = calibrator_oof * raw_si_oof

    return {
        "seed_probabilities": seed_oof_probabilities,
        "seed_logsi": seed_oof_logsi,
        "average_probability": average_probability,
        "average_logsi": average_logsi,
        "raw_si_oof": raw_si_oof,
        "calibrated_probability_oof": calibrator_oof,
        "calibrated_si_oof": calibrated_oof,
        "seed_blend_mse": seed_mses,
        "fold_metrics": all_fold_metrics,
        "classifier_roc_auc": float(
            roc_auc_score(activity, average_probability)
        ),
        "regressor_mse_log1p_active": float(
            mean_squared_error(
                np.log1p(target[active_indices]),
                average_logsi[active_indices],
            )
        ),
        "regressor_mse_raw_si_active": active_regression_mse,
        "uncalibrated_soft_blend_mse": float(
            mean_squared_error(target, average_probability * raw_si_oof)
        ),
        "cross_fitted_calibrated_blend_mse": float(
            mean_squared_error(target, calibrated_oof)
        ),
        "calibrated_blend_oof_caveat": "Calibration folds exclude validation labels, but base OOF models for calibration-training rows can include labels from other folds; this is diagnostic, not a fully nested unbiased score.",
    }


def fit_full_models(train, target, activity, raw, graphs):
    all_indices = np.arange(len(train))
    active_indices = np.flatnonzero(activity == 1)
    classifier_features, classifier_scaler = transform_features(raw, all_indices)
    regressor_features, regressor_scaler = transform_features(raw, active_indices)
    classifiers, regressors = [], []
    for seed in SEEDS:
        classifiers.append(
            fit_model(
                graphs,
                classifier_features,
                activity,
                all_indices,
                seed,
                "classifier",
                EPOCHS,
            )
        )
        regressors.append(
            fit_model(
                graphs,
                regressor_features,
                np.log1p(target),
                active_indices,
                seed + 100,
                "regressor",
                EPOCHS,
            )
        )
        print(f"Fitted final classifier/regressor seed={seed}", flush=True)
    return (
        classifiers,
        regressors,
        classifier_features,
        regressor_features,
        classifier_scaler,
        regressor_scaler,
    )


def create_final_submission(train, target, activity, raw, graphs, oof, test):
    classifiers, regressors, _, _, classifier_scaler, regressor_scaler = (
        fit_full_models(train, target, activity, raw, graphs)
    )
    test_raw = featurize_smiles(test["SMILES"].astype(str).tolist())
    test_graphs = [graph(smiles) for smiles in test["SMILES"].astype(str)]
    classifier_test_desc = np.clip(
        classifier_scaler.transform(test_raw["desc"]),
        -8.0,
        8.0,
    ).astype(np.float32)
    classifier_test_features = np.concatenate(
        [test_raw["fp2"], test_raw["maccs"], classifier_test_desc],
        axis=1,
    ).astype(np.float32)
    regressor_test_desc = np.clip(
        regressor_scaler.transform(test_raw["desc"]),
        -8.0,
        8.0,
    ).astype(np.float32)
    regressor_test_features = np.concatenate(
        [test_raw["fp2"], test_raw["maccs"], regressor_test_desc],
        axis=1,
    ).astype(np.float32)
    classifier_test_probabilities = np.mean(
        [
            predict_probability(
                model,
                test_graphs,
                classifier_test_features,
                np.arange(len(test)),
            )
            for model in classifiers
        ],
        axis=0,
    )
    regressor_test_logsi = np.mean(
        [
            predict(
                model,
                test_graphs,
                regressor_test_features,
                np.arange(len(test)),
            )
            for model in regressors
        ],
        axis=0,
    )
    calibrator = IsotonicRegression(
        increasing=True,
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    ).fit(
        oof["average_probability"],
        activity,
    )
    calibrated_probability = calibrator.predict(classifier_test_probabilities)
    regressor_si = np.expm1(np.clip(regressor_test_logsi, 0.0, 14.0))
    final_si = calibrated_probability * regressor_si
    submission = pd.DataFrame(
        {"index": test["index"].to_numpy(), "SI VACV": final_si}
    ).sort_values("index", kind="stable")
    validate_submission(submission)

    submission_dir = ROOT / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)
    submission_path = submission_dir / "submission_final.csv"
    submission.to_csv(submission_path, index=False)
    model_path = RESULTS / "final_strategy_c_models.joblib"
    joblib.dump(
        {
            "classifiers": classifiers,
            "regressors": regressors,
            "classifier_scaler": classifier_scaler,
            "regressor_scaler": regressor_scaler,
            "classifier_isotonic_calibrator": calibrator,
            "seeds": list(SEEDS),
            "epochs": EPOCHS,
            "target_definition": "active iff SI VACV > 0; regression trained on the 237 positive-SI rows",
        },
        model_path,
        compress=3,
    )
    return {
        "path": str(submission_path.relative_to(ROOT)),
        "rows": len(submission),
        "min": float(submission["SI VACV"].min()),
        "max": float(submission["SI VACV"].max()),
        "max_target_9000_met": bool(submission["SI VACV"].max() <= 9000.0),
        "mean": float(submission["SI VACV"].mean()),
        "median": float(submission["SI VACV"].median()),
        "calibrated_probability_min": float(calibrated_probability.min()),
        "calibrated_probability_max": float(calibrated_probability.max()),
        "regressor_si_max": float(regressor_si.max()),
        "model_path": str(model_path.relative_to(ROOT)),
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    train, target, activity = load_train()
    test = load_test()
    train_smiles = train["SMILES"].astype(str).tolist()
    train_raw = featurize_smiles(train_smiles)
    graphs = [graph(smiles) for smiles in train_smiles]
    oof = run_oof(train, target, activity, train_raw, graphs)
    np.save(RESULTS / "final_classifier_oof_probs.npy", oof["average_probability"])
    np.save(RESULTS / "final_regressor_oof_log1p.npy", oof["average_logsi"])
    np.save(RESULTS / "final_crossfitted_calibrated_oof.npy", oof["calibrated_si_oof"])

    final_submission = create_final_submission(
        train,
        target,
        activity,
        train_raw,
        graphs,
        oof,
        test,
    )
    report = {
        "model": "five_seed_GCN_classifier_regressor_with_OOF_isotonic_probability_calibration",
        "seeds": list(SEEDS),
        "folds_per_seed": N_FOLDS,
        "fold_splitter": "StratifiedGroupKFold(shuffle=True), SI bins [0.001, 10, 100, 1000], canonical SMILES groups",
        "epochs": EPOCHS,
        "training_rows": len(train),
        "positive_si_regression_rows": int(activity.sum()),
        "zero_si_rows": int(len(target) - activity.sum()),
        "calibration": {
            "method": "IsotonicRegression(increasing=True, y_min=0, y_max=1)",
            "input": "Mean of five-seed GroupKFold OOF classifier probabilities",
            "target": "Binary positive-SI activity labels",
            "fit_protocol": "OOF calibration fit on all training OOF predictions for test; diagnostic calibrated OOF scores use calibrators fitted only on other stratified group folds.",
            "friend_code_note": "The teammate's CrossFitCalibrator actually calibrates raw SI regression predictions to continuous SI targets, not classifier probabilities. This pipeline reuses its isotonic method/order on the requested classifier probabilities; it does not claim to reuse the teammate's fitted regression calibrator.",
        },
        "classifier_roc_auc": oof["classifier_roc_auc"],
        "regressor_mse_log1p_positive_si": oof["regressor_mse_log1p_active"],
        "regressor_mse_raw_si_positive_rows": oof["regressor_mse_raw_si_active"],
        "seed_oof_blend_mse_raw_si": oof["seed_blend_mse"],
        "uncalibrated_soft_blend_mse_raw_si": oof["uncalibrated_soft_blend_mse"],
        "cross_fitted_calibrated_blend_mse_raw_si": oof[
            "cross_fitted_calibrated_blend_mse"
        ],
        "final_submission": final_submission,
        "fold_metrics": oof["fold_metrics"],
        "oof_files": {
            "classifier_probability": "experiments/results/final_classifier_oof_probs.npy",
            "regressor_log1p": "experiments/results/final_regressor_oof_log1p.npy",
            "calibrated_blend": "experiments/results/final_crossfitted_calibrated_oof.npy",
        },
    }
    write_json(RESULTS / "final_pipeline_report.json", report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
