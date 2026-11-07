import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


ROOT = Path(__file__).resolve().parent
FRIEND = ROOT / "from_friend"
RESULTS = ROOT / "experiments" / "results"
SEEDS = (42, 43, 44, 45, 46)
N_FOLDS = 5
EPOCHS = 80


def load_friend_modules():
    sys.path.insert(0, str(FRIEND))
    from src.tail_research import TailRegressor
    from src.validation import folds
    import vacv_cpu_improved as cpu

    return TailRegressor, folds, cpu


def load_data():
    train_path = ROOT / "train_v2.csv"
    train = pd.read_csv(train_path, usecols=["SMILES", "SI VACV"])
    target = pd.to_numeric(train["SI VACV"], errors="raise").to_numpy(dtype=float)
    if len(train) != 320 or not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("Expected 320 finite nonnegative SI targets")

    test = pd.read_csv(ROOT / "test_clean.csv", usecols=["index", "SMILES"])
    if len(test) != 597 or test["index"].duplicated().any():
        raise ValueError("Expected 597 test rows with unique indices")

    friend_train = pd.read_csv(
        FRIEND / "train.csv",
        usecols=["SMILES", "SI VACV"],
    )
    train_hash = hashlib.sha256(train_path.read_bytes()).hexdigest()
    friend_hash = hashlib.sha256((FRIEND / "train.csv").read_bytes()).hexdigest()
    if train_hash != friend_hash:
        raise ValueError("train_v2.csv differs from the teammate's pinned training data")
    if not np.array_equal(
        train["SI VACV"].to_numpy(dtype=float),
        friend_train["SI VACV"].to_numpy(dtype=float),
    ):
        raise ValueError("Teammate and local train targets do not match")
    return train, target, test, train_hash


def run_seed(seed, train, target, test, TailRegressor, folds, cpu):
    features = cpu.chem_features(train["SMILES"].astype(str).tolist())
    groups = features["canon"]
    split = folds(target, groups, seed, N_FOLDS)
    oof = np.full(len(train), np.nan, dtype=float)
    fold_mses = []

    for fold_index, (train_indices, valid_indices) in enumerate(split, start=1):
        train_indices = np.asarray(train_indices, dtype=int)
        valid_indices = np.asarray(valid_indices, dtype=int)
        if set(groups[train_indices]) & set(groups[valid_indices]):
            raise ValueError(f"Canonical groups cross seed={seed}, fold={fold_index}")

        model = TailRegressor("control_log", epochs=EPOCHS).fit(
            train.iloc[train_indices].reset_index(drop=True),
            target[train_indices],
            seed * 100 + fold_index - 1,
        )
        oof[valid_indices] = model.predict(
            train.iloc[valid_indices].reset_index(drop=True)
        )
        fold_mse = float(mean_squared_error(target[valid_indices], oof[valid_indices]))
        fold_mses.append(fold_mse)
        print(
            f"seed={seed} fold={fold_index} rows={len(valid_indices)} "
            f"MSE_raw_SI={fold_mse:.6g}",
            flush=True,
        )

    if not np.isfinite(oof).all():
        raise RuntimeError(f"Incomplete OOF predictions for seed {seed}")

    final_model = TailRegressor("control_log", epochs=EPOCHS).fit(
        train.reset_index(drop=True),
        target,
        seed,
    )
    test_prediction = final_model.predict(test[["SMILES"]].copy())
    if test_prediction.shape != (len(test),) or not np.isfinite(test_prediction).all():
        raise ValueError(f"Invalid test predictions for seed {seed}")
    if (test_prediction < 0).any():
        raise ValueError(f"Negative test predictions for seed {seed}")

    return {
        "oof": oof,
        "oof_mse": float(mean_squared_error(target, oof)),
        "fold_mses": fold_mses,
        "test_prediction": test_prediction,
        "model": final_model,
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    TailRegressor, folds, cpu = load_friend_modules()
    train, target, test, training_hash = load_data()
    seed_results = {}
    test_predictions = []
    oof_predictions = []

    for seed in SEEDS:
        print(f"===== Seed {seed} =====", flush=True)
        result = run_seed(seed, train, target, test, TailRegressor, folds, cpu)
        seed_results[seed] = result
        test_predictions.append(result["test_prediction"])
        oof_predictions.append(result["oof"])
        np.save(RESULTS / f"exp53_gcn_seed_{seed}_oof.npy", result["oof"])
        joblib.dump(
            result["model"],
            RESULTS / f"exp53_gcn_seed_{seed}_model.joblib",
            compress=3,
        )

    oof_matrix = np.stack(oof_predictions)
    test_matrix = np.stack(test_predictions)
    mean_oof = oof_matrix.mean(axis=0)
    mean_test = test_matrix.mean(axis=0)
    if not np.isfinite(mean_test).all() or (mean_test < 0).any():
        raise ValueError("Seed-averaged test predictions are invalid")

    submission = pd.DataFrame(
        {"index": test["index"].to_numpy(), "SI VACV": mean_test}
    ).sort_values("index", kind="stable")
    if (
        len(submission) != 597
        or submission["index"].duplicated().any()
        or submission.isna().any().any()
        or not np.isfinite(submission["SI VACV"]).all()
        or (submission["SI VACV"] < 0).any()
    ):
        raise ValueError("Final seed-averaged submission validation failed")
    submission_path = ROOT / "submission_final_seed_avg.csv"
    submission.to_csv(submission_path, index=False)
    np.save(RESULTS / "exp53_gcn_seed_avg_oof.npy", mean_oof)

    per_seed_mses = {
        str(seed): seed_results[seed]["oof_mse"] for seed in SEEDS
    }
    report = {
        "model": "FriendGraphNet_control_log_seed_average",
        "seeds": list(SEEDS),
        "folds_per_seed": N_FOLDS,
        "epochs": EPOCHS,
        "training_rows": len(train),
        "test_rows": len(test),
        "training_sha256": training_hash,
        "per_seed": {
            str(seed): {
                "oof_mse_raw_si": seed_results[seed]["oof_mse"],
                "fold_mses": seed_results[seed]["fold_mses"],
                "oof_path": f"experiments/results/exp53_gcn_seed_{seed}_oof.npy",
                "model_path": f"experiments/results/exp53_gcn_seed_{seed}_model.joblib",
            }
            for seed in SEEDS
        },
        "mean_per_seed_oof_mse": float(np.mean(list(per_seed_mses.values()))),
        "seed_averaged_oof_mse": float(mean_squared_error(target, mean_oof)),
        "seed_averaged_oof_path": "experiments/results/exp53_gcn_seed_avg_oof.npy",
        "submission": {
            "path": str(submission_path.relative_to(ROOT)),
            "rows": len(submission),
            "prediction_min": float(submission["SI VACV"].min()),
            "prediction_max": float(submission["SI VACV"].max()),
            "prediction_mean": float(submission["SI VACV"].mean()),
            "prediction_median": float(submission["SI VACV"].median()),
        },
    }
    report_path = RESULTS / "exp53_seed_averaging.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
