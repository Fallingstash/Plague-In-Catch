import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.metrics import mean_squared_error


ROOT = Path(__file__).resolve().parent
FRIEND = ROOT / "from_friend"
RESULTS = ROOT / "experiments" / "results"
FIGSHARE_PATH = ROOT / "data" / "final_collected_dataset.csv"
BASELINE_OOF_PATH = FRIEND / "results" / "isotonic_gcn_7357_2643" / "gcn_oof.csv"
BASELINE_MANIFEST_PATH = FRIEND / "results" / "isotonic_gcn_7357_2643" / "manifest.json"
EXPECTED_BASELINE_MSE = 3_831_981.9529758967
BASELINE_TOLERANCE = 1e-6
SEEDS = (42, 43)
N_SPLITS = 5
EPOCHS = 80


def canonical(smiles):
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(f"Invalid or empty SMILES: {smiles}")
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_friend_modules():
    sys.path.insert(0, str(FRIEND))
    from src.tail_research import TailRegressor
    import vacv_cpu_improved as cpu
    from src.validation import folds, sha

    return TailRegressor, cpu, folds, sha


def reproduce_baseline(train, target, TailRegressor, cpu, folds):
    if hashlib.sha256((ROOT / "train_v2.csv").read_bytes()).hexdigest() != hashlib.sha256(
        (FRIEND / "train.csv").read_bytes()
    ).hexdigest():
        raise ValueError("train_v2.csv bytes differ from teammate's pinned train.csv")

    manifest = json.loads(BASELINE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("training_sha256") != hashlib.sha256(
        (ROOT / "train_v2.csv").read_bytes()
    ).hexdigest():
        raise ValueError("Teammate baseline manifest does not match train_v2.csv")
    if manifest.get("oof_seeds") != list(SEEDS) or manifest.get("oof_folds") != N_SPLITS:
        raise ValueError("Teammate baseline hyperparameter/split manifest differs")

    saved_oof = pd.read_csv(BASELINE_OOF_PATH)
    required = {"row", "fold", "true", "gcn_oof_seed_average"}
    if (
        len(saved_oof) != len(train)
        or not required.issubset(saved_oof.columns)
        or saved_oof["row"].duplicated().any()
        or set(saved_oof["row"]) != set(range(len(train)))
    ):
        raise ValueError("Teammate baseline OOF is incomplete or malformed")
    saved_oof = saved_oof.set_index("row").loc[np.arange(len(train))]
    if not np.array_equal(saved_oof["true"].to_numpy(dtype=float), target):
        raise ValueError("Teammate OOF targets do not exactly match train_v2.csv")

    raw = cpu.chem_features(train["SMILES"].astype(str).tolist())
    groups = raw["canon"]
    reproduced = np.zeros(len(target), dtype=float)
    fold_ids = np.zeros(len(target), dtype=int)
    for seed in SEEDS:
        seed_oof = np.full(len(target), np.nan)
        split = folds(target, groups, seed, N_SPLITS)
        for fold_index, (_, valid_indices) in enumerate(split, start=1):
            cache_path = (
                FRIEND
                / "experiments"
                / "tail_20261107"
                / f"{seed}_control_log"
                / f"fold_{fold_index}.joblib"
            )
            model = joblib.load(cache_path)
            if not isinstance(model, TailRegressor) or model.mode != "control_log":
                raise ValueError(f"Incompatible baseline checkpoint: {cache_path}")
            valid_indices = np.asarray(valid_indices, dtype=int)
            validation_frame = train.iloc[valid_indices].reset_index(drop=True)
            seed_oof[valid_indices] = model.predict(validation_frame)
            if seed == SEEDS[0]:
                fold_ids[valid_indices] = fold_index
        if not np.isfinite(seed_oof).all():
            raise RuntimeError(f"Baseline seed {seed} produced incomplete OOF predictions")
        reproduced += seed_oof / len(SEEDS)

    saved_prediction = saved_oof["gcn_oof_seed_average"].to_numpy(dtype=float)
    if not np.allclose(reproduced, saved_prediction, rtol=1e-5, atol=1e-6):
        max_prediction_difference = float(np.max(np.abs(reproduced - saved_prediction)))
        raise ValueError(
            "Reproduced baseline OOF predictions differ beyond floating-point tolerance: "
            f"max_abs_difference={max_prediction_difference}"
        )
    baseline_mse = float(mean_squared_error(target, reproduced))
    difference = baseline_mse - EXPECTED_BASELINE_MSE
    relative_difference = difference / EXPECTED_BASELINE_MSE
    np.save(RESULTS / "exp49_gcn_baseline_oof.npy", reproduced)
    if abs(relative_difference) > BASELINE_TOLERANCE:
        return {
            "status": "stopped_baseline_mismatch",
            "baseline_mse": baseline_mse,
            "expected_mse": EXPECTED_BASELINE_MSE,
            "difference": difference,
            "relative_difference": relative_difference,
            "oof_path": str((RESULTS / "exp49_gcn_baseline_oof.npy").relative_to(ROOT)),
        }
    return {
        "status": "matched",
        "baseline_mse": baseline_mse,
        "expected_mse": EXPECTED_BASELINE_MSE,
        "difference": difference,
        "relative_difference": relative_difference,
        "oof_path": str((RESULTS / "exp49_gcn_baseline_oof.npy").relative_to(ROOT)),
        "fold_ids": fold_ids,
    }


def load_figshare_and_combine(train):
    dataset = pd.read_csv(FIGSHARE_PATH)
    required = {
        "canonical_smiles",
        "IC50",
        "IC50_unit",
        "CC50",
        "CC50_unit",
        "SI",
        "source",
        "measurement_relation",
    }
    if not required.issubset(dataset.columns):
        raise ValueError(f"Collected dataset is missing columns: {sorted(required - set(dataset.columns))}")
    figshare = dataset[
        (dataset["source"] == "Figshare")
        & dataset["IC50"].notna()
        & dataset["CC50"].notna()
    ].copy()
    if len(figshare) != 31:
        raise ValueError(f"Expected 31 paired Figshare rows, got {len(figshare)}")
    exact = (
        (figshare["measurement_relation"] == "=")
        & (figshare["IC50_unit"] == figshare["CC50_unit"])
        & (pd.to_numeric(figshare["IC50"], errors="raise") > 0)
        & (pd.to_numeric(figshare["CC50"], errors="raise") > 0)
    )
    if not exact.all():
        raise ValueError("Figshare rows must be exact, positive pairs with matching units")
    figshare_smiles = figshare["canonical_smiles"].astype(str).map(canonical)
    reported_si = pd.to_numeric(figshare["SI"], errors="raise").to_numpy(dtype=float)
    calculated_si = (
        pd.to_numeric(figshare["CC50"], errors="raise").to_numpy(dtype=float)
        / pd.to_numeric(figshare["IC50"], errors="raise").to_numpy(dtype=float)
    )
    np.testing.assert_allclose(reported_si, calculated_si, rtol=1e-8, atol=1e-10)

    train_keys = [canonical(smiles) for smiles in train["SMILES"]]
    figshare_keys = figshare_smiles.tolist()
    if len(set(figshare_keys)) != len(figshare_keys):
        raise ValueError("Figshare contains duplicate canonical structures")
    overlap = set(train_keys) & set(figshare_keys)
    if overlap:
        raise ValueError(f"Figshare structures overlap train_v2: {len(overlap)}")

    external = pd.DataFrame(
        {
            "SMILES": figshare_keys,
            "SI VACV": calculated_si,
            "source": "Figshare",
        }
    )
    original = pd.DataFrame(
        {
            "SMILES": train["SMILES"].astype(str),
            "SI VACV": pd.to_numeric(train["SI VACV"], errors="raise"),
            "source": "train_v2",
        }
    )
    combined = pd.concat([original, external], ignore_index=True)
    target = combined["SI VACV"].to_numpy(dtype=float)
    if (
        len(combined) != 351
        or not np.isfinite(target).all()
        or (target < 0).any()
    ):
        raise ValueError("Combined dataset failed row-count or target validation")
    return combined, target, train_keys


def run_experiment_a(combined, target, train_keys, TailRegressor, cpu, folds):
    raw = cpu.chem_features(combined["SMILES"].astype(str).tolist())
    groups = raw["canon"]
    if len(set(groups[: len(train_keys)]) & set(groups[len(train_keys) :])):
        raise ValueError("Canonical group leakage between original train and Figshare")
    original_mask = np.arange(len(target)) < len(train_keys)
    averaged_oof = np.zeros(len(target), dtype=float)
    fold_sizes = []
    for seed in SEEDS:
        seed_oof = np.full(len(target), np.nan)
        split = folds(target, groups, seed, N_SPLITS)
        for fold_index, (train_indices, valid_indices) in enumerate(split, start=1):
            train_indices = np.asarray(train_indices, dtype=int)
            valid_indices = np.asarray(valid_indices, dtype=int)
            if set(groups[train_indices]) & set(groups[valid_indices]):
                raise ValueError(f"Canonical groups cross split boundary at fold {fold_index}")
            model = TailRegressor("control_log", epochs=EPOCHS).fit(
                combined.iloc[train_indices].reset_index(drop=True),
                target[train_indices],
                seed * 100 + fold_index - 1,
            )
            seed_oof[valid_indices] = model.predict(
                combined.iloc[valid_indices].reset_index(drop=True)
            )
            if seed == SEEDS[0]:
                fold_sizes.append(
                    {
                        "fold": fold_index,
                        "train_rows": int(len(train_indices)),
                        "validation_rows": int(len(valid_indices)),
                        "validation_original_rows": int(np.count_nonzero(valid_indices < len(train_keys))),
                        "validation_figshare_rows": int(np.count_nonzero(valid_indices >= len(train_keys))),
                    }
                )
            print(
                f"ExpA seed={seed} fold={fold_index} trained={len(train_indices)} "
                f"valid={len(valid_indices)}",
                flush=True,
            )
        if not np.isfinite(seed_oof).all():
            raise RuntimeError(f"ExpA seed {seed} produced incomplete OOF predictions")
        averaged_oof += seed_oof / len(SEEDS)

    original_prediction = averaged_oof[original_mask]
    original_target = target[original_mask]
    exp_a_mse = float(mean_squared_error(original_target, original_prediction))
    np.save(RESULTS / "exp49_gcn_figshare_oof.npy", original_prediction)
    return {
        "exp_a_mse": exp_a_mse,
        "difference_from_baseline": None,
        "relative_improvement": None,
        "fold_sizes_seed_42": fold_sizes,
        "oof_path": str((RESULTS / "exp49_gcn_figshare_oof.npy").relative_to(ROOT)),
        "averaged_oof_combined": averaged_oof,
    }


def train_final_and_submit(combined, target):
    from src.tail_research import TailRegressor

    models = [
        TailRegressor("control_log", epochs=EPOCHS).fit(
            combined.reset_index(drop=True), target, final_seed
        )
        for final_seed in (4242, 4343)
    ]
    model_path = RESULTS / "exp49_gcn_figshare_model.joblib"
    joblib.dump(
        {
            "models": models,
            "seeds": [4242, 4343],
            "epochs": EPOCHS,
            "training_rows": len(combined),
            "target": "log1p(SI VACV)",
            "groups": "canonical SMILES",
        },
        model_path,
        compress=3,
    )
    test = pd.read_csv(ROOT / "test_clean.csv", usecols=["index", "SMILES"])
    if len(test) != 597:
        raise ValueError(f"Expected 597 test rows, got {len(test)}")
    prediction = np.mean(
        [model.predict(test[["SMILES"]]) for model in models],
        axis=0,
    )
    submission = pd.DataFrame(
        {"index": test["index"].to_numpy(), "SI VACV": prediction}
    ).sort_values("index", kind="stable")
    if (
        len(submission) != 597
        or submission["index"].duplicated().any()
        or submission.isna().any().any()
        or not np.isfinite(prediction).all()
        or (prediction < 0).any()
    ):
        raise ValueError("Figshare submission failed validation")
    submission_path = ROOT / "submission_gcn_figshare.csv"
    submission.to_csv(submission_path, index=False)
    return {
        "model_path": str(model_path.relative_to(ROOT)),
        "submission_path": str(submission_path.relative_to(ROOT)),
        "rows": len(submission),
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
        "prediction_mean": float(prediction.mean()),
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS / "exp49_reproduce_and_expA.json"
    try:
        TailRegressor, cpu, folds, _ = load_friend_modules()
        train = pd.read_csv(ROOT / "train_v2.csv", usecols=["SMILES", "SI VACV"])
        target = pd.to_numeric(train["SI VACV"], errors="raise").to_numpy(dtype=float)
        if len(train) != 320 or not np.isfinite(target).all() or (target < 0).any():
            raise ValueError("Expected 320 finite nonnegative SI targets in train_v2.csv")

        print("===== Step 1: reproduce teammate baseline =====", flush=True)
        baseline = reproduce_baseline(train, target, TailRegressor, cpu, folds)
        report = {"baseline": {key: value for key, value in baseline.items() if key != "fold_ids"}}
        if baseline["status"] != "matched":
            write_json(report_path, report)
            print(json.dumps(report, indent=2))
            print("Baseline mismatch; stopping before Figshare experiment.")
            return 1
        print(
            f"Baseline MSE={baseline['baseline_mse']:.6f}; "
            f"expected={EXPECTED_BASELINE_MSE:.6f}",
            flush=True,
        )

        print("===== Step 2: Figshare experiment =====", flush=True)
        combined, combined_target, train_keys = load_figshare_and_combine(train)
        print(
            f"Combined rows={len(combined)}; original={len(train)}; Figshare=31",
            flush=True,
        )
        exp_a = run_experiment_a(
            combined,
            combined_target,
            train_keys,
            TailRegressor,
            cpu,
            folds,
        )
        exp_a["difference_from_baseline"] = (
            exp_a["exp_a_mse"] - baseline["baseline_mse"]
        )
        exp_a["relative_improvement"] = (
            baseline["baseline_mse"] - exp_a["exp_a_mse"]
        ) / baseline["baseline_mse"]
        exp_a["interpretation"] = (
            "Figshare improves the OOF MSE."
            if exp_a["exp_a_mse"] < baseline["baseline_mse"]
            else "Negative transfer: Figshare does not improve the OOF MSE."
        )
        del exp_a["averaged_oof_combined"]

        if exp_a["exp_a_mse"] < baseline["baseline_mse"] * 0.99:
            print("Improvement exceeds 1%; fitting full-data model and test submission.", flush=True)
            exp_a["final_model"] = train_final_and_submit(combined, combined_target)
        else:
            exp_a["final_model"] = None
            print("Improvement is not greater than 1%; skipping final model/submission.", flush=True)

        report["experiment_a"] = exp_a
        write_json(report_path, report)
        print(json.dumps(report, indent=2))
        return 0
    except Exception as error:
        failure = {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
        write_json(report_path, failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
