import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.model_selection import GroupKFold
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
FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def canonical(smiles):
    molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(molecule, canonical=True)


def featurize(smiles_values):
    names = [name for name, _ in Descriptors.descList]
    descriptor_rows = []
    fingerprint_rows = []
    RDLogger.DisableLog("rdApp.error")
    for row_index, smiles in enumerate(smiles_values):
        molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if molecule is None:
            raise ValueError(f"Invalid SMILES at row {row_index}: {smiles!r}")
        descriptor_rows.append([function(molecule) for _, function in Descriptors.descList])
        fingerprint_rows.append(
            FP_GENERATOR.GetFingerprintAsNumPy(molecule).astype(np.uint8)
        )
    desc = np.asarray(descriptor_rows, dtype=np.float32)
    fp = np.asarray(fingerprint_rows, dtype=np.float32)
    return np.concatenate([desc, fp], axis=1)


def model():
    return lgb.LGBMRegressor(**PARAMS)


def oof_group_cv(features, target, groups):
    folds = GroupKFold(n_splits=5)
    oof = np.full(len(target), np.nan, dtype=float)
    fold_mses = []
    for fold_id, (tr, va) in enumerate(folds.split(features, target, groups=groups)):
        fit = model()
        fit.fit(features[tr], target[tr])
        oof[va] = fit.predict(features[va])
        mse = float(mean_squared_error(target[va], oof[va]))
        fold_mses.append(mse)
        print(
            f"ChEMBL pIC50 fold={fold_id} MSE={mse:.6g}; "
            f"train={len(tr)}, val={len(va)}",
            flush=True,
        )
    if not np.isfinite(oof).all():
        raise RuntimeError("ChEMBL grouped OOF predictions are incomplete")
    return oof, fold_mses


def choose_vacv_target():
    report_path = RESULTS_DIR / "exp35outlier.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Run exp35 before exp36: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    scores = report["cv_mse_raw"]
    raw_mse = scores["raw_si"]["mse_all"]
    best_log_mse = min(
        scores[name]["mse_all"]
        for name in (
            "log1p_backtransform",
            "log1p_global_smearing",
            "log1p_fold_cross_fitted_smearing",
        )
    )
    if raw_mse < best_log_mse:
        return "raw_si", False
    return "log1p", False


def main():
    chembl_path = ROOT / "data" / "collected_raw.csv"
    if not chembl_path.is_file():
        raise FileNotFoundError(f"ChEMBL source data missing: {chembl_path}")
    chembl = pd.read_csv(chembl_path)
    chembl["IC50uM"] = pd.to_numeric(chembl["IC50uM"], errors="coerce")
    chembl = chembl.loc[
        chembl["SMILES"].notna()
        & np.isfinite(chembl["IC50uM"])
        & (chembl["IC50uM"] > 0)
    ].copy()
    chembl["canonical"] = chembl["SMILES"].map(canonical)
    chembl["pIC50"] = 6.0 - np.log10(chembl["IC50uM"].to_numpy(dtype=float))
    chembl = (
        chembl.groupby("canonical", sort=False, as_index=False)
        .agg({"SMILES": "first", "pIC50": "median", "IC50uM": "median"})
        .reset_index(drop=True)
    )
    if len(chembl) < 5:
        raise ValueError(f"Too few unique ChEMBL structures: {len(chembl)}")
    chembl_features = featurize(chembl["SMILES"].tolist())
    chembl_groups = chembl["canonical"].to_numpy()
    chembl_oof, chembl_fold_mses = oof_group_cv(
        chembl_features, chembl["pIC50"].to_numpy(dtype=float), chembl_groups
    )
    np.save(RESULTS_DIR / "exp36chembl_oof_pic50.npy", chembl_oof)

    train = pd.read_csv(ROOT / "train_clean_v2.csv")
    target = pd.to_numeric(train["SI VACV"], errors="coerce").to_numpy(dtype=float)
    folds = json.loads((RESULTS_DIR / "folds_v2.json").read_text(encoding="utf-8"))["folds"]
    features_train = np.concatenate(
        [
            pd.read_csv(ROOT / "featurestraindesc_v2.csv").to_numpy(dtype=np.float32),
            np.load(ROOT / "featurestrainfp_v2.npz")["fingerprints"].astype(np.float32),
        ],
        axis=1,
    )
    test = pd.read_csv(ROOT / "test_clean.csv")
    test_features = featurize(test["SMILES"].tolist())
    train_canonical = train["SMILES"].map(canonical).to_numpy()
    test_canonical = test["SMILES"].map(canonical).to_numpy()

    auxiliary_oof = np.full(len(train), np.nan, dtype=float)
    aux_fold_counts = []
    for fold in folds:
        tr = np.asarray(fold["train_indices"], dtype=int)
        va = np.asarray(fold["validation_indices"], dtype=int)
        excluded = set(train_canonical[va])
        keep = ~chembl["canonical"].isin(excluded).to_numpy()
        if int(keep.sum()) < 5:
            raise ValueError(f"Too few ChEMBL rows remain for VACV fold {fold['fold']}")
        auxiliary = model()
        auxiliary.fit(chembl_features[keep], chembl["pIC50"].to_numpy(dtype=float)[keep])
        auxiliary_oof[va] = auxiliary.predict(features_train[va])
        aux_fold_counts.append(
            {
                "fold": int(fold["fold"]),
                "chembl_unique_train": int(keep.sum()),
                "excluded_canonical_overlap_count": int((~keep).sum()),
            }
        )
        print(
            f"VACV fold={fold['fold']} pIC50 feature ready; "
            f"ChEMBL train={int(keep.sum())}, excluded overlap={int((~keep).sum())}",
            flush=True,
        )
    if not np.isfinite(auxiliary_oof).all():
        raise RuntimeError("Fold-safe VACV pIC50 feature is incomplete")

    selected_target, use_smearing = choose_vacv_target()
    metric_target = target if selected_target == "raw_si" else np.log1p(target)
    oof_model = np.full(len(target), np.nan, dtype=float)
    fold_mses = []
    best_iterations = []
    for fold in folds:
        tr = np.asarray(fold["train_indices"], dtype=int)
        va = np.asarray(fold["validation_indices"], dtype=int)
        augmented_train = np.column_stack([features_train[tr], auxiliary_oof[tr]])
        augmented_valid = np.column_stack([features_train[va], auxiliary_oof[va]])
        fit = model()
        fit.fit(
            augmented_train,
            metric_target[tr],
            eval_set=[(augmented_valid, metric_target[va])],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        iteration = int(fit.best_iteration_ or PARAMS["n_estimators"])
        prediction = fit.predict(augmented_valid, num_iteration=iteration)
        oof_model[va] = prediction
        best_iterations.append(iteration)
        fold_mses.append(
            float(mean_squared_error(metric_target[va], prediction))
        )
        print(
            f"VACV augmented fold={fold['fold']} transformed-target MSE={fold_mses[-1]:.6g}",
            flush=True,
        )
    if not np.isfinite(oof_model).all():
        raise RuntimeError("Augmented VACV OOF predictions are incomplete")

    if selected_target == "raw_si":
        oof_si = oof_model
        final_target = target
    else:
        oof_si = np.expm1(np.clip(oof_model, 0.0, 30.0))
        final_target = np.log1p(target)
    smearing_factor = None
    baseline_path = RESULTS_DIR / "exp08_v2.json"
    baseline_mse = float(json.loads(baseline_path.read_text(encoding="utf-8"))["cvmetric"])
    augmented_mse = float(mean_squared_error(target, oof_si))

    final_chembl_model = model().fit(
        chembl_features, chembl["pIC50"].to_numpy(dtype=float)
    )
    train_pic50_final = final_chembl_model.predict(features_train)
    test_pic50_final = final_chembl_model.predict(test_features)
    np.save(RESULTS_DIR / "exp36_vacv_pic50_oof.npy", auxiliary_oof)
    np.save(RESULTS_DIR / "exp36_vacv_pic50_full_train.npy", train_pic50_final)
    np.save(RESULTS_DIR / "exp36_test_pic50.npy", test_pic50_final)
    np.save(RESULTS_DIR / "exp36_vacv_oof_raw_si.npy", oof_si)
    pd.DataFrame(
        {"index": test["index"], "predicted_pIC50": test_pic50_final}
    ).to_csv(ROOT / "test_predicted_pic50_chembl.csv", index=False)
    result = {
        "model": "LGBM_v2_with_ChEMBL_pIC50_feature",
        "chembl_unique_structures": len(chembl),
        "chembl_source_rows": 454,
        "chembl_cv": {
            "method": "5-fold GroupKFold by canonical SMILES",
            "pic50_oof_mse": float(
                mean_squared_error(chembl["pIC50"].to_numpy(dtype=float), chembl_oof)
            ),
            "foldmetrics": chembl_fold_mses,
        },
        "vacv_cv": {
            "folds": 5,
            "leakage_control": "Auxiliary model excludes ChEMBL structures matching each validation fold.",
            "selected_vacv_target": selected_target,
            "smearing_applied": use_smearing,
            "smearing_factor": smearing_factor,
            "fold_auxiliary_fit_counts": aux_fold_counts,
            "foldmetrics_transformed_target": fold_mses,
            "best_iterations": best_iterations,
            "cvmetric": augmented_mse,
            "metricname": "MSE_raw",
            "baseline_mse_raw": baseline_mse,
            "delta_vs_baseline": augmented_mse - baseline_mse,
            "improved_over_baseline": augmented_mse < baseline_mse,
        },
        "outputs": {
            "chembl_oof_pic50": str(RESULTS_DIR / "exp36chembl_oof_pic50.npy"),
            "vacv_oof_pic50": str(RESULTS_DIR / "exp36_vacv_pic50_oof.npy"),
            "test_pic50": str(RESULTS_DIR / "exp36_test_pic50.npy"),
            "test_pic50_csv": str(ROOT / "test_predicted_pic50_chembl.csv"),
            "vacv_augmented_oof_si": str(RESULTS_DIR / "exp36_vacv_oof_raw_si.npy"),
        },
        "notes": (
            "Duplicate ChEMBL canonical structures were collapsed by median pIC50. "
            "Each VACV validation fold receives pIC50 features from a ChEMBL model "
            "that excluded exact canonical overlaps with that validation fold."
        ),
    }
    report_path = RESULTS_DIR / "exp36chembl.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"ChEMBL feature MSE={augmented_mse:.6g}; baseline={baseline_mse:.6g}; "
        f"delta={augmented_mse - baseline_mse:.6g}; target={selected_target}"
    )
    print(f"Saved result: {report_path}")


if __name__ == "__main__":
    main()
