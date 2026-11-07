import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"
CLASSIFIER_PARAMS = {
    "n_estimators": 500,
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
    "class_weight": "balanced",
}
REGRESSOR_PARAMS = {
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
EARLY_STOPPING_ROUNDS = 50


def load_expanded_data():
    expanded_path = ROOT / "traincleanexpanded.csv"
    chembl_path = ROOT / "data" / "collected_raw.csv"
    for path in (expanded_path, chembl_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required expanded data is missing: {path}")

    train = pd.read_csv(expanded_path)
    chembl = pd.read_csv(chembl_path)
    expanded_desc = pd.read_csv(ROOT / "featurestraindesc_expanded.csv")
    expanded_fp_path = ROOT / "featurestrainfp_expanded.npz"
    classifier_desc = pd.read_csv(ROOT / "featurestrainclassifierdesc_expanded.csv")
    classifier_fp_path = ROOT / "featurestrainclassifierfp_expanded.npz"
    for path in (expanded_fp_path, classifier_fp_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required expanded features are missing: {path}")

    with np.load(expanded_fp_path) as data:
        expanded_fp = data["fingerprints"]
    with np.load(classifier_fp_path) as data:
        chembl_fp = data["fingerprints"]
    expected_descriptor_count = len(expanded_desc.columns)
    if expanded_desc.shape != (349, expected_descriptor_count):
        raise ValueError(f"Unexpected expanded descriptor shape: {expanded_desc.shape}")
    if expanded_fp.shape != (349, 2048):
        raise ValueError(f"Unexpected expanded fingerprint shape: {expanded_fp.shape}")
    if len(train) != 349 or not np.array_equal(
        train["SMILES"].to_numpy(), pd.read_csv(ROOT / "featurestrainmeta_expanded.csv")["SMILES"].to_numpy()
    ):
        raise ValueError("Expanded features and training rows are misaligned")
    if len(chembl) != 454 or classifier_desc.shape[0] != len(chembl):
        raise ValueError("ChEMBL features and classifier rows are misaligned")
    if chembl_fp.shape != (len(chembl), 2048):
        raise ValueError(f"Unexpected ChEMBL fingerprint shape: {chembl_fp.shape}")
    if list(expanded_desc.columns) != list(classifier_desc.columns):
        raise ValueError("Expanded and ChEMBL descriptor columns are not aligned")

    x_base = np.concatenate(
        [expanded_desc.to_numpy(dtype=np.float32), expanded_fp.astype(np.float32)],
        axis=1,
    )
    x_chembl = np.concatenate(
        [classifier_desc.to_numpy(dtype=np.float32), chembl_fp.astype(np.float32)],
        axis=1,
    )
    if x_base.shape != (349, expected_descriptor_count + 2048):
        raise ValueError(f"Unexpected expanded feature matrix shape: {x_base.shape}")
    if x_chembl.shape != (454, expected_descriptor_count + 2048):
        raise ValueError(f"Unexpected ChEMBL feature matrix shape: {x_chembl.shape}")
    if np.isinf(x_base).any() or np.isinf(x_chembl).any():
        raise ValueError("Feature matrices contain infinite values")

    labels_base = train["is_active"].to_numpy(dtype=int)
    target_log = train["log10_SI"].to_numpy(dtype=float)
    target_raw = pd.to_numeric(train["SI VACV"], errors="coerce").to_numpy(dtype=float)
    labels_chembl = pd.to_numeric(chembl["IC50uM"], errors="coerce").notna().to_numpy(dtype=int)
    if labels_base.sum() != 268 or labels_chembl.sum() != len(chembl):
        raise ValueError("Unexpected active counts in expanded training sources")
    if not np.isfinite(target_log[labels_base == 1]).all():
        raise ValueError("Expanded active log targets contain missing or non-finite values")
    canonical_base = np.asarray(
        [canonical_smiles(smiles) for smiles in train["SMILES"]],
        dtype=object,
    )
    canonical_chembl = np.asarray(
        [canonical_smiles(smiles) for smiles in chembl["SMILES"]],
        dtype=object,
    )
    return {
        "train": train,
        "chembl": chembl,
        "x_base": x_base,
        "x_chembl": x_chembl,
        "x_classifier": np.concatenate([x_base, x_chembl], axis=0),
        "labels_base": labels_base,
        "labels_classifier": np.concatenate([labels_base, labels_chembl]),
        "target_log": target_log,
        "target_raw": target_raw,
        "active_base": labels_base == 1,
        "canonical_base": canonical_base,
        "canonical_classifier": np.concatenate([canonical_base, canonical_chembl]),
    }


def canonical_smiles(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def load_fold_artifact():
    path = RESULTS_DIR / "folds_expanded.json"
    if not path.is_file():
        raise FileNotFoundError(f"Expanded fold assignments are missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def fold_records(artifact, key):
    folds = artifact[key]
    n_rows = len(artifact["training_349_fold_ids"] if key == "training_349_folds" else artifact["classifier_803_fold_ids"])
    validations = []
    for fold_id, fold in enumerate(folds):
        if fold["fold"] != fold_id:
            raise ValueError("Expanded fold IDs must be ordered from 0 to 4")
        train_indices = set(fold["train_indices"])
        valid_indices = set(fold["validation_indices"])
        expected = set(range(n_rows))
        if train_indices & valid_indices or train_indices | valid_indices != expected:
            raise ValueError(f"Expanded fold {fold_id} does not partition all rows")
        validations.extend(fold["validation_indices"])
    if sorted(validations) != list(range(n_rows)):
        raise ValueError("Expanded folds must validate every row exactly once")
    return folds


def purge_canonical_duplicates(train_indices, valid_indices, canonical_keys):
    valid_canonical = {
        canonical_keys[index]
        for index in valid_indices
        if canonical_keys[index] is not None
    }
    keep_mask = np.asarray(
        [
            canonical_keys[index] is None or canonical_keys[index] not in valid_canonical
            for index in train_indices
        ],
        dtype=bool,
    )
    kept = train_indices[keep_mask]
    return kept, int(len(train_indices) - len(kept))


def write_result(filename, result):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path
