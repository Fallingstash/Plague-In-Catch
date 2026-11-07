import hashlib
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error


ROOT = Path(__file__).resolve().parent
FRIEND = ROOT / "from_friend"
RESULTS = ROOT / "experiments" / "results"
FRIEND_OOF = FRIEND / "results" / "isotonic_gcn_7357_2643" / "gcn_oof.csv"
FRIEND_BUNDLE = FRIEND / "results" / "isotonic_gcn_7357_2643" / "prediction_bundle.joblib"


def canonical(smiles):
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def mse(target, prediction):
    return float(mean_squared_error(target, prediction))


def fit_simplex_weights(features, target):
    def objective(weights):
        error = features @ weights - target
        return float(error @ error / len(target))

    def gradient(weights):
        return 2 * features.T @ (features @ weights - target) / len(target)

    result = minimize(
        objective,
        np.array([1.0, 0.0, 0.0]),
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * 3,
        constraints=[
            {
                "type": "eq",
                "fun": lambda weights: weights.sum() - 1.0,
                "jac": lambda weights: np.ones(3),
            }
        ],
        options={"maxiter": 1000, "ftol": 1e-14},
    )
    if not result.success:
        raise RuntimeError(f"Simplex optimization failed: {result.message}")
    return result.x


def load_aligned_oof():
    train_path = ROOT / "train_v2.csv"
    clean_path = ROOT / "train_clean_v2.csv"
    friend_train_path = FRIEND / "train.csv"
    train = pd.read_csv(train_path, usecols=["SMILES", "SI VACV"])
    clean = pd.read_csv(clean_path, usecols=["SMILES", "SI VACV"])
    friend_train = pd.read_csv(friend_train_path, usecols=["SMILES", "SI VACV"])
    target = train["SI VACV"].to_numpy(dtype=float)

    if len(train) != 320 or not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("Expected 320 finite nonnegative official v2 targets")
    if len(clean) != len(train) or len(friend_train) != len(train):
        raise ValueError("Training files have different row counts")

    train_keys = [(canonical(s), float(y)) for s, y in zip(train.SMILES, target)]
    clean_keys = [
        (canonical(s), float(y))
        for s, y in zip(clean.SMILES, clean["SI VACV"].to_numpy(dtype=float))
    ]
    friend_keys = [
        (canonical(s), float(y))
        for s, y in zip(
            friend_train.SMILES,
            friend_train["SI VACV"].to_numpy(dtype=float),
        )
    ]
    if train_keys != clean_keys:
        raise ValueError("train_clean_v2.csv is not aligned with train_v2.csv")

    friend_rows = defaultdict(deque)
    for row, key in enumerate(friend_keys):
        friend_rows[key].append(row)
    row_map = []
    for key in train_keys:
        if not friend_rows[key]:
            raise ValueError(f"No matching teammate row for canonical structure {key[0]}")
        row_map.append(friend_rows[key].popleft())
    if any(rows for rows in friend_rows.values()):
        raise ValueError("Teammate train has unmatched rows")

    frame = pd.read_csv(FRIEND_OOF)
    required = {"row", "fold", "true", "gcn_oof_seed_average"}
    if not required.issubset(frame.columns) or len(frame) != len(friend_train):
        raise ValueError("Unexpected teammate OOF schema or row count")
    if frame["row"].duplicated().any() or set(frame["row"]) != set(range(len(friend_train))):
        raise ValueError("Teammate OOF row indices are incomplete or duplicated")
    frame = frame.set_index("row")
    mapped_rows = np.asarray(row_map, dtype=int)
    friend_target = frame.loc[mapped_rows, "true"].to_numpy(dtype=float)
    if not np.array_equal(friend_target, target):
        raise ValueError("Teammate OOF targets do not match aligned train targets")
    friend_gcn = frame.loc[mapped_rows, "gcn_oof_seed_average"].to_numpy(dtype=float)
    if not np.isfinite(friend_gcn).all():
        raise ValueError("Teammate GCN OOF predictions contain invalid values")

    lgbm = np.load(RESULTS / "exp08_v2_oof_raw_si.npy")
    gin = np.load(RESULTS / "exp14_v2_oof_raw_si.npy")
    if any(values.shape != target.shape or not np.isfinite(values).all() for values in (lgbm, gin)):
        raise ValueError("Our v2 OOF arrays have invalid shape or values")
    return train, target, friend_gcn, lgbm, gin, frame.loc[mapped_rows, "fold"].to_numpy(int)


def evaluate_oof(target, gcn, lgbm, gin, folds):
    predictions = {
        "friend_gcn": gcn,
        "our_lgbm": lgbm,
        "our_gin": gin,
        "gcn_lgbm_50_50": (gcn + lgbm) / 2,
        "gcn_gin_50_50": (gcn + gin) / 2,
        "equal_thirds": (gcn + lgbm + gin) / 3,
    }
    residuals = [gcn - target, lgbm - target, gin - target]
    feature_matrix = np.column_stack([gcn, lgbm, gin]) / 10000.0
    scaled_target = target / 10000.0
    crossfit_prediction = np.full(len(target), np.nan)
    weight_report = []

    for fold in folds:
        train_indices = np.asarray(fold["train_indices"], dtype=int)
        valid_indices = np.asarray(fold["validation_indices"], dtype=int)
        weights = fit_simplex_weights(
            feature_matrix[train_indices],
            scaled_target[train_indices],
        )
        crossfit_prediction[valid_indices] = (
            feature_matrix[valid_indices] @ weights
        ) * 10000.0
        weight_report.append(
            {
                "fold": int(fold["fold"]),
                "weights_gcn_lgbm_gin": weights.tolist(),
                "validation_mse": mse(target[valid_indices], crossfit_prediction[valid_indices]),
            }
        )
    if not np.isfinite(crossfit_prediction).all():
        raise ValueError("Cross-fit ensemble predictions are incomplete")
    predictions["crossfit_optimized"] = crossfit_prediction

    full_weights = fit_simplex_weights(feature_matrix, scaled_target)
    return {
        "mse": {name: mse(target, prediction) for name, prediction in predictions.items()},
        "residual_correlation": {
            "friend_gcn_vs_our_lgbm": float(np.corrcoef(residuals[0], residuals[1])[0, 1]),
            "friend_gcn_vs_our_gin": float(np.corrcoef(residuals[0], residuals[2])[0, 1]),
        },
        "crossfit_weights": weight_report,
        "full_oof_weights_descriptive_only": full_weights.tolist(),
        "selected_test_weights_gcn_lgbm_gin": [1.0, 0.0, 0.0],
        "selected_model_reason": "GCN has the lowest standalone OOF MSE; tested blends did not improve it.",
    }


def create_submission():
    if not FRIEND_BUNDLE.is_file():
        raise FileNotFoundError(f"Missing teammate prediction bundle: {FRIEND_BUNDLE}")
    sys.path.insert(0, str(FRIEND))
    bundle = joblib.load(FRIEND_BUNDLE)
    if bundle.get("model_type") != "isotonic_gcn_blend" or len(bundle.get("gcn_models", [])) != 2:
        raise ValueError("Unexpected teammate GCN inference bundle")

    test = pd.read_csv(ROOT / "test_clean.csv", usecols=["index", "SMILES"])
    friend_test = pd.read_csv(FRIEND / "test.csv", usecols=["index", "SMILES"])
    if len(test) != 597 or len(friend_test) != len(test):
        raise ValueError("Expected 597 aligned test structures")
    if not np.array_equal(test["index"].to_numpy(), friend_test["index"].to_numpy()):
        raise ValueError("Teammate and local test indices do not match")
    if [canonical(s) for s in test.SMILES] != [canonical(s) for s in friend_test.SMILES]:
        raise ValueError("Teammate and local test structures do not match")

    prediction_frame = test[["SMILES"]].copy()
    components = [model.predict(prediction_frame) for model in bundle["gcn_models"]]
    prediction = np.mean(components, axis=0)
    if prediction.shape != (597,) or not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError("Invalid teammate GCN test predictions")

    submission = pd.DataFrame(
        {"index": test["index"].to_numpy(), "SI VACV": prediction}
    ).sort_values("index", kind="stable")
    if (
        len(submission) != 597
        or submission.isna().any().any()
        or submission["index"].duplicated().any()
        or (submission["SI VACV"] < 0).any()
    ):
        raise ValueError("Submission validation failed")
    output = ROOT / "submission_ensemble_gcn.csv"
    submission.to_csv(output, index=False)
    return {
        "path": str(output.relative_to(ROOT)),
        "rows": len(submission),
        "prediction_min": float(submission["SI VACV"].min()),
        "prediction_max": float(submission["SI VACV"].max()),
        "prediction_mean": float(submission["SI VACV"].mean()),
        "weights_gcn_lgbm_gin": [1.0, 0.0, 0.0],
    }


def main():
    for path in (FRIEND_OOF, FRIEND_BUNDLE, ROOT / "train_v2.csv", ROOT / "test_clean.csv"):
        if not path.is_file():
            raise FileNotFoundError(f"Required audit input is missing: {path}")
    train, target, gcn, lgbm, gin, friend_folds = load_aligned_oof()
    folds_manifest = json.loads((RESULTS / "folds_v2.json").read_text(encoding="utf-8"))
    if folds_manifest["row_count"] != len(target) or len(folds_manifest["folds"]) != 5:
        raise ValueError("Unexpected v2 fold manifest")
    if set(friend_folds) != {1, 2, 3, 4, 5}:
        raise ValueError("Teammate OOF fold assignment is incomplete")

    oof_report = evaluate_oof(target, gcn, lgbm, gin, folds_manifest["folds"])
    submission_report = create_submission()
    source_code = (FRIEND / "competition_data.py").read_text(encoding="utf-8-sig")
    if 'FORBIDDEN_PREFIXES = ("ic50", "cc50")' not in source_code or "usecols=allowed" not in source_code:
        raise ValueError("Could not verify teammate hidden-measurement exclusion")
    report = {
        "training_rows": len(train),
        "teammate_oof_rows": len(gcn),
        "canonical_row_alignment": "matched by canonical SMILES and SI; duplicate keys consumed one-to-one",
        "training_sha256": hashlib.sha256((FRIEND / "train.csv").read_bytes()).hexdigest(),
        "architecture": {
            "model": "GraphNet used by TailRegressor(control_log)",
            "graph": "one normalized adjacency aggregation layer; 10 atom features; untyped bonds",
            "pooling": "masked mean pooling",
            "head": "Linear(64,64) + GELU + Dropout(0.2) + Linear(64,1)",
            "additional_features": "Morgan fingerprints, MACCS keys, standardized RDKit descriptors",
            "target": "log1p(SI VACV), decoded with expm1",
            "oof_protocol": "canonical-grouped 5-fold OOF averaged over seeds 42 and 43",
        },
        "leakage_audit": {
            "uses_test_IC50_CC50_columns": False,
            "evidence": "competition_data.read_competition_csv drops ic50/cc50-prefixed columns; current GCN pipeline trains on SMILES and SI VACV only.",
            "historical_caveat": "The project audit documents an older test-max-based sweep; it is excluded from the current documented selection protocol.",
        },
        "oof": oof_report,
        "submission": submission_report,
    }
    report_path = RESULTS / "friend_ensemble_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
