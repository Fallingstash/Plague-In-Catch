import copy
import contextlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

from featurize_gnn import make_graph
from prep import load_and_clean_train
from train_gnn import (
    BATCH_SIZE,
    LEARNING_RATE,
    PATIENCE,
    WEIGHT_DECAY,
    GINRegressor,
    set_seed,
)


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "experiments" / "results"
SEED = 42
N_FOLDS = 5
LGBM_PARAMS = {
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
    "random_state": SEED,
    "verbose": -1,
    "n_jobs": -1,
}
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
    "random_state": SEED,
    "verbose": -1,
    "n_jobs": -1,
    "class_weight": "balanced",
}
MAX_GNN_EPOCHS = 300
FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
RUN_STATUS = []


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_step(name, function):
    log_path = RESULTS_DIR / f"{name}.log"
    started = datetime.now(timezone.utc).isoformat()
    print(f"\n===== {name} started =====", flush=True)
    try:
        result = function()
        status = "success"
        detail = None
        save_json(RESULTS_DIR / f"{name}.json", result or {"status": "success"})
        print(f"===== {name} succeeded =====", flush=True)
    except Exception as error:
        status = "failed"
        detail = f"{type(error).__name__}: {error}"
        print(detail, flush=True)
        traceback.print_exc()
        save_json(
            RESULTS_DIR / f"{name}.json",
            {
                "status": "failed",
                "error": detail,
                "traceback": traceback.format_exc(),
            },
        )
    completed = datetime.now(timezone.utc).isoformat()
    RUN_STATUS.append(
        {
            "step": name,
            "status": status,
            "error": detail,
            "started_at": started,
            "completed_at": completed,
            "log": str(log_path),
        }
    )
    return status


def load_v2_data():
    path = ROOT / "train_clean_v2.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Prepared v2 train file is missing: {path}")
    train = pd.read_csv(path)
    target = pd.to_numeric(train["SI VACV"], errors="coerce").to_numpy(dtype=float)
    labels = train["is_active"].to_numpy(dtype=int)
    if not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("V2 SI target must be finite and nonnegative")
    if not np.isin(labels, [0, 1]).all() or len(train) < N_FOLDS:
        raise ValueError("Invalid v2 activity labels or dataset size")
    if not np.array_equal(labels, train["IC50_uM_VACV"].notna().astype(int).to_numpy()):
        raise ValueError("is_active must be defined by non-missing IC50")
    inactive = labels == 0
    if not np.all(target[inactive] == 0):
        raise ValueError("Every inactive v2 molecule must have SI=0")
    return train, target, labels


def load_v2_folds(row_count):
    path = RESULTS_DIR / "folds_v2.json"
    if not path.is_file():
        raise FileNotFoundError(f"V2 folds are missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["row_count"] != row_count or data["random_state"] != SEED:
        raise ValueError("V2 fold manifest is not aligned with the prepared train")
    folds = data["folds"]
    expected = set(range(row_count))
    validation = []
    for fold in folds:
        train_indices = set(fold["train_indices"])
        valid_indices = set(fold["validation_indices"])
        if train_indices & valid_indices or train_indices | valid_indices != expected:
            raise ValueError(f"Fold {fold['fold']} does not partition v2 rows")
        validation.extend(fold["validation_indices"])
    if sorted(validation) != list(range(row_count)):
        raise ValueError("V2 folds must validate every row exactly once")
    return folds


def build_v2_features(train, folds):
    descriptor_names = [name for name, _ in Descriptors.descList]
    descriptors = []
    fingerprints = []
    valid_smiles = 0
    warnings_path = ROOT / "warnings_v2.log"
    RDLogger.DisableLog("rdApp.error")
    with warnings_path.open("w", encoding="utf-8") as warnings:
        for index, smiles in enumerate(train["SMILES"]):
            molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
            if molecule is None:
                warnings.write(f"row={index}\tsmiles={smiles!r}\terror=invalid SMILES\n")
                raise ValueError(f"Invalid train_v2 SMILES at row {index}: {smiles!r}")
            valid_smiles += 1
            descriptors.append([function(molecule) for _, function in Descriptors.descList])
            fingerprints.append(
                FP_GENERATOR.GetFingerprintAsNumPy(molecule).astype(np.uint8)
            )
    descriptors = pd.DataFrame(descriptors, columns=descriptor_names)
    fingerprints = np.asarray(fingerprints, dtype=np.uint8)
    if descriptors.shape != (len(train), 217) or fingerprints.shape != (len(train), 2048):
        raise ValueError(
            f"Unexpected v2 feature shapes: {descriptors.shape}, {fingerprints.shape}"
        )
    descriptors.to_csv(ROOT / "featurestraindesc_v2.csv", index=False)
    np.savez_compressed(
        ROOT / "featurestrainfp_v2.npz", fingerprints=fingerprints
    )
    fold_ids = np.full(len(train), -1, dtype=int)
    for fold in folds:
        fold_ids[np.asarray(fold["validation_indices"], dtype=int)] = int(fold["fold"])
    graphs = [
        make_graph(
            row,
            graph_index,
            fold_id=int(fold_ids[graph_index]),
            target_columns={"SI VACV": "y_si", "is_active": "y_active"},
        )
        for graph_index, (_, row) in enumerate(train.iterrows())
    ]
    torch.save(graphs, ROOT / "train_graphs_v2.pt")
    graph_fold_manifest = {
        "source": "experiments/results/folds_v2.json",
        "n_splits": N_FOLDS,
        "random_state": SEED,
        "fold_ids": fold_ids.tolist(),
    }
    save_json(ROOT / "train_graphs_v2_folds.json", graph_fold_manifest)

    test = pd.read_csv(ROOT / "test_clean.csv")
    test_graphs_path = ROOT / "test_graphs.pt"
    if test_graphs_path.is_file():
        test_graphs = torch.load(
            test_graphs_path, map_location="cpu", weights_only=False
        )
        if len(test_graphs) != len(test) or not np.array_equal(
            [graph.smiles for graph in test_graphs], test["SMILES"].to_numpy()
        ):
            raise ValueError("Existing test graph ordering differs from test_clean.csv")
    else:
        test_graphs = [
            make_graph(row, graph_index)
            for graph_index, (_, row) in enumerate(test.iterrows())
        ]
    torch.save(test_graphs, ROOT / "test_graphs_v2.pt")

    return {
        "rows": len(train),
        "descriptor_shape": list(descriptors.shape),
        "fingerprint_shape": list(fingerprints.shape),
        "train_graphs": len(graphs),
        "test_graphs": len(test_graphs),
        "parsed_train_smiles": valid_smiles,
        "warning_file": str(warnings_path),
    }


def step_prepare():
    source = ROOT / "train_v2.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Organizer-provided v2 data is missing: {source}")
    train = load_and_clean_train(source, v2=True)
    train.to_csv(ROOT / "train_clean_v2.csv", index=False)
    train.loc[train["is_active"] == 1].to_csv(
        ROOT / "train_active_v2.csv", index=False
    )
    labels = train["is_active"].to_numpy(dtype=int)
    target = pd.to_numeric(train["SI VACV"], errors="coerce").to_numpy(dtype=float)
    if (target < 0).any() or not np.isfinite(target).all():
        raise ValueError("V2 SI values must be finite and nonnegative")
    if not np.all(target[labels == 0] == 0):
        raise ValueError("Inactive v2 samples must have SI=0")

    folds = []
    splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold_id, (train_indices, valid_indices) in enumerate(
        splitter.split(np.arange(len(train)))
    ):
        folds.append(
            {
                "fold": fold_id,
                "train_indices": train_indices.tolist(),
                "validation_indices": valid_indices.tolist(),
            }
        )
    fold_manifest = {
        "source": "train_v2.csv",
        "row_count": len(train),
        "n_splits": N_FOLDS,
        "shuffle": True,
        "random_state": SEED,
        "folds": folds,
    }
    save_json(RESULTS_DIR / "folds_v2.json", fold_manifest)
    return {
        "source": str(source),
        "rows": len(train),
        "active": int(labels.sum()),
        "inactive": int((labels == 0).sum()),
        "zero_si": int((target == 0).sum()),
        "active_zero_si": int(((labels == 1) & (target == 0)).sum()),
        "inactive_si_all_zero": bool(np.all(target[labels == 0] == 0)),
        "fold_counts": [
            len(fold["validation_indices"]) for fold in folds
        ],
        "outputs": [
            str(ROOT / "train_clean_v2.csv"),
            str(RESULTS_DIR / "folds_v2.json"),
        ],
        "note": "Original train.csv and existing feature/fold artifacts were not replaced.",
    }


def step_features():
    train, _, _ = load_v2_data()
    folds = load_v2_folds(len(train))
    return build_v2_features(train, folds)


def load_feature_matrix():
    descriptor_path = ROOT / "featurestraindesc_v2.csv"
    fp_path = ROOT / "featurestrainfp_v2.npz"
    if not descriptor_path.is_file() or not fp_path.is_file():
        raise FileNotFoundError("Run v2 feature generation before model training")
    descriptors = pd.read_csv(descriptor_path).to_numpy(dtype=np.float32)
    with np.load(fp_path) as data:
        fingerprints = data["fingerprints"].astype(np.float32)
    features = np.concatenate([descriptors, fingerprints], axis=1)
    if features.shape[1] != 2265 or np.isinf(features).any():
        raise ValueError(f"Invalid v2 feature matrix: {features.shape}")
    return features


def step_lgbm():
    train, target, _ = load_v2_data()
    folds = load_v2_folds(len(train))
    features = load_feature_matrix()
    oof = np.full(len(train), np.nan, dtype=float)
    fold_metrics = []
    best_iterations = []
    for fold in folds:
        tr = np.asarray(fold["train_indices"], dtype=int)
        va = np.asarray(fold["validation_indices"], dtype=int)
        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            features[tr],
            target[tr],
            eval_set=[(features[va], target[va])],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        iteration = int(model.best_iteration_ or LGBM_PARAMS["n_estimators"])
        oof[va] = model.predict(features[va], num_iteration=iteration)
        mse = float(mean_squared_error(target[va], oof[va]))
        best_iterations.append(iteration)
        fold_metrics.append(mse)
        print(f"fold={fold['fold']} LGBM_v2 MSE_raw={mse:.6g}, trees={iteration}", flush=True)
    if not np.isfinite(oof).all():
        raise RuntimeError("LGBM_v2 OOF predictions are incomplete")
    mse = float(mean_squared_error(target, oof))
    np.save(RESULTS_DIR / "exp08_v2_oof_raw_si.npy", oof)
    return {
        "model": "LGBM_raw_SI_v2",
        "params": LGBM_PARAMS,
        "cvmetric": mse,
        "metricname": "MSE_raw",
        "foldmetrics": fold_metrics,
        "best_iterations": best_iterations,
        "rows": len(train),
        "target_zero_count": int((target == 0).sum()),
        "notes": "Trained on all v2 rows; target is raw SI including inactive zeros.",
    }


def step_classifier():
    train, _, labels = load_v2_data()
    folds = load_v2_folds(len(train))
    features = load_feature_matrix()
    oof = np.full(len(train), np.nan, dtype=float)
    fold_auc = []
    best_iterations = []
    for fold in folds:
        tr = np.asarray(fold["train_indices"], dtype=int)
        va = np.asarray(fold["validation_indices"], dtype=int)
        model = lgb.LGBMClassifier(**CLASSIFIER_PARAMS)
        model.fit(
            features[tr],
            labels[tr],
            eval_set=[(features[va], labels[va])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        iteration = int(model.best_iteration_ or CLASSIFIER_PARAMS["n_estimators"])
        oof[va] = model.predict_proba(features[va], num_iteration=iteration)[:, 1]
        fold_auc.append(float(roc_auc_score(labels[va], oof[va])))
        best_iterations.append(iteration)
        print(
            f"fold={fold['fold']} classifier_v2 ROC-AUC={fold_auc[-1]:.5f}, "
            f"trees={iteration}",
            flush=True,
        )
    if not np.isfinite(oof).all():
        raise RuntimeError("Classifier_v2 OOF probabilities are incomplete")
    predictions = (oof >= 0.5).astype(int)
    auc = float(roc_auc_score(labels, oof))
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    np.save(RESULTS_DIR / "exp02_v2_oof_probs.npy", oof)
    return {
        "model": "LGBMClassifier_v2",
        "params": CLASSIFIER_PARAMS,
        "cvmetric": auc,
        "metricname": "ROC_AUC",
        "foldmetrics": fold_auc,
        "best_iterations": best_iterations,
        "accuracy": float(accuracy_score(labels, predictions)),
        "confusion_matrix": matrix.tolist(),
        "rows": len(train),
    }


def graph_prediction(model, graphs, device):
    loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions.extend((model(batch).detach().cpu().numpy() * 10000.0).tolist())
    return np.asarray(predictions, dtype=float)


def train_gnn_fold(graphs, target, train_indices, valid_indices, fold_id, device):
    set_seed(SEED + fold_id)
    train_graphs = []
    for index in train_indices:
        graph = copy.copy(graphs[index])
        graph.y_norm = torch.tensor([target[index] / 10000.0], dtype=torch.float32)
        train_graphs.append(graph)
    valid_graphs = [graphs[index] for index in valid_indices]
    train_loader = DataLoader(
        train_graphs, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )
    valid_loader = DataLoader(
        valid_graphs, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )
    model = GINRegressor(input_dim=int(graphs[0].x.shape[1])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    best_mse = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    for epoch in range(1, MAX_GNN_EPOCHS + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = F.mse_loss(prediction, batch.y_norm.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_prediction = graph_prediction(model, valid_graphs, device)
        mse = float(mean_squared_error(target[valid_indices], validation_prediction))
        if mse < best_mse:
            best_mse = mse
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"fold={fold_id} epoch={epoch:03d} train_mse_norm={np.mean(losses):.6f} "
                f"val_mse_raw={mse:.6g} best={best_mse:.6g}",
                flush=True,
            )
        if wait >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError(f"GNN fold {fold_id} did not produce a checkpoint")
    model.load_state_dict(best_state)
    return graph_prediction(model, valid_graphs, device), best_epoch, best_mse


def step_gnn():
    train, target, _ = load_v2_data()
    folds = load_v2_folds(len(train))
    graph_path = ROOT / "train_graphs_v2.pt"
    if not graph_path.is_file():
        raise FileNotFoundError(f"V2 graphs are missing: {graph_path}")
    graphs = torch.load(graph_path, map_location="cpu", weights_only=False)
    if len(graphs) != len(train) or not np.array_equal(
        [graph.smiles for graph in graphs], train["SMILES"].to_numpy()
    ):
        raise ValueError("V2 training graphs do not align with train_clean_v2.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oof = np.full(len(train), np.nan, dtype=float)
    fold_metrics = []
    best_epochs = []
    for fold in folds:
        tr = np.asarray(fold["train_indices"], dtype=int)
        va = np.asarray(fold["validation_indices"], dtype=int)
        prediction, best_epoch, mse = train_gnn_fold(
            graphs, target, tr, va, int(fold["fold"]), device
        )
        oof[va] = prediction
        best_epochs.append(best_epoch)
        fold_metrics.append(mse)
        print(
            f"fold={fold['fold']} GNN_v2 best_epoch={best_epoch} MSE_raw={mse:.6g}",
            flush=True,
        )
    if not np.isfinite(oof).all():
        raise RuntimeError("GNN_v2 OOF predictions are incomplete")
    mse = float(mean_squared_error(target, oof))
    np.save(RESULTS_DIR / "exp14_v2_oof_raw_si.npy", oof)
    return {
        "model": "GINRegressor_raw_SI_v2",
        "params": {
            "num_layers": 4,
            "hidden_dim": 128,
            "dropout": 0.3,
            "target_normalization": "SI / 10000",
            "max_epochs": MAX_GNN_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "device": str(device),
        },
        "cvmetric": mse,
        "metricname": "MSE_raw",
        "foldmetrics": fold_metrics,
        "best_epochs": best_epochs,
        "rows": len(train),
    }


def step_stacking():
    train, target, labels = load_v2_data()
    folds = load_v2_folds(len(train))
    gnn = np.load(RESULTS_DIR / "exp14_v2_oof_raw_si.npy")
    lgbm = np.load(RESULTS_DIR / "exp08_v2_oof_raw_si.npy")
    probabilities = np.load(RESULTS_DIR / "exp02_v2_oof_probs.npy")
    if any(array.shape != (len(train),) or not np.isfinite(array).all() for array in (gnn, lgbm, probabilities)):
        raise ValueError("V2 stacking OOF inputs have invalid shape or values")
    meta_features = np.column_stack([gnn, lgbm, probabilities])
    oof = np.full(len(train), np.nan, dtype=float)
    fold_metrics = []
    fold_coefficients = []
    for fold in folds:
        tr = np.asarray(fold["train_indices"], dtype=int)
        va = np.asarray(fold["validation_indices"], dtype=int)
        meta = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        meta.fit(meta_features[tr], target[tr])
        oof[va] = np.maximum(meta.predict(meta_features[va]), 0.0)
        mse = float(mean_squared_error(target[va], oof[va]))
        fold_metrics.append(mse)
        fold_coefficients.append(
            {
                "fold": int(fold["fold"]),
                "ridge_coefficients": meta.named_steps["ridge"].coef_.tolist(),
                "intercept": float(meta.named_steps["ridge"].intercept_),
                "mse_raw": mse,
            }
        )
        print(f"fold={fold['fold']} stack_v2 MSE_raw={mse:.6g}", flush=True)
    if not np.isfinite(oof).all():
        raise RuntimeError("V2 stacking OOF predictions are incomplete")
    mse = float(mean_squared_error(target, oof))
    full_meta = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    full_meta.fit(meta_features, target)
    import joblib

    joblib.dump(full_meta, RESULTS_DIR / "exp30_v2_meta_model.joblib")
    np.save(RESULTS_DIR / "exp30_v2_oof_stacking.npy", oof)
    raw_blend = probabilities * (0.5 * gnn + 0.5 * lgbm)
    blend_mse = float(mean_squared_error(target, raw_blend))
    return {
        "model": "RidgeStacking_v2",
        "features": ["GNN_v2_raw_SI", "LGBM_v2_raw_SI", "Classifier_v2_P_active"],
        "feature_shape": list(meta_features.shape),
        "cvmetric": mse,
        "metricname": "MSE_raw",
        "foldmetrics": fold_metrics,
        "fold_coefficients": fold_coefficients,
        "gated_50_50_regression_blend_mse": blend_mse,
        "classifier_roc_auc": float(
            json.loads((RESULTS_DIR / "exp02_v2.json").read_text())["cvmetric"]
        ),
        "rows": len(train),
        "target_zeros": int((target == 0).sum()),
    }


def step_final_submission():
    train, target, labels = load_v2_data()
    folds = load_v2_folds(len(train))
    test = pd.read_csv(ROOT / "test_clean.csv")
    test_graphs = torch.load(
        ROOT / "test_graphs_v2.pt", map_location="cpu", weights_only=False
    )
    if len(test) != 597 or len(test_graphs) != len(test):
        raise ValueError("V2 final inference expects 597 aligned test graphs")
    if not np.array_equal(
        [graph.smiles for graph in test_graphs], test["SMILES"].to_numpy()
    ):
        raise ValueError("Test graph order does not match test_clean.csv")

    features_train = load_feature_matrix()
    descriptors_test = pd.read_csv(ROOT / "featurestestdesc.csv").to_numpy(dtype=np.float32)
    with np.load(ROOT / "featurestestfp.npz") as data:
        fingerprints_test = data["fingerprints"].astype(np.float32)
    features_test = np.concatenate([descriptors_test, fingerprints_test], axis=1)
    if features_test.shape != (len(test), 2265):
        raise ValueError(f"Unexpected test feature shape: {features_test.shape}")

    lgbm_report = json.loads((RESULTS_DIR / "exp08_v2.json").read_text(encoding="utf-8"))
    classifier_report = json.loads((RESULTS_DIR / "exp02_v2.json").read_text(encoding="utf-8"))
    gnn_report = json.loads((RESULTS_DIR / "exp14_v2.json").read_text(encoding="utf-8"))
    lgbm_params = dict(LGBM_PARAMS)
    lgbm_params["n_estimators"] = int(np.median(lgbm_report["best_iterations"]))
    lgbm_final = lgb.LGBMRegressor(**lgbm_params).fit(features_train, target)
    lgbm_test = lgbm_final.predict(features_test)

    classifier_params = dict(CLASSIFIER_PARAMS)
    classifier_params["n_estimators"] = int(np.median(classifier_report["best_iterations"]))
    classifier_final = lgb.LGBMClassifier(**classifier_params).fit(features_train, labels)
    probability_test = classifier_final.predict_proba(features_test)[:, 1]

    graph_data = torch.load(
        ROOT / "train_graphs_v2.pt", map_location="cpu", weights_only=False
    )
    best_epochs = [int(value) for value in gnn_report["best_epochs"]]
    final_epochs = max(1, int(np.median(best_epochs)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)
    train_graphs = []
    for graph, value in zip(graph_data, target):
        graph_copy = copy.copy(graph)
        graph_copy.y_norm = torch.tensor([value / 10000.0], dtype=torch.float32)
        train_graphs.append(graph_copy)
    train_loader = DataLoader(
        train_graphs, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        test_graphs, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )
    gnn_final = GINRegressor(input_dim=int(graph_data[0].x.shape[1])).to(device)
    optimizer = torch.optim.AdamW(
        gnn_final.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    for epoch in range(1, final_epochs + 1):
        gnn_final.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(gnn_final(batch), batch.y_norm.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn_final.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(
            f"final GNN epoch={epoch:03d}/{final_epochs} "
            f"train_mse_norm={np.mean(losses):.6f}",
            flush=True,
        )
    gnn_final.eval()
    gnn_test = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            gnn_test.extend((gnn_final(batch).cpu().numpy() * 10000.0).tolist())
    gnn_test = np.asarray(gnn_test, dtype=float)

    meta_features = np.column_stack([gnn_test, lgbm_test, probability_test])
    meta_model = __import__("joblib").load(RESULTS_DIR / "exp30_v2_meta_model.joblib")
    predictions = np.maximum(meta_model.predict(meta_features), 0.0)
    if predictions.shape != (len(test),) or not np.isfinite(predictions).all():
        raise RuntimeError("V2 stacked test predictions are invalid")
    submission = pd.DataFrame(
        {"index": test["index"].to_numpy(), "SI VACV": predictions}
    ).sort_values("index", kind="stable")
    if (
        len(submission) != 597
        or submission.isna().any().any()
        or (submission["SI VACV"] < 0).any()
        or submission["index"].duplicated().any()
        or not submission["index"].is_monotonic_increasing
    ):
        raise RuntimeError("V2 submission validation failed")
    output = ROOT / "submission_ml_final_v2.csv"
    submission.to_csv(output, index=False)
    return {
        "model": "RidgeStacking_v2",
        "rows": len(submission),
        "file_size_bytes": output.stat().st_size,
        "prediction_summary": {
            "min": float(submission["SI VACV"].min()),
            "max": float(submission["SI VACV"].max()),
            "median": float(submission["SI VACV"].median()),
            "mean": float(submission["SI VACV"].mean()),
        },
        "final_gnn_epochs": final_epochs,
        "final_lgbm_iterations": lgbm_params["n_estimators"],
        "final_classifier_iterations": classifier_params["n_estimators"],
        "output": str(output),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    steps = [
        ("prep_v2", step_prepare),
        ("features_v2", step_features),
        ("exp08_v2", step_lgbm),
        ("exp02_v2", step_classifier),
        ("exp14_v2", step_gnn),
        ("exp30_v2", step_stacking),
        ("submission_v2", step_final_submission),
    ]
    for name, function in steps:
        log_path = RESULTS_DIR / f"{name}.log"
        with log_path.open("w", encoding="utf-8") as log:
            output = Tee(sys.__stdout__, log)
            error_output = Tee(sys.__stderr__, log)
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error_output):
                status = run_step(name, function)
        print(f"{name}: {status}; log={log_path}", flush=True)

    summary = {
        "task": "TASK 31 train_v2 pipeline",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "steps": RUN_STATUS,
        "all_steps_succeeded": all(step["status"] == "success" for step in RUN_STATUS),
        "source_rows": len(pd.read_csv(ROOT / "train_v2.csv")),
        "outputs": [
            str(ROOT / "train_clean_v2.csv"),
            str(RESULTS_DIR / "folds_v2.json"),
            str(RESULTS_DIR / "exp08_v2.json"),
            str(RESULTS_DIR / "exp02_v2.json"),
            str(RESULTS_DIR / "exp14_v2.json"),
            str(RESULTS_DIR / "exp30_v2.json"),
            str(ROOT / "submission_ml_final_v2.csv"),
        ],
    }
    save_json(RESULTS_DIR / "run_all_upgrade_v2.json", summary)
    print(f"Pipeline complete: {json.dumps(summary, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
