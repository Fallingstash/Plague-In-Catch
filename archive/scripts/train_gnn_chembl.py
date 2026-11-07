import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool

from featurize_gnn import make_graph
from train_gnn import BATCH_SIZE, DROPOUT, HIDDEN_DIM, LEARNING_RATE, NUM_LAYERS, WEIGHT_DECAY, set_seed


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "experiments" / "results"
PRETRAIN_EPOCHS = 50
FROZEN_HEAD_EPOCHS = 20
FINETUNE_EPOCHS = 20
FINETUNE_LEARNING_RATE = 0.0001


class GINEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.convolutions = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        for layer_index in range(num_layers):
            layer_input_dim = input_dim if layer_index == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(layer_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convolutions.append(GINConv(mlp))
            self.normalizations.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout

    def forward(self, graph):
        x = graph.x
        for convolution, normalization in zip(self.convolutions, self.normalizations):
            x = convolution(x, graph.edge_index)
            x = normalization(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, graph.batch)


class GINRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.encoder = GINEncoder(input_dim, hidden_dim=hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, graph):
        return self.head(self.encoder(graph)).view(-1)


def canonical_smiles(smiles):
    molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if molecule is None:
        raise ValueError(f"Invalid ChEMBL SMILES: {smiles!r}")
    return Chem.MolToSmiles(molecule, canonical=True)


def load_chembl():
    path = ROOT / "data" / "collected_raw.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Required ChEMBL data is missing: {path}")
    data = pd.read_csv(path)
    required = {"SMILES", "IC50uM"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"ChEMBL data is missing columns: {sorted(missing)}")
    data["IC50uM"] = pd.to_numeric(data["IC50uM"], errors="coerce")
    data = data.loc[
        data["SMILES"].notna()
        & np.isfinite(data["IC50uM"])
        & (data["IC50uM"] > 0)
    ].copy()
    if len(data) != 454:
        raise ValueError(f"Expected 454 usable ChEMBL rows, got {len(data)}")
    data["canonical"] = data["SMILES"].map(canonical_smiles)
    data["pIC50"] = 6.0 - np.log10(data["IC50uM"].to_numpy(dtype=float))

    graphs = [
        make_graph(row, i, target_columns={"pIC50": "y_pic50"})
        for i, (_, row) in enumerate(data.iterrows())
    ]
    torch.save(graphs, ROOT / "chembl_graphs.pt")

    unique_data = (
        data.groupby("canonical", sort=False, as_index=False)
        .agg({"SMILES": "first", "pIC50": "median"})
        .reset_index(drop=True)
    )
    unique_graphs = [
        make_graph(row, i, target_columns={"pIC50": "y_pic50"})
        for i, (_, row) in enumerate(unique_data.iterrows())
    ]
    return data, unique_data, unique_graphs


def train_pretraining(chembl_graphs, seed, device, label):
    set_seed(seed)
    loader = DataLoader(
        chembl_graphs, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )
    model = GINRegressor(input_dim=int(chembl_graphs[0].x.shape[1])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        model.train()
        losses = []
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            target = batch.y_pic50.view(-1)
            loss = F.mse_loss(prediction, target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch in (1, 10, 25, 50):
            print(
                f"{label} pretrain epoch={epoch:02d}/{PRETRAIN_EPOCHS} "
                f"train_mse={np.mean(losses):.6f}",
                flush=True,
            )
    return {
        key: value.detach().cpu().clone()
        for key, value in model.encoder.state_dict().items()
    }


def predict_raw_si(model, graphs, scaler, device):
    loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            normalized = model(batch).detach().cpu().numpy().reshape(-1, 1)
            predictions.extend(scaler.inverse_transform(normalized).ravel().tolist())
    return np.asarray(predictions, dtype=float)


def train_si_model(model, train_graphs, normalized_targets, epochs, optimizer, device, label):
    loader = DataLoader(
        train_graphs, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )
    for epoch in range(1, epochs + 1):
        model.train()
        if label.endswith("frozen-head"):
            model.encoder.eval()
        losses = []
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = F.mse_loss(prediction, batch.y_norm.view(-1))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(
            f"{label} epoch={epoch:02d}/{epochs} train_mse_norm={np.mean(losses):.6f}",
            flush=True,
        )


def main():
    data, unique_data, unique_graphs = load_chembl()
    train_path = ROOT / "train_clean.csv"
    graphs_path = ROOT / "train_graphs.pt"
    folds_path = RESULTS_DIR / "folds.json"
    fold_ids_path = ROOT / "train_graphs_folds.json"
    for path in (train_path, graphs_path, folds_path, fold_ids_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")

    train = pd.read_csv(train_path)
    targets = pd.to_numeric(train["SI VACV"], errors="coerce").to_numpy(dtype=float)
    active = pd.to_numeric(train["is_active"], errors="coerce").to_numpy(dtype=int) == 1
    train_graphs = torch.load(graphs_path, map_location="cpu", weights_only=False)
    fold_data = json.loads(folds_path.read_text(encoding="utf-8"))
    fold_manifest = json.loads(fold_ids_path.read_text(encoding="utf-8"))
    folds = fold_data["folds"]
    fold_ids = np.asarray(fold_manifest["fold_ids"], dtype=int)
    if len(train) != 318 or len(train_graphs) != 318 or targets.shape != (318,):
        raise ValueError("Expected 318 aligned training rows, labels, and graphs")
    if not np.isfinite(targets).all() or (targets <= 0).any():
        raise ValueError("Training SI values must be finite and positive")
    if fold_ids.shape != (318,) or fold_manifest["source"] != "experiments/results/folds.json":
        raise ValueError("GNN fold manifest does not reference the saved original folds")
    if not np.array_equal(
        [graph.smiles for graph in train_graphs], train["SMILES"].to_numpy()
    ):
        raise ValueError("Training graph order does not match train_clean.csv")
    if not np.array_equal(
        fold_ids, np.asarray([int(graph.fold_id) for graph in train_graphs])
    ):
        raise ValueError("GNN graph fold IDs do not match the fold manifest")
    if sorted(np.concatenate([fold["validation_indices"] for fold in folds]).tolist()) != list(range(318)):
        raise ValueError("Saved folds must validate every training row exactly once")
    for fold in folds:
        expected_validation = np.flatnonzero(fold_ids == int(fold["fold"]))
        if not np.array_equal(
            np.sort(np.asarray(fold["validation_indices"], dtype=int)),
            expected_validation,
        ):
            raise ValueError(f"Fold {fold['fold']} does not match the saved graph fold IDs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"ChEMBL rows={len(data)}, unique canonical molecules={len(unique_data)}, "
        f"train active={int(active.sum())}; device={device}",
        flush=True,
    )
    full_encoder_state = train_pretraining(
        unique_graphs, seed=42, device=device, label="full-data"
    )
    torch.save(
        {
            "encoder_state_dict": full_encoder_state,
            "input_dim": int(unique_graphs[0].x.shape[1]),
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "pretrain_epochs": PRETRAIN_EPOCHS,
            "unique_canonical_molecules": len(unique_graphs),
            "target": "pIC50 = 6 - log10(IC50_uM)",
        },
        ROOT / "gnn_encoder_pretrained.pt",
    )

    chembl_by_canonical = {
        row.canonical: graph
        for row, graph in zip(unique_data.itertuples(index=False), unique_graphs)
    }
    oof = np.full(len(train), np.nan, dtype=float)
    fold_mses = []
    fold_pretrain_counts = []
    input_dim = int(train_graphs[0].x.shape[1])

    for fold in folds:
        fold_id = int(fold["fold"])
        train_indices = np.asarray(fold["train_indices"], dtype=int)
        valid_indices = np.asarray(fold["validation_indices"], dtype=int)
        active_train_indices = train_indices[active[train_indices]]
        if not len(active_train_indices):
            raise ValueError(f"Fold {fold_id} has no active training rows")

        validation_canonical = {
            canonical_smiles(train.iloc[index]["SMILES"]) for index in valid_indices
        }
        fold_chembl_graphs = [
            graph
            for canonical, graph in chembl_by_canonical.items()
            if canonical not in validation_canonical
        ]
        if not fold_chembl_graphs:
            raise RuntimeError(f"No ChEMBL graphs available for fold {fold_id}")
        fold_pretrain_counts.append(len(fold_chembl_graphs))
        fold_encoder_state = train_pretraining(
            fold_chembl_graphs,
            seed=4200 + fold_id,
            device=device,
            label=f"fold-{fold_id}",
        )

        scaler = RobustScaler().fit(targets[active_train_indices].reshape(-1, 1))
        normalized_targets = scaler.transform(targets.reshape(-1, 1)).ravel()
        fold_train_graphs = []
        for index in active_train_indices:
            graph = copy.copy(train_graphs[index])
            graph.y_norm = torch.tensor([normalized_targets[index]], dtype=torch.float32)
            fold_train_graphs.append(graph)
        fold_validation_graphs = [train_graphs[index] for index in valid_indices]

        model = GINRegressor(input_dim=input_dim).to(device)
        model.encoder.load_state_dict(fold_encoder_state, strict=True)
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
        head_optimizer = torch.optim.Adam(
            model.head.parameters(), lr=LEARNING_RATE
        )
        train_si_model(
            model,
            fold_train_graphs,
            normalized_targets,
            FROZEN_HEAD_EPOCHS,
            head_optimizer,
            device,
            f"fold-{fold_id} frozen-head",
        )

        for parameter in model.encoder.parameters():
            parameter.requires_grad = True
        finetune_optimizer = torch.optim.Adam(
            model.parameters(), lr=FINETUNE_LEARNING_RATE
        )
        train_si_model(
            model,
            fold_train_graphs,
            normalized_targets,
            FINETUNE_EPOCHS,
            finetune_optimizer,
            device,
            f"fold-{fold_id} fine-tune",
        )

        predictions = predict_raw_si(model, fold_validation_graphs, scaler, device)
        oof[valid_indices] = predictions
        fold_mse = float(mean_squared_error(targets[valid_indices], predictions))
        fold_mses.append(fold_mse)
        print(
            f"fold={fold_id} OOF_MSE_raw={fold_mse:.6g}; "
            f"pretrain_unique_molecules={len(fold_chembl_graphs)}",
            flush=True,
        )

    if not np.isfinite(oof).all():
        raise RuntimeError("Fine-tuned GNN OOF predictions are incomplete or non-finite")
    mse_raw = float(mean_squared_error(targets, oof))
    np.save(RESULTS_DIR / "exp28oof_raw_si.npy", oof)

    result = {
        "model": "GIN_ChEMBL_pretrained_finetuned_raw_SI",
        "params": {
            "encoder": "4-layer GINConv, hidden_dim=128, dropout=0.3",
            "pretraining_target": "pIC50 = -log10(IC50_uM * 1e-6)",
            "pretraining_epochs": PRETRAIN_EPOCHS,
            "pretraining_optimizer": "Adam",
            "pretraining_learning_rate": LEARNING_RATE,
            "chembl_raw_rows": len(data),
            "chembl_unique_canonical_molecules": len(unique_data),
            "fine_tuning_target": "raw SI, RobustScaler fit on active fold-training rows",
            "frozen_head_epochs": FROZEN_HEAD_EPOCHS,
            "unfrozen_fine_tuning_epochs": FINETUNE_EPOCHS,
            "fine_tuning_optimizer": "Adam",
            "fine_tuning_learning_rate": FINETUNE_LEARNING_RATE,
            "active_training_rows_per_fold": [int((active[np.asarray(f["train_indices"], dtype=int)]).sum()) for f in folds],
            "fold_pretraining_unique_molecules": fold_pretrain_counts,
            "folds": "experiments/results/folds.json",
            "seed": 42,
        },
        "cvmetric": mse_raw,
        "metricname": "MSE_raw",
        "foldmetrics": fold_mses,
        "notes": (
            "Fold-specific pretraining excludes ChEMBL canonical structures matching "
            "each validation fold to avoid exact-molecule target leakage. ChEMBL "
            "replicates were collapsed to the median pIC50 per canonical SMILES. "
            "OOF validation includes all rows; SI fine-tuning uses active training rows only."
        ),
    }
    result_path = RESULTS_DIR / "exp_28_gnn_pretrained.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"exp28 MSE_raw: {mse_raw:.6g}")
    exp14_path = RESULTS_DIR / "exp14gnn.json"
    if exp14_path.is_file():
        exp14_mse = float(json.loads(exp14_path.read_text(encoding="utf-8"))["cvmetric"])
        print(f"exp14 MSE_raw: {exp14_mse:.6g}; delta: {mse_raw - exp14_mse:.6g}")
    ensemble_path = RESULTS_DIR / "exp22ensemble_stability.json"
    if ensemble_path.is_file():
        ensemble_mse = float(
            json.loads(ensemble_path.read_text(encoding="utf-8"))["seed42_oof_optimized_mse"]
        )
        print(f"Ensemble OOF MSE_raw: {ensemble_mse:.6g}; delta: {mse_raw - ensemble_mse:.6g}")
    print(f"Saved result: {result_path}")


if __name__ == "__main__":
    main()
