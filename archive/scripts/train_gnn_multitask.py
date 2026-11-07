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
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch_geometric.loader import DataLoader

from featurize_gnn import make_graph
from train_gnn import (
    BATCH_SIZE,
    DROPOUT,
    HIDDEN_DIM,
    LEARNING_RATE,
    NUM_LAYERS,
    WEIGHT_DECAY,
    set_seed,
)
from torch_geometric.nn import GINConv, global_mean_pool


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "experiments" / "results"
SEED = 42
MULTITASK_EPOCHS = 40
VACV_FINETUNE_EPOCHS = 10
FINETUNE_LEARNING_RATE = 0.0001
PATIENCE = 10
BATCH_SIZE_MULTITASK = 32


class SharedGINEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.convolutions = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.dropout = dropout
        for layer_index in range(num_layers):
            layer_input = input_dim if layer_index == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(layer_input, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convolutions.append(GINConv(mlp))
            self.normalizations.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, graph):
        x = graph.x
        for convolution, normalization in zip(self.convolutions, self.normalizations):
            x = convolution(x, graph.edge_index)
            x = normalization(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, graph.batch)


class MultiTaskGIN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = SharedGINEncoder(input_dim)
        self.pic50_head = nn.Linear(HIDDEN_DIM, 1)
        self.logsi_head = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, graph):
        representation = self.encoder(graph)
        return (
            self.pic50_head(representation).view(-1),
            self.logsi_head(representation).view(-1),
        )


def canonical(smiles):
    molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(molecule, canonical=True)


def load_chembl_unique():
    table = pd.read_csv(ROOT / "data" / "collected_raw.csv")
    table["IC50uM"] = pd.to_numeric(table["IC50uM"], errors="coerce")
    table = table.loc[
        table["SMILES"].notna()
        & np.isfinite(table["IC50uM"])
        & (table["IC50uM"] > 0)
    ].copy()
    table["canonical"] = table["SMILES"].map(canonical)
    table["pIC50"] = 6.0 - np.log10(table["IC50uM"].to_numpy(dtype=float))
    unique = (
        table.groupby("canonical", sort=False, as_index=False)
        .agg({"SMILES": "first", "pIC50": "median"})
        .reset_index(drop=True)
    )
    graphs = [
        make_graph(row, i, target_columns={"pIC50": "y_pic50"})
        for i, (_, row) in enumerate(unique.iterrows())
    ]
    if len(graphs) < 5:
        raise ValueError(f"Too few unique ChEMBL structures: {len(graphs)}")
    return unique, graphs


def make_dataset(graphs, target, attribute):
    dataset = []
    for graph, value in zip(graphs, target):
        item = copy.copy(graph)
        setattr(item, attribute, torch.tensor([float(value)], dtype=torch.float32))
        dataset.append(item)
    return dataset


def make_train_loader(dataset, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE_MULTITASK,
        shuffle=shuffle,
        drop_last=False,
    )


def cycle_batches(loader):
    while True:
        for batch in loader:
            yield batch


def train_multitask_epoch(model, chem_loader, vacv_loader, optimizer, device, lambda_pic50=1.0, lambda_logsi=1.0):
    steps = max(len(chem_loader), len(vacv_loader))
    chem_iter = cycle_batches(chem_loader)
    vacv_iter = cycle_batches(vacv_loader)
    losses = []
    model.train()
    for _ in range(steps):
        chem_batch = next(chem_iter).to(device)
        vacv_batch = next(vacv_iter).to(device)
        optimizer.zero_grad(set_to_none=True)
        predicted_pic50, _ = model(chem_batch)
        _, predicted_logsi = model(vacv_batch)
        loss_pic50 = F.mse_loss(predicted_pic50, chem_batch.y_pic50.view(-1))
        loss_logsi = F.mse_loss(predicted_logsi, vacv_batch.y_logsi.view(-1))
        loss = lambda_pic50 * loss_pic50 + lambda_logsi * loss_logsi
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(
            (
                float(loss.detach().cpu()),
                float(loss_pic50.detach().cpu()),
                float(loss_logsi.detach().cpu()),
            )
        )
    return np.mean(np.asarray(losses), axis=0)


def predict_logsi(model, graphs, scaler, device):
    loader = make_train_loader(graphs, shuffle=False)
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, normalized = model(batch)
            raw_scaled = normalized.detach().cpu().numpy().reshape(-1, 1)
            predictions.extend(scaler.inverse_transform(raw_scaled).ravel().tolist())
    return np.asarray(predictions, dtype=float)


def main():
    results_dir = RESULTS_DIR
    train = pd.read_csv(ROOT / "train_clean_v2.csv")
    target_si = pd.to_numeric(train["SI VACV"], errors="coerce").to_numpy(dtype=float)
    if len(train) != 320 or not np.isfinite(target_si).all() or (target_si < 0).any():
        raise ValueError(f"Expected 320 valid v2 train rows, got {len(train)}")
    target_logsi = np.log1p(target_si)
    train_graphs = torch.load(
        ROOT / "train_graphs_v2.pt", map_location="cpu", weights_only=False
    )
    folds_data = json.loads((results_dir / "folds_v2.json").read_text(encoding="utf-8"))
    folds = folds_data["folds"]
    if len(train_graphs) != len(train) or not np.array_equal(
        [graph.smiles for graph in train_graphs], train["SMILES"].to_numpy()
    ):
        raise ValueError("V2 train graphs do not align with train rows")

    chembl, _ = load_chembl_unique()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = int(train_graphs[0].x.shape[1])
    oof = np.full(len(train), np.nan, dtype=float)
    fold_mses = []
    fold_epochs = []
    fold_chembl_counts = []
    fold_task_scales = []

    for fold in folds:
        fold_id = int(fold["fold"])
        train_indices = np.asarray(fold["train_indices"], dtype=int)
        valid_indices = np.asarray(fold["validation_indices"], dtype=int)
        validation_smiles = {
            canonical(train.iloc[index]["SMILES"]) for index in valid_indices
        }
        keep_chembl = ~chembl["canonical"].isin(validation_smiles).to_numpy()
        fold_chembl = chembl.loc[keep_chembl].reset_index(drop=True)
        if len(fold_chembl) < 5:
            raise ValueError(f"Too few ChEMBL structures in fold {fold_id}")
        fold_chembl_graphs = [
            make_graph(row, index, target_columns={"pIC50": "y_pic50"})
            for index, (_, row) in enumerate(fold_chembl.iterrows())
        ]

        pic50_scaler = StandardScaler().fit(
            fold_chembl["pIC50"].to_numpy(dtype=float).reshape(-1, 1)
        )
        logsi_scaler = StandardScaler().fit(
            target_logsi[train_indices].reshape(-1, 1)
        )
        pic50_normalized = pic50_scaler.transform(
            fold_chembl["pIC50"].to_numpy(dtype=float).reshape(-1, 1)
        ).ravel()
        logsi_normalized = logsi_scaler.transform(
            target_logsi.reshape(-1, 1)
        ).ravel()
        chem_dataset = make_dataset(
            fold_chembl_graphs, pic50_normalized, "y_pic50"
        )
        vacv_train_graphs = make_dataset(
            [train_graphs[index] for index in train_indices],
            logsi_normalized[train_indices],
            "y_logsi",
        )
        vacv_valid_graphs = [train_graphs[index] for index in valid_indices]
        chem_loader = make_train_loader(chem_dataset, shuffle=True)
        vacv_loader = make_train_loader(vacv_train_graphs, shuffle=True)

        set_seed(SEED + fold_id)
        model = MultiTaskGIN(input_dim).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        best_state = None
        best_mse = float("inf")
        best_epoch = 0
        wait = 0
        for epoch in range(1, MULTITASK_EPOCHS + 1):
            loss, loss_pic50, loss_logsi = train_multitask_epoch(
                model, chem_loader, vacv_loader, optimizer, device
            )
            prediction_logsi = predict_logsi(
                model, vacv_valid_graphs, logsi_scaler, device
            )
            prediction_si = np.maximum(np.expm1(prediction_logsi), 0.0)
            valid_mse = float(
                mean_squared_error(target_si[valid_indices], prediction_si)
            )
            if valid_mse < best_mse:
                best_mse = valid_mse
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                wait = 0
            else:
                wait += 1
            if epoch == 1 or epoch % 5 == 0:
                print(
                    f"fold={fold_id} multitask epoch={epoch:02d}/{MULTITASK_EPOCHS} "
                    f"loss={loss:.5f} pic50={loss_pic50:.5f} "
                    f"logsi={loss_logsi:.5f} val_mse_raw={valid_mse:.6g} "
                    f"best={best_mse:.6g}",
                    flush=True,
                )
            if wait >= PATIENCE:
                break
        if best_state is None:
            raise RuntimeError(f"No valid multitask checkpoint in fold {fold_id}")
        model.load_state_dict(best_state)

        for parameter in model.encoder.parameters():
            parameter.requires_grad = True
        fine_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=FINETUNE_LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        for epoch in range(1, VACV_FINETUNE_EPOCHS + 1):
            model.train()
            losses = []
            for batch in vacv_loader:
                batch = batch.to(device)
                fine_optimizer.zero_grad(set_to_none=True)
                _, predicted = model(batch)
                loss = F.mse_loss(predicted, batch.y_logsi.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                fine_optimizer.step()
                losses.append(float(loss.detach().cpu()))
            print(
                f"fold={fold_id} VACV-finetune epoch={epoch:02d}/{VACV_FINETUNE_EPOCHS} "
                f"loss_logsi={np.mean(losses):.5f}",
                flush=True,
            )

        fold_prediction_logsi = predict_logsi(
            model, vacv_valid_graphs, logsi_scaler, device
        )
        fold_prediction_si = np.maximum(np.expm1(fold_prediction_logsi), 0.0)
        oof[valid_indices] = fold_prediction_si
        fold_mse = float(
            mean_squared_error(target_si[valid_indices], fold_prediction_si)
        )
        fold_mses.append(fold_mse)
        fold_epochs.append(best_epoch)
        fold_chembl_counts.append(int(len(fold_chembl)))
        fold_task_scales.append(
            {
                "fold": fold_id,
                "pic50_mean": float(pic50_scaler.mean_[0]),
                "pic50_scale": float(pic50_scaler.scale_[0]),
                "logsi_mean": float(logsi_scaler.mean_[0]),
                "logsi_scale": float(logsi_scaler.scale_[0]),
            }
        )
        print(
            f"fold={fold_id} multitask-GNN best_epoch={best_epoch} "
            f"final_OOF_MSE_raw={fold_mse:.6g}; ChEMBL unique={len(fold_chembl)}",
            flush=True,
        )

    if not np.isfinite(oof).all():
        raise RuntimeError("Multitask GNN OOF predictions are incomplete")
    mse = float(mean_squared_error(target_si, oof))
    np.save(results_dir / "exp39oof_raw_si.npy", oof)
    baseline_gnn = float(
        json.loads((results_dir / "exp14_v2.json").read_text(encoding="utf-8"))["cvmetric"]
    )
    baseline_lgbm = float(
        json.loads((results_dir / "exp08_v2.json").read_text(encoding="utf-8"))["cvmetric"]
    )
    report = {
        "model": "multitask_GIN_pIC50_log1pSI",
        "params": {
            "encoder": "4-layer GIN, hidden_dim=128, dropout=0.3",
            "heads": ["pIC50", "log1p_SI"],
            "lambda_pic50": 1.0,
            "lambda_log1p_si": 1.0,
            "pIC50_target": "6 - log10(IC50_uM)",
            "SI_target": "log1p(SI)",
            "target_scaling": "per-fold StandardScaler",
            "multitask_epochs": MULTITASK_EPOCHS,
            "vacv_finetune_epochs": VACV_FINETUNE_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "fine_tuning_learning_rate": FINETUNE_LEARNING_RATE,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE_MULTITASK,
            "lambda_values": {"pic50": 1.0, "log1p_si": 1.0},
            "seed": SEED,
            "device": str(device),
            "chembl_source_rows": 454,
            "chembl_unique_canonical": len(chembl),
            "fold_pretraining_chembl_unique": fold_chembl_counts,
        },
        "cvmetric": mse,
        "metricname": "MSE_raw",
        "foldmetrics": fold_mses,
        "best_multitask_epochs": fold_epochs,
        "baseline_gnn_v2_mse": baseline_gnn,
        "baseline_lgbm_v2_mse": baseline_lgbm,
        "delta_vs_gnn_v2": mse - baseline_gnn,
        "delta_vs_lgbm_v2": mse - baseline_lgbm,
        "fold_task_scalers": fold_task_scales,
        "notes": (
            "Mixed ChEMBL/VACV batches train the two task heads and shared encoder; "
            "the model is then fine-tuned on VACV-only batches for ten epochs. "
            "ChEMBL structures matching each validation fold are excluded from that "
            "fold's auxiliary task training."
        ),
    }
    report_path = results_dir / "exp39_multitask_gnn.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Multi-task GNN MSE_raw={mse:.6g}")
    print(f"exp14_v2 GNN MSE_raw={baseline_gnn:.6g}")
    print(f"exp08_v2 LGBM MSE_raw={baseline_lgbm:.6g}")
    print(f"Saved result: {report_path}")


if __name__ == "__main__":
    main()
