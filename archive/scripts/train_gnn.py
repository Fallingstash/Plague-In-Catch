import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "experiments" / "results"
SEED = 42
HIDDEN_DIM = 128
NUM_LAYERS = 4
DROPOUT = 0.3
BATCH_SIZE = 32
MAX_EPOCHS = 300
PATIENCE = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5


class GINRegressor(nn.Module):
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
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, graph):
        x = graph.x
        for convolution, normalization in zip(self.convolutions, self.normalizations):
            x = convolution(x, graph.edge_index)
            x = normalization(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        pooled = global_mean_pool(x, graph.batch)
        return self.output(pooled).view(-1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def raw_prediction(model, loader, scaler, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            normalized = model(batch).detach().cpu().numpy().reshape(-1, 1)
            predictions.extend(scaler.inverse_transform(normalized).ravel().tolist())
    return np.asarray(predictions, dtype=float)


def main():
    graph_path = ROOT / "train_graphs.pt"
    train_path = ROOT / "train_clean.csv"
    folds_path = ROOT / "train_graphs_folds.json"
    for path in (graph_path, train_path, folds_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required GNN input is missing: {path}")

    graphs = torch.load(graph_path, map_location="cpu", weights_only=False)
    train = pd.read_csv(train_path)
    fold_manifest = json.loads(folds_path.read_text(encoding="utf-8"))
    if len(graphs) != 318 or len(train) != 318:
        raise ValueError(f"Expected 318 train graphs/labels, got {len(graphs)}/{len(train)}")
    if not np.array_equal(
        [graph.smiles for graph in graphs], train["SMILES"].to_numpy()
    ):
        raise ValueError("Graph SMILES do not align with train_clean.csv")
    fold_ids = np.asarray(fold_manifest["fold_ids"], dtype=int)
    if fold_ids.shape != (318,) or not np.array_equal(
        fold_ids, np.asarray([int(graph.fold_id) for graph in graphs])
    ):
        raise ValueError("Saved fold IDs do not align with train_graphs.pt")

    targets = pd.to_numeric(train["SI VACV"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(targets).all() or (targets <= 0).any():
        raise ValueError("Raw SI targets must be finite and positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training GIN on {device}; rows={len(graphs)}, folds={len(np.unique(fold_ids))}")
    oof_prediction = np.full(len(graphs), np.nan, dtype=float)
    fold_metrics = []
    best_epochs = []

    for fold_id in range(5):
        set_seed(SEED + fold_id)
        train_indices = np.flatnonzero(fold_ids != fold_id)
        valid_indices = np.flatnonzero(fold_ids == fold_id)
        scaler = RobustScaler()
        scaler.fit(targets[train_indices].reshape(-1, 1))
        normalized_targets = scaler.transform(targets.reshape(-1, 1)).ravel()

        fold_train_graphs = []
        fold_valid_graphs = []
        for index in train_indices:
            graph = copy.copy(graphs[index])
            graph.y_norm = torch.tensor([normalized_targets[index]], dtype=torch.float32)
            fold_train_graphs.append(graph)
        for index in valid_indices:
            fold_valid_graphs.append(graphs[index])
        train_loader = DataLoader(
            fold_train_graphs, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
        )
        valid_loader = DataLoader(
            fold_valid_graphs, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
        )

        model = GINRegressor(input_dim=int(graphs[0].x.shape[1])).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        best_mse = float("inf")
        best_state = None
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            train_losses = []
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch)
                loss = F.mse_loss(prediction, batch.y_norm.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))

            valid_prediction = raw_prediction(model, valid_loader, scaler, device)
            valid_mse = float(mean_squared_error(targets[valid_indices], valid_prediction))
            train_loss = float(np.mean(train_losses))
            improved = valid_mse < best_mse
            if improved:
                best_mse = valid_mse
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print(
                f"fold={fold_id} epoch={epoch:03d} "
                f"train_mse_norm={train_loss:.6f} "
                f"val_mse_raw={valid_mse:.6g} "
                f"best_val_mse_raw={best_mse:.6g} "
                f"best_epoch={best_epoch}",
                flush=True,
            )
            if epochs_without_improvement >= PATIENCE:
                print(
                    f"fold={fold_id} early stopping at epoch={epoch} "
                    f"(patience={PATIENCE})",
                    flush=True,
                )
                break

        if best_state is None:
            raise RuntimeError(f"Fold {fold_id} did not produce a valid model checkpoint")
        model.load_state_dict(best_state)
        best_prediction = raw_prediction(model, valid_loader, scaler, device)
        oof_prediction[valid_indices] = best_prediction
        final_fold_mse = float(mean_squared_error(targets[valid_indices], best_prediction))
        fold_metrics.append(final_fold_mse)
        best_epochs.append(best_epoch)
        print(
            f"fold={fold_id} completed best_epoch={best_epoch} "
            f"OOF_MSE_raw={final_fold_mse:.6g}",
            flush=True,
        )

    if not np.isfinite(oof_prediction).all():
        raise RuntimeError("GNN OOF predictions contain missing/non-finite values")
    overall_mse = float(mean_squared_error(targets, oof_prediction))
    np.save(RESULTS_DIR / "exp14oofrawsi.npy", oof_prediction)
    result = {
        "model": "GINRegressor_raw_SI",
        "params": {
            "num_layers": NUM_LAYERS,
            "hidden_dim": HIDDEN_DIM,
            "global_pooling": "mean",
            "dropout": DROPOUT,
            "target_scaler": "RobustScaler, fit on each training fold only",
            "loss": "MSELoss on normalized SI",
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "seed": SEED,
            "device": str(device),
        },
        "cvmetric": overall_mse,
        "metricname": "MSEraw",
        "foldmetrics": fold_metrics,
        "best_epochs": best_epochs,
        "notes": "Five-fold OOF raw-SI regression; fold IDs from train_graphs.pt; validation MSE computed after RobustScaler inverse transform.",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / "exp14gnn.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"GNN MSE_raw: {overall_mse:.6g}")
    print(f"Fold MSEs: {fold_metrics}")
    print(f"Best epochs: {best_epochs}")
    baseline = json.loads(
        (RESULTS_DIR / "exp08raw_target.json").read_text(encoding="utf-8")
    )["cvmetric"]
    print(f"exp08 MSE_raw: {baseline:.6g}")
    print(f"GNN / exp08: {overall_mse / baseline:.6g}")
    print(f"Saved result: {result_path}")


if __name__ == "__main__":
    main()
