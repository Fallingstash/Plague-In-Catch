import copy
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch_geometric.loader import DataLoader

from train_gnn import (
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    GINRegressor,
    set_seed,
)


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "experiments" / "results"
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
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}


def main():
    train = pd.read_csv(ROOT / "train_clean.csv")
    test = pd.read_csv(ROOT / "test_clean.csv")
    target = pd.to_numeric(train["SI VACV"], errors="coerce").to_numpy(dtype=float)
    if len(train) != 318 or len(test) != 597:
        raise ValueError(f"Expected 318 train and 597 test rows; got {len(train)} and {len(test)}")
    if not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError("Training SI values must be finite and positive")

    train_graphs = torch.load(ROOT / "train_graphs.pt", map_location="cpu", weights_only=False)
    test_graphs = torch.load(ROOT / "test_graphs.pt", map_location="cpu", weights_only=False)
    if len(train_graphs) != len(train) or len(test_graphs) != len(test):
        raise ValueError("Graph counts do not match train/test row counts")
    if not np.array_equal([graph.smiles for graph in train_graphs], train["SMILES"].to_numpy()):
        raise ValueError("Training graph order does not match train_clean.csv")
    if not np.array_equal([graph.smiles for graph in test_graphs], test["SMILES"].to_numpy()):
        raise ValueError("Test graph order does not match test_clean.csv")

    descriptors_train = pd.read_csv(ROOT / "featurestraindesc.csv")
    descriptors_test = pd.read_csv(ROOT / "featurestestdesc.csv")
    with np.load(ROOT / "featurestrainfp.npz") as data:
        fingerprints_train = data["fingerprints"]
    with np.load(ROOT / "featurestestfp.npz") as data:
        fingerprints_test = data["fingerprints"]
    features_train = np.concatenate(
        [descriptors_train.to_numpy(dtype=np.float32), fingerprints_train.astype(np.float32)],
        axis=1,
    )
    features_test = np.concatenate(
        [descriptors_test.to_numpy(dtype=np.float32), fingerprints_test.astype(np.float32)],
        axis=1,
    )
    if features_train.shape != (318, 2265) or features_test.shape != (597, 2265):
        raise ValueError(
            f"Unexpected feature shapes: {features_train.shape} and {features_test.shape}"
        )
    if np.isinf(features_train).any() or np.isinf(features_test).any():
        raise ValueError("Features contain infinite values")

    result_path = RESULTS_DIR / "exp22ensemble_stability.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    weights = result["weights"]
    gin_weight = float(weights["gin"])
    lgbm_weight = float(weights["lgbm_raw"])
    if not np.isclose(gin_weight + lgbm_weight, 1.0) or min(gin_weight, lgbm_weight) < 0:
        raise ValueError(f"Invalid ensemble weights in {result_path}")

    seed42 = next(item for item in result["seed_results"] if item["seed"] == 42)
    gnn_epochs = int(np.median(seed42["gnn_best_epochs"]))
    lgbm_iterations = int(np.median(seed42["lgbm_best_iterations"]))
    if gnn_epochs < 1 or lgbm_iterations < 1:
        raise ValueError("Invalid final model iteration counts")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    scaler = RobustScaler().fit(target.reshape(-1, 1))
    normalized_targets = scaler.transform(target.reshape(-1, 1)).ravel()
    full_train_graphs = []
    for graph, normalized_target in zip(train_graphs, normalized_targets):
        graph_copy = copy.copy(graph)
        graph_copy.y_norm = torch.tensor([normalized_target], dtype=torch.float32)
        full_train_graphs.append(graph_copy)
    train_loader = DataLoader(
        full_train_graphs, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    gin_model = GINRegressor(input_dim=int(train_graphs[0].x.shape[1])).to(device)
    optimizer = torch.optim.AdamW(
        gin_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    for epoch in range(1, gnn_epochs + 1):
        gin_model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = gin_model(batch)
            loss = torch.nn.functional.mse_loss(prediction, batch.y_norm.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gin_model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"GIN epoch={epoch:03d}/{gnn_epochs} train_mse_norm={np.mean(losses):.6f}")

    gin_model.eval()
    gnn_predictions = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            normalized = gin_model(batch).detach().cpu().numpy().reshape(-1, 1)
            gnn_predictions.extend(scaler.inverse_transform(normalized).ravel().tolist())
    gnn_predictions = np.asarray(gnn_predictions, dtype=float)

    lgbm_params = dict(LGBM_PARAMS)
    lgbm_params["n_estimators"] = lgbm_iterations
    lgbm_model = lgb.LGBMRegressor(**lgbm_params)
    lgbm_model.fit(features_train, target)
    lgbm_predictions = lgbm_model.predict(features_test)

    predictions = gin_weight * gnn_predictions + lgbm_weight * lgbm_predictions
    if predictions.shape != (len(test),) or not np.isfinite(predictions).all():
        raise RuntimeError("Ensemble predictions must be finite and match test row count")
    predictions = np.maximum(predictions, float(target.min()))
    submission = pd.DataFrame(
        {"index": test["index"].to_numpy(), "SI VACV": predictions}
    ).sort_values("index", kind="stable")
    if (
        len(submission) != 597
        or submission["index"].isna().any()
        or submission["SI VACV"].isna().any()
        or (submission["SI VACV"] <= 0).any()
        or not submission["index"].is_monotonic_increasing
        or submission["index"].duplicated().any()
    ):
        raise RuntimeError("Generated submission failed validation")

    output_path = ROOT / "submission_ensemble.csv"
    submission.to_csv(output_path, index=False)
    print(f"Final GIN epochs: {gnn_epochs}; LightGBM trees: {lgbm_iterations}")
    print(f"Ensemble weights: GIN={gin_weight:.6f}, LightGBM={lgbm_weight:.6f}")
    print(f"Submission preview:\n{submission.head().to_string(index=False)}")
    print(
        f"Prediction min={submission['SI VACV'].min():.6g}, "
        f"max={submission['SI VACV'].max():.6g}, "
        f"median={submission['SI VACV'].median():.6g}, "
        f"mean={submission['SI VACV'].mean():.6g}"
    )
    print(f"File size: {output_path.stat().st_size} bytes")
    print(f"Saved submission: {output_path}")


if __name__ == "__main__":
    main()
