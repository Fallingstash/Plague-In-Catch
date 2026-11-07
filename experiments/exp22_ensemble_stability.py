import copy
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from torch_geometric.loader import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from train_gnn import (  # noqa: E402
    BATCH_SIZE,
    DROPOUT,
    HIDDEN_DIM,
    LEARNING_RATE,
    MAX_EPOCHS,
    NUM_LAYERS,
    PATIENCE,
    WEIGHT_DECAY,
    GINRegressor,
    set_seed,
)

RESULTS_DIR = ROOT / "experiments" / "results"
SEEDS = (42, 43, 44)
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
LGBM_EARLY_STOPPING = 50


def load_oof(path_options):
    for path in path_options:
        if path.is_file():
            return np.load(path)
    raise FileNotFoundError("Missing OOF input: " + ", ".join(str(path) for path in path_options))


def fit_simplex_weights(predictions, target):
    prediction_matrix = np.column_stack(predictions)
    if prediction_matrix.shape[1] < 2:
        raise ValueError("At least two model predictions are required")

    def objective(weights):
        prediction = prediction_matrix @ weights
        return float(np.mean((target - prediction) ** 2))

    result = minimize(
        objective,
        x0=np.full(prediction_matrix.shape[1], 1.0 / prediction_matrix.shape[1]),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * prediction_matrix.shape[1],
        constraints=[{"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}],
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Ensemble weight optimization failed: {result.message}")
    return result.x, float(result.fun)


def make_seed_folds(seed, row_count):
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    folds = []
    fold_ids = np.full(row_count, -1, dtype=int)
    for fold_id, (train_indices, valid_indices) in enumerate(splitter.split(np.arange(row_count))):
        fold_ids[valid_indices] = fold_id
        folds.append(
            {
                "fold": fold_id,
                "train_indices": train_indices.tolist(),
                "validation_indices": valid_indices.tolist(),
            }
        )
    if (fold_ids < 0).any():
        raise RuntimeError(f"Seed {seed} did not assign every row to validation")
    return folds, fold_ids


def raw_prediction(model, loader, scaler, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            normalized = model(batch).detach().cpu().numpy().reshape(-1, 1)
            predictions.extend(scaler.inverse_transform(normalized).ravel().tolist())
    return np.asarray(predictions, dtype=float)


def fit_seed_gnn(graphs, target, folds, seed, device):
    oof = np.full(len(graphs), np.nan, dtype=float)
    best_epochs = []
    fold_mses = []
    for fold in folds:
        fold_id = int(fold["fold"])
        train_indices = np.asarray(fold["train_indices"], dtype=int)
        valid_indices = np.asarray(fold["validation_indices"], dtype=int)
        set_seed(seed * 100 + fold_id)
        scaler = RobustScaler().fit(target[train_indices].reshape(-1, 1))
        normalized_targets = scaler.transform(target.reshape(-1, 1)).ravel()
        train_graphs = []
        for index in train_indices:
            graph = copy.copy(graphs[index])
            graph.y_norm = torch.tensor([normalized_targets[index]], dtype=torch.float32)
            train_graphs.append(graph)
        valid_graphs = [graphs[index] for index in valid_indices]
        train_loader = DataLoader(
            train_graphs, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
        )
        valid_loader = DataLoader(
            valid_graphs, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
        )
        model = GINRegressor(int(graphs[0].x.shape[1])).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        best_mse = float("inf")
        best_state = None
        best_epoch = 0
        wait = 0
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            losses = []
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch)
                loss = torch.nn.functional.mse_loss(prediction, batch.y_norm.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            valid_prediction = raw_prediction(model, valid_loader, scaler, device)
            valid_mse = float(mean_squared_error(target[valid_indices], valid_prediction))
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
            if epoch == 1 or epoch % 10 == 0:
                print(
                    f"seed={seed} GIN fold={fold_id} epoch={epoch:03d} "
                    f"train_mse_norm={np.mean(losses):.6f} "
                    f"val_mse_raw={valid_mse:.6g} best={best_mse:.6g}",
                    flush=True,
                )
            if wait >= PATIENCE:
                break
        if best_state is None:
            raise RuntimeError(f"GIN seed={seed}, fold={fold_id}: no valid checkpoint")
        model.load_state_dict(best_state)
        oof[valid_indices] = raw_prediction(model, valid_loader, scaler, device)
        fold_mse = float(mean_squared_error(target[valid_indices], oof[valid_indices]))
        best_epochs.append(best_epoch)
        fold_mses.append(fold_mse)
        print(
            f"seed={seed} GIN fold={fold_id} done: best_epoch={best_epoch}, "
            f"OOF_MSE_raw={fold_mse:.6g}",
            flush=True,
        )
    if not np.isfinite(oof).all():
        raise RuntimeError(f"GIN seed={seed} OOF is incomplete")
    return oof, fold_mses, best_epochs


def fit_seed_lgbm(features, target, folds, seed):
    oof = np.full(len(target), np.nan, dtype=float)
    best_iterations = []
    fold_mses = []
    for fold in folds:
        fold_id = int(fold["fold"])
        train_indices = np.asarray(fold["train_indices"], dtype=int)
        valid_indices = np.asarray(fold["validation_indices"], dtype=int)
        params = dict(LGBM_PARAMS)
        params["random_state"] = seed
        model = lgb.LGBMRegressor(**params)
        model.fit(
            features[train_indices],
            target[train_indices],
            eval_set=[(features[valid_indices], target[valid_indices])],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(LGBM_EARLY_STOPPING, verbose=False)],
        )
        oof[valid_indices] = model.predict(
            features[valid_indices], num_iteration=model.best_iteration_
        )
        best_iteration = int(model.best_iteration_ or LGBM_PARAMS["n_estimators"])
        best_iterations.append(best_iteration)
        fold_mse = float(mean_squared_error(target[valid_indices], oof[valid_indices]))
        fold_mses.append(fold_mse)
        print(
            f"seed={seed} LGBM fold={fold_id} done: best_iteration={best_iteration}, "
            f"OOF_MSE_raw={fold_mse:.6g}",
            flush=True,
        )
    if not np.isfinite(oof).all():
        raise RuntimeError(f"LightGBM seed={seed} OOF is incomplete")
    return oof, fold_mses, best_iterations


def main():
    results = RESULTS_DIR
    target = pd.read_csv(ROOT / "train_clean.csv")["SI VACV"].to_numpy(dtype=float)
    graphs = torch.load(ROOT / "train_graphs.pt", map_location="cpu", weights_only=False)
    descriptors = pd.read_csv(ROOT / "featurestraindesc.csv")
    with np.load(ROOT / "featurestrainfp.npz") as data:
        fingerprints = data["fingerprints"]
    features = np.concatenate(
        [descriptors.to_numpy(dtype=np.float32), fingerprints.astype(np.float32)], axis=1
    )
    if len(target) != 318 or len(graphs) != 318 or features.shape != (318, 2265):
        raise ValueError("Expected the original aligned 318-row graph/feature dataset")
    if not np.isfinite(target).all() or np.isinf(features).any():
        raise ValueError("Invalid training targets or features")

    gnn_original = load_oof([results / "exp14oofrawsi.npy"])
    lgbm_original = load_oof(
        [results / "exp08oofrawsi.npy", results / "exp08oof_raw_si.npy"]
    )
    if gnn_original.shape != (318,) or lgbm_original.shape != (318,):
        raise ValueError("Existing OOF predictions must each contain 318 rows")
    optimal_weights, optimized_mse = fit_simplex_weights(
        [gnn_original, lgbm_original], target
    )
    equal_weight_mse = float(
        mean_squared_error(target, 0.5 * gnn_original + 0.5 * lgbm_original)
    )
    print(
        f"Seed-42 OOF optimal weights: GIN={optimal_weights[0]:.6f}, "
        f"LightGBM={optimal_weights[1]:.6f}; MSE={optimized_mse:.6g}"
    )
    print(f"Seed-42 50/50 MSE: {equal_weight_mse:.6g}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_results = []
    for seed in SEEDS:
        folds, fold_ids = make_seed_folds(seed, len(target))
        fold_path = results / f"folds_seed{seed}.json"
        fold_path.write_text(
            json.dumps(
                {
                    "n_splits": 5,
                    "shuffle": True,
                    "random_state": seed,
                    "folds": folds,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        gnn_oof, gnn_fold_mse, gnn_best_epochs = fit_seed_gnn(
            graphs, target, folds, seed, device
        )
        lgbm_oof, lgbm_fold_mse, lgbm_best_iterations = fit_seed_lgbm(
            features, target, folds, seed
        )
        ensemble = optimal_weights[0] * gnn_oof + optimal_weights[1] * lgbm_oof
        seed_mse = float(mean_squared_error(target, ensemble))
        np.save(results / f"exp22_seed{seed}_gnn_oof.npy", gnn_oof)
        np.save(results / f"exp22_seed{seed}_lgbm_oof.npy", lgbm_oof)
        np.save(results / f"exp22_seed{seed}_ensemble_oof.npy", ensemble)
        seed_results.append(
            {
                "seed": seed,
                "mse_ensemble": seed_mse,
                "mse_gnn": float(mean_squared_error(target, gnn_oof)),
                "mse_lgbm": float(mean_squared_error(target, lgbm_oof)),
                "gnn_fold_mses": gnn_fold_mse,
                "lgbm_fold_mses": lgbm_fold_mse,
                "gnn_best_epochs": gnn_best_epochs,
                "lgbm_best_iterations": lgbm_best_iterations,
                "fold_counts": np.bincount(fold_ids, minlength=5).tolist(),
            }
        )
        print(
            f"seed={seed} ensemble MSE_raw={seed_mse:.6g} "
            f"(GIN={seed_results[-1]['mse_gnn']:.6g}, "
            f"LGBM={seed_results[-1]['mse_lgbm']:.6g})",
            flush=True,
        )

    seed_mses = np.asarray([item["mse_ensemble"] for item in seed_results])
    mean_mse = float(seed_mses.mean())
    std_population = float(seed_mses.std(ddof=0))
    std_sample = float(seed_mses.std(ddof=1))
    result = {
        "model": "optimized_simplex_gin_lgbm_ensemble",
        "weights": {
            "gin": float(optimal_weights[0]),
            "lgbm_raw": float(optimal_weights[1]),
        },
        "seed42_oof_optimized_mse": optimized_mse,
        "seed42_oof_equal_weight_mse": equal_weight_mse,
        "stability_seeds": list(SEEDS),
        "seed_results": seed_results,
        "mse_mean": mean_mse,
        "mse_std_population": std_population,
        "mse_std_sample": std_sample,
        "stable_std_lt_0_5M": std_population < 500000.0,
        "notes": "Non-negative weights summing to one optimized on seed-42 OOF; same weights applied unchanged to each seed-specific OOF run.",
    }
    out = results / "exp22ensemble_stability.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Ensemble seed MSEs: {seed_mses.tolist()}")
    print(f"Mean MSE: {mean_mse:.6g}")
    print(f"Std MSE (population): {std_population:.6g}")
    print(f"Std MSE (sample): {std_sample:.6g}")
    print(f"Stable (<0.5M): {std_population < 500000.0}")
    print(f"Saved result: {out}")


if __name__ == "__main__":
    main()
