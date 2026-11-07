import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, roc_auc_score


ROOT = Path(__file__).resolve().parent
FRIEND = ROOT / "from_friend"
RESULTS = ROOT / "experiments" / "results"
SEEDS = (42, 43)
N_FOLDS = 5
EPOCHS = 80
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
BASELINE_MSE = 3_831_981.8649830604
TARGET_TOLERANCE = 1e-5


def load_friend_modules():
    sys.path.insert(0, str(FRIEND))
    from src.tail_research import batch, graph
    from versions.test.training import GraphNet, features, transform
    import vacv_cpu_improved as cpu

    return batch, graph, GraphNet, features, transform, cpu


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)


def load_train():
    train = pd.read_csv(
        ROOT / "train_v2.csv",
        usecols=["SMILES", "SI VACV", "IC50_uM_VACV"],
    )
    target_si = pd.to_numeric(train["SI VACV"], errors="raise").to_numpy(dtype=float)
    activity = train["IC50_uM_VACV"].notna().to_numpy(dtype=np.int64)
    if len(train) != 320 or not np.isfinite(target_si).all() or (target_si < 0).any():
        raise ValueError("Expected 320 finite nonnegative SI targets")
    if not np.all(target_si[activity == 0] == 0):
        raise ValueError("Inactive v2 rows must have SI=0")
    return train, target_si, activity


def model_features(train, fit_indices, cpu, features_fn):
    raw = cpu.chem_features(train["SMILES"].astype(str).tolist())
    return features_fn(raw, np.asarray(fit_indices, dtype=int))


def train_fold_model(
    model,
    graph_data,
    features,
    target,
    train_indices,
    batch_fn,
    seed,
    objective,
):
    set_seed(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    rng = np.random.default_rng(seed)
    train_indices = np.asarray(train_indices, dtype=int)
    steps = max(1, int(np.ceil(len(train_indices) / BATCH_SIZE)))
    target_tensor = torch.as_tensor(target, dtype=torch.float32)

    for _ in range(EPOCHS):
        model.train()
        shuffled = rng.permutation(train_indices)
        for batch_ids in np.array_split(shuffled, steps):
            optimizer.zero_grad(set_to_none=True)
            batch_data = batch_fn(graph_data, batch_ids, features)
            prediction = model(*batch_data)
            loss = objective(prediction, target_tensor[batch_ids])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=3.0,
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Nonfinite gradient")
            optimizer.step()
    return model


def predict_logits(model, graph_data, features, indices, batch_fn):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_ids in np.array_split(
            np.asarray(indices, dtype=int),
            max(1, int(np.ceil(len(indices) / BATCH_SIZE))),
        ):
            predictions.extend(model(*batch_fn(graph_data, batch_ids, features)).cpu().numpy())
    return np.asarray(predictions, dtype=float)


def cross_validate(train, target_si, activity, batch_fn, graph_fn, GraphNet, features_fn, cpu):
    from src.validation import folds

    target_logsi = np.log1p(target_si)
    all_indices = np.arange(len(train))
    active_indices = np.flatnonzero(activity == 1)
    canonical_groups = np.asarray(
        cpu.chem_features(train["SMILES"].astype(str).tolist())["canon"],
        dtype=object,
    )
    graph_data = [graph_fn(smiles) for smiles in train["SMILES"].astype(str)]
    classifier_oof = {seed: np.full(len(train), np.nan) for seed in SEEDS}
    regressor_oof = {seed: np.full(len(train), np.nan) for seed in SEEDS}
    fold_reports = []

    for seed in SEEDS:
        split = folds(target_si, canonical_groups, seed, N_FOLDS)
        for fold_index, (train_indices, valid_indices) in enumerate(split, start=1):
            train_indices = np.asarray(train_indices, dtype=int)
            valid_indices = np.asarray(valid_indices, dtype=int)
            if set(canonical_groups[train_indices]) & set(canonical_groups[valid_indices]):
                raise ValueError(f"Canonical groups cross fold {fold_index}")

            classifier_extra, _ = model_features(
                train, train_indices, cpu, features_fn
            )
            classifier = GraphNet(extra_dim=classifier_extra.shape[1], layers=1)
            classifier.head[-1] = torch.nn.Linear(64, 1)
            classifier = train_fold_model(
                classifier,
                graph_data,
                classifier_extra,
                activity,
                train_indices,
                batch_fn,
                seed * 10000 + fold_index,
                lambda output, target: F.binary_cross_entropy_with_logits(
                    output.view(-1),
                    target.view(-1),
                ),
            )
            logits = predict_logits(
                classifier,
                graph_data,
                classifier_extra,
                valid_indices,
                batch_fn,
            )
            classifier_oof[seed][valid_indices] = 1.0 / (1.0 + np.exp(-logits))

            active_train_indices = train_indices[activity[train_indices] == 1]
            regressor_extra, _ = model_features(
                train, active_train_indices, cpu, features_fn
            )
            regressor = GraphNet(extra_dim=regressor_extra.shape[1], layers=1)
            regressor.head[-1] = torch.nn.Linear(64, 1)
            regressor = train_fold_model(
                regressor,
                graph_data,
                regressor_extra,
                target_logsi,
                active_train_indices,
                batch_fn,
                seed * 20000 + fold_index,
                F.mse_loss,
            )
            regressor_log_prediction = predict_logits(
                regressor,
                graph_data,
                regressor_extra,
                valid_indices,
                batch_fn,
            )
            regressor_oof[seed][valid_indices] = regressor_log_prediction
            active_valid = valid_indices[activity[valid_indices] == 1]
            fold_reports.append(
                {
                    "seed": seed,
                    "fold": fold_index,
                    "classifier_train_rows": int(len(train_indices)),
                    "regressor_train_active_rows": int(len(active_train_indices)),
                    "validation_rows": int(len(valid_indices)),
                    "validation_active_rows": int(len(active_valid)),
                }
            )
            print(
                f"seed={seed} fold={fold_index} classifier={len(train_indices)} "
                f"regressor_active={len(active_train_indices)}",
                flush=True,
            )

    for seed in SEEDS:
        if not np.isfinite(classifier_oof[seed]).all():
            raise RuntimeError(f"Incomplete classifier OOF for seed {seed}")
        if not np.isfinite(regressor_oof[seed]).all():
            raise RuntimeError(f"Incomplete regressor OOF for seed {seed}")

    classifier_probability = np.mean(
        np.stack([classifier_oof[seed] for seed in SEEDS]),
        axis=0,
    )
    regressor_log = np.mean(
        np.stack([regressor_oof[seed] for seed in SEEDS]),
        axis=0,
    )
    regressor_si = np.expm1(np.clip(regressor_log, 0.0, 14.0))
    soft_blend = classifier_probability * regressor_si
    hard_blend = np.where(classifier_probability >= 0.5, regressor_si, 0.0)
    active_valid = activity == 1

    report = {
        "classifier_roc_auc": float(roc_auc_score(activity, classifier_probability)),
        "classifier_accuracy_at_0_5": float(
            np.mean((classifier_probability >= 0.5).astype(int) == activity)
        ),
        "regressor_oof_mse_log1p_active": float(
            mean_squared_error(target_logsi[active_valid], regressor_log[active_valid])
        ),
        "regressor_oof_rmse_log1p_active": float(
            np.sqrt(mean_squared_error(target_logsi[active_valid], regressor_log[active_valid]))
        ),
        "regressor_oof_mse_raw_si_active": float(
            mean_squared_error(target_si[active_valid], regressor_si[active_valid])
        ),
        "soft_blend_oof_mse_raw_si": float(mean_squared_error(target_si, soft_blend)),
        "hard_blend_oof_mse_raw_si": float(mean_squared_error(target_si, hard_blend)),
        "classifier_oof": classifier_probability,
        "regressor_log1p_oof": regressor_log,
        "soft_blend_oof": soft_blend,
        "folds": fold_reports,
    }
    return report


def fit_final_and_submit(train, target_si, activity, batch_fn, graph_fn, GraphNet, features_fn, cpu):
    all_indices = np.arange(len(train))
    active_indices = np.flatnonzero(activity == 1)
    graphs = [graph_fn(smiles) for smiles in train["SMILES"].astype(str)]

    classifier_extra, classifier_scaler = model_features(
        train, all_indices, cpu, features_fn
    )
    classifier = GraphNet(extra_dim=classifier_extra.shape[1], layers=1)
    classifier.head[-1] = torch.nn.Linear(64, 1)
    classifier = train_fold_model(
        classifier,
        graphs,
        classifier_extra,
        activity,
        all_indices,
        batch_fn,
        4242,
        lambda output, target: F.binary_cross_entropy_with_logits(
            output.view(-1),
            target.view(-1),
        ),
    )

    regressor_extra, regressor_scaler = model_features(
        train, active_indices, cpu, features_fn
    )
    regressor = GraphNet(extra_dim=regressor_extra.shape[1], layers=1)
    regressor.head[-1] = torch.nn.Linear(64, 1)
    regressor = train_fold_model(
        regressor,
        graphs,
        regressor_extra,
        np.log1p(target_si),
        active_indices,
        batch_fn,
        4343,
        F.mse_loss,
    )
    model_path = RESULTS / "exp51_strategy_c_gcn_model.pt"
    torch.save(
        {
            "classifier_state_dict": classifier.state_dict(),
            "classifier_scaler": classifier_scaler,
            "regressor_state_dict": regressor.state_dict(),
            "regressor_scaler": regressor_scaler,
            "classifier_extra_dim": int(classifier_extra.shape[1]),
            "regressor_extra_dim": int(regressor_extra.shape[1]),
            "training_rows": len(train),
            "training_active_rows": int(len(active_indices)),
            "seeds": [4242, 4343],
        },
        model_path,
    )

    test = pd.read_csv(ROOT / "test_clean.csv", usecols=["index", "SMILES"])
    test_graphs = [graph_fn(smiles) for smiles in test["SMILES"].astype(str)]
    classifier_raw = cpu.chem_features(
        train["SMILES"].astype(str).tolist() + test["SMILES"].astype(str).tolist()
    )
    classifier_test_extra = features_fn(classifier_raw, all_indices)[0]
    classifier_logits = predict_logits(
        classifier,
        graphs + test_graphs,
        classifier_test_extra.astype(np.float32),
        np.arange(len(train), len(train) + len(test)),
        batch_fn,
    )
    probability = 1.0 / (1.0 + np.exp(-classifier_logits))

    regressor_smiles = (
        train["SMILES"].astype(str).tolist()
        + test["SMILES"].astype(str).tolist()
    )
    regressor_raw = cpu.chem_features(regressor_smiles)
    regressor_test_extra = features_fn(regressor_raw, active_indices)[0]
    regressor_log = predict_logits(
        regressor,
        graphs + test_graphs,
        regressor_test_extra.astype(np.float32),
        np.arange(len(train), len(train) + len(test)),
        batch_fn,
    )
    predictions = probability * np.expm1(np.clip(regressor_log, 0.0, 14.0))
    submission = pd.DataFrame(
        {"index": test["index"].to_numpy(), "SI VACV": predictions}
    ).sort_values("index", kind="stable")
    if (
        len(submission) != 597
        or submission["index"].duplicated().any()
        or submission.isna().any().any()
        or not np.isfinite(predictions).all()
        or (predictions < 0).any()
    ):
        raise ValueError("Strategy C GCN submission validation failed")
    submission_path = ROOT / "submission_strategy_c_gcn.csv"
    submission.to_csv(submission_path, index=False)
    return {
        "model": str(model_path.relative_to(ROOT)),
        "submission": str(submission_path.relative_to(ROOT)),
        "rows": len(submission),
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
        "prediction_mean": float(predictions.mean()),
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    train, target_si, activity = load_train()
    baseline_path = RESULTS / "exp49_gcn_baseline_oof.npy"
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Missing reproduced GCN baseline OOF: {baseline_path}")
    baseline_oof = np.load(baseline_path)
    if baseline_oof.shape != target_si.shape or not np.isfinite(baseline_oof).all():
        raise ValueError("Reproduced GCN baseline OOF is invalid")
    baseline_mse = float(mean_squared_error(target_si, baseline_oof))
    if abs(baseline_mse - BASELINE_MSE) / BASELINE_MSE > TARGET_TOLERANCE:
        raise ValueError("Baseline OOF does not match the recorded GCN reference")

    batch_fn, graph_fn, GraphNet, features_fn, _, cpu = load_friend_modules()
    result = cross_validate(
        train,
        target_si,
        activity,
        batch_fn,
        graph_fn,
        GraphNet,
        features_fn,
        cpu,
    )
    np.save(RESULTS / "exp51_classifier_oof_probs.npy", result["classifier_oof"])
    np.save(RESULTS / "exp51_regressor_oof_log1p.npy", result["regressor_log1p_oof"])
    np.save(RESULTS / "exp51_strategy_c_oof_raw_si.npy", result["soft_blend_oof"])

    report = {
        "model": "FriendGraphNet_strategy_C",
        "architecture": {
            "encoder": "10 atom features -> Linear(10,64) -> normalized adjacency aggregation -> Linear(64,64); masked mean pooling",
            "molecular_features": "Morgan fp2 + MACCS + standardized RDKit descriptors",
            "classifier_loss": "BCEWithLogitsLoss",
            "regressor_loss": "MSE on log1p(SI), active rows only",
            "blend": "P(active) * expm1(regressor_log1p)",
        },
        "rows": int(len(train)),
        "active_rows": int(activity.sum()),
        "inactive_rows": int(len(activity) - activity.sum()),
        "group_cv": {"seeds": list(SEEDS), "folds": N_FOLDS, "epochs": EPOCHS},
        "baseline_mse_raw_si": baseline_mse,
        "classifier_roc_auc": result["classifier_roc_auc"],
        "classifier_accuracy_at_0_5": result["classifier_accuracy_at_0_5"],
        "regressor_oof_mse_log1p_active": result["regressor_oof_mse_log1p_active"],
        "regressor_oof_rmse_log1p_active": result["regressor_oof_rmse_log1p_active"],
        "regressor_oof_mse_raw_si_active": result["regressor_oof_mse_raw_si_active"],
        "soft_blend_oof_mse_raw_si": result["soft_blend_oof_mse_raw_si"],
        "hard_blend_oof_mse_raw_si": result["hard_blend_oof_mse_raw_si"],
        "difference_from_baseline": result["soft_blend_oof_mse_raw_si"] - baseline_mse,
        "relative_improvement": (
            baseline_mse - result["soft_blend_oof_mse_raw_si"]
        ) / baseline_mse,
        "folds": result["folds"],
        "oof_files": {
            "classifier_probability": "experiments/results/exp51_classifier_oof_probs.npy",
            "regressor_log1p": "experiments/results/exp51_regressor_oof_log1p.npy",
            "soft_blend_raw_si": "experiments/results/exp51_strategy_c_oof_raw_si.npy",
        },
        "submission": None,
    }
    if result["soft_blend_oof_mse_raw_si"] < baseline_mse:
        report["submission"] = fit_final_and_submit(
            train,
            target_si,
            activity,
            batch_fn,
            graph_fn,
            GraphNet,
            features_fn,
            cpu,
        )

    report_path = RESULTS / "exp51_strategy_c_gcn.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
