import copy
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import RobustScaler
from torch_geometric.loader import DataLoader

from experiments.upgrade_common import RESULTS_DIR, load_lgbm_features, load_oof, load_training_data
from train_gnn import BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, GINRegressor, set_seed


ROOT = Path(__file__).resolve().parent


def select_lgbm(use_tuned=False):
    tuning_path = RESULTS_DIR / "exp31tuning.json"
    tuned_oof_path = RESULTS_DIR / "exp31oof_raw_si.npy"
    if use_tuned and tuning_path.is_file() and tuned_oof_path.is_file():
        tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
        if tuning["improved_over_baseline"]:
            return tuning["best_params"], int(np.median(tuning["best_iterations"])), "exp31 tuned"
    if use_tuned:
        raise FileNotFoundError("Stack requested tuned LightGBM, but no successful tuned model is available")

    baseline = json.loads((RESULTS_DIR / "exp08raw_target.json").read_text(encoding="utf-8"))
    params = dict(baseline["params"])
    params.pop("early_stopping_rounds", None)
    params.pop("eval_metric", None)
    params.pop("target", None)
    return params, int(np.median(baseline["best_iterations"])), "exp08 baseline"


def predict_gnn(train_graphs, test_graphs, target, epochs, device):
    scaler = RobustScaler().fit(target.reshape(-1, 1))
    normalized_targets = scaler.transform(target.reshape(-1, 1)).ravel()
    full_graphs = []
    for graph, value in zip(train_graphs, normalized_targets):
        graph_copy = copy.copy(graph)
        graph_copy.y_norm = torch.tensor([value], dtype=torch.float32)
        full_graphs.append(graph_copy)
    train_loader = DataLoader(
        full_graphs, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    model = GINRegressor(input_dim=int(train_graphs[0].x.shape[1])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    for epoch in range(1, epochs + 1):
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
        print(
            f"GIN epoch={epoch:03d}/{epochs} train_mse_norm={np.mean(losses):.6f}",
            flush=True,
        )

    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            normalized = model(batch).detach().cpu().numpy().reshape(-1, 1)
            predictions.extend(scaler.inverse_transform(normalized).ravel().tolist())
    return np.asarray(predictions, dtype=float)


def main():
    train, target = load_training_data()
    test = pd.read_csv(ROOT / "test_clean.csv")
    if len(test) != 597:
        raise ValueError(f"Expected 597 test rows, got {len(test)}")
    features_train, features_test = load_lgbm_features()
    train_graphs = torch.load(ROOT / "train_graphs.pt", map_location="cpu", weights_only=False)
    test_graphs = torch.load(ROOT / "test_graphs.pt", map_location="cpu", weights_only=False)
    if len(train_graphs) != 318 or len(test_graphs) != 597:
        raise ValueError("Train/test graph counts do not match the datasets")
    if not np.array_equal(
        [graph.smiles for graph in train_graphs], train["SMILES"].to_numpy()
    ):
        raise ValueError("Training graph order differs from train_clean.csv")
    if not np.array_equal(
        [graph.smiles for graph in test_graphs], test["SMILES"].to_numpy()
    ):
        raise ValueError("Test graph order differs from test_clean.csv")

    meta_path = RESULTS_DIR / "exp30_meta_model.joblib"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Stacking meta-model is missing: {meta_path}")
    meta_model = joblib.load(meta_path)
    stack_report = json.loads(
        (RESULTS_DIR / "exp30stacking.json").read_text(encoding="utf-8")
    )
    use_tuned_lgbm = stack_report["selected_lgbm_oof"] == "exp31 tuned"
    params, lgbm_iterations, lgbm_source = select_lgbm(use_tuned_lgbm)
    if use_tuned_lgbm and lgbm_source != "exp31 tuned":
        raise RuntimeError("Stack meta-model expects tuned LightGBM, but tuning is unavailable")
    params = dict(params)
    params["n_estimators"] = lgbm_iterations
    params["random_state"] = 42
    params["verbose"] = -1
    params["n_jobs"] = -1

    gin_epochs = int(
        np.median(
            json.loads((RESULTS_DIR / "exp14gnn.json").read_text(encoding="utf-8"))[
                "best_epochs"
            ]
        )
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    gnn_test = predict_gnn(train_graphs, test_graphs, target, gin_epochs, device)

    lgbm_model = lgb.LGBMRegressor(**params)
    lgbm_model.fit(features_train, target)
    lgbm_test = lgbm_model.predict(features_test)

    classifier_report = json.loads(
        (RESULTS_DIR / "exp02classifier.json").read_text(encoding="utf-8")
    )
    classifier_params = dict(classifier_report["params"])
    classifier_params.pop("early_stopping_rounds", None)
    classifier_params.pop("eval_metric", None)
    classifier_params["n_estimators"] = int(
        np.median(classifier_report["bestiterations"])
    )
    classifier = lgb.LGBMClassifier(**classifier_params)
    labels = train["is_active"].to_numpy(dtype=int)
    classifier.fit(features_train, labels)
    raw_test_probability = classifier.predict_proba(features_test)[:, 1]

    oof_probability = load_oof("exp02oof_probs.npy")
    probability_calibrator = LogisticRegression(random_state=42, max_iter=1000)
    probability_calibrator.fit(oof_probability[:, None], labels)
    calibrated_test_probability = probability_calibrator.predict_proba(
        raw_test_probability[:, None]
    )[:, 1]

    selected_oof_lgbm = (
        load_oof("exp31oof_raw_si.npy")
        if use_tuned_lgbm
        else load_oof("exp08oof_raw_si.npy")
    )
    active = labels == 1
    regression_calibrator = LinearRegression().fit(
        selected_oof_lgbm[active, None], target[active]
    )
    calibrated_test_si = regression_calibrator.predict(lgbm_test[:, None])
    exp10_test = (
        (1.0 - calibrated_test_probability) * 10000.0
        + calibrated_test_probability * calibrated_test_si
    )
    meta_features = np.column_stack(
        [gnn_test, lgbm_test, exp10_test, raw_test_probability]
    )
    stack_predictions = meta_model.predict(meta_features)

    candidate_mses = {
        "stacking": float(stack_report["cvmetric"]),
        "weighted_gnn_lgbm_reference": float(
            json.loads((RESULTS_DIR / "exp22ensemble_stability.json").read_text())[
                "seed42_oof_optimized_mse"
            ]
        ),
    }
    crossfit_path = RESULTS_DIR / "exp29crossfit.json"
    if crossfit_path.is_file() and not use_tuned_lgbm:
        candidate_mses["crossfit_optimized_3_model"] = float(
            json.loads(crossfit_path.read_text(encoding="utf-8"))["cvmetric"]
        )
    best_candidate = min(candidate_mses, key=candidate_mses.get)
    if best_candidate == "stacking":
        predictions = stack_predictions
        selected_model = "Ridge stacking"
    elif best_candidate == "crossfit_optimized_3_model":
        weights = np.asarray(
            json.loads(crossfit_path.read_text(encoding="utf-8"))[
                "full_data_optimal_weights"
            ],
            dtype=float,
        )
        predictions = weights[0] * gnn_test + weights[1] * lgbm_test + weights[2] * exp10_test
        selected_model = "full-data optimized 3-model blend"
    else:
        weights = np.asarray([0.568825363320826, 0.43117463667917394])
        predictions = weights[0] * gnn_test + weights[1] * lgbm_test
        selected_model = "weighted GIN + LightGBM reference"

    if not np.isfinite(predictions).all():
        raise RuntimeError("Final predictions contain non-finite values")
    predictions = np.maximum(predictions, float(target.min()))
    submission = pd.DataFrame(
        {"index": test["index"].to_numpy(), "SI VACV": predictions}
    ).sort_values("index", kind="stable")
    if (
        len(submission) != 597
        or submission.isna().any().any()
        or not np.isfinite(submission["SI VACV"]).all()
        or (submission["SI VACV"] <= 0).any()
        or submission["index"].duplicated().any()
        or not submission["index"].is_monotonic_increasing
    ):
        raise RuntimeError("Final submission validation failed")
    output_path = ROOT / "submissionmlfinal.csv"
    submission.to_csv(output_path, index=False)

    result = {
        "model": selected_model,
        "selected_lgbm": lgbm_source,
        "gin_epochs": gin_epochs,
        "lgbm_iterations": lgbm_iterations,
        "classifier_iterations": classifier_params["n_estimators"],
        "candidate_oof_mse": candidate_mses,
        "prediction_summary": {
            "min": float(submission["SI VACV"].min()),
            "max": float(submission["SI VACV"].max()),
            "median": float(submission["SI VACV"].median()),
            "mean": float(submission["SI VACV"].mean()),
        },
        "rows": len(submission),
        "file_size_bytes": output_path.stat().st_size,
        "output": str(output_path),
        "notes": "Test CC50/IC50 columns are not used in features or predictions.",
    }
    report_path = RESULTS_DIR / "makesubmissionfinal.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Selected model: {selected_model}; CV comparison={candidate_mses}")
    print(f"Submission preview:\n{submission.head().to_string(index=False)}")
    print(f"Prediction summary: {result['prediction_summary']}")
    print(f"Saved {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
