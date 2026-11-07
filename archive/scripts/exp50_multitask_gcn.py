import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from torch import nn


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "experiments" / "results"
FRIEND = ROOT / "from_friend"
LAMBDAS = (0.1, 0.5, 1.0)
SEEDS = (42, 43)
N_FOLDS = 5
MAX_EPOCHS = 80
PATIENCE = 12
BATCH_SIZE = 32
BASELINE_MSE = 3_831_981.9529758967


def canonical(smiles):
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(f"Invalid or empty SMILES: {smiles}")
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def load_friend_modules():
    sys.path.insert(0, str(FRIEND))
    from src.tail_research import batch, graph
    import vacv_cpu_improved as cpu
    from versions.test.training import features

    return batch, graph, cpu, features


class MultiTaskGraphNet(nn.Module):
    def __init__(self, extra_dim):
        super().__init__()
        self.first = nn.Linear(10, 64)
        self.conv = nn.Linear(64, 64)
        self.extra_projection = nn.Linear(extra_dim, 64, bias=False)
        self.dropout = nn.Dropout(0.2)
        self.si_head = nn.Linear(64, 1)
        self.pic50_head = nn.Linear(64, 1)

    def encode(self, atoms, adjacency, mask, extra):
        nodes = torch.relu(self.first(atoms))
        nodes = torch.relu(self.conv(torch.bmm(adjacency, nodes)))
        pooled = (nodes * mask[:, :, None]).sum(dim=1)
        pooled = pooled / mask.sum(dim=1, keepdim=True).clamp_min(1)
        representation = torch.relu(pooled + self.extra_projection(extra))
        representation = self.dropout(representation)
        return representation

    def forward(self, atoms, adjacency, mask, extra):
        representation = self.encode(atoms, adjacency, mask, extra)
        return self.si_head(representation).squeeze(1), self.pic50_head(representation).squeeze(1)


def load_data():
    train = pd.read_csv(ROOT / "train_v2.csv", usecols=["SMILES", "SI VACV"])
    target_si = pd.to_numeric(train["SI VACV"], errors="raise").to_numpy(dtype=float)
    if len(train) != 320 or not np.isfinite(target_si).all() or (target_si < 0).any():
        raise ValueError("Expected 320 finite nonnegative SI targets")

    test = pd.read_csv(ROOT / "test_clean.csv", usecols=["SMILES"])
    test_groups = {canonical(smiles) for smiles in test["SMILES"]}
    chembl = pd.read_csv(ROOT / "data" / "collected_raw.csv")
    if not {"SMILES", "IC50uM"}.issubset(chembl.columns) or len(chembl) != 454:
        raise ValueError("Expected 454 ChEMBL rows with SMILES and IC50uM")
    chembl["IC50uM"] = pd.to_numeric(chembl["IC50uM"], errors="coerce")
    chembl = chembl[
        chembl["SMILES"].notna()
        & np.isfinite(chembl["IC50uM"])
        & (chembl["IC50uM"] > 0)
    ].copy()
    chembl["canonical"] = chembl["SMILES"].map(canonical)
    overlap = set(chembl["canonical"]) & test_groups
    overlap_rows = int(chembl["canonical"].isin(test_groups).sum())
    chembl = chembl.loc[~chembl["canonical"].isin(test_groups)].copy()
    if set(chembl["canonical"]) & test_groups:
        raise ValueError("ChEMBL training structures overlap test_clean.csv")

    chembl["pIC50"] = 6.0 - np.log10(chembl["IC50uM"].to_numpy(dtype=float))
    chembl = (
        chembl.groupby("canonical", sort=False, as_index=False)
        .agg({"SMILES": "first", "pIC50": "median"})
        .reset_index(drop=True)
    )
    if len(chembl) < 5:
        raise ValueError("Too few unique ChEMBL molecules remain after test exclusion")
    return train, target_si, chembl, {
        "input_rows": 454,
        "excluded_test_overlap_rows": overlap_rows,
        "excluded_test_overlap_unique_structures": len(overlap),
        "used_unique_structures": len(chembl),
    }


def prepare_model_data(train, chembl, fold_train_indices, graph_fn, cpu, features_fn):
    smiles = train["SMILES"].astype(str).tolist() + chembl["SMILES"].astype(str).tolist()
    raw = cpu.chem_features(smiles)
    extra, _ = features_fn(raw, np.asarray(fold_train_indices, dtype=int))
    graphs = [graph_fn(smiles_item) for smiles_item in smiles]
    return graphs, extra.astype(np.float32)


def batched_predictions(model, graphs, extra, indices, batch_fn, target_head):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_ids in np.array_split(
            np.asarray(indices, dtype=int),
            max(1, int(np.ceil(len(indices) / BATCH_SIZE))),
        ):
            atoms, adjacency, mask, batch_extra = batch_fn(graphs, batch_ids, extra)
            si_prediction, pic50_prediction = model(atoms, adjacency, mask, batch_extra)
            output = si_prediction if target_head == "si" else pic50_prediction
            predictions.extend(output.cpu().numpy().tolist())
    return np.asarray(predictions, dtype=float)


def train_model(
    graphs,
    extra,
    target_logsi,
    target_pic50,
    si_train_indices,
    chem_train_indices,
    batch_fn,
    seed,
    lambda_pic50,
    max_epochs=MAX_EPOCHS,
    validation_indices=None,
):
    torch.set_num_threads(4)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = MultiTaskGraphNet(extra.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    si_train_indices = np.asarray(si_train_indices, dtype=int)
    chem_train_indices = np.asarray(chem_train_indices, dtype=int)
    if not len(si_train_indices) or not len(chem_train_indices):
        raise ValueError("Both tasks need at least one training row")

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    best_raw_mse = float("inf")
    wait = 0
    steps = max(
        int(np.ceil(len(si_train_indices) / BATCH_SIZE)),
        int(np.ceil(len(chem_train_indices) / BATCH_SIZE)),
    )
    for epoch in range(1, max_epochs + 1):
        model.train()
        si_order = rng.permutation(si_train_indices)
        chem_order = rng.permutation(chem_train_indices)
        si_batches = np.array_split(si_order, steps)
        chem_batches = np.array_split(chem_order, steps)
        for si_ids, chem_ids in zip(si_batches, chem_batches):
            optimizer.zero_grad(set_to_none=True)
            si_graph_batch = batch_fn(graphs, si_ids, extra)
            chem_graph_batch = batch_fn(graphs, chem_ids, extra)
            si_prediction, _ = model(*si_graph_batch)
            _, pic50_prediction = model(*chem_graph_batch)
            si_loss = F.mse_loss(
                si_prediction,
                torch.as_tensor(target_logsi[si_ids], dtype=torch.float32),
            )
            pic50_loss = F.mse_loss(
                pic50_prediction,
                torch.as_tensor(target_pic50[chem_ids], dtype=torch.float32),
            )
            loss = si_loss + lambda_pic50 * pic50_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite multitask loss")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite gradient")
            optimizer.step()

        if validation_indices is not None:
            log_prediction = batched_predictions(
                model, graphs, extra, validation_indices, batch_fn, "si"
            )
            raw_prediction = np.expm1(np.clip(log_prediction, 0.0, 14.0))
            validation_loss = float(
                mean_squared_error(target_logsi[validation_indices], log_prediction)
            )
            raw_mse = float(
                mean_squared_error(
                    np.expm1(target_logsi[validation_indices]),
                    raw_prediction,
                )
            )
            if validation_loss < best_loss - 1e-8:
                best_loss = validation_loss
                best_epoch = epoch
                best_raw_mse = raw_mse
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
            if wait >= PATIENCE:
                break

    if validation_indices is not None:
        if best_state is None:
            raise RuntimeError("No best model was selected during early stopping")
        model.load_state_dict(best_state)
        return model, best_epoch, best_loss, best_raw_mse
    return model, max_epochs, None, None


def make_inner_split(target_si, groups, outer_train_indices, seed, cpu):
    outer_train_indices = np.asarray(outer_train_indices, dtype=int)
    inner_splits = cpu.make_folds(
        target_si[outer_train_indices],
        groups[outer_train_indices],
        n_folds=5,
        seed=seed,
    )
    inner_train_local, inner_valid_local = inner_splits[0]
    return (
        outer_train_indices[np.asarray(inner_train_local, dtype=int)],
        outer_train_indices[np.asarray(inner_valid_local, dtype=int)],
    )


def run_cv(train, target_si, chembl, batch_fn, graph_fn, cpu, features_fn):
    from src.validation import folds

    target_logsi = np.log1p(target_si)
    chembl_offset = len(train)
    chembl_smiles = chembl["SMILES"].astype(str).tolist()
    chembl_target = chembl["pIC50"].to_numpy(dtype=float)
    full_pic50 = np.full(chembl_offset + len(chembl), np.nan, dtype=float)
    full_pic50[chembl_offset:] = chembl_target
    raw_main = cpu.chem_features(train["SMILES"].astype(str).tolist())
    main_groups = np.asarray(raw_main["canon"], dtype=object)
    chem_groups = np.asarray(
        [canonical(smiles) for smiles in chembl_smiles], dtype=object
    )

    oof_by_seed = {seed: np.full(len(train), np.nan, dtype=float) for seed in SEEDS}
    chosen_lambdas = []
    fold_reports = []
    lambda_validation_mses = {str(value): [] for value in LAMBDAS}

    for seed in SEEDS:
        outer_folds = folds(target_si, main_groups, seed, N_FOLDS)
        for fold_index, (outer_train, outer_valid) in enumerate(outer_folds, start=1):
            outer_train = np.asarray(outer_train, dtype=int)
            outer_valid = np.asarray(outer_valid, dtype=int)
            if set(main_groups[outer_train]) & set(main_groups[outer_valid]):
                raise ValueError("Train structures cross outer CV fold")
            inner_train, inner_valid = make_inner_split(
                target_si,
                main_groups,
                outer_train,
                seed * 100 + fold_index,
                cpu,
            )
            graphs, extra = prepare_model_data(
                train, chembl, inner_train, graph_fn, cpu, features_fn
            )
            valid_groups = set(main_groups[inner_valid]) | set(main_groups[outer_valid])
            inner_chem = np.flatnonzero(
                ~np.isin(chem_groups, list(valid_groups))
            ) + chembl_offset
            if not len(inner_chem):
                raise ValueError("No ChEMBL data remain for inner training")

            tuning = []
            for lambda_pic50 in LAMBDAS:
                model, best_epoch, log_mse, raw_mse = train_model(
                    graphs,
                    extra,
                    target_logsi,
                    full_pic50,
                    inner_train,
                    inner_chem,
                    batch_fn,
                    seed * 10000 + fold_index * 100 + int(lambda_pic50 * 10),
                    lambda_pic50,
                    validation_indices=inner_valid,
                )
                tuning.append(
                    {
                        "lambda": lambda_pic50,
                        "best_epoch": best_epoch,
                        "validation_log_mse": log_mse,
                        "validation_raw_mse": raw_mse,
                    }
                )
                lambda_validation_mses[str(lambda_pic50)].append(raw_mse)

            best_tuning = min(tuning, key=lambda value: value["validation_log_mse"])
            selected_lambda = float(best_tuning["lambda"])
            chosen_lambdas.append(selected_lambda)
            outer_groups = set(main_groups[outer_valid])
            outer_chem = np.flatnonzero(
                ~np.isin(chem_groups, list(outer_groups))
            ) + chembl_offset
            outer_graphs, outer_extra = prepare_model_data(
                train, chembl, outer_train, graph_fn, cpu, features_fn
            )
            model, _, _, _ = train_model(
                outer_graphs,
                outer_extra,
                target_logsi,
                full_pic50,
                outer_train,
                outer_chem,
                batch_fn,
                seed * 10000 + fold_index,
                selected_lambda,
                max_epochs=int(best_tuning["best_epoch"]),
            )
            log_predictions = batched_predictions(
                model, outer_graphs, outer_extra, outer_valid, batch_fn, "si"
            )
            predictions = np.expm1(np.clip(log_predictions, 0.0, 14.0))
            oof_by_seed[seed][outer_valid] = predictions
            fold_mse = float(mean_squared_error(target_si[outer_valid], predictions))
            fold_reports.append(
                {
                    "seed": seed,
                    "fold": fold_index,
                    "lambda": selected_lambda,
                    "early_stopping_epoch": int(best_tuning["best_epoch"]),
                    "inner_tuning": tuning,
                    "outer_mse_raw_si": fold_mse,
                    "outer_train_rows": int(len(outer_train)),
                    "outer_valid_rows": int(len(outer_valid)),
                    "chEMBL_train_rows": int(len(outer_chem)),
                }
            )
            print(
                f"seed={seed} fold={fold_index} lambda={selected_lambda:g} "
                f"raw_SI_MSE={fold_mse:.6g}",
                flush=True,
            )

    for seed, prediction in oof_by_seed.items():
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"Incomplete OOF predictions for seed {seed}")
    averaged_oof = np.mean(np.stack(list(oof_by_seed.values())), axis=0)
    mse = float(mean_squared_error(target_si, averaged_oof))
    return {
        "oof": averaged_oof,
        "mse": mse,
        "fold_reports": fold_reports,
        "selected_lambdas": chosen_lambdas,
        "lambda_validation_mse_mean": {
            key: float(np.mean(values)) for key, values in lambda_validation_mses.items()
        },
    }


def train_submission(
    train, target_si, chembl, best_lambda, batch_fn, graph_fn, cpu, features_fn
):
    from sklearn.model_selection import GroupShuffleSplit

    indices = np.arange(len(train))
    groups = np.asarray(
        [canonical(smiles) for smiles in train["SMILES"].astype(str)],
        dtype=object,
    )
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=4242)
    inner_train, inner_valid = next(splitter.split(indices, target_si, groups))
    inner_train = indices[inner_train]
    inner_valid = indices[inner_valid]
    graphs, extra = prepare_model_data(
        train, chembl, inner_train, graph_fn, cpu, features_fn
    )
    chem_offset = len(train)
    all_groups = np.asarray(
        [canonical(smiles) for smiles in train["SMILES"].astype(str)]
        + [canonical(smiles) for smiles in chembl["SMILES"].astype(str)],
        dtype=object,
    )
    validation_groups = set(groups[inner_valid])
    chem_indices = np.flatnonzero(
        ~np.isin(all_groups[chem_offset:], list(validation_groups))
    ) + chem_offset
    target_logsi = np.log1p(target_si)
    pic50 = np.full(len(train) + len(chembl), np.nan, dtype=float)
    pic50[chem_offset:] = chembl["pIC50"].to_numpy(dtype=float)
    model, best_epoch, _, _ = train_model(
        graphs,
        extra,
        target_logsi,
        pic50,
        inner_train,
        chem_indices,
        batch_fn,
        4242,
        best_lambda,
        validation_indices=inner_valid,
    )

    graphs, extra = prepare_model_data(
        train, chembl, indices, graph_fn, cpu, features_fn
    )
    model, _, _, _ = train_model(
        graphs,
        extra,
        target_logsi,
        pic50,
        indices,
        np.arange(chem_offset, len(pic50)),
        batch_fn,
        4242,
        best_lambda,
        max_epochs=best_epoch,
    )
    model_path = RESULTS / "exp50_multitask_gcn_model.pt"
    train_smiles = (
        train["SMILES"].astype(str).tolist()
        + chembl["SMILES"].astype(str).tolist()
    )
    train_raw_features = cpu.chem_features(train_smiles)
    _, feature_scaler = features_fn(train_raw_features, indices)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_scaler": feature_scaler,
            "extra_dim": int(extra.shape[1]),
            "lambda_pic50": float(best_lambda),
            "epochs": int(best_epoch),
            "training_rows": len(train),
            "chembl_unique_rows": len(chembl),
        },
        model_path,
    )

    test = pd.read_csv(ROOT / "test_clean.csv", usecols=["index", "SMILES"])
    test_graphs = [graph_fn(smiles) for smiles in test["SMILES"].astype(str)]
    raw = cpu.chem_features(
        train["SMILES"].astype(str).tolist()
        + chembl["SMILES"].astype(str).tolist()
        + test["SMILES"].astype(str).tolist()
    )
    test_extra, _ = features_fn(raw, indices)
    test_indices = np.arange(len(train) + len(chembl), len(train) + len(chembl) + len(test))
    all_graphs = graphs + test_graphs
    log_predictions = batched_predictions(
        model, all_graphs, test_extra.astype(np.float32), test_indices, batch_fn, "si"
    )
    predictions = np.expm1(np.clip(log_predictions, 0.0, 14.0))
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
        raise ValueError("Invalid multitask GCN submission")
    path = ROOT / "submission_multitask.csv"
    submission.to_csv(path, index=False)
    return {
        "model": str(model_path.relative_to(ROOT)),
        "submission": str(path.relative_to(ROOT)),
        "rows": len(submission),
        "best_lambda": float(best_lambda),
        "early_stopping_epoch": int(best_epoch),
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
        "prediction_mean": float(predictions.mean()),
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    batch_fn, graph_fn, cpu, features_fn = load_friend_modules()
    train, target_si, chembl, chembl_info = load_data()
    print(
        f"Train={len(train)}, ChEMBL used={len(chembl)} unique, "
        f"test-overlap ChEMBL rows excluded={chembl_info['excluded_test_overlap_rows']}",
        flush=True,
    )
    baseline_path = RESULTS / "exp49_gcn_baseline_oof.npy"
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Missing reproduced baseline OOF: {baseline_path}")
    baseline_oof = np.load(baseline_path)
    if baseline_oof.shape != target_si.shape or not np.isfinite(baseline_oof).all():
        raise ValueError("Reproduced baseline OOF is invalid")
    baseline_mse = float(mean_squared_error(target_si, baseline_oof))
    if abs(baseline_mse - BASELINE_MSE) / BASELINE_MSE > 1e-5:
        raise ValueError("Reproduced baseline OOF differs from the expected reference")

    cv = run_cv(train, target_si, chembl, batch_fn, graph_fn, cpu, features_fn)
    np.save(RESULTS / "exp50_multitask_gcn_oof.npy", cv["oof"])
    improvement = (baseline_mse - cv["mse"]) / baseline_mse
    report = {
        "model": "FriendGraphNet_multitask_GCN",
        "architecture": {
            "atom_encoder": "Linear(10,64) + ReLU",
            "graph_layer": "normalized adjacency aggregation, Linear(64,64) + ReLU",
            "pooling": "masked mean",
            "molecular_features": "Morgan fp2 + MACCS + standardized RDKit descriptors projected to 64",
            "heads": "Linear(64,1) for log1p(SI) and pIC50",
            "loss": "MSE(log1p(SI)) + lambda * MSE(pIC50)",
        },
        "chembl": chembl_info,
        "rows": int(len(train)),
        "oof_mse_raw_si": cv["mse"],
        "baseline_mse_raw_si": baseline_mse,
        "difference_from_baseline": cv["mse"] - baseline_mse,
        "relative_improvement": (baseline_mse - cv["mse"]) / baseline_mse,
        "lambda_candidates": list(LAMBDAS),
        "best_lambda_by_inner_validation_mean_raw_mse": min(
            cv["lambda_validation_mse_mean"],
            key=cv["lambda_validation_mse_mean"].get,
        ),
        "lambda_validation_mse_mean": cv["lambda_validation_mse_mean"],
        "selected_lambdas_by_outer_fold": cv["selected_lambdas"],
        "folds": cv["fold_reports"],
        "oof_path": str((RESULTS / "exp50_multitask_gcn_oof.npy").relative_to(ROOT)),
        "submission": None,
    }
    if improvement > 0:
        best_lambda = float(report["best_lambda_by_inner_validation_mean_raw_mse"])
        report["submission"] = train_submission(
            train, target_si, chembl, best_lambda, batch_fn, graph_fn, cpu, features_fn
        )
    report_path = RESULTS / "exp50_multitask_gcn.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
