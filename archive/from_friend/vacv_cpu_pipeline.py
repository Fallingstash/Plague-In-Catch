# -*- coding: utf-8 -*-
r"""CPU-only VACV SI model: grouped CV, PyTorch neural ensembles, ExtraTrees,
Tanimoto neighbors, saved model, single SMILES and CSV inference.

Do not use CC50_uM or IC50_uM_VACV as features. The single target is SI VACV.

Examples (PowerShell):
  .\.venv\Scripts\python.exe vacv_cpu_pipeline.py train --train-csv "train(1).csv" --test-csv "test(1).csv" --sample-csv "sample_submit(1).csv"
  .\.venv\Scripts\python.exe vacv_cpu_pipeline.py predict --smiles "CCO"
  .\.venv\Scripts\python.exe vacv_cpu_pipeline.py similar --smiles "CCO" --top 5
  .\.venv\Scripts\python.exe vacv_cpu_pipeline.py batch --input "test(1).csv" --output "my_predictions.csv"
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
import copy
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.nn import functional as F
from competition_data import read_competition_csv

RDLogger.DisableLog("rdApp.*")
FP_BITS = 1024
Y_SCALE = 5000.0
DEVICE = torch.device("cpu")  # IMPORTANT: force CPU, even if CUDA is installed.
FGEN2 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS, includeChirality=True)
FGEN3 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=FP_BITS, includeChirality=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def chem_features(smiles):
    f2, f3, mac, desc, canon, scaff = [], [], [], [], [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            raise ValueError(f"Invalid SMILES at row {i}: {s!r}")
        sm = Chem.MolToSmiles(m, isomericSmiles=True)
        canon.append(sm)
        scaff.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m) or sm)
        f2.append(FGEN2.GetFingerprintAsNumPy(m))
        f3.append(FGEN3.GetFingerprintAsNumPy(m))
        mac.append(np.asarray(MACCSkeys.GenMACCSKeys(m)))
        desc.append([
            Descriptors.MolWt(m), Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m),
            rdMolDescriptors.CalcNumHBA(m), rdMolDescriptors.CalcNumHBD(m),
            rdMolDescriptors.CalcNumRotatableBonds(m),
            rdMolDescriptors.CalcNumAromaticRings(m),
            rdMolDescriptors.CalcNumAliphaticRings(m),
            rdMolDescriptors.CalcNumRings(m),
            rdMolDescriptors.CalcFractionCSP3(m),
            rdMolDescriptors.CalcNumHeavyAtoms(m),
            rdMolDescriptors.CalcNumHeteroatoms(m),
            Descriptors.NumValenceElectrons(m), Descriptors.MolMR(m),
            rdMolDescriptors.CalcNumAmideBonds(m),
            rdMolDescriptors.CalcNumAtomStereoCenters(m),
        ])
    return dict(
        fp2=np.asarray(f2, dtype=np.float32),
        fp3=np.asarray(f3, dtype=np.float32),
        maccs=np.asarray(mac, dtype=np.float32),
        desc=np.nan_to_num(np.asarray(desc, dtype=np.float32), nan=0, posinf=0, neginf=0),
        canon=np.array(canon), scaffold=np.array(scaff),
    )


def metadata_from_df(df):
    # Features consist of SMILES and optionally Culture/strain only.
    return df.reindex(columns=["Culture", "strain"], fill_value="unknown").fillna("unknown").astype(str).apply(
        lambda col: col.str.lower().str.strip()
    ).reset_index(drop=True)


def fit_transform_features(raw, meta, tr_idx, smiles_only):
    scaler = StandardScaler().fit(raw["desc"][tr_idx])
    descriptors = np.clip(scaler.transform(raw["desc"]), -8, 8).astype(np.float32)
    parts = [raw["fp2"], raw["maccs"], descriptors]
    if smiles_only:
        encoder = None
    else:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(meta.iloc[tr_idx])
        parts.append(encoder.transform(meta).astype(np.float32))
    x = np.concatenate(parts, axis=1).astype(np.float32)
    return x, scaler, encoder


def transform_features(raw, meta, scaler, encoder):
    desc = np.clip(scaler.transform(raw["desc"]), -8, 8).astype(np.float32)
    parts = [raw["fp2"], raw["maccs"], desc]
    if encoder is not None:
        parts.append(encoder.transform(meta).astype(np.float32))
    return np.concatenate(parts, axis=1).astype(np.float32)


def tanimoto(a, b):
    inter = a @ b.T
    den = a.sum(1, keepdims=True) + b.sum(1)[None, :] - inter
    return np.divide(inter, den, out=np.zeros_like(inter, dtype=np.float32), where=den > 0)


def combined_similarity(ref_raw, query_raw, ref_meta, query_meta, meta_weight):
    sim = .6 * tanimoto(query_raw["fp2"], ref_raw["fp2"]) + .4 * tanimoto(query_raw["fp3"], ref_raw["fp3"])
    if meta_weight and ref_meta is not None and query_meta is not None:
        r = ref_meta.to_numpy(dtype=str)
        q = query_meta.to_numpy(dtype=str)
        m = ((q[:, None, 0] == r[None, :, 0]).astype(np.float32)
             + (q[:, None, 1] == r[None, :, 1]).astype(np.float32)) / 2
        sim = (sim + meta_weight * m) / (1 + meta_weight)
    return sim


def knn(sim, labels, k=20, power=8):
    k = min(k, len(labels))
    pos = np.argpartition(sim, -k, axis=1)[:, -k:]
    ss = np.take_along_axis(sim, pos, axis=1).astype(np.float64)
    w = np.maximum(ss, 1e-6) ** power
    yy = np.asarray(labels, dtype=np.float64)[pos]
    return np.maximum(0, (w * yy).sum(1) / np.maximum(w.sum(1), 1e-20))


class Net(nn.Module):
    """'raw': direct SI, 'log': log(SI+1), 'hurdle': zero-probability * direct SI."""
    def __init__(self, in_dim, width=192, dropout=.20, kind="raw"):
        super().__init__()
        self.kind = kind
        self.body = nn.Sequential(
            nn.Linear(in_dim, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(64, 1)
        self.zero_head = nn.Linear(64, 1) if kind == "hurdle" else None

    def forward(self, x):
        z = self.body(x)
        main = self.head(z).flatten()
        if self.kind == "log":
            return main, None
        main = F.softplus(main)
        if self.kind == "hurdle":
            zero_logit = self.zero_head(z).flatten()
            return main * torch.sigmoid(-zero_logit), zero_logit
        return main, None


def net_predict(model, x):
    model.eval()
    with torch.no_grad():
        pred, _ = model(torch.from_numpy(x))
        pred = pred.numpy().astype(np.float64)
    if model.kind == "log":
        pred = np.expm1(np.clip(pred, 0, np.log1p(40000)))
    else:
        pred = pred * Y_SCALE
    return np.maximum(pred, 0)


def net_loss(model, x, y):
    p, zero_logit = model(x)
    if model.kind == "log":
        return F.mse_loss(p, torch.log1p(y))
    loss = F.mse_loss(p, y / Y_SCALE)
    if model.kind == "hurdle":
        loss = loss + .05 * F.binary_cross_entropy_with_logits(zero_logit, (y == 0).float())
    return loss


def run_net_epochs(x, y, tr_idx, seed, kind, epochs, width, dropout, lr, wd):
    set_seed(seed)
    model = Net(x.shape[1], width, dropout, kind).to(DEVICE)
    xx = torch.from_numpy(x)
    yy = torch.from_numpy(y.astype(np.float32))
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    rg = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        for idx in np.array_split(rg.permutation(tr_idx), max(1, int(np.ceil(len(tr_idx)/64)))):
            loss = net_loss(model, xx[idx], yy[idx])
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optim.step()
    model.eval()
    return model


def early_stopping_epochs(x, y, outer_tr, groups, seed, kind, epochs, patience, width, dropout, lr, wd):
    set_seed(seed)
    splitter = GroupShuffleSplit(n_splits=1, test_size=.18, random_state=seed)
    a, b = next(splitter.split(outer_tr, groups=groups[outer_tr]))
    train_ids, valid_ids = outer_tr[a], outer_tr[b]
    if len(train_ids) < 15 or not len(valid_ids):
        return max(8, epochs // 2)
    model = Net(x.shape[1], width, dropout, kind).to(DEVICE)
    xx = torch.from_numpy(x)
    yy = torch.from_numpy(y.astype(np.float32))
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    rng = np.random.default_rng(seed)
    best_loss, best_epoch, stale = float("inf"), 1, 0
    for ep in range(1, epochs + 1):
        model.train()
        for ids in np.array_split(rng.permutation(train_ids), max(1, int(np.ceil(len(train_ids)/64)))):
            loss = net_loss(model, xx[ids], yy[ids])
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optim.step()
        pred = net_predict(model, x[valid_ids])
        mse = np.mean((pred-y[valid_ids])**2)
        if mse < best_loss - 1e-6:
            best_loss, best_epoch, stale = float(mse), ep, 0
        else:
            stale += 1
        if stale >= patience:
            break
    return max(3, best_epoch)


def make_folds(y, groups, n_folds, seed):
    strata = np.digitize(y, bins=[.001, 10, 100, 1000])
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), strata, groups))


def save_csv(path, sample, predictions):
    if len(sample) != len(predictions):
        raise ValueError("Sample submission/test mismatch")
    arr = np.asarray(predictions, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("NaN or Inf in prediction")
    pd.DataFrame({"index": sample["index"].values, "SI VACV": np.maximum(arr, 0)}).to_csv(path, index=False)


def make_nn_config(kind, args):
    return dict(kind=kind, width=args.width, dropout=args.dropout, lr=args.lr, wd=args.weight_decay)


def train_main(args):
    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    train = read_competition_csv(args.train_csv, "train")
    test = read_competition_csv(args.test_csv, "test")
    sample = pd.read_csv(args.sample_csv)
    assert "SMILES" in train and "SMILES" in test and "SI VACV" in train
    assert "index" in sample and len(sample) == len(test)
    if "index" in test and not np.array_equal(test["index"].to_numpy(), sample["index"].to_numpy()):
        raise ValueError("test/index differs from sample submission")
    y = pd.to_numeric(train["SI VACV"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Invalid target SI VACV")
    if (np.isclose(y, 10000)).sum() > 20:
        raise ValueError("Looks like the OLD WRONG dataset with SI=10000. Use the corrected train(1).csv!")
    print(f"TRAIN {len(train)}, TEST {len(test)} | CPU | zeros={(y == 0).sum()} | max={y.max():.3f}", flush=True)
    raw = chem_features(train["SMILES"].astype(str).tolist())
    meta = metadata_from_df(train)
    n = len(train)
    folds = make_folds(y, raw["canon"][:n] if args.cv == "canonical" else raw["scaffold"][:n], args.folds, args.seed)
    group_labels = raw["canon"][:n] if args.cv == "canonical" else raw["scaffold"][:n]
    kinds = [s.strip() for s in args.nn_models.split(",") if s.strip()]
    if not set(kinds).issubset({"raw", "log", "hurdle"}):
        raise ValueError("nn-models allowed: raw,log,hurdle")
    oof = {k: np.zeros(n, dtype=np.float64) for k in kinds}
    oof.update({"knn": np.zeros(n), "extra_trees": np.zeros(n)})
    fold_episodes = {k: [] for k in kinds}
    fold_no = np.zeros(n, dtype=int)
    meta_w = 0 if args.smiles_only else args.meta_weight

    for fold, (tr, va) in enumerate(folds, 1):
        fold_no[va] = fold
        x, _, _ = fit_transform_features(raw, meta, tr, args.smiles_only)
        tr_raw = {k: v[tr] for k, v in raw.items() if k in ("fp2", "fp3")}
        va_raw = {k: v[va] for k, v in raw.items() if k in ("fp2", "fp3")}
        sim = combined_similarity(tr_raw, va_raw, meta.iloc[tr] if meta_w else None, meta.iloc[va] if meta_w else None, meta_w)
        oof["knn"][va] = knn(sim, y[tr], args.k, args.power)
        tree = ExtraTreesRegressor(n_estimators=250, min_samples_leaf=2, max_features=.8, random_state=args.seed+fold, n_jobs=args.threads)
        tree.fit(x[tr], y[tr])
        oof["extra_trees"][va] = np.maximum(0, tree.predict(x[va]))
        line = [f"Fold {fold}/{len(folds)}", f"KNN {mean_squared_error(y[va],oof['knn'][va]):,.0f}"]
        for kind in kinds:
            preds = []
            for i in range(args.seeds):
                seed = args.seed + 100 * i + 17 * fold
                chosen = early_stopping_epochs(x, y, tr, group_labels, seed, kind,
                    args.epochs, args.patience, args.width, args.dropout, args.lr, args.weight_decay)
                fold_episodes[kind].append(chosen)
                model = run_net_epochs(x, y, tr, seed, kind, chosen,
                    args.width, args.dropout, args.lr, args.weight_decay)
                preds.append(net_predict(model, x[va]))
            oof[kind][va] = np.mean(preds, axis=0)
            line.append(f"NN-{kind} {mean_squared_error(y[va],oof[kind][va]):,.0f}")
        print(" | ".join(line), flush=True)

    # Conservative linear blend search. Select the best base, then allow 10-30% of one other.
    scores = {k: float(mean_squared_error(y, v)) for k, v in oof.items()}
    best_base = min(scores, key=scores.get)
    weights = {k: 0.0 for k in oof}
    weights[best_base] = 1.0
    best_oof = oof[best_base].copy()
    best_mse = scores[best_base]
    for other in oof:
        if other == best_base:
            continue
        for w in (.1, .2, .3):
            p = (1-w)*oof[best_base] + w*oof[other]
            mse = mean_squared_error(y,p)
            if mse < best_mse:
                best_mse, best_oof = float(mse), p
                weights = {k: 0.0 for k in oof}
                weights[best_base], weights[other] = 1-w, w
    print("\nCV MSE:", json.dumps(scores, indent=2), flush=True)
    print(f"ENSEMBLE MSE={best_mse:,.2f} | weights={weights}", flush=True)
    print("NOTE: parameters/weights selected using same CV folds; score may be optimistic.", flush=True)
    print("MEAN BASELINE MSE:", float(np.mean((y - y.mean())**2)), flush=True)

    # Retrain on ALL labeled molecules. Persist trained networks and transform pipeline.
    all_tr = np.arange(n)
    x, scaler, encoder = fit_transform_features(raw, meta, all_tr, args.smiles_only)
    tree = ExtraTreesRegressor(n_estimators=350, min_samples_leaf=2, max_features=.8, random_state=args.seed, n_jobs=args.threads)
    tree.fit(x[:n], y)
    models = []
    final_nets = {kind: [] for kind in kinds}
    for kind in kinds:
        epochs = int(np.clip(np.median(fold_episodes[kind]), 5, args.epochs))
        for i in range(args.seeds):
            seed = args.seed + 1000 + 100*i
            model = run_net_epochs(x, y, all_tr, seed, kind, epochs,
                args.width, args.dropout, args.lr, args.weight_decay)
            final_nets[kind].append(model)
            models.append(dict(kind=kind, width=args.width, dropout=args.dropout,
                state={k: v.detach().cpu().clone() for k,v in model.state_dict().items()}))
        print(f"Final NN-{kind}: {args.seeds} networks x {epochs} epochs", flush=True)
    test_raw = chem_features(test["SMILES"].astype(str).tolist())
    test_meta = metadata_from_df(test)
    x_test = transform_features(test_raw, test_meta, scaler, encoder)
    test_preds = {"extra_trees": np.maximum(0,tree.predict(x_test))}
    test_preds["knn"] = knn(combined_similarity(
        {k:v for k,v in raw.items() if k in ("fp2","fp3")},
        {k:v for k,v in test_raw.items() if k in ("fp2","fp3")},
        meta if meta_w else None, test_meta if meta_w else None, meta_w), y, args.k, args.power)
    for kind in kinds:
        test_preds[kind] = np.mean([net_predict(model,x_test) for model in final_nets[kind]], axis=0)
    ensemble = sum(weights[k] * test_preds[k] for k in weights)
    save_csv(out / "submission.csv", sample, ensemble)
    for name, pred in test_preds.items():
        save_csv(out / f"submission_{name}.csv", sample, pred)
    pd.DataFrame({"fold": fold_no, "true_SI": y, **{f"pred_{k}":v for k,v in oof.items()}, "pred_ensemble":best_oof}).to_csv(out / "oof_predictions.csv",index=False)
    report = dict(scores_mse=scores, ensemble_mse=best_mse, baseline_mse=float(np.mean((y-y.mean())**2)),
        weights=weights, cv=args.cv, seeds=args.seeds, epochs=args.epochs,
        warning="CV reflects selection of blend weights on the same folds; Kaggle score unknown")
    (out / "metrics.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    bundle = dict(
        scaler=scaler, encoder=encoder, tree=tree, nets=models, weights=weights,
        ref_smiles=train.SMILES.astype(str).tolist(), ref_titles=train.Title.astype(str).tolist(),
        ref_si=y, ref_raw={k:v for k,v in raw.items() if k in ("fp2","fp3")},
        ref_meta=meta.copy(), meta_weight=meta_w, k=args.k, power=args.power,
        smiles_only=args.smiles_only,
    )
    joblib.dump(bundle, out / "models" / "model_bundle.joblib", compress=3)
    print(f"\nDONE: {out/'submission.csv'}", flush=True)
    print(f"Saved CPU inference bundle: {out/'models'/'model_bundle.joblib'}", flush=True)


def bundle_predict(b, df):
    if "SMILES" not in df:
        raise ValueError("DataFrame must have SMILES")
    raw = chem_features(df.SMILES.astype(str).tolist())
    meta = metadata_from_df(df)
    x = transform_features(raw, meta, b["scaler"], b["encoder"])
    preds = {}
    if b["weights"].get("extra_trees",0):
        preds["extra_trees"] = np.maximum(0,b["tree"].predict(x))
    if b["weights"].get("knn",0):
        preds["knn"] = knn(combined_similarity(
            b["ref_raw"], raw,
            b["ref_meta"] if b["meta_weight"] else None,
            meta if b["meta_weight"] else None,
            b["meta_weight"]),b["ref_si"],b["k"],b["power"])
    for kind in {m["kind"] for m in b["nets"]}:
        if not b["weights"].get(kind,0):
            continue
        pp=[]
        for d in b["nets"]:
            if d["kind"] != kind:
                continue
            model = Net(x.shape[1],d["width"],d["dropout"],kind)
            model.load_state_dict(d["state"])
            pp.append(net_predict(model,x))
        preds[kind]=np.mean(pp,axis=0)
    return np.maximum(0,sum(b["weights"][k]*v for k,v in preds.items()))


def main():
    p=argparse.ArgumentParser(description="VACV CPU neural + similarity model")
    sub=p.add_subparsers(dest="command",required=True)
    t=sub.add_parser("train")
    t.add_argument("--train-csv",default="train(1).csv")
    t.add_argument("--test-csv",default="test(1).csv")
    t.add_argument("--sample-csv",default="sample_submit(1).csv")
    t.add_argument("--out",default="vacv_cpu_results")
    t.add_argument("--cv",choices=["canonical","scaffold"],default="canonical")
    t.add_argument("--smiles-only",action="store_true",help="use ONLY SMILES features")
    t.add_argument("--nn-models",default="raw,log,hurdle")
    t.add_argument("--folds",type=int,default=5)
    t.add_argument("--seeds",type=int,default=2)
    t.add_argument("--epochs",type=int,default=180)
    t.add_argument("--patience",type=int,default=25)
    t.add_argument("--width",type=int,default=192)
    t.add_argument("--dropout",type=float,default=.20)
    t.add_argument("--lr",type=float,default=.001)
    t.add_argument("--weight-decay",type=float,default=.01)
    t.add_argument("--threads",type=int,default=4)
    t.add_argument("--seed",type=int,default=42)
    t.add_argument("--k",type=int,default=20)
    t.add_argument("--power",type=int,default=8)
    t.add_argument("--meta-weight",type=float,default=.7)
    for command in ("predict","batch","similar"):
        v=sub.add_parser(command)
        v.add_argument("--model",default="vacv_cpu_results/models/model_bundle.joblib")
        v.add_argument("--smiles",default=None)
        v.add_argument("--culture",default="unknown")
        v.add_argument("--strain",default="unknown")
        v.add_argument("--input",default=None)
        v.add_argument("--output",default="predictions.csv")
        v.add_argument("--top",type=int,default=10)
    args=p.parse_args()
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    set_seed(42)
    if args.command=="train":
        train_main(args)
        return
    b=joblib.load(args.model)
    if args.command=="batch":
        if not args.input:
            raise ValueError("--input file.csv required")
        df=read_competition_csv(args.input, "inference")
        pred=bundle_predict(b,df)
        df["SI VACV"] = pred
        df.to_csv(args.output,index=False)
        print(f"Saved {len(df)} predictions in {args.output}")
        return
    if not args.smiles:
        raise ValueError("--smiles required")
    query=pd.DataFrame([dict(SMILES=args.smiles,Culture=args.culture,strain=args.strain)])
    if args.command=="predict":
        print(json.dumps({"SMILES":args.smiles,"predicted_SI_VACV":float(bundle_predict(b,query)[0]),
            "metadata_warning":"Unknown culture/strain may degrade predictions" if (args.culture=="unknown" or args.strain=="unknown") and not b["smiles_only"] else None},ensure_ascii=False,indent=2))
    else:
        raw=chem_features([args.smiles])
        sim=combined_similarity(b["ref_raw"],raw,None,None,0)[0]
        ids=np.argsort(-sim)[:max(1,args.top)]
        for j,i in enumerate(ids,1):
            print(f"{j:2d}. similarity={sim[i]:.3f} | SI={b['ref_si'][i]:.3f} | {b['ref_titles'][i]} | {b['ref_smiles'][i]}")


if __name__=="__main__":
    main()
