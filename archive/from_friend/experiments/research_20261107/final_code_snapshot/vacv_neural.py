# -*- coding: utf-8 -*-
"""VACV SI, CPU-only. Train on the labelled training split and predict test.

PowerShell examples:
  .\\.venv\\Scripts\\python.exe vacv_neural.py train --train train.csv --test test.csv --sample sample_submit.csv --out results_nn
  .\\.venv\\Scripts\\python.exe vacv_neural_full.py predict --model results_nn/models/bundle.joblib --smiles 'CCO'
  .\\.venv\\Scripts\\python.exe vacv_neural_full.py similar --model results_nn/models/bundle.joblib --smiles 'CCO' --top 10
  .\\.venv\\Scripts\\python.exe vacv_neural_full.py batch --model results_nn/models/bundle.joblib --input 'test(1).csv' --output predictions.csv

Only the provided SI VACV column from train.csv is a training target.
Test rows are not included in CV, preprocessing fits, or model fitting.
IC50/CC50 measurements are never model features or imported labels.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import joblib
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.nn import functional as F
from competition_data import read_competition_csv

RDLogger.DisableLog('rdApp.*')
FP_BITS = 1024
FP2 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS, includeChirality=True)
FP3 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=FP_BITS, includeChirality=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def standardized_smiles(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m, isomericSmiles=True) if m is not None else None


def chemical_features(smiles):
    fp2, fp3, maccs, desc, canon, scaf = [], [], [], [], [], []
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            raise ValueError(f"Invalid SMILES at row {i}: {s!r}")
        standard = Chem.MolToSmiles(mol, isomericSmiles=True)
        canon.append(standard)
        scaf.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or standard)
        fp2.append(FP2.GetFingerprintAsNumPy(mol))
        fp3.append(FP3.GetFingerprintAsNumPy(mol))
        maccs.append(np.asarray(MACCSkeys.GenMACCSKeys(mol)))
        desc.append([
            Descriptors.MolWt(mol), Crippen.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumHBA(mol), rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcNumAliphaticRings(mol), rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol), rdMolDescriptors.CalcNumHeavyAtoms(mol),
            rdMolDescriptors.CalcNumHeteroatoms(mol), Descriptors.NumValenceElectrons(mol),
            Descriptors.MolMR(mol), rdMolDescriptors.CalcNumAmideBonds(mol),
            rdMolDescriptors.CalcNumAtomStereoCenters(mol),
        ])
    return {
        'fp2': np.asarray(fp2, dtype=np.float32),
        'fp3': np.asarray(fp3, dtype=np.float32),
        'maccs': np.asarray(maccs, dtype=np.float32),
        'desc': np.nan_to_num(np.asarray(desc, dtype=np.float32), nan=0, posinf=0, neginf=0),
        'canonical': np.asarray(canon),
        'scaffold': np.asarray(scaf),
    }


def normalized_metadata(df):
    # Intentional whitelist: no Title, IC50, CC50, DOI as predictors.
    return (df.reindex(columns=['Culture', 'strain'])
            .fillna('unknown').astype(str)
            .apply(lambda c: c.str.casefold().str.strip())
            .reset_index(drop=True))


def fit_features(chem, meta, train_ids, use_metadata):
    scaler = StandardScaler().fit(chem['desc'][train_ids])
    parts = [chem['fp2'], chem['maccs'],
             np.clip(scaler.transform(chem['desc']), -8, 8)]
    encoder = None
    if use_metadata:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoder.fit(meta.iloc[train_ids])
        parts.append(encoder.transform(meta))
    return np.concatenate(parts, axis=1).astype(np.float32), scaler, encoder


def inference_features(chem, meta, scaler, encoder):
    parts = [chem['fp2'], chem['maccs'], np.clip(scaler.transform(chem['desc']), -8, 8)]
    if encoder is not None:
        parts.append(encoder.transform(meta))
    return np.concatenate(parts, axis=1).astype(np.float32)


class VACVNet(nn.Module):
    """Small regularized neural model with two optional objective functions."""
    def __init__(self, n_inputs, mode='log', hidden=128, dropout=0.3):
        super().__init__()
        self.mode = mode
        self.body = nn.Sequential(
            nn.Linear(n_inputs, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout),
        )
        self.log_head = nn.Linear(64, 1)
        if mode == 'hybrid':
            self.raw_head = nn.Linear(64, 1)
        elif mode != 'log':
            raise ValueError('Unknown model mode')

    def forward(self, x):
        z = self.body(x)
        log_out = self.log_head(z).flatten()
        raw_out = F.softplus(self.raw_head(z).flatten()) * 1000 if self.mode == 'hybrid' else None
        return log_out, raw_out


def model_loss(model, x, y):
    log_out, raw_out = model(x)
    # Stable training for many SI=0 and large dynamic range.
    log_loss = F.smooth_l1_loss(log_out, torch.log1p(y), beta=1.0)
    if model.mode == 'log':
        return log_loss
    # Hybrid considers the competition's raw MSE (but is NOT guaranteed better).
    raw_loss = F.mse_loss(raw_out / 1000.0, y / 1000.0)
    return log_loss + 0.10 * raw_loss


def prediction(model, x):
    model.eval()
    with torch.no_grad():
        log_out, raw_out = model(torch.from_numpy(x))
        if model.mode == 'log':
            # Do not impose arbitrary cap like SI<=500.
            p = torch.expm1(torch.clamp(log_out, min=0, max=14))
        else:
            p = raw_out
        return np.maximum(0, p.numpy().astype(np.float64))


def train_epochs(x, y, ids, epochs, seed, mode, args):
    seed_all(seed)
    model = VACVNet(x.shape[1], mode, args.hidden, args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tx, ty = torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        shuffled = rng.permutation(ids)
        for batch in np.array_split(shuffled, max(1, int(np.ceil(len(ids)/64)))):
            opt.zero_grad()
            loss = model_loss(model, tx[batch], ty[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
    model.eval()
    return model


def select_epochs(x, y, train_ids, canonical, seed, mode, args, chem=None, meta=None):
    # Crucial: inner CV uses only outer-training molecules.
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    ia, ib = next(split.split(train_ids, groups=canonical[train_ids]))
    inner_train, inner_val = train_ids[ia], train_ids[ib]
    if chem is not None:
        x, _, _ = fit_features(chem, meta, inner_train, args.use_metadata)
    seed_all(seed)
    model = VACVNet(x.shape[1], mode, args.hidden, args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tx, ty = torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))
    rng = np.random.default_rng(seed)
    best_mse, best_epoch, stale = float('inf'), 1, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = rng.permutation(inner_train)
        for batch in np.array_split(shuffled, max(1, int(np.ceil(len(shuffled)/64)))):
            opt.zero_grad()
            loss = model_loss(model, tx[batch], ty[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
        mse = float(np.mean((prediction(model, x[inner_val]) - y[inner_val]) ** 2))
        if np.isfinite(mse) and mse < best_mse - 1e-6:
            best_mse, best_epoch, stale = mse, epoch, 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    return best_epoch


def cv_split(y, groups, folds, seed):
    bins = np.digitize(y, [0.000001, 1, 10, 100, 1000])
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), bins, groups))


def fit_train(args):
    from competition_data import require_safe_output
    require_safe_output(args.out)
    if args.use_metadata:
        raise ValueError('Competition model inputs are SMILES only; Culture/strain metadata are disabled')
    torch.set_num_threads(max(1, args.threads))
    seed_all(args.seed)
    original = read_competition_csv(args.train, 'train')
    test = read_competition_csv(args.test, 'test')
    sample = pd.read_csv(args.sample)
    if not {'SMILES', 'SI VACV'}.issubset(original) or 'SMILES' not in test:
        raise ValueError('Training must contain SMILES and SI VACV; test must contain SMILES')
    if len(sample) != len(test) or not {'index', 'SI VACV'}.issubset(sample):
        raise ValueError('Invalid sample submission format')
    if 'index' in test and not np.array_equal(test['index'].to_numpy(), sample['index'].to_numpy()):
        raise ValueError('test index != sample index')
    if 'SI VACV' in test:
        raise ValueError('Refusing test table that contains target SI VACV')
    y_original = pd.to_numeric(original['SI VACV'], errors='raise').to_numpy(np.float64)
    if not np.all(np.isfinite(y_original)) or np.any(y_original < 0):
        raise ValueError('Invalid SI values')
    if np.isclose(y_original, 10000).sum() > 20:
        raise ValueError('Seems old dataset: SI=10000 for inactive rows. Use corrected train(1).csv')
    n_original = len(original)
    main = original
    chem = chemical_features(main['SMILES'].astype(str).tolist())
    meta = normalized_metadata(main)
    y = pd.to_numeric(main['SI VACV']).to_numpy(np.float64)
    original_groups = chem['canonical'] if args.cv == 'canonical' else chem['scaffold']
    folds = cv_split(y_original, original_groups, args.folds, args.seed)
    oof = {name: np.zeros(n_original, dtype=np.float64) for name in ('log', 'hybrid')}
    epochs_by_mode = {name: [] for name in oof}
    fold_report = []
    print(f'Train {n_original} | test reserved for inference ({len(test)} rows) | zeros {(y_original==0).sum()} | CPU', flush=True)
    for i, (tr, va) in enumerate(folds, start=1):
        ids_train = tr
        x, _, _ = fit_features(chem, meta, ids_train, args.use_metadata)
        info = {'fold': i, 'original_validation': len(va)}
        for mode in oof:
            preds = []
            for seed_id in range(args.seeds):
                s = args.seed + i * 997 + seed_id * 79 + (50000 if mode == 'hybrid' else 0)
                best_epoch = select_epochs(x, y, ids_train, chem['canonical'], s, mode, args, chem, meta)
                epochs_by_mode[mode].append(best_epoch)
                net = train_epochs(x, y, ids_train, best_epoch, s, mode, args)
                preds.append(prediction(net, x[va]))
            oof[mode][va] = np.mean(preds, axis=0)
            info[f'{mode}_MSE'] = float(mean_squared_error(y_original[va], oof[mode][va]))
        fold_report.append(info)
        print(f"Fold {i}/{len(folds)} | log MSE={info['log_MSE']:,.0f} | hybrid MSE={info['hybrid_MSE']:,.0f}", flush=True)
    scores = {name: float(mean_squared_error(y_original, pred)) for name, pred in oof.items()}
    selected = min(scores, key=scores.get)
    output = Path(args.out)
    (output / 'models').mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'SI_true': y_original, **{f'pred_{k}': p for k,p in oof.items()}}).to_csv(output/'oof_predictions.csv', index=False)
    report = {'OOF_MSE': scores, 'selected': selected, 'mean_baseline_MSE': float(np.mean((y_original-y_original.mean())**2)),
              'CV': args.cv, 'training_n': n_original, 'external_n': 0, 'test_used_for_training': False, 'folds': fold_report,
              'epochs': epochs_by_mode,
              'note': 'Model selection using same folds is slightly optimistic; Kaggle result can differ. No test labels used.'}
    (output / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('OOF CV MSE', json.dumps(scores, ensure_ascii=False), 'selected:', selected, flush=True)
    all_train_ids = np.arange(n_original)
    x, scaler, encoder = fit_features(chem, meta, all_train_ids, args.use_metadata)
    final_epochs = max(5, int(np.median(epochs_by_mode[selected])))
    final_models, final_nets = [], []
    for j in range(args.seeds):
        s = args.seed + 90000 + j * 79
        net = train_epochs(x, y, all_train_ids, final_epochs, s, selected, args)
        final_nets.append(net)
        final_models.append({
            'mode': selected, 'state_dict': {k:v.detach().cpu().clone() for k,v in net.state_dict().items()},
            'hidden': args.hidden, 'dropout': args.dropout, 'seed': s, 'epochs': final_epochs,
        })
    # Test structures are featurized only after all fitting and CV decisions are complete.
    test_chem = chemical_features(test['SMILES'].astype(str).tolist())
    test_meta = normalized_metadata(test)
    x_test = inference_features(test_chem, test_meta, scaler, encoder)
    predicted = [prediction(net, x_test) for net in final_nets]
    predictions = np.mean(predicted, axis=0)
    pd.DataFrame({'index': sample['index'], 'SI VACV': predictions}).to_csv(output/'submission.csv', index=False)
    bundle = {'models': final_models, 'selected': selected, 'scaler': scaler, 'encoder': encoder,
              'training_smiles': main['SMILES'].astype(str).tolist(), 'training_SI': y,
              'training_titles': main['Title'].fillna('').astype(str).tolist() if 'Title' in main else ['']*n_original,
              'training_fp2': chem['fp2'], 'training_fp3': chem['fp3'],
              'source_doi': main['DOI'].fillna('').astype(str).tolist() if 'DOI' in main else ['']*n_original,
              'use_metadata': args.use_metadata}
    joblib.dump(bundle, output/'models'/'bundle.joblib', compress=3)
    print(f'Saved {output / "submission.csv"}; model: {output / "models" / "bundle.joblib"}', flush=True)


def load_model_predictions(bundle, frame):
    if 'SMILES' not in frame:
        raise ValueError('Prediction table must have SMILES')
    chem = chemical_features(frame['SMILES'].astype(str).tolist())
    x = inference_features(chem, normalized_metadata(frame), bundle['scaler'], bundle['encoder'])
    arr = []
    for obj in bundle['models']:
        net = VACVNet(x.shape[1], obj['mode'], obj['hidden'], obj['dropout'])
        net.load_state_dict(obj['state_dict'])
        arr.append(prediction(net, x))
    return np.mean(arr, axis=0)


def similar(bundle, smiles, top):
    query = chemical_features([smiles])
    def tanimoto(A, B):
        inter = A @ B.T
        union = A.sum(axis=1, keepdims=True) + B.sum(axis=1)[None,:] - inter
        return np.divide(inter, union, out=np.zeros_like(inter), where=union>0).ravel()
    s = .6*tanimoto(bundle['training_fp2'], query['fp2']) + .4*tanimoto(bundle['training_fp3'], query['fp3'])
    ids = np.argsort(-s)[:max(1, min(top, len(s)))]
    return [{'similarity_Tanimoto': round(float(s[i]),4), 'SMILES': bundle['training_smiles'][i],
             'SI_observed':float(bundle['training_SI'][i]),'Title':bundle['training_titles'][i],
             'DOI':bundle['source_doi'][i]} for i in ids]


def main():
    p = argparse.ArgumentParser(description='VACV QSAR CPU neural network pipeline')
    commands = p.add_subparsers(dest='cmd', required=True)
    t = commands.add_parser('train')
    t.add_argument('--train', default='train.csv')
    t.add_argument('--test', default='test.csv')
    t.add_argument('--sample', default='sample_submit.csv')
    t.add_argument('--out', default='results_nn')
    t.add_argument('--cv', choices=['canonical', 'scaffold'], default='canonical')
    t.add_argument('--folds', type=int, default=5)
    t.add_argument('--seeds', type=int, default=2)
    t.add_argument('--epochs', type=int, default=180)
    t.add_argument('--patience', type=int, default=30)
    t.add_argument('--hidden', type=int, default=128)
    t.add_argument('--dropout', type=float, default=0.30)
    t.add_argument('--lr', type=float, default=0.001)
    t.add_argument('--weight-decay', type=float, default=0.02)
    t.add_argument('--seed', type=int, default=42)
    t.add_argument('--threads', type=int, default=4)
    t.add_argument('--use-metadata', action='store_true', help='Add Culture and strain as model inputs')
    for name in ('predict', 'batch', 'similar'):
        c = commands.add_parser(name)
        c.add_argument('--model', default='results_nn/models/bundle.joblib')
        c.add_argument('--smiles')
        c.add_argument('--input')
        c.add_argument('--output', default='predictions.csv')
        c.add_argument('--top', type=int, default=10)
        c.add_argument('--culture', default='unknown')
        c.add_argument('--strain', default='unknown')
    a = p.parse_args()
    if a.cmd == 'train':
        fit_train(a)
    else:
        bundle = joblib.load(a.model)   # Only load files you created/trust.
        if a.cmd == 'batch':
            if not a.input:
                raise ValueError('batch requires --input')
            df = read_competition_csv(a.input, 'inference')
            df['predicted_SI_VACV'] = load_model_predictions(bundle, df)
            from competition_data import require_safe_output
            require_safe_output(a.output)
            df.to_csv(a.output, index=False)
            print(f'Saved {len(df)} predictions -> {a.output}')
        else:
            if not a.smiles:
                raise ValueError('Specify --smiles')
            if a.cmd == 'similar':
                result = similar(bundle, a.smiles, a.top)
            else:
                frame = pd.DataFrame([{'SMILES': a.smiles, 'Culture': a.culture, 'strain': a.strain}])
                result = {'SMILES': a.smiles, 'predicted_SI_VACV': float(load_model_predictions(bundle, frame)[0])}
            print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
