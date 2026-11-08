"""Train a reproducible CUDA/CPU molecular fingerprint model and export CSV predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import GroupKFold
from competition_data import read_competition_csv, require_safe_output
from src.validation import training_data, metrics, sha

BITS = 2048

def fingerprints(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=BITS, includeChirality=True)
    arrays, groups = [], []
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(s))
        if mol is None or not mol.GetNumAtoms():
            raise ValueError(f'Invalid SMILES at row {i + 2}: {s}')
        groups.append(Chem.MolToSmiles(mol, isomericSmiles=True))
        arrays.append(gen.GetFingerprintAsNumPy(mol).astype(np.float32))
    if not arrays:
        raise ValueError('Input CSV is empty')
    return np.stack(arrays), np.array(groups)

def network():
    return torch.nn.Sequential(torch.nn.Linear(BITS, 128), torch.nn.ReLU(), torch.nn.Dropout(0.15), torch.nn.Linear(128, 1))

def fit(x, y, epochs, seed, device):
    torch.manual_seed(seed)
    net = network().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01)
    features = torch.as_tensor(x, device=device)
    target = torch.as_tensor(np.log1p(y).astype(np.float32), device=device).reshape(-1, 1)
    for _ in range(epochs):
        net.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(net(features), target)
        loss.backward()
        optimizer.step()
    return net.cpu().eval()

def predict(net, x, device):
    net = net.to(device).eval()
    with torch.no_grad():
        z = net(torch.as_tensor(x, device=device)).flatten().cpu().numpy()
    return np.expm1(np.clip(z, 0, 20))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train')
    train.add_argument('--train', default='train.csv')
    train.add_argument('--out', required=True)
    train.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    train.add_argument('--epochs', type=int, default=150)
    train.add_argument('--folds', type=int, default=5)
    train.add_argument('--seed', type=int, default=42)
    infer = commands.add_parser('predict')
    infer.add_argument('--model', required=True)
    infer.add_argument('--input', required=True)
    infer.add_argument('--output', required=True)
    infer.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    args = parser.parse_args()
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
    if device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable in the installed PyTorch')
    if args.command == 'train':
        if args.epochs < 1 or args.folds < 2:
            parser.error('epochs must be positive and folds must be at least 2')
        out = require_safe_output(args.out)
        out.mkdir(parents=True, exist_ok=False)
        df = training_data(args.train)
        x, groups = fingerprints(df.SMILES)
        y = df['SI VACV'].to_numpy(float)
        oof = np.zeros(len(y))
        fold_id = np.zeros(len(y), dtype=int)
        for i, (tr, va) in enumerate(GroupKFold(n_splits=args.folds).split(x, y, groups)):
            net = fit(x[tr], y[tr], args.epochs, args.seed + i, device)
            oof[va] = predict(net, x[va], device)
            fold_id[va] = i
        report = {'device': device, 'train_sha256': sha(args.train), 'epochs': args.epochs, 'folds': args.folds, 'seed': args.seed, 'metrics': metrics(y, oof), 'note': 'Experimental model; grouped OOF is not an independent benchmark after model selection.'}
        pd.DataFrame({'row': np.arange(len(y)), 'fold': fold_id, 'observed': y, 'predicted': oof}).to_csv(out / 'oof.csv', index=False)
        net = fit(x, y, args.epochs, args.seed, device)
        torch.save({'state_dict': net.state_dict(), 'bits': BITS, 'train_sha256': report['train_sha256']}, out / 'model.pt')
        (out / 'metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        output = require_safe_output(args.output)
        if output.exists() or output.resolve() in (Path(args.input).resolve(), Path(args.model).resolve()):
            raise FileExistsError('Output must be a new file')
        df = read_competition_csv(args.input, 'inference')
        if 'SMILES' not in df: raise ValueError('CSV requires a SMILES column')
        x, _ = fingerprints(df.SMILES)
        checkpoint = torch.load(args.model, map_location='cpu', weights_only=True)
        if checkpoint['bits'] != BITS: raise ValueError('Unsupported fingerprint size')
        net = network(); net.load_state_dict(checkpoint['state_dict'])
        df['predicted_SI_VACV'] = predict(net, x, device)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        print(f'Exported {len(df)} predictions to {output}')

if __name__ == '__main__':
    main()
