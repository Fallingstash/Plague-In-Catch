"""Experimental two-stage continuation on provisional ChEMBL pairs, not for contest use."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from competition_data import read_competition_csv, require_safe_output
from src.validation import training_data, sha
from src.gpu_training import fingerprints, network, predict, BITS


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {'mse': float(np.mean((y-p)**2)), 'mae': float(np.mean(np.abs(y-p))),
            'log_mae': float(np.mean(np.abs(np.log1p(y)-np.log1p(p))))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--external', default='data/external/processed/chembl_quarantine.csv')
    ap.add_argument('--checkpoint', default='results/distill_top1_20261108/student.pt')
    ap.add_argument('--teacher-train', default='results/distill_top1_20261108/train_predictions.csv')
    ap.add_argument('--train', default='train.csv')
    ap.add_argument('--test', default='test.csv')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device
    if device == 'cuda' and not torch.cuda.is_available(): ap.error('CUDA unavailable')
    out = require_safe_output(a.out); out.mkdir(parents=True, exist_ok=False)
    official = training_data(a.train)
    ref = pd.read_csv(a.teacher_train)
    if len(ref) != len(official) or not np.array_equal(ref.SMILES, official.SMILES):
        raise ValueError('Teacher predictions do not match official train')
    ext = pd.read_csv(a.external)
    required = {'SMILES','SI VACV','ic50_relation','cc50_relation','DOI','source_url'}
    if not required.issubset(ext.columns): raise ValueError('External provenance fields missing')
    ext['SI VACV'] = pd.to_numeric(ext['SI VACV'], errors='coerce')
    valid = np.isfinite(ext['SI VACV']) & (ext['SI VACV'] >= 0) & (ext.ic50_relation == '=') & (ext.cc50_relation == '=') & ext.DOI.notna() & ext.source_url.notna()
    ext = ext.loc[valid].copy().reset_index(drop=True)
    ex, groups = fingerprints(ext.SMILES)
    ox, og = fingerprints(official.SMILES)
    _, tg = fingerprints(read_competition_csv(a.test, 'test').SMILES)
    keep = ~np.isin(groups, np.concatenate([og,tg]))
    ext, ex, groups = ext.loc[keep].reset_index(drop=True), ex[keep], groups[keep]
    if len(set(groups)) < 5: raise ValueError('Not enough distinct non-overlapping external molecules')
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=a.seed).split(ex, groups=groups))
    ext['split'] = 'validation'; ext.loc[tr, 'split'] = 'train'
    ext.to_csv(out/'external_used.csv', index=False)
    target = ext['SI VACV'].to_numpy(float)
    torch.manual_seed(a.seed)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('bits') != BITS or checkpoint.get('train_sha256') != sha(a.train): raise ValueError('Unexpected starting checkpoint')
    net = network(); net.load_state_dict(checkpoint['state_dict']); net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=0.0001, weight_decay=0.01)
    x = torch.as_tensor(ex[tr], device=device); y = torch.as_tensor(np.log1p(target[tr]).astype(np.float32), device=device).view(-1,1)
    tx = torch.as_tensor(ox, device=device); ty = torch.as_tensor(np.log1p(ref.teacher.to_numpy(np.float32)), device=device).view(-1,1)
    report = {'source_sha256':sha(a.external), 'checkpoint_sha256':sha(a.checkpoint), 'teacher_train_sha256':sha(a.teacher_train), 'train_sha256':sha(a.train), 'device':device, 'seed':a.seed, 'external_rows_after_filter':len(ext), 'external_train_rows':len(tr), 'external_validation_rows':len(va), 'external_validation_unique_molecules':len(set(groups[va])), 'warning':'Provisional computed SI; no primary-source verification. This model is not eligible for the documented competition protocol. Validation is small and teacher was trained on official train.', 'stages':{}}
    def evaluate(name):
        prediction = predict(net, ex[va], device)
        refpred = predict(net, ox, device)
        report['stages'][name] = {'validation_vs_provisional_SI':metrics(target[va],prediction), 'official_train_vs_teacher':metrics(ref.teacher, refpred)}
        pd.DataFrame({'SMILES':ext.SMILES.iloc[va].to_numpy(), 'provisional_SI':target[va], 'predicted':prediction}).to_csv(out/(name+'_validation.csv'),index=False)
    evaluate('baseline')
    for stage in (1,2):
        net.to(device).train()
        for _ in range(100):
            net.train(); opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(net(x),y) + 0.5*torch.nn.functional.mse_loss(net(tx),ty)
            loss.backward(); opt.step()
        torch.save({'state_dict':{k:v.detach().cpu() for k,v in net.state_dict().items()}, 'bits':BITS, 'train_sha256':sha(a.train), 'external_sha256':sha(a.external), 'epoch':stage*100}, out/f'stage{stage}.pt')
        evaluate(f'stage{stage}')
        print(f'Completed {stage*100} epochs', flush=True)
    (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
