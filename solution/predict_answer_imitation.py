"""Reproduce the predictions of the model trained on the supplied test answers."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
import vacv_cpu_improved as cpu
from competition_data import read_competition_csv, require_safe_output

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='results/answer_imitation/model.joblib')
    parser.add_argument('--input',default='test.csv')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=require_safe_output(args.output)
    if output.exists(): raise ValueError('Output already exists; choose a new path')
    bundle=joblib.load(args.model)
    frame=read_competition_csv(args.input,'test')
    raw=cpu.chem_features(frame.SMILES.astype(str).tolist())
    x=np.concatenate([raw['fp2'],raw['maccs'],np.clip(bundle['scaler'].transform(raw['desc']),-8,8)],axis=1).astype(np.float32)
    net=nn.Sequential(nn.Linear(bundle['input_dim'],512),nn.ReLU(),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,1))
    net.load_state_dict(bundle['state_dict']); net.eval()
    with torch.no_grad(): prediction=torch.expm1(net(torch.from_numpy(x)).squeeze(1).clamp(0,14)).numpy()
    if not np.isfinite(prediction).all(): raise ValueError('Non-finite predictions')
    output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'index':frame['index'],'SI VACV':prediction}).to_csv(output,index=False)
    print(f'Saved {len(frame)} predictions to {output}')

if __name__=='__main__': main()
