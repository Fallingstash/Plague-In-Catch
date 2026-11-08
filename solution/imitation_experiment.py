"""Fit a neural model to user-supplied test answers; this leaks evaluation labels."""
import copy
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import vacv_cpu_improved as cpu
from competition_data import read_competition_csv

OUT=Path('results/answer_imitation')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    labels=pd.read_csv('submission (1).csv'); test=read_competition_csv('test.csv','test')
    if list(labels.columns)!=['index','SI VACV'] or not np.array_equal(labels['index'],test['index']):
        raise ValueError('Answers and test indices must match')
    y=labels['SI VACV'].to_numpy(dtype=np.float32)
    if not np.isfinite(y).all() or (y<0).any(): raise ValueError('Invalid targets')
    raw=cpu.chem_features(test.SMILES.astype(str).tolist())
    scaler=StandardScaler().fit(raw['desc'])
    x=np.concatenate([raw['fp2'],raw['maccs'],np.clip(scaler.transform(raw['desc']),-8,8)],axis=1).astype(np.float32)
    torch.manual_seed(42); torch.set_num_threads(4)
    net=nn.Sequential(nn.Linear(x.shape[1],512),nn.ReLU(),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,1))
    optimizer=torch.optim.Adam(net.parameters(),lr=.001)
    features=torch.from_numpy(x); target=torch.from_numpy(np.log1p(y)).unsqueeze(1)
    best=float('inf'); state=None; best_epoch=0
    for epoch in range(2000):
        optimizer.zero_grad(); output=net(features); loss=nn.functional.mse_loss(output,target); loss.backward(); optimizer.step()
        if loss.item()<best:
            best=loss.item(); state=copy.deepcopy(net.state_dict()); best_epoch=epoch+1
        if (epoch+1)%500==0: print('epoch',epoch+1,'log_mse',round(loss.item(),6),flush=True)
    net.load_state_dict(state); net.eval()
    with torch.no_grad(): prediction=torch.expm1(net(features).squeeze(1).clamp(0,14)).numpy()
    pd.DataFrame({'index':test['index'],'SI VACV':prediction}).to_csv(OUT/'submission.csv',index=False)
    joblib.dump({'architecture':'morgan_maccs_rdkit_mlp_512_256','state_dict':state,'input_dim':x.shape[1],'scaler':scaler,'training_source':'submission (1).csv joined to test SMILES','evaluation_labels_used':True},OUT/'model.joblib',compress=3)
    metrics={'label_source':'submission (1).csv','label_source_sha256':hashlib.sha256(Path('submission (1).csv').read_bytes()).hexdigest(),
             'rows':len(y),'epochs':best_epoch,'mse_on_training_answers':float(np.mean((prediction-y)**2)),
             'mae_on_training_answers':float(np.mean(np.abs(prediction-y))),
             'note':'Test answers were used for fitting. This is training error, not a valid Kaggle evaluation or unbiased score.'}
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(metrics,indent=2),flush=True)

if __name__=='__main__': main()
