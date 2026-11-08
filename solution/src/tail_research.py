"""Train-only experiments addressing large SI errors without editing targets."""
import argparse
import json
import time
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
import vacv_cpu_improved as cpu
from versions.test.training import GraphNet,graph,batch,features,transform
from src.validation import ROOT,training_data,folds,metrics,sha,write_json,assert_protected_hashes

WORK=ROOT/'experiments/tail_20261107'
MODES=('control_log','raw_mse','root_mse','weighted_log','log_raw1','log_raw10','dual_raw','mixture')
SCALE=5000.0

def decode(output,mode):
    if mode=='raw_mse':return nn.functional.softplus(output)*SCALE
    if mode=='root_mse':return output.clamp_min(0).square()*SCALE
    if mode=='dual_raw':return nn.functional.softplus(output[:,1])*SCALE
    if mode=='mixture':
        gate=torch.sigmoid(output[:,0]);low=torch.expm1(output[:,1].clamp(0,14));high=torch.expm1(output[:,2].clamp(0,14))
        return (1-gate)*low+gate*high
    # Same numerical link as the original GCN, not a ceiling selected using test.
    return torch.expm1(output.clamp(0,14))

def objective(output,y,mode,weight_scale,tail_threshold):
    logy=torch.log1p(y)
    if mode=='raw_mse':return nn.functional.mse_loss(decode(output,mode)/SCALE,y/SCALE)
    if mode=='root_mse':return nn.functional.mse_loss(output,torch.sqrt(y/SCALE))
    if mode=='weighted_log':
        w=1+torch.sqrt(y/weight_scale)
        return (w*(output-logy).square()).sum()/w.sum()
    if mode=='dual_raw':
        return nn.functional.mse_loss(output[:,0],logy)+nn.functional.mse_loss(decode(output,mode)/SCALE,y/SCALE)
    if mode=='mixture':
        high=y>=tail_threshold;low=~high
        loss=nn.functional.binary_cross_entropy_with_logits(output[:,0],high.float())
        if low.any():loss=loss+nn.functional.mse_loss(output[low,1],logy[low])
        if high.any():loss=loss+nn.functional.mse_loss(output[high,2],logy[high])
        return loss+nn.functional.mse_loss(decode(output,mode)/SCALE,y/SCALE)
    loss=nn.functional.mse_loss(output,logy)
    if mode in ('log_raw1','log_raw10'):
        loss=loss+(1. if mode=='log_raw1' else 10.)*nn.functional.mse_loss(decode(output,mode)/SCALE,y/SCALE)
    return loss

class TailRegressor:
    def __init__(self,mode='control_log',epochs=80):
        if mode not in MODES:raise ValueError('Unknown tail model')
        self.mode=mode;self.epochs=epochs
    def fit(self,frame,y,seed):
        torch.set_num_threads(4);cpu.set_seed(seed);before=np.asarray(y).copy()
        raw=cpu.chem_features(frame.SMILES.tolist());ids=np.arange(len(y));extra,self.scaler=features(raw,ids)
        graphs=[graph(s) for s in frame.SMILES];positive=np.asarray(y)[np.asarray(y)>0]
        self.weight_scale=max(float(np.quantile(positive,.75)),1.) if len(positive) else 1.
        self.tail_threshold=float(np.quantile(positive,.9)) if len(positive) else 1.
        self.training_target_max=float(np.max(y));self.training_rows=len(y);self.model=GraphNet(extra.shape[1])
        if self.mode in ('dual_raw','mixture'):self.model.head[-1]=nn.Linear(64,2 if self.mode=='dual_raw' else 3)
        opt=torch.optim.AdamW(self.model.parameters(),lr=.001,weight_decay=.01)
        yy=torch.tensor(y,dtype=torch.float32);rng=np.random.default_rng(seed);self.history=[]
        for epoch in range(self.epochs):
            self.model.train();losses=[]
            for b in np.array_split(rng.permutation(ids),max(1,int(np.ceil(len(ids)/32)))):
                output=self.model(*batch(graphs,b,extra));loss=objective(output,yy[b],self.mode,self.weight_scale,self.tail_threshold)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                opt.zero_grad();loss.backward();norm=nn.utils.clip_grad_norm_(self.model.parameters(),3.)
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
                opt.step();losses.append(float(loss.detach()))
            self.history.append({'epoch':epoch+1,'mean_training_loss':float(np.mean(losses)),'largest_batch_loss':max(losses)})
        np.testing.assert_array_equal(y,before)
        self.model.eval();return self
    def predict(self,frame):
        raw=cpu.chem_features(frame.SMILES.tolist());extra=transform(raw,self.scaler);graphs=[graph(s) for s in frame.SMILES]
        result=[];self.model.eval()
        with torch.no_grad():
            for b in np.array_split(np.arange(len(frame)),max(1,int(np.ceil(len(frame)/32)))):
                result.extend(decode(self.model(*batch(graphs,b,extra)),self.mode).numpy())
        result=np.asarray(result,float)
        if not np.isfinite(result).all() or (result<0).any():raise ValueError('Invalid prediction')
        return result

if __name__=='__main__':sys.modules['src.tail_research']=sys.modules[__name__]
TailRegressor.__module__='src.tail_research'
if not hasattr(sys.modules['__main__'],'TailRegressor'):sys.modules['__main__'].TailRegressor=TailRegressor

def run(seeds=(42,43),modes=MODES):
    assert_protected_hashes();df=training_data();y=df['SI VACV'].to_numpy(float);before=y.copy();raw=cpu.chem_features(df.SMILES.tolist())
    WORK.mkdir(parents=True,exist_ok=True)
    write_json(WORK/'protocol.json',{'modes':list(MODES),'epochs':80,'scale':SCALE,'grouped_folds':5,'seeds':[42,43],
               'training_sha256':sha(ROOT/'train.csv'),'test_used':False,'targets_modified':False,
               'note':'Fixed configurations. Screening results are not independent after selecting a winner; no automatic submission or test-based clipping.'})
    for seed in seeds:
        pairs=folds(y,raw['canon'],seed,5);reference=pd.read_csv(ROOT/f'experiments/research_20261107/runs/canonical_{seed}_reference350/oof.csv').sort_values('row')
        np.testing.assert_allclose(reference.true,y)
        for mode in modes:
            out=WORK/f'{seed}_{mode}';out.mkdir(exist_ok=True)
            if (out/'result.json').exists():continue
            begin=time.monotonic();p=np.full(len(y),np.nan);fold_ids=np.zeros(len(y),int);training_max=np.zeros(len(y));train_metrics=[]
            for j,(tr,va) in enumerate(pairs):
                checkpoint=out/f'fold_{j+1}.joblib'
                if checkpoint.exists():model=joblib.load(checkpoint)
                else:
                    model=TailRegressor(mode).fit(df.iloc[tr].reset_index(drop=True),y[tr],seed*100+j)
                    joblib.dump(model,checkpoint,compress=3)
                p[va]=model.predict(df.iloc[va].reset_index(drop=True));fold_ids[va]=j+1;training_max[va]=model.training_target_max
                train_metrics.append({'fold':j+1,'training_MSE':float(np.mean((y[tr]-model.predict(df.iloc[tr].reset_index(drop=True)))**2))})
                pd.DataFrame(model.history).to_csv(out/f'training_history_{j+1}.csv',index=False)
                print(f'tail {mode} seed {seed} fold {j+1}: MSE {np.mean((y[va]-p[va])**2):.0f}',flush=True)
            np.testing.assert_array_equal(fold_ids,reference.fold)
            blend=.7*reference.cpu_pred.to_numpy()+.3*p
            pd.DataFrame({'row':np.arange(len(y)),'seed':seed,'fold':fold_ids,'true':y,'new_gcn_pred':p,'blend_pred':blend,
                          'reference_pred':reference.pred,'training_target_max':training_max}).to_csv(out/'oof.csv',index=False)
            report={'mode':mode,'seed':seed,'gcn_metrics':metrics(y,p),'blend_metrics':metrics(y,blend),'reference_metrics':metrics(y,reference.pred),
                    'training_metrics':train_metrics,'fold_metrics':{str(j):metrics(y[fold_ids==j],blend[fold_ids==j]) for j in range(1,6)},
                    'extreme_row':int(np.argmax(y)),'extreme_prediction':float(blend[np.argmax(y)]),'training_sha256':sha(ROOT/'train.csv'),
                    'code_sha256':sha(__file__),'seconds':time.monotonic()-begin,'status':'screening_only_no_kaggle_score'}
            write_json(out/'result.json',report);summarize()
    np.testing.assert_array_equal(y,before);assert_protected_hashes();summarize()

def summarize():
    records=[]
    for file in WORK.glob('*/result.json'):
        r=json.loads(file.read_text());records.append({'mode':r['mode'],'seed':r['seed'],'blend_MSE':r['blend_metrics']['MSE'],
            'reference_MSE':r['reference_metrics']['MSE'],'gcn_MSE':r['gcn_metrics']['MSE'],'MAE':r['blend_metrics']['MAE'],
            'extreme_prediction':r['extreme_prediction'],'MSE_without_extreme_DIAGNOSTIC':r['blend_metrics']['MSE_without_max_DIAGNOSTIC_ONLY'],
            'extreme_error_share':r['blend_metrics']['max_target_error_share']})
    if records:pd.DataFrame(records).to_csv(ROOT/'reports/tail_model_comparison.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',default='42,43');p.add_argument('--modes',default=','.join(MODES));a=p.parse_args()
    run(tuple(int(s) for s in a.seeds.split(',')),a.modes.split(','))
