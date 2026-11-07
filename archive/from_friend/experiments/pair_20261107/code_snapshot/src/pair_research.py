"""Pairwise delta-SI regression. Pairs and anchors come only from a train fold."""
import argparse
import json
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import Ridge
import vacv_cpu_improved as cpu
from src.validation import ROOT,training_data,folds,metrics,sha,write_json,assert_protected_hashes
from src.features import tanimoto
from versions.test.training import features,transform

WORK=ROOT/'experiments/pair_20261107'
MODES=('anchor_mean','ridge_log','ridge_raw','kernel_log','kernel_raw','neural_log','neural_raw')
SCALE=5000.

def similarity(a,b):
    return .6*tanimoto(a['fp2'],b['fp2'])+.4*tanimoto(a['fp3'],b['fp3'])

def training_pairs(raw,seed,neighbors=8,random_neighbors=2):
    sim=similarity(raw,raw);rng=np.random.default_rng(seed);edges=set();n=len(raw['canon'])
    for b in range(n):
        eligible=np.flatnonzero(raw['canon']!=raw['canon'][b])
        if not len(eligible):continue
        ordered=eligible[np.argsort(-sim[b,eligible],kind='stable')]
        chosen=list(ordered[:neighbors]);remaining=np.setdiff1d(eligible,chosen)
        if len(remaining):chosen.extend(rng.choice(remaining,min(random_neighbors,len(remaining)),replace=False).tolist())
        for a in chosen:edges.add((min(int(a),b),max(int(a),b)))
    if not edges:raise ValueError('Need at least two different training structures')
    directed=sorted([(a,b) for a,b in edges]+[(b,a) for a,b in edges])
    a=np.array([x[0] for x in directed]);b=np.array([x[1] for x in directed]);w=.25+sim[a,b]
    return a,b,w/w.mean()

class PairNet(nn.Module):
    """Odd difference branch × symmetric context: delta(a,b)=-delta(b,a)."""
    def __init__(self,width,hidden=64):
        super().__init__();self.difference=nn.Linear(width,hidden,bias=False);self.context=nn.Linear(width,hidden);self.dropout=nn.Dropout(.1);self.output=nn.Linear(hidden,1,bias=False)
    def forward(self,a,b):
        z=torch.tanh(self.difference(b-a))*torch.sigmoid(self.context((a+b)*.5))
        return self.output(self.dropout(z)).squeeze(1)

class PairRegressor:
    def __init__(self,mode='neural_log',epochs=60):
        if mode not in MODES:raise ValueError('Unknown pair method')
        self.mode=mode;self.epochs=epochs
    def fit(self,frame,y,seed=42,row_ids=None):
        torch.set_num_threads(4);cpu.set_seed(seed);before=np.asarray(y).copy()
        self.reference_raw=cpu.chem_features(frame.SMILES.tolist());self.row_ids=np.arange(len(y)) if row_ids is None else np.asarray(row_ids).copy()
        self.x,self.scaler=features(self.reference_raw,np.arange(len(y)));self.y=np.asarray(y,float).copy()
        self.log=self.mode.endswith('_log');self.z=np.log1p(self.y) if self.log else self.y/SCALE
        self.pair_a,self.pair_b,self.pair_weights=training_pairs(self.reference_raw,seed)
        a,b,w=self.pair_a,self.pair_b,self.pair_weights;delta=self.z[b]-self.z[a];self.history=[]
        if self.mode.startswith('ridge'):
            self.model=Ridge(alpha=10.,fit_intercept=False).fit(self.x[b]-self.x[a],delta,sample_weight=w)
        elif self.mode.startswith('kernel'):
            kernel=similarity(self.reference_raw,self.reference_raw);lap=np.zeros((len(y),len(y)))
            np.add.at(lap,(a,a),w);np.add.at(lap,(b,b),w);np.add.at(lap,(a,b),-w);np.add.at(lap,(b,a),-w)
            lap*=len(y)/len(a)
            self.kernel_coefficients=np.linalg.solve(lap@kernel+np.eye(len(y)),lap@self.z)
            self.reference_potential=kernel@self.kernel_coefficients
        elif self.mode.startswith('neural'):
            self.model=PairNet(self.x.shape[1]);opt=torch.optim.AdamW(self.model.parameters(),lr=.001,weight_decay=.01)
            xx=torch.from_numpy(self.x);yy=torch.tensor(delta,dtype=torch.float32);ww=torch.tensor(w,dtype=torch.float32);rng=np.random.default_rng(seed)
            for ep in range(self.epochs):
                self.model.train();losses=[]
                for batch in np.array_split(rng.permutation(len(a)),max(1,int(np.ceil(len(a)/128)))):
                    output=self.model(xx[a[batch]],xx[b[batch]]);loss=(ww[batch]*(output-yy[batch]).square()).mean()
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite pair loss')
                    opt.zero_grad();loss.backward();norm=nn.utils.clip_grad_norm_(self.model.parameters(),3.)
                    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite pair gradient')
                    opt.step();losses.append(float(loss.detach()))
                self.history.append({'epoch':ep+1,'weighted_pair_training_loss':float(np.mean(losses))})
            self.model.eval()
        np.testing.assert_array_equal(y,before);return self
    def predict_details(self,frame):
        raw=cpu.chem_features(frame.SMILES.tolist());x=transform(raw,self.scaler);sim=similarity(raw,self.reference_raw)
        anchors=np.argsort(-sim,axis=1,kind='stable')[:,:min(5,len(self.y))]
        q=np.repeat(np.arange(len(frame)),anchors.shape[1]);a=anchors.ravel()
        if self.mode=='anchor_mean':delta=np.zeros(len(a))
        elif self.mode.startswith('ridge'):delta=self.model.predict(x[q]-self.x[a])
        elif self.mode.startswith('kernel'):
            potential=sim@self.kernel_coefficients;delta=potential[q]-self.reference_potential[a]
        else:
            with torch.no_grad():
                self.model.eval();delta=[]
                for ids in np.array_split(np.arange(len(a)),max(1,int(np.ceil(len(a)/256)))):
                    delta.extend(self.model(torch.from_numpy(self.x[a[ids]]),torch.from_numpy(x[q[ids]])).numpy())
                delta=np.asarray(delta,float)
        z=self.z[a]+delta
        values=np.expm1(np.clip(z,0,14)) if self.log else np.maximum(z*SCALE,0.)
        values=values.reshape(anchors.shape);w=np.maximum(np.take_along_axis(sim,anchors,axis=1),1e-6)**4
        p=(values*w).sum(1)/w.sum(1)
        if not np.isfinite(p).all() or (p<0).any():raise ValueError('Invalid pair predictions')
        return p,anchors,sim.max(1)
    def predict(self,frame):return self.predict_details(frame)[0]

if __name__=='__main__':sys.modules['src.pair_research']=sys.modules[__name__]
for cls in (PairNet,PairRegressor):cls.__module__='src.pair_research'

def run(seeds=(42,43),modes=MODES):
    assert_protected_hashes();df=training_data();y=df['SI VACV'].to_numpy(float);raw=cpu.chem_features(df.SMILES.tolist());WORK.mkdir(parents=True,exist_ok=True)
    write_json(WORK/'protocol.json',{'training_sha256':sha(ROOT/'train.csv'),'modes':list(MODES),'seeds':[42,43],'folds':5,'epochs':60,
               'pair_neighbors':8,'random_pairs_per_molecule':2,'query_anchors':5,'anchor_weight_power':4,'test_used':False,
               'predeclared_comparisons':['standalone pair','90% reference + 10% pair','70% CPU + 30% pair'],
               'note':'Pairs are correlated transformations of 320 original labels, not new independent observations. No test-derived tuning.'})
    for seed in seeds:
        reference=pd.read_csv(ROOT/f'experiments/research_20261107/runs/canonical_{seed}_reference350/oof.csv').sort_values('row')
        np.testing.assert_allclose(reference.true,y);pairs=folds(y,raw['canon'],seed,5)
        for mode in modes:
            out=WORK/f'{seed}_{mode}';out.mkdir(exist_ok=True)
            if (out/'result.json').exists():continue
            begin=time.monotonic();p=np.full(len(y),np.nan);fold_id=np.zeros(len(y),int);max_sim=np.zeros(len(y));anchor_records=[];counts=[]
            for j,(tr,va) in enumerate(pairs,1):
                checkpoint=out/f'fold_{j}.joblib'
                if checkpoint.exists():model=joblib.load(checkpoint)
                else:
                    model=PairRegressor(mode).fit(df.iloc[tr].reset_index(drop=True),y[tr],seed*100+j-1,tr)
                    joblib.dump(model,checkpoint,compress=3)
                assert set(model.row_ids).isdisjoint(va) and set(model.row_ids)==set(tr)
                pd.DataFrame({'anchor_row':model.row_ids[model.pair_a],'query_row':model.row_ids[model.pair_b],'weight':model.pair_weights}).to_csv(out/f'training_pairs_{j}.csv',index=False)
                p[va],anchor_ids,max_sim[va]=model.predict_details(df.iloc[va].reset_index(drop=True));fold_id[va]=j
                counts.append({'fold':j,'training_molecules':len(tr),'directed_pairs':len(model.pair_a)})
                for i,ids in zip(va,anchor_ids):anchor_records.append({'row':int(i),'fold':j,'anchor_training_rows':json.dumps(model.row_ids[ids].tolist())})
                if model.history:pd.DataFrame(model.history).to_csv(out/f'pair_training_history_{j}.csv',index=False)
                print(f'pair {mode} {seed} fold {j}: MSE {np.mean((y[va]-p[va])**2):.0f}',flush=True)
            np.testing.assert_array_equal(fold_id,reference.fold)
            add=.9*reference.pred.to_numpy()+.1*p;replace=.7*reference.cpu_pred.to_numpy()+.3*p
            pd.DataFrame({'row':np.arange(len(y)),'seed':seed,'fold':fold_id,'true':y,'pair_pred':p,'add10_pred':add,'replace30_pred':replace,'reference_pred':reference.pred,'max_train_similarity':max_sim}).to_csv(out/'oof.csv',index=False)
            pd.DataFrame(anchor_records).sort_values('row').to_csv(out/'query_anchors.csv',index=False)
            write_json(out/'result.json',{'mode':mode,'seed':seed,'pair_metrics':metrics(y,p),'add10_metrics':metrics(y,add),'replace30_metrics':metrics(y,replace),'reference_metrics':metrics(y,reference.pred),
                       'extreme_pair_prediction':float(p[np.argmax(y)]),'extreme_add10_prediction':float(add[np.argmax(y)]),'extreme_replace30_prediction':float(replace[np.argmax(y)]),
                       'pair_counts':counts,'fold_metrics':{str(j):{'pair':metrics(y[fold_id==j],p[fold_id==j]),'add10':metrics(y[fold_id==j],add[fold_id==j]),'replace30':metrics(y[fold_id==j],replace[fold_id==j])} for j in range(1,6)},
                       'training_sha256':sha(ROOT/'train.csv'),'source_sha256':sha(__file__),'seconds':time.monotonic()-begin,'status':'screening_only_no_Kaggle_score'})
            summarize()
    assert_protected_hashes();summarize()

def summarize():
    rows=[]
    for file in WORK.glob('*/result.json'):
        r=json.loads(file.read_text())
        for variant,key in [('standalone','pair_metrics'),('add10','add10_metrics'),('replace30','replace30_metrics')]:
            score=r[key];rows.append({'mode':r['mode'],'seed':r['seed'],'variant':variant,'MSE':score['MSE'],'MAE':score['MAE'],'reference_MSE':r['reference_metrics']['MSE'],
                 'MSE_without_max_DIAGNOSTIC':score['MSE_without_max_DIAGNOSTIC_ONLY'],'extreme_error_share':score['max_target_error_share']})
    if rows:pd.DataFrame(rows).to_csv(ROOT/'reports/pair_model_comparison.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--modes',default=','.join(MODES));p.add_argument('--seeds',default='42,43');a=p.parse_args()
    run(tuple(int(s) for s in a.seeds.split(',')),a.modes.split(','))
