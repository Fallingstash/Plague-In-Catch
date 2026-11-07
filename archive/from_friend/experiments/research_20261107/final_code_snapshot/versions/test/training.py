"""GCN-style molecular regression, CPU, grouped out-of-fold comparison."""
import copy, json, random, hashlib, joblib, shutil
from datetime import datetime, timezone
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import argparse
import numpy as np, pandas as pd, torch
from torch import nn
from rdkit import Chem
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import vacv_cpu_improved as cpu
from competition_data import read_competition_csv, require_safe_output

torch.set_num_threads(4)
class GraphNet(nn.Module):
 def __init__(self,extra_dim=0,layers=1):
  super().__init__(); self.layers=layers; self.first=nn.Linear(10,64); self.conv=nn.Linear(64,64); self.conv2=nn.Linear(64,64) if layers==2 else None; self.head=nn.Sequential(nn.Linear(64+extra_dim,64),nn.GELU(),nn.Dropout(.2),nn.Linear(64,1))
 def forward(self,atoms,adj,mask,extra):
  x=torch.relu(self.first(atoms)); x=torch.relu(self.conv(torch.bmm(adj,x)))
  if self.conv2 is not None: x=x+torch.relu(self.conv2(torch.bmm(adj,x)))
  x=(x*mask[:,:,None]).sum(dim=1)/mask.sum(dim=1,keepdim=True).clamp_min(1)
  if extra is not None: x=torch.cat([x,extra],dim=1)
  return self.head(x).squeeze(1)

def graph(smiles):
 mol=Chem.MolFromSmiles(smiles)
 if mol is None: raise ValueError('invalid SMILES')
 f=[]
 for a in mol.GetAtoms():
  f.append([a.GetAtomicNum()/100.,a.GetTotalDegree()/6.,a.GetFormalCharge()/4.,a.GetTotalNumHs()/4.,int(a.GetIsAromatic()),a.GetMass()/200.,float(a.IsInRing()),int(a.GetHybridization())/10.,int(a.GetChiralTag())/5.,a.GetTotalValence()/8.])
 n=len(f); adj=np.eye(n,dtype=np.float32)
 for b in mol.GetBonds():
  i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx();adj[i,j]=1;adj[j,i]=1
 degree=adj.sum(axis=1);adj=adj/np.sqrt(degree[:,None]*degree[None,:])
 return np.asarray(f,dtype=np.float32),adj

def batch(graphs,ids,extra):
 size=max(len(graphs[i][0]) for i in ids); atoms=np.zeros((len(ids),size,10),np.float32); adj=np.zeros((len(ids),size,size),np.float32);mask=np.zeros((len(ids),size),np.float32)
 for j,i in enumerate(ids):
  f,a=graphs[i]; n=len(f); atoms[j,:n]=f;adj[j,:n,:n]=a;mask[j,:n]=1
 return torch.from_numpy(atoms),torch.from_numpy(adj),torch.from_numpy(mask),None if extra is None else torch.from_numpy(extra[ids])

def fit(graphs,extra,y,ids,seed,epochs,layers=1):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed); model=GraphNet(0 if extra is None else extra.shape[1],layers);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);rng=np.random.default_rng(seed)
 for _ in range(epochs):
  model.train()
  for ids_batch in np.array_split(rng.permutation(ids),int(np.ceil(len(ids)/32))):
   opt.zero_grad(); out=model(*batch(graphs,ids_batch,extra)); loss=nn.functional.mse_loss(out,torch.log1p(torch.tensor(y[ids_batch],dtype=torch.float32))); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
 return model

def pred(model,graphs,extra,ids):
 model.eval(); p=[]
 with torch.no_grad():
  for b in np.array_split(ids,max(1,int(np.ceil(len(ids)/32)))):
   p.extend(torch.expm1(model(*batch(graphs,b,extra)).clamp(0,14)).numpy())
 return np.array(p,dtype=float)

def features(raw,tr):
 scaler=StandardScaler().fit(raw['desc'][tr]); desc=np.clip(scaler.transform(raw['desc']),-8,8);return np.concatenate([raw['fp2'],raw['maccs'],desc],axis=1).astype(np.float32),scaler

def transform(raw,scaler):
 desc=np.clip(scaler.transform(raw['desc']),-8,8);return np.concatenate([raw['fp2'],raw['maccs'],desc],axis=1).astype(np.float32)

def main():
 p=argparse.ArgumentParser(description='Train and validate a VACV graph model using train.csv only')
 p.add_argument('--train',default='train.csv'); p.add_argument('--test',default='test.csv'); p.add_argument('--sample',default='sample_submit.csv')
 p.add_argument('--out',default='versions/test'); p.add_argument('--folds',type=int,default=5); p.add_argument('--split-seeds',default='42,43')
 p.add_argument('--epochs',type=int,default=80); p.add_argument('--final-seed',type=int,default=4242); p.add_argument('--graph-weight',type=float,default=.30)
 p.add_argument('--base-submission',help='Deprecated; a complete saved base model is required')
 p.add_argument('--base-model',default='versions/reference/model.joblib',help='Complete saved CPU bundle for reproducible blending')
 p.add_argument('--layers',type=int,choices=[1,2],default=1,help='Number of graph aggregation layers')
 args=p.parse_args(); out=require_safe_output(args.out); out.mkdir(parents=True,exist_ok=True)
 if args.base_submission: p.error('Use --base-model; a saved test submission is not a general prediction model')
 if not 0<=args.graph_weight<=1: raise ValueError('--graph-weight must be between 0 and 1')
 train=read_competition_csv(args.train,'train')
 if (out/'submission.csv').exists():
  archive=ROOT/'experiments/legacy_archive'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f');archive.mkdir(parents=True,exist_ok=True)
  for old in out.iterdir():
   if old.is_file() and old.suffix in {'.csv','.json','.joblib','.md'}:shutil.copy2(old,archive/old.name)
 y=train['SI VACV'].to_numpy(float); n=len(y)
 raw=cpu.chem_features(train.SMILES.astype(str).tolist()); graphs=[graph(s) for s in train.SMILES.astype(str)]; groups=raw['canon']
 results=[]; predictions=[]; seeds=[int(s.strip()) for s in args.split_seeds.split(',') if s.strip()]
 if not seeds or args.folds<2: raise ValueError('Provide split seeds and at least two folds')
 for seed in seeds:
  folds=cpu.make_folds(y,groups,args.folds,seed); pp={name:np.zeros(n) for name in ['gcn_morgan_L1','gcn_morgan_L2']}
  for j,(tr,va) in enumerate(folds):
   if not set(groups[tr]).isdisjoint(groups[va]): raise AssertionError('canonical structure crosses a CV fold')
   x,_=features(raw,tr)
   for name in pp:
    layer_count=int(name[-1]); ex=x
    net=fit(graphs,ex,y,tr,seed*100+j,args.epochs,layers=layer_count); pp[name][va]=pred(net,graphs,ex,va)
    print(seed,j+1,name,round(mean_squared_error(y[va],pp[name][va])),flush=True)
  for name,predictions_oof in pp.items():
   results.append({'seed':seed,'model':name,'MSE':mean_squared_error(y,predictions_oof),'MAE':mean_absolute_error(y,predictions_oof)})
   predictions.append(pd.DataFrame({'seed':seed,'model':name,'row':np.arange(n),'true':y,'pred':predictions_oof}))
 metrics=pd.DataFrame(results); metrics.to_csv(out/'metrics.csv',index=False); pd.concat(predictions,ignore_index=True).to_csv(out/'oof.csv',index=False)
 print(metrics.to_string(index=False),flush=True)
 x,scaler=features(raw,np.arange(n)); net=fit(graphs,x,y,np.arange(n),args.final_seed,args.epochs,layers=args.layers)
 # Test structures are featurized only after CV and final model fitting.
 test=read_competition_csv(args.test,'test'); sample=pd.read_csv(args.sample)
 if list(sample.columns)!=['index','SI VACV'] or 'index' not in test or len(test)!=len(sample) or not np.array_equal(test['index'],sample['index']):raise ValueError('test and sample indices do not match')
 test_raw=cpu.chem_features(test.SMILES.astype(str).tolist()); x_test=transform(test_raw,scaler); test_graphs=[graph(s) for s in test.SMILES.astype(str)]
 result=pred(net,test_graphs,x_test,np.arange(len(test)))
 if not np.isfinite(result).all() or (result<0).any(): raise ValueError('Invalid test predictions')
 allowed_train=train.reindex(columns=['SMILES','SI VACV','Culture','strain']).to_csv(index=False).encode('utf-8')
 bundle={'state_dict':copy.deepcopy(net.state_dict()),'extra_dim':x.shape[1],'layers':args.layers,'scaler':scaler,'mode':'gcn_morgan', 'seed':args.final_seed,'epochs':args.epochs,'training_rows':n,'training_sha256':hashlib.sha256(allowed_train).hexdigest(),'feature_version':f'gcn_atom10_{args.layers}layer_cpu_fp2_maccs_desc16_v1','graph_weight':args.graph_weight}
 joblib.dump(bundle,out/'model.joblib',compress=3)
 loaded=joblib.load(out/'model.joblib'); restored=GraphNet(loaded['extra_dim'],loaded.get('layers',1)); restored.load_state_dict(loaded['state_dict']); np.testing.assert_allclose(result,pred(restored,test_graphs,x_test,np.arange(len(test))),rtol=1e-6,atol=1e-6)
 base=joblib.load(args.base_model);reference_pred=cpu.bundle_predict(base,test)
 submission=(1-args.graph_weight)*reference_pred+args.graph_weight*result
 if not np.isfinite(submission).all() or (submission<0).any(): raise ValueError('Invalid blended predictions')
 pd.DataFrame({'index':sample['index'],'SI VACV':submission}).to_csv(out/'submission.csv',index=False)
 from src.validation import write_json,sha
 composite={'model_type':'reference_graph_blend','base_bundle':base,'graph_bundle':bundle,'graph_weight':args.graph_weight,'train_rows':n,'test_rows':len(test)}
 joblib.dump(composite,out/'prediction_bundle.joblib',compress=3)
 write_json(out/'manifest.json',{'status':'experimental_pending_kaggle','public_mse':None,'graph_weight':args.graph_weight,'layers':args.layers,'training_sha256':sha(args.train),'test_sha256':sha(args.test),'submission_sha256':sha(out/'submission.csv'),'model_sha256':sha(out/'model.joblib'),'prediction_model_sha256':sha(out/'prediction_bundle.joblib'),'note':'OOF graph metrics are screening only. No test labels, feature selection, normalization fitting, or weight tuning on test.'})
 print(f'Saved train-only {1-args.graph_weight:.0%} reference + {args.graph_weight:.0%} graph submission; max prediction {float(submission.max()):.3f}',flush=True)
if __name__=='__main__':main()
