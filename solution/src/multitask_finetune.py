"""Joint fine-tuning on SI and auxiliary VACV concentration endpoints."""
import argparse,json
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from src.gpu_training import fingerprints,network,predict,BITS
from src.validation import training_data,sha
from competition_data import require_safe_output,read_competition_csv

def metric(y,p):
 y=np.asarray(y,float); p=np.asarray(p,float)
 return {'MAE':float(np.mean(abs(y-p))),'MSE':float(np.mean((y-p)**2)),'log_MAE':float(np.mean(abs(np.log1p(y)-np.log1p(p))))}

def main():
 p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto'); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
 device=('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device
 if device=='cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable')
 out=require_safe_output(a.out); out.mkdir(parents=True,exist_ok=False)
 source='data/external/processed/vacv_all_20261108/vacv_activities.csv'; df=pd.read_csv(source); df=df[df.endpoint.isin(['IC50','EC50']) & df.relation.isin(['=','>','>=','<','<='])].copy()
 df['value']=pd.to_numeric(df['value'],errors='coerce'); df=df[np.isfinite(df.value)&(df.value>0)].copy(); df=df[df.data_validity_comment.isna()].copy()
 factors={'nM':1.,'uM':1000.,'µM':1000.,'μM':1000.,'mM':1e6,'M':1e9,'pM':.001}
 def convert(r):
  if r['units'] in factors:return r['value']*factors[r['units']]
  if r['units'] in ('ug.mL-1','ug/ml','µg/mL'):
   mol=Chem.MolFromSmiles(r.SMILES); mw=Descriptors.MolWt(mol) if mol else 0
   return r['value']*1e6/mw if mw>0 else np.nan
  return np.nan
 df['value_nM']=df.apply(convert,axis=1); df=df[np.isfinite(df.value_nM)&(df.value_nM>0)].reset_index(drop=True)
 test=read_competition_csv('test.csv','test'); _,testg=fingerprints(test.SMILES); ax,ag=fingerprints(df.SMILES); keep=~np.isin(ag,testg); df=df.loc[keep].reset_index(drop=True); ax=ax[keep]; ag=ag[keep]
 si=pd.read_csv('results/all_open_si_20261108/training_data.csv'); sx,sg=fingerprints(si.SMILES); sy=si['SI VACV'].to_numpy(float); val=np.flatnonzero(si.evaluation_split=='validation'); train=np.flatnonzero(si.evaluation_split=='train')
 official=training_data(); ox,_=fingerprints(official.SMILES); teacher=pd.read_csv('results/distill_top1_20261108/train_predictions.csv')
 if len(teacher)!=len(official) or not np.array_equal(teacher.SMILES,official.SMILES):raise ValueError('Teacher/train mismatch')
 ck=torch.load('results/distill_top1_20261108/student.pt',map_location='cpu',weights_only=True)
 if ck['bits']!=BITS or ck['train_sha256']!=sha('train.csv'):raise ValueError('Unexpected checkpoint')
 df['evaluation_split']=np.where(np.isin(ag,sg[val]),'excluded_for_validation','train'); df.to_csv(out/'auxiliary_data.csv',index=False)
 aux_training=np.flatnonzero(df.evaluation_split=='train'); y=np.log10(df.value_nM.to_numpy(np.float32)); types=(df.endpoint=='EC50').to_numpy(np.int64); rel=df.relation.to_numpy(); relation=np.where(np.isin(rel,['>','>=']),1,np.where(np.isin(rel,['<','<=']),-1,0))
 report={'device':device,'seed':a.seed,'source_sha256':sha(source),'si_rows':len(si),'auxiliary_rows':len(df),'auxiliary_train_rows_for_evaluation':len(aux_training),'excluded_auxiliary_rows_matching_si_validation':int(len(df)-len(aux_training)),'auxiliary_endpoints':df.endpoint.value_counts().to_dict(),'auxiliary_relations':df.relation.value_counts().to_dict(),'conversion':'Molar units converted to nM; mass concentration converted using RDKit molecular weight. Approximate salt/form assumptions remain.','warning':'Exploratory provisional data; molecular activity depends on assay conditions. The holdout evaluates only SI; the full model uses all SI rows. External data violate the documented contest protocol.','evaluation':{}}
 def new_model():
  torch.manual_seed(a.seed); net=network(); net.load_state_dict(ck['state_dict']); net.to(device); head=torch.nn.Linear(128,2).to(device); return net,head
 def evaluate(net,key):
  outp=predict(net,sx[val],device); report['evaluation'][key]={'holdout_si':metric(sy[val],outp),'teacher_train':metric(teacher.teacher,predict(net,ox,device))}; pd.DataFrame({'SMILES':sg[val],'observed':sy[val],'predicted':outp}).to_csv(out/(key+'_validation.csv'),index=False)
 def run(net,head,siidx,auxidx,callback):
  torch.manual_seed(a.seed); opt=torch.optim.AdamW(list(net.parameters())+list(head.parameters()),lr=.0001,weight_decay=.01)
  xx=torch.as_tensor(sx[siidx],device=device); yy=torch.as_tensor(np.log1p(sy[siidx]).astype(np.float32),device=device).view(-1,1)
  aa=torch.as_tensor(ax[auxidx],device=device); ay=torch.as_tensor(y[auxidx],device=device); at=torch.as_tensor(types[auxidx],device=device); ar=torch.as_tensor(relation[auxidx],device=device)
  tx=torch.as_tensor(ox,device=device); ty=torch.as_tensor(np.log1p(teacher.teacher.to_numpy(np.float32)),device=device).view(-1,1)
  for stage in (1,2):
   for _ in range(100):
    net.train(); head.train(); opt.zero_grad(set_to_none=True)
    # Separate output heads share the molecular representation.
    z=net[2](net[1](net[0](aa))); pred=head(z).gather(1,at[:,None]).flatten(); diff=pred-ay
    residual=torch.where(ar>0,torch.minimum(diff,torch.zeros_like(diff)),torch.where(ar<0,torch.maximum(diff,torch.zeros_like(diff)),diff))
    loss=torch.nn.functional.mse_loss(net(xx),yy)+.5*torch.nn.functional.mse_loss(net(tx),ty)+.2*torch.mean(residual**2)
    loss.backward(); opt.step()
   callback(net,head,stage)
 net,head=new_model(); evaluate(net,'baseline'); run(net,head,train,aux_training,lambda n,h,s:evaluate(n,f'epoch_{100*s}'))
 net,head=new_model()
 def save(n,h,s):
  torch.save({'state_dict':{k:v.detach().cpu() for k,v in n.state_dict().items()},'aux_head':{k:v.detach().cpu() for k,v in h.state_dict().items()},'bits':BITS,'epoch':s*100,'train_sha256':sha('train.csv'),'auxiliary_sha256':sha(source)},out/f'full_epoch_{s*100}.pt'); print(f'Trained {s*100} epochs',flush=True)
 run(net,head,np.arange(len(si)),np.arange(len(df)),save)
 (out/'metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8'); print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
