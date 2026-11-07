"""Inference for GCN model trained by gcn_experiment.py."""
import argparse
from pathlib import Path
import joblib, numpy as np, pandas as pd
from versions.test.training import GraphNet,graph,pred
import vacv_cpu_improved as cpu
from competition_data import read_competition_csv, require_safe_output

def predict(bundle,smiles):
 graphs=[graph(str(s)) for s in smiles]; raw=cpu.chem_features([str(s) for s in smiles])
 desc=np.clip(bundle['scaler'].transform(raw['desc']),-8,8)
 x=np.concatenate([raw['fp2'],raw['maccs'],desc],axis=1).astype(np.float32)
 if x.shape[1]!=bundle['extra_dim']: raise ValueError('Feature dimension differs from saved model')
 net=GraphNet(bundle['extra_dim'],bundle.get('layers',1));net.load_state_dict(bundle['state_dict'])
 return pred(net,graphs,x,np.arange(len(graphs)))

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',default='versions/test/model.joblib');p.add_argument('--reference-model',default='versions/reference/model.joblib');p.add_argument('--graph-weight',type=float);p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--sample');a=p.parse_args()
 frame=read_competition_csv(a.input,'test' if a.sample else 'inference');bundle=joblib.load(a.model);graph_pred=predict(bundle,frame['SMILES'])
 reference=joblib.load(a.reference_model);reference_pred=cpu.bundle_predict(reference,frame)
 weight=bundle.get('graph_weight',.05) if a.graph_weight is None else a.graph_weight
 if not 0<=weight<=1: raise ValueError('--graph-weight must be between 0 and 1')
 y=(1-weight)*reference_pred+weight*graph_pred
 if not np.isfinite(y).all() or (y<0).any(): raise ValueError('Invalid model predictions')
 if a.sample:
  sample=pd.read_csv(a.sample)
  if list(sample.columns)!=['index','SI VACV'] or len(sample)!=len(frame) or 'index' not in frame or not np.array_equal(sample['index'],frame['index']): raise ValueError('Test and sample do not match')
  result=pd.DataFrame({'index':sample['index'],'SI VACV':y})
 else: result=frame.assign(predicted_SI=y)
 require_safe_output(a.output);result.to_csv(a.output,index=False);print(f'Saved {len(result)} rows to {a.output}')
if __name__=='__main__':main()
