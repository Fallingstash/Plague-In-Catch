"""CPU inference for the previously trained VACV neural ensemble."""
import argparse, json, subprocess, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
from vacv_neural import load_model_predictions, similar
from competition_data import read_competition_csv

def main():
 p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
 train=sub.add_parser('train', help='Train on labeled molecules and predict the test set')
 train.add_argument('--train',default='train.csv'); train.add_argument('--test',default='test.csv')
 train.add_argument('--sample',default='sample_submit.csv'); train.add_argument('--out',default='results_new')
 train.add_argument('--epochs',type=int,default=180); train.add_argument('--seeds',type=int,default=2)
 train.add_argument('--folds',type=int,default=5); train.add_argument('--seed',type=int,default=42)
 train.add_argument('--repeats',type=int,default=3, help='Independent grouped CV splits; all folds count')
 train.add_argument('--device',choices=['cpu'],default='cpu')
 for name in ('predict','similar','batch','submit'):
  q=sub.add_parser(name); q.add_argument('--model',default='versions/reference/blend_model.joblib')
  if name in ('predict','similar'): q.add_argument('--smiles',required=True)
  if name=='similar': q.add_argument('--top',type=int,default=10)
  if name=='batch': q.add_argument('--input',required=True); q.add_argument('--output',required=True)
  if name=='submit': q.add_argument('--test',default='test.csv'); q.add_argument('--sample',default='sample_submit.csv'); q.add_argument('--output',default='results/submission.csv')
 a=p.parse_args()
 if a.cmd=='train':
  output=Path(a.out)
  if a.repeats < 1: p.error('--repeats must be positive')
  if a.repeats > 1:
   if output.exists() and any(output.iterdir()): p.error('Output directory is not empty; choose a new --out')
   output.mkdir(parents=True,exist_ok=True)
   summaries=[]
   for repeat in range(a.repeats):
    run_dir=output/f'repeat_{repeat+1}'
    cmd=[sys.executable,str(Path(__file__)), 'train','--train',a.train,'--test',a.test,'--sample',a.sample,
         '--out',str(run_dir),'--epochs',str(a.epochs),'--seeds',str(a.seeds),'--folds',str(a.folds),
         '--seed',str(a.seed+repeat),'--repeats','1']
    subprocess.run(cmd,check=True)
    metrics=json.loads((run_dir/'metrics.json').read_text(encoding='utf-8'))
    selected=metrics['selected']
    for fold in metrics['folds']:
     summaries.append({'repeat':repeat+1,'seed':a.seed+repeat,'fold':fold['fold'],
                       'n':fold['original_validation'],'selected':selected,'MSE':fold[selected+'_MSE']})
   table=pd.DataFrame(summaries); table.to_csv(output/'repeated_folds.csv',index=False)
   score=np.average(table['MSE'],weights=table['n'])
   diagnostic=[]
   for repeat,folds in table.groupby('repeat'):
    worst=folds['MSE'].idxmax(); remaining=folds.drop(index=worst)
    diagnostic.append({'repeat':repeat,'excluded_fold':int(folds.loc[worst,'fold']),
                       'excluded_MSE':float(folds.loc[worst,'MSE']),
                       'MSE_without_worst_fold':float(np.average(remaining['MSE'],weights=remaining['n']))})
   pd.DataFrame(diagnostic).to_csv(output/'diagnostic_without_worst_fold.csv',index=False)
   print(f'All {len(table)} folds included: pooled MSE = {score:,.2f}. Per-repeat models are in {output}.')
   print(f'Diagnostic median fold MSE = {table["MSE"].median():,.2f}; this does not replace pooled MSE.')
   print('Diagnostic scores excluding the worst fold are saved separately; they are biased and do not select the model.')
   print('Model selection uses the same folds; this score is optimistic and is not nested CV.')
   return
  if (output/'models'/'bundle.joblib').exists():
   p.error('Output already contains a model; choose a new --out directory to preserve it')
  cmd=[sys.executable, str(Path(__file__).with_name('vacv_neural.py')), 'train',
       '--train',a.train,'--test',a.test,'--sample',a.sample,'--out',a.out,
       '--epochs',str(a.epochs),'--seeds',str(a.seeds),'--folds',str(a.folds),'--seed',str(a.seed)]
  subprocess.run(cmd,check=True)
  from vacv_neural import chemical_features, cv_split
  data=read_competition_csv(a.train,'train'); oof=pd.read_csv(output/'oof_predictions.csv')
  groups=chemical_features(data['SMILES'].tolist())['canonical']; y=data['SI VACV'].to_numpy(float)
  metrics=json.loads((output/'metrics.json').read_text(encoding='utf-8'))
  pred=oof['pred_'+metrics['selected']].to_numpy(); records=[]
  for fold,(_,valid) in enumerate(cv_split(y,groups,a.folds,a.seed),1):
   for i in valid:
    records.append({'fold':fold,'row':int(i),'SMILES':data.iloc[i]['SMILES'],'observed_SI':y[i],
                    'predicted_SI':pred[i],'squared_error':(y[i]-pred[i])**2})
  errors=pd.DataFrame(records).sort_values('squared_error',ascending=False)
  errors.to_csv(output/'fold_errors.csv',index=False)
  print('All folds included in OOF MSE. Largest errors:')
  print(errors[['fold','row','observed_SI','predicted_SI','squared_error']].head(5).to_string(index=False))
  return
 bundle=joblib.load(a.model)
 if bundle.get('model_type')=='reference_graph_blend':
  import vacv_cpu_improved as cpu
  from gcn_predict import predict as graph_predict
  is_cpu=False
  base=bundle['base_bundle']; graph=bundle['graph_bundle']; graph_weight=bundle['graph_weight']
  def model_predict(frame):
   return (1-graph_weight)*cpu.bundle_predict(base,frame)+graph_weight*graph_predict(graph,frame['SMILES'])
  def model_similar(smiles,top):
   query=cpu.chem_features([smiles]); sim=cpu.combined_similarity(base['ref_raw'],query,None,None,0)[0]
   ids=np.argsort(-sim)[:max(1,min(top,len(sim)))]
   return [{'similarity_Tanimoto':float(sim[i]),'SMILES':base['ref_smiles'][i], 'SI_observed':float(base['ref_si'][i]),'Title':base['ref_titles'][i]} for i in ids]
 elif 'nets' in bundle:
  is_cpu=True
 else:
  is_cpu=False
 if is_cpu:
  import vacv_cpu_improved as cpu
  def model_predict(frame): return cpu.bundle_predict(bundle,frame)
  def model_similar(smiles,top):
   query=cpu.chem_features([smiles]); sim=cpu.combined_similarity(bundle['ref_raw'],query,None,None,0)[0]
   ids=np.argsort(-sim)[:max(1,min(top,len(sim)))]
   return [{'similarity_Tanimoto':float(sim[i]),'SMILES':bundle['ref_smiles'][i], 'SI_observed':float(bundle['ref_si'][i]),'Title':bundle['ref_titles'][i]} for i in ids]
 else:
  if bundle.get('model_type')!='reference_graph_blend':
   def model_predict(frame): return load_model_predictions(bundle,frame)
   def model_similar(smiles,top): return similar(bundle,smiles,top)
 if a.cmd=='similar': print(json.dumps(model_similar(a.smiles,a.top),ensure_ascii=False,indent=2)); return
 if a.cmd=='predict':
  frame=pd.DataFrame({'SMILES':[a.smiles]}); neighbours=model_similar(a.smiles,1); pred=model_predict(frame)[0]
  print(json.dumps({'SMILES':a.smiles,'predicted_SI_VACV':float(pred),'nearest':neighbours[0], 'warning':'Low similarity to training data' if neighbours[0]['similarity_Tanimoto']<0.3 else None},ensure_ascii=False,indent=2)); return
 frame=read_competition_csv(a.input,'inference') if a.cmd=='batch' else read_competition_csv(a.test,'test')
 if a.cmd=='submit':
  sample=pd.read_csv(a.sample)
  if list(sample.columns)!=['index','SI VACV'] or len(sample)!=len(frame) or 'index' not in frame or not np.array_equal(sample['index'],frame['index']): raise ValueError('Sample and test indices or schema do not match')
 pred=model_predict(frame)
 if not np.isfinite(pred).all() or (pred<0).any(): raise ValueError('Invalid predictions')
 if a.cmd=='batch': frame['predicted_SI_VACV']=pred; result=frame
 else: result=pd.DataFrame({'index':sample['index'],'SI VACV':pred})
 Path(a.output).parent.mkdir(parents=True,exist_ok=True); result.to_csv(a.output,index=False); print(f'Saved {len(result)} rows to {a.output}')
if __name__=='__main__': main()

