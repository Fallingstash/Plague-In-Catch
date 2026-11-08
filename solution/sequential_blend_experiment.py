"""Blend the saved CPU/GCN1 model with GCN2 in two successive 70/30 steps."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from competition_data import read_competition_csv
from src.artifacts import validate_submission
from src.inference import predict_bundle
from src.validation import sha

OUT=Path('results/sequential_70_30_20261108')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    first=joblib.load('versions/top1/blend_model.joblib')
    second=joblib.load('versions/candidates/historical_two_layer_70_30/model_bundle.joblib')
    if first['graph_weight'] != .3 or second['graph_weight'] != .3:
        raise ValueError('Expected saved 70/30 reference models')
    model={'model_type':'multi_graph_blend','base_bundle':first['base_bundle'],
           'cpu_weight':.49,'graph_weights':[.21,.30],
           'graph_bundles':[first['graph_bundle'],second['graph_bundle']]}
    model_path=OUT/'prediction_bundle.joblib'; joblib.dump(model,model_path,compress=3)
    test=read_competition_csv('test.csv','test'); sample=pd.read_csv('sample_submit.csv')
    if not np.array_equal(test['index'],sample['index']): raise ValueError('Test indices mismatch')
    pred=predict_bundle(model,test); file=OUT/'submission.csv'
    pd.DataFrame({'index':sample['index'],'SI VACV':pred}).to_csv(file,index=False)
    manifest={'status':'experimental_pending_kaggle','public_mse':None,'weights':{'cpu':.49,'gcn1':.21,'gcn2':.30},
              'recipe':'0.70 * (0.70 * CPU + 0.30 * GCN1) + 0.30 * GCN2',
              'training_sha256':sha('train.csv'),'test_sha256':sha('test.csv'),
              'submission_sha256':sha(file),'prediction_model_sha256':sha(model_path),'prediction_model_path':'prediction_bundle.joblib'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    oof=pd.read_csv('reports/matched_blend_oof_predictions.csv')
    old=pd.read_csv('reports/gcn_layer_oof_comparison.csv')
    merged=oof.merge(old[['seed','row','true','pred_1layer','pred_2layer']],on=['seed','row','true'],validate='one_to_one')
    if len(merged)!=len(oof): raise ValueError('OOF predictions do not align')
    np.testing.assert_allclose(merged.gcn_oof,merged.pred_1layer)
    merged['sequential']=.49*merged.cpu_oof+.21*merged.pred_1layer+.30*merged.pred_2layer
    merged['baseline_15_15']=.70*merged.cpu_oof+.15*merged.pred_1layer+.15*merged.pred_2layer
    scores=[]
    for seed,part in merged.groupby('seed'):
        for name in ('baseline_15_15','sequential'):
            scores.append({'seed':seed,'candidate':name,'mse':float(np.mean((part['true']-part[name])**2)),'mae':float(np.mean(abs(part['true']-part[name])))})
    pd.DataFrame(scores).to_csv(OUT/'metrics.csv',index=False)
    print(pd.DataFrame(scores).to_string(index=False),flush=True)
    print(validate_submission(file),flush=True)

if __name__=='__main__': main()
