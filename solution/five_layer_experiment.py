"""Evaluate and package a 70% CPU / 30% five-layer sequential GCN ensemble."""
import copy
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import vacv_cpu_improved as cpu
from competition_data import read_competition_csv
from src.artifacts import validate_submission
from src.inference import predict_bundle
from src.validation import folds, sha, training_data
from versions.test.training import features, fit, graph, pred

OUT = Path('results/five_layer_gcn_20261108')

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df=training_data(); y=df['SI VACV'].to_numpy(float); n=len(df)
    raw=cpu.chem_features(df.SMILES.tolist()); graphs=[graph(s) for s in df.SMILES]
    previous=pd.read_csv('reports/matched_blend_oof_predictions.csv')
    previous_layers=pd.read_csv('reports/gcn_layer_oof_comparison.csv')
    rows=[]; scores=[]
    for seed in (42,43):
        old=previous[previous.seed==seed].sort_values('row').reset_index(drop=True)
        old_layers=previous_layers[previous_layers.seed==seed].sort_values('row').reset_index(drop=True)
        assert np.array_equal(old.row,np.arange(n)) and np.array_equal(old_layers.row,np.arange(n))
        np.testing.assert_allclose(old['true'],y); np.testing.assert_allclose(old_layers['true'],y)
        np.testing.assert_allclose(old.gcn_oof,old_layers.pred_1layer)
        predictions=np.zeros(n)
        for j,(tr,va) in enumerate(folds(y,raw['canon'],seed)):
            x,_=features(raw,tr)
            net=fit(graphs,x,y,tr,seed*100+j,80,layers=5)
            predictions[va]=pred(net,graphs,x,va)
            print(f'seed={seed} fold={j+1} complete',flush=True)
        frame=pd.DataFrame({'seed':seed,'row':np.arange(n),'true':y,'cpu':old.cpu_oof,'gcn1':old_layers.pred_1layer,'gcn2':old_layers.pred_2layer,'gcn5':predictions})
        rows.append(frame)
        for name,p in {'cpu70_gcn1_15_gcn2_15':.70*frame.cpu+.15*frame.gcn1+.15*frame.gcn2,
                       'cpu70_gcn5_30':.70*frame.cpu+.30*frame.gcn5}.items():
            scores.append({'candidate':name,'seed':seed,'mse':float(np.mean((y-p)**2)),'mae':float(np.mean(abs(y-p)))})
    pd.concat(rows,ignore_index=True).to_csv(OUT/'oof.csv',index=False)
    pd.DataFrame(scores).to_csv(OUT/'metrics.csv',index=False)
    print(pd.DataFrame(scores).to_string(index=False),flush=True)
    x,scaler=features(raw,np.arange(n)); net=fit(graphs,x,y,np.arange(n),4242,80,layers=5)
    graph_bundle={'state_dict':copy.deepcopy(net.state_dict()),'extra_dim':x.shape[1],'layers':5,'scaler':scaler,'seed':4242,'epochs':80,'training_rows':n}
    base=joblib.load('versions/top1/blend_model.joblib')['base_bundle']
    model={'model_type':'multi_graph_blend','base_bundle':base,'cpu_weight':.70,'graph_weights':[.30],'graph_bundles':[graph_bundle]}
    model_path=OUT/'prediction_bundle.joblib'; joblib.dump(model,model_path,compress=3)
    test=read_competition_csv('test.csv','test'); sample=pd.read_csv('sample_submit.csv')
    assert np.array_equal(test['index'],sample['index'])
    p=predict_bundle(model,test); file=OUT/'submission.csv'
    pd.DataFrame({'index':sample['index'],'SI VACV':p}).to_csv(file,index=False)
    manifest={'status':'experimental_pending_kaggle','public_mse':None,'weights':{'cpu':.70,'gcn5':.30},'gcn_layers':5,'training_sha256':sha('train.csv'),'test_sha256':sha('test.csv'),'submission_sha256':sha(file),'prediction_model_sha256':sha(model_path),'prediction_model_path':'prediction_bundle.joblib','validation':'paired canonical grouped OOF, seeds 42 and 43, trained only on train.csv'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(validate_submission(file),flush=True)

if __name__=='__main__': main()
