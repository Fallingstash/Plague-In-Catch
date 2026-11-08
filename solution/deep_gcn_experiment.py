"""Compare fixed 1–4 layer mixtures using paired train-only grouped OOF predictions."""
import copy
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import vacv_cpu_improved as cpu
from competition_data import read_competition_csv
from src.validation import training_data, folds, sha
from src.inference import predict_bundle
from src.artifacts import validate_submission
from versions.test.training import graph, features, fit, pred

OUT = Path('results/deep_gcn_20261108')
WEIGHTS = {
    'baseline_15_15': (0.70, 0.15, 0.15, 0., 0.),
    'gcn3_10': (0.60, 0.15, 0.15, 0.10, 0.),
    'gcn4_10': (0.60, 0.15, 0.15, 0., 0.10),
    'gcn3_5_gcn4_5': (0.60, 0.15, 0.15, 0.05, 0.05),
}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df=training_data(); y=df['SI VACV'].to_numpy(float); n=len(df)
    raw=cpu.chem_features(df.SMILES.tolist()); graphs=[graph(s) for s in df.SMILES]
    old=pd.read_csv('reports/matched_blend_oof_predictions.csv')
    layers=pd.read_csv('reports/gcn_layer_oof_comparison.csv')
    all_rows=[]
    for seed in (42,43):
        a=old[old.seed==seed].sort_values('row').reset_index(drop=True)
        b=layers[layers.seed==seed].sort_values('row').reset_index(drop=True)
        assert np.array_equal(a.row,np.arange(n)) and np.array_equal(b.row,np.arange(n))
        np.testing.assert_allclose(a['true'],y); np.testing.assert_allclose(b['true'],y)
        np.testing.assert_allclose(a.gcn_oof,b.pred_1layer)
        deep={3:np.zeros(n),4:np.zeros(n)}
        for j,(tr,va) in enumerate(folds(y,raw['canon'],seed)):
            x,_=features(raw,tr)
            for depth in (3,4):
                net=fit(graphs,x,y,tr,seed*100+j,80,layers=depth)
                deep[depth][va]=pred(net,graphs,x,va)
                print(f'seed={seed} fold={j+1} layers={depth} complete',flush=True)
        part=pd.DataFrame({'seed':seed,'row':np.arange(n),'true':y,'cpu':a.cpu_oof,'gcn1':b.pred_1layer,'gcn2':b.pred_2layer,'gcn3':deep[3],'gcn4':deep[4]})
        all_rows.append(part)
    oof=pd.concat(all_rows,ignore_index=True); oof.to_csv(OUT/'oof.csv',index=False)
    metrics=[]
    for name,weights in WEIGHTS.items():
        for seed,part in oof.groupby('seed'):
            p=part[['cpu','gcn1','gcn2','gcn3','gcn4']].to_numpy() @ np.array(weights)
            metrics.append({'candidate':name,'seed':seed,'mse':float(np.mean((part['true']-p)**2)), 'mae':float(np.mean(abs(part['true']-p)))})
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)
    # Fixed mixtures are declared before looking at test predictions.
    x,scaler=features(raw,np.arange(n)); deep_bundles=[]
    for depth in (3,4):
        net=fit(graphs,x,y,np.arange(n),4242,80,layers=depth)
        deep_bundles.append({'state_dict':copy.deepcopy(net.state_dict()),'extra_dim':x.shape[1],'layers':depth,'scaler':scaler,'seed':4242,'epochs':80,'training_rows':n})
    champion=joblib.load('versions/top1/blend_model.joblib')
    two=joblib.load('versions/candidates/historical_two_layer_70_30/model_bundle.joblib')
    graph_bundles=[champion['graph_bundle'],two['graph_bundle'],*deep_bundles]
    test=read_competition_csv('test.csv','test'); sample=pd.read_csv('sample_submit.csv')
    assert np.array_equal(test['index'],sample['index'])
    for name,weights in WEIGHTS.items():
        out=OUT/name; out.mkdir(exist_ok=True)
        model={'model_type':'multi_graph_blend','base_bundle':champion['base_bundle'],'cpu_weight':weights[0],'graph_weights':weights[1:],'graph_bundles':graph_bundles}
        model_path=out/'prediction_bundle.joblib'; joblib.dump(model,model_path,compress=3)
        p=predict_bundle(model,test)
        file=out/'submission.csv'; pd.DataFrame({'index':sample['index'],'SI VACV':p}).to_csv(file,index=False)
        manifest={'status':'experimental_pending_kaggle','public_mse':None,'weights':dict(zip(['cpu','gcn1','gcn2','gcn3','gcn4'],weights)),'training_sha256':sha('train.csv'),'test_sha256':sha('test.csv'),'submission_sha256':sha(file),'prediction_model_sha256':sha(model_path),'prediction_model_path':'prediction_bundle.joblib','validation':'paired canonical grouped OOF, seeds 42 and 43; train only; not an independent model-selection estimate'}
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        print(name,validate_submission(file),flush=True)

if __name__=='__main__': main()
