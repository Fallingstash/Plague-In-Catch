"""Immutable candidate packaging and recovery of previously reported artifacts."""
import re
import shutil
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.validation import ROOT,training_data,sha,write_json,protect_output,assert_protected_hashes
from src.artifacts import validate_submission
from src.inference import predict_bundle
from competition_data import read_competition_csv

def package_candidate(identifier,bundle,config,metric_info,public_score=None,original_submission=None):
    if not re.fullmatch('[A-Za-z0-9_-]+',identifier):raise ValueError('Invalid experiment ID')
    out=protect_output(ROOT/'versions/candidates'/identifier)
    if (out/'manifest.json').exists():
        return validate_submission(out/'submission.csv',model=out/'model_bundle.joblib')
    out.mkdir(parents=True,exist_ok=True)
    checkpoint=out/'model_bundle.joblib';joblib.dump(bundle,checkpoint,compress=3)
    # This stage is reached only after a fixed model/configuration has been chosen.
    sample=pd.read_csv(ROOT/'sample_submit.csv')
    if original_submission is not None:shutil.copy2(original_submission,out/'submission.csv')
    else:
        frame=read_competition_csv(ROOT/'test.csv','test');p=predict_bundle(bundle,frame)
        pd.DataFrame({'index':sample['index'],'SI VACV':p}).to_csv(out/'submission.csv',index=False)
    from src.research import source_digest
    manifest={'id':identifier,'status':'historical_reported_score_restored' if public_score is not None else 'pending_kaggle_unproven',
              'public_mse':public_score,'training_sha256':sha(ROOT/'train.csv'),'test_sha256':sha(ROOT/'test.csv'),
              'submission_sha256':sha(out/'submission.csv'),'prediction_model_sha256':sha(checkpoint),'code_sha256':source_digest(),
              'score_source':'previous user-reported Kaggle score; restored file not re-submitted' if public_score is not None else None}
    write_json(out/'manifest.json',manifest);write_json(out/'config.json',config);write_json(out/'metrics.json',metric_info)
    write_json(out/'model_refs.json',{'prediction_checkpoint':'model_bundle.joblib','sha256':sha(checkpoint),'training_sha256':sha(ROOT/'train.csv')})
    (out/'README.md').write_text(f'# {identifier}\n\nTrain-only model; test used for final inference. See config.json and metrics.json.\n\nReported Kaggle score: {public_score}. A null score is not a confirmed improvement.\n',encoding='utf-8')
    result=validate_submission(out/'submission.csv',model=checkpoint)
    write_json(out/'validation.json',result);assert_protected_hashes();return result

def preserve_legacy_candidate():
    import json
    directory=ROOT/'versions/test';manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'));w=manifest['graph_weight']
    bundle={'model_type':'reference_graph_blend','base_bundle':joblib.load(ROOT/'versions/reference/model.joblib'),
            'graph_bundle':joblib.load(ROOT/'versions/reference/gcn_model.joblib'),'graph_weight':w,'train_rows':320,'test_rows':597}
    joblib.dump(bundle,directory/'prediction_bundle.joblib',compress=3)
    manifest.update(prediction_model_sha256=sha(directory/'prediction_bundle.joblib'),training_sha256=sha(ROOT/'train.csv'),
                    prediction_model_path='prediction_bundle.joblib',selection_note='Legacy pending candidate; no new quality claim or test-based selection under the new protocol.')
    write_json(directory/'manifest.json',manifest)
    result=package_candidate('legacy_35_pending',bundle,{'graph_weight':w,'provenance':'pre-existing pending candidate'},
                             {'selection':'historical; not selected by new nested protocol'},original_submission=directory/'submission.csv')
    print('Legacy pending candidate preserved and reproducible:',result,flush=True)

def restore_two_layer():
    import torch
    import vacv_cpu_improved as cpu
    from versions.test.training import fit,features,graph
    torch.set_num_threads(4);df=training_data();y=df['SI VACV'].to_numpy(float)
    raw=cpu.chem_features(df.SMILES.tolist());x,scaler=features(raw,np.arange(len(df)))
    checkpoint=ROOT/'experiments/research_20261107/restored_two_layer.joblib'
    if checkpoint.exists():graph_bundle=joblib.load(checkpoint)
    else:
        net=fit([graph(s) for s in df.SMILES],x,y,np.arange(len(y)),4242,80,layers=2)
        graph_bundle={'state_dict':net.state_dict(),'extra_dim':x.shape[1],'layers':2,'scaler':scaler,'seed':4242,'epochs':80,'graph_weight':.3,'training_rows':320}
        joblib.dump(graph_bundle,checkpoint,compress=3)
    bundle={'model_type':'reference_graph_blend','base_bundle':joblib.load(ROOT/'versions/reference/model.joblib'),'graph_bundle':graph_bundle,'graph_weight':.3}
    recorded=pd.read_csv(ROOT/'reports/gcn_layer_test_prediction_comparison.csv').sort_values('index')
    sample=pd.read_csv(ROOT/'sample_submit.csv');assert np.array_equal(recorded['index'],sample['index'])
    recovered=ROOT/'experiments/research_20261107/restored_two_layer_submission.csv'
    pd.DataFrame({'index':sample['index'],'SI VACV':recorded.pred_2layer_70_30.to_numpy()}).to_csv(recovered,index=False)
    result=package_candidate('historical_two_layer_70_30',bundle,{'graph_weight':.3,'layers':2,'seed':4242,'epochs':80,'recovery_source':'reports/gcn_layer_test_prediction_comparison.csv'},
                             {'historical_reported_public_mse':898313.50898,'purpose':'Restore previously overwritten second-best reported model; no new improvement claim'},898313.50898,recovered)
    write_json(ROOT/'versions/ranking.json',{'top1':{'path':'versions/top1','public_mse':886520.98398},
              'second_best_known':{'path':'versions/candidates/historical_two_layer_70_30','public_mse':898313.50898},
              'preserved_top2_folder':{'path':'versions/top2','public_mse':936601.51958,'note':'Historical older top2, now third among reported results. Kept immutable.'}})
    print('Historical two-layer model restored:',result,flush=True)

if __name__=='__main__':
    preserve_legacy_candidate();restore_two_layer()
