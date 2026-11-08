"""Submission validation, provenance and immutable candidate directories."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from competition_data import read_competition_csv
from src.validation import ROOT,sha,write_json,protect_output,assert_protected_hashes
from src.inference import predict_path

def validate_submission(file,manifest=None,model=None,reproduce=True):
    file=Path(file);manifest=Path(manifest) if manifest else file.parent/'manifest.json'
    x=pd.read_csv(file)
    sample=pd.read_csv(ROOT/'sample_submit.csv');indices=pd.read_csv(ROOT/'test.csv',usecols=['index'])
    if list(x.columns)!=['index','SI VACV'] or len(x)!=597 or len(sample)!=len(x):
        raise ValueError('Expected exactly index,SI VACV and 597 rows')
    if x['index'].duplicated().any() or not np.array_equal(x['index'],sample['index']) or not np.array_equal(x['index'],indices['index']):
        raise ValueError('Submission indices differ from official test/sample, duplicated or missing')
    if not np.isfinite(x['SI VACV'].to_numpy(float)).all() or (x['SI VACV']<0).any():
        raise ValueError('Non-finite/negative SI')
    if not manifest.exists():raise ValueError('Missing manifest; submission is not ready')
    m=json.loads(manifest.read_text(encoding='utf-8'))
    if m.get('submission_sha256')!=sha(file):raise ValueError('Submission checksum mismatch')
    if model is None:
        recorded=m.get('prediction_model_path')
        if recorded:
            candidate=Path(recorded)
            if not candidate.is_absolute():candidate=file.parent/candidate
            if candidate.exists():model=candidate
    if model is None:
        for name in ('prediction_bundle.joblib','blend_model.joblib','model_bundle.joblib'):
            if (file.parent/name).exists():model=file.parent/name;break
    result={'rows':len(x),'submission_sha256':sha(file),'schema_valid':True,'model_reproduced':False}
    if reproduce:
        if model is None:raise ValueError('No complete prediction checkpoint; provide --model')
        model=Path(model)
        expected=m.get('prediction_model_sha256',m.get('model_sha256'))
        if not expected:raise ValueError('Manifest lacks the prediction model checksum')
        if sha(model)!=expected:raise ValueError('Prediction model checksum mismatch')
        frame=read_competition_csv(ROOT/'test.csv','test')
        p=predict_path(model,frame)
        np.testing.assert_allclose(x['SI VACV'],p,rtol=1e-9,atol=1e-8)
        result.update(model_reproduced=True,max_abs_delta=float(np.max(np.abs(x['SI VACV'].to_numpy()-p))))
    assert_protected_hashes()
    return result

def make_submission(model,output):
    output=protect_output(output);model=Path(model)
    if output.exists():raise ValueError('Output submission already exists; choose a new --output to preserve history')
    frame=read_competition_csv(ROOT/'test.csv','test');sample=pd.read_csv(ROOT/'sample_submit.csv')
    p=predict_path(model,frame);output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'index':sample['index'],'SI VACV':p}).to_csv(output,index=False)
    manifest=output.parent/'manifest.json'
    m=json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else {}
    m.update({'status':'pending_kaggle','public_mse':None,'training_sha256':sha(ROOT/'train.csv'),'test_sha256':sha(ROOT/'test.csv'),'submission_sha256':sha(output),'prediction_model_sha256':sha(model),'prediction_model_path':str(model.resolve())})
    write_json(manifest,m)
    return validate_submission(output,model=model)
