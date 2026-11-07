"""Behavioral checks for leakage, persistence, grouped CV and artifact integrity."""
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from src.validation import ROOT,training_data,folds,sha,assert_protected_hashes,protect_output
from src.features import features,canonical,tanimoto,subset
from src.models import MoleculeRegressor,TanimotoKernel
from src.artifacts import validate_submission
from competition_data import read_competition_csv,require_safe_output

@pytest.fixture(scope='module')
def train():return training_data()

@pytest.fixture(scope='module')
def raw(train):return features(train.SMILES.iloc[:30])

def test_official_training_source_and_allowed_fields(train):
    assert train.shape==(320,2)
    assert list(train.columns)==['SMILES','SI VACV']
    assert (train['SI VACV']==0).sum()==83

@pytest.mark.parametrize('source',['test.csv','research_data_collection/exports/research_candidates.csv'])
def test_test_and_external_labels_cannot_be_used_for_training(source):
    with pytest.raises(ValueError,match='official|Training'):
        training_data(ROOT/source)

def test_modified_training_targets_rejected(tmp_path,train):
    data=train.copy();data.loc[0,'SI VACV']=123456
    p=tmp_path/'train.csv';data.to_csv(p,index=False)
    with pytest.raises(ValueError):read_competition_csv(p,'train')

def test_forbidden_measurements_do_not_change_feature_arrays(tmp_path):
    p=tmp_path/'molecules.csv'
    pd.DataFrame({'SMILES':['CCO','CCN'],'IC50_uM_VACV':[1,2],'CC50_uM':[10,20],' CC50_derived':[3,4]}).to_csv(p,index=False)
    a=read_competition_csv(p,'inference');fa=features(a.SMILES)
    pd.DataFrame({'SMILES':['CCO','CCN'],'IC50_uM_VACV':[9999,-123],'CC50_uM':['bad','hidden'],' CC50_derived':['foo','bar']}).to_csv(p,index=False)
    b=read_competition_csv(p,'inference');fb=features(b.SMILES)
    assert list(a.columns)==list(b.columns)==['SMILES']
    for key in fa:np.testing.assert_array_equal(fa[key],fb[key])

def test_malformed_hidden_target_rejected(tmp_path):
    p=tmp_path/'test.csv';pd.DataFrame({'SMILES':['CCO'],'SI VACV':[9]}).to_csv(p,index=False)
    with pytest.raises(ValueError,match='target'):read_competition_csv(p,'test')

def test_canonicalization_and_stereochemistry():
    assert canonical('OCC')==canonical('CCO')
    assert canonical('N[C@H](C)O')!=canonical('N[C@@H](C)O')
    f=features(['N[C@H](C)O','N[C@@H](C)O'])
    assert not np.array_equal(f['morgan2_2048'][0],f['morgan2_2048'][1])
    with pytest.raises(ValueError):canonical('not a SMILES')
    with pytest.raises(ValueError):canonical('')

def test_cache_keeps_row_order_and_molecular_identity(tmp_path):
    a=features(['CCO','CCN','OCC'],tmp_path);b=features(['CCN','CCO'],tmp_path)
    np.testing.assert_array_equal(a['morgan2_1024'][0],a['morgan2_1024'][2])
    np.testing.assert_array_equal(a['morgan2_1024'][1],b['morgan2_1024'][0])
    assert len(list(tmp_path.glob('*.joblib')))==2

@pytest.mark.parametrize('seed',[42,43])
@pytest.mark.parametrize('grouping',['canonical','scaffold'])
def test_group_disjoint_folds_and_exact_oof_coverage(train,seed,grouping):
    raw=features(train.SMILES);y=train['SI VACV'].to_numpy();g=raw[grouping]
    a=folds(y,g,seed);b=folds(y,g,seed);coverage=np.zeros(len(y),int)
    for (tr,va),(tr2,va2) in zip(a,b):
        assert set(g[tr]).isdisjoint(g[va]);np.testing.assert_array_equal(va,va2);coverage[va]+=1
    assert np.all(coverage==1)

def test_fingerprints_and_positive_semidefinite_kernel(raw):
    assert raw['morgan2_512'].shape==(30,512)
    assert raw['morgan3_2048'].shape==(30,2048)
    k=tanimoto(raw['morgan2_1024'],raw['morgan2_1024'])
    np.testing.assert_allclose(k,k.T);np.testing.assert_allclose(np.diag(k),1)
    assert np.linalg.eigvalsh(k).min()>-1e-8

@pytest.mark.parametrize('family',['krr','gp','catboost'])
def test_model_round_trip_does_not_refit(tmp_path,train,raw,family):
    model=MoleculeRegressor({'family':family,'target':'log','alpha':.3,'iterations':12,'depth':2}).fit(raw,train['SI VACV'].iloc[:30].to_numpy(),42)
    expected=model.predict(raw);p=tmp_path/'model.joblib';joblib.dump(model,p);loaded=joblib.load(p)
    np.testing.assert_allclose(loaded.predict(raw),expected,rtol=1e-12,atol=1e-12)
    assert model.train_rows==30
    if family=='gp':assert np.isfinite(loaded.uncertainty(raw)).all()

def test_validation_extreme_features_do_not_refit_imputer(train,raw):
    model=MoleculeRegressor({'family':'catboost','target':'log','iterations':10,'depth':2}).fit(raw,train['SI VACV'].iloc[:30].to_numpy(),42)
    before=model.imputer.statistics_.copy();query={k:v.copy() for k,v in raw.items()};query['desc'][:]=1e12
    model.predict(query);np.testing.assert_array_equal(before,model.imputer.statistics_)

def test_training_prepare_does_not_read_test_or_external(monkeypatch):
    from src.research import prepare
    real=pd.read_csv
    def guarded(path,*a,**kw):
        if 'test.csv' in str(path) or 'external' in str(path):raise AssertionError('Selection touched forbidden data')
        return real(path,*a,**kw)
    monkeypatch.setattr(pd,'read_csv',guarded)
    df,_,y=prepare();assert len(df)==len(y)==320

@pytest.mark.parametrize('name',['top1','top2'])
def test_confirmed_artifacts_are_write_protected(name):
    for function in (protect_output,require_safe_output):
        with pytest.raises(ValueError):function(ROOT/'versions'/name/'nested'/'submission.csv')
    assert_protected_hashes()

def copy_submission(tmp_path):
    s=pd.read_csv(ROOT/'versions/top1/submission.csv');p=tmp_path/'submission.csv';s.to_csv(p,index=False)
    (tmp_path/'manifest.json').write_text(json.dumps({'submission_sha256':sha(p)}))
    return s,p

def test_valid_submission_schema_and_manifest(tmp_path):
    _,p=copy_submission(tmp_path)
    assert validate_submission(p,reproduce=False)['schema_valid']

@pytest.mark.parametrize('problem',['nan','negative','duplicate','order','missing','extra_column','hash'])
def test_invalid_submissions_are_rejected(tmp_path,problem):
    s,p=copy_submission(tmp_path)
    if problem=='nan':s.loc[0,'SI VACV']=np.nan
    elif problem=='negative':s.loc[0,'SI VACV']=-1
    elif problem=='duplicate':s.loc[1,'index']=s.loc[0,'index']
    elif problem=='order':s=s.iloc[::-1]
    elif problem=='missing':s=s.iloc[:-1]
    elif problem=='extra_column':s['extra']=1
    elif problem=='hash':s.loc[0,'SI VACV']+=1
    s.to_csv(p,index=False)
    with pytest.raises(ValueError):validate_submission(p,reproduce=False)

def test_top1_complete_prediction_checkpoint_reproduces_submission():
    r=validate_submission(ROOT/'versions/top1/submission.csv')
    assert r['model_reproduced'] and r['max_abs_delta']<1e-7

def test_directed_edges_are_reverse_pairs_and_cpu_padding_independent():
    from src.neural_models import directed_graph,directed_batch,DirectedMPNN
    a=directed_graph('CCO');b=directed_graph('CCCCCC')
    np.testing.assert_array_equal(a[1][a[3]],a[2])
    torch.manual_seed(42);model=DirectedMPNN(3);model.eval();extra=np.ones((2,3),np.float32)
    with torch.no_grad():
        x=model(*directed_batch([a,b],[0],extra));z=model(*directed_batch([a,b],[0,1],extra))
    torch.testing.assert_close(x[0],z[0],rtol=1e-6,atol=1e-6)
    assert all(p.device.type=='cpu' for p in model.parameters())

@pytest.mark.parametrize('tree_estimators',[250,350])
def test_nested_inner_rows_never_include_outer_validation(train,tree_estimators):
    from src.research import WORK
    y=train['SI VACV'].to_numpy();raw=features(train.SMILES)
    for seed in (42,43):
        for j,(tr,va) in enumerate(folds(y,raw['canonical'],seed),1):
            suffix='_tree350' if tree_estimators==350 else ''
            path=WORK/'nested'/f'canonical_{seed}{suffix}'/f'inner_oof_{j}.csv'
            if not path.exists():pytest.skip('Nested experiment not run yet')
            df=pd.read_csv(path)
            assert set(df.row)==set(tr) and set(df.row).isdisjoint(va)
            np.testing.assert_allclose(df.true,y[df.row.to_numpy()])

def test_windows_snapshot_paths_actually_check_model_hashes(monkeypatch):
    import src.validation as v
    real=v.sha
    def changed(path):
        if Path(path).name=='blend_model.joblib':return 'changed'
        return real(path)
    monkeypatch.setattr(v,'sha',changed)
    with pytest.raises(ValueError,match='Protected artifact changed'):v.assert_protected_hashes()

def test_json_writer_cannot_modify_confirmed_directory():
    from src.validation import write_json
    with pytest.raises(ValueError):write_json(ROOT/'versions/top1/should_never_exist.json',{'bad':True})
    assert not (ROOT/'versions/top1/should_never_exist.json').exists()

def test_neural_fit_tensors_do_not_depend_on_injected_concentrations(train):
    from src.neural_models import NeuralRegressor
    frame=train.iloc[:8].copy();y=frame['SI VACV'].to_numpy()
    a=NeuralRegressor('hybrid_gcn',epochs=1).fit(frame,y,42)
    frame['IC50_uM_VACV']='hidden';frame['CC50_uM']=1e100
    b=NeuralRegressor('hybrid_gcn',epochs=1).fit(frame,y,42)
    for x,z in zip(a.models[0].parameters(),b.models[0].parameters()):torch.testing.assert_close(x,z,rtol=0,atol=0)

def test_cli_prediction_defaults_to_confirmed_top1(monkeypatch,capsys):
    import src.cli as cli
    used=[]
    monkeypatch.setattr(sys,'argv',['main.py','predict','--smiles','CCO'])
    monkeypatch.setattr(cli,'predict_path',lambda path,frame:used.append(Path(path)) or np.array([1.]))
    monkeypatch.setattr(cli,'similar',lambda *args:[{'row':1}])
    cli.main()
    assert used==[ROOT/'versions/top1/blend_model.joblib']

def test_ready_submission_requires_model_checksum(tmp_path):
    _,p=copy_submission(tmp_path)
    with pytest.raises(ValueError,match='model checksum'):
        validate_submission(p,model=ROOT/'versions/top1/blend_model.joblib')

def test_submission_generation_never_overwrites_history(tmp_path):
    from src.artifacts import make_submission
    _,p=copy_submission(tmp_path)
    with pytest.raises(ValueError,match='already exists'):
        make_submission(ROOT/'versions/top1/blend_model.joblib',p)

def test_research_checkpoint_loads_in_a_fresh_python_process():
    import subprocess
    path=ROOT/'experiments/research_20261107/checkpoints/reference_canonical_42_1.joblib'
    code='import joblib; m=joblib.load('+repr(str(path))+'); assert type(m).__module__=="src.research"'
    result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True,timeout=40)
    assert result.returncode==0,result.stderr

def test_saved_champion_contains_only_official_training_references(train):
    b=joblib.load(ROOT/'versions/top1/blend_model.joblib')
    assert b['base_bundle']['ref_smiles']==train.SMILES.tolist()
    np.testing.assert_array_equal(b['base_bundle']['ref_si'],train['SI VACV'])
    assert b['base_bundle']['encoder'] is None
    assert int(b['base_bundle']['scaler'].n_samples_seen_)==len(train)
    assert int(b['graph_bundle']['scaler'].n_samples_seen_)==len(train)

@pytest.mark.parametrize('invalid',[{'cc50_relation':'>'},{'ic50_units':'mg/L'}])
def test_scientific_review_cannot_approve_invalid_measurements(invalid):
    from research_data_collection.validation import validate_record
    row={'SMILES':'CCO','source_url':'https://example.org/paper','ic50_value':1,'ic50_units':'uM','ic50_relation':'=',
         'cc50_value':10,'cc50_units':'uM','cc50_relation':'=','culture':'Vero','strain':'VACV','review_status':'approved',
         'compatibility_evidence':'same assay context','reviewer':'unit-test'}
    row.update(invalid)
    with pytest.raises(ValueError):validate_record(row)

def test_scientific_pending_record_is_never_approved():
    from research_data_collection.validation import validate_record
    assert validate_record({'SMILES':'CCO','source_url':'https://example.org/paper','review_status':'pending'})['scientifically_validated'] is False
