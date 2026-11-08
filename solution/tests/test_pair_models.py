import json
import joblib
import numpy as np
import pytest
import torch
import vacv_cpu_improved as cpu
from src.pair_research import PairNet,PairRegressor,training_pairs,similarity
from src.validation import training_data

def test_pair_network_is_antisymmetric_and_identity_zero():
    torch.manual_seed(42);model=PairNet(12);model.eval();a=torch.randn(5,12);b=torch.randn(5,12)
    with torch.no_grad():
        torch.testing.assert_close(model(a,b),-model(b,a),rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(model(a,a),torch.zeros(5),rtol=0,atol=0)

def test_pairs_exclude_self_and_identical_canonical_structures():
    raw=cpu.chem_features(['CCO','OCC','CCN','CCC','CCCl'])
    a,b,w=training_pairs(raw,42,2,1);edges=set(zip(a,b))
    assert np.all(a!=b) and np.all(raw['canon'][a]!=raw['canon'][b])
    assert all((j,i) in edges for i,j in edges)
    assert np.isfinite(w).all() and (w>0).all()
    aa,bb,_=training_pairs(raw,42,2,1);np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(b,bb)

@pytest.mark.parametrize('mode',['anchor_mean','ridge_log','ridge_raw','kernel_log','kernel_raw','neural_log','neural_raw'])
def test_pair_regression_persistence_training_scope_and_targets(mode,tmp_path):
    df=training_data();frame=df.iloc[:16];y=frame['SI VACV'].to_numpy();before=y.copy();ids=np.arange(16)
    model=PairRegressor(mode,epochs=2).fit(frame,y,42,ids)
    p,anchors,_=model.predict_details(df.iloc[16:19]);assert set(model.row_ids[anchors.ravel()]).issubset(ids)
    assert not set(model.row_ids[anchors.ravel()]).intersection(range(16,19))
    np.testing.assert_array_equal(y,before);assert int(model.scaler.n_samples_seen_)==16
    path=tmp_path/'pair.joblib';joblib.dump(model,path)
    np.testing.assert_allclose(joblib.load(path).predict(df.iloc[16:19]),p,rtol=1e-7,atol=1e-7)

def test_anchor_ablation_matches_weighted_known_train_si():
    df=training_data();frame=df.iloc[:16];model=PairRegressor('anchor_mean').fit(frame,frame['SI VACV'].to_numpy(),42)
    query=df.iloc[16:18];p,anchors,_=model.predict_details(query);s=similarity(cpu.chem_features(query.SMILES.tolist()),model.reference_raw)
    w=np.maximum(np.take_along_axis(s,anchors,axis=1),1e-6)**4
    expected=(model.y[anchors]*w).sum(1)/w.sum(1);np.testing.assert_allclose(p,expected)

def test_forbidden_columns_cannot_change_pair_model_fit():
    df=training_data().iloc[:16].copy();y=df['SI VACV'].to_numpy()
    a=PairRegressor('ridge_log').fit(df,y,42)
    df['IC50_uM_VACV']='forbidden';df['CC50_uM']=1e100
    b=PairRegressor('ridge_log').fit(df,y,42)
    np.testing.assert_array_equal(a.model.coef_,b.model.coef_)
    np.testing.assert_array_equal(a.pair_a,b.pair_a)

def test_actual_pair_experiments_have_no_endpoint_or_anchor_leakage():
    import pandas as pd
    from src.pair_research import WORK,MODES
    from src.validation import folds
    df=training_data();y=df['SI VACV'].to_numpy();raw=cpu.chem_features(df.SMILES.tolist())
    for seed in (42,43):
        split=folds(y,raw['canon'],seed)
        for mode in MODES:
            oof=pd.read_csv(WORK/f'{seed}_{mode}/oof.csv');assert len(oof)==len(df) and oof.row.nunique()==len(df)
            anchors=pd.read_csv(WORK/f'{seed}_{mode}/query_anchors.csv')
            for j,(tr,va) in enumerate(split,1):
                pairs=pd.read_csv(WORK/f'{seed}_{mode}'/f'training_pairs_{j}.csv')
                assert set(pairs.anchor_row).issubset(tr) and set(pairs.query_row).issubset(tr)
                for _,row in anchors[anchors.fold==j].iterrows():
                    assert int(row.row) in va and set(json.loads(row.anchor_training_rows)).issubset(tr)

def test_nested_pair_weight_fit_never_uses_outer_validation_rows():
    import pandas as pd
    from src.pair_research import WORK
    from src.validation import folds
    df=training_data();y=df['SI VACV'].to_numpy();raw=cpu.chem_features(df.SMILES.tolist())
    for seed in (42,43):
        for j,(tr,va) in enumerate(folds(y,raw['canon'],seed),1):
            inner=pd.read_csv(WORK/'nested'/str(seed)/f'inner_oof_{j}.csv')
            assert set(inner.row)==set(tr) and set(inner.row).isdisjoint(va)
