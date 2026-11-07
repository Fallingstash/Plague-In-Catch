import joblib
import numpy as np
import pytest
import torch
from src.tail_research import MODES,TailRegressor,objective,decode
from src.validation import training_data

@pytest.mark.parametrize('mode',MODES)
def test_tail_losses_keep_targets_and_gradients_finite(mode):
    y=torch.tensor([0.,1.,100.,5000.,36066.667]);before=y.clone()
    width=2 if mode=='dual_raw' else 3 if mode=='mixture' else None
    output=torch.zeros((len(y),width) if width else (len(y),),requires_grad=True)
    loss=objective(output,y,mode,100.,1000.);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(output.grad).all()
    assert (decode(output,mode)>=0).all()
    torch.testing.assert_close(y,before,rtol=0,atol=0)

def test_tail_persistence_and_prediction_do_not_refit_thresholds(tmp_path):
    df=training_data().iloc[:12].copy();y=df['SI VACV'].to_numpy();before=y.copy()
    model=TailRegressor('mixture',epochs=2).fit(df,y,42)
    threshold=model.tail_threshold;scale=model.weight_scale;p=model.predict(df)
    path=tmp_path/'tail.joblib';joblib.dump(model,path);loaded=joblib.load(path)
    np.testing.assert_allclose(p,loaded.predict(df),rtol=1e-7,atol=1e-7)
    assert loaded.tail_threshold==threshold and loaded.weight_scale==scale
    np.testing.assert_array_equal(y,before)

@pytest.mark.parametrize('kind',['affine','isotonic'])
def test_crossfit_calibration_keeps_targets_and_persists(kind,tmp_path):
    from src.tail_calibration import CrossFitCalibrator
    p=np.array([0.,1.,10.,100.,1000.]);y=np.array([0.,5.,20.,500.,4000.]);before=y.copy()
    c=CrossFitCalibrator(kind).fit(p,y);expected=c.predict(p)
    file=tmp_path/'calibrator.joblib';joblib.dump(c,file);np.testing.assert_allclose(joblib.load(file).predict(p),expected)
    np.testing.assert_array_equal(y,before)
    assert (c.predict(np.array([0.,1e6]))>=0).all()

def test_calibrator_training_rows_exclude_outer_validation():
    import pandas as pd
    from src.tail_research import WORK
    from src.research import prepare,folds_and_ids
    df,raw,y=prepare()
    for seed in (42,43):
        for j,(tr,va) in enumerate(folds_and_ids(y,raw,'canonical',seed)[0],1):
            frame=pd.read_csv(WORK/'calibration'/f'inner_oof_{seed}_{j}.csv')
            assert set(frame.row)==set(tr) and set(frame.row).isdisjoint(va)
