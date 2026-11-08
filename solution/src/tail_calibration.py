"""Outer grouped CV of calibration learned exclusively from inner OOF."""
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.isotonic import IsotonicRegression
from src.research import WORK as BASE,prepare,folds_and_ids
from src.tail_research import WORK
from src.validation import ROOT,folds,metrics,write_json,sha,assert_protected_hashes

class CrossFitCalibrator:
    def __init__(self,kind='affine'):self.kind=kind
    def fit(self,p,y):
        p,y=np.asarray(p,float),np.asarray(y,float)
        if self.kind=='isotonic':self.model=IsotonicRegression(increasing=True,y_min=0.,out_of_bounds='clip').fit(p,y)
        elif self.kind=='affine':
            self.input_scale=max(float(np.sqrt(np.mean(p*p))),1.);self.target_scale=max(float(np.sqrt(np.mean(y*y))),1.)
            a=np.column_stack([p/self.input_scale,np.ones(len(p))]);target=y/self.target_scale
            penalty=.01*len(p);prior=np.array([self.input_scale/self.target_scale,0.])
            self.coef=lsq_linear(np.vstack([a,np.sqrt(penalty)*np.eye(2)]),np.r_[target,np.sqrt(penalty)*prior],bounds=(0,np.inf)).x
        else:raise ValueError('Unknown calibration')
        return self
    def predict(self,p):
        p=np.asarray(p,float)
        if self.kind=='isotonic':return self.model.predict(p)
        return self.target_scale*(self.coef[0]*p/self.input_scale+self.coef[1])

if __name__=='__main__':sys.modules['src.tail_calibration']=sys.modules[__name__]
CrossFitCalibrator.__module__='src.tail_calibration'

def run():
    assert_protected_hashes();df,raw,y=prepare();records=[]
    for seed in (42,43):
        pairs,ids=folds_and_ids(y,raw,'canonical',seed)
        reference=pd.read_csv(BASE/'runs'/f'canonical_{seed}_reference350'/'oof.csv').sort_values('row')
        outputs={name:np.zeros(len(y)) for name in ('affine_reference','isotonic_reference','affine_gcn','isotonic_gcn')}
        for j,(tr,va) in enumerate(pairs,1):
            inner=folds(y[tr],raw['canonical'][tr],seed+100+j,3)
            q_cpu=np.zeros(len(tr));q_gcn=np.zeros(len(tr))
            for k,(itr,iva) in enumerate(inner,1):
                checkpoint=BASE/'checkpoints'/f'nested_reference350_canonical_{seed}_{j}_{k}.joblib'
                model=joblib.load(checkpoint)
                q_cpu[iva],q_gcn[iva]=model.predict_components(df.iloc[tr[iva]].reset_index(drop=True))
            inner_reference=.7*q_cpu+.3*q_gcn
            prior=pd.read_csv(BASE/'nested'/f'canonical_{seed}_tree350'/f'inner_oof_{j}.csv')
            np.testing.assert_array_equal(prior.row,tr);np.testing.assert_allclose(inner_reference,prior.baseline_inner_oof,rtol=1e-9,atol=1e-7)
            for name in outputs:
                kind,component=name.split('_');input_pred=inner_reference if component=='reference' else q_gcn
                calibrator=CrossFitCalibrator(kind).fit(input_pred,y[tr])
                q=reference.pred.to_numpy()[va] if component=='reference' else reference.gcn_pred.to_numpy()[va]
                output=calibrator.predict(q)
                if component=='gcn':output=.7*reference.cpu_pred.to_numpy()[va]+.3*output
                outputs[name][va]=output
                directory=WORK/'calibration'/f'{seed}_{name}';directory.mkdir(parents=True,exist_ok=True)
                joblib.dump(calibrator,directory/f'fold_{j}.joblib',compress=3)
            pd.DataFrame({'row':tr,'true':y[tr],'cpu_inner_oof':q_cpu,'gcn_inner_oof':q_gcn}).to_csv(WORK/'calibration'/f'inner_oof_{seed}_{j}.csv',index=False)
            print(f'Calibration seed {seed}, outer fold {j}: inner-only fits completed',flush=True)
        for name,p in outputs.items():
            out=WORK/'calibration'/f'{seed}_{name}';score=metrics(y,p)
            pd.DataFrame({'row':np.arange(len(y)),'seed':seed,'fold':ids,'true':y,'pred':p,'reference_pred':reference.pred}).to_csv(out/'oof.csv',index=False)
            report={'name':name,'seed':seed,'metrics':score,'reference_metrics':metrics(y,reference.pred),'extreme_prediction':float(p[np.argmax(y)]),
                    'fold_metrics':{str(j):metrics(y[ids==j],p[ids==j]) for j in range(1,6)},'train_sha256':sha(ROOT/'train.csv'),
                    'note':'Calibrator learned only from 3-fold inner OOF of outer training data. Test untouched; no target modification.'}
            write_json(out/'result.json',report)
            records.append({'mode':name,'seed':seed,'MSE':score['MSE'],'MAE':score['MAE'],'reference_MSE':report['reference_metrics']['MSE'],
                            'extreme_prediction':report['extreme_prediction'],'MSE_without_extreme_DIAGNOSTIC':score['MSE_without_max_DIAGNOSTIC_ONLY']})
    pd.DataFrame(records).to_csv(ROOT/'reports/tail_calibration_comparison.csv',index=False);assert_protected_hashes()
    print(pd.DataFrame(records).to_string(index=False))

if __name__=='__main__':run()
