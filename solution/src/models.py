"""Compact molecular regressors with train-fold-only fitted transforms."""
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from src.features import kernel, tanimoto

class TanimotoKernel(Kernel):
    def __init__(self): pass
    def __call__(self,X,Y=None,eval_gradient=False):
        k=tanimoto(X,X if Y is None else Y)
        return (k,np.empty((*k.shape,0))) if eval_gradient else k
    def diag(self,X): return np.ones(len(X))
    def is_stationary(self): return False

class MoleculeRegressor:
    def __init__(self, config): self.config=dict(config)
    def fit(self,raw,y,seed=42):
        c=self.config; self.train_rows=len(y); self.keys=c.get('keys',['morgan2_2048'])
        self.log=c.get('target','log')=='log'; target=np.log1p(y) if self.log else np.asarray(y,float)
        family=c['family']
        if family=='krr':
            self.reference={k:raw[k].copy() for k in self.keys}
            self.model=KernelRidge(alpha=c.get('alpha',1.),kernel='precomputed').fit(kernel(raw,raw,self.keys),target)
        elif family=='gp':
            self.model=GaussianProcessRegressor(kernel=TanimotoKernel(),alpha=c.get('alpha',.1),optimizer=None,normalize_y=True,random_state=seed).fit(raw[self.keys[0]],target)
        else:
            x=self.tabular(raw)
            self.imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(x)
            x=self.imputer.transform(x); self.selector=VarianceThreshold().fit(x);x=self.selector.transform(x)
            if family=='catboost':
                from catboost import CatBoostRegressor
                self.model=CatBoostRegressor(iterations=c.get('iterations',180),depth=c.get('depth',4),learning_rate=c.get('lr',.05),l2_leaf_reg=c.get('l2',5.),loss_function='RMSE',random_seed=seed,thread_count=4,verbose=False,allow_writing_files=False,task_type='CPU')
            elif family=='extra_trees':
                self.model=ExtraTreesRegressor(n_estimators=200,min_samples_leaf=c.get('leaf',2),max_features=.8,random_state=seed,n_jobs=4)
            else: raise ValueError(f'Unknown family: {family}')
            self.model.fit(x,target)
        return self
    def tabular(self,raw):
        x=np.concatenate([raw[k] for k in self.config.get('tabular_keys',['morgan2_1024','maccs','desc'])],axis=1).astype(float)
        x[~np.isfinite(x)]=np.nan
        return x
    def predict(self,raw):
        c=self.config; family=c['family']
        if family=='krr': p=self.model.predict(kernel(raw,self.reference,self.keys))
        elif family=='gp': p=self.model.predict(raw[self.keys[0]])
        else: p=self.model.predict(self.selector.transform(self.imputer.transform(self.tabular(raw))))
        with np.errstate(over='raise'):
            p=np.expm1(p) if self.log else p
        p=np.maximum(np.asarray(p,float),0)
        if not np.isfinite(p).all(): raise ValueError('Non-finite predictions')
        return p
    def uncertainty(self,raw):
        if self.config['family']!='gp': raise ValueError('Uncertainty only implemented for GP')
        _,sd=self.model.predict(raw[self.keys[0]],return_std=True)
        return sd  # in transformed-target units for log GP; never label it SI std.

def configurations():
    result={}
    for key in ('morgan2_1024','morgan2_2048','morgan3_2048','atom_pair','torsion','rdkit2048'):
        for target in ('raw','log'):
            name=f'krr_{key}_{target}'
            result[name]={'family':'krr','keys':[key],'alpha':1.,'target':target}
    for alpha in (.03,.3,3.,30.):
        result[f'krr_mix_{alpha}']={'family':'krr','keys':['morgan2_2048','morgan3_2048'],'alpha':alpha,'target':'log'}
    for target in ('raw','log'):
        result[f'gp_{target}']={'family':'gp','keys':['morgan2_2048'],'alpha':.3,'target':target}
        for depth in (4,6):
            result[f'catboost_d{depth}_{target}']={'family':'catboost','depth':depth,'target':target,'iterations':180,'lr':.05,'l2':5.}
        result[f'catboost_desc_{target}']={'family':'catboost','depth':4,'target':target,'iterations':180,'tabular_keys':['desc']}
    result['extra_trees_log']={'family':'extra_trees','target':'log','leaf':2}
    return result
