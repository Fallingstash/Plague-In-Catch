"""Resumable CPU research. Selection code never opens test.csv."""
import argparse
import copy
import hashlib
import json
import time
import warnings
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor
from src.validation import ROOT, training_data, folds, metrics, write_json, sha, assert_protected_hashes
from src.features import features, subset, tanimoto, VERSION
from src.models import MoleculeRegressor, configurations
import vacv_cpu_improved as cpu
from versions.test.training import fit as graph_fit, pred as graph_pred, graph, features as graph_features, transform

WORK = ROOT/'experiments/research_20261107'
torch.set_num_threads(4)

class ReferenceFit:
    """Fixed 70/30 recipe, each component fit on the current training fold only."""
    def fit(self, frame, y, seed, fold, epochs=60,tree_estimators=350):
        raw=cpu.chem_features(frame.SMILES.tolist()); meta=cpu.metadata_from_df(frame)
        ids=np.arange(len(y)); groups=raw['canon']
        x,self.scaler,self.encoder=cpu.fit_transform_features(raw,meta,ids,True)
        self.tree=ExtraTreesRegressor(n_estimators=tree_estimators,min_samples_leaf=2,max_features=.8,random_state=seed+fold,n_jobs=4).fit(x,y)
        self.nets=[];self.epochs=[]
        for i in range(2):
            init=seed+100*i+17*fold
            ep=cpu.early_stopping_epochs(x,y,ids,groups,init,'log',epochs,25,192,.2,.001,.01,raw,meta,True)
            self.epochs.append(ep)
            self.nets.append(cpu.run_net_epochs(x,y,ids,init,'log',ep,192,.2,.001,.01))
        gx,self.graph_scaler=graph_features(raw,ids)
        self.graph_net=graph_fit([graph(s) for s in frame.SMILES],gx,y,ids,seed*100+fold-1,80)
        self.training_rows=len(y)
        return self
    def predict_components(self,frame):
        raw=cpu.chem_features(frame.SMILES.tolist());meta=cpu.metadata_from_df(frame)
        x=cpu.transform_features(raw,meta,self.scaler,None)
        a=.7*np.mean([cpu.net_predict(m,x) for m in self.nets],axis=0)+.3*np.maximum(self.tree.predict(x),0)
        b=graph_pred(self.graph_net,[graph(s) for s in frame.SMILES],transform(raw,self.graph_scaler),np.arange(len(frame)))
        return a,b
    def predict(self,frame):
        a,b=self.predict_components(frame);return .7*a+.3*b

# Stable pickle namespace even when research is launched with `python -m`.
if __name__=='__main__':sys.modules['src.research']=sys.modules[__name__]
ReferenceFit.__module__='src.research'
# Compatibility for checkpoints created before the namespace fix.
if not hasattr(sys.modules['__main__'],'ReferenceFit'):
    sys.modules['__main__'].ReferenceFit=ReferenceFit

def prepare():
    assert_protected_hashes()
    df=training_data();raw=features(df.SMILES,WORK/'feature_cache')
    return df,raw,df['SI VACV'].to_numpy(float)

def source_digest():
    paths=list((ROOT/'src').glob('*.py'))+[ROOT/'main.py',ROOT/'competition_data.py',ROOT/'vacv_cpu_improved.py',ROOT/'vacv_neural.py',ROOT/'gcn_predict.py',ROOT/'versions/test/training.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}

def folds_and_ids(y,raw,cv,seed,count=5):
    groups=raw['canonical' if cv=='canonical' else 'scaffold']
    pairs=folds(y,groups,seed,count);ids=np.zeros(len(y),int)
    for j,(_,va) in enumerate(pairs,1):ids[va]=j
    return pairs,ids

def save_run(name,cv,seed,config,p,y,fold_ids,duration,extra=None):
    out=WORK/'runs'/f'{cv}_{seed}_{name}';out.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame({'row':np.arange(len(y)),'seed':seed,'fold':fold_ids,'true':y,'pred':p})
    if extra:
        for k,v in extra.items():frame[k]=v
    frame.to_csv(out/'oof.csv',index=False)
    report={'model':name,'cv':cv,'seed':seed,'config':config,'training_sha256':sha(ROOT/'train.csv'),'seconds':duration,
            'metrics':metrics(y,p),'fold_metrics':{str(f):metrics(y[fold_ids==f],p[fold_ids==f]) for f in np.unique(fold_ids)},
            'code_sha256':source_digest(),'feature_version':VERSION,'selection_note':'Screening OOF only; not an independent estimate after selecting a configuration.'}
    write_json(out/'result.json',report)
    update_leaderboard()
    return report

def update_leaderboard():
    entries=[]
    for path in (WORK/'runs').glob('*/result.json'):
        r=json.loads(path.read_text(encoding='utf-8'))
        entries.append({'id':path.parent.name,'model':r['model'],'cv':r['cv'],'seed':r['seed'],
                        'training_mse':None,'oof_mse':r['metrics']['MSE'],'oof_mae':r['metrics']['MAE'],
                        'nested_cv_mse':None,'kaggle_public_mse':None,'kaggle_private_mse':None,'seconds':r['seconds']})
    for path in (WORK/'nested').glob('*/result.json'):
        r=json.loads(path.read_text(encoding='utf-8'))
        entries.append({'id':path.parent.name,'model':f'nested_selection_tree{r.get("tree_estimators",250)}','cv':r['cv'],'seed':r['seed'],
                        'training_mse':None,'oof_mse':None,'oof_mae':r['metrics']['MAE'],'nested_cv_mse':r['metrics']['MSE'],
                        'kaggle_public_mse':None,'kaggle_private_mse':None,'seconds':r['seconds']})
    if entries:pd.DataFrame(entries).to_csv(ROOT/'reports/experiment_leaderboard.csv',index=False)

def reference_cv(cv='canonical',seeds=(42,43),force=False):
    df,raw,y=prepare()
    for seed in seeds:
        pairs,ids=folds_and_ids(y,raw,cv,seed)
        out=WORK/'runs'/f'{cv}_{seed}_reference'
        if (out/'result.json').exists() and not force:
            old=json.loads((out/'result.json').read_text(encoding='utf-8'))
            if old['training_sha256']!=sha(ROOT/'train.csv'):raise ValueError('Cached reference uses different training data')
            print(f'cached reference {cv} {seed}: MSE {old["metrics"]["MSE"]:.2f}',flush=True)
            continue
        begin=time.monotonic();a=np.zeros(len(y));b=np.zeros(len(y));sim=np.zeros(len(y));new_scaffold=np.zeros(len(y),bool)
        for j,(tr,va) in enumerate(pairs,1):
            file=WORK/'checkpoints'/f'reference_{cv}_{seed}_{j}.joblib';file.parent.mkdir(parents=True,exist_ok=True)
            if file.exists():model=joblib.load(file)
            else:
                model=ReferenceFit().fit(df.iloc[tr].reset_index(drop=True),y[tr],seed,j,tree_estimators=250)
                joblib.dump(model,file,compress=3)
            a[va],b[va]=model.predict_components(df.iloc[va].reset_index(drop=True))
            sim[va]=tanimoto(raw['morgan2_2048'][va],raw['morgan2_2048'][tr]).max(axis=1)
            new_scaffold[va]=~np.isin(raw['scaffold'][va],raw['scaffold'][tr])
            print(f'reference {cv} {seed} fold {j}: MSE {np.mean((y[va]-(.7*a[va]+.3*b[va]))**2):.0f}',flush=True)
        save_run('reference',cv,seed,{'cpu_weight':.7,'gcn_weight':.3,'cpu_epochs_max':60,'gcn_epochs':80},.7*a+.3*b,y,ids,time.monotonic()-begin,
                 {'cpu_pred':a,'gcn_pred':b,'max_train_similarity':sim,'new_scaffold':new_scaffold})

def evaluate(cv='canonical',seeds=(42,43),names=None):
    df,raw,y=prepare();configs=configurations()
    if (WORK/'optuna_best.json').exists():
        configs['optuna_catboost_tuned']=json.loads((WORK/'optuna_best.json').read_text(encoding='utf-8'))['config']
    if names:configs={n:configs[n] for n in names}
    for seed in seeds:
        pairs,ids=folds_and_ids(y,raw,cv,seed)
        write_json(WORK/'folds'/f'{cv}_{seed}.json',{'training_sha256':sha(ROOT/'train.csv'),'folds':[{'train':tr.tolist(),'validation':va.tolist()} for tr,va in pairs]})
        for name,c in configs.items():
            out=WORK/'runs'/f'{cv}_{seed}_{name}'
            models=out/'models';models.mkdir(parents=True,exist_ok=True)
            if (out/'result.json').exists():
                previous=json.loads((out/'result.json').read_text(encoding='utf-8'))
                model_file=str(Path('src/models.py'))
                if previous['config']!=c or previous['training_sha256']!=sha(ROOT/'train.csv') or previous['feature_version']!=VERSION or previous['code_sha256'].get(model_file)!=sha(ROOT/model_file):
                    raise ValueError('Cached model provenance changed; use a new research experiment ID')
                if all((models/f'fold_{j}.joblib').exists() for j in range(1,6)):continue
            begin=time.monotonic();p=np.full(len(y),np.nan)
            uncertainty=np.full(len(y),np.nan) if c['family']=='gp' else None
            try:
                for j,(tr,va) in enumerate(pairs,1):
                    checkpoint=models/f'fold_{j}.joblib'
                    if checkpoint.exists():model=joblib.load(checkpoint)
                    else:
                        model=MoleculeRegressor(c).fit(subset(raw,tr),y[tr],seed+j)
                        joblib.dump(model,checkpoint,compress=3)
                    p[va]=model.predict(subset(raw,va))
                    if uncertainty is not None:uncertainty[va]=model.uncertainty(subset(raw,va))
                save_run(name,cv,seed,c,p,y,ids,time.monotonic()-begin,{'predictive_std_transformed_units':uncertainty} if uncertainty is not None else None)
                print(f'{cv} {seed} {name}: MSE {np.mean((y-p)**2):.0f} [{time.monotonic()-begin:.1f}s]',flush=True)
            except Exception as exc:
                write_json(out/'failure.json',{'config':c,'error':repr(exc),'seconds':time.monotonic()-begin})
                raise

def audit():
    df,raw,y=prepare(); rows=[];details=[]
    saved=joblib.load(ROOT/'versions/top1/blend_model.joblib');base=saved['base_bundle'];gcn=saved['graph_bundle']
    official_raw=cpu.chem_features(df.SMILES.tolist())
    assert base['ref_smiles']==df.SMILES.tolist() and np.array_equal(np.asarray(base['ref_si']),y)
    assert base['encoder'] is None and base['meta_weight']==0
    np.testing.assert_allclose(base['scaler'].mean_,official_raw['desc'].mean(axis=0,dtype=np.float64),rtol=1e-8,atol=1e-8)
    np.testing.assert_allclose(gcn['scaler'].mean_,base['scaler'].mean_,rtol=1e-8,atol=1e-8)
    assert int(base['scaler'].n_samples_seen_)==320 and int(gcn['scaler'].n_samples_seen_)==320
    saved_summary={'graph_weight':saved['graph_weight'],'cpu_weights':base['weights'],'cpu_tree_estimators':base['tree'].n_estimators,
                   'cpu_tree_input_features':base['tree'].n_features_in_,'cpu_saved_networks':len(base['nets']),
                   'cpu_descriptor_columns':int(base['scaler'].n_features_in_),'graph_extra_dim':gcn['extra_dim'],
                   'graph_epochs':gcn['epochs'],'graph_seed':gcn['seed'],'training_reference_rows':len(base['ref_si']),
                   'graph_training_rows':gcn['training_rows'],'all_scalers_match_official_train':True,'metadata_used':False}
    write_json(ROOT/'reports/saved_model_audit.json',saved_summary)
    old=pd.read_csv(ROOT/'reports/matched_blend_oof_predictions.csv')
    for seed,g in old.groupby('seed'):
        g=g.sort_values('row');pairs,ids=folds_and_ids(y,raw,'canonical',int(seed))
        assert np.allclose(g.true,y) and np.array_equal(g.fold.to_numpy(),ids)
        similarity=np.zeros(len(y));new=np.zeros(len(y),bool)
        for tr,va in pairs:
            similarity[va]=tanimoto(raw['morgan2_2048'][va],raw['morgan2_2048'][tr]).max(1)
            new[va]=~np.isin(raw['scaffold'][va],raw['scaffold'][tr])
        a=g.cpu_oof.to_numpy();b=g.gcn_oof.to_numpy();p=.7*a+.3*b
        for i in range(len(y)):
            details.append({'seed':int(seed),'row':i,'fold':int(ids[i]),'SMILES':df.SMILES.iloc[i],'canonical':raw['canonical'][i],'scaffold':raw['scaffold'][i],
                            'true':float(y[i]),'cpu_pred':float(a[i]),'gcn_pred':float(b[i]),'blend_pred':float(p[i]),
                            'cpu_squared_error':float((a[i]-y[i])**2),'gcn_squared_error':float((b[i]-y[i])**2),'blend_squared_error':float((p[i]-y[i])**2),
                            'cpu_absolute_error':float(abs(a[i]-y[i])),'gcn_absolute_error':float(abs(b[i]-y[i])),
                            'max_train_similarity':float(similarity[i]),'new_scaffold':bool(new[i])})
        rows.append({'seed':int(seed),'cpu':metrics(y,a),'gcn':metrics(y,b),'blend':metrics(y,p),
                     'cpu_gcn_error_correlation':float(np.corrcoef(a-y,b-y)[0,1]),'gcn_wins':int(((b-y)**2<(a-y)**2).sum()),
                     'max_target_row':int(np.argmax(y))})
    pd.DataFrame(details).to_csv(ROOT/'reports/molecule_error_audit.csv',index=False)
    collision={k:int(len(set(raw['canonical']))-len(set(np.packbits(raw[k].astype(np.uint8),axis=1).tobytes()[i*int(raw[k].shape[1]/8):(i+1)*int(raw[k].shape[1]/8)] for i in range(len(y))))) for k in []}
    feature_diagnostics={k:{'width':int(raw[k].shape[1]),'constant_columns':int((np.ptp(raw[k],axis=0)==0).sum()),'unique_fingerprints':len(np.unique(raw[k],axis=0))} for k in raw if k not in ('canonical','scaffold','desc')}
    diagnostic={'train_sha256':sha(ROOT/'train.csv'),'rows':len(y),'canonical_groups':len(set(raw['canonical'])),'scaffolds':len(set(raw['scaffold'])),
                'zeros':int((y==0).sum()),'max_SI':float(y.max()),'descriptors_nonfinite':int((~np.isfinite(raw['desc'])).sum()),'features':feature_diagnostics,'scores':rows}
    write_json(ROOT/'reports/data_audit_details.json',diagnostic)
    lines=['# Data and model audit','',f'Official train SHA256: `{sha(ROOT/"train.csv")}`.',f'{len(y)} rows; {len(set(raw["canonical"]))} canonical groups; {(y==0).sum()} zeros; max SI={y.max()}.',
           'Targets are never recomputed. Only official SMILES and SI enter the new pipeline.',
           'Old 1%-sweep with a test max-prediction constraint was transductive model selection and is excluded from future selection.',
           'No Git repository exists; code snapshots and SHA256 manifests are used instead.',
           'The original second-layer submission (898313.50898) is the actual second-best reported score. Its overwritten checkpoint and original CSV have now been recovered in versions/candidates/historical_two_layer_70_30. The older protected top2 folder is preserved; versions/ranking.json records the actual ranking.',
           'The old default inference model was a stale 5% blend; CLI default now targets top1.',
           'CPU selects epochs in a group-disjoint inner holdout. GCN uses 80 fixed epochs, no early stopping. Graph adjacency ignores bond type; mean pooling may dilute rare substructures.',
           'Legacy CPU CV used 250 trees while final CPU used 350. Legacy runs reproduce the old numbers; new reference350 and nested tree350 runs correct that mismatch without changing historical artifacts.',
           'GCN inverse transform clamps log predictions to [0,14] for numerical behavior; it is not a validation-derived SI cutoff.',
           'Existing ensemble OOF weights were tuned on the same OOF and are exploratory. Prospective nested selection is reported separately.','']
    for r in rows:
        lines.append(f'Seed {r["seed"]}: CPU MSE {r["cpu"]["MSE"]:.2f}; GCN {r["gcn"]["MSE"]:.2f}; fixed 70/30 {r["blend"]["MSE"]:.2f}. Extreme row error share: {r["blend"]["max_target_error_share"]:.2%}. GCN wins on {r["gcn_wins"]}/320 rows. Error correlation {r["cpu_gcn_error_correlation"]:.4f}.')
    Path(ROOT/'reports/data_and_model_audit.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    print('Audit saved; protected hashes verified',flush=True)

def optimize(trials=8):
    import optuna
    df,raw,y=prepare(); pairs,_=folds_and_ids(y,raw,'canonical',42,3)
    study=optuna.create_study(study_name='train_only_catboost_v1',storage='sqlite:///'+str(WORK/'optuna.sqlite3').replace('\\','/'),load_if_exists=True,direction='minimize',sampler=optuna.samplers.TPESampler(seed=42),pruner=optuna.pruners.MedianPruner(n_startup_trials=3,n_warmup_steps=1))
    def objective(trial):
        c={'family':'catboost','depth':trial.suggest_int('depth',3,6),'iterations':trial.suggest_int('iterations',100,300,step=100),'l2':trial.suggest_float('l2',1.,20.,log=True),'lr':trial.suggest_float('lr',.025,.1,log=True),'target':trial.suggest_categorical('target',['raw','log']),'tabular_keys':trial.suggest_categorical('feature_set',['desc','combined'])}
        c['tabular_keys']=['desc'] if c['tabular_keys']=='desc' else ['morgan2_1024','maccs','desc']
        trial.set_user_attr('config',c);trial.set_user_attr('training_sha256',sha(ROOT/'train.csv'));p=np.zeros(len(y));seen=[]
        for j,(tr,va) in enumerate(pairs):
            model=MoleculeRegressor(c).fit(subset(raw,tr),y[tr],42+j);p[va]=model.predict(subset(raw,va));seen.extend(va.tolist())
            trial.report(float(np.mean((y[seen]-p[seen])**2)),j)
            if trial.should_prune():raise optuna.TrialPruned()
        return float(np.mean((y-p)**2))
    study.optimize(objective,n_trials=trials,timeout=180)
    study.trials_dataframe().to_csv(WORK/'optuna_trials.csv',index=False)
    write_json(WORK/'optuna_best.json',{'config':study.best_trial.user_attrs['config'],'screening_mse':study.best_value,'note':'Three-fold training-only tuning; not an independent assessment.'})

def neural_cv(cv='canonical',seeds=(42,43),modes=None):
    from src.neural_models import NeuralRegressor
    df,raw,y=prepare()
    for seed in seeds:
        pairs,ids=folds_and_ids(y,raw,cv,seed)
        for mode in modes or ['gcn_mean3','bond_gcn','hybrid_gcn','residual_mlp','dmpnn']:
            out=WORK/'runs'/f'{cv}_{seed}_{mode}'
            if (out/'result.json').exists():continue
            begin=time.monotonic();p=np.zeros(len(y))
            for j,(tr,va) in enumerate(pairs):
                model=NeuralRegressor(mode).fit(df.iloc[tr].reset_index(drop=True),y[tr],seed*100+j)
                p[va]=model.predict(df.iloc[va].reset_index(drop=True))
                checkpoint=WORK/'checkpoints'/f'{cv}_{seed}_{mode}_{j}.joblib';checkpoint.parent.mkdir(parents=True,exist_ok=True);joblib.dump(model,checkpoint,compress=3)
                print(f'{mode} {cv} {seed} fold {j+1}: MSE {np.mean((y[va]-p[va])**2):.0f}',flush=True)
            save_run(mode,cv,seed,{'family':'neural','mode':mode,'epochs':80},p,y,ids,time.monotonic()-begin)

def upgrade_tree(model,frame,y,seed,fold):
    model=copy.deepcopy(model)
    raw=cpu.chem_features(frame.SMILES.tolist());x=cpu.transform_features(raw,cpu.metadata_from_df(frame),model.scaler,None)
    model.tree=ExtraTreesRegressor(n_estimators=350,min_samples_leaf=2,max_features=.8,random_state=seed+fold,n_jobs=4).fit(x,y)
    return model

def matched_reference_cv(cv='canonical',seeds=(42,43)):
    reference_cv(cv,seeds);df,raw,y=prepare()
    for seed in seeds:
        pairs,ids=folds_and_ids(y,raw,cv,seed);out=WORK/'runs'/f'{cv}_{seed}_reference350'
        if (out/'result.json').exists():continue
        begin=time.monotonic();a=np.zeros(len(y));b=np.zeros(len(y))
        for j,(tr,va) in enumerate(pairs,1):
            checkpoint=WORK/'checkpoints'/f'reference350_{cv}_{seed}_{j}.joblib'
            if checkpoint.exists():model=joblib.load(checkpoint)
            else:
                prior=joblib.load(WORK/'checkpoints'/f'reference_{cv}_{seed}_{j}.joblib')
                model=upgrade_tree(prior,df.iloc[tr].reset_index(drop=True),y[tr],seed,j)
                joblib.dump(model,checkpoint,compress=3)
            a[va],b[va]=model.predict_components(df.iloc[va].reset_index(drop=True))
        save_run('reference350',cv,seed,{'cpu_weight':.7,'gcn_weight':.3,'tree_estimators':350,'purpose':'Match the final saved CPU recipe; reuse only training-fold neural fits'},.7*a+.3*b,y,ids,time.monotonic()-begin,{'cpu_pred':a,'gcn_pred':b})
        print(f'matched350 reference {cv} {seed}: MSE {np.mean((y-(.7*a+.3*b))**2):.2f}',flush=True)

def nested_cv(cv='canonical',seeds=(42,43),budget_seconds=900,tree_estimators=350):
    """All family/weight choices use inner OOF, never the held-out outer targets."""
    df,raw,y=prepare();started=time.monotonic()
    catalog=configurations()
    # Predeclared search space, not filtered using full-data screening results.
    pool={k:catalog[k] for k in ['krr_mix_0.03','krr_mix_0.3','krr_morgan2_2048_raw','gp_log','catboost_d4_raw','catboost_d4_log']}
    write_json(WORK/f'nested_search_space_{tree_estimators}.json',{'pool':pool,'weights':[0,.1,.2,.3,1.],
               'gates':['static','similarity','ridge_stacking'],'inner_folds':3,
               'note':'The search space was fixed before evaluating outer-fold targets. Global Optuna results are not imported.'})
    if tree_estimators==350:matched_reference_cv(cv,seeds)
    else:reference_cv(cv,seeds)
    for seed in seeds:
        suffix='_tree350' if tree_estimators==350 else ''
        out=WORK/'nested'/f'{cv}_{seed}{suffix}';out.mkdir(parents=True,exist_ok=True)
        if (out/'result.json').exists():continue
        begin=time.monotonic();pairs,ids=folds_and_ids(y,raw,cv,seed);outer_p=np.full(len(y),np.nan);decisions=[]
        for j,(tr,va) in enumerate(pairs,1):
            saved=out/f'fold_{j}.joblib'
            if saved.exists():
                artifact=joblib.load(saved);outer_p[va]=artifact['pred'];decisions.append(artifact['selection']);continue
            if time.monotonic()-started>budget_seconds:
                write_json(out/'incomplete.json',{'completed_folds':len(decisions),'status':'budget_exhausted_resume_required'});return
            prior_inner_file=WORK/'nested'/f'{cv}_{seed}'/f'inner_oof_{j}.csv'
            prior_inner=pd.read_csv(prior_inner_file) if tree_estimators==350 and prior_inner_file.exists() else None
            if prior_inner is not None:assert np.array_equal(prior_inner.row.to_numpy(),tr)
            inner_groups=raw['canonical' if cv=='canonical' else 'scaffold'][tr]
            inner=folds(y[tr],inner_groups,seed+100+j,3)
            baseline=np.zeros(len(tr));similarity=np.zeros(len(tr));newp={name:np.zeros(len(tr)) for name in pool}
            for k,(itr,iva) in enumerate(inner,1):
                a,b=tr[itr],tr[iva]
                cache=WORK/'checkpoints'/f'nested_reference{350 if tree_estimators==350 else ""}_{cv}_{seed}_{j}_{k}.joblib'
                if cache.exists():base=joblib.load(cache)
                else:
                    legacy=WORK/'checkpoints'/f'nested_reference_{cv}_{seed}_{j}_{k}.joblib'
                    if tree_estimators==350 and legacy.exists():base=upgrade_tree(joblib.load(legacy),df.iloc[a].reset_index(drop=True),y[a],seed+1000+j*10,k)
                    else:base=ReferenceFit().fit(df.iloc[a].reset_index(drop=True),y[a],seed+1000+j*10,k,tree_estimators=tree_estimators)
                    joblib.dump(base,cache,compress=3)
                baseline[iva]=base.predict(df.iloc[b].reset_index(drop=True))
                similarity[iva]=tanimoto(raw['morgan2_2048'][b],raw['morgan2_2048'][a]).max(1)
                for name,c in pool.items():
                    if prior_inner is not None:newp[name][iva]=prior_inner[name].to_numpy()[iva]
                    else:
                        model=MoleculeRegressor(c).fit(subset(raw,a),y[a],seed+j*10+k)
                        newp[name][iva]=model.predict(subset(raw,b))
                print(f'nested {cv} {seed} outer {j}/5 inner {k}/3 completed',flush=True)
            target=y[tr];best={'model':'reference','mode':'static','weight':0.,'inner_mse':float(np.mean((target-baseline)**2))}
            for name,p in newp.items():
                delta=p-baseline
                learned=float(np.clip(np.dot(delta,target-baseline)/(np.dot(delta,delta)*1.1+1e-12),0,.3))
                choices=[('static',w) for w in (.1,.2,.3,1.)]+[('similarity',w) for w in (.1,.2,.3)]+[('ridge_stacking',learned)]
                for mode,w in choices:
                    weight=w*(1-.5*similarity) if mode=='similarity' else w
                    mse=float(np.mean((target-(baseline+weight*delta))**2))
                    if mse<best['inner_mse']:best={'model':name,'mode':mode,'weight':w,'inner_mse':mse}
            reference_name='reference350' if tree_estimators==350 else 'reference'
            base=joblib.load(WORK/'checkpoints'/f'{reference_name}_{cv}_{seed}_{j}.joblib')
            p0=base.predict(df.iloc[va].reset_index(drop=True));chosen=None
            if best['model']=='reference':p=p0
            else:
                chosen=MoleculeRegressor(pool[best['model']]).fit(subset(raw,tr),y[tr],seed+j)
                p1=chosen.predict(subset(raw,va));weight=best['weight']
                if best['mode']=='similarity':weight=weight*(1-.5*tanimoto(raw['morgan2_2048'][va],raw['morgan2_2048'][tr]).max(1))
                p=(1-weight)*p0+weight*p1
            outer_p[va]=p;best['outer_fold']=j;decisions.append(best)
            joblib.dump({'selection':best,'pred':p,'validation_rows':va,'model':chosen},saved,compress=3)
            inner_frame=pd.DataFrame({'row':tr,'true':target,'baseline_inner_oof':baseline,'max_similarity':similarity,**newp});inner_frame.to_csv(out/f'inner_oof_{j}.csv',index=False)
            print(f'nested {seed} outer {j}: selected {best}; outer MSE {np.mean((y[va]-p)**2):.0f}',flush=True)
        reference_name='reference350' if tree_estimators==350 else 'reference'
        base_frame=pd.read_csv(WORK/'runs'/f'{cv}_{seed}_{reference_name}'/'oof.csv')
        pd.DataFrame({'row':np.arange(len(y)),'seed':seed,'fold':ids,'true':y,'pred':outer_p,'reference_pred':base_frame.pred}).to_csv(out/'oof.csv',index=False)
        result={'cv':cv,'seed':seed,'tree_estimators':tree_estimators,'metrics':metrics(y,outer_p),'reference_metrics':metrics(y,base_frame.pred),
                'fold_metrics':{str(f):metrics(y[ids==f],outer_p[ids==f]) for f in np.unique(ids)},
                'selections':decisions,'seconds':time.monotonic()-begin,'train_sha256':sha(ROOT/'train.csv'),
                'note':'Family, regularization, gate and weights selected on inner OOF only. Outer targets used only for final scoring. Historical baseline recipe was developed previously on this same dataset, so this is not a new independent dataset.'}
        write_json(out/'result.json',result);update_leaderboard()

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['audit','reference','screen','optimize','neural','nested']);p.add_argument('--cv',choices=['canonical','scaffold'],default='canonical');p.add_argument('--seeds',default='42,43');p.add_argument('--models');p.add_argument('--trials',type=int,default=8)
    a=p.parse_args();seeds=tuple(int(x) for x in a.seeds.split(','))
    if a.command=='audit':audit()
    elif a.command=='reference':reference_cv(a.cv,seeds)
    elif a.command=='screen':evaluate(a.cv,seeds,a.models.split(',') if a.models else None)
    elif a.command=='optimize':optimize(a.trials)
    elif a.command=='neural':neural_cv(a.cv,seeds,a.models.split(',') if a.models else None)
    elif a.command=='nested':nested_cv(a.cv,seeds)

if __name__=='__main__':main()
