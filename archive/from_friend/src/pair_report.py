"""Pairwise-learning results, original-molecule confidence and leakage audit."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from src.pair_research import WORK,MODES,PairRegressor,similarity
from src.validation import ROOT,training_data,sha,write_json,assert_protected_hashes
from src.features import canonical
from src.reporting import paired_bootstrap,table
from versions.test.training import transform
import vacv_cpu_improved as cpu

def build():
    df=training_data();y=df['SI VACV'].to_numpy(float);extreme=int(np.argmax(y));groups=np.array([canonical(s) for s in df.SMILES]);errors=[];anchors=[];nested=[]
    comparison=pd.read_csv(ROOT/'reports/pair_model_comparison.csv')
    assert len(comparison)==len(MODES)*2*3 and comparison.groupby(['mode','variant']).seed.nunique().eq(2).all()
    for seed in (42,43):
        for mode in MODES:
            out=WORK/f'{seed}_{mode}';data=pd.read_csv(out/'oof.csv').sort_values('row');np.testing.assert_allclose(data.true,y)
            for variant,column in [('standalone','pair_pred'),('add10','add10_pred'),('replace30','replace30_pred')]:
                for i in range(len(y)):
                    errors.append({'mode':mode,'variant':variant,'seed':seed,'row':i,'squared_error_delta':float((data[column].iloc[i]-y[i])**2-(data.reference_pred.iloc[i]-y[i])**2)})
        ridge=pd.read_csv(WORK/f'{seed}_ridge_log/oof.csv').sort_values('row');fold=int(ridge.fold.iloc[extreme])
        model=joblib.load(WORK/f'{seed}_ridge_log/fold_{fold}.joblib')
        query=df.iloc[[extreme]];raw=cpu.chem_features(query.SMILES.tolist());x=transform(raw,model.scaler);p,idx,_=model.predict_details(query)
        ids=idx[0];delta=model.model.predict(x[0]-model.x[ids]);values=np.expm1(np.clip(model.z[ids]+delta,0,14));s=similarity(raw,model.reference_raw)[0,ids];w=np.maximum(s,1e-6)**4;w/=w.sum()
        for a,d,v,sim,weight in zip(ids,delta,values,s,w):
            anchors.append({'seed':seed,'query_row':extreme,'anchor_row':int(model.row_ids[a]),'anchor_SI':float(model.y[a]),'similarity':float(sim),'predicted_delta_logSI':float(d),'reconstructed_SI':float(v),'anchor_weight':float(weight)})
        r=json.loads((WORK/'nested'/str(seed)/'result.json').read_text());nested.append({'seed':seed,'MSE':r['metrics']['MSE'],'reference_MSE':r['reference_metrics']['MSE'],'extreme_prediction':r['extreme_prediction'],'selected_weights':[s['selected_pair_weight'] for s in r['selections']]})
    detail=pd.DataFrame(errors);detail.to_csv(WORK/'paired_error_deltas.csv',index=False)
    bounds={mode+'_'+variant:paired_bootstrap(g,groups) for (mode,variant),g in detail.groupby(['mode','variant'])}
    write_json(WORK/'paired_group_bootstrap.json',bounds)
    pd.DataFrame(anchors).to_csv(ROOT/'reports/pair_extreme_anchors.csv',index=False)
    summary=comparison.groupby(['mode','variant']).agg(mean_MSE=('MSE','mean'),reference_MSE=('reference_MSE','mean'),mean_MAE=('MAE','mean'),without_max_DIAGNOSTIC=('MSE_without_max_DIAGNOSTIC','mean')).reset_index().sort_values('mean_MSE')
    summary.to_csv(ROOT/'reports/pair_summary.csv',index=False);pd.DataFrame(nested).to_csv(ROOT/'reports/pair_nested_comparison.csv',index=False)
    baseline=float(comparison.reference_MSE.mean());best=float(summary[(summary['mode']=='ridge_log')&(summary.variant=='add10')].mean_MSE.iloc[0]);nested_mean=float(np.mean([r['MSE'] for r in nested]))
    lines=['# Pairwise SI learning',
           'Inspired by the pairwise property-difference idea in [DeepDelta](https://pmc.ncbi.nlm.nih.gov/articles/PMC10605784/). This is a from-scratch fingerprint/descriptor implementation, not downloaded DeepDelta weights or a reproduction of its benchmarks.',
           'Official 320 training labels only. Pairs use 8 nearest structural neighbors plus 2 random partners per training molecule, both directions, excluding identical canonical structures. These are similarity pairs, not certified matched-molecular-pair fragment decompositions.',
           'Every pair endpoint and every inference anchor belongs to the current training fold. Descriptors are scaled on that fold. Validation and test targets are never used as anchors. The number of pairs is not the number of independent experimental observations.',
           'Seven configurations: anchor-only ablation; Ridge log/raw; pair Tanimoto-kernel log/raw; antisymmetric context-gated neural log/raw. Fixed comparisons: standalone, reference90/pair10, CPU70/pair30. Same canonical grouped 5 folds, seeds 42/43; every row included.',
           table(summary),
           f'Fixed reference90 + pair-ridge10 scored mean OOF MSE {best:,.2f} versus {baseline:,.2f} for reference, a {(1-best/baseline)*100:.2f}% decrease. The error excluding the maximum target increased; the small apparent benefit concentrates in the largest observation.',
           '## Conditional nested follow-up',table(pd.DataFrame(nested)),
           f'Within each outer training fold, a three-fold inner OOF check selected pair weight among 0, 0.1 and 0.3. Mean outer MSE was {nested_mean:,.2f}; the gain did not survive. Both extreme-target folds selected weight 0.',
           'The family was selected after exploratory screening, so this follow-up is conditional rather than an independent estimate of the entire seven-family search. No Kaggle test inference was used to choose a configuration.',
           'Raw pair neural networks and raw Ridge produced large errors on some folds; no difficult fold was removed. Log-delta Ridge was more stable than the larger pair network.',
           'Conclusion: preserve top1. The pair method provides a mildly different predictor, but this experiment does not justify spending a Kaggle submission. All checkpoints, pair endpoints, anchors and OOF predictions are saved.',
           'Detailed files: pair_model_comparison.csv, pair_summary.csv, pair_nested_comparison.csv and pair_extreme_anchors.csv in reports/. Confidence intervals bootstrap original canonical molecule groups; repeated seeds are averaged per original row.',
           'Commands: python -m src.pair_research; python -m src.pair_nested; python -m src.pair_report.']
    (ROOT/'reports/pair_learning_report.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    write_json(WORK/'checkpoint_hashes.json',{str(p.relative_to(WORK)):sha(p) for p in WORK.rglob('*.joblib')})
    write_json(WORK/'manifest.json',{'status':'completed','training_sha256':sha(ROOT/'train.csv'),'test_used':False,'targets_modified':False,'model_configurations':7,'split_seeds':[42,43],
               'folds':5,'fixed_pair10_mean_MSE':best,'reference_mean_MSE':baseline,'conditional_nested_mean_MSE':nested_mean,'improvement_confirmed':False,
               'recommended_submission':'versions/top1/submission.csv','source_sha256':{name:sha(ROOT/name) for name in ['src/pair_research.py','src/pair_nested.py','src/pair_report.py']}})
    assert_protected_hashes();print('Pair learning report saved. Fixed-weight screening improves slightly; conditional nested check rejects promotion.')

if __name__=='__main__':build()
