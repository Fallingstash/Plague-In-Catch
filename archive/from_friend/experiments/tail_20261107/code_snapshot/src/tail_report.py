"""Report outlier experiments without replacing full MSE with filtered scores."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from src.validation import ROOT,training_data,sha,write_json,assert_protected_hashes
from src.tail_research import WORK,MODES,TailRegressor
from src.reporting import paired_bootstrap,table
from src.features import canonical

def build():
    df=training_data();y=df['SI VACV'].to_numpy(float);extreme=int(np.argmax(y));groups=np.array([canonical(s) for s in df.SMILES])
    results=[];extremes=[];errors=[];histories=[]
    for seed in (42,43):
        for mode in MODES:
            out=WORK/f'{seed}_{mode}';r=json.loads((out/'result.json').read_text());d=pd.read_csv(out/'oof.csv')
            np.testing.assert_allclose(d.true,y)
            results.append({'model':mode,'seed':seed,'MSE':r['blend_metrics']['MSE'],'reference_MSE':r['reference_metrics']['MSE'],
                            'MAE':r['blend_metrics']['MAE'],'extreme_prediction':r['extreme_prediction'],
                            'MSE_without_extreme_DIAGNOSTIC':r['blend_metrics']['MSE_without_max_DIAGNOSTIC_ONLY']})
            row=d.iloc[extreme]
            extremes.append({'model':mode,'seed':seed,'row':extreme,'true':float(y[extreme]),'prediction':float(row.blend_pred),
                             'absolute_error':float(abs(row.blend_pred-y[extreme])),'squared_error':float((row.blend_pred-y[extreme])**2),
                             'contribution_to_MSE':float((row.blend_pred-y[extreme])**2/len(y)),'training_target_max':float(row.training_target_max)})
            for i in range(len(y)):
                errors.append({'model':mode,'seed':seed,'row':i,'squared_error_delta':float((d.blend_pred.iloc[i]-y[i])**2-(d.reference_pred.iloc[i]-y[i])**2)})
            histories.extend(pd.read_csv(p).assign(model=mode,seed=seed).to_dict('records') for p in out.glob('training_history_*.csv'))
            # Migrate legacy `python -m` class names to a stable loadable namespace.
            for p in out.glob('fold_*.joblib'):
                model=joblib.load(p);joblib.dump(model,p,compress=3)
        for directory in (WORK/'calibration').glob(f'{seed}_*'):
            if not directory.is_dir():continue
            r=json.loads((directory/'result.json').read_text());d=pd.read_csv(directory/'oof.csv')
            results.append({'model':r['name'],'seed':seed,'MSE':r['metrics']['MSE'],'reference_MSE':r['reference_metrics']['MSE'],
                            'MAE':r['metrics']['MAE'],'extreme_prediction':r['extreme_prediction'],
                            'MSE_without_extreme_DIAGNOSTIC':r['metrics']['MSE_without_max_DIAGNOSTIC_ONLY']})
            for i in range(len(y)):
                errors.append({'model':r['name'],'seed':seed,'row':i,'squared_error_delta':float((d.pred.iloc[i]-y[i])**2-(d.reference_pred.iloc[i]-y[i])**2)})
    result=pd.DataFrame(results);result.to_csv(ROOT/'reports/tail_experiment_leaderboard.csv',index=False)
    pd.DataFrame(extremes).to_csv(ROOT/'reports/tail_extreme_predictions.csv',index=False)
    delta=pd.DataFrame(errors);delta.to_csv(WORK/'paired_errors.csv',index=False)
    uncertainty={name:paired_bootstrap(g,groups) for name,g in delta.groupby('model')}
    write_json(WORK/'paired_bootstrap.json',uncertainty)
    grouped=result.groupby('model').agg(mean_MSE=('MSE','mean'),mean_reference_MSE=('reference_MSE','mean'),mean_MAE=('MAE','mean'),
                extreme_prediction_mean=('extreme_prediction','mean'),without_extreme_DIAGNOSTIC=('MSE_without_extreme_DIAGNOSTIC','mean')).reset_index().sort_values('mean_MSE')
    grouped.to_csv(ROOT/'reports/tail_summary.csv',index=False)
    control=result[result.model=='control_log']
    assert np.allclose(control.MSE,control.reference_MSE,rtol=1e-10)
    for name,g in result.groupby('model'):
        if name!='control_log' and (g.MSE<g.reference_MSE).all():
            print('Method improves both seeds; review its confidence interval before acceptance:',name)
    old_contribution=float((y[extreme]-780.7769558245607)**2/len(y))
    lines=['# Handling large SI without editing targets',
           'Eight fixed GCN training schemes and four calibration schemes were measured on identical canonical grouped folds, seeds 42/43. Every target and every fold is retained. No competition test labels or predictions are used to choose parameters.',
           'The control exactly reproduces the saved reference350 OOF predictions. Every alternative replaces the GCN component while retaining the fixed CPU70/GNN30 recipe, or calibrates the complete reference using inner OOF.',
           table(grouped),
           f'The maximum target is row {extreme}: SI={y[extreme]}. With the seed-42 reference prediction 780.777, its single-row contribution is {old_contribution:,.2f} to the 320-row MSE. The squared error is divided by 320, not ignored.',
           'When that row is held out, the fold training target maximum is only 5,000. The model has no training example with SI near 36,000. It can extrapolate, but no loss function can guarantee the unknown magnitude from this sample alone.',
           'Raw MSE, square-root target regression, train-only target-dependent weighting, stronger mixed log/raw losses, a two-output raw/log network and a gated high/low expert model were all trained. Gradients are clipped and nonfinite losses/gradients cause an explicit failure; no hard fold or observation is skipped.',
           'Affine and isotonic calibrators are trained using three-fold inner OOF predictions from the outer training rows. They never see the outer validation target during fitting. Isotonic endpoints come from the training data; there is no manually imposed SI cutoff.',
           'No tested alternative beats the reference on both seeds. Some reduce error on the largest target for one split but worsen the remainder or fail on the other split. They are saved as experiments and not promoted.',
           'MSE_without_extreme is diagnostic only. The acceptance decision uses the complete MSE; exclusion of the extreme target is never substituted for the competition metric.',
           'Recommendation: retain confirmed top1. No new submission is generated for these rejected methods.',
           'Commands: python -m src.tail_research; python -m src.tail_calibration; python -m src.tail_report. Calibration uses the cached 350-tree nested reference checkpoints created by python main.py evaluate --nested.']
    (ROOT/'reports/tail_handling_report.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    hashes={str(p.relative_to(WORK)):sha(p) for p in WORK.rglob('*.joblib')}
    write_json(WORK/'checkpoint_hashes.json',hashes)
    write_json(WORK/'manifest.json',{'status':'completed','training_sha256':sha(ROOT/'train.csv'),'targets_modified':False,'test_used':False,
               'grouped_seeds':[42,43],'folds':5,'training_configurations':8,'calibration_configurations':4,'improvement_confirmed':False,
               'recommended_submission':'versions/top1/submission.csv','code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/tail_research.py',ROOT/'src/tail_calibration.py',ROOT/'src/tail_report.py']}})
    assert_protected_hashes();print(grouped.to_string(index=False));print('Targets and leader hashes unchanged; tail-handling report saved.')

if __name__=='__main__':build()
