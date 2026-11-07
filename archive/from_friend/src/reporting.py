"""Evidence-based reports. Never tunes a model using competition test data."""
import json
import platform
import os
from pathlib import Path
import numpy as np
import pandas as pd
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'experiments/research_20261107/matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.validation import ROOT,training_data,sha,write_json,metrics,assert_protected_hashes
from src.features import features
from src.research import WORK,update_leaderboard,save_run,source_digest

def table(df):
    if df.empty:return 'No completed results.'
    cols=list(df.columns);lines=['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
    for _,r in df.iterrows():
        vals=[]
        for v in r:
            if isinstance(v,(float,np.floating)):vals.append(f'{v:,.2f}' if np.isfinite(v) else '')
            else:vals.append(str(v))
        lines.append('| '+' | '.join(vals)+' |')
    return '\n'.join(lines)

def paired_bootstrap(detail,groups):
    # Average repeated split errors per original row, then bootstrap molecular groups.
    means=detail.groupby('row')['squared_error_delta'].mean().reindex(np.arange(len(groups))).to_numpy()
    unique=np.unique(groups);sums=np.array([means[groups==g].sum() for g in unique]);counts=np.array([(groups==g).sum() for g in unique])
    rng=np.random.default_rng(2026);samples=rng.integers(0,len(unique),size=(2000,len(unique)))
    boot=sums[samples].sum(1)/counts[samples].sum(1)
    return {'mean_squared_error_delta':float(means.mean()),'group_bootstrap_95_low':float(np.quantile(boot,.025)),
            'group_bootstrap_95_high':float(np.quantile(boot,.975)),
            'note':'Diagnostic paired group bootstrap; repeated seeds are averaged per row, not treated as independent objects. Extreme-observation uncertainty remains large.'}

def build_reports():
    assert_protected_hashes();df=training_data();y=df['SI VACV'].to_numpy(float);raw=features(df.SMILES,WORK/'feature_cache')
    # Expose component scores and predeclared fixed 70/30 neural replacements.
    fixed_results=[];deltas=[]
    for ref_path in (WORK/'runs').glob('*_reference*/oof.csv'):
        r=json.loads((ref_path.parent/'result.json').read_text());cv,seed=r['cv'],r['seed'];ref=pd.read_csv(ref_path)
        for component,col in [('cpu_component350' if r['model']=='reference350' else 'cpu_component','cpu_pred'),('gcn_single','gcn_pred')]:
            target=WORK/'runs'/f'{cv}_{seed}_{component}'/'result.json'
            if not target.exists():save_run(component,cv,seed,{'source_checkpoint':str(ref_path.parent),'family':'reference_component'},ref[col].to_numpy(),y,ref.fold.to_numpy(),0)
        for mode in ('gcn_mean3','bond_gcn','hybrid_gcn','residual_mlp','dmpnn'):
            source=WORK/'runs'/f'{cv}_{seed}_{mode}'/'oof.csv'
            if not source.exists():continue
            pp=pd.read_csv(source).sort_values('row');p=.7*ref.cpu_pred.to_numpy()+.3*pp.pred.to_numpy()
            score=metrics(y,p);score.update(model=f'CPU70+{mode}30',reference_recipe=r['model'],cv=cv,seed=seed,reference_mse=r['metrics']['MSE'],relative_mse_change=(score['MSE']/r['metrics']['MSE']-1))
            fixed_results.append(score)
            if cv=='canonical' and r['model']=='reference350':
                for i in range(len(y)):deltas.append({'model':mode,'seed':seed,'row':i,'squared_error_delta':float((p[i]-y[i])**2-(ref.pred.iloc[i]-y[i])**2)})
    fixed=pd.DataFrame(fixed_results);fixed.to_csv(ROOT/'reports/fixed_neural_blend_comparison.csv',index=False)
    if deltas:
        delta=pd.DataFrame(deltas);delta.to_csv(ROOT/'reports/fixed_neural_blend_paired_errors.csv',index=False)
        write_json(ROOT/'reports/neural_group_bootstrap.json',{mode:paired_bootstrap(g,raw['canonical']) for mode,g in delta.groupby('model')})
    update_leaderboard();board=pd.read_csv(ROOT/'reports/experiment_leaderboard.csv')
    canonical=board[(board.cv=='canonical')&board.oof_mse.notna()]
    grouped=canonical.groupby('model').agg(mean_oof_mse=('oof_mse','mean'),mean_mae=('oof_mae','mean'),split_seeds=('seed','count')).reset_index().sort_values('mean_oof_mse')
    grouped.to_csv(ROOT/'reports/research_model_summary.csv',index=False)
    scaffold=board[(board.cv=='scaffold')&board.oof_mse.notna()][['model','seed','oof_mse','oof_mae']].sort_values('oof_mse')
    nested_results=[];nested_errors=[]
    for path in (WORK/'nested').glob('*/result.json'):
        r=json.loads(path.read_text());nested_results.append({'cv':r['cv'],'seed':r['seed'],'tree_estimators':r.get('tree_estimators',250),'reference_mse':r['reference_metrics']['MSE'],'nested_mse':r['metrics']['MSE'],'relative_change':r['metrics']['MSE']/r['reference_metrics']['MSE']-1})
        z=pd.read_csv(path.parent/'oof.csv')
        for _,row in z.iterrows():nested_errors.append({'tree_estimators':r.get('tree_estimators',250),'row':int(row.row),'squared_error_delta':float((row.pred-row.true)**2-(row.reference_pred-row.true)**2)})
    nested=pd.DataFrame(nested_results);nested.to_csv(ROOT/'reports/nested_comparison.csv',index=False)
    if nested_errors:write_json(ROOT/'reports/nested_group_bootstrap.json',{str(n):paired_bootstrap(g,raw['canonical']) for n,g in pd.DataFrame(nested_errors).groupby('tree_estimators')})
    # Correlations excluding nothing: retain raw and absolute error diagnostics.
    matrix={}
    for _,rr in canonical[canonical.seed==42].iterrows():
        path=WORK/'runs'/str(rr.id)/'oof.csv';z=pd.read_csv(path).sort_values('row');matrix[rr.model]=z.pred.to_numpy()-y
    pd.DataFrame(matrix).corr().to_csv(ROOT/'reports/oof_error_correlations.csv')
    pd.DataFrame(matrix).abs().corr().to_csv(ROOT/'reports/oof_absolute_error_correlations.csv')
    audit=pd.read_csv(ROOT/'reports/molecule_error_audit.csv');audit['similarity_bin']=pd.cut(audit.max_train_similarity,[0,.3,.5,.7,1.000001],include_lowest=True)
    audit.groupby(['seed','similarity_bin'],observed=True).agg(rows=('row','size'),cpu_mse=('cpu_squared_error','mean'),gcn_mse=('gcn_squared_error','mean'),blend_mse=('blend_squared_error','mean')).to_csv(ROOT/'reports/error_by_similarity.csv')
    grouped_targets=pd.DataFrame({'canonical':raw['canonical'],'SI':y}).groupby('canonical').agg(rows=('SI','size'),min_SI=('SI','min'),max_SI=('SI','max')).reset_index()
    conflicts=grouped_targets[(grouped_targets.rows>1)&(grouped_targets.min_SI!=grouped_targets.max_SI)]
    conflicts.to_csv(ROOT/'reports/canonical_target_conflicts.csv',index=False)
    allowed=WORK/'official_training_used.csv';df.to_csv(allowed,index=False)
    details=json.loads((ROOT/'reports/data_audit_details.json').read_text())
    lines=['# Reproducible train-only research',
           f'Official train: {len(df)} rows, {len(np.unique(raw["canonical"]))} canonical groups, {int((y==0).sum())} zeros; SHA256 `{sha(ROOT/"train.csv")}`.',
           f'Targets remain unchanged. {len(conflicts)} canonical groups contain inconsistent repeated SI measurements; all rows are retained and assigned as groups.',
           'No Git repository was present. Initial files were snapshotted and protected top1/top2 hashes have remained unchanged.',
           'The official SHA256-pinned loader rejects external or test data as training input. Feature generation takes only SMILES; forbidden concentrations are excluded before parsing.',
           'Repeated seeds are scored separately. The headline mean is the mean of seed MSE values, never 640 independent molecules. All folds and all 320 targets are included.',
           'The largest target contributes about 95% of baseline OOF SSE. Removing it is diagnostic only; it is never removed from model selection.',
           'Nested CV uses 5 outer grouped folds and 3 inner grouped folds. Family/regularization/static weights/similarity gates/regularized stacking weights are selected on inner OOF. Full-data Optuna results are not fed into this outer assessment.',
           'The legacy 250-tree baseline is retained for reproduction. New reference350 runs and separate nested tree350 runs use 350 trees consistently with final training. Old and corrected protocols are never pooled as one algorithm.',
           'The historical baseline was previously developed on this same train; no repeated CV can turn it into a wholly untouched independent dataset.',
           'The old test-max threshold sweep is retained as historical data and excluded from all new selection. No arbitrary SI ceiling is fitted or chosen from competition test predictions.',
           'GCN graph depth 2 also introduced a residual connection, so the previous comparison was not a pure depth-only ablation. Its MAE improved but MSE worsened.',
           'Compact D-MPNN, bond-aware GCN, mixed log/raw GCN loss, three-initialization GCN mean and residual MLP were actually trained on both canonical seeds. Scores are exploratory unless nested-assessed.',
           'CatBoost/Optuna/pytest were installed. Optuna uses SQLite, a finite 8-trial run, median pruning, three grouped tuning folds and a 180-second timeout. Its tuning score is not reported as independent validation.',
           'Scientific export contains 59 real ChEMBL candidate pairs, 0 primary-source-validated pairs, and training_allowed=False. It cannot be used as competition training data.',
           'Historical second-best 2-layer checkpoint and submission were recovered exactly from fixed architecture/seed and recorded per-row predictions. Its original CSV SHA256 was reproduced. The existing top2 folder is an older preserved third-best; accurate ranking is in versions/ranking.json.',
           'All checked submission/model pairs reproduce to floating point tolerance. Main CLI now defaults to top1 rather than the stale 5% reference bundle.',
           'Optuna-selected settings are not asserted to improve Kaggle. No Kaggle upload was performed.',
           'Primary implementation references: [KernelRidge](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html), [CatBoost parameters](https://catboost.ai/docs/en/references/training-parameters/common), [Optuna persistent studies](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html).']
    (ROOT/'reports/research_audit.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    comparison=['# Measured model comparison','Canonical 5-fold grouped CV, seeds 42/43. MSE is on original SI. These are standalone models except reference, which is the fixed CPU70/GCN30 blend.',table(grouped),
                '## Fixed 70/30 architecture replacements',table(fixed[['model','reference_recipe','cv','seed','MSE','reference_mse','relative_mse_change']]),
                '## Nested selection: independent outer-fold scoring',table(nested),
                '## Scaffold CV',table(scaffold),
                'Screening winners are not independent estimates after selecting from this table. See paired error CSVs and bootstrap uncertainty. Test labels were never used.']
    (ROOT/'reports/model_comparison.md').write_text('\n\n'.join(comparison)+'\n',encoding='utf-8')
    recommendation=['# Final recommendation',
                    'Keep the confirmed top1, CPU70% + one-layer GCN30%, Public MSE 886520.98398. No new upload is justified by this research batch.',
                    'Recommended submission: versions/top1/submission.csv. Prediction checkpoint: versions/top1/blend_model.joblib.',
                    'The nested search worsened mean outer MSE compared with the fixed reference on both seeds. Some fixed neural replacements show small average changes, but no sufficiently robust, independent improvement supports replacing the champion.',
                    table(nested),
                    'The recovered historical two-layer model has reported Public MSE 898313.50898. It is retained in versions/candidates/historical_two_layer_70_30 and is the actual second-best reported result. The older immutable top2 folder has score 936601.51958; versions/ranking.json records this distinction.',
                    'Legacy 35% GCN remains unconfirmed and is archived with a complete, validated checkpoint. It is not recommended as a new improved model by this batch.',
                    'No public score has been inferred from local MSE. No test-derived max-prediction constraint is used for the new decision.',
                    'Next useful work should require a predeclared validation improvement across additional molecular groups and decompose the extreme-row contribution. Do not spend uploads on the old monotonic percentage sweep.',
                    'Commands: python main.py predict --smiles "CCO"; python main.py similar --smiles "CCO" --top 10; python main.py batch --input molecules.csv --output predictions.csv; python main.py validate-submission --file versions/top1/submission.csv; python main.py evaluate --cv canonical; python main.py evaluate --cv scaffold --seeds 42; python main.py optimize --trials 8.']
    (ROOT/'reports/final_recommendation.md').write_text('\n\n'.join(recommendation)+'\n',encoding='utf-8')
    # Preserve the previous CSV and append recovered history without inventing scores.
    history=ROOT/'reports/submission_leaderboard.csv';hist=pd.read_csv(history)
    backup=WORK/'submission_leaderboard_before_research.csv'
    if not backup.exists():backup.write_bytes(history.read_bytes())
    if not (hist['rank']=='historical_second_best').any():
        hist=pd.concat([hist,pd.DataFrame([{'rank':'historical_second_best','blend':'CPU70 + two-layer GCN30 restored','kaggle_public_mse':898313.50898,'local_oof_mse_mean':None,'status':'exact historical CSV and model restored; not newly uploaded'}])],ignore_index=True)
    hist.to_csv(history,index=False)
    # Publication-style diagnostic; no test data enter this figure.
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].hist(np.log1p(y),bins=25,color='#4169a1');axes[0].set(xlabel='log(1 + SI VACV)',ylabel='Training molecules',title='Official target distribution')
    subset_summary=grouped[grouped.model.isin(['reference','gcn_single','gcn_mean3','hybrid_gcn','dmpnn','krr_mix_0.03','catboost_d4_log'])]
    axes[1].barh(subset_summary.model,subset_summary.mean_oof_mse/1e6,color='#4169a1');axes[1].set(xlabel='Mean OOF MSE (millions)',title='Both canonical seeds; all rows')
    fig.tight_layout();fig.savefig(ROOT/'reports/research_comparison.png',dpi=160);plt.close(fig)
    manifest={'status':'completed_bounded_research','train_sha256':sha(ROOT/'train.csv'),'allowed_training_export_sha256':sha(allowed),'code_sha256':source_digest(),'platform':platform.platform(),'recommended_submission':'versions/top1/submission.csv',
              'recommended_submission_sha256':sha(ROOT/'versions/top1/submission.csv'),'top1_changed':False,'top2_changed':False,'selection_uses_test':False,
              'canonical_screening_runs':int(len(canonical)),'nested_runs':len(nested_results),'scaffold_runs':int(len(scaffold)),'improvement_confirmed':False}
    test_report=WORK/'pytest_results.xml'
    if test_report.exists():
        import xml.etree.ElementTree as ET
        suites=ET.parse(test_report).getroot()
        suite=list(suites)[0] if suites.tag=='testsuites' else suites
        manifest['pytest']={k:suite.attrib.get(k) for k in ('tests','failures','errors','skipped')}
        manifest['pytest_report_sha256']=sha(test_report)
    write_json(WORK/'manifest.json',manifest);assert_protected_hashes()
    print('Reports saved. Recommendation: keep confirmed top1; no unsupported new Kaggle upload.',flush=True)

if __name__=='__main__':build_reports()
