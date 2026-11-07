"""Conditional nested check of the promising pair-ridge family; inner-only weights."""
import json
import joblib
import numpy as np
import pandas as pd
from src.pair_research import WORK,PairRegressor
from src.research import WORK as BASE,prepare,folds_and_ids
from src.validation import ROOT,folds,metrics,write_json,assert_protected_hashes,sha

def run():
    df,raw,y=prepare();weights=(0.,.1,.3)
    write_json(WORK/'nested_protocol.json',{'family':'ridge_log','weights':list(weights),'inner_folds':3,'outer_folds':5,
               'seeds':[42,43],'note':'Family selected after exploratory screening. Weight selection uses inner OOF only; this is a conditional follow-up, not independent validation of the entire exploratory family search.'})
    for seed in (42,43):
        pairs,ids=folds_and_ids(y,raw,'canonical',seed);ref=pd.read_csv(BASE/'runs'/f'canonical_{seed}_reference350'/'oof.csv').sort_values('row')
        out=WORK/'nested'/str(seed);out.mkdir(parents=True,exist_ok=True);pred=np.zeros(len(y));decisions=[]
        for j,(tr,va) in enumerate(pairs,1):
            inner=folds(y[tr],raw['canonical'][tr],seed+100+j,3)
            base_inner=pd.read_csv(BASE/'nested'/f'canonical_{seed}_tree350'/f'inner_oof_{j}.csv');np.testing.assert_array_equal(base_inner.row,tr)
            p=np.zeros(len(tr))
            for k,(itr,iva) in enumerate(inner,1):
                checkpoint=out/f'outer_{j}_inner_{k}.joblib'
                if checkpoint.exists():model=joblib.load(checkpoint)
                else:
                    model=PairRegressor('ridge_log').fit(df.iloc[tr[itr]].reset_index(drop=True),y[tr[itr]],seed*100+j*10+k,tr[itr])
                    joblib.dump(model,checkpoint,compress=3)
                assert set(model.row_ids).isdisjoint(va) and set(model.row_ids).isdisjoint(tr[iva])
                p[iva]=model.predict(df.iloc[tr[iva]].reset_index(drop=True))
            b=base_inner.baseline_inner_oof.to_numpy();scores={str(w):float(np.mean((y[tr]-((1-w)*b+w*p))**2)) for w in weights}
            selected=min(weights,key=lambda w:scores[str(w)])
            outer_pair=joblib.load(WORK/f'{seed}_ridge_log'/f'fold_{j}.joblib')
            pair_pred=outer_pair.predict(df.iloc[va].reset_index(drop=True));pred[va]=(1-selected)*ref.pred.to_numpy()[va]+selected*pair_pred
            decisions.append({'fold':j,'selected_pair_weight':selected,'inner_MSE':scores})
            pd.DataFrame({'row':tr,'true':y[tr],'reference_inner_oof':b,'pair_inner_oof':p}).to_csv(out/f'inner_oof_{j}.csv',index=False)
            print(f'Pair nested seed {seed} outer fold {j}: selected weight {selected}',flush=True)
        pd.DataFrame({'row':np.arange(len(y)),'seed':seed,'fold':ids,'true':y,'pred':pred,'reference_pred':ref.pred}).to_csv(out/'oof.csv',index=False)
        write_json(out/'result.json',{'seed':seed,'metrics':metrics(y,pred),'reference_metrics':metrics(y,ref.pred),'selections':decisions,
                   'extreme_prediction':float(pred[np.argmax(y)]),'training_sha256':sha(ROOT/'train.csv'),
                   'note':'Conditional nested check of fixed pair-ridge family. Test not read.'})
    assert_protected_hashes()

if __name__=='__main__':run()
