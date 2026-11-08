"""Command line interface. Evaluation/optimization never load competition test."""
import argparse
import json
import time
from pathlib import Path
import joblib
import pandas as pd
from competition_data import read_competition_csv
from src.validation import ROOT,training_data,protect_output,sha,write_json
from src.inference import predict_path,similar
from src.artifacts import validate_submission,make_submission

def main():
    p=argparse.ArgumentParser(description='Train-only VACV molecular regression')
    commands=p.add_subparsers(dest='command',required=True)
    for name in ('predict','similar','batch','submit','make-submission'):
        q=commands.add_parser(name);q.add_argument('--model',default=str(ROOT/'versions/top1/blend_model.joblib'))
        if name in ('predict','similar'):q.add_argument('--smiles',required=True)
        if name=='similar':q.add_argument('--top',type=int,default=10)
        if name=='batch':q.add_argument('--input',required=True);q.add_argument('--output',required=True)
        if name in ('submit','make-submission'):q.add_argument('--output',default='results/submission.csv')
    q=commands.add_parser('validate-submission');q.add_argument('--file',required=True);q.add_argument('--model');q.add_argument('--manifest')
    q=commands.add_parser('evaluate');q.add_argument('--cv',choices=['canonical','scaffold'],default='canonical');q.add_argument('--seeds',default='42,43');q.add_argument('--models',default='reference');q.add_argument('--nested',action='store_true')
    q=commands.add_parser('optimize');q.add_argument('--trials',type=int,default=8)
    q=commands.add_parser('research');q.add_argument('--phase',choices=['audit','screen','neural','nested','reports'],default='audit');q.add_argument('--seeds',default='42,43');q.add_argument('--cv',choices=['canonical','scaffold'],default='canonical')
    q=commands.add_parser('train');q.add_argument('--train',default=str(ROOT/'train.csv'));q.add_argument('--out',default='results/new_training');q.add_argument('--device',choices=['cpu'],default='cpu');q.add_argument('--config');q.add_argument('--seed',type=int,default=42);q.add_argument('--epochs',type=int,default=60)
    a=p.parse_args()
    if a.command=='predict':
        frame=pd.DataFrame({'SMILES':[a.smiles]});print(json.dumps({'SMILES':a.smiles,'predicted_SI_VACV':float(predict_path(a.model,frame)[0]),'nearest':similar(a.smiles,1)[0]},ensure_ascii=False,indent=2))
    elif a.command=='similar':print(json.dumps(similar(a.smiles,a.top),ensure_ascii=False,indent=2))
    elif a.command=='batch':
        out=protect_output(a.output);frame=read_competition_csv(a.input,'inference');frame['predicted_SI_VACV']=predict_path(a.model,frame);out.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(out,index=False);print(f'Saved {len(frame)} rows to {out}')
    elif a.command in ('submit','make-submission'):print(json.dumps(make_submission(a.model,a.output),indent=2))
    elif a.command=='validate-submission':print(json.dumps(validate_submission(a.file,a.manifest,a.model),indent=2))
    elif a.command=='evaluate':
        from src.research import matched_reference_cv,evaluate,nested_cv
        seeds=tuple(int(s) for s in a.seeds.split(','))
        if a.nested:nested_cv(a.cv,seeds)
        elif a.models=='reference':matched_reference_cv(a.cv,seeds)
        else:evaluate(a.cv,seeds,None if a.models=='all' else a.models.split(','))
    elif a.command=='optimize':
        from src.research import optimize
        if a.trials<1:p.error('--trials must be positive')
        optimize(a.trials)
    elif a.command=='research':
        from src.research import audit,evaluate,neural_cv,nested_cv
        seeds=tuple(int(s) for s in a.seeds.split(','))
        if a.phase=='audit':audit()
        elif a.phase=='screen':evaluate(a.cv,seeds)
        elif a.phase=='neural':neural_cv(a.cv,seeds)
        elif a.phase=='nested':nested_cv(a.cv,seeds)
        else:
            from src.reporting import build_reports
            build_reports()
    elif a.command=='train':
        from src.research import ReferenceFit,source_digest
        from src.models import MoleculeRegressor
        from src.features import features
        df=training_data(a.train);out=protect_output(a.out)
        if (out/'model_bundle.joblib').exists():raise ValueError('Model already exists; choose a fresh --out')
        begin=time.monotonic();config=json.loads(Path(a.config).read_text(encoding='utf-8')) if a.config else {'family':'reference'}
        if config.get('family')=='reference':
            model=ReferenceFit().fit(df,df['SI VACV'].to_numpy(float),a.seed,1,a.epochs);kind='reference_fit'
        else:
            model=MoleculeRegressor(config).fit(features(df.SMILES),df['SI VACV'].to_numpy(float),a.seed);kind='research_regressor'
        out.mkdir(parents=True,exist_ok=True);bundle={'model_type':kind,'model':model,'config':config,'training_sha256':sha(a.train)};joblib.dump(bundle,out/'model_bundle.joblib',compress=3)
        write_json(out/'manifest.json',{'status':'trained_not_submitted','training_sha256':sha(a.train),'code_sha256':source_digest(),'config':config,'seed':a.seed,'seconds':time.monotonic()-begin,'prediction_model_sha256':sha(out/'model_bundle.joblib')})
        print(f'Saved trained model to {out}/model_bundle.joblib; no test data used')

if __name__ == '__main__':
    main()
