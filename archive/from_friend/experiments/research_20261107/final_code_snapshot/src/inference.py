"""Single inference path for legacy and research checkpoints."""
import numpy as np
import joblib
from src.features import features
from src.validation import training_data

def predict_bundle(bundle, frame):
    from src.features import canonical
    if 'SMILES' not in frame or not len(frame):raise ValueError('A nonempty SMILES table is required')
    for sm in frame.SMILES:canonical(sm)
    kind=bundle.get('model_type')
    if kind=='reference_graph_blend':
        import vacv_cpu_improved as cpu
        from gcn_predict import predict
        w=bundle['graph_weight']
        p=(1-w)*cpu.bundle_predict(bundle['base_bundle'],frame)+w*predict(bundle['graph_bundle'],frame.SMILES)
    elif kind=='research_regressor':
        p=bundle['model'].predict(features(frame.SMILES))
    elif kind in ('reference_fit','neural_regressor'):
        p=bundle['model'].predict(frame)
    elif kind=='fixed_research_blend':
        p=(1-bundle['weight'])*predict_bundle(bundle['base'],frame)+bundle['weight']*predict_bundle(bundle['new'],frame)
    elif 'nets' in bundle:
        import vacv_cpu_improved as cpu
        p=cpu.bundle_predict(bundle,frame)
    else:
        from vacv_neural import load_model_predictions
        p=load_model_predictions(bundle,frame)
    p=np.asarray(p,float)
    if len(p)!=len(frame) or not np.isfinite(p).all() or (p<0).any():
        raise ValueError('Invalid model predictions')
    return p

def predict_path(path,frame): return predict_bundle(joblib.load(path),frame)

def similar(smiles,top=10):
    import vacv_cpu_improved as cpu
    df=training_data();ref=cpu.chem_features(df.SMILES.tolist());query=cpu.chem_features([smiles])
    similarity=cpu.combined_similarity(ref,query,None,None,0)[0]
    ids=np.argsort(-similarity)[:max(1,min(top,len(df)))]
    from competition_data import read_competition_csv
    from src.validation import ROOT
    source=read_competition_csv(ROOT/'train.csv','train')
    return [{'row':int(i),'SMILES':df.SMILES.iloc[i],'Title':str(source.iloc[i].get('Title','')),'SI_observed':float(df['SI VACV'].iloc[i]),'similarity_Tanimoto':float(similarity[i])} for i in ids]
