"""Continue student on provisional direct and derived SI datasets, with grouped holdout."""
import argparse, json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from competition_data import require_safe_output, read_competition_csv
from src.validation import training_data, sha
from src.gpu_training import fingerprints, network, predict, BITS

def score(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    return {'mse':float(np.mean((y-p)**2)), 'mae':float(np.mean(abs(y-p))), 'log_mae':float(np.mean(abs(np.log1p(y)-np.log1p(p))))}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',required=True); ap.add_argument('--device',choices=['auto','cuda','cpu'],default='auto'); ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--checkpoint',default='results/distill_top1_20261108/student.pt'); ap.add_argument('--teacher',default='results/distill_top1_20261108/train_predictions.csv')
    a=ap.parse_args(); device=('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device
    if device=='cuda' and not torch.cuda.is_available(): ap.error('CUDA unavailable')
    out=require_safe_output(a.out); out.mkdir(parents=True,exist_ok=False)
    inputs=[('pubchem','data/external/processed/pubchem_vaccinia_20261108/si_comparable_candidates.csv'),('chembl_pair','data/external/processed/chembl_quarantine.csv'),('chembl_direct','data/external/processed/vacv_all_20261108/direct_si_candidates.csv')]
    parts=[]
    for name,path in inputs:
        data=pd.read_csv(path); data['source_dataset']=name; data['source_row']=np.arange(len(data)); data['source_file']=path
        data=data[['SMILES','SI VACV','source_dataset','source_row','source_file']].copy(); data['SI VACV']=pd.to_numeric(data['SI VACV'],errors='coerce'); parts.append(data)
    data=pd.concat(parts,ignore_index=True); data=data[np.isfinite(data['SI VACV']) & (data['SI VACV']>=0)].reset_index(drop=True)
    x,groups=fingerprints(data.SMILES); data.SMILES=groups
    # Avoid multiplying a duplicated identical measurement when PubChem mirrors ChEMBL.
    data['_key']=data.SMILES+'|'+data['SI VACV'].map(lambda z: format(z,'.10g'))
    keep=~data.duplicated('_key'); data=data.loc[keep].drop(columns='_key').reset_index(drop=True); x=x[keep]; groups=groups[keep]
    official=training_data(); ox,og=fingerprints(official.SMILES); _,tg=fingerprints(read_competition_csv('test.csv','test').SMILES)
    keep=~np.isin(groups,tg); data=data.loc[keep].reset_index(drop=True); x=x[keep]; groups=groups[keep]
    labels=data['SI VACV'].to_numpy(float)
    eligible=np.flatnonzero(~np.isin(groups,og))
    if len(set(groups[eligible]))<5: raise ValueError('Not enough independent molecules')
    localtr,localva=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=a.seed).split(x[eligible],groups=groups[eligible]))
    va=eligible[localva]; tr=np.flatnonzero(~np.isin(groups,groups[va])); data['evaluation_split']='train'; data.loc[va,'evaluation_split']='validation'; data.to_csv(out/'training_data.csv',index=False)
    teacher=pd.read_csv(a.teacher)
    if len(teacher)!=len(official) or not np.array_equal(teacher.SMILES,official.SMILES): raise ValueError('Teacher does not match train')
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if ck.get('bits')!=BITS or ck.get('train_sha256')!=sha('train.csv'): raise ValueError('Unexpected checkpoint')
    def make_net():
        net=network(); net.load_state_dict(ck['state_dict']); return net.to(device)
    def run(net,indices,stage_callback):
        torch.manual_seed(a.seed); opt=torch.optim.AdamW(net.parameters(),lr=.0001,weight_decay=.01)
        ex=torch.as_tensor(x[indices],device=device); ey=torch.as_tensor(np.log1p(labels[indices]).astype(np.float32),device=device).view(-1,1)
        tx=torch.as_tensor(ox,device=device); ty=torch.as_tensor(np.log1p(teacher.teacher.to_numpy(np.float32)),device=device).view(-1,1)
        for stage in (1,2):
            for _ in range(100):
                net.train(); opt.zero_grad(set_to_none=True)
                loss=torch.nn.functional.mse_loss(net(ex),ey)+.5*torch.nn.functional.mse_loss(net(tx),ty)
                loss.backward(); opt.step()
            stage_callback(net,stage)
    report={'device':device,'seed':a.seed,'input_files':{p:sha(p) for _,p in inputs},'initial_checkpoint_sha256':sha(a.checkpoint),'rows_total':len(data),'unique_molecules':len(set(groups)),'evaluation_train_rows':len(tr),'validation_rows':len(va),'validation_unique_molecules':len(set(groups[va])),'sources':data.source_dataset.value_counts().to_dict(),'warning':'External SI values are provisional, may vary by assay and include duplicate experimental measurements; model not eligible for documented contest protocol. Teacher was fitted on full official train. Only the external holdout excluding official train offers an independent molecular split.','evaluation':{}}
    net=make_net()
    def evaluate(net,name):
        pred=predict(net,x[va],device); report['evaluation'][name]={'holdout_vs_provisional_si':score(labels[va],pred),'official_train_vs_teacher':score(teacher.teacher,predict(net,ox,device))}
        pd.DataFrame({'SMILES':groups[va],'observed_provisional_si':labels[va],'predicted':pred}).to_csv(out/f'{name}_validation.csv',index=False)
    evaluate(net,'baseline')
    run(net,tr,lambda net,stage:evaluate(net,f'epoch_{stage*100}'))
    full=make_net()
    def save(net,stage):
        torch.save({'state_dict':{k:v.detach().cpu() for k,v in net.state_dict().items()},'bits':BITS,'train_sha256':sha('train.csv'),'epoch':stage*100,'external_training_sha256':sha(out/'training_data.csv')},out/f'full_epoch_{stage*100}.pt')
        print(f'Full-data training: {stage*100} epochs complete',flush=True)
    run(full,np.arange(len(data)),save)
    (out/'metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
