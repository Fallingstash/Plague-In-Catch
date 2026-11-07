import argparse
import json
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
from src.validation import ROOT,sha,write_json
from research_data_collection.validation import validate_record

def export_existing(source,output):
    source=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(source,keep_default_na=False);records=[];validated=[]
    for _,r in df.iterrows():
        result=validate_record(r)
        row={'SMILES':r.SMILES,'IC50':r.get('IC50_uM_VACV',''),'CC50':r.get('CC50_uM',''),'SI':r.get('SI VACV',''),
             'DOI':r.get('DOI',''),'source_url':r.get('source_url',r.get('Source_URL','')),'strain':r.get('strain',''),
             'culture':r.get('culture',r.get('Culture','')),'units':'uM','ic50_relation':r.get('ic50_relation',''),
             'cc50_relation':r.get('cc50_relation',''),'si_provenance':'derived_from_assay_pair_PENDING_PRIMARY_SOURCE_REVIEW',
             'review_status':r.get('review_status','pending'),'scientifically_validated':result['scientifically_validated'],
             'training_allowed':False,'raw_record_json':json.dumps(r.to_dict(),ensure_ascii=False)}
        records.append(row)
        if result['scientifically_validated']:validated.append(row)
    pd.DataFrame(records).to_csv(out/'research_candidates.csv',index=False)
    pd.DataFrame(validated,columns=pd.DataFrame(records).columns).to_csv(out/'validated_measurements.csv',index=False)
    caches=[]
    for path in (ROOT/'data/external/raw').glob('*/manifest.json'):
        original=json.loads(path.read_text(encoding='utf-8'))
        caches.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'metadata':{k:v for k,v in original.items() if k!='requests'}})
    write_json(out/'manifest.json',{'source_file':str(source),'source_sha256':sha(source),'cached_download_manifests':caches,'exported_utc':datetime.now(timezone.utc).isoformat(),
                'candidate_count':len(records),'validated_count':len(validated),'training_allowed':False,
                'note':'Real existing ChEMBL records; candidate pairs are not validated labels. No competition model imports this module.'})
    print(f'Exported {len(records)} research-only candidates; {len(validated)} verified records. Training use forbidden.')

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
    c=sub.add_parser('collect');c.add_argument('--out',default='research_data_collection/raw/chembl');c.add_argument('--max-pages',type=int,default=30)
    c=sub.add_parser('export');c.add_argument('--source',default='data/external/processed/chembl_quarantine.csv');c.add_argument('--out',default='research_data_collection/exports')
    a=p.parse_args()
    if a.cmd=='export':export_existing(a.source,a.out)
    else:
        from research_data_collection.chembl import download
        download(a.out,a.max_pages)

if __name__=='__main__':main()
