"""Collect and export VACV ChEMBL activities, preserving source metadata.

This is a research dataset, not evidence that each endpoint is an SI label.
"""
import argparse, csv, hashlib, json, math, time, urllib.parse, urllib.request
from collections import Counter
from pathlib import Path
from rdkit import Chem, RDLogger
import pandas as pd
from competition_data import read_competition_csv, require_safe_output

BASE='https://www.ebi.ac.uk/chembl/api/data/'

def fetch(url):
    for attempt in range(4):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'VACV-data-research/1.0','Accept':'application/json'})
            with urllib.request.urlopen(req,timeout=60) as response: return json.load(response)
        except Exception:
            if attempt==3: raise
            time.sleep(2**attempt)

def download(cache, out):
    cache=Path(cache); out=require_safe_output(out); out.mkdir(parents=True,exist_ok=True)
    source=json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
    if source.get('status')!='complete': raise ValueError('Incomplete assay cache')
    assays={}
    for entry in source['requests']:
        if entry['kind'] != 'assays': continue
        payload=json.loads((cache/entry['file']).read_text(encoding='utf-8'))
        for row in payload.get('assays',[]):
            if 'vaccinia' in str(row.get('assay_organism','')).casefold(): assays[row['assay_chembl_id']]=row
    if not assays: raise ValueError('No confirmed vaccinia assays found')
    (out/'assays.json').write_text(json.dumps(assays,ensure_ascii=False),encoding='utf-8')
    ids=sorted(assays); requests=[]
    for batch in range(0,len(ids),40):
        offset=0
        while True:
            params={'assay_chembl_id__in':','.join(ids[batch:batch+40]),'limit':1000,'offset':offset,'order_by':'activity_id'}
            url=BASE+'activity.json?'+urllib.parse.urlencode(params)
            file=out/f'activities_{batch//40:03d}_{offset:07d}.json'
            if file.exists(): payload=json.loads(file.read_text(encoding='utf-8'))
            else:
                payload=fetch(url)
                file.write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
                time.sleep(.25)
            requests.append({'url':url,'file':file.name,'count':len(payload.get('activities',[]))})
            print(f'Batch {batch//40+1}/{math.ceil(len(ids)/40)} offset {offset}: {requests[-1]["count"]}',flush=True)
            next_url=payload.get('page_meta',{}).get('next')
            if not next_url: break
            offset+=int(payload['page_meta']['limit'])
    (out/'manifest.json').write_text(json.dumps({'status':'complete','source':'ChEMBL','assay_cache':str(cache),'assay_count':len(ids),'requests':requests},indent=2),encoding='utf-8')

def canonical(value):
    mol=Chem.MolFromSmiles(str(value))
    return Chem.MolToSmiles(mol,isomericSmiles=True) if mol and mol.GetNumAtoms() else None

def export(raw, out, test):
    raw=Path(raw); out=require_safe_output(out); out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((raw/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='complete': raise ValueError('Incomplete raw download')
    assays=json.loads((raw/'assays.json').read_text(encoding='utf-8'))
    excluded=set(read_competition_csv(test,'test').SMILES.map(canonical)); excluded.discard(None)
    seen=set(); records=[]; counts=Counter(); excluded_count=0
    for request in manifest['requests']:
        data=json.loads((raw/request['file']).read_text(encoding='utf-8'))
        for act in data.get('activities',[]):
            aid=act.get('assay_chembl_id'); assay=assays.get(aid)
            if not assay: raise ValueError(f'Unexpected assay {aid}')
            key=act.get('activity_id')
            if key in seen: continue
            seen.add(key)
            smiles=canonical(act.get('canonical_smiles') or '')
            if not smiles: counts['missing_structure']+=1; continue
            if smiles in excluded: excluded_count+=1; continue
            typ=str(act.get('standard_type') or '')
            description=str(act.get('assay_description') or assay.get('description') or '')
            relation=act.get('standard_relation') or act.get('relation') or ''
            value=act.get('standard_value') or act.get('value') or ''
            direct_si=typ.casefold() in ('selectivity index','selectivity','si') and any(term in description.casefold() for term in ('selectivity','selective index'))
            try: numeric=float(value)
            except (ValueError,TypeError): numeric=float('nan')
            exact=direct_si and relation=='=' and math.isfinite(numeric) and numeric>=0 and not act.get('data_validity_comment')
            row={'SMILES':smiles,'SI VACV':numeric if exact else '', 'endpoint':typ,'value':value,'relation':relation,'units':act.get('standard_units'),'direct_si_candidate':direct_si,'exact_si_candidate':exact,'activity_id':key,'molecule_chembl_id':act.get('molecule_chembl_id'),'assay_chembl_id':aid,'document_chembl_id':act.get('document_chembl_id'),'assay_description':description,'assay_organism':assay.get('assay_organism'),'assay_cell_type':assay.get('assay_cell_type'),'data_validity_comment':act.get('data_validity_comment'),'potential_duplicate':act.get('potential_duplicate'),'source_url':f'https://www.ebi.ac.uk/chembl/assay_report_card/{aid}/','review_status':'pending_primary_source_review'}
            records.append(row); counts[typ]+=1
    df=pd.DataFrame(records)
    if df.empty: raise ValueError('No usable molecular structures')
    df.to_csv(out/'vacv_activities.csv',index=False)
    df[df.exact_si_candidate].to_csv(out/'direct_si_candidates.csv',index=False)
    report={'source':'ChEMBL', 'source_manifest_sha256':hashlib.sha256((raw/'manifest.json').read_bytes()).hexdigest(), 'rows':len(df),'unique_smiles':int(df.SMILES.nunique()),'excluded_test_records':excluded_count,'direct_exact_si_candidates':int(df.exact_si_candidate.sum()), 'endpoint_counts':dict(counts), 'note':'All rows require source and assay review. Non-SI endpoints and censored bounds are not exact SI targets.'}
    (out/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

def main():
    RDLogger.DisableLog('rdApp.error')
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='cmd',required=True)
    d=sub.add_parser('download'); d.add_argument('--assay-cache',default='data/external/raw/chembl_batch'); d.add_argument('--out',required=True)
    e=sub.add_parser('export'); e.add_argument('--raw',required=True); e.add_argument('--out',required=True); e.add_argument('--test',default='test.csv')
    a=p.parse_args()
    if a.cmd=='download': download(a.assay_cache,a.out)
    else: export(a.raw,a.out,a.test)
if __name__=='__main__': main()

