"""Parse downloaded PubChem SI assays to reviewable, provenance-rich CSV."""
import argparse,json,math,re
from pathlib import Path
import pandas as pd
from rdkit import RDLogger
from competition_data import require_safe_output, read_competition_csv
from data_collection.vacv_activities import canonical

def main():
    RDLogger.DisableLog('rdApp.error')
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--raw',required=True); p.add_argument('--out',required=True); p.add_argument('--test',default='test.csv'); a=p.parse_args()
    raw=Path(a.raw); out=require_safe_output(a.out); out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((raw/'manifest.json').read_text(encoding='utf-8'))
    idx=pd.read_csv(raw/'assay_index.csv').set_index('aid')
    excluded=set(read_competition_csv(a.test,'test').SMILES.map(canonical)); excluded.discard(None)
    rows=[]; counts={'assays':0,'excluded_test_rows':0,'missing_structure':0,'unrecognized_format':0}
    for req in manifest['downloads']:
        if req['status']!='downloaded': continue
        aid=int(req['aid']); meta=idx.loc[aid]; title=str(meta['name']); lower=title.lower()
        # A target-virus ratio involving another virus or protein is not antiviral selectivity.
        if not ('vaccinia' in lower and ('selectivity index' in lower or 'selective index' in lower) and any(x in lower for x in ('cc50','tc50','mcc','cytotoxic'))): continue
        df=pd.read_csv(raw/req['file'],skiprows=[1,2],dtype=str,keep_default_na=False)
        columns={'PUBCHEM_EXT_DATASOURCE_SMILES','Standard Value','Standard Relation','Standard Type'}
        if not columns.issubset(df.columns): counts['unrecognized_format']+=1; continue
        counts['assays']+=1
        for _,r in df.iterrows():
            smiles=canonical(r['PUBCHEM_EXT_DATASOURCE_SMILES'])
            if not smiles: counts['missing_structure']+=1;continue
            if smiles in excluded: counts['excluded_test_rows']+=1;continue
            try: value=float(r['Standard Value'])
            except ValueError: continue
            if not math.isfinite(value) or value<0: continue
            relation=r['Standard Relation'].strip(); typ=r['Standard Type'].strip()
            rows.append({'SMILES':smiles,'SI VACV':value if relation=='=' else '', 'value':value,'relation':relation,'endpoint':typ,'assay_title':title,'assay_description':meta['description'],'source':meta['source'],'assay_id':aid,'pubchem_cid':r.get('PUBCHEM_CID',''),'pubchem_sid':r.get('PUBCHEM_SID',''),'source_url':f'https://pubchem.ncbi.nlm.nih.gov/bioassay/{aid}','review_status':'pending_primary_source_review','note':'Curated from depositor data; verify assay type, source, licence and repeat measurements.'})
    fields=['SMILES','SI VACV','value','relation','endpoint','assay_title','assay_description','source','assay_id','pubchem_cid','pubchem_sid','source_url','review_status','note']
    df=pd.DataFrame(rows,columns=fields); df.to_csv(out/'si_all_relations.csv',index=False)
    df[df.relation=='='].to_csv(out/'si_exact_candidates.csv',index=False)
    title = df.assay_title.str.lower()
    comparable = (df.relation=='=') & title.str.contains('cc50',regex=False) & (title.str.contains('ic50',regex=False) | title.str.contains('ec50',regex=False)) & ~title.str.contains('p37',regex=False)
    df[comparable].to_csv(out/'si_comparable_candidates.csv',index=False)
    counts.update({'rows':len(df),'unique_smiles':int(df.SMILES.nunique()),'exact_rows':int((df.relation=='=').sum()),'censored_rows':int((df.relation!='=').sum()),'comparable_candidates':int(comparable.sum())})
    (out/'manifest.json').write_text(json.dumps(counts,indent=2),encoding='utf-8'); print(json.dumps(counts,indent=2))
if __name__=='__main__':main()

