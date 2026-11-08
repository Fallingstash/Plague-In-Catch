"""Build a training quarantine from exact ChEMBL VACV IC50/CC50 pairs.

The output retains only exact uM-convertible concentrations from same compound and
publication, explicitly VACV antiviral assays, cytotoxicity assays, and matching
annotated cell lines. It always excludes every canonical structure in test.csv.
Rows remain pending for primary-source review; nothing is silently auto-approved.
"""
import argparse,json,math
from pathlib import Path
import numpy as np,pandas as pd
from rdkit import Chem
from competition_data import read_competition_csv
FACTORS={'M':1e6,'mM':1e3,'uM':1.,'µM':1.,'μM':1.,'nM':1e-3,'pM':1e-6}
def canon(s):
 m=Chem.MolFromSmiles(str(s));return Chem.MolToSmiles(m,isomericSmiles=True) if m else None
def value(x,u,rel):
 if str(rel).strip()!='=' or str(u).strip() not in FACTORS:return np.nan
 try:v=float(x)*FACTORS[str(u).strip()]
 except (TypeError,ValueError):return np.nan
 return v if math.isfinite(v) and v>0 else np.nan
def main():
 p=argparse.ArgumentParser();p.add_argument('--candidates',default='data/external/processed/chembl_candidates.csv');p.add_argument('--test',default='test.csv');p.add_argument('--out',default='data/external/processed');a=p.parse_args()
 d=pd.read_csv(a.candidates,keep_default_na=False); test=set(read_competition_csv(a.test,'test').SMILES.map(canon));test.discard(None)
 rows=[];stats={'pairs':len(d),'test_structure_excluded':0,'not_exact_or_units':0,'not_explicit_vacv':0,'not_cytotoxicity':0,'cell_mismatch':0,'eligible_pending':0}
 for _,r in d.iterrows():
  s=canon(r.get('SMILES',''))
  if not s or s in test:stats['test_structure_excluded']+=1;continue
  if str(r.get('ic50_relation',''))!='=' or str(r.get('cc50_relation',''))!='=':stats['not_exact_or_units']+=1;continue
  ic=value(r.get('ic50_value'),r.get('ic50_units'),r.get('ic50_relation'));cc=value(r.get('cc50_value'),r.get('cc50_units'),r.get('cc50_relation'))
  if not np.isfinite(ic+cc):stats['not_exact_or_units']+=1;continue
  viral=str(r.get('viral_assay_description','')).casefold()
  if not ('vaccinia' in viral or 'vacv' in viral):stats['not_explicit_vacv']+=1;continue
  tox=str(r.get('cc50_assay_description','')).casefold()
  if 'cytotox' not in tox:stats['not_cytotoxicity']+=1;continue
  c1=str(r.get('ic50_assay_cell_type','')).strip().casefold();c2=str(r.get('cc50_assay_cell_type','')).strip().casefold()
  if not c1 or not c2 or c1!=c2:stats['cell_mismatch']+=1;continue
  rows.append({'SMILES':s,'SI VACV':cc/ic,'IC50_uM_VACV':ic,'CC50_uM':cc,'ic50_value':r.get('ic50_value',''),'ic50_units':r.get('ic50_units',''),'ic50_relation':r.get('ic50_relation',''),'cc50_value':r.get('cc50_value',''),'cc50_units':r.get('cc50_units',''),'cc50_relation':r.get('cc50_relation',''),'Culture':r.get('ic50_assay_cell_type',''),'culture':r.get('ic50_assay_cell_type',''),'strain':'','DOI':r.get('DOI',''),'Source_URL':r.get('source_url',''),'source_url':r.get('source_url',''),'compatibility_evidence':'','reviewer':'','document_chembl_id':r.get('document_chembl_id',''),'molecule_chembl_id':r.get('molecule_chembl_id',''),'ic50_assay_id':r.get('ic50_assay_id',''),'cc50_assay_id':r.get('cc50_assay_id',''),'ic50_description':r.get('viral_assay_description',''),'cc50_description':r.get('cc50_assay_description',''),'review_status':'pending_primary_source_review','review_notes':'same DOI, exact values, explicit VACV and matching cell label; verify primary-source pair and conditions before training'})
 stats['eligible_pending']=len(rows);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out/'chembl_quarantine.csv',index=False);(out/'chembl_filter_audit.json').write_text(json.dumps(stats,indent=2),encoding='utf-8');print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
