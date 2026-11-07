"""Extract candidate same-publication pairs; candidates are NOT approved labels."""
import argparse,csv,json
from pathlib import Path
import pandas as pd
from rdkit import Chem
from competition_data import read_competition_csv

def canonical(smiles):
 mol=Chem.MolFromSmiles(str(smiles))
 return Chem.MolToSmiles(mol,isomericSmiles=True) if mol is not None else None

def main():
 p=argparse.ArgumentParser(); p.add_argument('--raw',default='data/external/raw/chembl'); p.add_argument('--output',default='data/external/processed/chembl_candidates.csv'); p.add_argument('--exclude-test',default='test.csv',help='Never export ChEMBL labels for structures in Kaggle test'); a=p.parse_args()
 folder=Path(a.raw); manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
 if manifest.get('status')!='complete': raise ValueError('Refusing incomplete source download')
 test=read_competition_csv(a.exclude_test,'test'); test_structures={canonical(s) for s in test['SMILES']}; test_structures.discard(None)
 assays={}; molecules={}; docs={}
 # Resolve structure IDs first. Do not inspect activity values for any test molecule.
 for request in manifest['requests']:
  kind=request['kind']
  if kind not in ('assays','cc50_assays','molecule','molecules','document','documents'): continue
  data=json.loads((folder/request['file']).read_text(encoding='utf-8'))
  if kind in ('assays','cc50_assays'): assays.update({x['assay_chembl_id']:x for x in data.get('assays',[]) if x.get('assay_chembl_id')})
  elif kind=='molecule': molecules[data['molecule_chembl_id']]=data
  elif kind=='molecules': molecules.update({x['molecule_chembl_id']:x for x in data.get('molecules',[]) if x.get('molecule_chembl_id')})
  elif kind=='document': docs[data['document_chembl_id']]=data
  elif kind=='documents': docs.update({x['document_chembl_id']:x for x in data.get('documents',[]) if x.get('document_chembl_id')})
 test_molecule_ids=set()
 for mid,mol in molecules.items():
  smiles=(mol.get('molecule_structures') or {}).get('canonical_smiles','')
  if canonical(smiles) in test_structures: test_molecule_ids.add(mid)
 virus=[]; tox=[]
 for request in manifest['requests']:
  kind=request['kind']
  if kind not in ('activities','cc50'): continue
  data=json.loads((folder/request['file']).read_text(encoding='utf-8'))
  # Exclude by ChEMBL ID before reading or retaining any measurement fields.
  kept=[x for x in data.get('activities',[]) if x.get('molecule_chembl_id') not in test_molecule_ids]
  (virus if kind=='activities' else tox).extend(kept)
 by_key={}
 for record in tox:
  if record.get('molecule_chembl_id') and record.get('document_chembl_id'): by_key.setdefault((record['molecule_chembl_id'],record['document_chembl_id']),[]).append(record)
 rows=[]
 excluded_test=0
 for v in virus:
  if str(v.get('standard_type','')).upper()!='IC50': continue
  mid=v.get('molecule_chembl_id'); did=v.get('document_chembl_id')
  for cc in by_key.get((mid,did),[]):
   doc=docs.get(did,{}); mol=molecules.get(mid,{}); assay=assays.get(v.get('assay_chembl_id'),{}); tox_assay=assays.get(cc.get('assay_chembl_id'),{})
   smiles=(mol.get('molecule_structures') or {}).get('canonical_smiles',''); canon=canonical(smiles)
   if not canon or canon in test_structures:
    excluded_test += int(canon in test_structures); continue
   rows.append({'SMILES':canon,'molecule_chembl_id':mid,'document_chembl_id':did,'DOI':doc.get('doi',''),
    'source_url':f'https://www.ebi.ac.uk/chembl/document_report_card/{did}/','study_title':doc.get('title',''),
    'ic50_value':v.get('standard_value'),'ic50_units':v.get('standard_units'),'ic50_relation':v.get('standard_relation'), 'ic50_activity_id':v.get('activity_id'),
    'cc50_value':cc.get('standard_value'),'cc50_units':cc.get('standard_units'),'cc50_relation':cc.get('standard_relation'),'cc50_activity_id':cc.get('activity_id'),
    'ic50_assay_id':v.get('assay_chembl_id'),'cc50_assay_id':cc.get('assay_chembl_id'),'viral_assay_description':assay.get('description',''),
    'viral_organism':assay.get('assay_organism',''),'ic50_assay_cell_type':assay.get('assay_cell_type',''),
    'cc50_assay_description':tox_assay.get('description',''),'cc50_assay_cell_type':tox_assay.get('assay_cell_type',''),
    'culture':'','strain':'','compatibility_evidence':'','reviewer':'','review_status':'pending'})
 fields=['SMILES','molecule_chembl_id','document_chembl_id','DOI','source_url','study_title','ic50_value','ic50_units','ic50_relation','ic50_activity_id','cc50_value','cc50_units','cc50_relation','cc50_activity_id','ic50_assay_id','cc50_assay_id','viral_assay_description','viral_organism','ic50_assay_cell_type','cc50_assay_description','cc50_assay_cell_type','culture','strain','compatibility_evidence','reviewer','review_status']
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 print(f'{len(rows)} candidate pairs -> {out}; excluded {len(test_molecule_ids)} ChEMBL IDs matching canonical test structures and {excluded_test} pairs. No pair is approved for training.')
if __name__=='__main__': main()
