"""Validate manually reviewed exact, compatible VACV measurement pairs."""
import argparse,math
from pathlib import Path
import pandas as pd
from rdkit import Chem
from competition_data import read_competition_csv

FACTORS={'M':1e6,'mM':1e3,'uM':1.0,'µM':1.0,'μM':1.0,'nM':1e-3,'pM':1e-6}
REQUIRED=['SMILES','source_url','ic50_value','ic50_units','ic50_relation','cc50_value','cc50_units','cc50_relation','culture','strain','compatibility_evidence','reviewer','review_status']
def concentration(value,unit,relation):
 if str(relation).strip()!='=': raise ValueError('censored or non-exact relation')
 if str(unit).strip() not in FACTORS: raise ValueError('unsupported unit')
 try: val=float(value)
 except (TypeError,ValueError): raise ValueError('invalid concentration')
 if not math.isfinite(val) or val<=0: raise ValueError('concentration must be finite and positive')
 return val*FACTORS[str(unit).strip()]

def canonical_smiles(smiles):
 mol=Chem.MolFromSmiles(str(smiles))
 if mol is None: raise ValueError('invalid SMILES')
 return Chem.MolToSmiles(mol,isomericSmiles=True)

def validate(row,excluded_test_structures=None):
 for key in REQUIRED:
  if not str(row.get(key,'')).strip(): raise ValueError('missing '+key)
 if str(row['review_status']).strip().lower()!='approved': raise ValueError('not manually approved')
 if not str(row['source_url']).startswith(('https://','http://')): raise ValueError('invalid source URL')
 canonical=canonical_smiles(row['SMILES'])
 if excluded_test_structures and canonical in {canonical_smiles(s) for s in excluded_test_structures}: raise ValueError('structure is present in competition test set')
 mol=Chem.MolFromSmiles(canonical)
 ic=concentration(row['ic50_value'],row['ic50_units'],row['ic50_relation']); cc=concentration(row['cc50_value'],row['cc50_units'],row['cc50_relation'])
 output=dict(row); output.update({'SMILES':Chem.MolToSmiles(mol,isomericSmiles=True),'IC50_uM_VACV':ic,'CC50_uM':cc,'SI VACV':cc/ic,'Culture':row['culture'],'strain':row['strain'],'Source_URL':row['source_url']})
 return output

def main():
 p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',default='data/external/processed/validated.csv'); p.add_argument('--rejected',default='data/external/processed/rejected.csv'); p.add_argument('--test',default='test.csv'); a=p.parse_args()
 test_df=read_competition_csv(a.test,'test'); test_structures={canonical_smiles(x) for x in test_df['SMILES']}
 df=pd.read_csv(a.input,dtype=str,keep_default_na=False); missing=set(REQUIRED)-set(df.columns)
 if missing: raise ValueError('Missing columns: '+', '.join(sorted(missing)))
 accepted=[]; rejected=[]
 for _,row in df.iterrows():
  try: accepted.append(validate(row.to_dict(),test_structures))
  except ValueError as exc: rejected.append({**row.to_dict(),'rejection_reason':str(exc)})
 out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); reject=Path(a.rejected); reject.parent.mkdir(parents=True,exist_ok=True)
 columns=list(dict.fromkeys(list(df.columns)+['IC50_uM_VACV','CC50_uM','SI VACV','Culture','strain','Source_URL']))
 pd.DataFrame(accepted,columns=columns).to_csv(out,index=False); pd.DataFrame(rejected,columns=list(df.columns)+['rejection_reason']).to_csv(reject,index=False)
 print(f'Approved {len(accepted)}, rejected {len(rejected)}; reviewer attestation is not independent proof of compatibility.')
if __name__=='__main__': main()
