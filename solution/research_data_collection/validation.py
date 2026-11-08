"""Research provenance validation; no automatic biological compatibility claims."""
from rdkit import Chem

def validate_record(row):
    if Chem.MolFromSmiles(str(row.get('SMILES',''))) is None:raise ValueError('Invalid research SMILES')
    if not str(row.get('source_url','')).startswith(('https://','http://')):raise ValueError('Missing source URL')
    if row.get('review_status')!='approved':
        return {'scientifically_validated':False,'reason':'Primary-source pairing and conditions have not been manually verified'}
    if not row.get('compatibility_evidence') or not row.get('reviewer'):
        raise ValueError('Approval requires evidence and reviewer')
    from data_collection.validate import validate
    validate(dict(row))  # exact relations, convertible units, positive concentrations and assay context
    return {'scientifically_validated':True,'reason':'Explicit reviewer attestation; consult original source'}
