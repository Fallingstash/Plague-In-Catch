"""Unsupervised per-molecule cache. No target or test-set fitting."""
import hashlib
from pathlib import Path
import joblib
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

VERSION = 'rdkit-' + rdBase.rdkitVersion + '-smiles-only-v1'
FAMILIES = [f'morgan{r}_{bits}' for r in (2,3) for bits in (512,1024,2048)] + ['maccs','rdkit2048','atom_pair','torsion']

def canonical(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or not mol.GetNumAtoms():
        raise ValueError('Invalid/empty SMILES')
    return Chem.MolToSmiles(mol, isomericSmiles=True)

def single(smiles):
    key = canonical(smiles); mol = Chem.MolFromSmiles(key)
    result = {'canonical':key, 'scaffold':MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or key}
    for r in (2,3):
        for bits in (512,1024,2048):
            result[f'morgan{r}_{bits}'] = rdFingerprintGenerator.GetMorganGenerator(radius=r, fpSize=bits, includeChirality=True).GetFingerprintAsNumPy(mol).astype(np.float32)
    result['maccs'] = np.asarray(MACCSkeys.GenMACCSKeys(mol),dtype=np.float32)
    result['rdkit2048'] = np.asarray(Chem.RDKFingerprint(mol,fpSize=2048),dtype=np.float32)
    result['atom_pair'] = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048, includeChirality=True).GetFingerprintAsNumPy(mol).astype(np.float32)
    result['torsion'] = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048, includeChirality=True).GetFingerprintAsNumPy(mol).astype(np.float32)
    result['desc'] = np.asarray([fn(mol) for _,fn in Descriptors._descList],float)
    return result

def features(smiles, cache=None):
    cache = Path(cache) if cache else None
    if cache: cache.mkdir(parents=True,exist_ok=True)
    values = []
    for sm in smiles:
        canonical_sm = canonical(sm)
        filename = hashlib.sha256((VERSION+canonical_sm).encode()).hexdigest()+'.joblib'
        path = cache/filename if cache else None
        if path and path.exists():
            result = joblib.load(path)
        else:
            result = single(canonical_sm)
            if path: joblib.dump(result,path,compress=3)
        values.append(result)
    return {key:np.asarray([v[key] for v in values]) for key in values[0]}

def tanimoto(a,b):
    a,b = np.asarray(a,float),np.asarray(b,float)
    inter = a@b.T; union = a.sum(1)[:,None]+b.sum(1)[None,:]-inter
    return np.divide(inter,union,out=np.zeros_like(inter),where=union>0)

def kernel(a,b,keys):
    return sum(tanimoto(a[k],b[k]) for k in keys)/len(keys)

def subset(raw, ids):
    return {k:v[ids] for k,v in raw.items()}
