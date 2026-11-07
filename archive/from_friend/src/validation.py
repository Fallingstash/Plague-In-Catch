"""Data provenance, grouped folds and metrics shared by all experiments."""
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from competition_data import read_competition_csv
import vacv_cpu_improved as cpu

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_TRAIN_SHA256 = 'd9a876a67b5623d9cdb6592fc53d31ffaa7c535b09968c3a538138740fe246e8'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    from competition_data import require_safe_output
    path = require_safe_output(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)

def training_data(path=None):
    path = Path(path or ROOT / 'train.csv')
    if sha(path) != OFFICIAL_TRAIN_SHA256:
        raise ValueError('Only the SHA256-pinned official train.csv is allowed; external/test labels are forbidden')
    df = read_competition_csv(path, 'train')[['SMILES', 'SI VACV']].copy()
    y = df['SI VACV'].to_numpy(float)
    if len(df) != 320 or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Invalid official training targets')
    return df

def folds(y, groups, seed=42, count=5):
    result = cpu.make_folds(np.asarray(y), np.asarray(groups), count, seed)
    coverage = np.zeros(len(y), int)
    for tr, va in result:
        if not set(groups[tr]).isdisjoint(groups[va]):
            raise ValueError('Molecular groups cross train/validation boundary')
        coverage[va] += 1
    if not np.all(coverage == 1):
        raise ValueError('Each row must have exactly one OOF prediction per split seed')
    return result

def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if y.shape != p.shape or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Invalid or incomplete predictions')
    err = (y-p)**2
    subgroup = lambda mask: float(err[mask].mean()) if mask.any() else None
    return dict(MSE=float(err.mean()), RMSE=float(np.sqrt(err.mean())), MAE=float(mean_absolute_error(y,p)),
                MSE_zero=subgroup(y == 0), MSE_positive=subgroup(y > 0), MSE_high=subgroup(y >= 1000),
                max_target_error_share=float(err[np.argmax(y)] / err.sum()) if err.sum() else 0.,
                MSE_without_max_DIAGNOSTIC_ONLY=float(np.delete(err,np.argmax(y)).mean()))

def protect_output(path):
    path = Path(path).resolve()
    for name in ('top1','top2'):
        protected = (ROOT / 'versions' / name).resolve()
        if path == protected or protected in path.parents:
            raise ValueError('Confirmed top1/top2 are read-only; promotion requires a separate confirmed-score operation')
    return path

def assert_protected_hashes(snapshot=None):
    snapshot = Path(snapshot or ROOT/'experiments/research_20261107/initial_hashes.json')
    initial = json.loads(snapshot.read_text(encoding='utf-8'))
    checked={'top1':0,'top2':0}
    for name, expected in initial.items():
        normalized=name.replace('\\','/')
        for rank in checked:
            if normalized.startswith(f'versions/{rank}/'):
                if sha(ROOT/normalized) != expected:raise ValueError(f'Protected artifact changed: {name}')
                checked[rank]+=1
    if not all(checked.values()):raise ValueError('Snapshot does not cover both protected leader directories')
    return True
