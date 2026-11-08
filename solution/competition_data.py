"""Strict readers for the competition split.

SI VACV from train is the only target. IC50/CC50 columns are excluded at CSV
read time for both splits; test may contain hidden measurements, which must not
be read, used for features, CV, model selection, or training.
"""
from pathlib import Path
import hashlib
import pandas as pd

TARGET = "SI VACV"
FORBIDDEN_PREFIXES = ("ic50", "cc50")
OFFICIAL_TRAIN_SHA256 = 'd9a876a67b5623d9cdb6592fc53d31ffaa7c535b09968c3a538138740fe246e8'

def require_safe_output(path):
    path=Path(path).resolve()
    root=Path(__file__).resolve().parent
    for name in ('top1','top2'):
        protected=(root/'versions'/name).resolve()
        if path==protected or protected in path.parents:
            raise ValueError('Refusing to overwrite protected top1/top2')
    return path


def read_competition_csv(path, split):
    if split not in {"train", "test", "inference"}:
        raise ValueError("split must be train, test, or inference")
    path = Path(path)
    if split == 'train' and hashlib.sha256(path.read_bytes()).hexdigest() != OFFICIAL_TRAIN_SHA256:
        raise ValueError('Training requires the SHA256-pinned official train.csv; external/test training is forbidden')
    columns = list(pd.read_csv(path, nrows=0).columns)
    if split == "train" and TARGET not in columns:
        raise ValueError(f"Training file must contain {TARGET!r}")
    if split == "test" and TARGET in columns:
        raise ValueError("Refusing test file that contains the target SI VACV")
    # usecols prevents pandas from parsing prohibited concentration values.
    allowed = [c for c in columns if not c.strip().casefold().startswith(FORBIDDEN_PREFIXES)]
    return pd.read_csv(path, usecols=allowed)
