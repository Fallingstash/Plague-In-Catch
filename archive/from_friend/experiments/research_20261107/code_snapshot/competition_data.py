"""Strict readers for the competition split.

SI VACV from train is the only target. IC50/CC50 columns are excluded at CSV
read time for both splits; test may contain hidden measurements, which must not
be read, used for features, CV, model selection, or training.
"""
from pathlib import Path
import pandas as pd

TARGET = "SI VACV"
FORBIDDEN_PREFIXES = ("ic50", "cc50")


def read_competition_csv(path, split):
    if split not in {"train", "test", "inference"}:
        raise ValueError("split must be train, test, or inference")
    path = Path(path)
    columns = list(pd.read_csv(path, nrows=0).columns)
    if split == "train" and TARGET not in columns:
        raise ValueError(f"Training file must contain {TARGET!r}")
    if split == "test" and TARGET in columns:
        raise ValueError("Refusing test file that contains the target SI VACV")
    # usecols prevents pandas from parsing prohibited concentration values.
    allowed = [c for c in columns if not c.casefold().startswith(FORBIDDEN_PREFIXES)]
    return pd.read_csv(path, usecols=allowed)
