from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _find_input(filename):
    candidates = (
        DATA / "inputs" / filename,
        DATA / filename,
        ROOT / filename,
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Required input is missing: {filename}")


def load_train():
    path = _find_input("train_v2.csv")
    frame = pd.read_csv(
        path,
        usecols=["SMILES", "SI VACV"],
    )
    frame["SI VACV"] = pd.to_numeric(frame["SI VACV"], errors="raise")
    target = frame["SI VACV"].to_numpy(dtype=np.float64)
    if len(frame) != 320 or not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("Expected 320 finite, nonnegative SI targets")
    if frame["SMILES"].isna().any():
        raise ValueError("Training SMILES must not be missing")
    active = (target > 0).astype(np.int64)
    if int(active.sum()) != 237:
        raise ValueError(
            f"Expected 237 positive-SI regression rows, got {int(active.sum())}"
        )
    return frame, target, active


def load_test():
    path = _find_input("test_clean.csv")
    frame = pd.read_csv(path, usecols=["index", "SMILES"])
    if (
        len(frame) != 597
        or frame["index"].isna().any()
        or frame["index"].duplicated().any()
        or frame["SMILES"].isna().any()
    ):
        raise ValueError("Expected 597 test rows with unique indices and SMILES")
    return frame


def validate_submission(frame):
    values = frame["SI VACV"].to_numpy(dtype=np.float64)
    if (
        list(frame.columns) != ["index", "SI VACV"]
        or len(frame) != 597
        or frame["index"].duplicated().any()
        or frame.isna().any().any()
        or not np.isfinite(values).all()
        or (values < 0).any()
    ):
        raise ValueError("Submission must contain 597 valid nonnegative SI predictions")
