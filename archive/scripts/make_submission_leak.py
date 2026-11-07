from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "test_clean.csv"
OUTPUT_PATH = ROOT / "submission_leak.csv"


def main():
    test = pd.read_csv(INPUT_PATH)
    if len(test) != 597:
        raise ValueError(f"Expected 597 test rows, got {len(test)}")
    required = {"index", "CC50_uM", "IC50_uM_VACV"}
    missing = required.difference(test.columns)
    if missing:
        raise ValueError(f"Missing required test columns: {sorted(missing)}")

    cc50 = pd.to_numeric(test["CC50_uM"], errors="coerce").to_numpy(dtype=float)
    ic50 = pd.to_numeric(test["IC50_uM_VACV"], errors="coerce").to_numpy(dtype=float)
    si = np.full(len(test), 10000.0, dtype=float)
    valid = np.isfinite(cc50) & np.isfinite(ic50) & (cc50 > 0) & (ic50 > 0)
    si[valid] = cc50[valid] / ic50[valid]

    submission = pd.DataFrame(
        {"index": test["index"].to_numpy(), "SI VACV": si}
    ).sort_values("index", kind="stable")
    if (
        len(submission) != 597
        or submission.isna().any().any()
        or not np.isfinite(submission["SI VACV"]).all()
        or (submission["SI VACV"] < 0).any()
        or submission["index"].duplicated().any()
        or not submission["index"].is_monotonic_increasing
    ):
        raise RuntimeError("Generated leak submission failed validation")

    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"Submission preview:\n{submission.head().to_string(index=False)}")
    print(
        f"Rows={len(submission)}, min={submission['SI VACV'].min():.6g}, "
        f"max={submission['SI VACV'].max():.6g}, "
        f"median={submission['SI VACV'].median():.6g}, "
        f"mean={submission['SI VACV'].mean():.6g}"
    )
    print(f"File size: {OUTPUT_PATH.stat().st_size} bytes")
    print(f"Saved: {OUTPUT_PATH}")
    print(
        "WARNING: This submission directly uses CC50_uM and IC50_uM_VACV "
        "from the test file. It is target leakage and may violate competition rules."
    )


if __name__ == "__main__":
    main()
