import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_prep import load_test, load_train, validate_submission
from models import featurize_smiles
from pipeline import SEEDS, make_folds


def main():
    train, target, activity = load_train()
    test = load_test()
    submission_path = ROOT / "submission" / "submission_final.csv"
    report_path = ROOT / "experiments" / "results" / "final_pipeline_report.json"
    probability_path = ROOT / "experiments" / "results" / "final_classifier_oof_probs.npy"
    logsi_path = ROOT / "experiments" / "results" / "final_regressor_oof_log1p.npy"
    calibrated_path = ROOT / "experiments" / "results" / "final_crossfitted_calibrated_oof.npy"

    submission = pd.read_csv(submission_path)
    validate_submission(submission)
    if not np.array_equal(
        submission["index"].to_numpy(),
        np.sort(test["index"].to_numpy()),
    ):
        raise ValueError("Submission indices do not match test indices")

    raw = featurize_smiles(train["SMILES"].astype(str).tolist())
    for seed in SEEDS:
        for train_indices, valid_indices in make_folds(target, raw["canonical"], seed):
            if set(raw["canonical"][train_indices]) & set(raw["canonical"][valid_indices]):
                raise ValueError(f"Canonical SMILES overlap in seed {seed} folds")

    probabilities = np.load(probability_path)
    logsi = np.load(logsi_path)
    calibrated = np.load(calibrated_path)
    for name, values in (
        ("probabilities", probabilities),
        ("log1p(SI)", logsi),
        ("calibrated blend", calibrated),
    ):
        if values.shape != (len(train),) or not np.isfinite(values).all():
            raise ValueError(f"Invalid OOF {name} array: {values.shape}")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("OOF classifier probabilities must be in [0, 1]")
    if (calibrated < 0).any():
        raise ValueError("Calibrated SI predictions must be nonnegative")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["final_submission"]["rows"] != len(submission):
        raise ValueError("Pipeline report and submission row counts differ")
    print(
        json.dumps(
            {
                "submission_rows": len(submission),
                "submission_min": float(submission["SI VACV"].min()),
                "submission_max": float(submission["SI VACV"].max()),
                "submission_mean": float(submission["SI VACV"].mean()),
                "submission_median": float(submission["SI VACV"].median()),
                "max_target_9000_met": bool(submission["SI VACV"].max() <= 9000.0),
                "classifier_roc_auc_oof": float(report["classifier_roc_auc"]),
                "diagnostic_calibrated_oof_mse": float(
                    mean_squared_error(target, calibrated)
                ),
                "fold_group_leakage": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
