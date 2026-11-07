import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "experiments" / "results"
OURS_PATH = ROOT / "submission_final_seed_avg.csv"
FRIEND_PATH = ROOT / "from_friend" / "results" / "isotonic_gcn_7357_2643" / "submission.csv"
TRAIN_PATH = ROOT / "train_v2.csv"
TEST_PATH = ROOT / "test_clean.csv"


def load_submission(path, name):
    frame = pd.read_csv(path, usecols=["index", "SI VACV"])
    frame["SI VACV"] = pd.to_numeric(frame["SI VACV"], errors="raise")
    values = frame["SI VACV"].to_numpy(dtype=float)
    if (
        len(frame) != 597
        or frame["index"].duplicated().any()
        or not np.isfinite(values).all()
        or (values < 0).any()
    ):
        raise ValueError(f"{name} submission is invalid")
    return frame


def summarize(values):
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)),
        "zero_count": int(np.count_nonzero(values == 0)),
        "below_0_1_count": int(np.count_nonzero(values < 0.1)),
        "below_1_count": int(np.count_nonzero(values < 1.0)),
        "unique_values": int(np.unique(values).size),
        "max_exact_value_frequency": int(pd.Series(values).value_counts().max()),
        "repeated_exact_value_count": int(
            np.count_nonzero(pd.Series(values).value_counts().to_numpy() > 1)
        ),
    }


def threshold_check(friend_values):
    candidates = (0.0, 0.1, 1.0, 10.0, 100.0, 10000.0)
    exact_counts = {
        str(value): int(np.count_nonzero(friend_values == value))
        for value in candidates
    }
    rounded_counts = {
        str(value): int(np.count_nonzero(np.round(friend_values, 2) == value))
        for value in candidates
    }
    max_frequency = int(pd.Series(friend_values).value_counts().max())
    plateau = max_frequency >= 5 or any(
        exact_counts[str(value)] >= 5 for value in candidates
    )
    return {
        "detected": bool(plateau),
        "rule": "At least 5 teammate predictions share an exact value or a listed sentinel.",
        "candidate_exact_counts": exact_counts,
        "candidate_rounded_2dp_counts": rounded_counts,
        "maximum_exact_value_frequency": max_frequency,
        "sentinel_threshold": None,
        "decision": "No supported threshold; leave our predictions unchanged.",
    }


def zero_target_diagnostics(target, predictions):
    if target.shape != predictions.shape or not np.isfinite(predictions).all():
        raise ValueError("OOF target and prediction arrays are misaligned")
    zero = target == 0
    squared_errors = np.square(predictions - target)
    total_sse = float(squared_errors.sum())
    zero_sse = float(squared_errors[zero].sum())
    return {
        "rows": int(len(target)),
        "zero_target_rows": int(zero.sum()),
        "zero_target_fraction_rows": float(zero.mean()),
        "oof_mse_all_rows": float(squared_errors.mean()),
        "mse_on_zero_target_rows": float(squared_errors[zero].mean()),
        "zero_row_sse_contribution_to_overall_mse": float(zero_sse / len(target)),
        "zero_row_share_of_total_oof_squared_error": (
            float(zero_sse / total_sse) if total_sse else 0.0
        ),
        "mse_on_nonzero_target_rows": float(squared_errors[~zero].mean()),
    }


def main():
    for path in (OURS_PATH, FRIEND_PATH, TRAIN_PATH, TEST_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Missing diagnostic input: {path}")

    ours = load_submission(OURS_PATH, "ours")
    friend = load_submission(FRIEND_PATH, "friend")
    merged = ours.merge(
        friend,
        on="index",
        suffixes=("_ours", "_friend"),
        validate="one_to_one",
    )
    if len(merged) != 597:
        raise ValueError("Submissions do not contain the same 597 indices")
    ours_values = merged["SI VACV_ours"].to_numpy(dtype=float)
    friend_values = merged["SI VACV_friend"].to_numpy(dtype=float)
    difference = np.abs(ours_values - friend_values)
    merged["absolute_difference"] = difference

    test = pd.read_csv(
        TEST_PATH,
        usecols=["index", "Title", "SMILES"],
    )
    top10 = (
        merged.nlargest(10, "absolute_difference")
        .merge(test, on="index", validate="one_to_one")
        .sort_values("absolute_difference", ascending=False)
    )

    train = pd.read_csv(TRAIN_PATH, usecols=["SI VACV"])
    target = pd.to_numeric(train["SI VACV"], errors="raise").to_numpy(dtype=float)
    if len(target) != 320 or not np.isfinite(target).all():
        raise ValueError("Expected 320 valid v2 SI targets")
    seed_avg_oof_path = RESULTS / "exp53_gcn_seed_avg_oof.npy"
    friend_oof_path = FRIEND_PATH.parent / "gcn_oof.csv"
    if not seed_avg_oof_path.is_file() or not friend_oof_path.is_file():
        raise FileNotFoundError("Required OOF data for zero-target error attribution is missing")

    our_oof = np.load(seed_avg_oof_path)
    friend_oof = pd.read_csv(friend_oof_path).sort_values("row")
    if (
        our_oof.shape != target.shape
        or len(friend_oof) != len(target)
        or not np.array_equal(
            friend_oof["true"].to_numpy(dtype=float),
            target,
        )
    ):
        raise ValueError("OOF diagnostics are not aligned to train_v2.csv")
    our_zero_error = zero_target_diagnostics(target, our_oof)
    friend_zero_error = zero_target_diagnostics(
        target,
        friend_oof["gcn_oof_seed_average"].to_numpy(dtype=float),
    )

    threshold = threshold_check(friend_values)
    thresholded_path = ROOT / "submission_ours_thresholded.csv"
    thresholded_submission = None
    if threshold["detected"]:
        sentinel = threshold["sentinel_threshold"]
        if sentinel is None:
            raise ValueError("A plateau was detected but no supported sentinel was identified")
        thresholded_values = np.where(ours_values < sentinel, sentinel, ours_values)
        thresholded = pd.DataFrame(
            {"index": merged["index"], "SI VACV": thresholded_values}
        ).sort_values("index", kind="stable")
        thresholded.to_csv(thresholded_path, index=False)
        thresholded_submission = str(thresholded_path.relative_to(ROOT))

    report = {
        "submissions": {
            "ours": {
                "path": str(OURS_PATH.relative_to(ROOT)),
                **summarize(ours_values),
            },
            "friend": {
                "path": str(FRIEND_PATH.relative_to(ROOT)),
                **summarize(friend_values),
            },
        },
        "prediction_correlation": float(np.corrcoef(ours_values, friend_values)[0, 1]),
        "top10_absolute_prediction_differences": [
            {
                "index": int(row["index"]),
                "title": str(row["Title"]),
                "smiles": str(row["SMILES"]),
                "ours": float(row["SI VACV_ours"]),
                "friend": float(row["SI VACV_friend"]),
                "absolute_difference": float(row["absolute_difference"]),
            }
            for _, row in top10.iterrows()
        ],
        "threshold_check": threshold,
        "thresholded_submission": thresholded_submission,
        "train_zero_target_error": {
            "ours_seed_averaged_oof": our_zero_error,
            "friend_gcn_oof": friend_zero_error,
            "note": "The friend submission is a CPU+isotonic-GCN blend; its available OOF file is the raw GCN component, not the complete submitted blend.",
        },
        "interpretation": {
            "test_target_labels_are_unavailable": True,
            "score_gap_cannot_be_attributed_to_a_threshold_from_submissions_alone": True,
            "friend_submission_is_not_the_same_model_as_ours": (
                "Friend submission is the fixed CPU/isotonic-GCN blend; ours is a five-seed raw GCN average."
            ),
        },
    }
    report_path = RESULTS / "diagnose_kaggle_gap.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
