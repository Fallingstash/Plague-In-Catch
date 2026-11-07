import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "experiments" / "results"


def load_first_existing(*paths):
    for path in paths:
        if path.is_file():
            return np.load(path)
    raise FileNotFoundError("None of the expected OOF files exists: " + ", ".join(map(str, paths)))


def main():
    train = pd.read_csv(ROOT / "train_clean.csv")
    expanded_probability = load_first_existing(
        RESULTS_DIR / "exp02oofprobsexpanded.npy",
        RESULTS_DIR / "exp02oof_probs_expanded.npy",
    )
    original_probability = load_first_existing(
        RESULTS_DIR / "exp02oofprobs.npy",
        RESULTS_DIR / "exp02oof_probs.npy",
    )
    if len(train) != 318 or expanded_probability.shape != (803,):
        raise ValueError(
            f"Expected 318 original labels and 803 expanded probabilities, "
            f"got {len(train)} and {expanded_probability.shape}"
        )
    if original_probability.shape != (318,):
        raise ValueError(
            f"Expected original probabilities with shape (318,), got {original_probability.shape}"
        )
    if not np.isfinite(expanded_probability).all() or not np.isfinite(original_probability).all():
        raise ValueError("Probability arrays contain missing or non-finite values")

    labels = train["is_active"].to_numpy(dtype=int)
    expanded_original_probability = expanded_probability[:318]
    original_prediction = (original_probability >= 0.5).astype(int)
    expanded_prediction = (expanded_original_probability >= 0.5).astype(int)
    expanded_confusion = confusion_matrix(labels, expanded_prediction, labels=[0, 1])
    original_confusion = confusion_matrix(labels, original_prediction, labels=[0, 1])
    active_mean = float(expanded_original_probability[labels == 1].mean())
    inactive_mean = float(expanded_original_probability[labels == 0].mean())
    expanded_auc = float(roc_auc_score(labels, expanded_original_probability))
    original_auc = float(roc_auc_score(labels, original_probability))
    predicted_positive_rate = float(expanded_prediction.mean())

    print(f"Expanded classifier mean P(active), active rows: {active_mean:.6f}")
    print(f"Expanded classifier mean P(active), inactive rows: {inactive_mean:.6f}")
    print(f"Expanded classifier ROC-AUC (original 318): {expanded_auc:.6f}")
    print(f"Original classifier ROC-AUC: {original_auc:.6f}")
    print(f"Predicted active rate (expanded, threshold=0.5): {predicted_positive_rate:.6f}")
    print(f"Original confusion matrix [[tn, fp], [fn, tp]]:\n{original_confusion}")
    print(f"Expanded confusion matrix [[tn, fp], [fn, tp]]:\n{expanded_confusion}")
    if expanded_confusion[0, 1] + expanded_confusion[0, 0] == 0:
        raise RuntimeError("No inactive rows in the original labels")
    if expanded_confusion[0, 0] == 0:
        print("Diagnosis: expanded classifier predicts every inactive row as active.")
    elif expanded_confusion[0, 0] < expanded_confusion[0, 1]:
        print("Diagnosis: expanded classifier favors active labels on inactive rows, but does not classify all rows as active.")
    else:
        print("Diagnosis: no evidence that the classifier simply predicts everything active.")


if __name__ == "__main__":
    main()
