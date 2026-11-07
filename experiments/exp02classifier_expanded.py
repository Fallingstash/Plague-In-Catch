import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

from expanded_common import (
    CLASSIFIER_PARAMS,
    EARLY_STOPPING_ROUNDS,
    RESULTS_DIR,
    fold_records,
    load_expanded_data,
    load_fold_artifact,
    purge_canonical_duplicates,
    write_result,
)


def main():
    data = load_expanded_data()
    folds = fold_records(load_fold_artifact(), "classifier_803_folds")
    oof_probability = np.full(len(data["labels_classifier"]), np.nan, dtype=float)
    fold_metrics = []
    best_iterations = []
    purged_rows = []
    for fold in folds:
        train_indices = np.asarray(fold["train_indices"], dtype=int)
        valid_indices = np.asarray(fold["validation_indices"], dtype=int)
        train_indices, purged_count = purge_canonical_duplicates(
            train_indices, valid_indices, data["canonical_classifier"]
        )
        purged_rows.append(purged_count)
        model = lgb.LGBMClassifier(**CLASSIFIER_PARAMS)
        model.fit(
            data["x_classifier"][train_indices],
            data["labels_classifier"][train_indices],
            eval_set=[
                (
                    data["x_classifier"][valid_indices],
                    data["labels_classifier"][valid_indices],
                )
            ],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        oof_probability[valid_indices] = model.predict_proba(
            data["x_classifier"][valid_indices],
            num_iteration=model.best_iteration_,
        )[:, 1]
        best = int(model.best_iteration_ or CLASSIFIER_PARAMS["n_estimators"])
        best_iterations.append(best)
        original_valid = valid_indices[valid_indices < 318]
        fold_metrics.append(
            float(
                roc_auc_score(
                    data["labels_classifier"][original_valid],
                    oof_probability[original_valid],
                )
            )
        )
        print(
            f"Fold {fold['fold']}: ROC-AUC original rows="
            f"{fold_metrics[-1]:.6f}, best={best}"
        )

    if not np.isfinite(oof_probability).all():
        raise RuntimeError("Expanded classifier OOF contains missing or non-finite values")
    original_labels = data["labels_classifier"][:318]
    original_probability = oof_probability[:318]
    original_prediction = (original_probability >= 0.5).astype(int)
    roc_auc = float(roc_auc_score(original_labels, original_probability))
    pr_auc = float(average_precision_score(original_labels, original_probability))
    accuracy = float(accuracy_score(original_labels, original_prediction))
    confusion = confusion_matrix(
        original_labels, original_prediction, labels=[0, 1]
    ).tolist()
    all_roc_auc = float(
        roc_auc_score(data["labels_classifier"], oof_probability)
    )

    np.save(RESULTS_DIR / "exp02oof_probs_expanded.npy", oof_probability)
    path = write_result(
        "exp02classifier_expanded.json",
        {
            "model": "lgbmclassifier_expanded_803",
            "params": CLASSIFIER_PARAMS
            | {"early_stopping_rounds": EARLY_STOPPING_ROUNDS, "eval_metric": "auc"},
            "cvmetric": roc_auc,
            "metricname": "ROCAUC_original_318",
            "pr_auc_original_318": pr_auc,
            "accuracy_original_318": accuracy,
            "confusion_matrix_original_318": confusion,
            "roc_auc_all_803": all_roc_auc,
            "foldmetrics": fold_metrics,
            "bestiterations": best_iterations,
            "canonical_duplicate_rows_purged_per_fold": purged_rows,
            "notes": "Models train on 803 rows; primary ROC-AUC is OOF on the original 318 labels for comparability; canonical-SMILES duplicates are purged from each fold's training data; no IC50/CC50 features.",
        },
    )
    print(f"Classifier rows: {len(data['labels_classifier'])}")
    print(f"ROC-AUC original 318: {roc_auc:.6f}")
    print(f"ROC-AUC all 803 (source-mixed): {all_roc_auc:.6f}")
    print(f"PR-AUC: {pr_auc:.6f}; accuracy: {accuracy:.6f}")
    print(f"Confusion matrix: {confusion}")
    print(f"Best iterations: {best_iterations}")
    print(f"Saved result: {path}")


if __name__ == "__main__":
    main()
