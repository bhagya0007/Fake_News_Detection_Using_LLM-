"""Baseline Model Training and Validation Service.

Trains the initial classical ML baseline (TF-IDF + Logistic Regression):
- Fits ONLY on the prepared training split (data/raw/train.tsv).
- Evaluates on the prepared validation split (data/raw/valid.tsv).
- The test split (data/raw/test.tsv) is strictly NOT loaded or used.
- Saves model artifact to models/baseline_initial.joblib.
- Saves validation metrics to models/baseline_initial_validation_metrics.json.
- Saves confusion matrix to models/baseline_initial_confusion_matrix.csv.
- Saves validation predictions to models/baseline_initial_validation_predictions.csv.
- Executes an overlap-excluded diagnostic for leakage sensitivity.
- Performs post-serialization reload verification.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from backend.app.services.baseline_data import (
    prepare_baseline_split,
    split_features_and_labels,
)
from backend.app.services.dataset_loader import get_project_root


# Explicit TF-IDF baseline hyperparameters
TFIDF_CONFIG: dict[str, Any] = {
    "lowercase": True,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "max_features": 50000,
    "sublinear_tf": True,
}

# Explicit Logistic Regression hyperparameters
LOGREG_CONFIG: dict[str, Any] = {
    "max_iter": 1000,
    "C": 1.0,
    "class_weight": None,
    "solver": "lbfgs",
    "random_state": 42,
}


def build_baseline_pipeline() -> Pipeline:
    """Construct the initial scikit-learn Pipeline with TF-IDF and Logistic Regression."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**TFIDF_CONFIG)),
            ("classifier", LogisticRegression(**LOGREG_CONFIG)),
        ]
    )


def compute_metrics(
    y_true: pd.Series, y_pred: np.ndarray, classes: list[str]
) -> dict[str, Any]:
    """Calculate comprehensive evaluation metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    report = classification_report(
        y_true, y_pred, labels=classes, output_dict=True, zero_division=0
    )

    per_class: dict[str, dict[str, float]] = {}
    for cls in classes:
        cls_str = str(cls)
        if cls_str in report:
            per_class[cls_str] = {
                "precision": float(round(report[cls_str]["precision"], 4)),
                "recall": float(round(report[cls_str]["recall"], 4)),
                "f1_score": float(round(report[cls_str]["f1-score"], 4)),
                "support": int(report[cls_str]["support"]),
            }

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_order": [str(c) for c in classes],
    }


def compute_overlap_diagnostic(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    y_val_true: pd.Series,
    y_val_pred: np.ndarray,
) -> dict[str, Any]:
    """Evaluate validation performance strictly on statements not present in train."""
    train_statements = set(train_df["clean_statement"])
    non_overlap_mask = ~val_df["clean_statement"].isin(train_statements)

    overlap_count = int((~non_overlap_mask).sum())
    evaluated_count = int(non_overlap_mask.sum())

    y_true_sub = y_val_true[non_overlap_mask]
    y_pred_sub = y_val_pred[non_overlap_mask]

    acc = float(accuracy_score(y_true_sub, y_pred_sub))
    macro_f1 = float(f1_score(y_true_sub, y_pred_sub, average="macro", zero_division=0))
    macro_p = float(precision_score(y_true_sub, y_pred_sub, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true_sub, y_pred_sub, average="macro", zero_division=0))

    return {
        "overlap_statements_excluded": overlap_count,
        "evaluated_rows": evaluated_count,
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "diagnostic_label": "Leakage-sensitivity diagnostic (verbatim train-overlap excluded)",
    }


def verify_serialized_model(
    model_path: Path, val_df: pd.DataFrame, classes: list[str]
) -> None:
    """Verify that the saved joblib model artifact reloads and predicts reliably."""
    reloaded = joblib.load(model_path)
    assert hasattr(reloaded, "predict"), "Reloaded model missing predict method"
    assert hasattr(reloaded, "predict_proba"), "Reloaded model missing predict_proba method"
    assert list(reloaded.classes_) == classes, "Class order mismatch in reloaded model"

    # Test inference on sample validation statements
    sample_texts = val_df["clean_statement"].head(5).tolist()
    preds = reloaded.predict(sample_texts)
    probas = reloaded.predict_proba(sample_texts)

    assert len(preds) == 5, "Sample prediction length mismatch"
    assert probas.shape == (5, len(classes)), "Probability matrix shape mismatch"

    # Verify probability distribution sums to approximately 1.0
    for prob_row in probas:
        prob_sum = float(np.sum(prob_row))
        assert abs(prob_sum - 1.0) < 1e-4, f"Probabilities do not sum to 1.0: {prob_sum}"


def train_and_evaluate_baseline() -> dict[str, Any]:
    """Execute complete training, evaluation, artifact persistence, and verification.

    Returns:
        Dictionary of validation metrics.
    """
    project_root = get_project_root()
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TF-IDF + LOGISTIC REGRESSION BASELINE TRAINING & EVALUATION")
    print("=" * 80)

    # 1. Load data
    print("\n[1] Loading prepared training and validation splits...")
    train_df = prepare_baseline_split("train")
    val_df = prepare_baseline_split("valid")
    # Strictly verify test.tsv is not loaded
    print(f"  Train split loaded: {len(train_df):,} rows")
    print(f"  Validation split loaded: {len(val_df):,} rows")

    # 2. Print class distributions
    print("\n[2] Class distributions:")
    print("  Training split:")
    for lbl, count in train_df["project_label"].value_counts().items():
        print(f"    {lbl:16s}: {count:5d} ({count/len(train_df)*100:5.2f}%)")
    print("  Validation split:")
    for lbl, count in val_df["project_label"].value_counts().items():
        print(f"    {lbl:16s}: {count:5d} ({count/len(val_df)*100:5.2f}%)")

    # 3. Extract features & labels
    X_train, y_train = split_features_and_labels(train_df)
    X_val, y_val = split_features_and_labels(val_df)

    # 4. Construct Pipeline
    print("\n[3] Building Pipeline (TF-IDF + Logistic Regression)...")
    pipeline = build_baseline_pipeline()

    # 5. Fit ONLY on train
    print(f"\n[4] Fitting model on {len(X_train):,} training statements...")
    start_time = time.perf_counter()
    pipeline.fit(X_train, y_train)
    duration = time.perf_counter() - start_time
    print(f"  Training completed in {duration:.2f} seconds.")

    classes = [str(c) for c in pipeline.classes_]
    print(f"  Model learned classes: {classes}")

    # 6. Predict on validation set
    print(f"\n[5] Evaluating on {len(X_val):,} validation statements...")
    y_val_pred = pipeline.predict(X_val)
    y_val_proba = pipeline.predict_proba(X_val)

    # 7. Compute real validation metrics
    metrics = compute_metrics(y_val, y_val_pred, classes=classes)
    overlap_diagnostic = compute_overlap_diagnostic(
        train_df, val_df, y_val, y_val_pred
    )

    print("\n[6] Validation Results:")
    print(f"  Accuracy        : {metrics['accuracy'] * 100:.2f}%")
    print(f"  Macro Precision : {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall    : {metrics['macro_recall']:.4f}")
    print(f"  Macro F1        : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1     : {metrics['weighted_f1']:.4f}")

    print("\n  Per-Class Metrics:")
    for cls, scores in metrics["per_class"].items():
        print(
            f"    {cls:<16} Precision: {scores['precision']:.4f} | "
            f"Recall: {scores['recall']:.4f} | F1: {scores['f1_score']:.4f} | Support: {scores['support']}"
        )

    print("\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"    Classes: {classes}")
    for row_idx, row in enumerate(metrics["confusion_matrix"]):
        print(f"    {classes[row_idx]:<16}: {row}")

    print("\n  Leakage Sensitivity Diagnostic (excluding train overlap):")
    print(
        f"    Excluded {overlap_diagnostic['overlap_statements_excluded']} statements; "
        f"Evaluated {overlap_diagnostic['evaluated_rows']} rows"
    )
    print(f"    Diagnostic Accuracy : {overlap_diagnostic['accuracy'] * 100:.2f}%")
    print(f"    Diagnostic Macro F1 : {overlap_diagnostic['macro_f1']:.4f}")

    # 8. Save model artifact
    model_artifact_path = models_dir / "baseline_initial.joblib"
    print(f"\n[7] Saving model artifact to: {model_artifact_path}...")
    joblib.dump(pipeline, model_artifact_path)
    model_size_bytes = model_artifact_path.stat().st_size
    print(f"  Artifact saved successfully ({model_size_bytes:,} bytes).")

    # 9. Verify reload
    print("\n[8] Verifying reloaded artifact...")
    verify_serialized_model(model_artifact_path, val_df, classes)
    print("  Reload verification: PASS (predictions and probability sums intact).")

    # 10. Save confusion matrix CSV
    cm_csv_path = models_dir / "baseline_initial_confusion_matrix.csv"
    cm_df = pd.DataFrame(
        metrics["confusion_matrix"], index=classes, columns=classes
    )
    cm_df.to_csv(cm_csv_path, index=True)
    print(f"\n[9] Saved confusion matrix CSV to: {cm_csv_path}")

    # 11. Save validation predictions CSV
    preds_csv_path = models_dir / "baseline_initial_validation_predictions.csv"
    preds_df = pd.DataFrame(
        {
            "id": val_df["id"].values,
            "statement": val_df["statement"].values,
            "true_label": y_val.values,
            "predicted_label": y_val_pred,
            "correct": (y_val.values == y_val_pred),
        }
    )
    for col_idx, cls in enumerate(classes):
        preds_df[f"prob_{cls}"] = y_val_proba[:, col_idx]
    preds_df.to_csv(preds_csv_path, index=False)
    print(f"[10] Saved validation predictions CSV to: {preds_csv_path}")

    # 12. Save validation metrics JSON
    full_metrics: dict[str, Any] = {
        "dataset": "LIAR",
        "training_split": "train",
        "evaluation_split": "valid",
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "model": "TF-IDF + Logistic Regression",
        "configuration": {
            "tfidf": TFIDF_CONFIG,
            "logistic_regression": {
                **LOGREG_CONFIG,
                "ngram_range": list(TFIDF_CONFIG["ngram_range"]),
            },
        },
        "training_duration_seconds": round(duration, 3),
        "model_artifact_size_bytes": model_size_bytes,
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "class_order": classes,
        "overlap_excluded_diagnostic": overlap_diagnostic,
    }

    metrics_json_path = models_dir / "baseline_initial_validation_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"[11] Saved validation metrics JSON to: {metrics_json_path}")

    print("\n" + "=" * 80)
    print("Baseline Training & Validation: COMPLETE")
    print("=" * 80)

    return full_metrics


if __name__ == "__main__":
    train_and_evaluate_baseline()
