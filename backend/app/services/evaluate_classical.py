"""Service for One-Time Classical Model Evaluation on Untouched LIAR Test Split (Step 7).

Evaluates the frozen classical baseline pipeline (models/classical_best_validation.joblib)
strictly on the official LIAR test split (data/raw/test.tsv).

STRICT POLICIES:
1. Zero model fitting: .fit() and .partial_fit() are NEVER called.
2. The model artifact is loaded frozen from disk.
3. Official test metrics evaluate all 1,283 test rows without omission.
4. Overlap-excluded diagnostic is computed as a secondary metric (1,279 rows).
5. Descriptive probability analysis is performed (explicitly not assumed to be calibrated).
6. 'unverifiable' is never emitted by the classical baseline.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from backend.app.services.baseline_data import prepare_baseline_split
from backend.app.services.dataset_loader import (
    ALLOWED_PROJECT_LABELS,
    get_project_root,
)

# Reference validation performance from Step 6 / 6.5
VALIDATION_REF_ACCURACY: float = 0.4688
VALIDATION_REF_MACRO_F1: float = 0.4687


def load_frozen_classical_artifact(
    model_path: Optional[Path] = None,
) -> Pipeline:
    """Load and validate the frozen classical model artifact.

    Args:
        model_path: Optional custom path to classical_best_validation.joblib.

    Returns:
        Loaded and verified scikit-learn Pipeline.

    Raises:
        FileNotFoundError: If the model artifact does not exist.
        TypeError: If the artifact is not a Pipeline or lacks classes_/predict_proba.
    """
    if model_path is None:
        model_path = get_project_root() / "models" / "classical_best_validation.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Frozen model artifact not found at {model_path}. "
            "Step 6.5 model selection must be completed first."
        )

    pipeline = joblib.load(model_path)

    if not hasattr(pipeline, "predict"):
        raise TypeError(f"Loaded object from {model_path} does not implement predict.")
    if not hasattr(pipeline, "predict_proba"):
        raise TypeError(f"Loaded object from {model_path} does not implement predict_proba.")
    if not hasattr(pipeline, "classes_"):
        raise TypeError(f"Loaded object from {model_path} does not expose classes_.")

    # Verify classes match expected project labels
    if set(pipeline.classes_) != ALLOWED_PROJECT_LABELS:
        raise ValueError(
            f"Model classes {pipeline.classes_} do not match allowed project labels {ALLOWED_PROJECT_LABELS}."
        )

    return pipeline


def compute_test_metrics(
    y_true: list[str] | np.ndarray,
    y_pred: list[str] | np.ndarray,
    classes: list[str],
) -> dict[str, Any]:
    """Compute comprehensive test classification metrics with zero_division=0.

    Args:
        y_true: Ground truth project labels.
        y_pred: Predicted project labels.
        classes: Ordered list of class labels from model.classes_.

    Returns:
        Structured metrics dictionary.
    """
    acc = float(accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    report = classification_report(
        y_true, y_pred, labels=classes, output_dict=True, zero_division=0
    )

    per_class: dict[str, dict[str, Any]] = {}
    for cls in classes:
        if cls in report:
            per_class[cls] = {
                "precision": round(float(report[cls]["precision"]), 4),
                "recall": round(float(report[cls]["recall"]), 4),
                "f1_score": round(float(report[cls]["f1-score"]), 4),
                "support": int(report[cls]["support"]),
            }

    cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "class_order": classes,
    }


def compute_probability_diagnostic(
    probs: np.ndarray,
    is_correct: np.ndarray,
) -> dict[str, Any]:
    """Perform descriptive probability output diagnostic.

    Args:
        probs: Array of predicted class probability estimates, shape (N, num_classes).
        is_correct: Boolean array indicating whether prediction was correct.

    Returns:
        Descriptive statistics dictionary for max-probability outputs.
    """
    max_probs = np.max(probs, axis=1)

    correct_max = max_probs[is_correct]
    incorrect_max = max_probs[~is_correct]

    return {
        "diagnostic_note": (
            "Descriptive model-output diagnostic. Probability estimates returned by "
            "Logistic Regression are not assumed to be calibrated."
        ),
        "correct_predictions": {
            "count": int(len(correct_max)),
            "mean_max_probability": round(float(np.mean(correct_max)), 4),
            "median_max_probability": round(float(np.median(correct_max)), 4),
            "min_max_probability": round(float(np.min(correct_max)), 4),
            "max_max_probability": round(float(np.max(correct_max)), 4),
        },
        "incorrect_predictions": {
            "count": int(len(incorrect_max)),
            "mean_max_probability": round(float(np.mean(incorrect_max)), 4),
            "median_max_probability": round(float(np.median(incorrect_max)), 4),
            "min_max_probability": round(float(np.min(incorrect_max)), 4),
            "max_max_probability": round(float(np.max(incorrect_max)), 4),
        },
    }


def compute_test_overlap_diagnostic(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_test: list[str] | np.ndarray,
    y_test_pred: list[str] | np.ndarray,
    classes: list[str],
) -> dict[str, Any]:
    """Compute secondary diagnostic excluding test statements appearing verbatim in train.

    Args:
        train_df: Prepared training DataFrame.
        test_df: Prepared test DataFrame.
        y_test: Actual test labels.
        y_test_pred: Predicted test labels.
        classes: Ordered list of class labels.

    Returns:
        Diagnostic metrics dictionary.
    """
    train_statements = set(train_df["clean_statement"].str.strip())
    test_statements = test_df["clean_statement"].str.strip()

    is_overlap = test_statements.isin(train_statements).to_numpy()
    overlap_count = int(is_overlap.sum())
    diag_mask = ~is_overlap

    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_test_pred)

    diag_y_true = y_test_arr[diag_mask]
    diag_y_pred = y_pred_arr[diag_mask]

    return {
        "diagnostic_label": "Secondary Leakage Diagnostic (Exact Train Overlaps Excluded)",
        "overlap_statements_excluded": overlap_count,
        "evaluated_rows": int(len(diag_y_true)),
        "accuracy": round(float(accuracy_score(diag_y_true, diag_y_pred)), 4),
        "macro_precision": round(
            float(precision_score(diag_y_true, diag_y_pred, average="macro", zero_division=0)),
            4,
        ),
        "macro_recall": round(
            float(recall_score(diag_y_true, diag_y_pred, average="macro", zero_division=0)),
            4,
        ),
        "macro_f1": round(
            float(f1_score(diag_y_true, diag_y_pred, average="macro", zero_division=0)), 4
        ),
    }


def evaluate_frozen_classical_model(
    model_path: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Execute complete, one-time final test evaluation of the frozen classical model.

    ZERO MODEL FITTING IS PERFORMED.

    Args:
        model_path: Optional path to the frozen model artifact.
        raw_dir: Optional path to data/raw directory.
        output_dir: Optional path to models/ directory.

    Returns:
        Structured dictionary containing all test metrics, diagnostics, and comparisons.
    """
    project_root = get_project_root()
    if output_dir is None:
        output_dir = project_root / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FINAL CLASSICAL MODEL TEST EVALUATION (STEP 7)")
    print("=" * 80)

    # 1. Load frozen model artifact
    print("\n[1] Loading frozen classical model artifact...")
    pipeline = load_frozen_classical_artifact(model_path)
    classes = [str(c) for c in pipeline.classes_]
    print(f"  Model loaded: {pipeline}")
    print(f"  Learned class order: {classes}")

    # Record model parameter state before evaluation to verify zero mutation
    classifier_step = pipeline.named_steps["classifier"]
    coef_before = np.copy(classifier_step.coef_)
    intercept_before = np.copy(classifier_step.intercept_)

    # 2. Load official LIAR test split (first time in project)
    print("\n[2] Loading official LIAR test split (data/raw/test.tsv)...")
    test_df = prepare_baseline_split("test", raw_dir=raw_dir)
    test_rows = len(test_df)
    print(f"  Test set loaded: {test_rows:,} rows")

    X_test = test_df["clean_statement"]
    y_test = test_df["project_label"].tolist()

    # Actual test class distribution
    test_dist = test_df["project_label"].value_counts().to_dict()
    print("  Actual test class distribution:")
    for cls, count in test_dist.items():
        pct = (count / test_rows) * 100
        print(f"    - {cls:<16}: {count:4d} ({pct:.2f}%)")

    # 3. Generate predictions and probability estimates (INFERENCE ONLY)
    print("\n[3] Generating predictions and class probability estimates (ZERO FITTING)...")
    infer_start = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    y_pred = [str(p) for p in y_pred]
    probs = pipeline.predict_proba(X_test)
    infer_duration = time.perf_counter() - infer_start
    print(f"  Inference completed in {infer_duration:.3f} seconds.")

    # Guard: classical baseline must never emit 'unverifiable'
    for pred in y_pred:
        if pred == "unverifiable":
            raise RuntimeError(
                "Classical baseline illegally emitted 'unverifiable'. "
                "Abstention is reserved for downstream fusion logic."
            )
        if pred not in ALLOWED_PROJECT_LABELS:
            raise ValueError(f"Predicted illegal label '{pred}'. Allowed: {ALLOWED_PROJECT_LABELS}")

    # Verify model parameters were NOT modified during inference
    np.testing.assert_array_equal(
        coef_before,
        classifier_step.coef_,
        err_msg="Model coefficients mutated during test evaluation!",
    )
    np.testing.assert_array_equal(
        intercept_before,
        classifier_step.intercept_,
        err_msg="Model intercept mutated during test evaluation!",
    )

    # 4. Map probability columns matching model.classes_
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    prob_fake = probs[:, class_to_idx["likely_fake"]]
    prob_reliable = probs[:, class_to_idx["likely_reliable"]]
    prob_misleading = probs[:, class_to_idx["misleading"]]

    # 5. Compute official test metrics
    print("\n[4] Computing official test metrics...")
    metrics = compute_test_metrics(y_test, y_pred, classes=classes)

    # Prediction distribution
    pred_series = pd.Series(y_pred)
    pred_dist = pred_series.value_counts().to_dict()
    print("  Predicted class distribution:")
    for cls in classes:
        count = pred_dist.get(cls, 0)
        pct = (count / test_rows) * 100
        print(f"    - {cls:<16}: {count:4d} ({pct:.2f}%)")

    # 6. Save test predictions CSV
    predictions_df = pd.DataFrame(
        {
            "id": test_df["id"],
            "statement": test_df["statement"],
            "true_label": y_test,
            "predicted_label": y_pred,
            "correct": [t == p for t, p in zip(y_test, y_pred)],
            "prob_likely_fake": [round(float(p), 6) for p in prob_fake],
            "prob_likely_reliable": [round(float(p), 6) for p in prob_reliable],
            "prob_misleading": [round(float(p), 6) for p in prob_misleading],
        }
    )
    predictions_csv_path = output_dir / "classical_test_predictions.csv"
    predictions_df.to_csv(predictions_csv_path, index=False)
    print(f"\n[5] Saved test predictions CSV to: {predictions_csv_path}")

    # 7. Save confusion matrix CSV
    cm_df = pd.DataFrame(metrics["confusion_matrix"], index=classes, columns=classes)
    cm_csv_path = output_dir / "classical_test_confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path, index=True)
    print(f"[6] Saved confusion matrix CSV to: {cm_csv_path}")

    # 8. Probability output diagnostic
    is_correct_arr = np.array([t == p for t, p in zip(y_test, y_pred)])
    prob_diagnostic = compute_probability_diagnostic(probs, is_correct_arr)

    # 9. Overlap-excluded diagnostic
    train_df = prepare_baseline_split("train", raw_dir=raw_dir)
    overlap_diagnostic = compute_test_overlap_diagnostic(
        train_df, test_df, y_test, y_pred, classes=classes
    )

    # 10. Validation vs Test Comparison
    delta_acc = metrics["accuracy"] - VALIDATION_REF_ACCURACY
    delta_f1 = metrics["macro_f1"] - VALIDATION_REF_MACRO_F1

    # Error statistics
    cm = np.array(metrics["confusion_matrix"])
    confusion_pairs = {}
    for i, true_cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            if i != j:
                confusion_pairs[f"{true_cls} -> {pred_cls}"] = int(cm[i, j])
    most_frequent_confusion = max(confusion_pairs.items(), key=lambda x: x[1])

    lowest_prec_cls = min(classes, key=lambda c: metrics["per_class"][c]["precision"])
    lowest_rec_cls = min(classes, key=lambda c: metrics["per_class"][c]["recall"])
    lowest_f1_cls = min(classes, key=lambda c: metrics["per_class"][c]["f1_score"])
    most_pred_cls = max(pred_dist.items(), key=lambda x: x[1])[0]
    least_pred_cls = min(pred_dist.items(), key=lambda x: x[1])[0]

    error_stats = {
        "most_frequent_confusion_pair": {
            "pair": most_frequent_confusion[0],
            "count": most_frequent_confusion[1],
        },
        "all_confusion_pairs": confusion_pairs,
        "class_with_lowest_precision": {
            "class": lowest_prec_cls,
            "precision": metrics["per_class"][lowest_prec_cls]["precision"],
        },
        "class_with_lowest_recall": {
            "class": lowest_rec_cls,
            "recall": metrics["per_class"][lowest_rec_cls]["recall"],
        },
        "class_with_lowest_f1": {
            "class": lowest_f1_cls,
            "f1_score": metrics["per_class"][lowest_f1_cls]["f1_score"],
        },
        "most_frequently_predicted_class": {
            "class": most_pred_cls,
            "count": pred_dist[most_pred_cls],
        },
        "least_frequently_predicted_class": {
            "class": least_pred_cls,
            "count": pred_dist[least_pred_cls],
        },
    }

    # 11. Compile complete test metadata JSON
    test_metadata = {
        "dataset": "LIAR",
        "split": "test",
        "model_artifact": "models/classical_best_validation.joblib",
        "test_rows": test_rows,
        "model": "Word TF-IDF + Logistic Regression",
        "configuration": {
            "tfidf": {
                "lowercase": True,
                "ngram_range": [1, 2],
                "min_df": 1,
                "max_df": 0.95,
                "max_features": 50000,
                "sublinear_tf": True,
            },
            "classifier": {
                "type": "LogisticRegression",
                "C": 0.5,
                "class_weight": "balanced",
                "solver": "lbfgs",
                "max_iter": 1000,
                "random_state": 42,
            },
        },
        "inference_duration_seconds": round(infer_duration, 4),
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "validation_reference": {
            "validation_accuracy": VALIDATION_REF_ACCURACY,
            "validation_macro_f1": VALIDATION_REF_MACRO_F1,
            "delta_accuracy": round(delta_acc, 4),
            "delta_macro_f1": round(delta_f1, 4),
        },
        "actual_test_distribution": test_dist,
        "predicted_test_distribution": pred_dist,
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "class_order": classes,
        "probability_diagnostic": prob_diagnostic,
        "overlap_excluded_diagnostic": overlap_diagnostic,
        "error_statistics": error_stats,
    }

    metrics_json_path = output_dir / "classical_test_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(test_metadata, f, indent=2)
    print(f"[7] Saved test metrics JSON to: {metrics_json_path}")

    # Print summary to console
    print("\n" + "=" * 80)
    print("OFFICIAL LIAR TEST EVALUATION RESULTS")
    print("=" * 80)
    print(f"  Test Accuracy   : {metrics['accuracy']*100:.2f}% (Validation: {VALIDATION_REF_ACCURACY*100:.2f}%, delta: {delta_acc*100:+.2f}%)")
    print(f"  Test Macro F1   : {metrics['macro_f1']:.4f} (Validation: {VALIDATION_REF_MACRO_F1:.4f}, delta: {delta_f1:+.4f})")
    print(f"  Macro Precision : {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall    : {metrics['macro_recall']:.4f}")
    print(f"  Weighted F1     : {metrics['weighted_f1']:.4f}")
    print("\n  Per-Class Metrics:")
    for cls in classes:
        pc = metrics["per_class"][cls]
        print(f"    - {cls:<16}: Prec={pc['precision']:.4f}, Rec={pc['recall']:.4f}, F1={pc['f1_score']:.4f} (Support={pc['support']})")
    print("=" * 80)

    return test_metadata
