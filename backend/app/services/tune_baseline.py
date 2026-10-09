"""Baseline Hyperparameter Tuning Service.

Executes a controlled, validation-based grid search over TF-IDF and Logistic Regression
hyperparameters using ONLY the training split (for fitting) and the validation split
(for model selection).

The official test split (data/raw/test.tsv) is strictly untouched.

Primary selection metric: Macro F1.
Secondary metrics: Accuracy, Macro Precision, Macro Recall, Weighted F1, Per-Class F1.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from backend.app.services.baseline_data import (
    prepare_baseline_split,
    split_features_and_labels,
)
from backend.app.services.dataset_loader import get_project_root
from backend.app.services.train_baseline import (
    compute_metrics,
    compute_overlap_diagnostic,
    verify_serialized_model,
)

# Search space definition: 2 * 3 * 2 * 4 * 2 = 96 configurations
SEARCH_SPACE: dict[str, list[Any]] = {
    "ngram_range": [(1, 1), (1, 2)],
    "min_df": [1, 2, 3],
    "max_features": [30000, 50000],
    "C": [0.5, 1.0, 2.0, 4.0],
    "class_weight": [None, "balanced"],
}

# Fixed parameters
FIXED_CONFIG: dict[str, Any] = {
    "lowercase": True,
    "max_df": 0.95,
    "sublinear_tf": True,
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": 42,
}


def generate_tuning_configurations() -> list[dict[str, Any]]:
    """Generate all 96 discrete hyperparameter configurations without duplicates."""
    keys = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())
    configs: list[dict[str, Any]] = []

    for combination in itertools.product(*values):
        config = dict(zip(keys, combination))
        configs.append(config)

    return configs


def build_candidate_pipeline(config: dict[str, Any]) -> Pipeline:
    """Build a fresh Pipeline for a specific hyperparameter configuration."""
    tfidf_params = {
        "lowercase": FIXED_CONFIG["lowercase"],
        "ngram_range": config["ngram_range"],
        "min_df": config["min_df"],
        "max_df": FIXED_CONFIG["max_df"],
        "max_features": config["max_features"],
        "sublinear_tf": FIXED_CONFIG["sublinear_tf"],
    }

    logreg_params = {
        "max_iter": FIXED_CONFIG["max_iter"],
        "C": config["C"],
        "class_weight": config["class_weight"],
        "solver": FIXED_CONFIG["solver"],
        "random_state": FIXED_CONFIG["random_state"],
    }

    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("classifier", LogisticRegression(**logreg_params)),
        ]
    )


def ranking_key(item: dict[str, Any]) -> tuple[Any, ...]:
    """Sort key for ranking models: primary = Macro F1, tie-breakers = simplicity & speed."""
    ngram_complexity = 1 if item["ngram_range"] == [1, 1] or item["ngram_range"] == (1, 1) else 2
    cw_complexity = 0 if item["class_weight"] is None else 1
    return (
        round(item["macro_f1"], 5),
        round(item["accuracy"], 5),
        -ngram_complexity,
        -item["max_features"],
        item["min_df"],
        -cw_complexity,
        -item["training_seconds"],
    )


def evaluate_single_config(
    pipeline: Pipeline,
    X_train: pd.Series,
    y_train: pd.Series,
    X_val: pd.Series,
    y_val: pd.Series,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Fit a candidate pipeline on train and evaluate on validation set."""
    start_time = time.perf_counter()
    pipeline.fit(X_train, y_train)
    duration = time.perf_counter() - start_time

    classes = [str(c) for c in pipeline.classes_]
    y_val_pred = pipeline.predict(X_val)
    metrics = compute_metrics(y_val, y_val_pred, classes=classes)

    return {
        "ngram_range": list(config["ngram_range"]),
        "min_df": config["min_df"],
        "max_features": config["max_features"],
        "C": config["C"],
        "class_weight": config["class_weight"],
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "training_seconds": round(duration, 3),
        "per_class": metrics["per_class"],
    }


def run_hyperparameter_search(
    raw_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Execute the full 96-configuration hyperparameter search and persist results.

    Returns:
        Dictionary containing summary of the search and best configuration metrics.
    """
    project_root = get_project_root()
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CONTROLLED BASELINE HYPERPARAMETER SEARCH (STEP 6)")
    print("=" * 80)

    # 1. Load data (TRAIN and VALID only; never TEST)
    print("\n[1] Loading prepared train and valid splits (test.tsv is NOT loaded)...")
    train_df = prepare_baseline_split("train", raw_dir=raw_dir)
    val_df = prepare_baseline_split("valid", raw_dir=raw_dir)

    X_train, y_train = split_features_and_labels(train_df)
    X_val, y_val = split_features_and_labels(val_df)

    # 2. Generate search configurations
    configs = generate_tuning_configurations()
    total_configs = len(configs)
    print(f"\n[2] Search space: {total_configs} unique configurations generated.")

    # 3. Iterate and evaluate each candidate
    print(f"\n[3] Executing grid search across {total_configs} configurations...")
    results: list[dict[str, Any]] = []
    overall_start = time.perf_counter()

    for idx, cfg in enumerate(configs, 1):
        pipe = build_candidate_pipeline(cfg)
        res = evaluate_single_config(pipe, X_train, y_train, X_val, y_val, cfg)
        results.append(res)

        if idx % 16 == 0 or idx == total_configs:
            elapsed = time.perf_counter() - overall_start
            print(
                f"  [{idx:2d}/{total_configs}] Completed. Elapsed: {elapsed:.1f}s | "
                f"Latest macro F1: {res['macro_f1']:.4f} (acc: {res['accuracy']:.4f})"
            )

    total_tuning_duration = time.perf_counter() - overall_start
    print(f"\n[4] Grid search completed in {total_tuning_duration:.2f} seconds.")

    # 4. Rank configurations primarily by macro_f1
    ranked_results = sorted(results, key=ranking_key, reverse=True)
    best_result = ranked_results[0]

    print("\n" + "=" * 80)
    print("TOP 5 HYPERPARAMETER CONFIGURATIONS (by Macro F1)")
    print("=" * 80)
    for rank, r in enumerate(ranked_results[:5], 1):
        cw_str = str(r["class_weight"])
        print(
            f"  Rank {rank}: Macro F1 = {r['macro_f1']:.4f} | Acc = {r['accuracy']*100:.2f}% | "
            f"ngrams={r['ngram_range']}, min_df={r['min_df']}, max_feat={r['max_features']}, "
            f"C={r['C']}, class_weight={cw_str} ({r['training_seconds']:.2f}s)"
        )

    # 5. Save full tuning results CSV
    tuning_csv_path = models_dir / "baseline_tuning_results.csv"
    csv_rows = []
    for r in ranked_results:
        csv_rows.append(
            {
                "ngram_range": str(r["ngram_range"]),
                "min_df": r["min_df"],
                "max_features": r["max_features"],
                "C": r["C"],
                "class_weight": str(r["class_weight"]),
                "accuracy": r["accuracy"],
                "macro_precision": r["macro_precision"],
                "macro_recall": r["macro_recall"],
                "macro_f1": r["macro_f1"],
                "weighted_f1": r["weighted_f1"],
                "training_seconds": r["training_seconds"],
            }
        )
    pd.DataFrame(csv_rows).to_csv(tuning_csv_path, index=False)
    print(f"\n[5] Saved tuning results CSV: {tuning_csv_path}")

    # 6. Save full tuning results JSON
    tuning_json_path = models_dir / "baseline_tuning_results.json"
    with open(tuning_json_path, "w", encoding="utf-8") as f:
        json.dump(ranked_results, f, indent=2)
    print(f"[6] Saved tuning results JSON: {tuning_json_path}")

    # 7. Train fresh Pipeline for the best configuration on train split ONLY
    best_config = {
        "ngram_range": tuple(best_result["ngram_range"]),
        "min_df": best_result["min_df"],
        "max_features": best_result["max_features"],
        "C": best_result["C"],
        "class_weight": best_result["class_weight"],
    }
    print(f"\n[7] Training fresh Pipeline with best configuration: {best_config}")
    best_pipeline = build_candidate_pipeline(best_config)
    refit_start = time.perf_counter()
    best_pipeline.fit(X_train, y_train)
    refit_duration = time.perf_counter() - refit_start

    classes = [str(c) for c in best_pipeline.classes_]

    # Evaluate best model
    y_val_pred = best_pipeline.predict(X_val)
    y_val_proba = best_pipeline.predict_proba(X_val)

    best_metrics = compute_metrics(y_val, y_val_pred, classes=classes)
    best_diagnostic = compute_overlap_diagnostic(
        train_df, val_df, y_val, y_val_pred
    )

    # 8. Save best model artifact
    best_model_path = models_dir / "baseline_best_validation.joblib"
    joblib.dump(best_pipeline, best_model_path)
    best_model_size = best_model_path.stat().st_size
    print(f"[8] Saved best model artifact: {best_model_path} ({best_model_size:,} bytes)")

    # 9. Verify reloaded best artifact
    verify_serialized_model(best_model_path, val_df, classes)
    print("  Reload verification: PASS")

    # 10. Save confusion matrix CSV
    best_cm_csv_path = models_dir / "baseline_best_validation_confusion_matrix.csv"
    best_cm_df = pd.DataFrame(
        best_metrics["confusion_matrix"], index=classes, columns=classes
    )
    best_cm_df.to_csv(best_cm_csv_path, index=True)
    print(f"[9] Saved best confusion matrix CSV: {best_cm_csv_path}")

    # 11. Save predictions CSV
    best_preds_csv_path = models_dir / "baseline_best_validation_predictions.csv"
    best_preds_df = pd.DataFrame(
        {
            "id": val_df["id"].values,
            "statement": val_df["statement"].values,
            "true_label": y_val.values,
            "predicted_label": y_val_pred,
            "correct": (y_val.values == y_val_pred),
        }
    )
    for col_idx, cls in enumerate(classes):
        best_preds_df[f"prob_{cls}"] = y_val_proba[:, col_idx]
    best_preds_df.to_csv(best_preds_csv_path, index=False)
    print(f"[10] Saved best predictions CSV: {best_preds_csv_path}")

    # 12. Load Step 5 initial baseline metrics for comparison
    initial_metrics_path = models_dir / "baseline_initial_validation_metrics.json"
    initial_metrics: dict[str, Any] = {}
    if initial_metrics_path.is_file():
        with open(initial_metrics_path, "r", encoding="utf-8") as f:
            initial_metrics = json.load(f)

    initial_macro_f1 = initial_metrics.get("macro_f1", 0.4465)
    initial_acc = initial_metrics.get("accuracy", 0.4564)
    macro_f1_delta = round(best_metrics["macro_f1"] - initial_macro_f1, 4)
    acc_delta = round(best_metrics["accuracy"] - initial_acc, 4)

    # Error analysis calculations
    cm = best_metrics["confusion_matrix"]
    confusion_pairs = []
    for i, true_cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            if i != j:
                confusion_pairs.append(((true_cls, pred_cls), cm[i][j]))
    most_common_pair = max(confusion_pairs, key=lambda x: x[1])

    lowest_recall_cls = min(
        best_metrics["per_class"].items(), key=lambda x: x[1]["recall"]
    )[0]
    lowest_f1_cls = min(
        best_metrics["per_class"].items(), key=lambda x: x[1]["f1_score"]
    )[0]

    # Analyze effect of class_weight
    balanced_f1s = [r["macro_f1"] for r in results if r["class_weight"] == "balanced"]
    unweighted_f1s = [r["macro_f1"] for r in results if r["class_weight"] is None]
    avg_balanced_f1 = float(np.mean(balanced_f1s))
    avg_unweighted_f1 = float(np.mean(unweighted_f1s))
    cw_effect = (
        f"Mean Macro F1 with class_weight='balanced': {avg_balanced_f1:.4f} vs "
        f"unweighted: {avg_unweighted_f1:.4f}. "
        f"Balanced weighting {'improved' if avg_balanced_f1 > avg_unweighted_f1 else 'decreased'} average Macro F1."
    )

    error_analysis = {
        "most_common_confusion_pair": {
            "true_label": most_common_pair[0][0],
            "predicted_label": most_common_pair[0][1],
            "count": most_common_pair[1],
        },
        "lowest_recall_class": lowest_recall_cls,
        "lowest_f1_class": lowest_f1_cls,
        "class_weight_effect": cw_effect,
    }

    comparison = {
        "initial_macro_f1": initial_macro_f1,
        "best_macro_f1": best_metrics["macro_f1"],
        "macro_f1_improvement": macro_f1_delta,
        "initial_accuracy": initial_acc,
        "best_accuracy": best_metrics["accuracy"],
        "accuracy_improvement": acc_delta,
    }

    # 13. Save best validation metrics JSON
    best_full_metrics = {
        "dataset": "LIAR",
        "training_split": "train",
        "evaluation_split": "valid",
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "model": "TF-IDF + Logistic Regression (Validation-Tuned)",
        "selection_metric": "macro_f1",
        "total_configurations_tested": total_configs,
        "total_tuning_duration_seconds": round(total_tuning_duration, 2),
        "best_configuration": best_result,
        "accuracy": best_metrics["accuracy"],
        "macro_precision": best_metrics["macro_precision"],
        "macro_recall": best_metrics["macro_recall"],
        "macro_f1": best_metrics["macro_f1"],
        "weighted_f1": best_metrics["weighted_f1"],
        "per_class": best_metrics["per_class"],
        "confusion_matrix": best_metrics["confusion_matrix"],
        "class_order": classes,
        "overlap_excluded_diagnostic": best_diagnostic,
        "comparison_to_initial_baseline": comparison,
        "error_analysis": error_analysis,
    }

    best_json_path = models_dir / "baseline_best_validation_metrics.json"
    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump(best_full_metrics, f, indent=2)
    print(f"[11] Saved best validation metrics JSON: {best_json_path}")

    print("\n" + "=" * 80)
    print("HYPERPARAMETER SELECTION SUMMARY")
    print("=" * 80)
    print(f"  Initial Macro F1 : {initial_macro_f1:.4f} (Accuracy: {initial_acc*100:.2f}%)")
    print(f"  Best Macro F1    : {best_metrics['macro_f1']:.4f} (Accuracy: {best_metrics['accuracy']*100:.2f}%)")
    print(f"  Macro F1 Delta   : {macro_f1_delta:+.4f}")
    print(f"  Accuracy Delta   : {acc_delta:+.4f}")
    print(f"  Most common error: Actual '{most_common_pair[0][0]}' -> Predicted '{most_common_pair[0][1]}' ({most_common_pair[1]} instances)")
    print(f"  Class weight note: {cw_effect}")
    print("=" * 80)

    return best_full_metrics
