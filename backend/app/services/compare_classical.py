"""Classical Model Comparison Service (Step 6.5).

Compares the current tuned Logistic Regression baseline against:
1. LinearSVC with word TF-IDF (Model B)
2. Word + character TF-IDF with Logistic Regression (Model C)
3. Word + character TF-IDF with LinearSVC (Model D)

All fitting is performed ONLY on train.tsv.
All comparison and selection is performed ONLY on valid.tsv.
The official test split (data/raw/test.tsv) is strictly untouched.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from backend.app.services.baseline_data import (
    prepare_baseline_split,
    split_features_and_labels,
)
from backend.app.services.dataset_loader import (
    ALLOWED_PROJECT_LABELS,
    get_project_root,
)
from backend.app.services.train_baseline import (
    compute_metrics,
    compute_overlap_diagnostic,
)

# Reference metrics from Step 6 tuned Logistic Regression
REFERENCE_MACRO_F1: float = 0.4687
REFERENCE_ACCURACY: float = 0.4688
MIN_MEANINGFUL_IMPROVEMENT: float = 0.0020


def generate_comparison_configurations() -> list[dict[str, Any]]:
    """Generate all discrete candidate model configurations for comparison.

    Includes:
    - Model A: Current Tuned Logistic Regression (Step 6 baseline reference)
    - Model B: Word TF-IDF + LinearSVC (6 configs: C in [0.5, 1.0, 2.0], class_weight in [None, 'balanced'])
    - Model C: Word + Char TF-IDF + Logistic Regression (8 configs: char_max_features in [30000, 50000], C in [0.5, 1.0], class_weight in ['balanced', None])
    - Model D: Word + Char TF-IDF + LinearSVC (8 configs: char_max_features in [30000, 50000], C in [0.5, 1.0], class_weight in [None, 'balanced'])
    """
    configs: list[dict[str, Any]] = []

    # Model A: Step 6 reference configuration
    configs.append(
        {
            "model_name": "Tuned Logistic Regression (Step 6)",
            "model_group": "A",
            "classifier": "LogisticRegression",
            "word_ngram_range": [1, 2],
            "word_max_features": 50000,
            "char_enabled": False,
            "char_ngram_range": None,
            "char_max_features": None,
            "C": 0.5,
            "class_weight": "balanced",
        }
    )

    # Model B: Word TF-IDF + LinearSVC
    for c_val in [0.5, 1.0, 2.0]:
        for cw in [None, "balanced"]:
            configs.append(
                {
                    "model_name": "Word TF-IDF + LinearSVC",
                    "model_group": "B",
                    "classifier": "LinearSVC",
                    "word_ngram_range": [1, 2],
                    "word_max_features": 50000,
                    "char_enabled": False,
                    "char_ngram_range": None,
                    "char_max_features": None,
                    "C": c_val,
                    "class_weight": cw,
                }
            )

    # Model C: Word + Char TF-IDF + Logistic Regression
    for char_max in [30000, 50000]:
        for c_val in [0.5, 1.0]:
            for cw in ["balanced", None]:
                configs.append(
                    {
                        "model_name": "Word + Char TF-IDF + LogisticRegression",
                        "model_group": "C",
                        "classifier": "LogisticRegression",
                        "word_ngram_range": [1, 2],
                        "word_max_features": 50000,
                        "char_enabled": True,
                        "char_ngram_range": [3, 5],
                        "char_max_features": char_max,
                        "C": c_val,
                        "class_weight": cw,
                    }
                )

    # Model D: Word + Char TF-IDF + LinearSVC
    for char_max in [30000, 50000]:
        for c_val in [0.5, 1.0]:
            for cw in [None, "balanced"]:
                configs.append(
                    {
                        "model_name": "Word + Char TF-IDF + LinearSVC",
                        "model_group": "D",
                        "classifier": "LinearSVC",
                        "word_ngram_range": [1, 2],
                        "word_max_features": 50000,
                        "char_enabled": True,
                        "char_ngram_range": [3, 5],
                        "char_max_features": char_max,
                        "C": c_val,
                        "class_weight": cw,
                    }
                )

    return configs


def build_comparison_pipeline(config: dict[str, Any]) -> Pipeline:
    """Build a scikit-learn Pipeline corresponding to a comparison configuration."""
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        lowercase=True,
        ngram_range=tuple(config["word_ngram_range"]),
        min_df=1,
        max_df=0.95,
        max_features=config["word_max_features"],
        sublinear_tf=True,
    )

    if config.get("char_enabled", False):
        char_tfidf = TfidfVectorizer(
            analyzer="char_wb",
            lowercase=True,
            ngram_range=tuple(config["char_ngram_range"]),
            min_df=2,
            max_features=config["char_max_features"],
            sublinear_tf=True,
        )
        features: Any = FeatureUnion(
            [
                ("word", word_tfidf),
                ("char", char_tfidf),
            ]
        )
    else:
        features = word_tfidf

    if config["classifier"] == "LogisticRegression":
        clf: Any = LogisticRegression(
            C=config["C"],
            class_weight=config["class_weight"],
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )
    elif config["classifier"] == "LinearSVC":
        clf = LinearSVC(
            C=config["C"],
            class_weight=config["class_weight"],
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported classifier: {config['classifier']}")

    return Pipeline(
        [
            ("features", features),
            ("classifier", clf),
        ]
    )


def evaluate_candidate(
    pipeline: Pipeline,
    X_train: pd.Series,
    y_train: pd.Series,
    X_val: pd.Series,
    y_val: pd.Series,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Fit a candidate pipeline on train and evaluate on validation set."""
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fit_duration = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_val_pred = pipeline.predict(X_val)
    pred_duration = time.perf_counter() - t1

    classes = [str(c) for c in pipeline.classes_]
    metrics = compute_metrics(y_val, y_val_pred, classes=classes)

    macro_f1 = metrics["macro_f1"]
    delta_f1 = round(macro_f1 - REFERENCE_MACRO_F1, 4)

    has_proba = hasattr(pipeline, "predict_proba")

    return {
        "model_name": config["model_name"],
        "model_group": config["model_group"],
        "classifier": config["classifier"],
        "word_ngram_range": config["word_ngram_range"],
        "word_max_features": config["word_max_features"],
        "char_enabled": config["char_enabled"],
        "char_ngram_range": config["char_ngram_range"],
        "char_max_features": config["char_max_features"],
        "C": config["C"],
        "class_weight": config["class_weight"],
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": macro_f1,
        "weighted_f1": metrics["weighted_f1"],
        "delta_macro_f1": delta_f1,
        "training_seconds": round(fit_duration, 3),
        "prediction_seconds": round(pred_duration, 3),
        "native_probabilities": has_proba,
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "class_order": classes,
    }


def select_best_model(
    results: list[dict[str, Any]],
    reference_model_config: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Select the best classical model based on Macro F1 and engineering trade-offs.

    Rule:
    - Primary metric: Macro F1 descending.
    - If highest Macro F1 candidate beats reference by < MIN_MEANINGFUL_IMPROVEMENT (0.002),
      or if tied, retain the reference Tuned Logistic Regression (simpler model, native probabilities).
    - If a candidate beats reference by >= 0.002, select that candidate.

    Returns:
        Tuple of (selected_result_dict, justification_string).
    """
    ranked = sorted(results, key=lambda x: x["macro_f1"], reverse=True)
    top_candidate = ranked[0]

    delta = top_candidate["macro_f1"] - REFERENCE_MACRO_F1

    if delta < MIN_MEANINGFUL_IMPROVEMENT:
        # Reference is selected due to simplicity & native probabilities
        ref_results = [
            r for r in results if r["model_group"] == "A"
        ]
        chosen = ref_results[0] if ref_results else top_candidate
        justification = (
            f"Highest candidate '{top_candidate['model_name']}' achieved Macro F1 = "
            f"{top_candidate['macro_f1']:.4f} (delta = {delta:+.4f} vs reference {REFERENCE_MACRO_F1:.4f}). "
            f"Because this improvement is below the meaningful threshold ({MIN_MEANINGFUL_IMPROVEMENT:.4f}), "
            f"the simpler Tuned Logistic Regression model was retained for lower complexity and native probability support."
        )
        return chosen, justification
    else:
        chosen = top_candidate
        justification = (
            f"Candidate '{chosen['model_name']}' achieved Macro F1 = {chosen['macro_f1']:.4f} "
            f"(delta = {delta:+.4f} vs reference {REFERENCE_MACRO_F1:.4f}), exceeding the "
            f"minimum meaningful threshold (+{MIN_MEANINGFUL_IMPROVEMENT:.4f}). Selected as best model."
        )
        return chosen, justification


def run_classical_comparison(
    raw_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Execute complete classical model comparison and persist results."""
    project_root = get_project_root()
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CLASSICAL MODEL COMPARISON (STEP 6.5)")
    print("=" * 80)

    # 1. Load train and valid data ONLY
    print("\n[1] Loading train and valid splits (data/raw/test.tsv is strictly NOT loaded)...")
    train_df = prepare_baseline_split("train", raw_dir=raw_dir)
    val_df = prepare_baseline_split("valid", raw_dir=raw_dir)

    X_train, y_train = split_features_and_labels(train_df)
    X_val, y_val = split_features_and_labels(val_df)

    print(f"  Training set size   : {len(X_train):,} rows")
    print(f"  Validation set size : {len(X_val):,} rows")

    # 2. Generate comparison configurations
    configs = generate_comparison_configurations()
    total_configs = len(configs)
    print(f"\n[2] Generated {total_configs} candidate configurations across 4 model families:")
    print("  - Model A: Tuned Logistic Regression (Step 6 reference)")
    print("  - Model B: Word TF-IDF + LinearSVC (6 configs)")
    print("  - Model C: Word + Char TF-IDF + LogisticRegression (8 configs)")
    print("  - Model D: Word + Char TF-IDF + LinearSVC (8 configs)")

    # 3. Evaluate each configuration
    print(f"\n[3] Fitting and evaluating all {total_configs} candidates...")
    results: list[dict[str, Any]] = []
    overall_start = time.perf_counter()

    for idx, cfg in enumerate(configs, 1):
        pipe = build_comparison_pipeline(cfg)
        res = evaluate_candidate(pipe, X_train, y_train, X_val, y_val, cfg)
        results.append(res)
        cw_str = str(cfg["class_weight"])
        char_info = f"char={cfg['char_max_features']}" if cfg["char_enabled"] else "word_only"
        print(
            f"  [{idx:2d}/{total_configs}] {cfg['model_group']}: {cfg['classifier']:<18} "
            f"C={cfg['C']:<3} cw={cw_str:<8} {char_info:<12} -> "
            f"Macro F1: {res['macro_f1']:.4f} (Acc: {res['accuracy']*100:.2f}%) [{res['training_seconds']:.2f}s]"
        )

    total_duration = time.perf_counter() - overall_start
    print(f"\n[4] All evaluations completed in {total_duration:.2f} seconds.")

    # 4. Save comparison CSV
    csv_rows = []
    for r in results:
        csv_rows.append(
            {
                "model_name": r["model_name"],
                "model_group": r["model_group"],
                "classifier": r["classifier"],
                "word_ngram_range": str(r["word_ngram_range"]),
                "word_max_features": r["word_max_features"],
                "char_enabled": r["char_enabled"],
                "char_ngram_range": str(r["char_ngram_range"]),
                "char_max_features": r["char_max_features"],
                "C": r["C"],
                "class_weight": str(r["class_weight"]),
                "accuracy": r["accuracy"],
                "macro_precision": r["macro_precision"],
                "macro_recall": r["macro_recall"],
                "macro_f1": r["macro_f1"],
                "weighted_f1": r["weighted_f1"],
                "delta_macro_f1": r["delta_macro_f1"],
                "training_seconds": r["training_seconds"],
                "prediction_seconds": r["prediction_seconds"],
                "native_probabilities": r["native_probabilities"],
                "likely_fake_f1": r["per_class"]["likely_fake"]["f1_score"],
                "likely_reliable_f1": r["per_class"]["likely_reliable"]["f1_score"],
                "misleading_f1": r["per_class"]["misleading"]["f1_score"],
            }
        )
    comp_csv_path = models_dir / "classical_model_comparison.csv"
    pd.DataFrame(csv_rows).to_csv(comp_csv_path, index=False)
    print(f"\n[5] Saved comparison CSV to: {comp_csv_path}")

    # 5. Save comparison JSON
    comp_json_path = models_dir / "classical_model_comparison.json"
    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[6] Saved comparison JSON to: {comp_json_path}")

    # 6. Model Selection
    selected_result, justification = select_best_model(results, configs[0])

    print("\n" + "=" * 80)
    print("MODEL SELECTION DECISION")
    print("=" * 80)
    print(f"  Selected Model: {selected_result['model_name']}")
    print(f"  Classifier    : {selected_result['classifier']}")
    print(f"  Configuration : C={selected_result['C']}, class_weight={selected_result['class_weight']}, char_enabled={selected_result['char_enabled']}")
    print(f"  Macro F1      : {selected_result['macro_f1']:.4f} (vs reference {REFERENCE_MACRO_F1:.4f}, delta: {selected_result['delta_macro_f1']:+.4f})")
    print(f"  Accuracy      : {selected_result['accuracy']*100:.2f}% (vs reference {REFERENCE_ACCURACY*100:.2f}%)")
    print(f"  Justification : {justification}")
    print("=" * 80)

    # 7. Fit fresh selected Pipeline on train split ONLY and serialize
    print(f"\n[7] Fitting fresh selected pipeline on train set...")
    selected_cfg = {
        "word_ngram_range": selected_result["word_ngram_range"],
        "word_max_features": selected_result["word_max_features"],
        "char_enabled": selected_result["char_enabled"],
        "char_ngram_range": selected_result["char_ngram_range"],
        "char_max_features": selected_result["char_max_features"],
        "classifier": selected_result["classifier"],
        "C": selected_result["C"],
        "class_weight": selected_result["class_weight"],
    }
    selected_pipe = build_comparison_pipeline(selected_cfg)
    fit_start = time.perf_counter()
    selected_pipe.fit(X_train, y_train)
    fit_time = time.perf_counter() - fit_start

    classes = [str(c) for c in selected_pipe.classes_]
    y_val_pred = selected_pipe.predict(X_val)

    selected_metrics = compute_metrics(y_val, y_val_pred, classes=classes)
    selected_diagnostic = compute_overlap_diagnostic(
        train_df, val_df, y_val, y_val_pred
    )

    # Save selected artifact
    selected_artifact_path = models_dir / "classical_best_validation.joblib"
    joblib.dump(selected_pipe, selected_artifact_path)
    artifact_size = selected_artifact_path.stat().st_size
    print(f"[8] Saved selected model artifact: {selected_artifact_path} ({artifact_size:,} bytes)")

    # 8. Save selected confusion matrix CSV
    selected_cm_path = models_dir / "classical_best_validation_confusion_matrix.csv"
    cm_df = pd.DataFrame(
        selected_metrics["confusion_matrix"], index=classes, columns=classes
    )
    cm_df.to_csv(selected_cm_path, index=True)
    print(f"[9] Saved selected confusion matrix CSV: {selected_cm_path}")

    # 9. Save selected model metadata JSON
    metadata = {
        "dataset": "LIAR",
        "training_split": "train",
        "evaluation_split": "valid",
        "model_name": selected_result["model_name"],
        "classifier_type": selected_result["classifier"],
        "feature_representation": (
            "Word + Character TF-IDF" if selected_result["char_enabled"] else "Word TF-IDF"
        ),
        "configuration": selected_cfg,
        "native_probabilities": selected_result["native_probabilities"],
        "selection_metric": "macro_f1",
        "training_duration_seconds": round(fit_time, 3),
        "artifact_size_bytes": artifact_size,
        "accuracy": selected_metrics["accuracy"],
        "macro_precision": selected_metrics["macro_precision"],
        "macro_recall": selected_metrics["macro_recall"],
        "macro_f1": selected_metrics["macro_f1"],
        "weighted_f1": selected_metrics["weighted_f1"],
        "delta_macro_f1_vs_step6": selected_result["delta_macro_f1"],
        "per_class": selected_metrics["per_class"],
        "confusion_matrix": selected_metrics["confusion_matrix"],
        "class_order": classes,
        "overlap_excluded_diagnostic": selected_diagnostic,
        "selection_justification": justification,
    }

    metadata_path = models_dir / "classical_best_validation_metrics.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[10] Saved selected model metadata JSON: {metadata_path}")

    return {
        "results": results,
        "selected_result": selected_result,
        "justification": justification,
        "metadata": metadata,
    }
