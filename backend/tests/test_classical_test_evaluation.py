"""Unit tests for Classical Model Test Evaluation (Step 7).

Verifies:
1. Test evaluator accepts a fitted model pipeline.
2. Evaluator strictly NEVER calls .fit() or .partial_fit().
3. Metrics output contains all required fields and correct types.
4. Confusion matrix shape is exactly 3x3.
5. Predicted probabilities sum to approximately 1.0 for every sample.
6. Predictions CSV has the exact required columns and non-null values.
7. Only allowed project labels are emitted.
8. 'unverifiable' is strictly NEVER emitted.
9. Test evaluation does not alter model parameters (coefficients/intercept).
10. Frozen artifact reloads successfully and functions cleanly.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from backend.app.services.dataset_loader import (
    ALLOWED_PROJECT_LABELS,
    get_project_root,
)
from backend.app.services.evaluate_classical import (
    compute_probability_diagnostic,
    compute_test_metrics,
    compute_test_overlap_diagnostic,
    evaluate_frozen_classical_model,
    load_frozen_classical_artifact,
)


@pytest.fixture
def synthetic_test_corpus() -> tuple[list[str], list[str]]:
    """Fast in-memory synthetic dataset for evaluation unit tests."""
    texts = [
        "Economy expanded by two percent in the latest quarter.",
        "Official census records verify the national population.",
        "Fabricated rumor that the ocean turned into soda.",
        "Fake news story regarding microchip injections.",
        "Claim stated taxes doubled, though rates only rose slightly.",
        "Candidate claimed unemployment dropped, but figures are mixed.",
    ]
    labels = [
        "likely_reliable",
        "likely_reliable",
        "likely_fake",
        "likely_fake",
        "misleading",
        "misleading",
    ]
    return texts, labels


@pytest.fixture
def fitted_synthetic_pipeline(synthetic_test_corpus: tuple[list[str], list[str]]) -> Pipeline:
    """Pre-fitted synthetic pipeline for isolated unit tests."""
    texts, labels = synthetic_test_corpus
    pipe = Pipeline(
        [
            ("features", TfidfVectorizer(ngram_range=(1, 2))),
            ("classifier", LogisticRegression(C=0.5, class_weight="balanced", random_state=42)),
        ]
    )
    pipe.fit(texts, labels)
    return pipe


def test_frozen_artifact_reloads_successfully() -> None:
    """Test 1: Verify the frozen classical artifact exists, reloads, and has expected attributes."""
    pipeline = load_frozen_classical_artifact()
    assert hasattr(pipeline, "predict")
    assert hasattr(pipeline, "predict_proba")
    assert hasattr(pipeline, "classes_")
    assert set(pipeline.classes_) == ALLOWED_PROJECT_LABELS


def test_evaluator_never_calls_fit_or_partial_fit(
    tmp_path: Path,
    fitted_synthetic_pipeline: Pipeline,
) -> None:
    """Test 2: CRITICAL GUARANTEE - Evaluator strictly never calls .fit() or .partial_fit()."""
    import ast

    # 1. Structural / AST source code check: no Call node calls 'fit' or 'partial_fit'
    for path in [
        Path("backend/app/services/evaluate_classical.py"),
        Path("scripts/evaluate_classical_test.py"),
    ]:
        assert path.exists()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    assert node.func.attr not in {
                        "fit",
                        "partial_fit",
                        "fit_transform",
                        "fit_predict",
                    }, f"Illegal call to '{node.func.attr}' found in {path.name} at line {node.lineno}!"

    # 2. Runtime mock check: calling .fit raises an error if attempted during evaluation
    temp_model_path = tmp_path / "test_model.joblib"
    joblib.dump(fitted_synthetic_pipeline, temp_model_path)

    with patch.object(Pipeline, "fit", side_effect=RuntimeError("Illegal call to Pipeline.fit()")):
        with patch.object(
            LogisticRegression, "fit", side_effect=RuntimeError("Illegal call to LogisticRegression.fit()")
        ):
            metrics = evaluate_frozen_classical_model(
                model_path=temp_model_path,
                output_dir=tmp_path,
            )
            assert metrics is not None
            assert metrics["accuracy"] >= 0.0


def test_metrics_output_contains_required_fields(
    synthetic_test_corpus: tuple[list[str], list[str]],
    fitted_synthetic_pipeline: Pipeline,
) -> None:
    """Test 3: Verify compute_test_metrics returns all required structure and valid ranges."""
    texts, labels = synthetic_test_corpus
    classes = [str(c) for c in fitted_synthetic_pipeline.classes_]
    preds = fitted_synthetic_pipeline.predict(texts)

    metrics = compute_test_metrics(labels, preds, classes=classes)

    required_keys = {
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "per_class",
        "confusion_matrix",
        "class_order",
    }
    assert required_keys.issubset(metrics.keys())
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0

    for cls in classes:
        assert cls in metrics["per_class"]
        pc = metrics["per_class"][cls]
        assert "precision" in pc
        assert "recall" in pc
        assert "f1_score" in pc
        assert "support" in pc


def test_confusion_matrix_shape_is_3x3(
    synthetic_test_corpus: tuple[list[str], list[str]],
    fitted_synthetic_pipeline: Pipeline,
) -> None:
    """Test 4: Verify confusion matrix is exactly 3x3 for the 3 project classes."""
    texts, labels = synthetic_test_corpus
    classes = [str(c) for c in fitted_synthetic_pipeline.classes_]
    preds = fitted_synthetic_pipeline.predict(texts)

    metrics = compute_test_metrics(labels, preds, classes=classes)
    cm = metrics["confusion_matrix"]
    assert len(cm) == 3
    for row in cm:
        assert len(row) == 3


def test_probabilities_sum_to_one(
    synthetic_test_corpus: tuple[list[str], list[str]],
    fitted_synthetic_pipeline: Pipeline,
) -> None:
    """Test 5: Verify class probability estimates sum to 1.0 across all predictions."""
    texts, _ = synthetic_test_corpus
    probs = fitted_synthetic_pipeline.predict_proba(texts)
    sums = np.sum(probs, axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-5)


def test_predictions_csv_schema_and_allowed_labels(tmp_path: Path) -> None:
    """Test 6: Verify generated predictions CSV has correct schema and valid labels."""
    output_dir = tmp_path / "models"
    output_dir.mkdir()

    evaluate_frozen_classical_model(output_dir=output_dir)

    pred_csv = output_dir / "classical_test_predictions.csv"
    assert pred_csv.exists()

    df = pd.read_csv(pred_csv)
    expected_cols = [
        "id",
        "statement",
        "true_label",
        "predicted_label",
        "correct",
        "prob_likely_fake",
        "prob_likely_reliable",
        "prob_misleading",
    ]
    assert list(df.columns) == expected_cols
    assert len(df) == 1283

    # Check allowed labels
    assert set(df["predicted_label"].unique()).issubset(ALLOWED_PROJECT_LABELS)
    assert set(df["true_label"].unique()).issubset(ALLOWED_PROJECT_LABELS)


def test_unverifiable_never_appears(tmp_path: Path) -> None:
    """Test 7: 'unverifiable' is strictly NEVER emitted by the classical baseline."""
    output_dir = tmp_path / "models"
    output_dir.mkdir()

    metrics = evaluate_frozen_classical_model(output_dir=output_dir)
    assert "unverifiable" not in metrics["class_order"]
    assert "unverifiable" not in metrics["per_class"]

    pred_csv = output_dir / "classical_test_predictions.csv"
    df = pd.read_csv(pred_csv)
    assert "unverifiable" not in df["predicted_label"].values
    assert "unverifiable" not in df["true_label"].values


def test_evaluation_does_not_mutate_model_parameters(
    synthetic_test_corpus: tuple[list[str], list[str]],
    fitted_synthetic_pipeline: Pipeline,
    tmp_path: Path,
) -> None:
    """Test 8: Verify evaluation inference does not mutate classifier weights."""
    clf = fitted_synthetic_pipeline.named_steps["classifier"]
    coef_orig = np.copy(clf.coef_)
    intercept_orig = np.copy(clf.intercept_)

    model_file = tmp_path / "param_check_model.joblib"
    joblib.dump(fitted_synthetic_pipeline, model_file)

    evaluate_frozen_classical_model(model_path=model_file, output_dir=tmp_path)

    # Reload and check coefficients
    reloaded = joblib.load(model_file)
    clf_reloaded = reloaded.named_steps["classifier"]
    np.testing.assert_array_equal(coef_orig, clf_reloaded.coef_)
    np.testing.assert_array_equal(intercept_orig, clf_reloaded.intercept_)


def test_probability_diagnostic_computation() -> None:
    """Test 9: Verify descriptive probability diagnostic computes correctly."""
    probs = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.3, 0.6, 0.1],
            [0.2, 0.3, 0.5],
            [0.4, 0.3, 0.3],
        ]
    )
    is_correct = np.array([True, True, False, False])
    diag = compute_probability_diagnostic(probs, is_correct)

    assert "correct_predictions" in diag
    assert "incorrect_predictions" in diag
    assert diag["correct_predictions"]["count"] == 2
    assert diag["incorrect_predictions"]["count"] == 2
    assert 0.0 <= diag["correct_predictions"]["mean_max_probability"] <= 1.0
    assert 0.0 <= diag["incorrect_predictions"]["mean_max_probability"] <= 1.0


def test_overlap_diagnostic_computation() -> None:
    """Test 10: Verify secondary overlap diagnostic correctly filters train overlaps."""
    train_df = pd.DataFrame(
        {
            "clean_statement": ["Statement A", "Statement B"],
            "project_label": ["likely_reliable", "likely_fake"],
        }
    )
    test_df = pd.DataFrame(
        {
            "clean_statement": ["Statement A", "Statement C", "Statement D"],
            "project_label": ["likely_reliable", "likely_fake", "misleading"],
        }
    )
    y_test = ["likely_reliable", "likely_fake", "misleading"]
    y_pred = ["likely_reliable", "likely_fake", "likely_fake"]
    classes = ["likely_fake", "likely_reliable", "misleading"]

    diag = compute_test_overlap_diagnostic(train_df, test_df, y_test, y_pred, classes)
    assert diag["overlap_statements_excluded"] == 1
    assert diag["evaluated_rows"] == 2
    assert "accuracy" in diag
    assert "macro_f1" in diag
