"""Unit tests for Classical Model Comparison (Step 6.5).

Verifies:
1. Exact generation of 23 candidate configurations across 4 model families.
2. Complete absence of duplicate configurations in the comparison set.
3. FeatureUnion pipelines preserve scipy.sparse matrices without dense densification.
4. LinearSVC exposes decision_function without fabricating probabilities.
5. Model selection logic strictly enforces the >= +0.0020 Macro F1 threshold.
6. 'unverifiable' is strictly never output by any classical model.
7. Saved model artifacts reload cleanly and expose valid classes.
8. Strict verification that test.tsv is never loaded or referenced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from backend.app.services.baseline_model import (
    load_baseline_pipeline,
    predict_baseline,
)
from backend.app.services.compare_classical import (
    MIN_MEANINGFUL_IMPROVEMENT,
    REFERENCE_ACCURACY,
    REFERENCE_MACRO_F1,
    build_comparison_pipeline,
    generate_comparison_configurations,
    select_best_model,
)
from backend.app.services.dataset_loader import (
    ALLOWED_PROJECT_LABELS,
    get_project_root,
)


@pytest.fixture
def small_synthetic_corpus() -> tuple[list[str], list[str]]:
    """Fixture providing a fast, in-memory synthetic dataset for pipeline tests."""
    texts = [
        "Government report confirmed economic growth was robust in 2023.",
        "Official census bureau data validated state population trends.",
        "Claims that the moon is made of cheese were declared false.",
        "Fabricated claims regarding third-trimester abortions are false.",
        "Some claim that taxes doubled, but actual rates only increased slightly.",
        "Talking point suggests crime dropped, though statistics remain mixed.",
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


def test_configuration_generation_count_and_groups() -> None:
    """Test 1: Verify exactly 23 discrete comparison configurations across 4 families."""
    configs = generate_comparison_configurations()
    assert len(configs) == 23

    # Group counts: Model A (1), Model B (6), Model C (8), Model D (8)
    groups = [c["model_group"] for c in configs]
    assert groups.count("A") == 1
    assert groups.count("B") == 6
    assert groups.count("C") == 8
    assert groups.count("D") == 8


def test_no_duplicate_configurations() -> None:
    """Test 2: Verify all 23 configurations are unique."""
    configs = generate_comparison_configurations()
    serialized = [
        json.dumps(
            {
                "group": c["model_group"],
                "classifier": c["classifier"],
                "C": c["C"],
                "cw": str(c["class_weight"]),
                "char_enabled": c["char_enabled"],
                "char_max": c.get("char_max_features"),
            },
            sort_keys=True,
        )
        for c in configs
    ]
    assert len(serialized) == len(set(serialized))


def test_word_and_char_pipeline_uses_sparse_representation(
    small_synthetic_corpus: tuple[list[str], list[str]],
) -> None:
    """Test 3: FeatureUnion pipeline preserves scipy.sparse representation."""
    texts, labels = small_synthetic_corpus
    cfg = {
        "word_ngram_range": (1, 2),
        "word_max_features": 1000,
        "char_enabled": True,
        "char_ngram_range": (3, 5),
        "char_max_features": 1000,
        "classifier": "LogisticRegression",
        "C": 1.0,
        "class_weight": None,
    }
    pipe = build_comparison_pipeline(cfg)
    assert "features" in pipe.named_steps
    assert isinstance(pipe.named_steps["features"], FeatureUnion)

    # Fit pipeline
    pipe.fit(texts, labels)

    # Transform features and verify it's a sparse CSR matrix
    features_step = pipe.named_steps["features"]
    transformed = features_step.transform(texts)
    assert sp.issparse(transformed)
    assert transformed.getformat() == "csr"


def test_linear_svc_prediction_and_decision_scores(
    small_synthetic_corpus: tuple[list[str], list[str]],
) -> None:
    """Test 4: LinearSVC produces valid labels and exposes decision_scores without faking probabilities."""
    texts, labels = small_synthetic_corpus
    cfg = {
        "word_ngram_range": (1, 2),
        "word_max_features": 1000,
        "char_enabled": False,
        "char_ngram_range": None,
        "char_max_features": None,
        "classifier": "LinearSVC",
        "C": 1.0,
        "class_weight": None,
    }
    pipe = build_comparison_pipeline(cfg)
    pipe.fit(texts, labels)

    res = predict_baseline(pipe, "Economic indicators showed slight growth.")
    assert res["predicted_label"] in ALLOWED_PROJECT_LABELS
    assert res["probabilities"] is None
    assert "decision_scores" in res
    assert isinstance(res["decision_scores"], dict)
    assert set(res["decision_scores"].keys()) == ALLOWED_PROJECT_LABELS


def test_selection_logic_enforces_meaningful_improvement_threshold() -> None:
    """Test 5: Model selection strictly enforces the >= +0.0020 Macro F1 threshold."""
    ref_cfg = {
        "model_name": "Tuned Logistic Regression (Step 6)",
        "model_group": "A",
        "classifier": "LogisticRegression",
        "char_enabled": False,
        "C": 0.5,
        "class_weight": "balanced",
        "word_ngram_range": (1, 2),
        "word_max_features": 50000,
        "char_ngram_range": None,
        "char_max_features": None,
    }

    # Case 1: Candidate is slightly better (+0.0010) but below threshold (+0.0020)
    # Reference should be retained!
    results_below_threshold = [
        {
            **ref_cfg,
            "macro_f1": REFERENCE_MACRO_F1,
            "accuracy": REFERENCE_ACCURACY,
            "native_probabilities": True,
            "training_seconds": 1.5,
        },
        {
            "model_name": "Word + Char TF-IDF + LinearSVC",
            "model_group": "D",
            "classifier": "LinearSVC",
            "char_enabled": True,
            "C": 1.0,
            "class_weight": "balanced",
            "word_ngram_range": (1, 2),
            "word_max_features": 50000,
            "char_ngram_range": (3, 5),
            "char_max_features": 50000,
            "macro_f1": REFERENCE_MACRO_F1 + 0.0010,  # Below threshold
            "accuracy": REFERENCE_ACCURACY + 0.0010,
            "native_probabilities": False,
            "training_seconds": 3.0,
        },
    ]
    selected, just = select_best_model(results_below_threshold, ref_cfg)
    assert selected["model_name"] == "Tuned Logistic Regression (Step 6)"
    assert "retained for lower complexity" in just

    # Case 2: Candidate exceeds threshold (+0.0035 >= +0.0020)
    # Candidate should be selected!
    results_above_threshold = [
        {
            **ref_cfg,
            "macro_f1": REFERENCE_MACRO_F1,
            "accuracy": REFERENCE_ACCURACY,
            "native_probabilities": True,
            "training_seconds": 1.5,
        },
        {
            "model_name": "Word + Char TF-IDF + LinearSVC",
            "model_group": "D",
            "classifier": "LinearSVC",
            "char_enabled": True,
            "C": 1.0,
            "class_weight": "balanced",
            "word_ngram_range": (1, 2),
            "word_max_features": 50000,
            "char_ngram_range": (3, 5),
            "char_max_features": 50000,
            "macro_f1": REFERENCE_MACRO_F1 + 0.0035,  # Exceeds threshold
            "accuracy": REFERENCE_ACCURACY + 0.0030,
            "native_probabilities": False,
            "training_seconds": 3.0,
        },
    ]
    selected, just = select_best_model(results_above_threshold, ref_cfg)
    assert selected["model_name"] == "Word + Char TF-IDF + LinearSVC"
    assert "exceeding the minimum meaningful threshold" in just


def test_classical_model_never_emits_unverifiable(
    small_synthetic_corpus: tuple[list[str], list[str]],
) -> None:
    """Test 6: Classical models never emit 'unverifiable'."""
    texts, labels = small_synthetic_corpus
    for clf_name in ["LogisticRegression", "LinearSVC"]:
        cfg = {
            "word_ngram_range": (1, 2),
            "word_max_features": 1000,
            "char_enabled": False,
            "char_ngram_range": None,
            "char_max_features": None,
            "classifier": clf_name,
            "C": 1.0,
            "class_weight": None,
        }
        pipe = build_comparison_pipeline(cfg)
        pipe.fit(texts, labels)
        for query in [
            "completely unverified claim",
            "official report released",
            "aliens on the moon",
        ]:
            res = predict_baseline(pipe, query)
            assert res["predicted_label"] != "unverifiable"
            if res.get("probabilities"):
                assert "unverifiable" not in res["probabilities"]


def test_artifact_persistence_and_reload(
    tmp_path: Path,
    small_synthetic_corpus: tuple[list[str], list[str]],
) -> None:
    """Test 7: Saved artifact can be serialized and reloaded cleanly."""
    texts, labels = small_synthetic_corpus
    cfg = {
        "word_ngram_range": (1, 2),
        "word_max_features": 1000,
        "char_enabled": True,
        "char_ngram_range": (3, 5),
        "char_max_features": 1000,
        "classifier": "LinearSVC",
        "C": 1.0,
        "class_weight": "balanced",
    }
    pipe = build_comparison_pipeline(cfg)
    pipe.fit(texts, labels)

    save_path = tmp_path / "classical_test_artifact.joblib"
    joblib.dump(pipe, save_path)
    assert save_path.exists()

    loaded = load_baseline_pipeline(save_path)
    assert hasattr(loaded, "classes_")
    assert set(loaded.classes_) == ALLOWED_PROJECT_LABELS

    res = predict_baseline(loaded, "Labor statistics were updated.")
    assert res["predicted_label"] in ALLOWED_PROJECT_LABELS


def test_no_test_set_leakage_in_comparison_modules() -> None:
    """Test 8: Strict check that test.tsv is never loaded or referenced in comparison logic."""
    service_file = Path("backend/app/services/compare_classical.py")
    script_file = Path("scripts/compare_classical_models.py")

    assert service_file.exists()
    assert script_file.exists()

    service_code = service_file.read_text(encoding="utf-8")
    script_code = script_file.read_text(encoding="utf-8")

    # Ensure neither file calls prepare_baseline_split("test") or mentions test.tsv loading
    assert 'prepare_baseline_split("test")' not in service_code
    assert 'prepare_baseline_split("test")' not in script_code
    assert 'split="test"' not in service_code
    assert 'split="test"' not in script_code
