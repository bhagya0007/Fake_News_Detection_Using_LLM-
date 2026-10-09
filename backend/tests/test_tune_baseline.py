"""Unit tests for baseline hyperparameter tuning and ranking logic.

Verifies:
1. Generation of 96 tuning configurations.
2. Complete absence of duplicate configurations in the search space.
3. Ranking function prioritizes highest Macro F1 and applies simplicity tie-breakers.
4. Tuning results serialization to JSON and CSV formats.
5. Serialized best model reloads cleanly and exposes classes.
6. Inference on tuned model produces allowed project labels.
7. 'unverifiable' is never emitted by the tuned baseline model.
8. Probability output remains normalized and contains all classes.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from backend.app.services.baseline_model import (
    load_baseline_pipeline,
    predict_baseline,
)
from backend.app.services.dataset_loader import ALLOWED_PROJECT_LABELS
from backend.app.services.tune_baseline import (
    generate_tuning_configurations,
    ranking_key,
)


def test_tuning_configuration_generation() -> None:
    """Test 1: Verify exactly 96 discrete hyperparameter configurations are generated."""
    configs = generate_tuning_configurations()
    assert len(configs) == 96


def test_no_duplicate_configurations() -> None:
    """Test 2: Verify all 96 generated configurations are unique."""
    configs = generate_tuning_configurations()
    # Serialize each config to a deterministic JSON string to test uniqueness
    serialized = [json.dumps(c, sort_keys=True) for c in configs]
    assert len(serialized) == len(set(serialized))


def test_ranking_chooses_highest_macro_f1() -> None:
    """Test 3: Verify ranking primarily chooses the highest Macro F1."""
    mock_results = [
        {
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_features": 50000,
            "C": 1.0,
            "class_weight": None,
            "accuracy": 0.45,
            "macro_f1": 0.4400,
            "training_seconds": 1.0,
        },
        {
            "ngram_range": [1, 1],
            "min_df": 2,
            "max_features": 30000,
            "C": 2.0,
            "class_weight": "balanced",
            "accuracy": 0.46,
            "macro_f1": 0.4650,  # Highest
            "training_seconds": 0.8,
        },
        {
            "ngram_range": [1, 2],
            "min_df": 3,
            "max_features": 30000,
            "C": 0.5,
            "class_weight": None,
            "accuracy": 0.44,
            "macro_f1": 0.4350,
            "training_seconds": 0.9,
        },
    ]

    ranked = sorted(mock_results, key=ranking_key, reverse=True)
    assert ranked[0]["macro_f1"] == 0.4650
    assert ranked[0]["C"] == 2.0


def test_ranking_tie_breaker_prefers_simpler_model() -> None:
    """Test 3b: When Macro F1 and accuracy are tied, simpler model (unigrams) is preferred."""
    tied_results = [
        {
            "ngram_range": [1, 2],  # More complex
            "min_df": 2,
            "max_features": 50000,
            "C": 1.0,
            "class_weight": None,
            "accuracy": 0.4500,
            "macro_f1": 0.4500,
            "training_seconds": 1.2,
        },
        {
            "ngram_range": [1, 1],  # Simpler
            "min_df": 2,
            "max_features": 30000,
            "C": 1.0,
            "class_weight": None,
            "accuracy": 0.4500,
            "macro_f1": 0.4500,
            "training_seconds": 0.8,
        },
    ]

    ranked = sorted(tied_results, key=ranking_key, reverse=True)
    # The (1, 1) model should be ranked first due to lower complexity
    assert ranked[0]["ngram_range"] == [1, 1]


def test_result_serialization(tmp_path: Path) -> None:
    """Test 4: Verify tuning results serialize to valid JSON and CSV."""
    sample_res = [
        {
            "ngram_range": [1, 1],
            "min_df": 2,
            "max_features": 30000,
            "C": 1.0,
            "class_weight": None,
            "accuracy": 0.456,
            "macro_precision": 0.455,
            "macro_recall": 0.448,
            "macro_f1": 0.447,
            "weighted_f1": 0.451,
            "training_seconds": 0.5,
        }
    ]

    json_path = tmp_path / "test_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sample_res, f)
    assert json_path.stat().st_size > 0

    csv_path = tmp_path / "test_results.csv"
    pd.DataFrame(sample_res).to_csv(csv_path, index=False)
    assert csv_path.stat().st_size > 0


@pytest.fixture(scope="module")
def small_tuned_pipeline(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Fixture creating and saving a fast synthetic model."""
    import joblib

    tmp_dir = tmp_path_factory.mktemp("models")
    model_path = tmp_dir / "test_tuned_model.joblib"

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 1))),
            ("classifier", LogisticRegression(C=2.0, random_state=42)),
        ]
    )
    texts = [
        "Economy grew rapidly this year.",
        "Official data confirmed employment numbers.",
        "Fake claims about vaccine microchips.",
        "Fabricated stories on election machines.",
        "Taxes might rise according to some reports.",
        "Candidate claimed rates changed moderately.",
    ]
    labels = [
        "likely_reliable",
        "likely_reliable",
        "likely_fake",
        "likely_fake",
        "misleading",
        "misleading",
    ]
    pipe.fit(texts, labels)
    joblib.dump(pipe, model_path)
    return model_path


def test_best_model_can_be_reloaded(small_tuned_pipeline: Path) -> None:
    """Test 5: Verify saved model artifact reloads cleanly and exposes classes_."""
    loaded = load_baseline_pipeline(small_tuned_pipeline)
    assert hasattr(loaded, "predict")
    assert hasattr(loaded, "classes_")
    assert set(loaded.classes_) == ALLOWED_PROJECT_LABELS


def test_inference_produces_valid_class(small_tuned_pipeline: Path) -> None:
    """Test 6: Inference on reloaded model produces one of the allowed project labels."""
    loaded = load_baseline_pipeline(small_tuned_pipeline)
    res = predict_baseline(loaded, "The federal budget expanded in 2024.")
    assert res["predicted_label"] in ALLOWED_PROJECT_LABELS


def test_no_unverifiable(small_tuned_pipeline: Path) -> None:
    """Test 7: 'unverifiable' is never output by the baseline model."""
    loaded = load_baseline_pipeline(small_tuned_pipeline)
    test_texts = [
        "Unconfirmed rumor on social media.",
        "No evidence was provided.",
        "Completely uncertain proposition.",
    ]
    for text in test_texts:
        res = predict_baseline(loaded, text)
        assert res["predicted_label"] != "unverifiable"
        assert "unverifiable" not in res["probabilities"]


def test_probability_output_remains_valid(small_tuned_pipeline: Path) -> None:
    """Test 8: Probabilities sum to approximately 1.0 and cover all 3 classes."""
    loaded = load_baseline_pipeline(small_tuned_pipeline)
    res = predict_baseline(loaded, "Testing probability normalization.")
    probs = res["probabilities"]
    assert set(probs.keys()) == ALLOWED_PROJECT_LABELS
    assert abs(sum(probs.values()) - 1.0) < 1e-4
