"""Unit tests for baseline model inference service.

Verifies:
1. Pipeline trains on a lightweight synthetic in-memory dataset.
2. Prediction produces one of the allowed project labels.
3. Probability output includes all fitted classes.
4. Probabilities sum to approximately 1.0.
5. Invalid/empty inputs are rejected with expected exceptions.
6. Serialized model artifact can be saved and reloaded.
7. Inference does not mutate or retrain the model.
8. 'unverifiable' is strictly NOT produced by the classical baseline.
9. Batch prediction functions correctly.
"""

from pathlib import Path
import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from backend.app.services.baseline_model import (
    load_baseline_pipeline,
    predict_baseline,
    predict_baseline_batch,
)
from backend.app.services.dataset_loader import ALLOWED_PROJECT_LABELS


@pytest.fixture(scope="module")
def synthetic_pipeline() -> Pipeline:
    """Fixture providing a fast, in-memory fitted pipeline for unit testing."""
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

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )
    pipe.fit(texts, labels)
    return pipe


def test_pipeline_trains_on_synthetic_data(synthetic_pipeline: Pipeline) -> None:
    """Test 1: Pipeline can train on synthetic dataset and exposes learned classes."""
    assert hasattr(synthetic_pipeline, "classes_")
    classes = set(synthetic_pipeline.classes_)
    assert classes == ALLOWED_PROJECT_LABELS


def test_prediction_produces_allowed_label(synthetic_pipeline: Pipeline) -> None:
    """Test 2: Prediction produces one of the allowed project labels."""
    result = predict_baseline(synthetic_pipeline, "Official data shows economic expansion.")
    assert "predicted_label" in result
    assert result["predicted_label"] in ALLOWED_PROJECT_LABELS


def test_probabilities_include_all_classes(synthetic_pipeline: Pipeline) -> None:
    """Test 3: Probability dictionary includes all fitted classes."""
    result = predict_baseline(synthetic_pipeline, "Claims about taxes were debated.")
    probs = result["probabilities"]
    assert set(probs.keys()) == set(synthetic_pipeline.classes_)
    for cls_name, prob in probs.items():
        assert 0.0 <= prob <= 1.0


def test_probabilities_sum_to_one(synthetic_pipeline: Pipeline) -> None:
    """Test 4: Class probability estimates sum approximately to 1.0."""
    result = predict_baseline(synthetic_pipeline, "Another statement for testing.")
    probs = result["probabilities"]
    prob_sum = sum(probs.values())
    assert abs(prob_sum - 1.0) < 1e-4


@pytest.mark.parametrize(
    "invalid_input, expected_err",
    [
        (None, ValueError),
        (12345, TypeError),
        ("", ValueError),
        ("   ", ValueError),
        ("\t\n", ValueError),
    ],
)
def test_invalid_input_rejected(
    synthetic_pipeline: Pipeline, invalid_input: object, expected_err: type[Exception]
) -> None:
    """Test 5: Empty, None, non-string, or whitespace-only inputs are rejected."""
    with pytest.raises(expected_err):
        predict_baseline(synthetic_pipeline, invalid_input)  # type: ignore[arg-type]


def test_saved_model_can_be_loaded(synthetic_pipeline: Pipeline, tmp_path: Path) -> None:
    """Test 6: A saved Pipeline can be persisted and reloaded cleanly."""
    import joblib

    model_file = tmp_path / "temp_pipeline.joblib"
    joblib.dump(synthetic_pipeline, model_file)

    loaded = load_baseline_pipeline(model_path=model_file)
    assert hasattr(loaded, "predict")
    assert list(loaded.classes_) == list(synthetic_pipeline.classes_)

    res = predict_baseline(loaded, "Census bureau data was released.")
    assert res["predicted_label"] in ALLOWED_PROJECT_LABELS


def test_inference_does_not_retrain_model(synthetic_pipeline: Pipeline) -> None:
    """Test 7: Inference does not alter classifier coefficients or retrain the model."""
    classifier = synthetic_pipeline.named_steps["classifier"]
    coef_before = np.copy(classifier.coef_)

    # Run multiple predictions
    predict_baseline(synthetic_pipeline, "Sample statement 1")
    predict_baseline(synthetic_pipeline, "Sample statement 2")

    coef_after = classifier.coef_
    np.testing.assert_array_equal(coef_before, coef_after)


def test_unverifiable_never_produced(synthetic_pipeline: Pipeline) -> None:
    """Test 8: 'unverifiable' is never output as a label or probability key."""
    statements = [
        "Unemployment went down.",
        "Aliens landed yesterday.",
        "Uncertain statement with no evidence.",
    ]
    for s in statements:
        res = predict_baseline(synthetic_pipeline, s)
        assert res["predicted_label"] != "unverifiable"
        assert "unverifiable" not in res["probabilities"]


def test_batch_prediction(synthetic_pipeline: Pipeline) -> None:
    """Test 9: Batch prediction returns correct number of structured results."""
    statements = [
        "Report shows growth.",
        "False claims about medicine.",
    ]
    batch_results = predict_baseline_batch(synthetic_pipeline, statements)
    assert len(batch_results) == 2
    for r in batch_results:
        assert r["predicted_label"] in ALLOWED_PROJECT_LABELS
        assert abs(sum(r["probabilities"].values()) - 1.0) < 1e-4
