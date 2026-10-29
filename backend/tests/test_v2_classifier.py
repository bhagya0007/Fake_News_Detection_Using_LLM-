"""Unit and Contract Tests for V2 Classifiers & Epistemic Abstention Boundaries.

Tests verify:
1. Strict 3-class label contract (likely_reliable, misleading, likely_fake).
2. Probabilities strictly sum to 1.0.
3. Zero unverifiable predictions from ML classifiers.
4. Clean integration with HybridAnalysisEngine via MLAssessment schema.
5. Immutability of frozen V1 baseline and benchmark datasets.
6. Classifier training pool integrity (0 unverifiable records in pool).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from backend.app.config import MODELS_DIR, PROJECT_ROOT
from backend.app.schemas import MLAssessment, MLLabel
from backend.app.services.v2_classifier_service import (
    ALLOWED_LABELS,
    V2ClassifierService,
)

CANONICAL_V1_BASELINE_SHA256 = "d4c3636277761f5399166ee41c0eec57cafe7b9216bacfc18d81f26fea415c92"
CANONICAL_LIAR_TEST_TSV_SHA256 = "ee41e2c741e0019216952625145af8c4aa8c8d3057ec68284b134267bb845e6f"


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class TestV2ClassifierContracts:
    """Validate architectural contracts and runtime guards of V2 classifiers."""

    def test_v1_baseline_immutability(self):
        """Historical V1 classical baseline artifact must NEVER be modified or retrained."""
        v1_path = MODELS_DIR / "classical_best_validation.joblib"
        assert v1_path.exists(), f"Missing canonical V1 artifact: {v1_path}"
        actual_hash = compute_sha256(v1_path)
        assert actual_hash == CANONICAL_V1_BASELINE_SHA256, "V1 baseline hash mismatch!"

    def test_liar_test_split_immutability(self):
        """Official frozen LIAR test split must remain completely untouched."""
        test_path = PROJECT_ROOT / "data" / "raw" / "test.tsv"
        assert test_path.exists(), f"Missing LIAR test split: {test_path}"
        actual_hash = compute_sha256(test_path)
        assert actual_hash == CANONICAL_LIAR_TEST_TSV_SHA256, "LIAR test split hash mismatch!"

    def test_classifier_training_pool_excludes_unverifiable(self):
        """Classifier training pool must strictly exclude unverifiable / Not Enough Evidence."""
        pool_path = PROJECT_ROOT / "data" / "v2_processed" / "classifier_train_pool.jsonl"
        assert pool_path.exists(), f"Missing classifier pool: {pool_path}"
        
        count = 0
        with open(pool_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                label = item.get("harmonized_label")
                assert label != "unverifiable", f"Found illegal unverifiable label in training pool: {item}"
                assert label in ALLOWED_LABELS, f"Unexpected label in pool: {label}"
                count += 1
        
        assert count == 11926, f"Expected 11,926 records in pool, found {count}"

    def test_v2_tfidf_service_prediction_contract(self):
        """V2 TF-IDF classifier predicts one of 3 classes and probabilities sum to 1.0."""
        service = V2ClassifierService(model_type="tfidf")
        res = service.predict_dict("The economy grew by 3 percent according to official statistics.")
        
        assert res["predicted_label"] in ALLOWED_LABELS
        assert res["predicted_label"] != "unverifiable"
        
        probs = res["probabilities"]
        assert set(probs.keys()) == ALLOWED_LABELS
        total_prob = sum(probs.values())
        assert abs(total_prob - 1.0) < 1e-4

    def test_v2_tfidf_ml_assessment_schema_compliance(self):
        """V2 TF-IDF classifier outputs valid MLAssessment compliant with HybridAnalysisEngine."""
        service = V2ClassifierService(model_type="tfidf")
        assessment = service.predict("Vaccines cause magnetic fields.")
        
        assert isinstance(assessment, MLAssessment)
        assert assessment.label in [MLLabel.likely_fake, MLLabel.likely_reliable, MLLabel.misleading]
        assert assessment.model_name == "V2 TFIDF Classifier"
        assert assessment.model_version == "v2.5_frozen"
        
        prob_sum = (
            assessment.probabilities.likely_fake +
            assessment.probabilities.likely_reliable +
            assessment.probabilities.misleading
        )
        assert abs(prob_sum - 1.0) < 1e-4

    def test_v2_minilm_service_prediction_contract(self):
        """V2 MiniLM classifier predicts one of 3 classes and probabilities sum to 1.0."""
        service = V2ClassifierService(model_type="minilm")
        res = service.predict_dict("The state government passed a bill funding renewable energy.")
        
        assert res["predicted_label"] in ALLOWED_LABELS
        assert res["predicted_label"] != "unverifiable"
        
        probs = res["probabilities"]
        assert set(probs.keys()) == ALLOWED_LABELS
        total_prob = sum(probs.values())
        assert abs(total_prob - 1.0) < 1e-4

    def test_v2_minilm_ml_assessment_compliance(self):
        """V2 MiniLM classifier outputs valid MLAssessment for downstream fusion."""
        service = V2ClassifierService(model_type="minilm")
        assessment = service.predict("Aliens landed in downtown Chicago today.")
        
        assert isinstance(assessment, MLAssessment)
        assert assessment.label in [MLLabel.likely_fake, MLLabel.likely_reliable, MLLabel.misleading]
        assert assessment.model_name == "V2 MINILM Classifier"

    def test_ml_label_strictly_rejects_unverifiable(self):
        """MLLabel enum strictly rejects 'unverifiable' at the type schema level."""
        with pytest.raises(ValueError):
            MLLabel("unverifiable")

    def test_v2_tfidf_metadata_hash_integrity(self):
        """Metadata recorded SHA-256 matches actual serialized artifact."""
        meta_path = MODELS_DIR / "v2" / "tfidf_logreg" / "metadata.json"
        artifact_path = MODELS_DIR / "v2" / "tfidf_logreg" / "model.joblib"
        assert meta_path.exists()
        assert artifact_path.exists()
        
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        actual_hash = compute_sha256(artifact_path)
        assert meta["artifact_sha256"] == actual_hash

    def test_v2_minilm_metadata_hash_integrity(self):
        """Metadata recorded SHA-256 for MiniLM model matches actual serialized artifact."""
        meta_path = MODELS_DIR / "v2" / "minilm_logreg" / "metadata.json"
        artifact_path = MODELS_DIR / "v2" / "minilm_logreg" / "model.joblib"
        assert meta_path.exists()
        assert artifact_path.exists()
        
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        actual_hash = compute_sha256(artifact_path)
        assert meta["artifact_sha256"] == actual_hash
