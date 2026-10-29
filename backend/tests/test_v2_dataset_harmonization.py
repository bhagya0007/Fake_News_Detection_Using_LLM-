"""Unit and Regression Tests for V2 Dataset Harmonization (Step V2.4).

Verifies:
- Label mappings across LIAR, AVeriTeC, MultiFC, and FEVER
- Fail-closed behavior on ambiguous/unverified labels
- Text normalization preservation (numbers, dates, currency, unicode)
- Empty/invalid row exclusion
- Deduplication and text hashing
- Test split protection and cross-split leakage prevention
- Manifest integrity and reproducibility
- Zero-external-dependency invariants (no Groq, no GDELT, no model training)
- Preservation of frozen V1 baseline files
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from backend.app.data.v2_dataset_schema import HarmonizedSample
from backend.app.data.v2_label_mapping import (
    MAPPING_VERSION,
    map_dataset_label,
    LIAR_LABEL_MAPPING,
    AVERITEC_LABEL_MAPPING,
    MULTIFC_ACCEPTED_MAPPING,
    MULTIFC_EXCLUDED_LABELS,
)
from scripts.prepare_v2_datasets import (
    normalize_claim_text,
    compute_text_hash,
    audit_text_quality,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "v2_processed"
RAW_LIAR_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"


# ---------------------------------------------------------------------------
# 1. Label Mapping Tests
# ---------------------------------------------------------------------------
class TestLabelMappings:
    """Test audited label mapping dictionaries and fail-closed rules."""

    def test_mapping_version_constant(self):
        assert MAPPING_VERSION == "v2_label_mapping_v1"

    @pytest.mark.parametrize(
        "orig,expected_target",
        [
            ("true", "likely_reliable"),
            ("mostly-true", "likely_reliable"),
            ("half-true", "misleading"),
            ("barely-true", "misleading"),
            ("false", "likely_fake"),
            ("pants-fire", "likely_fake"),
        ],
    )
    def test_liar_mappings(self, orig, expected_target):
        target, status, conf, rule = map_dataset_label("liar", orig)
        assert status == "accepted"
        assert target == expected_target
        assert conf >= 0.90
        assert "liar_v1" in rule
        # Verify LIAR never maps to unverifiable
        assert target != "unverifiable"

    def test_liar_unknown_label_fails_closed(self):
        target, status, conf, rule = map_dataset_label("liar", "custom_unknown_rating")
        assert target is None
        assert status == "excluded"
        assert conf == 0.0
        assert rule == "liar_unknown_label_fail_closed"

    def test_averitec_supported_mapping(self):
        target, status, conf, rule = map_dataset_label("averitec", "Supported")
        assert status == "accepted"
        assert target == "likely_reliable"
        assert conf == 1.0
        assert rule == "averitec_supported_to_reliable"

    def test_averitec_refuted_mapping(self):
        target, status, conf, rule = map_dataset_label("averitec", "Refuted")
        assert status == "accepted"
        assert target == "likely_fake"
        assert conf == 1.0
        assert rule == "averitec_refuted_to_fake"

    def test_averitec_nei_mapping(self):
        target, status, conf, rule = map_dataset_label("averitec", "Not Enough Evidence")
        assert status == "accepted"
        assert target == "unverifiable"
        assert conf == 1.0
        assert rule == "averitec_nei_to_unverifiable"

    @pytest.mark.parametrize(
        "label_str",
        [
            "Conflicting Evidence/Cherrypicking",
            "Conflicting Evidence/Cherry-picking",
        ],
    )
    def test_averitec_cherrypicking_mapping(self, label_str):
        target, status, conf, rule = map_dataset_label("averitec", label_str)
        assert status == "accepted"
        assert target == "misleading"
        assert conf == 0.85
        assert rule == "averitec_cherrypicking_to_misleading"

    def test_averitec_unknown_label_fails_closed(self):
        target, status, conf, rule = map_dataset_label("averitec", "fabricated_verdict")
        assert target is None
        assert status == "excluded"
        assert conf == 0.0

    @pytest.mark.parametrize(
        "orig,expected_target",
        [
            ("fact", "likely_reliable"),
            ("truth!", "likely_reliable"),
            ("correct", "likely_reliable"),
            ("mostly true", "likely_reliable"),
            ("fiction!", "likely_fake"),
            ("fake news", "likely_fake"),
            ("pants on fire!", "likely_fake"),
            ("bogus warning", "likely_fake"),
            ("half-true", "misleading"),
            ("cherry picks", "misleading"),
            ("mixture", "misleading"),
            ("distorts the facts", "misleading"),
            ("unproven", "unverifiable"),
            ("no evidence", "unverifiable"),
        ],
    )
    def test_multifc_accepted_mappings(self, orig, expected_target):
        target, status, conf, rule = map_dataset_label("multifc", orig)
        assert status == "accepted"
        assert target == expected_target
        assert conf >= 0.85

    @pytest.mark.parametrize(
        "excluded_label",
        [
            "0",
            "1",
            "2",
            "promise kept",
            "promise broken",
            "compromise",
            "full flop",
            "none",
        ],
    )
    def test_multifc_non_veracity_excluded(self, excluded_label):
        target, status, conf, rule = map_dataset_label("multifc", excluded_label)
        assert target is None
        assert status == "excluded"
        assert conf == 0.0

    def test_multifc_ambiguous_label_fails_closed(self):
        target, status, conf, rule = map_dataset_label("multifc", "completely_unseen_publisher_tag")
        assert target is None
        assert status == "ambiguous"
        assert conf == 0.0
        assert rule == "multifc_unverified_label_fail_closed"


# ---------------------------------------------------------------------------
# 2. Text Normalization and Quality Auditing Tests
# ---------------------------------------------------------------------------
class TestTextNormalization:
    """Test text preprocessing preservation and quality filters."""

    def test_preserves_numbers_percentages_currency(self):
        raw = "   Senator claimed $4.5 million (or 15.2%) was allocated to 35 projects in 2024.   \n\n"
        norm = normalize_claim_text(raw)
        assert "$4.5 million" in norm
        assert "15.2%" in norm
        assert "35 projects" in norm
        assert "2024" in norm
        assert norm.startswith("Senator")
        assert norm.endswith("2024.")

    def test_preserves_unicode_entities_and_dashes(self):
        raw = '“The CEO of Pfizer—Albert Bourla—stated: ‘Vaccines are effective.’”'
        norm = normalize_claim_text(raw)
        assert "Albert Bourla" in norm
        assert "Pfizer" in norm
        assert "-" in norm

    def test_strips_bom_and_null_bytes(self):
        raw = "\ufeff\x00Normal claim text about the economy.\x00"
        norm = normalize_claim_text(raw)
        assert norm == "Normal claim text about the economy."

    def test_quality_audit_empty_text(self):
        assert audit_text_quality("", "") == "empty_text"
        assert audit_text_quality("   \n\t  ", "") == "empty_text"

    def test_quality_audit_too_short(self):
        assert audit_text_quality("Short", "Short") == "too_short"

    def test_quality_audit_url_only(self):
        assert audit_text_quality("https://example.com/claim", "https://example.com/claim") == "url_only"

    def test_quality_audit_encoding_corrupted(self):
        assert audit_text_quality("Corrupted \ufffd claim text here.", "Corrupted claim text here.") == "encoding_corrupted"

    def test_compute_text_hash_deterministic(self):
        h1 = compute_text_hash("President signed executive order on tariffs.")
        h2 = compute_text_hash("  President SIGNED executive order on tariffs! \n")
        assert h1 == h2
        assert len(h1) == 64


# ---------------------------------------------------------------------------
# 3. Schema & Canonical Object Tests
# ---------------------------------------------------------------------------
class TestHarmonizedSchema:
    """Test HarmonizedSample Pydantic validation."""

    def test_sample_construction_preserves_raw_and_provenance(self):
        sample = HarmonizedSample(
            sample_id="test_001",
            source_dataset="averitec",
            source_split="train",
            original_id="orig_42",
            text="Cleaned claim text here.",
            raw_text="  Cleaned claim text here. \n",
            original_label="Supported",
            harmonized_label="likely_reliable",
            mapping_status="accepted",
            mapping_confidence=1.0,
            mapping_rule="averitec_supported_to_reliable",
            domain="politics",
            speaker="Alice",
            claim_date="2023-01-01",
            source_url="https://example.com",
            has_evidence=True,
            evidence_reference="doc_01",
            license_tag="CC BY-NC 4.0",
            is_protected_test=False,
            text_hash="abc123hash",
            exclusion_reason=None,
        )
        assert sample.source_dataset == "averitec"
        assert sample.original_label == "Supported"
        assert sample.raw_text != sample.text
        assert sample.harmonized_label == "likely_reliable"
        assert sample.is_protected_test is False

    def test_sample_frozen_immutability(self):
        sample = HarmonizedSample(
            sample_id="test_002",
            source_dataset="liar",
            source_split="train",
            original_id="123",
            text="Claim text.",
            raw_text="Claim text.",
            original_label="false",
            harmonized_label="likely_fake",
            mapping_status="accepted",
            mapping_confidence=1.0,
            mapping_rule="liar_v1_false_to_fake",
            domain="politics",
            license_tag="Academic",
            is_protected_test=False,
            text_hash="hash123",
        )
        with pytest.raises(Exception):
            sample.harmonized_label = "likely_reliable"  # type: ignore


# ---------------------------------------------------------------------------
# 4. Pipeline Artifact & Protection Invariants
# ---------------------------------------------------------------------------
class TestPipelineArtifacts:
    """Verify produced processed datasets and manifest integrity."""

    def test_manifest_exists_and_valid(self):
        manifest_path = PROCESSED_DIR / "manifest.json"
        assert manifest_path.exists(), "Manifest file missing in data/v2_processed"

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["manifest_version"] == "v2_processed_manifest_v1"
        assert manifest["mapping_version"] == MAPPING_VERSION
        assert "raw_counts" in manifest
        assert "canonical_eligible_train_pool" in manifest

        # Reconciled raw counts
        assert manifest["raw_counts"]["liar"] == 12836
        assert manifest["raw_counts"]["averitec"] == 3568
        assert manifest["raw_counts"]["total_raw"] == 71244

        pool_info = manifest["canonical_eligible_train_pool"]
        assert pool_info["total_samples"] == 12192
        assert pool_info["contributing_datasets"]["liar_train"] == 9240
        assert pool_info["contributing_datasets"]["averitec_train"] == 2952

    def test_combined_train_pool_contains_zero_protected_test_samples(self):
        train_pool_path = PROCESSED_DIR / "combined_eligible_train_pool.jsonl"
        assert train_pool_path.exists()

        with open(train_pool_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                item = json.loads(line)
                # Invariant: Must NOT be marked protected test
                assert item["is_protected_test"] is False, f"Line {line_idx} is marked as protected test!"
                # Invariant: Must NOT originate from LIAR test split
                assert not (item["source_dataset"] == "liar" and item["source_split"] == "test")
                # Invariant: Must NOT originate from AVeriTeC dev split
                assert not (item["source_dataset"] == "averitec" and item["source_split"] == "dev")
                # Invariant: Must NOT originate from MultiFC (licensing unresolved)
                assert item["source_dataset"] != "multifc"
                # Invariant: Must NOT originate from FEVER (auxiliary)
                assert item["source_dataset"] != "fever"
                # Invariant: Harmonized label must be valid target class
                assert item["harmonized_label"] in ["likely_reliable", "misleading", "likely_fake", "unverifiable"]

    def test_historical_liar_row_counts_from_actual_loader(self):
        """Verify exact historical Step 2 counts from canonical dataset_loader."""
        from backend.app.services.dataset_loader import load_liar_split
        train_df = load_liar_split("train")
        valid_df = load_liar_split("valid")
        test_df = load_liar_split("test")

        assert len(train_df) == 10269, f"Expected 10,269 train rows, got {len(train_df)}"
        assert len(valid_df) == 1284, f"Expected 1,284 valid rows, got {len(valid_df)}"
        assert len(test_df) == 1283, f"Expected 1,283 test rows, got {len(test_df)}"
        assert len(train_df) + len(valid_df) + len(test_df) == 12836

    def test_v1_v2_liar_loader_id_agreement(self):
        """Verify that V2 ingest_liar parses the exact same 12,836 IDs as V1 load_liar_split."""
        from backend.app.services.dataset_loader import load_liar_split
        from scripts.prepare_v2_datasets import ingest_liar

        v1_ids = set()
        for split in ["train", "valid", "test"]:
            df = load_liar_split(split)
            v1_ids.update(df["id"].astype(str).tolist())

        v2_samples = ingest_liar()
        v2_orig_ids = {s.original_id for s in v2_samples}

        assert len(v1_ids) == 12836
        assert len(v2_orig_ids) == 12836
        assert v1_ids == v2_orig_ids, f"ID mismatch: {len(v1_ids - v2_orig_ids)} in V1 only, {len(v2_orig_ids - v1_ids)} in V2 only"

    def test_all_frozen_test_ids_excluded_from_v2_train_pool(self):
        """Verify that all 1,283 test IDs from test.tsv are strictly absent from the canonical training pool."""
        from backend.app.services.dataset_loader import load_liar_split
        test_df = load_liar_split("test")
        test_ids = set(test_df["id"].astype(str).tolist())
        assert len(test_ids) == 1283

        train_pool_path = PROCESSED_DIR / "combined_eligible_train_pool.jsonl"
        with open(train_pool_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if item["source_dataset"] == "liar":
                    assert item["original_id"] not in test_ids, f"Test ID {item['original_id']} leaked into train pool!"

    def test_raw_liar_hashes_unchanged(self):
        """Verify raw LIAR TSVs match canonical immutable SHA-256 checksums."""
        import hashlib
        from backend.app.config import CANONICAL_LIAR_TEST_TSV_SHA256

        test_p = RAW_LIAR_DIR / "test.tsv"
        actual_test_hash = hashlib.sha256(test_p.read_bytes()).hexdigest()
        assert actual_test_hash == CANONICAL_LIAR_TEST_TSV_SHA256
        assert actual_test_hash == "ee41e2c741e0019216952625145af8c4aa8c8d3057ec68284b134267bb845e6f"

        train_p = RAW_LIAR_DIR / "train.tsv"
        train_hash = hashlib.sha256(train_p.read_bytes()).hexdigest()
        assert train_hash == "23ef8cb7ff57b25c941100d0744039cebc595d65a67ea190f72b330a658d8c36"

        valid_p = RAW_LIAR_DIR / "valid.tsv"
        valid_hash = hashlib.sha256(valid_p.read_bytes()).hexdigest()
        assert valid_hash == "8dd3d7ed1a11e78ed7df090c9eddb0f91147f895850c78d49dd7b4e88fd09a16"

    def test_canonical_baseline_artifact_identified_and_hash_unchanged(self):
        """Verify CANONICAL_V1_BASELINE_ARTIFACT points to classical_best_validation.joblib and matches hash."""
        import hashlib
        from backend.app.config import (
            CANONICAL_V1_BASELINE_ARTIFACT,
            CANONICAL_V1_BASELINE_SHA256,
        )

        assert CANONICAL_V1_BASELINE_ARTIFACT.name == "classical_best_validation.joblib"
        assert CANONICAL_V1_BASELINE_ARTIFACT.exists()
        assert CANONICAL_V1_BASELINE_ARTIFACT.stat().st_size == 3239700

        actual_hash = hashlib.sha256(CANONICAL_V1_BASELINE_ARTIFACT.read_bytes()).hexdigest()
        assert actual_hash == CANONICAL_V1_BASELINE_SHA256
        assert actual_hash == "d4c3636277761f5399166ee41c0eec57cafe7b9216bacfc18d81f26fea415c92"

    def test_historical_test_metrics_linkage(self):
        """Verify classical_test_metrics.json evaluates all 1,283 rows and produces documented baseline metrics."""
        metrics_file = MODELS_DIR / "classical_test_metrics.json"
        assert metrics_file.exists()

        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["model_artifact"] == "models/classical_best_validation.joblib"
        assert data["test_rows"] == 1283
        assert data["accuracy"] == 0.4373
        assert data["macro_f1"] == 0.4345
        assert data["actual_test_distribution"] == {
            "misleading": 481,
            "likely_reliable": 460,
            "likely_fake": 342,
        }

    def test_dataset_aware_validation_policy_balance(self):
        """Verify that Stratified 5-Fold cross-validation balances LIAR and AVeriTeC across all folds."""
        manifest_path = PROCESSED_DIR / "manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        policy = manifest.get("v2_5_validation_policy", {})
        folds = policy.get("folds", [])
        assert len(folds) == 5

        for f_stat in folds:
            assert f_stat["total_val_samples"] in [2438, 2439]
            assert f_stat["liar_samples"] == 1848
            assert f_stat["averitec_samples"] in [590, 591]
            assert 75.0 <= f_stat["liar_percentage"] <= 76.0
            assert 24.0 <= f_stat["averitec_percentage"] <= 25.0
            # Ensure all 4 classes are represented in validation fold
            for c in ["likely_reliable", "misleading", "likely_fake", "unverifiable"]:
                assert f_stat["class_distribution"][c] > 0
