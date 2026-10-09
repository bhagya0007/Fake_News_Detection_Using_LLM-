"""Automated Tests for System Evaluation Suite (Step 19).

Verifies evaluation dataset integrity, evaluation metrics schema, fusion safety invariants,
database cleanliness, and canonical artifact hash preservation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List

import pytest

from backend.app.config import DATA_DIR, MODELS_DIR, PROJECT_ROOT
from backend.app.schemas import (
    Claim,
    ClaimType,
    EvidenceItem,
    EvidenceRetrievalResult,
    EvidenceStance,
    EvidenceSupport,
    InternalConsistency,
    LLMAssessment,
    MLAssessment,
    MLLabel,
    RedFlagFeatures,
    RedFlagResult,
    RedFlagSummary,
    SourceAttribution,
    VerdictLabel,
)
from backend.app.services.fusion import fuse

EVAL_DATA_DIR = PROJECT_ROOT / "data" / "evaluation"
REPORTS_DIR = PROJECT_ROOT / "reports" / "final_evaluation"

CANONICAL_TEST_TSV_SHA256 = "ee41e2c741e0019216952625145af8c4aa8c8d3057ec68284b134267bb845e6f"
CANONICAL_MODEL_SHA256 = "d4c3636277761f5399166ee41c0eec57cafe7b9216bacfc18d81f26fea415c92"


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


# =============================================================================
# 1. Dataset Integrity Tests
# =============================================================================
def test_controlled_benchmark_dataset_integrity() -> None:
    """Verify hybrid_controlled_eval.jsonl contains 44 balanced cases across 15 categories."""
    path = EVAL_DATA_DIR / "hybrid_controlled_eval.jsonl"
    assert path.exists(), f"Missing dataset: {path}"

    cases: List[Dict[str, Any]] = []
    verdicts: Dict[str, int] = {}
    categories: set[str] = set()

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            assert "case_id" in case
            assert "category" in case
            assert "input_text" in case
            assert "expected_verdict" in case
            assert "expected_abstained" in case
            assert "claims" in case

            v = case["expected_verdict"]
            verdicts[v] = verdicts.get(v, 0) + 1
            categories.add(case["category"])
            cases.append(case)

    assert len(cases) == 44, f"Expected 44 cases, got {len(cases)}"
    assert len(categories) >= 15, f"Expected at least 15 categories, got {len(categories)}"

    # Exact balance: 11 of each class
    for label in ["likely_reliable", "likely_fake", "misleading", "unverifiable"]:
        assert verdicts.get(label) == 11, f"Expected 11 cases for {label}, got {verdicts.get(label)}"


def test_production_abstention_dataset_integrity() -> None:
    """Verify production_abstention_eval.json contains 20 varied statements across domains."""
    path = EVAL_DATA_DIR / "production_abstention_eval.json"
    assert path.exists(), f"Missing dataset: {path}"

    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    assert len(cases) == 20
    domains = {c["domain"] for c in cases}
    assert len(domains) >= 10, f"Expected at least 10 domains, got {len(domains)}"
    for c in cases:
        assert "statement_id" in c
        assert "domain" in c
        assert "statement" in c
        assert "has_checkable_claims" in c


def test_claim_extraction_dataset_integrity() -> None:
    """Verify claim_extraction_eval.json contains 15 synthetic cases."""
    path = EVAL_DATA_DIR / "claim_extraction_eval.json"
    assert path.exists(), f"Missing dataset: {path}"

    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    assert len(cases) == 15
    for c in cases:
        assert "test_id" in c
        assert "text" in c
        assert "expected_claim_count" in c
        assert "expected_checkable" in c


def test_retrieval_dataset_integrity() -> None:
    """Verify retrieval_eval.json contains 16 queries with both in-domain and OOD queries."""
    path = EVAL_DATA_DIR / "retrieval_eval.json"
    assert path.exists(), f"Missing dataset: {path}"

    with open(path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    assert len(queries) == 16
    in_domain = [q for q in queries if not q["is_out_of_domain"]]
    out_domain = [q for q in queries if q["is_out_of_domain"]]
    assert len(in_domain) == 13
    assert len(out_domain) == 3


# =============================================================================
# 2. Evaluation Metrics Artifacts Tests
# =============================================================================
def test_evaluation_report_artifacts_exist() -> None:
    """Verify all 14 evaluation artifacts exist in reports/final_evaluation/."""
    expected_files = [
        "controlled_results.csv",
        "controlled_metrics.json",
        "controlled_confusion_matrix.csv",
        "production_behavior.csv",
        "production_behavior_summary.json",
        "classical_test_metrics.json",
        "classical_test_confusion_matrix.csv",
        "claim_extraction_results.csv",
        "claim_extraction_metrics.json",
        "retrieval_results.csv",
        "retrieval_metrics.json",
        "fusion_invariants.json",
        "performance_metrics.json",
        "final_evaluation_report.md",
    ]
    for filename in expected_files:
        p = REPORTS_DIR / filename
        assert p.exists(), f"Missing evaluation artifact: {p}"
        assert p.stat().st_size > 0, f"Empty artifact: {p}"


def test_controlled_metrics_safety_invariants() -> None:
    """Verify controlled benchmark achieved 0 unsafe decisions and 0 grounding violations."""
    metrics_path = REPORTS_DIR / "controlled_metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_cases"] == 44
    assert data["overall_accuracy"] == 1.0
    assert data["macro_f1"] == 1.0
    assert data["abstention_accuracy"] == 1.0
    assert data["unsafe_decisions"] == 0
    assert data["evidence_grounding_violations"] == 0
    assert data["synthetic_quarantine_violations"] == 0


def test_production_behavior_summary() -> None:
    """Verify Track B production behavior achieved 100% abstention and short-circuiting."""
    summary_path = REPORTS_DIR / "production_behavior_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_statements"] == 20
    assert data["abstention_count"] == 20
    assert data["abstention_rate"] == 1.0
    assert data["unverifiable_rate"] == 1.0
    assert data["mean_support_score"] == 0.0
    assert data["llm_short_circuit_rate"] == 1.0


def test_classical_test_metrics_values() -> None:
    """Verify Track C classical metrics match Step 7 frozen numbers (0.4373 / 0.4345)."""
    metrics_path = REPORTS_DIR / "classical_test_metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_test_samples"] == 1283
    assert data["test_accuracy"] == 0.4373
    assert data["test_macro_f1"] == 0.4345


def test_fusion_invariants_pass_all() -> None:
    """Verify all 10 fusion safety invariants passed."""
    inv_path = REPORTS_DIR / "fusion_invariants.json"
    with open(inv_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["all_invariants_passed"] is True
    assert data["total_invariants"] == 10
    assert data["passed_invariants"] == 10


def test_synthesis_report_streamlit_smoke_wording() -> None:
    """Verify report contains exact Streamlit smoke test wording."""
    report_path = REPORTS_DIR / "final_evaluation_report.md"
    content = report_path.read_text(encoding="utf-8")
    assert "Streamlit server started on port 8501." in content
    assert "Uvicorn server started on port 8501" not in content


# =============================================================================
# 3. Direct Fusion Safety Invariant Tests
# =============================================================================
def test_direct_fusion_invariant_empty_evidence_abstention() -> None:
    """Verify fuse() strictly abstains when evidence is None or empty."""
    ml_high = MLAssessment(
        label=MLLabel.likely_fake,
        probabilities={"likely_reliable": 0.01, "likely_fake": 0.98, "misleading": 0.01},
        model_name="test",
    )
    result = fuse(ml=ml_high, evidence=None, has_checkable_claims=True)
    assert result.verdict == VerdictLabel.unverifiable
    assert result.abstained is True
    assert result.support_score == 0.0


def test_direct_fusion_invariant_synthetic_quarantine() -> None:
    """Verify synthetic fixtures are rejected in production mode."""
    synth_item = EvidenceItem(
        id="synth_test",
        claim_summary="Test",
        evidence_text="Test",
        source_name="Demo",
        is_synthetic=True,
        stance=EvidenceStance.supports,
    )
    ev_res = EvidenceRetrievalResult(query="q", items=[synth_item], retrieval_backend="test", threshold=0.45)
    result = fuse(evidence=ev_res, allow_synthetic_evidence=False, has_checkable_claims=True)
    assert result.verdict == VerdictLabel.unverifiable
    assert result.abstained is True
    assert result.support_score == 0.0
    assert "synthetic_quarantine" in result.signals_used


def test_direct_fusion_invariant_support_score_equivalence() -> None:
    """Verify abstained == True strictly implies support_score == 0.0."""
    res_no_claims = fuse(has_checkable_claims=False)
    assert res_no_claims.abstained is True
    assert res_no_claims.support_score == 0.0


# =============================================================================
# 4. Canonical Integrity & Cleanliness Tests
# =============================================================================
def test_production_database_cleanliness() -> None:
    """Verify production database contains exactly 0 analyses and 0 feedback records."""
    db_path = DATA_DIR / "app" / "fake_news_analysis.db"
    assert db_path.exists()

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM analyses")
    analyses_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM feedback")
    feedback_count = cur.fetchone()[0]
    conn.close()

    assert analyses_count == 0, f"Expected 0 analyses in production DB, found {analyses_count}"
    assert feedback_count == 0, f"Expected 0 feedback in production DB, found {feedback_count}"


def test_frozen_artifacts_canonical_hashes() -> None:
    """Verify data/raw/test.tsv and classical_best_validation.joblib hashes are untouched."""
    test_tsv = DATA_DIR / "raw" / "test.tsv"
    model_joblib = MODELS_DIR / "classical_best_validation.joblib"

    assert test_tsv.exists()
    assert model_joblib.exists()

    tsv_hash = _compute_sha256(test_tsv)
    model_hash = _compute_sha256(model_joblib)

    assert tsv_hash == CANONICAL_TEST_TSV_SHA256, f"test.tsv hash changed: {tsv_hash}"
    assert model_hash == CANONICAL_MODEL_SHA256, f"model hash changed: {model_hash}"
