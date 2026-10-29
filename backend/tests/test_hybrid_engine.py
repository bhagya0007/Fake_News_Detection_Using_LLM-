"""Unit and Integration Tests for End-to-End HybridAnalysisEngine (Step 15).

Tests all 42 required scenarios:
- Input validation (None, non-string, empty, whitespace, oversized)
- Text preservation (original vs cleaned)
- Component calling and signal passing (ClaimExtractor, ML, RedFlags, Retriever, LLM, Fusion)
- Evidence retrieval per checkable claim, deduplication, skipping on 0 claims
- Evidence gate: no evidence -> unverifiable
- Synthetic evidence quarantine (rejected in production, enabled in demo)
- Graceful degradation under failures (LLM quota 429, timeout, retrieval, ML, red flags, claim extractor)
- Component execution statuses tracking
- Non-negative timing instrumentation
- Bounded support score (NOT named probability)
- Final verdict single authority in fusion
- Determinism and isolation (autouse socket guard, frozen assets untouched)
"""

from __future__ import annotations

import hashlib
import socket
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pytest
from pydantic import ValidationError

from backend.app.config import DATA_DIR, MODELS_DIR
from backend.app.schemas import (
    AnalysisResponse,
    Claim,
    ClaimAssessment,
    ClaimExtractionResult,
    ClaimResult,
    ClassProbabilities,
    ComponentExecutionStatus,
    ComponentStatus,
    EvidenceItem,
    EvidenceRetrievalResult,
    EvidenceStance,
    EvidenceSupport,
    FinalAssessment,
    InternalConsistency,
    LLMAssessment,
    MLAssessment,
    MLLabel,
    RedFlag,
    RedFlagFeatures,
    RedFlagResult,
    RedFlagSummary,
    Severity,
    SourceAttribution,
    VerdictLabel,
)
from backend.app.services.hybrid_engine import (
    DefaultMLService,
    DefaultRedFlagService,
    HybridAnalysisEngine,
)


# ---------------------------------------------------------------------------
# Network Isolation Guard (Autouse)
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def block_network_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strict guard: prohibit any network connections during unit testing."""

    def disabled_socket(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Prohibited real network call during test execution!")

    monkeypatch.setattr(socket, "socket", disabled_socket)
    monkeypatch.setenv("LLM_PROVIDER", "mock")


# ---------------------------------------------------------------------------
# Test Doubles / Fakes
# ---------------------------------------------------------------------------
class FakeClaimExtractor:
    def __init__(self, claims: Optional[list[Claim]] = None, should_fail: bool = False, fail_exc: Optional[Exception] = None) -> None:
        self.calls: list[str] = []
        self.claims = claims if claims is not None else [
            Claim(id=1, text="Test claim proposition.", checkable=True)
        ]
        self.should_fail = should_fail
        self.fail_exc = fail_exc or RuntimeError("LLM extraction failed")

    def extract_claims(self, text: str) -> ClaimExtractionResult:
        self.calls.append(text)
        if self.should_fail:
            raise self.fail_exc
        return ClaimExtractionResult(claims=self.claims, source_text_length=len(text))


class FakeMLService:
    def __init__(self, assessment: Optional[MLAssessment] = None, should_fail: bool = False) -> None:
        self.calls: list[str] = []
        self.should_fail = should_fail
        if assessment is not None:
            self.assessment = assessment
        else:
            probs = ClassProbabilities(likely_fake=0.1, likely_reliable=0.85, misleading=0.05)
            self.assessment = MLAssessment(
                label=MLLabel.likely_reliable,
                probabilities=probs,
                model_name="Tuned TF-IDF + Logistic Regression",
                model_version="validation_frozen",
            )

    def predict(self, text: str) -> MLAssessment:
        self.calls.append(text)
        if self.should_fail:
            raise RuntimeError("ML model pipeline inference error")
        return self.assessment


class FakeRedFlagService:
    def __init__(self, result: Optional[RedFlagResult] = None, should_fail: bool = False) -> None:
        self.calls: list[str] = []
        self.should_fail = should_fail
        if result is not None:
            self.result = result
        else:
            features = RedFlagFeatures(
                character_count=50,
                word_count=10,
                uppercase_letter_ratio=0.0,
                exclamation_count=0,
                question_mark_count=0,
                repeated_punctuation_count=0,
                percentage_count=0,
                statistical_expression_count=0,
                url_count=0,
                sensational_phrase_count=0,
                certainty_phrase_count=0,
                urgency_phrase_count=0,
                attribution_phrase_count=1,
                quote_count=0,
                excessive_punctuation_ratio=0.0,
            )
            summary = RedFlagSummary(
                total_flags=0,
                high_severity_flags=0,
                medium_severity_flags=0,
                low_severity_flags=0,
                has_attribution=True,
                has_statistical_claim=False,
            )
            self.result = RedFlagResult(flags=[], features=features, summary=summary)

    def analyze(self, text: str) -> RedFlagResult:
        self.calls.append(text)
        if self.should_fail:
            raise RuntimeError("Red-flag regex service failed")
        return self.result


class FakeEvidenceRetriever:
    def __init__(self, items: Optional[list[EvidenceItem]] = None, should_fail: bool = False) -> None:
        self.queries: list[str] = []
        self.items = items if items is not None else []
        self.should_fail = should_fail
        self.backend_name = "fake_retriever"
        self.threshold = 0.45

    def retrieve(self, claim_text: str, top_k: int = 3) -> EvidenceRetrievalResult:
        self.queries.append(claim_text)
        if self.should_fail:
            raise RuntimeError("Evidence index corrupted or disk I/O error")
        return EvidenceRetrievalResult(
            query=claim_text,
            items=self.items[:top_k],
            retrieval_backend=self.backend_name,
            threshold=self.threshold,
            no_sufficient_evidence=len(self.items) == 0,
        )


class FakeLLMReasoner:
    def __init__(
        self,
        assessment: Optional[LLMAssessment] = None,
        should_fail: bool = False,
        fail_exc: Optional[Exception] = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.should_fail = should_fail
        self.fail_exc = fail_exc or RuntimeError("LLM reasoner failure")
        if assessment is not None:
            self.assessment = assessment
        else:
            self.assessment = LLMAssessment(
                label=VerdictLabel.likely_reliable,
                source_attribution=SourceAttribution.strong,
                evidence_support=EvidenceSupport.supported,
                internal_consistency=InternalConsistency.consistent,
                short_reason="Corroborated by verified evidence.",
            )

    def assess(self, text: str, claims: list[Claim], evidence: Optional[list[EvidenceItem]] = None) -> LLMAssessment:
        self.calls.append({"text": text, "claims": claims, "evidence": evidence})
        if self.should_fail:
            raise self.fail_exc
        return self.assessment


# ---------------------------------------------------------------------------
# Test 1–7: Engine Init and Input Validation
# ---------------------------------------------------------------------------
def test_1_engine_initialization() -> None:
    """Test 1: Engine initializes cleanly with custom or default components."""
    fake_ext = FakeClaimExtractor()
    engine = HybridAnalysisEngine(claim_extractor=fake_ext)
    assert engine.claim_extractor is fake_ext
    assert engine.allow_synthetic_evidence is False
    assert engine.top_k == 3


def test_2_valid_text_input() -> None:
    """Test 2: Valid text input runs through the pipeline and returns AnalysisResponse."""
    engine = HybridAnalysisEngine(
        claim_extractor=FakeClaimExtractor(),
        ml_service=FakeMLService(),
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=FakeEvidenceRetriever(),
        llm_reasoner=FakeLLMReasoner(),
    )
    resp = engine.analyze_text("NASA confirmed that water ice exists on the Moon's south pole.")
    assert isinstance(resp, AnalysisResponse)
    assert resp.input_text == "NASA confirmed that water ice exists on the Moon's south pole."
    assert resp.verdict in {VerdictLabel.likely_reliable, VerdictLabel.unverifiable}


def test_3_none_input_rejected() -> None:
    """Test 3: None input raises ValueError."""
    engine = HybridAnalysisEngine(claim_extractor=FakeClaimExtractor())
    with pytest.raises(ValueError, match="cannot be None"):
        engine.analyze_text(None)  # type: ignore[arg-type]


def test_4_non_string_input_rejected() -> None:
    """Test 4: Non-string input raises TypeError."""
    engine = HybridAnalysisEngine(claim_extractor=FakeClaimExtractor())
    with pytest.raises(TypeError, match="must be a string"):
        engine.analyze_text(12345)  # type: ignore[arg-type]


def test_5_empty_input_rejected() -> None:
    """Test 5: Empty string input raises ValueError."""
    engine = HybridAnalysisEngine(claim_extractor=FakeClaimExtractor())
    with pytest.raises(ValueError, match="cannot be empty"):
        engine.analyze_text("")


def test_6_whitespace_only_input_rejected() -> None:
    """Test 6: Whitespace-only input raises ValueError."""
    engine = HybridAnalysisEngine(claim_extractor=FakeClaimExtractor())
    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        engine.analyze_text("   \n\t  ")


def test_7_oversized_input_rejected() -> None:
    """Test 7: Input text exceeding 20,000 characters is rejected without truncation."""
    engine = HybridAnalysisEngine(claim_extractor=FakeClaimExtractor())
    oversized = "A" * 20_001
    with pytest.raises(ValueError, match="exceeds maximum allowed length"):
        engine.analyze_text(oversized)


# ---------------------------------------------------------------------------
# Test 8–12: Text Preservation and Calling Order
# ---------------------------------------------------------------------------
def test_8_original_text_preserved() -> None:
    """Test 8: Original text with erratic spacing and casing is preserved in response."""
    raw = "  BREAKING NEWS!!!   NASA finds water on   Mars???   "
    engine = HybridAnalysisEngine(
        claim_extractor=FakeClaimExtractor(),
        ml_service=FakeMLService(),
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=FakeEvidenceRetriever(),
        llm_reasoner=FakeLLMReasoner(),
    )
    resp = engine.analyze_text(raw)
    assert resp.input_text == raw


def test_9_cleaned_text_handled_correctly() -> None:
    """Test 9: Conservative cleaning normalizes spaces for ML and claims while preserving cues."""
    raw = "Water   detected on   Moon.\n\nDetails   below."
    claim_ext = FakeClaimExtractor()
    ml_svc = FakeMLService()
    engine = HybridAnalysisEngine(
        claim_extractor=claim_ext,
        ml_service=ml_svc,
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=FakeEvidenceRetriever(),
        llm_reasoner=FakeLLMReasoner(),
    )
    engine.analyze_text(raw)
    assert claim_ext.calls[0] == "Water detected on Moon. Details below."
    assert ml_svc.calls[0] == "Water detected on Moon. Details below."


def test_10_claim_extractor_called() -> None:
    """Test 10: ClaimExtractor is called exactly once with cleaned input."""
    claim_ext = FakeClaimExtractor()
    engine = HybridAnalysisEngine(
        claim_extractor=claim_ext,
        ml_service=FakeMLService(),
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=FakeEvidenceRetriever(),
        llm_reasoner=FakeLLMReasoner(),
    )
    engine.analyze_text("A simple test statement.")
    assert len(claim_ext.calls) == 1


def test_11_ml_service_called() -> None:
    """Test 11: Classical ML service is called exactly once with cleaned input."""
    ml_svc = FakeMLService()
    engine = HybridAnalysisEngine(
        claim_extractor=FakeClaimExtractor(),
        ml_service=ml_svc,
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=FakeEvidenceRetriever(),
        llm_reasoner=FakeLLMReasoner(),
    )
    engine.analyze_text("A simple test statement.")
    assert len(ml_svc.calls) == 1


def test_12_red_flags_called() -> None:
    """Test 12: Red flag service is called on the unnormalized raw text to preserve formatting."""
    raw = "   BREAKING NEWS!!!   "
    rf_svc = FakeRedFlagService()
    engine = HybridAnalysisEngine(
        claim_extractor=FakeClaimExtractor(),
        ml_service=FakeMLService(),
        red_flag_service=rf_svc,
        evidence_retriever=FakeEvidenceRetriever(),
        llm_reasoner=FakeLLMReasoner(),
    )
    engine.analyze_text(raw)
    assert rf_svc.calls[0] == raw


# ---------------------------------------------------------------------------
# Test 13–15: Evidence Retrieval & Deduplication
# ---------------------------------------------------------------------------
def test_13_retrieval_called_per_checkable_claim() -> None:
    """Test 13: Evidence retriever is queried using Claim.text for checkable claims."""
    claims = [
        Claim(id=1, text="Claim Alpha", checkable=True),
        Claim(id=2, text="Claim Beta", checkable=True),
        Claim(id=3, text="Opinion Gamma", checkable=False),
    ]
    retriever = FakeEvidenceRetriever()
    engine = HybridAnalysisEngine(
        claim_extractor=FakeClaimExtractor(claims=claims),
        ml_service=FakeMLService(),
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=retriever,
        llm_reasoner=FakeLLMReasoner(),
    )
    engine.analyze_text("Some statement.")
    assert retriever.queries == ["Claim Alpha", "Claim Beta"]


def test_14_no_claims_skips_retrieval() -> None:
    """Test 14: If no claims or only non-checkable claims exist, retrieval is skipped."""
    claims = [Claim(id=1, text="Chocolate ice cream is the best.", checkable=False)]
    retriever = FakeEvidenceRetriever()
    engine = HybridAnalysisEngine(
        claim_extractor=FakeClaimExtractor(claims=claims),
        ml_service=FakeMLService(),
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=retriever,
        llm_reasoner=FakeLLMReasoner(),
    )
    resp = engine.analyze_text("Chocolate ice cream is the best.")
    assert len(retriever.queries) == 0
    status_map = {s.component: s.status for s in resp.component_statuses}
    assert status_map["evidence_retrieval"] == ComponentExecutionStatus.skipped
    assert status_map["llm_reasoner"] == ComponentExecutionStatus.skipped


def test_15_evidence_deduplication() -> None:
    """Test 15: If the same evidence item is retrieved for multiple claims, it is deduplicated in response.evidence."""
    shared_ev = EvidenceItem(
        id="EV-100",
        claim_summary="Summary",
        evidence_text="Common verified statement text.",
        source_name="Gov Source",
        stance=EvidenceStance.supports,
        is_synthetic=False,
    )
    claims = [
        Claim(id=1, text="First claim.", checkable=True),
        Claim(id=2, text="Second claim.", checkable=True),
    ]
    retriever = FakeEvidenceRetriever(items=[shared_ev])
    engine = HybridAnalysisEngine(
        claim_extractor=FakeClaimExtractor(claims=claims),
        ml_service=FakeMLService(),
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=retriever,
        llm_reasoner=FakeLLMReasoner(),
    )
    resp = engine.analyze_text("Two claims text.")
    # In overall response, deduplicated to 1 item
    assert len(resp.evidence) == 1
    assert resp.evidence[0].id == "EV-100"
    # But per-claim evidence still shows association
    assert len(resp.claims[0].evidence) == 1
    assert len(resp.claims[1].evidence) == 1


# ---------------------------------------------------------------------------
# Test 16–18: Reasoner & Fusion Signal Flow
# ---------------------------------------------------------------------------
def test_16_llm_reasoner_receives_claims_and_evidence() -> None:
    """Test 16: LLM Reasoner is passed cleaned text, claims, and deduplicated evidence."""
    ev = EvidenceItem(
        id="EV-200",
        claim_summary="Test",
        evidence_text="Evidence text.",
        source_name="Source",
        stance=EvidenceStance.supports,
        is_synthetic=False,
    )
    claims = [Claim(id=1, text="Test claim.", checkable=True)]
    reasoner = FakeLLMReasoner()
    engine = HybridAnalysisEngine(
        claim_extractor=FakeClaimExtractor(claims=claims),
        ml_service=FakeMLService(),
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=FakeEvidenceRetriever(items=[ev]),
        llm_reasoner=reasoner,
    )
    engine.analyze_text("Test statement.")
    assert len(reasoner.calls) == 1
    assert reasoner.calls[0]["claims"] == claims
    assert len(reasoner.calls[0]["evidence"]) == 1


def test_17_fusion_receives_all_expected_signals() -> None:
    """Test 17: Fusion engine receives ML, red flags, LLM, evidence, and checkable claims flag."""
    captured: dict[str, Any] = {}

    def custom_fusion(**kwargs: Any) -> FinalAssessment:
        captured.update(kwargs)
        return FinalAssessment(
            verdict=VerdictLabel.unverifiable,
            support_score=0.0,
            reason="Custom fusion reason.",
            abstained=True,
            signals_used=["custom"],
        )

    claims = [Claim(id=1, text="Checkable claim.", checkable=True)]
    engine = HybridAnalysisEngine(
        claim_extractor=FakeClaimExtractor(claims=claims),
        ml_service=FakeMLService(),
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=FakeEvidenceRetriever(),
        llm_reasoner=FakeLLMReasoner(),
        fusion_engine=custom_fusion,
    )
    engine.analyze_text("Checkable statement.")
    assert "ml" in captured
    assert "red_flags" in captured
    assert "llm" in captured
    assert "evidence" in captured
    assert captured["has_checkable_claims"] is True


def test_18_final_verdict_comes_from_fusion() -> None:
    """Test 18: Engine strictly adopts the verdict and support score from fusion."""

    def custom_fusion(**kwargs: Any) -> FinalAssessment:
        return FinalAssessment(
            verdict=VerdictLabel.unverifiable,
            support_score=0.0,
            reason="Abstained by fusion authority.",
            abstained=True,
            signals_used=["fusion_test"],
        )

    engine = HybridAnalysisEngine(
        claim_extractor=FakeClaimExtractor(),
        ml_service=FakeMLService(),
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=FakeEvidenceRetriever(),
        llm_reasoner=FakeLLMReasoner(),
        fusion_engine=custom_fusion,
    )
    resp = engine.analyze_text("Any text.")
    assert resp.verdict == VerdictLabel.unverifiable
    assert resp.support_score == 0.0
    assert resp.reason == "Abstained by fusion authority."


# ---------------------------------------------------------------------------
# Test 19–22: Evidence Gating and Abstention
# ---------------------------------------------------------------------------
def test_19_no_evidence_abstains() -> None:
    """Test 19: Checkable claim with 0 evidence results in unverifiable and skips LLM reasoner."""
    reasoner = FakeLLMReasoner()
    engine = HybridAnalysisEngine(
        claim_extractor=FakeClaimExtractor(claims=[Claim(id=1, text="Moon is made of cheese.", checkable=True)]),
        ml_service=FakeMLService(),
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=FakeEvidenceRetriever(items=[]),
        llm_reasoner=reasoner,
    )
    resp = engine.analyze_text("The Moon is made of green cheese.")
    assert resp.verdict == VerdictLabel.unverifiable
    assert resp.support_score == 0.0
    assert len(reasoner.calls) == 0
    assert resp.llm is None
    status_map = {s.component: s.status for s in resp.component_statuses}
    assert status_map["llm_reasoner"] == ComponentExecutionStatus.skipped
    reasoner_status = next(s for s in resp.component_statuses if s.component == "llm_reasoner")
    assert "no usable external evidence" in reasoner_status.message


def test_20_ml_fake_plus_no_evidence_still_abstains() -> None:
    """Test 20: Even if classical ML predicts likely_fake, absence of evidence forces unverifiable and skips reasoner."""
    fake_probs = ClassProbabilities(likely_fake=0.95, likely_reliable=0.02, misleading=0.03)
    ml_fake = MLAssessment(
        label=MLLabel.likely_fake,
        probabilities=fake_probs,
        model_name="Tuned TF-IDF + Logistic Regression",
    )
    reasoner = FakeLLMReasoner()
    engine = HybridAnalysisEngine(
        claim_extractor=FakeClaimExtractor(claims=[Claim(id=1, text="Moon cheese claim.", checkable=True)]),
        ml_service=FakeMLService(assessment=ml_fake),
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=FakeEvidenceRetriever(items=[]),
        llm_reasoner=reasoner,
    )
    resp = engine.analyze_text("Moon cheese claim.")
    assert resp.verdict == VerdictLabel.unverifiable
    assert resp.support_score == 0.0
    assert len(reasoner.calls) == 0
    assert resp.llm is None
    status_map = {s.component: s.status for s in resp.component_statuses}
    assert status_map["llm_reasoner"] == ComponentExecutionStatus.skipped


def test_21_ml_reliable_plus_no_evidence_still_abstains() -> None:
    """Test 21: Even if classical ML predicts likely_reliable, absence of evidence forces unverifiable and skips reasoner."""
    rel_probs = ClassProbabilities(likely_fake=0.05, likely_reliable=0.90, misleading=0.05)
    ml_rel = MLAssessment(
        label=MLLabel.likely_reliable,
        probabilities=rel_probs,
        model_name="Tuned TF-IDF + Logistic Regression",
    )
    reasoner = FakeLLMReasoner()
    engine = HybridAnalysisEngine(
        claim_extractor=FakeClaimExtractor(claims=[Claim(id=1, text="Sensational claim.", checkable=True)]),
        ml_service=FakeMLService(assessment=ml_rel),
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=FakeEvidenceRetriever(items=[]),
        llm_reasoner=reasoner,
    )
    resp = engine.analyze_text("Sensational claim.")
    assert resp.verdict == VerdictLabel.unverifiable
    assert resp.support_score == 0.0
    assert len(reasoner.calls) == 0
    assert resp.llm is None
    status_map = {s.component: s.status for s in resp.component_statuses}
    assert status_map["llm_reasoner"] == ComponentExecutionStatus.skipped


def test_22_no_checkable_claims_abstains() -> None:
    """Test 22: Pure opinion input results in unverifiable without querying evidence or reasoner."""
    claims = [Claim(id=1, text="I think chocolate is superior to vanilla.", checkable=False)]
    retriever = FakeEvidenceRetriever()
    reasoner = FakeLLMReasoner()
    engine = HybridAnalysisEngine(
        claim_extractor=FakeClaimExtractor(claims=claims),
        ml_service=FakeMLService(),
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=retriever,
        llm_reasoner=reasoner,
    )
    resp = engine.analyze_text("I think chocolate is superior to vanilla.")
    assert resp.verdict == VerdictLabel.unverifiable
    assert resp.support_score == 0.0
    assert len(retriever.queries) == 0
    assert len(reasoner.calls) == 0


# ---------------------------------------------------------------------------
# Test 23–24: Synthetic Evidence Quarantine
# ---------------------------------------------------------------------------
def test_23_synthetic_evidence_rejected_in_production() -> None:
    """Test 23: In default production mode, synthetic demo evidence is rejected, causing short-circuit and abstention."""
    synth_item = EvidenceItem(
        id="SYNTH-1",
        claim_summary="Acme factory",
        evidence_text="Synthetic test record.",
        source_name="Synthetic Wire",
        stance=EvidenceStance.supports,
        is_synthetic=True,
    )
    claims = [Claim(id=1, text="Acme built a factory.", checkable=True)]
    reasoner = FakeLLMReasoner()
    engine = HybridAnalysisEngine(
        claim_extractor=FakeClaimExtractor(claims=claims),
        ml_service=FakeMLService(),
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=FakeEvidenceRetriever(items=[synth_item]),
        llm_reasoner=reasoner,
        allow_synthetic_evidence=False,  # Production default
    )
    resp = engine.analyze_text("Acme built a factory.")
    assert resp.verdict == VerdictLabel.unverifiable
    assert resp.support_score == 0.0
    assert "synthetic_quarantine" in resp.reason or "No usable provenance-bearing" in resp.reason
    assert len(reasoner.calls) == 0
    assert resp.llm is None
    status_map = {s.component: s.status for s in resp.component_statuses}
    assert status_map["llm_reasoner"] == ComponentExecutionStatus.skipped
    reasoner_status = next(s for s in resp.component_statuses if s.component == "llm_reasoner")
    assert "no usable external evidence" in reasoner_status.message


def test_24_synthetic_evidence_works_only_in_explicit_demo_mode() -> None:
    """Test 24: In explicit demo mode, synthetic evidence is usable and produces grounded verdict."""
    synth_item = EvidenceItem(
        id="SYNTH-2",
        claim_summary="Acme factory",
        evidence_text="Synthetic test record.",
        source_name="Synthetic Wire",
        stance=EvidenceStance.supports,
        is_synthetic=True,
    )
    claims = [Claim(id=1, text="Acme built a factory.", checkable=True)]
    reasoner = FakeLLMReasoner()
    engine = HybridAnalysisEngine.demo(
        claim_extractor=FakeClaimExtractor(claims=claims),
        ml_service=FakeMLService(),
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=FakeEvidenceRetriever(items=[synth_item]),
        llm_reasoner=reasoner,
    )
    resp = engine.analyze_text("Acme built a factory.")
    assert resp.verdict == VerdictLabel.likely_reliable
    assert resp.support_score > 0.0
    assert "[DEMO / SYNTHETIC EVIDENCE]" in resp.reason
    assert len(reasoner.calls) == 1
    assert resp.llm is not None
    status_map = {s.component: s.status for s in resp.component_statuses}
    assert status_map["llm_reasoner"] == ComponentExecutionStatus.success


# ---------------------------------------------------------------------------
# Test 25–30: Graceful Degradation & Failure Handling
# ---------------------------------------------------------------------------
def test_25_llm_quota_failure_safe_abstention() -> None:
    """Test 25: 429 quota exhaustion in LLM reasoner safely degrades to unverifiable without crashing."""
    ev = EvidenceItem(
        id="EV-300",
        claim_summary="Summary",
        evidence_text="Evidence text.",
        source_name="Gov Source",
        stance=EvidenceStance.supports,
        is_synthetic=False,
    )
    engine = HybridAnalysisEngine(
        claim_extractor=FakeClaimExtractor(),
        ml_service=FakeMLService(),
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=FakeEvidenceRetriever(items=[ev]),
        llm_reasoner=FakeLLMReasoner(
            should_fail=True,
            fail_exc=RuntimeError("GoogleAPIError: 429 ResourceExhausted: Quota exceeded"),
        ),
    )
    resp = engine.analyze_text("Valid text.")
    assert resp.verdict == VerdictLabel.unverifiable
    status_map = {s.component: s.status for s in resp.component_statuses}
    assert status_map["llm_reasoner"] == ComponentExecutionStatus.unavailable
    assert resp.llm is None


def test_26_llm_timeout_safe_abstention() -> None:
    """Test 26: Timeout in LLM reasoner safely degrades to unverifiable without crashing."""
    ev = EvidenceItem(
        id="EV-301",
        claim_summary="Summary",
        evidence_text="Evidence text.",
        source_name="Gov Source",
        stance=EvidenceStance.supports,
        is_synthetic=False,
    )
    engine = HybridAnalysisEngine(
        claim_extractor=FakeClaimExtractor(),
        ml_service=FakeMLService(),
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=FakeEvidenceRetriever(items=[ev]),
        llm_reasoner=FakeLLMReasoner(
            should_fail=True,
            fail_exc=TimeoutError("Request timed out after 30.0s"),
        ),
    )
    resp = engine.analyze_text("Valid text.")
    assert resp.verdict == VerdictLabel.unverifiable
    status_map = {s.component: s.status for s in resp.component_statuses}
    assert status_map["llm_reasoner"] == ComponentExecutionStatus.unavailable
    assert resp.llm is None


def test_27_evidence_retriever_failure_safe_abstention() -> None:
    """Test 27: Failure in evidence retriever safely degrades to unverifiable."""
    engine = HybridAnalysisEngine(
        claim_extractor=FakeClaimExtractor(),
        ml_service=FakeMLService(),
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=FakeEvidenceRetriever(should_fail=True),
        llm_reasoner=FakeLLMReasoner(),
    )
    resp = engine.analyze_text("Valid text.")
    assert resp.verdict == VerdictLabel.unverifiable
    status_map = {s.component: s.status for s in resp.component_statuses}
    assert status_map["evidence_retrieval"] == ComponentExecutionStatus.failed


def test_28_ml_failure_does_not_crash_pipeline() -> None:
    """Test 28: Classical ML failure records unavailable status and allows pipeline to continue."""
    ev = EvidenceItem(
        id="EV-302",
        claim_summary="Summary",
        evidence_text="Verified real evidence text.",
        source_name="Gov Source",
        stance=EvidenceStance.supports,
        is_synthetic=False,
    )
    engine = HybridAnalysisEngine(
        claim_extractor=FakeClaimExtractor(),
        ml_service=FakeMLService(should_fail=True),
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=FakeEvidenceRetriever(items=[ev]),
        llm_reasoner=FakeLLMReasoner(),
    )
    resp = engine.analyze_text("Valid text.")
    status_map = {s.component: s.status for s in resp.component_statuses}
    assert status_map["classical_ml"] == ComponentExecutionStatus.unavailable
    assert resp.ml is None
    # Because evidence + LLM are strong, pipeline still arrives at a valid verdict
    assert resp.verdict == VerdictLabel.likely_reliable


def test_29_red_flag_failure_does_not_crash_pipeline() -> None:
    """Test 29: Red flag failure records unavailable status and allows pipeline to continue."""
    ev = EvidenceItem(
        id="EV-303",
        claim_summary="Summary",
        evidence_text="Verified real evidence text.",
        source_name="Gov Source",
        stance=EvidenceStance.supports,
        is_synthetic=False,
    )
    engine = HybridAnalysisEngine(
        claim_extractor=FakeClaimExtractor(),
        ml_service=FakeMLService(),
        red_flag_service=FakeRedFlagService(should_fail=True),
        evidence_retriever=FakeEvidenceRetriever(items=[ev]),
        llm_reasoner=FakeLLMReasoner(),
    )
    resp = engine.analyze_text("Valid text.")
    status_map = {s.component: s.status for s in resp.component_statuses}
    assert status_map["red_flags"] == ComponentExecutionStatus.unavailable
    assert resp.red_flags is None
    assert resp.verdict == VerdictLabel.likely_reliable


def test_30_claim_extractor_failure_handled_safely() -> None:
    """Test 30: Failure in claim extractor safely yields unverifiable verdict without raising unhandled exception."""
    engine = HybridAnalysisEngine(
        claim_extractor=FakeClaimExtractor(should_fail=True),
        ml_service=FakeMLService(),
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=FakeEvidenceRetriever(),
        llm_reasoner=FakeLLMReasoner(),
    )
    resp = engine.analyze_text("Valid text.")
    assert resp.verdict == VerdictLabel.unverifiable
    status_map = {s.component: s.status for s in resp.component_statuses}
    assert status_map["claim_extraction"] == ComponentExecutionStatus.failed
    assert status_map["evidence_retrieval"] == ComponentExecutionStatus.skipped
    assert status_map["llm_reasoner"] == ComponentExecutionStatus.skipped


# ---------------------------------------------------------------------------
# Test 31–37: Telemetry, Schemas, and Isolation
# ---------------------------------------------------------------------------
def test_31_component_statuses_accurate() -> None:
    """Test 31: All 7 expected pipeline subsystems are present in component_statuses."""
    engine = HybridAnalysisEngine(
        claim_extractor=FakeClaimExtractor(),
        ml_service=FakeMLService(),
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=FakeEvidenceRetriever(),
        llm_reasoner=FakeLLMReasoner(),
    )
    resp = engine.analyze_text("Normal factual text.")
    components = [s.component for s in resp.component_statuses]
    expected = [
        "cleaning",
        "claim_extraction",
        "classical_ml",
        "red_flags",
        "evidence_retrieval",
        "llm_reasoner",
        "fusion",
    ]
    for exp in expected:
        assert exp in components


def test_32_timing_fields_non_negative() -> None:
    """Test 32: All timing fields in PipelineTimings are present and >= 0.0."""
    engine = HybridAnalysisEngine(
        claim_extractor=FakeClaimExtractor(),
        ml_service=FakeMLService(),
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=FakeEvidenceRetriever(),
        llm_reasoner=FakeLLMReasoner(),
    )
    resp = engine.analyze_text("Normal factual text.")
    assert resp.timings is not None
    assert resp.timings.cleaning_ms >= 0.0
    assert resp.timings.claim_extraction_ms >= 0.0
    assert resp.timings.ml_ms >= 0.0
    assert resp.timings.red_flags_ms >= 0.0
    assert resp.timings.retrieval_ms >= 0.0
    assert resp.timings.llm_ms >= 0.0
    assert resp.timings.fusion_ms >= 0.0
    assert resp.timings.total_ms >= 0.0


def test_33_support_score_bounded() -> None:
    """Test 33: support_score is strictly clamped in [0.0, 1.0]."""
    engine = HybridAnalysisEngine(
        claim_extractor=FakeClaimExtractor(),
        ml_service=FakeMLService(),
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=FakeEvidenceRetriever(),
        llm_reasoner=FakeLLMReasoner(),
    )
    resp = engine.analyze_text("Some text.")
    assert 0.0 <= resp.support_score <= 1.0


def test_34_support_score_not_named_probability() -> None:
    """Test 34: support_score field is named support_score, not truth_probability or confidence_probability."""
    engine = HybridAnalysisEngine(
        claim_extractor=FakeClaimExtractor(),
        ml_service=FakeMLService(),
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=FakeEvidenceRetriever(),
        llm_reasoner=FakeLLMReasoner(),
    )
    resp = engine.analyze_text("Some text.")
    assert hasattr(resp, "support_score")
    assert not hasattr(resp, "truth_probability")
    assert not hasattr(resp, "confidence_probability")


def test_35_analysis_response_schema_validation() -> None:
    """Test 35: AnalysisResponse validates and roundtrips through JSON serialization."""
    engine = HybridAnalysisEngine(
        claim_extractor=FakeClaimExtractor(),
        ml_service=FakeMLService(),
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=FakeEvidenceRetriever(),
        llm_reasoner=FakeLLMReasoner(),
    )
    resp = engine.analyze_text("Sample input.")
    json_data = resp.model_dump_json()
    restored = AnalysisResponse.model_validate_json(json_data)
    assert restored.verdict == resp.verdict
    assert restored.support_score == resp.support_score


def test_36_no_chain_of_thought_fields() -> None:
    """Test 36: Response and all child models contain zero hidden chain-of-thought fields."""
    engine = HybridAnalysisEngine(
        claim_extractor=FakeClaimExtractor(),
        ml_service=FakeMLService(),
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=FakeEvidenceRetriever(),
        llm_reasoner=FakeLLMReasoner(),
    )
    resp = engine.analyze_text("Sample text.")
    dumped = resp.model_dump()
    forbidden = ["chain_of_thought", "private_reasoning", "hidden_thought", "internal_thought"]
    for key in forbidden:
        assert key not in dumped
    if resp.llm is not None:
        llm_dump = resp.llm.model_dump()
        for key in forbidden:
            assert key not in llm_dump


def test_37_no_api_key_leakage() -> None:
    """Test 37: No API key or authorization header is exposed in component status messages or reason."""
    dummy_key = "AIzaSySecretApiKey123456789"
    engine = HybridAnalysisEngine(
        claim_extractor=FakeClaimExtractor(
            should_fail=True,
            fail_exc=RuntimeError(f"Error communicating with Gemini (key: {dummy_key})"),
        ),
        ml_service=FakeMLService(),
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=FakeEvidenceRetriever(),
        llm_reasoner=FakeLLMReasoner(),
    )
    resp = engine.analyze_text("Input text.")
    for cs in resp.component_statuses:
        if cs.message:
            assert dummy_key not in cs.message
    assert dummy_key not in resp.reason


# ---------------------------------------------------------------------------
# Test 38–42: Isolation, Integrity & Authority
# ---------------------------------------------------------------------------
def test_38_no_real_network_call() -> None:
    """Test 38: Confirmed by autouse socket guard (any attempt to connect would raise RuntimeError)."""
    engine = HybridAnalysisEngine(
        claim_extractor=FakeClaimExtractor(),
        ml_service=FakeMLService(),
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=FakeEvidenceRetriever(),
        llm_reasoner=FakeLLMReasoner(),
    )
    resp = engine.analyze_text("Offline verified text.")
    assert resp.verdict in {VerdictLabel.likely_reliable, VerdictLabel.unverifiable}


def test_39_no_test_tsv_access() -> None:
    """Test 39: SHA-256 of data/raw/test.tsv remains completely untouched."""
    test_tsv = DATA_DIR / "raw" / "test.tsv"
    if test_tsv.exists():
        expected_hash = "ee41e2c741e0019216952625145af8c4aa8c8d3057ec68284b134267bb845e6f"
        actual_hash = hashlib.sha256(test_tsv.read_bytes()).hexdigest()
        assert actual_hash == expected_hash


def test_40_classical_artifact_not_modified() -> None:
    """Test 40: SHA-256 of models/classical_best_validation.joblib is unmodified."""
    model_path = MODELS_DIR / "classical_best_validation.joblib"
    if model_path.exists():
        expected_hash = "d4c3636277761f5399166ee41c0eec57cafe7b9216bacfc18d81f26fea415c92"
        actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        assert actual_hash == expected_hash


def test_41_same_deterministic_mocked_input_gives_same_result() -> None:
    """Test 41: Running the engine twice on identical input with identical fakes yields bitwise identical verdict and score."""
    engine = HybridAnalysisEngine(
        claim_extractor=FakeClaimExtractor(),
        ml_service=FakeMLService(),
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=FakeEvidenceRetriever(),
        llm_reasoner=FakeLLMReasoner(),
    )
    resp1 = engine.analyze_text("Identical query.")
    resp2 = engine.analyze_text("Identical query.")
    assert resp1.verdict == resp2.verdict
    assert resp1.support_score == resp2.support_score
    assert resp1.reason == resp2.reason


def test_42_fusion_is_only_final_decision_authority() -> None:
    """Test 42: The engine never overrides fusion, even if classical ML predicts a conflicting label."""
    fake_probs = ClassProbabilities(likely_fake=0.99, likely_reliable=0.0, misleading=0.01)
    ml_fake = MLAssessment(
        label=MLLabel.likely_fake,
        probabilities=fake_probs,
        model_name="Tuned TF-IDF + Logistic Regression",
    )

    def authoritative_fusion(**kwargs: Any) -> FinalAssessment:
        # Fusion dictates the verdict, e.g. unverifiable because of missing evidence
        return FinalAssessment(
            verdict=VerdictLabel.unverifiable,
            support_score=0.0,
            reason="Fusion solely determines this verdict.",
            abstained=True,
            signals_used=["fusion_authority"],
        )

    engine = HybridAnalysisEngine(
        claim_extractor=FakeClaimExtractor(),
        ml_service=FakeMLService(assessment=ml_fake),
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=FakeEvidenceRetriever(),
        llm_reasoner=FakeLLMReasoner(),
        fusion_engine=authoritative_fusion,
    )
    resp = engine.analyze_text("Fictitious claim.")
    # Must NOT be 'likely_fake' just because ML predicted fake!
    assert resp.verdict == VerdictLabel.unverifiable
    assert resp.verdict.value != ml_fake.label.value


def test_43_step_15_5_no_evidence_short_circuit_invariants() -> None:
    """Test 43: Comprehensive validation of Step 15.5 quota-aware short-circuit invariants.
    
    1. Checkable claim with NO evidence:
       - Claim extractor runs.
       - Evidence retriever is queried.
       - Reasoner is NOT invoked (0 calls).
       - Reasoner status is 'skipped' with message 'Skipped because no usable external evidence was available.'.
       - resp.llm is None.
       - Fusion abstains -> unverifiable, support_score = 0.0.
    2. Real usable evidence present:
       - Reasoner is invoked (1 call).
       - Reasoner status is 'success'.
       - resp.llm is populated.
    """
    # 1. No evidence scenario
    reasoner_no_ev = FakeLLMReasoner()
    retriever_no_ev = FakeEvidenceRetriever(items=[])
    claim_extractor_no_ev = FakeClaimExtractor(
        claims=[Claim(id=1, text="Fact-checkable claim without KB match.", checkable=True)]
    )
    engine_no_ev = HybridAnalysisEngine(
        claim_extractor=claim_extractor_no_ev,
        ml_service=FakeMLService(),
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=retriever_no_ev,
        llm_reasoner=reasoner_no_ev,
        allow_synthetic_evidence=False,
    )
    resp_no_ev = engine_no_ev.analyze_text("Fact-checkable claim without KB match.")

    # Invariants:
    assert len(claim_extractor_no_ev.calls) == 1
    assert retriever_no_ev.queries == ["Fact-checkable claim without KB match."]
    assert len(reasoner_no_ev.calls) == 0, "LLM reasoner must NOT be called when no usable evidence exists!"
    assert resp_no_ev.llm is None
    status_map = {s.component: s.status for s in resp_no_ev.component_statuses}
    assert status_map["llm_reasoner"] == ComponentExecutionStatus.skipped
    reasoner_status = next(s for s in resp_no_ev.component_statuses if s.component == "llm_reasoner")
    assert reasoner_status.message == "Skipped because no usable external evidence was available."
    assert resp_no_ev.verdict == VerdictLabel.unverifiable
    assert resp_no_ev.support_score == 0.0

    # 2. Real evidence scenario
    real_ev = EvidenceItem(
        id="EV-REAL-1",
        claim_summary="Verified claim",
        evidence_text="Verified facts from authoritative wire.",
        source_name="Official Wire",
        stance=EvidenceStance.supports,
        is_synthetic=False,
    )
    reasoner_real = FakeLLMReasoner()
    retriever_real = FakeEvidenceRetriever(items=[real_ev])
    engine_real = HybridAnalysisEngine(
        claim_extractor=FakeClaimExtractor(
            claims=[Claim(id=1, text="Verified claim proposition.", checkable=True)]
        ),
        ml_service=FakeMLService(),
        red_flag_service=FakeRedFlagService(),
        evidence_retriever=retriever_real,
        llm_reasoner=reasoner_real,
        allow_synthetic_evidence=False,
    )
    resp_real = engine_real.analyze_text("Verified claim proposition.")

    assert len(reasoner_real.calls) == 1, "LLM reasoner must be invoked when usable evidence exists!"
    assert resp_real.llm is not None
    real_status_map = {s.component: s.status for s in resp_real.component_statuses}
    assert real_status_map["llm_reasoner"] == ComponentExecutionStatus.success
    assert resp_real.verdict != VerdictLabel.unverifiable

