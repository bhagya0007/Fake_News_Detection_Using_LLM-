"""Unit tests for the Deterministic Hybrid Decision Fusion & Abstention Engine (Step 14).

Verifies all 35 required test scenarios:
1. No-checkable-claim abstention
2. No-evidence abstention
3. Synthetic-only evidence rejected in production mode
4. Synthetic evidence optionally allowed in explicit demo mode
5. Related evidence alone is insufficient (abstains)
6. Unknown stance alone is insufficient (abstains)
7. Supported evidence + reliable LLM produces likely_reliable
8. Contradicted evidence + fake LLM produces likely_fake
9. Partial evidence + misleading LLM produces misleading
10. LLM unavailable results in safe abstention
11. LLM unverifiable is strictly respected
12. ML agreement without evidence still abstains
13. ML disagreement with strong support does not flip verdict
14. ML disagreement with contradiction does not flip verdict
15. Red flags alone cannot produce likely_fake
16. Red flags alone cannot produce misleading
17. Red flags cannot overturn supported factual evidence
18. Support score is strictly bounded within [0.0, 1.0]
19. Abstained consistency is strictly maintained
20. signals_used is populated across all scenarios
21. Synthetic evidence count is accurate
22. Real evidence count is accurate
23. Original evidence objects are not mutated
24. Deterministic output guaranteed
25. Same input produces identical output
26. Zero external API calls made
27. Zero LLM client calls inside fusion
28. Zero retrieval engine calls inside fusion
29. Zero classical model inference calls inside fusion
30. No hidden chain-of-thought fields exist
31. Model probability treated as decision strength, not truth probability
32. Inconsistent signal states safely result in abstention
33. Empty evidence list handled gracefully
34. Component failure behavior safe via FusionInput
35. Network socket access strictly prohibited during tests
"""

from __future__ import annotations

import socket
from datetime import date
from typing import Any, List
from unittest.mock import patch

import pytest

from backend.app.schemas import (
    ClassProbabilities,
    ComponentExecutionStatus,
    ComponentStatus,
    EvidenceItem,
    EvidenceRetrievalResult,
    EvidenceStance,
    EvidenceSupport,
    FinalAssessment,
    FusionInput,
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
from backend.app.services.fusion import fuse, fuse_input, get_usable_real_evidence


# ---------------------------------------------------------------------------
# Offline Socket Guard (Guarantees zero network calls during tests)
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def block_external_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prohibit any network socket connections during unit test execution."""
    real_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Network connection prohibited in unit tests!")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


# ---------------------------------------------------------------------------
# Fixture Helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def base_ml_reliable() -> MLAssessment:
    return MLAssessment(
        label=MLLabel.likely_reliable,
        probabilities=ClassProbabilities(likely_reliable=0.80, likely_fake=0.10, misleading=0.10),
        model_name="Tuned TF-IDF + Logistic Regression",
    )


@pytest.fixture
def base_ml_fake() -> MLAssessment:
    return MLAssessment(
        label=MLLabel.likely_fake,
        probabilities=ClassProbabilities(likely_reliable=0.10, likely_fake=0.80, misleading=0.10),
        model_name="Tuned TF-IDF + Logistic Regression",
    )


@pytest.fixture
def base_ml_misleading() -> MLAssessment:
    return MLAssessment(
        label=MLLabel.misleading,
        probabilities=ClassProbabilities(likely_reliable=0.20, likely_fake=0.20, misleading=0.60),
        model_name="Tuned TF-IDF + Logistic Regression",
    )


@pytest.fixture
def clean_red_flags() -> RedFlagResult:
    features = RedFlagFeatures(
        character_count=50,
        word_count=8,
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
        attribution_phrase_count=0,
        quote_count=0,
        excessive_punctuation_ratio=0.0,
    )
    summary = RedFlagSummary(
        total_flags=0,
        low_severity_flags=0,
        medium_severity_flags=0,
        high_severity_flags=0,
    )
    return RedFlagResult(flags=[], features=features, summary=summary)


@pytest.fixture
def high_red_flags() -> RedFlagResult:
    features = RedFlagFeatures(
        character_count=100,
        word_count=15,
        uppercase_letter_ratio=0.35,
        exclamation_count=5,
        question_mark_count=0,
        repeated_punctuation_count=2,
        percentage_count=0,
        statistical_expression_count=0,
        url_count=0,
        sensational_phrase_count=3,
        certainty_phrase_count=1,
        urgency_phrase_count=2,
        attribution_phrase_count=0,
        quote_count=0,
        excessive_punctuation_ratio=0.08,
    )
    flags = [
        RedFlag(type="excessive_caps", severity=Severity.high, message="Excessive capitalization"),
        RedFlag(type="urgency_phrase", severity=Severity.high, message="Urgency calls detected"),
    ]
    summary = RedFlagSummary(
        total_flags=2,
        low_severity_flags=0,
        medium_severity_flags=0,
        high_severity_flags=2,
    )
    return RedFlagResult(flags=flags, features=features, summary=summary)


@pytest.fixture
def real_supported_evidence() -> EvidenceRetrievalResult:
    items = [
        EvidenceItem(
            id="real_001",
            claim_summary="Economic growth reached 2%",
            evidence_text="The Bureau of Economic Analysis confirmed GDP rose 2% this quarter.",
            source_name="BEA Official",
            source_url="https://bea.gov/gdp-report",
            similarity_score=0.91,
            stance=EvidenceStance.supports,
            is_synthetic=False,
        )
    ]
    return EvidenceRetrievalResult(
        query="Economic growth reached 2%",
        items=items,
        retrieval_backend="sentence_transformer",
        threshold=0.45,
        no_sufficient_evidence=False,
    )


@pytest.fixture
def real_contradicted_evidence() -> EvidenceRetrievalResult:
    items = [
        EvidenceItem(
            id="real_002",
            claim_summary="Vaccine causes immediate heart failure",
            evidence_text="Extensive clinical trials and CDC bulletins show no causal link to acute heart failure.",
            source_name="CDC Bulletin",
            source_url="https://cdc.gov/vaccine-safety",
            similarity_score=0.88,
            stance=EvidenceStance.contradicts,
            is_synthetic=False,
        )
    ]
    return EvidenceRetrievalResult(
        query="Vaccine causes immediate heart failure",
        items=items,
        retrieval_backend="sentence_transformer",
        threshold=0.45,
        no_sufficient_evidence=False,
    )


@pytest.fixture
def real_partial_evidence() -> EvidenceRetrievalResult:
    items = [
        EvidenceItem(
            id="real_003",
            claim_summary="Unemployment rose 15%",
            evidence_text="Unemployment rose 15% among seasonal workers, while national overall rate dropped 0.2%.",
            source_name="BLS Report",
            source_url="https://bls.gov/employment",
            similarity_score=0.84,
            stance=EvidenceStance.supports,
            is_synthetic=False,
        )
    ]
    return EvidenceRetrievalResult(
        query="Unemployment rose 15%",
        items=items,
        retrieval_backend="sentence_transformer",
        threshold=0.45,
        no_sufficient_evidence=False,
    )


@pytest.fixture
def synthetic_evidence() -> EvidenceRetrievalResult:
    items = [
        EvidenceItem(
            id="synth_001",
            claim_summary="Acme Corp factory in Texas",
            evidence_text="Acme Corp announced today that it will construct a new manufacturing facility in Texas.",
            source_name="Synthetic Demo Source",
            source_url="https://example.com/synthetic/acme",
            similarity_score=0.92,
            stance=EvidenceStance.supports,
            is_synthetic=True,
        )
    ]
    return EvidenceRetrievalResult(
        query="Acme Corp factory in Texas",
        items=items,
        retrieval_backend="sentence_transformer",
        threshold=0.45,
        no_sufficient_evidence=False,
    )


@pytest.fixture
def llm_reliable() -> LLMAssessment:
    return LLMAssessment(
        label=VerdictLabel.likely_reliable,
        source_attribution=SourceAttribution.strong,
        evidence_support=EvidenceSupport.supported,
        internal_consistency=InternalConsistency.consistent,
        short_reason="Directly corroborated by official statistics.",
    )


@pytest.fixture
def llm_fake() -> LLMAssessment:
    return LLMAssessment(
        label=VerdictLabel.likely_fake,
        source_attribution=SourceAttribution.weak,
        evidence_support=EvidenceSupport.contradicted,
        internal_consistency=InternalConsistency.consistent,
        short_reason="Refuted directly by verified medical bulletin.",
    )


@pytest.fixture
def llm_misleading() -> LLMAssessment:
    return LLMAssessment(
        label=VerdictLabel.misleading,
        source_attribution=SourceAttribution.partial,
        evidence_support=EvidenceSupport.partial,
        internal_consistency=InternalConsistency.questionable,
        short_reason="Selective focus on seasonal workers misrepresents overall figures.",
    )


# ---------------------------------------------------------------------------
# Test Cases (All 35 Scenarios)
# ---------------------------------------------------------------------------

def test_1_no_checkable_claim_abstention(
    base_ml_reliable: MLAssessment, clean_red_flags: RedFlagResult
) -> None:
    """1. No checkable claims returns unverifiable with abstained=True."""
    res = fuse(
        ml=base_ml_reliable,
        red_flags=clean_red_flags,
        has_checkable_claims=False,
    )
    assert res.verdict == VerdictLabel.unverifiable
    assert res.abstained is True
    assert "No checkable factual claim" in res.reason


def test_2_no_evidence_abstention(
    base_ml_reliable: MLAssessment, clean_red_flags: RedFlagResult
) -> None:
    """2. Externally checkable claim with no evidence returns unverifiable."""
    res = fuse(
        ml=base_ml_reliable,
        red_flags=clean_red_flags,
        evidence=None,
        has_checkable_claims=True,
    )
    assert res.verdict == VerdictLabel.unverifiable
    assert res.abstained is True
    assert "No usable provenance-bearing evidence was available" in res.reason


def test_3_synthetic_only_evidence_rejected_in_production(
    base_ml_reliable: MLAssessment,
    clean_red_flags: RedFlagResult,
    synthetic_evidence: EvidenceRetrievalResult,
    llm_reliable: LLMAssessment,
) -> None:
    """3. Synthetic fixtures are rejected in production mode (allow_synthetic_evidence=False)."""
    res = fuse(
        ml=base_ml_reliable,
        red_flags=clean_red_flags,
        llm=llm_reliable,
        evidence=synthetic_evidence,
        allow_synthetic_evidence=False,
    )
    assert res.verdict == VerdictLabel.unverifiable
    assert res.abstained is True
    assert "synthetic demo fixtures were rejected" in res.reason
    assert res.fusion_signals.synthetic_evidence_count == 1
    assert res.fusion_signals.real_evidence_count == 0


def test_4_synthetic_evidence_allowed_in_demo(
    base_ml_reliable: MLAssessment,
    clean_red_flags: RedFlagResult,
    synthetic_evidence: EvidenceRetrievalResult,
    llm_reliable: LLMAssessment,
) -> None:
    """4. Synthetic evidence is permitted when allow_synthetic_evidence=True and marked as demo."""
    res = fuse(
        ml=base_ml_reliable,
        red_flags=clean_red_flags,
        llm=llm_reliable,
        evidence=synthetic_evidence,
        allow_synthetic_evidence=True,
    )
    assert res.verdict == VerdictLabel.likely_reliable
    assert res.abstained is False
    assert "[DEMO / SYNTHETIC EVIDENCE]" in res.reason


def test_5_related_evidence_alone_insufficient(
    base_ml_reliable: MLAssessment,
    clean_red_flags: RedFlagResult,
    llm_reliable: LLMAssessment,
) -> None:
    """5. Evidence with only 'related' stance is insufficient for verification and abstains."""
    related_item = EvidenceItem(
        id="rel_001",
        claim_summary="Topic",
        evidence_text="Contextual snippet.",
        source_name="Gov Source",
        source_url="https://example.gov",
        stance=EvidenceStance.related,
        is_synthetic=False,
    )
    ev_result = EvidenceRetrievalResult(
        query="Topic",
        items=[related_item],
        retrieval_backend="sentence_transformer",
        threshold=0.45,
        no_sufficient_evidence=False,
    )
    res = fuse(
        ml=base_ml_reliable,
        red_flags=clean_red_flags,
        llm=llm_reliable,
        evidence=ev_result,
    )
    assert res.verdict == VerdictLabel.unverifiable
    assert res.abstained is True
    assert "only topically related or has an unknown stance" in res.reason


def test_6_unknown_stance_alone_insufficient(
    base_ml_reliable: MLAssessment,
    clean_red_flags: RedFlagResult,
    llm_reliable: LLMAssessment,
) -> None:
    """6. Evidence with only 'unknown' stance is insufficient and abstains."""
    unk_item = EvidenceItem(
        id="unk_001",
        claim_summary="Topic",
        evidence_text="Contextual snippet.",
        source_name="Gov Source",
        source_url="https://example.gov",
        stance=EvidenceStance.unknown,
        is_synthetic=False,
    )
    ev_result = EvidenceRetrievalResult(
        query="Topic",
        items=[unk_item],
        retrieval_backend="sentence_transformer",
        threshold=0.45,
        no_sufficient_evidence=False,
    )
    res = fuse(
        ml=base_ml_reliable,
        red_flags=clean_red_flags,
        llm=llm_reliable,
        evidence=ev_result,
    )
    assert res.verdict == VerdictLabel.unverifiable
    assert res.abstained is True


def test_7_supported_evidence_and_reliable_llm(
    base_ml_reliable: MLAssessment,
    clean_red_flags: RedFlagResult,
    real_supported_evidence: EvidenceRetrievalResult,
    llm_reliable: LLMAssessment,
) -> None:
    """7. Supported evidence + reliable LLM yields likely_reliable."""
    res = fuse(
        ml=base_ml_reliable,
        red_flags=clean_red_flags,
        llm=llm_reliable,
        evidence=real_supported_evidence,
    )
    assert res.verdict == VerdictLabel.likely_reliable
    assert res.abstained is False
    assert res.support_score >= 0.70


def test_8_contradicted_evidence_and_fake_llm(
    base_ml_fake: MLAssessment,
    clean_red_flags: RedFlagResult,
    real_contradicted_evidence: EvidenceRetrievalResult,
    llm_fake: LLMAssessment,
) -> None:
    """8. Contradicted evidence + fake LLM yields likely_fake."""
    res = fuse(
        ml=base_ml_fake,
        red_flags=clean_red_flags,
        llm=llm_fake,
        evidence=real_contradicted_evidence,
    )
    assert res.verdict == VerdictLabel.likely_fake
    assert res.abstained is False
    assert res.support_score >= 0.70


def test_9_partial_evidence_and_misleading_llm(
    base_ml_misleading: MLAssessment,
    clean_red_flags: RedFlagResult,
    real_partial_evidence: EvidenceRetrievalResult,
    llm_misleading: LLMAssessment,
) -> None:
    """9. Partial evidence + misleading LLM yields misleading."""
    res = fuse(
        ml=base_ml_misleading,
        red_flags=clean_red_flags,
        llm=llm_misleading,
        evidence=real_partial_evidence,
    )
    assert res.verdict == VerdictLabel.misleading
    assert res.abstained is False


def test_10_llm_unavailable_abstains(
    base_ml_reliable: MLAssessment,
    clean_red_flags: RedFlagResult,
    real_supported_evidence: EvidenceRetrievalResult,
) -> None:
    """10. When LLM reasoner is unavailable (llm=None), the system safely abstains."""
    res = fuse(
        ml=base_ml_reliable,
        red_flags=clean_red_flags,
        llm=None,
        evidence=real_supported_evidence,
    )
    assert res.verdict == VerdictLabel.unverifiable
    assert res.abstained is True
    assert "LLM reasoning component was unavailable" in res.reason


def test_11_llm_unverifiable_respected(
    base_ml_reliable: MLAssessment,
    clean_red_flags: RedFlagResult,
    real_supported_evidence: EvidenceRetrievalResult,
) -> None:
    """11. When LLM reasoner returns unverifiable, fusion respects it and abstains."""
    llm_unv = LLMAssessment(
        label=VerdictLabel.unverifiable,
        source_attribution=SourceAttribution.weak,
        evidence_support=EvidenceSupport.unavailable,
        internal_consistency=InternalConsistency.consistent,
        short_reason="Evidence incomplete.",
    )
    res = fuse(
        ml=base_ml_reliable,
        red_flags=clean_red_flags,
        llm=llm_unv,
        evidence=real_supported_evidence,
    )
    assert res.verdict == VerdictLabel.unverifiable
    assert res.abstained is True


def test_12_ml_agreement_without_evidence_still_abstains(
    base_ml_fake: MLAssessment,
    clean_red_flags: RedFlagResult,
) -> None:
    """12. Agreement between ML and LLM without evidence is NOT verification; system abstains."""
    llm_no_ev_fake = LLMAssessment(
        label=VerdictLabel.likely_fake,
        source_attribution=SourceAttribution.none,
        evidence_support=EvidenceSupport.unavailable,
        internal_consistency=InternalConsistency.consistent,
        short_reason="Pattern matches known hoax style.",
    )
    res = fuse(
        ml=base_ml_fake,
        red_flags=clean_red_flags,
        llm=llm_no_ev_fake,
        evidence=None,
    )
    assert res.verdict == VerdictLabel.unverifiable
    assert res.abstained is True


def test_13_ml_disagreement_with_strong_support_does_not_flip(
    base_ml_fake: MLAssessment,
    clean_red_flags: RedFlagResult,
    real_supported_evidence: EvidenceRetrievalResult,
    llm_reliable: LLMAssessment,
) -> None:
    """13. ML predicting likely_fake cannot flip evidence-supported likely_reliable."""
    res = fuse(
        ml=base_ml_fake,
        red_flags=clean_red_flags,
        llm=llm_reliable,
        evidence=real_supported_evidence,
    )
    assert res.verdict == VerdictLabel.likely_reliable
    assert res.abstained is False
    assert "ml_disagreement" in res.signals_used
    assert "factual evidence took precedence" in res.reason


def test_14_ml_disagreement_with_contradiction_does_not_flip(
    base_ml_reliable: MLAssessment,
    clean_red_flags: RedFlagResult,
    real_contradicted_evidence: EvidenceRetrievalResult,
    llm_fake: LLMAssessment,
) -> None:
    """14. ML predicting likely_reliable cannot flip evidence-contradicted likely_fake."""
    res = fuse(
        ml=base_ml_reliable,
        red_flags=clean_red_flags,
        llm=llm_fake,
        evidence=real_contradicted_evidence,
    )
    assert res.verdict == VerdictLabel.likely_fake
    assert res.abstained is False
    assert "ml_disagreement" in res.signals_used
    assert "overridden by factual evidence" in res.reason


def test_15_red_flags_alone_cannot_produce_fake(
    base_ml_fake: MLAssessment,
    high_red_flags: RedFlagResult,
) -> None:
    """15. Red flags alone without evidence cannot produce likely_fake (must abstain)."""
    res = fuse(
        ml=base_ml_fake,
        red_flags=high_red_flags,
        evidence=None,
    )
    assert res.verdict == VerdictLabel.unverifiable
    assert res.abstained is True


def test_16_red_flags_alone_cannot_produce_misleading(
    base_ml_misleading: MLAssessment,
    high_red_flags: RedFlagResult,
) -> None:
    """16. Red flags alone without evidence cannot produce misleading."""
    res = fuse(
        ml=base_ml_misleading,
        red_flags=high_red_flags,
        evidence=None,
    )
    assert res.verdict == VerdictLabel.unverifiable
    assert res.abstained is True


def test_17_red_flags_cannot_overturn_supported_evidence(
    base_ml_reliable: MLAssessment,
    high_red_flags: RedFlagResult,
    real_supported_evidence: EvidenceRetrievalResult,
    llm_reliable: LLMAssessment,
) -> None:
    """17. Sensational headlines (high red flags) with supported evidence remain likely_reliable."""
    res = fuse(
        ml=base_ml_reliable,
        red_flags=high_red_flags,
        llm=llm_reliable,
        evidence=real_supported_evidence,
    )
    assert res.verdict == VerdictLabel.likely_reliable
    assert res.abstained is False
    assert "red flags were detected" in res.reason
    assert res.support_score <= 0.85  # Slight penalty


def test_18_support_score_bounded_0_to_1(
    base_ml_reliable: MLAssessment,
    clean_red_flags: RedFlagResult,
    real_supported_evidence: EvidenceRetrievalResult,
    llm_reliable: LLMAssessment,
) -> None:
    """18. Support score is always strictly within [0.0, 1.0]."""
    res = fuse(
        ml=base_ml_reliable,
        red_flags=clean_red_flags,
        llm=llm_reliable,
        evidence=real_supported_evidence,
    )
    assert 0.0 <= res.support_score <= 1.0


def test_19_abstained_consistency(
    base_ml_reliable: MLAssessment,
    clean_red_flags: RedFlagResult,
) -> None:
    """19. FinalAssessment validation strictly enforces abstained == (verdict == unverifiable)."""
    res = fuse(ml=base_ml_reliable, red_flags=clean_red_flags, evidence=None)
    assert res.verdict == VerdictLabel.unverifiable
    assert res.abstained is True


def test_20_signals_used_populated(
    base_ml_reliable: MLAssessment,
    clean_red_flags: RedFlagResult,
    real_supported_evidence: EvidenceRetrievalResult,
    llm_reliable: LLMAssessment,
) -> None:
    """20. signals_used is populated with relevant pipeline identifiers."""
    res = fuse(
        ml=base_ml_reliable,
        red_flags=clean_red_flags,
        llm=llm_reliable,
        evidence=real_supported_evidence,
    )
    assert len(res.signals_used) > 0
    assert "evidence_support" in res.signals_used


def test_21_synthetic_evidence_count_correct(
    base_ml_reliable: MLAssessment,
    clean_red_flags: RedFlagResult,
    synthetic_evidence: EvidenceRetrievalResult,
) -> None:
    """21. Synthetic count in fusion_signals matches the number of synthetic items."""
    res = fuse(
        ml=base_ml_reliable,
        red_flags=clean_red_flags,
        evidence=synthetic_evidence,
    )
    assert res.fusion_signals.synthetic_evidence_count == 1
    assert res.fusion_signals.real_evidence_count == 0


def test_22_real_evidence_count_correct(
    base_ml_reliable: MLAssessment,
    clean_red_flags: RedFlagResult,
    real_supported_evidence: EvidenceRetrievalResult,
) -> None:
    """22. Real evidence count matches non-synthetic items."""
    res = fuse(
        ml=base_ml_reliable,
        red_flags=clean_red_flags,
        evidence=real_supported_evidence,
    )
    assert res.fusion_signals.real_evidence_count == 1
    assert res.fusion_signals.synthetic_evidence_count == 0


def test_23_original_evidence_objects_not_mutated(
    synthetic_evidence: EvidenceRetrievalResult,
) -> None:
    """23. get_usable_real_evidence does not mutate the original EvidenceRetrievalResult or items."""
    original_len = len(synthetic_evidence.items)
    usable = get_usable_real_evidence(synthetic_evidence, allow_synthetic_evidence=False)
    assert usable == []
    assert len(synthetic_evidence.items) == original_len


def test_24_deterministic_output(
    base_ml_reliable: MLAssessment,
    clean_red_flags: RedFlagResult,
    real_supported_evidence: EvidenceRetrievalResult,
    llm_reliable: LLMAssessment,
) -> None:
    """24. Calling fuse multiple times produces bitwise deterministic output."""
    res1 = fuse(
        ml=base_ml_reliable,
        red_flags=clean_red_flags,
        llm=llm_reliable,
        evidence=real_supported_evidence,
    )
    res2 = fuse(
        ml=base_ml_reliable,
        red_flags=clean_red_flags,
        llm=llm_reliable,
        evidence=real_supported_evidence,
    )
    assert res1.verdict == res2.verdict
    assert res1.support_score == res2.support_score
    assert res1.reason == res2.reason
    assert res1.signals_used == res2.signals_used


def test_25_same_input_produces_same_result(
    base_ml_fake: MLAssessment,
    clean_red_flags: RedFlagResult,
    real_contradicted_evidence: EvidenceRetrievalResult,
    llm_fake: LLMAssessment,
) -> None:
    """25. Identical inputs produce identical outputs across different execution instances."""
    res1 = fuse(ml=base_ml_fake, red_flags=clean_red_flags, llm=llm_fake, evidence=real_contradicted_evidence)
    res2 = fuse(ml=base_ml_fake, red_flags=clean_red_flags, llm=llm_fake, evidence=real_contradicted_evidence)
    assert res1.model_dump() == res2.model_dump()


def test_26_zero_api_calls_made(
    base_ml_reliable: MLAssessment,
    clean_red_flags: RedFlagResult,
) -> None:
    """26. Fusion executes completely offline with zero API calls."""
    with patch("urllib.request.urlopen") as mock_url:
        fuse(ml=base_ml_reliable, red_flags=clean_red_flags)
        mock_url.assert_not_called()


def test_27_zero_llm_client_calls_inside_fusion(
    base_ml_reliable: MLAssessment,
    clean_red_flags: RedFlagResult,
) -> None:
    """27. Fusion does not call LLMClient."""
    with patch("backend.app.ai.llm_client.LLMClient") as mock_client:
        fuse(ml=base_ml_reliable, red_flags=clean_red_flags)
        mock_client.assert_not_called()


def test_28_zero_retrieval_calls_inside_fusion(
    base_ml_reliable: MLAssessment,
    clean_red_flags: RedFlagResult,
) -> None:
    """28. Fusion does not instantiate or call EvidenceRetriever."""
    with patch("backend.app.services.evidence_retriever.EvidenceRetriever") as mock_retriever:
        fuse(ml=base_ml_reliable, red_flags=clean_red_flags)
        mock_retriever.assert_not_called()


def test_29_zero_classical_inference_calls_inside_fusion(
    base_ml_reliable: MLAssessment,
    clean_red_flags: RedFlagResult,
) -> None:
    """29. Fusion does not run predict_baseline."""
    with patch("backend.app.services.baseline_model.predict_baseline") as mock_pred:
        fuse(ml=base_ml_reliable, red_flags=clean_red_flags)
        mock_pred.assert_not_called()


def test_30_no_chain_of_thought_fields_in_output(
    base_ml_reliable: MLAssessment,
    clean_red_flags: RedFlagResult,
    real_supported_evidence: EvidenceRetrievalResult,
    llm_reliable: LLMAssessment,
) -> None:
    """30. Output FinalAssessment contains no hidden chain-of-thought fields."""
    res = fuse(
        ml=base_ml_reliable,
        red_flags=clean_red_flags,
        llm=llm_reliable,
        evidence=real_supported_evidence,
    )
    prohibited = {"chain_of_thought", "cot", "reasoning_steps", "private_reasoning"}
    assert prohibited.isdisjoint(set(res.model_dump().keys()))


def test_31_model_probability_treated_as_score_not_calibrated_truth(
    clean_red_flags: RedFlagResult,
    real_supported_evidence: EvidenceRetrievalResult,
    llm_reliable: LLMAssessment,
) -> None:
    """31. Varying ML model probability modulates the internal decision score without being treated as real-world probability."""
    ml_high = MLAssessment(
        label=MLLabel.likely_reliable,
        probabilities=ClassProbabilities(likely_reliable=0.95, likely_fake=0.03, misleading=0.02),
        model_name="Test",
    )
    ml_low = MLAssessment(
        label=MLLabel.likely_reliable,
        probabilities=ClassProbabilities(likely_reliable=0.40, likely_fake=0.30, misleading=0.30),
        model_name="Test",
    )
    res_high = fuse(ml=ml_high, red_flags=clean_red_flags, llm=llm_reliable, evidence=real_supported_evidence)
    res_low = fuse(ml=ml_low, red_flags=clean_red_flags, llm=llm_reliable, evidence=real_supported_evidence)
    assert res_high.support_score > res_low.support_score


def test_32_inconsistent_signals_safely_abstain(
    base_ml_reliable: MLAssessment,
    clean_red_flags: RedFlagResult,
    real_contradicted_evidence: EvidenceRetrievalResult,
) -> None:
    """32. Inconsistent LLM claims (e.g. likely_reliable label despite contradicted evidence) result in safe abstention."""
    llm_inconsistent = LLMAssessment(
        label=VerdictLabel.likely_reliable,
        source_attribution=SourceAttribution.strong,
        evidence_support=EvidenceSupport.contradicted,  # Incompatible!
        internal_consistency=InternalConsistency.consistent,
        short_reason="Contradicted by evidence but marked reliable.",
    )
    res = fuse(
        ml=base_ml_reliable,
        red_flags=clean_red_flags,
        llm=llm_inconsistent,
        evidence=real_contradicted_evidence,
    )
    assert res.verdict == VerdictLabel.unverifiable
    assert res.abstained is True
    assert "signal_conflict" in res.signals_used


def test_33_empty_evidence_list_handled(
    base_ml_reliable: MLAssessment,
    clean_red_flags: RedFlagResult,
) -> None:
    """33. EvidenceRetrievalResult with empty items list abstains cleanly."""
    empty_ev = EvidenceRetrievalResult(
        query="Test query",
        items=[],
        retrieval_backend="sentence_transformer",
        threshold=0.45,
        no_sufficient_evidence=True,
    )
    res = fuse(
        ml=base_ml_reliable,
        red_flags=clean_red_flags,
        evidence=empty_ev,
    )
    assert res.verdict == VerdictLabel.unverifiable
    assert res.abstained is True


def test_34_component_failure_safe_via_fusion_input(
    base_ml_reliable: MLAssessment,
    clean_red_flags: RedFlagResult,
    real_supported_evidence: EvidenceRetrievalResult,
) -> None:
    """34. Component failure indicated by llm_available=False in FusionInput abstains safely."""
    inp = FusionInput(
        ml=base_ml_reliable,
        red_flags=clean_red_flags,
        llm=None,
        evidence=real_supported_evidence,
        has_checkable_claims=True,
        llm_available=False,
        component_statuses=[
            ComponentStatus(
                component="llm",
                status=ComponentExecutionStatus.unavailable,
                message="Quota limit reached",
            )
        ],
    )
    res = fuse_input(inp)
    assert res.verdict == VerdictLabel.unverifiable
    assert res.abstained is True


def test_35_network_socket_blocked() -> None:
    """35. Proving that any attempt to open a network socket in tests raises RuntimeError."""
    with pytest.raises(RuntimeError, match="Network connection prohibited"):
        s = socket.socket()
        s.connect(("1.1.1.1", 80))
