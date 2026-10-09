"""Unit tests for the Evidence-Aware LLM Reasoner Service (Step 12).

Verifies structured reasoning against supplied evidence, strict adherence to the
unverifiable policy when evidence is absent, rubric dimension evaluations, schema
contract guarantees, zero hidden chain-of-thought, and complete mock network isolation.
"""

from datetime import date
from pathlib import Path
import socket
from typing import Any
import pytest
from pydantic import ValidationError

from backend.app.ai.exceptions import LLMResponseValidationError
from backend.app.ai.llm_client import LLMClient
from backend.app.ai.providers.mock_provider import MockLLMProvider
from backend.app.schemas import (
    MAX_ANALYSIS_TEXT_LENGTH,
    Claim,
    ClaimType,
    EvidenceItem,
    EvidenceStance,
    EvidenceSupport,
    InternalConsistency,
    LLMAssessment,
    LLMCallMetadata,
    SourceAttribution,
    VerdictLabel,
)
from backend.app.services.llm_reasoner import LLMReasoner


@pytest.fixture(autouse=True)
def guard_against_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strict guard: Intercept any external socket connections during test execution."""
    def forbidden_connect(self: socket.socket, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("External network connection attempted during test execution!")

    monkeypatch.setattr(socket.socket, "connect", forbidden_connect)


def make_assessment(
    label: VerdictLabel = VerdictLabel.unverifiable,
    source_attribution: SourceAttribution = SourceAttribution.none,
    evidence_support: EvidenceSupport = EvidenceSupport.unavailable,
    internal_consistency: InternalConsistency = InternalConsistency.consistent,
    rhetorical_cues: list[str] | None = None,
    unsupported_claims: list[str] | None = None,
    short_reason: str = "Test justification sentence.",
) -> LLMAssessment:
    """Helper to construct a valid LLMAssessment instance for mocking."""
    return LLMAssessment(
        label=label,
        source_attribution=source_attribution,
        evidence_support=evidence_support,
        internal_consistency=internal_consistency,
        rhetorical_cues=rhetorical_cues or [],
        unsupported_claims=unsupported_claims or [],
        short_reason=short_reason,
    )


# ---------------------------------------------------------------------------
# Test 1: Claim with No Evidence -> Unverifiable
# ---------------------------------------------------------------------------
def test_claim_with_no_evidence_evaluates_to_unverifiable() -> None:
    text = "Company X earned $10 billion last year."
    claims = [Claim(text="Company X earned $10 billion last year.", checkable=True, claim_type=ClaimType.numeric)]
    provider = MockLLMProvider()
    provider.queue_response(
        make_assessment(
            label=VerdictLabel.unverifiable,
            source_attribution=SourceAttribution.none,
            evidence_support=EvidenceSupport.unavailable,
            unsupported_claims=["Company X earned $10 billion last year."],
            short_reason="No external financial records or evidence were provided to substantiate this figure.",
        )
    )
    reasoner = LLMReasoner(llm_client=LLMClient(provider=provider))
    result = reasoner.assess(text, claims=claims, evidence=None)

    assert result.label == VerdictLabel.unverifiable
    assert result.evidence_support == EvidenceSupport.unavailable
    assert len(result.unsupported_claims) == 1


# ---------------------------------------------------------------------------
# Test 2: Clearly Supporting Supplied Evidence -> Likely Reliable
# ---------------------------------------------------------------------------
def test_supporting_evidence_evaluates_to_likely_reliable() -> None:
    text = "Acme announced a factory in Texas."
    claims = [Claim(text="Acme announced a factory in Texas.", checkable=True, claim_type=ClaimType.event)]
    evidence = [
        EvidenceItem(
            claim_summary="Acme factory announcement",
            evidence_text="Acme's official release states the company announced a Texas factory.",
            source_name="Acme Press Office",
            stance=EvidenceStance.supports,
        )
    ]
    provider = MockLLMProvider()
    provider.queue_response(
        make_assessment(
            label=VerdictLabel.likely_reliable,
            source_attribution=SourceAttribution.strong,
            evidence_support=EvidenceSupport.supported,
            short_reason="The official press release directly confirms the factory announcement.",
        )
    )
    reasoner = LLMReasoner(llm_client=LLMClient(provider=provider))
    result = reasoner.assess(text, claims=claims, evidence=evidence)

    assert result.label == VerdictLabel.likely_reliable
    assert result.evidence_support == EvidenceSupport.supported


# ---------------------------------------------------------------------------
# Test 3: Clearly Contradictory Supplied Evidence -> Likely Fake
# ---------------------------------------------------------------------------
def test_contradictory_evidence_evaluates_to_likely_fake() -> None:
    text = "Acme announced a factory in Texas."
    claims = [Claim(text="Acme announced a factory in Texas.", checkable=True, claim_type=ClaimType.event)]
    evidence = [
        EvidenceItem(
            claim_summary="Acme factory announcement",
            evidence_text="Acme's official release states no Texas factory has been announced.",
            source_name="Acme Press Office",
            stance=EvidenceStance.contradicts,
        )
    ]
    provider = MockLLMProvider()
    provider.queue_response(
        make_assessment(
            label=VerdictLabel.likely_fake,
            source_attribution=SourceAttribution.strong,
            evidence_support=EvidenceSupport.contradicted,
            short_reason="The company's official release explicitly refutes the factory announcement.",
        )
    )
    reasoner = LLMReasoner(llm_client=LLMClient(provider=provider))
    result = reasoner.assess(text, claims=claims, evidence=evidence)

    assert result.label == VerdictLabel.likely_fake
    assert result.evidence_support == EvidenceSupport.contradicted


# ---------------------------------------------------------------------------
# Test 4: Partial Evidence -> Misleading or Unverifiable
# ---------------------------------------------------------------------------
def test_partial_evidence_evaluates_to_misleading() -> None:
    text = "Company Y doubled revenue and fired all executives."
    claims = [
        Claim(text="Company Y doubled revenue.", checkable=True, claim_type=ClaimType.numeric),
        Claim(text="Company Y fired all executives.", checkable=True, claim_type=ClaimType.event),
    ]
    evidence = [
        EvidenceItem(
            claim_summary="Revenue growth",
            evidence_text="Company Y reported that its revenue doubled this quarter.",
            source_name="Financial Times",
            stance=EvidenceStance.supports,
        )
    ]
    provider = MockLLMProvider()
    provider.queue_response(
        make_assessment(
            label=VerdictLabel.misleading,
            source_attribution=SourceAttribution.none,
            evidence_support=EvidenceSupport.partial,
            unsupported_claims=["Company Y fired all executives."],
            short_reason="Evidence corroborates the revenue growth but lacks verification for executive dismissals.",
        )
    )
    reasoner = LLMReasoner(llm_client=LLMClient(provider=provider))
    result = reasoner.assess(text, claims=claims, evidence=evidence)

    assert result.label in (VerdictLabel.misleading, VerdictLabel.unverifiable)
    assert result.evidence_support == EvidenceSupport.partial


# ---------------------------------------------------------------------------
# Test 5: Opinion-Only Text -> Unverifiable
# ---------------------------------------------------------------------------
def test_opinion_only_text_evaluates_to_unverifiable() -> None:
    text = "This movie is completely awful and uninspired."
    claims: list[Claim] = []
    provider = MockLLMProvider()
    provider.queue_response(
        make_assessment(
            label=VerdictLabel.unverifiable,
            source_attribution=SourceAttribution.none,
            evidence_support=EvidenceSupport.not_applicable,
            short_reason="The text contains purely subjective aesthetic opinions and no checkable factual claims.",
        )
    )
    reasoner = LLMReasoner(llm_client=LLMClient(provider=provider))
    result = reasoner.assess(text, claims=claims, evidence=None)

    assert result.label == VerdictLabel.unverifiable
    assert result.evidence_support == EvidenceSupport.not_applicable


# ---------------------------------------------------------------------------
# Test 6: Sensational Wording Without Evidence -> NOT Likely Fake
# ---------------------------------------------------------------------------
def test_sensational_wording_without_evidence_is_not_fake() -> None:
    text = "SHOCKING BOMBSHELL: New trade deal signed in Geneva!"
    claims = [Claim(text="New trade deal signed in Geneva.", checkable=True, claim_type=ClaimType.event)]
    provider = MockLLMProvider()
    provider.queue_response(
        make_assessment(
            label=VerdictLabel.unverifiable,
            source_attribution=SourceAttribution.none,
            evidence_support=EvidenceSupport.unavailable,
            rhetorical_cues=["sensational wording", "emotional appeal"],
            short_reason="Sensational headline phrasing is noted, but no external evidence was supplied to verify the deal.",
        )
    )
    reasoner = LLMReasoner(llm_client=LLMClient(provider=provider))
    result = reasoner.assess(text, claims=claims, evidence=None)

    assert result.label == VerdictLabel.unverifiable
    assert result.label != VerdictLabel.likely_fake
    assert "sensational wording" in result.rhetorical_cues


# ---------------------------------------------------------------------------
# Test 7: Missing Source Without Evidence -> NOT Automatically Likely Fake
# ---------------------------------------------------------------------------
def test_missing_source_without_evidence_is_not_fake() -> None:
    text = "Unemployment dropped by 1.2% this quarter."
    claims = [Claim(text="Unemployment dropped by 1.2% this quarter.", checkable=True, claim_type=ClaimType.numeric)]
    provider = MockLLMProvider()
    provider.queue_response(
        make_assessment(
            label=VerdictLabel.unverifiable,
            source_attribution=SourceAttribution.none,
            evidence_support=EvidenceSupport.unavailable,
            short_reason="No attribution is given, but absence of evidence requires an unverifiable verdict.",
        )
    )
    reasoner = LLMReasoner(llm_client=LLMClient(provider=provider))
    result = reasoner.assess(text, claims=claims, evidence=None)

    assert result.label == VerdictLabel.unverifiable
    assert result.label != VerdictLabel.likely_fake
    assert result.source_attribution == SourceAttribution.none


# ---------------------------------------------------------------------------
# Test 8: False-Looking Claim Without Evidence -> Unverifiable (No Memory Correction)
# ---------------------------------------------------------------------------
def test_false_looking_claim_without_evidence_is_unverifiable() -> None:
    text = "The Moon is made entirely of cheese."
    claims = [Claim(text="The Moon is made entirely of cheese.", checkable=True, claim_type=ClaimType.existence)]
    provider = MockLLMProvider()
    provider.queue_response(
        make_assessment(
            label=VerdictLabel.unverifiable,
            source_attribution=SourceAttribution.none,
            evidence_support=EvidenceSupport.unavailable,
            unsupported_claims=["The Moon is made entirely of cheese."],
            short_reason="No physical or astronomical evidence was supplied within the pipeline.",
        )
    )
    reasoner = LLMReasoner(llm_client=LLMClient(provider=provider))
    result = reasoner.assess(text, claims=claims, evidence=None)

    # Must NOT be classified as likely_fake from model pretraining memory alone
    assert result.label == VerdictLabel.unverifiable


# ---------------------------------------------------------------------------
# Test 9: Attributed Claim Preserves Attributed Event
# ---------------------------------------------------------------------------
def test_attributed_claim_preserves_attribution() -> None:
    text = "The finance minister said inflation fell to 4.2%."
    claims = [
        Claim(
            text="The finance minister said inflation fell to 4.2%.",
            checkable=True,
            claim_type=ClaimType.attribution,
        )
    ]
    provider = MockLLMProvider()
    provider.queue_response(
        make_assessment(
            label=VerdictLabel.unverifiable,
            source_attribution=SourceAttribution.strong,
            evidence_support=EvidenceSupport.unavailable,
            short_reason="Attribution to the finance minister is clear, but official records were not supplied.",
        )
    )
    reasoner = LLMReasoner(llm_client=LLMClient(provider=provider))
    result = reasoner.assess(text, claims=claims, evidence=None)

    assert result.source_attribution == SourceAttribution.strong
    assert result.label == VerdictLabel.unverifiable


# ---------------------------------------------------------------------------
# Test 10-12: Source Attribution Rubric Dimensions
# ---------------------------------------------------------------------------
def test_source_attribution_strong() -> None:
    res = make_assessment(source_attribution=SourceAttribution.strong)
    assert res.source_attribution == "strong"


def test_source_attribution_partial() -> None:
    res = make_assessment(source_attribution=SourceAttribution.partial)
    assert res.source_attribution == "partial"


def test_source_attribution_none() -> None:
    res = make_assessment(source_attribution=SourceAttribution.none)
    assert res.source_attribution == "none"


# ---------------------------------------------------------------------------
# Test 13-16: Evidence Support Rubric Dimensions
# ---------------------------------------------------------------------------
def test_evidence_support_supported() -> None:
    res = make_assessment(evidence_support=EvidenceSupport.supported)
    assert res.evidence_support == "supported"


def test_evidence_support_contradicted() -> None:
    res = make_assessment(evidence_support=EvidenceSupport.contradicted)
    assert res.evidence_support == "contradicted"


def test_evidence_support_partial() -> None:
    res = make_assessment(evidence_support=EvidenceSupport.partial)
    assert res.evidence_support == "partial"


def test_evidence_support_unavailable() -> None:
    res = make_assessment(evidence_support=EvidenceSupport.unavailable)
    assert res.evidence_support == "unavailable"


# ---------------------------------------------------------------------------
# Test 17-18: Internal Consistency Rubric Dimensions
# ---------------------------------------------------------------------------
def test_internal_consistency_consistent() -> None:
    res = make_assessment(internal_consistency=InternalConsistency.consistent)
    assert res.internal_consistency == "consistent"


def test_internal_consistency_contradictory() -> None:
    res = make_assessment(internal_consistency=InternalConsistency.contradictory)
    assert res.internal_consistency == "contradictory"


# ---------------------------------------------------------------------------
# Test 19: Malformed Provider JSON Rejected
# ---------------------------------------------------------------------------
def test_malformed_provider_json_rejected() -> None:
    provider = MockLLMProvider()
    provider.queue_response("{ broken json syntax")
    reasoner = LLMReasoner(llm_client=LLMClient(provider=provider))

    with pytest.raises(LLMResponseValidationError) as exc_info:
        reasoner.assess("Sample text.", claims=[])
    assert "not valid JSON" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 20: Invalid Verdict Rejected
# ---------------------------------------------------------------------------
def test_invalid_verdict_rejected() -> None:
    provider = MockLLMProvider()
    # "true" is not a valid VerdictLabel enum member
    provider.queue_response(
        {
            "label": "true",
            "source_attribution": "none",
            "evidence_support": "unavailable",
            "internal_consistency": "consistent",
            "short_reason": "Justification.",
        }
    )
    reasoner = LLMReasoner(llm_client=LLMClient(provider=provider))

    with pytest.raises(LLMResponseValidationError) as exc_info:
        reasoner.assess("Sample text.", claims=[])
    assert "validation error" in str(exc_info.value).lower() or "input should be" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Test 21: Empty short_reason Rejected
# ---------------------------------------------------------------------------
def test_empty_short_reason_rejected() -> None:
    with pytest.raises(ValidationError):
        LLMAssessment(
            label=VerdictLabel.unverifiable,
            source_attribution=SourceAttribution.none,
            evidence_support=EvidenceSupport.unavailable,
            internal_consistency=InternalConsistency.consistent,
            short_reason="   \n  ",
        )


# ---------------------------------------------------------------------------
# Test 22: Rhetorical Cues Max Length Bounded
# ---------------------------------------------------------------------------
def test_rhetorical_cues_max_length_bounded() -> None:
    with pytest.raises(ValidationError):
        LLMAssessment(
            label=VerdictLabel.unverifiable,
            source_attribution=SourceAttribution.none,
            evidence_support=EvidenceSupport.unavailable,
            internal_consistency=InternalConsistency.consistent,
            rhetorical_cues=[f"cue_{i}" for i in range(25)],
            short_reason="Valid justification.",
        )


# ---------------------------------------------------------------------------
# Test 23: Unsupported Claims Max Length Bounded
# ---------------------------------------------------------------------------
def test_unsupported_claims_max_length_bounded() -> None:
    with pytest.raises(ValidationError):
        LLMAssessment(
            label=VerdictLabel.unverifiable,
            source_attribution=SourceAttribution.none,
            evidence_support=EvidenceSupport.unavailable,
            internal_consistency=InternalConsistency.consistent,
            unsupported_claims=[f"claim_{i}" for i in range(25)],
            short_reason="Valid justification.",
        )


# ---------------------------------------------------------------------------
# Test 24: No Chain-of-Thought Fields in Schema
# ---------------------------------------------------------------------------
def test_no_chain_of_thought_fields_in_schema() -> None:
    fields = set(LLMAssessment.model_fields.keys())
    prohibited = {
        "chain_of_thought",
        "cot",
        "reasoning",
        "thinking",
        "internal_monologue",
        "hidden_reasoning",
        "private_reasoning",
    }
    assert fields.isdisjoint(prohibited)


# ---------------------------------------------------------------------------
# Test 25: Autouse Network Guard Enforced
# ---------------------------------------------------------------------------
def test_autouse_network_guard_enforced() -> None:
    with pytest.raises(RuntimeError) as exc_info:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("8.8.8.8", 53))
    assert "External network connection attempted" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 26: No Vendor SDK Imports in llm_reasoner.py
# ---------------------------------------------------------------------------
def test_no_vendor_sdk_imports_in_llm_reasoner() -> None:
    import backend.app.services.llm_reasoner as reasoner_module
    module_src = Path(reasoner_module.__file__).read_text(encoding="utf-8")
    assert "google.genai" not in module_src
    assert "openai" not in module_src
    assert "anthropic" not in module_src


# ---------------------------------------------------------------------------
# Test 27: Reasoner Communicates Exclusively via LLMClient
# ---------------------------------------------------------------------------
def test_reasoner_communicates_via_llm_client() -> None:
    provider = MockLLMProvider()
    client = LLMClient(provider=provider)
    reasoner = LLMReasoner(llm_client=client)

    provider.queue_response(make_assessment())
    reasoner.assess("Sample text.", claims=[])

    assert len(provider.call_history) == 1
    call = provider.call_history[0]
    assert "Source Text to Assess" in call["user_prompt"]


# ---------------------------------------------------------------------------
# Test 28: Prompt Version Exposed
# ---------------------------------------------------------------------------
def test_prompt_version_exposed() -> None:
    provider = MockLLMProvider()
    provider.queue_response(make_assessment())
    reasoner = LLMReasoner(llm_client=LLMClient(provider=provider), prompt_version="reasoning_rubric_v1")

    assert reasoner.prompt_version == "reasoning_rubric_v1"
    res = reasoner.assess("Text.", claims=[])
    assert res.prompt_version == "reasoning_rubric_v1"


# ---------------------------------------------------------------------------
# Test 29: No Evidence Does Not Imply Fake
# ---------------------------------------------------------------------------
def test_no_evidence_does_not_imply_fake() -> None:
    provider = MockLLMProvider()
    provider.queue_response(
        make_assessment(
            label=VerdictLabel.unverifiable,
            evidence_support=EvidenceSupport.unavailable,
            short_reason="Without external evidence, this factual claim cannot be verified.",
        )
    )
    reasoner = LLMReasoner(llm_client=LLMClient(provider=provider))
    res = reasoner.assess("Unchecked claim.", claims=[Claim(text="Unchecked claim.", checkable=True)])

    assert res.label == VerdictLabel.unverifiable
    assert res.label != VerdictLabel.likely_fake


# ---------------------------------------------------------------------------
# Test 30: Sensational Language Does Not Imply Fake
# ---------------------------------------------------------------------------
def test_sensational_language_does_not_imply_fake() -> None:
    provider = MockLLMProvider()
    provider.queue_response(
        make_assessment(
            label=VerdictLabel.unverifiable,
            rhetorical_cues=["sensational wording"],
            short_reason="Sensational tone identified; external facts unverified.",
        )
    )
    reasoner = LLMReasoner(llm_client=LLMClient(provider=provider))
    res = reasoner.assess("SHOCKING headline!", claims=[Claim(text="Headline claim.", checkable=True)])

    assert res.label != VerdictLabel.likely_fake


# ---------------------------------------------------------------------------
# Test 31: Statistical Claim Does Not Imply Fake
# ---------------------------------------------------------------------------
def test_statistical_claim_does_not_imply_fake() -> None:
    provider = MockLLMProvider()
    provider.queue_response(
        make_assessment(
            label=VerdictLabel.unverifiable,
            short_reason="Statistical assertion requires external validation.",
        )
    )
    reasoner = LLMReasoner(llm_client=LLMClient(provider=provider))
    res = reasoner.assess("Profits jumped 45.2%.", claims=[Claim(text="Profits jumped 45.2%.", checkable=True)])

    assert res.label != VerdictLabel.likely_fake


# ---------------------------------------------------------------------------
# Test 32: Prompt User Formatter Embeds Claims and Evidence Correctly
# ---------------------------------------------------------------------------
def test_prompt_embeds_claims_and_evidence_correctly() -> None:
    provider = MockLLMProvider()
    provider.queue_response(make_assessment())
    reasoner = LLMReasoner(llm_client=LLMClient(provider=provider))

    claims = [Claim(text="Claim A.", checkable=True, claim_type=ClaimType.numeric)]
    evidence = [
        EvidenceItem(
            claim_summary="Summary A",
            evidence_text="Evidence text excerpt for A.",
            source_name="Official Gazette",
            similarity_score=0.92,
            stance=EvidenceStance.supports,
        )
    ]

    reasoner.assess("Statement text.", claims=claims, evidence=evidence)

    last_prompt = provider.call_history[0]["user_prompt"]
    assert "Claim A." in last_prompt
    assert "Official Gazette" in last_prompt
    assert "Evidence text excerpt for A." in last_prompt
    assert "0.9200" in last_prompt
    assert "supports" in last_prompt
    assert "does NOT imply factual support" in last_prompt


# ---------------------------------------------------------------------------
# Test 33: Input Validation Edge Cases
# ---------------------------------------------------------------------------
def test_input_validation_edge_cases() -> None:
    reasoner = LLMReasoner(llm_client=LLMClient(provider=MockLLMProvider()))

    with pytest.raises(ValueError):
        reasoner.assess(None, claims=[])  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        reasoner.assess(12345, claims=[])  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        reasoner.assess("   \n ", claims=[])

    with pytest.raises(ValueError):
        reasoner.assess("a" * (MAX_ANALYSIS_TEXT_LENGTH + 1), claims=[])

    with pytest.raises(ValueError):
        reasoner.assess("Valid text.", claims=None)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        reasoner.assess("Valid text.", claims="not_a_list")  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        reasoner.assess("Valid text.", claims=[{"text": "not_a_claim_obj"}])  # type: ignore[list-item]

    with pytest.raises(TypeError):
        reasoner.assess("Valid text.", claims=[], evidence="not_a_list")  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        reasoner.assess("Valid text.", claims=[], evidence=[{"not": "evidence"}])  # type: ignore[list-item]
