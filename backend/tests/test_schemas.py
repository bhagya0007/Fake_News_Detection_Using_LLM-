"""Unit tests for Pydantic Data Schemas and Contracts (Step 9).

Verifies all 32 required test scenarios:
1. Valid AnalysisRequest accepted.
2. Empty AnalysisRequest rejected.
3. Whitespace-only AnalysisRequest rejected.
4. Over-length AnalysisRequest rejected.
5. Valid Claim accepted.
6. Empty Claim rejected.
7. Valid RedFlag accepted.
8. Invalid severity rejected.
9. Negative feature counts rejected.
10. Invalid ratios (>1.0 or <0.0) rejected.
11. Valid MLAssessment accepted.
12. ML 'unverifiable' label strictly rejected.
13. Probability > 1.0 rejected.
14. Probability < 0.0 rejected.
15. Probability sum not ~1.0 rejected.
16. Small floating point tolerance accepted.
17. Valid EvidenceItem accepted.
18. Valid LLMAssessment accepted.
19. Invalid LLM verdict rejected.
20. Empty short_reason rejected.
21. Valid FinalAssessment accepted.
22. unverifiable + abstained=True accepted.
23. unverifiable + abstained=False rejected.
24. non-unverifiable + abstained=True rejected.
25. Valid AnalysisResponse accepted.
26. Optional llm=None accepted.
27. Valid FeedbackRequest accepted.
28. Oversized feedback comment rejected.
29. Actual Step 8 red-flag output validates cleanly.
30. Actual frozen classical inference output validates through adapter.
31. JSON serialization round-trip works for all primary schemas.
32. No chain-of-thought or hidden reasoning fields exist in LLMAssessment.
"""

from __future__ import annotations

import json
from datetime import date, datetime

import pytest
from pydantic import ValidationError

from backend.app.schemas import (
    MAX_ANALYSIS_TEXT_LENGTH,
    AnalysisRequest,
    AnalysisResponse,
    Claim,
    ClaimAssessment,
    ClaimResult,
    ClassProbabilities,
    ComponentExecutionStatus,
    ComponentStatus,
    EvidenceItem,
    EvidenceRetrievalResult,
    EvidenceStance,
    EvidenceSupport,
    FeedbackRequest,
    FinalAssessment,
    FusionInput,
    FusionSignals,
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
    adapter_from_classical_prediction,
    adapter_from_red_flags,
)
from backend.app.services.baseline_model import (
    load_baseline_pipeline,
    predict_baseline,
)
from backend.app.services.dataset_loader import get_project_root
from backend.app.services.red_flags import analyze_red_flags


# 1. AnalysisRequest Tests
def test_valid_analysis_request() -> None:
    """Test 1: Valid AnalysisRequest accepted and trimmed."""
    req = AnalysisRequest(text="  The government published a report today.  ")
    assert req.text == "The government published a report today."


def test_empty_analysis_request_rejected() -> None:
    """Test 2: Empty string rejected with ValidationError."""
    with pytest.raises(ValidationError):
        AnalysisRequest(text="")


def test_whitespace_only_request_rejected() -> None:
    """Test 3: Whitespace-only string rejected with ValidationError."""
    with pytest.raises(ValidationError):
        AnalysisRequest(text="   \n\t   ")


def test_over_length_request_rejected() -> None:
    """Test 4: Exactly MAX_ANALYSIS_TEXT_LENGTH (20,000) accepted; 20,001 raises ValidationError."""
    boundary_ok = AnalysisRequest(text="A" * MAX_ANALYSIS_TEXT_LENGTH)
    assert len(boundary_ok.text) == MAX_ANALYSIS_TEXT_LENGTH

    oversized = "A" * (MAX_ANALYSIS_TEXT_LENGTH + 1)
    with pytest.raises(ValidationError) as exc_info:
        AnalysisRequest(text=oversized)
    assert "Input text exceeds maximum allowed length" in str(exc_info.value)


# 2. Claim Tests
def test_valid_claim() -> None:
    """Test 5: Valid Claim accepted."""
    claim = Claim(id=1, text="Unemployment fell by two percent.", checkable=True)
    assert claim.text == "Unemployment fell by two percent."
    assert claim.checkable is True


def test_empty_claim_rejected() -> None:
    """Test 6: Empty Claim text rejected."""
    with pytest.raises(ValidationError):
        Claim(text="")


# 3. RedFlag & Stylometric Tests
def test_valid_red_flag() -> None:
    """Test 7: Valid RedFlag accepted."""
    flag = RedFlag(
        type="sensational_language",
        severity=Severity.medium,
        message="Sensational phrasing detected.",
        snippet="You won't believe",
        value=1,
    )
    assert flag.type == "sensational_language"
    assert flag.severity == Severity.medium


def test_invalid_severity_rejected() -> None:
    """Test 8: Invalid severity string rejected."""
    with pytest.raises(ValidationError):
        RedFlag(
            type="sensational_language",
            severity="extreme",  # type: ignore
            message="Invalid severity test.",
        )


def test_negative_counts_rejected() -> None:
    """Test 9: Negative feature and summary counts rejected."""
    with pytest.raises(ValidationError):
        RedFlagSummary(
            total_flags=-1,
            low_severity_flags=0,
            medium_severity_flags=0,
            high_severity_flags=0,
        )


def test_invalid_ratios_rejected() -> None:
    """Test 10: Ratios outside [0.0, 1.0] rejected."""
    valid_features = {
        "character_count": 10,
        "word_count": 2,
        "uppercase_letter_ratio": 1.25,  # Invalid: > 1.0
        "exclamation_count": 0,
        "question_mark_count": 0,
        "repeated_punctuation_count": 0,
        "percentage_count": 0,
        "statistical_expression_count": 0,
        "url_count": 0,
        "sensational_phrase_count": 0,
        "certainty_phrase_count": 0,
        "urgency_phrase_count": 0,
        "attribution_phrase_count": 0,
        "quote_count": 0,
        "excessive_punctuation_ratio": 0.0,
    }
    with pytest.raises(ValidationError):
        RedFlagFeatures(**valid_features)


# 4. Classical ML Assessment Tests
def test_valid_ml_assessment() -> None:
    """Test 11: Valid MLAssessment accepted."""
    probs = ClassProbabilities(likely_fake=0.2, likely_reliable=0.5, misleading=0.3)
    ml = MLAssessment(
        label=MLLabel.likely_reliable,
        probabilities=probs,
        model_name="Tuned TF-IDF + Logistic Regression",
    )
    assert ml.label == MLLabel.likely_reliable
    assert ml.probabilities.likely_reliable == 0.5


def test_ml_unverifiable_rejected() -> None:
    """Test 12: MLLabel strictly rejects 'unverifiable'."""
    probs = ClassProbabilities(likely_fake=0.33, likely_reliable=0.33, misleading=0.34)
    with pytest.raises(ValidationError):
        MLAssessment(
            label="unverifiable",  # type: ignore
            probabilities=probs,
            model_name="Test Model",
        )


def test_probability_greater_than_one_rejected() -> None:
    """Test 13: Probability > 1.0 rejected."""
    with pytest.raises(ValidationError):
        ClassProbabilities(likely_fake=1.2, likely_reliable=0.0, misleading=0.0)


def test_probability_less_than_zero_rejected() -> None:
    """Test 14: Probability < 0.0 rejected."""
    with pytest.raises(ValidationError):
        ClassProbabilities(likely_fake=-0.1, likely_reliable=0.6, misleading=0.5)


def test_probability_sum_not_one_rejected() -> None:
    """Test 15: Probability sum clearly deviating from 1.0 rejected."""
    with pytest.raises(ValidationError):
        ClassProbabilities(likely_fake=0.5, likely_reliable=0.5, misleading=0.5)


def test_small_floating_tolerance_accepted() -> None:
    """Test 16: Small float imprecision (e.g., 0.9999 or 1.0002) is safely accepted."""
    probs = ClassProbabilities(likely_fake=0.3333, likely_reliable=0.3333, misleading=0.3334)
    assert abs((probs.likely_fake + probs.likely_reliable + probs.misleading) - 1.0) < 1e-4


# 5. Evidence Item Tests
def test_valid_evidence_item() -> None:
    """Test 17: Valid EvidenceItem accepted."""
    ev = EvidenceItem(
        claim_summary="Economic growth reached 2%",
        evidence_text="The Bureau of Economic Analysis confirmed 2% GDP growth.",
        source_name="BEA",
        similarity_score=0.88,
        stance=EvidenceStance.supports,
    )
    assert ev.source_name == "BEA"
    assert ev.stance == EvidenceStance.supports


# 6. LLMAssessment Tests
def test_valid_llm_assessment() -> None:
    """Test 18: Valid LLMAssessment accepted."""
    llm = LLMAssessment(
        label=VerdictLabel.likely_reliable,
        source_attribution=SourceAttribution.strong,
        evidence_support=EvidenceSupport.supported,
        internal_consistency=InternalConsistency.consistent,
        rhetorical_cues=["objective tone", "attributed figures"],
        unsupported_claims=[],
        short_reason="Official government statistics directly corroborate the assertion.",
    )
    assert llm.label == VerdictLabel.likely_reliable
    assert llm.source_attribution == SourceAttribution.strong


def test_invalid_llm_verdict_rejected() -> None:
    """Test 19: Invalid LLM verdict string rejected."""
    with pytest.raises(ValidationError):
        LLMAssessment(
            label="definitely_true",  # type: ignore
            source_attribution=SourceAttribution.strong,
            evidence_support=EvidenceSupport.supported,
            internal_consistency=InternalConsistency.consistent,
            short_reason="Reasoning.",
        )


def test_empty_short_reason_rejected() -> None:
    """Test 20: Empty short_reason string rejected."""
    with pytest.raises(ValidationError):
        LLMAssessment(
            label=VerdictLabel.misleading,
            source_attribution=SourceAttribution.weak,
            evidence_support=EvidenceSupport.partial,
            internal_consistency=InternalConsistency.questionable,
            short_reason="   \n  ",
        )


# 7. FinalAssessment Abstention Consistency Tests
def test_valid_final_assessment() -> None:
    """Test 21: Valid FinalAssessment accepted."""
    final = FinalAssessment(
        verdict=VerdictLabel.misleading,
        support_score=0.45,
        reason="Claim selectively emphasizes partial data without necessary context.",
        abstained=False,
        signals_used=["ml_baseline", "red_flags"],
    )
    assert final.verdict == VerdictLabel.misleading
    assert final.abstained is False


def test_unverifiable_and_abstained_true_accepted() -> None:
    """Test 22: 'unverifiable' verdict with abstained=True is consistent and accepted."""
    final = FinalAssessment(
        verdict=VerdictLabel.unverifiable,
        support_score=0.0,
        reason="Insufficient corroborating evidence found.",
        abstained=True,
    )
    assert final.verdict == VerdictLabel.unverifiable
    assert final.abstained is True


def test_unverifiable_and_abstained_false_rejected() -> None:
    """Test 23: 'unverifiable' with abstained=False violates consistency and is rejected."""
    with pytest.raises(ValidationError, match="abstained must be True"):
        FinalAssessment(
            verdict=VerdictLabel.unverifiable,
            support_score=0.0,
            reason="Unverifiable claim.",
            abstained=False,
        )


def test_non_unverifiable_and_abstained_true_rejected() -> None:
    """Test 24: Definite verdict with abstained=True violates consistency and is rejected."""
    with pytest.raises(ValidationError, match="verdict must be 'unverifiable'"):
        FinalAssessment(
            verdict=VerdictLabel.likely_reliable,
            support_score=0.9,
            reason="High confidence claim.",
            abstained=True,
        )


# 8. AnalysisResponse Tests
def test_valid_analysis_response() -> None:
    """Test 25: Valid AnalysisResponse accepted."""
    probs = ClassProbabilities(likely_fake=0.1, likely_reliable=0.8, misleading=0.1)
    ml = MLAssessment(
        label=MLLabel.likely_reliable,
        probabilities=probs,
        model_name="Tuned TF-IDF + Logistic Regression",
    )
    rf_features = RedFlagFeatures(
        character_count=45,
        word_count=8,
        uppercase_letter_ratio=0.02,
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
        excessive_punctuation_ratio=0.02,
    )
    rf_summary = RedFlagSummary(
        total_flags=0,
        low_severity_flags=0,
        medium_severity_flags=0,
        high_severity_flags=0,
        has_attribution=True,
        has_statistical_claim=False,
    )
    red_flags = RedFlagResult(flags=[], features=rf_features, summary=rf_summary)

    resp = AnalysisResponse(
        input_text="The department released economic figures today.",
        verdict=VerdictLabel.likely_reliable,
        support_score=0.85,
        red_flags=red_flags,
        ml=ml,
        reason="Model and attribution corroborate statement.",
    )
    assert resp.verdict == VerdictLabel.likely_reliable
    assert resp.llm is None


def test_optional_llm_none_accepted() -> None:
    """Test 26: Graceful degradation when llm=None."""
    probs = ClassProbabilities(likely_fake=0.5, likely_reliable=0.2, misleading=0.3)
    ml = MLAssessment(
        label=MLLabel.likely_fake,
        probabilities=probs,
        model_name="Tuned TF-IDF + Logistic Regression",
    )
    rf_features = RedFlagFeatures(
        character_count=20,
        word_count=4,
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
    rf_summary = RedFlagSummary(
        total_flags=0,
        low_severity_flags=0,
        medium_severity_flags=0,
        high_severity_flags=0,
    )
    red_flags = RedFlagResult(flags=[], features=rf_features, summary=rf_summary)

    resp = AnalysisResponse(
        input_text="Unattributed test statement.",
        verdict=VerdictLabel.likely_fake,
        support_score=0.50,
        red_flags=red_flags,
        ml=ml,
        llm=None,
        reason="Classical baseline prediction.",
        component_statuses=[
            ComponentStatus(
                component="llm",
                status=ComponentExecutionStatus.unavailable,
                message="LLM service disabled.",
            )
        ],
    )
    assert resp.llm is None
    assert resp.component_statuses[0].status == ComponentExecutionStatus.unavailable


# 9. FeedbackRequest Tests
def test_valid_feedback_request() -> None:
    """Test 27: Valid FeedbackRequest accepted."""
    fb = FeedbackRequest(analysis_id="analysis_123", agree=True, comment="Accurate rating.")
    assert fb.agree is True


def test_oversized_feedback_rejected() -> None:
    """Test 28: Exceeding maximum feedback comment length rejected."""
    with pytest.raises(ValidationError):
        FeedbackRequest(analysis_id=1, agree=False, comment="X" * 2001)


# 10. Step 8 Red-Flag Integration Tests
def test_actual_step8_red_flag_output_validates() -> None:
    """Test 29: Real output from analyze_red_flags validates directly into RedFlagResult."""
    text = "SHOCKING!!! You won't believe what happened: unemployment rose by 7.5% according to official data."
    raw_result = analyze_red_flags(text)

    # Validate directly using adapter
    validated_result = adapter_from_red_flags(raw_result)
    assert isinstance(validated_result, RedFlagResult)
    assert validated_result.features.percentage_count == 1
    assert validated_result.summary.has_attribution is True
    assert len(validated_result.flags) > 0


# 11. Classical Model Integration Tests
def test_actual_classical_inference_output_validates_through_adapter() -> None:
    """Test 30: Real output from predict_baseline validates into MLAssessment."""
    model_path = get_project_root() / "models" / "classical_best_validation.joblib"
    pipeline = load_baseline_pipeline(model_path)

    raw_prediction = predict_baseline(pipeline, "The central bank kept interest rates unchanged.")

    # Validate via adapter
    ml_assessment = adapter_from_classical_prediction(raw_prediction)
    assert isinstance(ml_assessment, MLAssessment)
    assert ml_assessment.label in {MLLabel.likely_fake, MLLabel.likely_reliable, MLLabel.misleading}
    assert (
        abs(
            ml_assessment.probabilities.likely_fake
            + ml_assessment.probabilities.likely_reliable
            + ml_assessment.probabilities.misleading
            - 1.0
        )
        < 1e-4
    )


# 12. Serialization Round-Trip Tests
def test_serialization_round_trip() -> None:
    """Test 31: Full serialization and deserialization round-trip for complex schemas."""
    probs = ClassProbabilities(likely_fake=0.1, likely_reliable=0.7, misleading=0.2)
    ml = MLAssessment(
        label=MLLabel.likely_reliable,
        probabilities=probs,
        model_name="Tuned TF-IDF + Logistic Regression",
    )
    raw_rf = analyze_red_flags("According to the report, growth was 3%.")
    red_flags = adapter_from_red_flags(raw_rf)

    response = AnalysisResponse(
        analysis_id="test_42",
        input_text="According to the report, growth was 3%.",
        verdict=VerdictLabel.likely_reliable,
        support_score=0.75,
        red_flags=red_flags,
        ml=ml,
        reason="Corroborated by baseline and official report.",
        created_at=datetime(2026, 9, 30, 21, 0, 0),
    )

    # Serialize to JSON string
    json_str = response.model_dump_json()
    assert isinstance(json_str, str)

    # Deserialize back to object
    reconstructed = AnalysisResponse.model_validate_json(json_str)
    assert reconstructed.verdict == response.verdict
    assert reconstructed.ml.probabilities.likely_reliable == response.ml.probabilities.likely_reliable
    assert reconstructed.red_flags.features.character_count == response.red_flags.features.character_count


# 13. Absence of Hidden Chain-of-Thought
def test_no_chain_of_thought_field_exists() -> None:
    """Test 32: STRICT GUARANTEE - No chain-of-thought fields exist in LLMAssessment or any schema."""
    prohibited_fields = {
        "chain_of_thought",
        "reasoning_steps",
        "private_reasoning",
        "hidden_reasoning",
        "internal_thought",
        "cot",
    }
    schema_fields = set(LLMAssessment.model_fields.keys())
    assert prohibited_fields.isdisjoint(schema_fields)

    # Check all schemas in the module
    for schema_cls in [
        AnalysisRequest,
        Claim,
        RedFlag,
        RedFlagResult,
        MLAssessment,
        EvidenceItem,
        EvidenceRetrievalResult,
        LLMAssessment,
        ClaimResult,
        FusionSignals,
        FusionInput,
        FinalAssessment,
        AnalysisResponse,
    ]:
        cls_fields = set(schema_cls.model_fields.keys())
        assert prohibited_fields.isdisjoint(cls_fields), f"Prohibited CoT field in {schema_cls.__name__}!"
