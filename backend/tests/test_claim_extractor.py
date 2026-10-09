"""Unit tests for the LLM-based Claim Extraction Service (Step 11).

Verifies strict input validation, factual proposition extraction, numeric and
entity preservation, attribution retention, max-5-claim limit, deduplication,
schema guarantees, and zero external network calls.
"""

import socket
from typing import Any
import pytest
from pydantic import BaseModel, ValidationError

from backend.app.ai.exceptions import LLMResponseValidationError
from backend.app.ai.llm_client import LLMClient
from backend.app.ai.providers.mock_provider import MockLLMProvider
from backend.app.schemas import (
    MAX_ANALYSIS_TEXT_LENGTH,
    Claim,
    ClaimExtractionResult,
    ClaimType,
    LLMCallMetadata,
)
from backend.app.services.claim_extractor import ClaimExtractor


@pytest.fixture(autouse=True)
def guard_against_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strict guard: Prevent ANY network socket connection during test execution."""
    def forbidden_connect(self: socket.socket, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("External network connection attempted during test execution!")

    monkeypatch.setattr(socket.socket, "connect", forbidden_connect)


# ---------------------------------------------------------------------------
# Test 1: Simple factual claim
# ---------------------------------------------------------------------------
def test_simple_factual_claim() -> None:
    text = "India's population exceeded 1.4 billion."
    provider = MockLLMProvider()
    provider.queue_response(
        ClaimExtractionResult(
            claims=[
                Claim(
                    text="India's population exceeded 1.4 billion.",
                    checkable=True,
                    claim_type=ClaimType.numeric,
                )
            ]
        )
    )
    extractor = ClaimExtractor(llm_client=LLMClient(provider=provider))
    result = extractor.extract_claims(text)

    assert len(result.claims) == 1
    assert result.claims[0].text == "India's population exceeded 1.4 billion."
    assert result.claims[0].checkable is True
    assert result.source_text_length == len(text)


# ---------------------------------------------------------------------------
# Test 2: Numeric preservation
# ---------------------------------------------------------------------------
def test_numeric_preservation() -> None:
    text = "Revenue increased by 27.5% to $4.2 billion in 2025."
    provider = MockLLMProvider()
    provider.queue_response(
        ClaimExtractionResult(
            claims=[
                Claim(
                    text="Revenue increased by 27.5% to $4.2 billion in 2025.",
                    checkable=True,
                    claim_type=ClaimType.numeric,
                )
            ]
        )
    )
    extractor = ClaimExtractor(llm_client=LLMClient(provider=provider))
    result = extractor.extract_claims(text)

    assert len(result.claims) == 1
    claim_text = result.claims[0].text
    assert "27.5%" in claim_text
    assert "$4.2 billion" in claim_text
    assert "2025" in claim_text


# ---------------------------------------------------------------------------
# Test 3: Multiple claims
# ---------------------------------------------------------------------------
def test_multiple_claims() -> None:
    text = "The company opened a factory in Ohio, hired 500 workers, and began exporting to Canada."
    provider = MockLLMProvider()
    provider.queue_response(
        ClaimExtractionResult(
            claims=[
                Claim(text="The company opened a factory in Ohio.", checkable=True, claim_type=ClaimType.event),
                Claim(text="The company hired 500 workers.", checkable=True, claim_type=ClaimType.numeric),
                Claim(text="The company began exporting to Canada.", checkable=True, claim_type=ClaimType.event),
            ]
        )
    )
    extractor = ClaimExtractor(llm_client=LLMClient(provider=provider))
    result = extractor.extract_claims(text)

    assert len(result.claims) == 3
    assert result.claims[0].text == "The company opened a factory in Ohio."
    assert result.claims[1].text == "The company hired 500 workers."
    assert result.claims[2].text == "The company began exporting to Canada."


# ---------------------------------------------------------------------------
# Test 4: Maximum 5 claims enforced
# ---------------------------------------------------------------------------
def test_maximum_5_claims_enforced() -> None:
    # 1. Pydantic schema validation directly rejects > 5 claims
    excess_claims = [
        {"text": f"Factual assertion #{i}", "checkable": True}
        for i in range(1, 7)
    ]
    provider = MockLLMProvider()
    provider.queue_response({"claims": excess_claims})
    extractor = ClaimExtractor(llm_client=LLMClient(provider=provider))

    with pytest.raises(LLMResponseValidationError) as exc_info:
        extractor.extract_claims("Source text with multiple statements.")
    err_msg = str(exc_info.value).lower()
    assert "failed schema validation" in err_msg or "too_long" in err_msg or "validation error" in err_msg


# ---------------------------------------------------------------------------
# Test 5: Opinion-only input
# ---------------------------------------------------------------------------
def test_opinion_only_input() -> None:
    text = "I think this film is terrible and completely uninspired."
    provider = MockLLMProvider()
    provider.queue_response(ClaimExtractionResult(claims=[]))
    extractor = ClaimExtractor(llm_client=LLMClient(provider=provider))
    result = extractor.extract_claims(text)

    assert len(result.claims) == 0
    assert result.source_text_length == len(text)


# ---------------------------------------------------------------------------
# Test 6: Mixed opinion + fact
# ---------------------------------------------------------------------------
def test_mixed_opinion_and_fact() -> None:
    text = "This policy is terrible, and parliament passed it on Tuesday."
    provider = MockLLMProvider()
    provider.queue_response(
        ClaimExtractionResult(
            claims=[
                Claim(text="Parliament passed this policy on Tuesday.", checkable=True, claim_type=ClaimType.event)
            ]
        )
    )
    extractor = ClaimExtractor(llm_client=LLMClient(provider=provider))
    result = extractor.extract_claims(text)

    assert len(result.claims) == 1
    assert result.claims[0].text == "Parliament passed this policy on Tuesday."
    assert "terrible" not in result.claims[0].text.lower()


# ---------------------------------------------------------------------------
# Test 7: Rhetorical question
# ---------------------------------------------------------------------------
def test_rhetorical_question() -> None:
    text = "Can you believe prices increased again?"
    provider = MockLLMProvider()
    # Should not fabricate unsupported factual details
    provider.queue_response(
        ClaimExtractionResult(
            claims=[
                Claim(text="Prices increased.", checkable=True, claim_type=ClaimType.event)
            ]
        )
    )
    extractor = ClaimExtractor(llm_client=LLMClient(provider=provider))
    result = extractor.extract_claims(text)

    assert len(result.claims) == 1
    assert result.claims[0].text == "Prices increased."


# ---------------------------------------------------------------------------
# Test 8: False-looking claim preservation (No correction)
# ---------------------------------------------------------------------------
def test_false_looking_claim_preservation() -> None:
    text = "The Moon is made of cheese."
    provider = MockLLMProvider()
    provider.queue_response(
        ClaimExtractionResult(
            claims=[
                Claim(text="The Moon is made of cheese.", checkable=True, claim_type=ClaimType.existence)
            ]
        )
    )
    extractor = ClaimExtractor(llm_client=LLMClient(provider=provider))
    result = extractor.extract_claims(text)

    # Must NOT correct the claim to "The Moon is not made of cheese"
    assert len(result.claims) == 1
    assert result.claims[0].text == "The Moon is made of cheese."
    assert "not" not in result.claims[0].text.lower()


# ---------------------------------------------------------------------------
# Test 9: No external attribution hallucination
# ---------------------------------------------------------------------------
def test_no_external_attribution_hallucination() -> None:
    text = "Inflation rose by 10%."
    provider = MockLLMProvider()
    provider.queue_response(
        ClaimExtractionResult(
            claims=[
                Claim(text="Inflation rose by 10%.", checkable=True, claim_type=ClaimType.numeric)
            ]
        )
    )
    extractor = ClaimExtractor(llm_client=LLMClient(provider=provider))
    result = extractor.extract_claims(text)

    assert len(result.claims) == 1
    assert "government data" not in result.claims[0].text.lower()


# ---------------------------------------------------------------------------
# Test 10: Entity preservation
# ---------------------------------------------------------------------------
def test_entity_preservation() -> None:
    text = "NASA launched Artemis I from Kennedy Space Center in Florida."
    provider = MockLLMProvider()
    provider.queue_response(
        ClaimExtractionResult(
            claims=[
                Claim(
                    text="NASA launched Artemis I from Kennedy Space Center in Florida.",
                    checkable=True,
                    claim_type=ClaimType.event,
                )
            ]
        )
    )
    extractor = ClaimExtractor(llm_client=LLMClient(provider=provider))
    result = extractor.extract_claims(text)

    claim_text = result.claims[0].text
    assert "NASA" in claim_text
    assert "Artemis I" in claim_text
    assert "Kennedy Space Center" in claim_text
    assert "Florida" in claim_text


# ---------------------------------------------------------------------------
# Test 11: Date preservation
# ---------------------------------------------------------------------------
def test_date_preservation() -> None:
    text = "The treaty was ratified on October 24, 1945."
    provider = MockLLMProvider()
    provider.queue_response(
        ClaimExtractionResult(
            claims=[
                Claim(
                    text="The treaty was ratified on October 24, 1945.",
                    checkable=True,
                    claim_type=ClaimType.event,
                )
            ]
        )
    )
    extractor = ClaimExtractor(llm_client=LLMClient(provider=provider))
    result = extractor.extract_claims(text)

    assert "October 24, 1945" in result.claims[0].text


# ---------------------------------------------------------------------------
# Test 12: Percentage preservation
# ---------------------------------------------------------------------------
def test_percentage_preservation() -> None:
    text = "The unemployment rate dropped to 3.4%."
    provider = MockLLMProvider()
    provider.queue_response(
        ClaimExtractionResult(
            claims=[
                Claim(
                    text="The unemployment rate dropped to 3.4%.",
                    checkable=True,
                    claim_type=ClaimType.numeric,
                )
            ]
        )
    )
    extractor = ClaimExtractor(llm_client=LLMClient(provider=provider))
    result = extractor.extract_claims(text)

    assert "3.4%" in result.claims[0].text


# ---------------------------------------------------------------------------
# Test 13: Money-value preservation
# ---------------------------------------------------------------------------
def test_money_value_preservation() -> None:
    text = "The municipal budget allocated $15.5 million for public parks."
    provider = MockLLMProvider()
    provider.queue_response(
        ClaimExtractionResult(
            claims=[
                Claim(
                    text="The municipal budget allocated $15.5 million for public parks.",
                    checkable=True,
                    claim_type=ClaimType.numeric,
                )
            ]
        )
    )
    extractor = ClaimExtractor(llm_client=LLMClient(provider=provider))
    result = extractor.extract_claims(text)

    assert "$15.5 million" in result.claims[0].text


# ---------------------------------------------------------------------------
# Test 14: Empty input rejected
# ---------------------------------------------------------------------------
def test_empty_input_rejected() -> None:
    extractor = ClaimExtractor(llm_client=LLMClient(provider=MockLLMProvider()))
    with pytest.raises(ValueError) as exc_info:
        extractor.extract_claims("")
    assert "cannot be empty" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 15: Whitespace input rejected
# ---------------------------------------------------------------------------
def test_whitespace_input_rejected() -> None:
    extractor = ClaimExtractor(llm_client=LLMClient(provider=MockLLMProvider()))
    with pytest.raises(ValueError) as exc_info:
        extractor.extract_claims("   \t\n  ")
    assert "cannot be empty or whitespace-only" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 16: None rejected
# ---------------------------------------------------------------------------
def test_none_input_rejected() -> None:
    extractor = ClaimExtractor(llm_client=LLMClient(provider=MockLLMProvider()))
    with pytest.raises(ValueError) as exc_info:
        extractor.extract_claims(None)  # type: ignore[arg-type]
    assert "cannot be None" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 17: Non-string rejected
# ---------------------------------------------------------------------------
def test_non_string_rejected() -> None:
    extractor = ClaimExtractor(llm_client=LLMClient(provider=MockLLMProvider()))
    with pytest.raises(TypeError) as exc_info:
        extractor.extract_claims(123456)  # type: ignore[arg-type]
    assert "must be a string" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 18: Oversized input rejected
# ---------------------------------------------------------------------------
def test_oversized_input_rejected() -> None:
    extractor = ClaimExtractor(llm_client=LLMClient(provider=MockLLMProvider()))
    oversized = "a" * (MAX_ANALYSIS_TEXT_LENGTH + 1)
    with pytest.raises(ValueError) as exc_info:
        extractor.extract_claims(oversized)
    assert "exceeds maximum allowed length" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 19: Malformed provider JSON rejected
# ---------------------------------------------------------------------------
def test_malformed_provider_json_rejected() -> None:
    provider = MockLLMProvider()
    provider.queue_response("{ broken json syntax here")
    extractor = ClaimExtractor(llm_client=LLMClient(provider=provider))

    with pytest.raises(LLMResponseValidationError) as exc_info:
        extractor.extract_claims("Valid source text.")
    assert "not valid JSON" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 20: Schema-invalid provider output rejected
# ---------------------------------------------------------------------------
def test_schema_invalid_provider_output_rejected() -> None:
    provider = MockLLMProvider()
    # Missing required 'text' in claim
    provider.queue_response({"claims": [{"checkable": True}]})
    extractor = ClaimExtractor(llm_client=LLMClient(provider=provider))

    with pytest.raises(LLMResponseValidationError) as exc_info:
        extractor.extract_claims("Valid source text.")
    err_msg = str(exc_info.value).lower()
    assert "failed schema validation" in err_msg or "validation error" in err_msg or "missing" in err_msg


# ---------------------------------------------------------------------------
# Test 21: Duplicate claims handled safely
# ---------------------------------------------------------------------------
def test_duplicate_claims_handled_safely() -> None:
    text = "The company reported a profit. The company reported a profit."
    provider = MockLLMProvider()
    provider.queue_response(
        ClaimExtractionResult(
            claims=[
                Claim(text="The company reported a profit.", checkable=True),
                Claim(text="The company reported a profit.", checkable=True),
            ]
        )
    )
    extractor = ClaimExtractor(llm_client=LLMClient(provider=provider))
    result = extractor.extract_claims(text)

    # Deduplicated to 1 unique claim
    assert len(result.claims) == 1
    assert result.claims[0].text == "The company reported a profit."


# ---------------------------------------------------------------------------
# Test 22: Deterministic mock behavior
# ---------------------------------------------------------------------------
def test_deterministic_mock_behavior() -> None:
    provider = MockLLMProvider()
    extractor = ClaimExtractor(llm_client=LLMClient(provider=provider))

    res1 = extractor.extract_claims("Source text one")
    res2 = extractor.extract_claims("Source text two")

    assert res1.claims[0].text == res2.claims[0].text
    assert res1.claims[0].checkable == res2.claims[0].checkable


# ---------------------------------------------------------------------------
# Test 23: No truthfulness verdict in extraction result
# ---------------------------------------------------------------------------
def test_no_truthfulness_verdict_in_extraction_result() -> None:
    fields = set(ClaimExtractionResult.model_fields.keys())
    prohibited_verdict_fields = {
        "verdict",
        "label",
        "is_fake",
        "truth_value",
        "truthfulness",
        "credibility",
        "score",
        "fake_probability",
    }
    assert fields.isdisjoint(prohibited_verdict_fields)

    # Check individual Claim fields as well
    claim_fields = set(Claim.model_fields.keys())
    assert claim_fields.isdisjoint(prohibited_verdict_fields)


# ---------------------------------------------------------------------------
# Test 24: No chain-of-thought field
# ---------------------------------------------------------------------------
def test_no_chain_of_thought_field() -> None:
    fields = set(ClaimExtractionResult.model_fields.keys()) | set(Claim.model_fields.keys())
    prohibited_cot_fields = {
        "chain_of_thought",
        "cot",
        "reasoning",
        "thinking",
        "internal_monologue",
        "hidden_reasoning",
    }
    assert fields.isdisjoint(prohibited_cot_fields)


# ---------------------------------------------------------------------------
# Test 25: No external provider called during tests
# ---------------------------------------------------------------------------
def test_no_external_provider_called_during_tests() -> None:
    # Attempting socket connection triggers autouse fixture guard
    with pytest.raises(RuntimeError) as exc_info:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("1.1.1.1", 80))
    assert "External network connection attempted" in str(exc_info.value)

    # ClaimExtractor runs with zero socket connections
    provider = MockLLMProvider()
    extractor = ClaimExtractor(llm_client=LLMClient(provider=provider))
    res = extractor.extract_claims("Safe claim text")
    assert len(res.claims) >= 1


# ---------------------------------------------------------------------------
# Test 26: Provider metadata does not expose credentials
# ---------------------------------------------------------------------------
def test_provider_metadata_does_not_expose_credentials() -> None:
    provider = MockLLMProvider()
    extractor = ClaimExtractor(llm_client=LLMClient(provider=provider))
    extractor.extract_claims("Sample test text.")

    meta = extractor.last_metadata
    assert meta is not None
    assert isinstance(meta, LLMCallMetadata)
    assert not hasattr(meta, "api_key")
    assert not hasattr(meta, "authorization")
    assert "api_key" not in meta.model_dump()


# ---------------------------------------------------------------------------
# Test 27: Claim order preserved
# ---------------------------------------------------------------------------
def test_claim_order_preserved() -> None:
    text = "First assertion. Second assertion. Third assertion."
    provider = MockLLMProvider()
    provider.queue_response(
        ClaimExtractionResult(
            claims=[
                Claim(text="First assertion.", checkable=True),
                Claim(text="Second assertion.", checkable=True),
                Claim(text="Third assertion.", checkable=True),
            ]
        )
    )
    extractor = ClaimExtractor(llm_client=LLMClient(provider=provider))
    result = extractor.extract_claims(text)

    assert result.claims[0].text == "First assertion."
    assert result.claims[1].text == "Second assertion."
    assert result.claims[2].text == "Third assertion."


# ---------------------------------------------------------------------------
# Test 28: Empty claims list accepted
# ---------------------------------------------------------------------------
def test_empty_claims_list_accepted() -> None:
    provider = MockLLMProvider()
    provider.queue_response(ClaimExtractionResult(claims=[]))
    extractor = ClaimExtractor(llm_client=LLMClient(provider=provider))
    result = extractor.extract_claims("Text with no verifiable claims.")

    assert isinstance(result, ClaimExtractionResult)
    assert result.claims == []


# ---------------------------------------------------------------------------
# Test 29: Checkable field validates as bool
# ---------------------------------------------------------------------------
def test_checkable_field_validates_as_bool() -> None:
    claim_true = Claim(text="Valid claim", checkable=True)
    claim_false = Claim(text="Valid claim", checkable=False)
    assert claim_true.checkable is True
    assert claim_false.checkable is False

    with pytest.raises(ValidationError):
        Claim(text="Valid claim", checkable="not_a_bool")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Test 30: Service communicates through LLMClient only
# ---------------------------------------------------------------------------
def test_service_communicates_through_llm_client_only() -> None:
    provider = MockLLMProvider()
    client = LLMClient(provider=provider)
    extractor = ClaimExtractor(llm_client=client)

    assert extractor.llm_client is client
    extractor.extract_claims("Testing client routing.")
    assert len(provider.call_history) == 1


# ---------------------------------------------------------------------------
# Test 31: Article-like input handling
# ---------------------------------------------------------------------------
def test_article_like_input() -> None:
    article_text = (
        "Acme Corp announced on Monday that it will invest $2 billion in a new factory in Texas. "
        "The company says the facility will employ 3,000 workers. "
        "Critics called the announcement unrealistic."
    )
    provider = MockLLMProvider()
    provider.queue_response(
        ClaimExtractionResult(
            claims=[
                Claim(
                    text="Acme Corp announced on Monday that it will invest $2 billion in a new factory in Texas.",
                    checkable=True,
                    claim_type=ClaimType.attribution,
                ),
                Claim(
                    text="The company says the facility will employ 3,000 workers.",
                    checkable=True,
                    claim_type=ClaimType.attribution,
                ),
            ]
        )
    )
    extractor = ClaimExtractor(llm_client=LLMClient(provider=provider))
    result = extractor.extract_claims(article_text)

    assert len(result.claims) == 2
    assert "invest $2 billion" in result.claims[0].text
    assert "Texas" in result.claims[0].text
    assert "3,000 workers" in result.claims[1].text


# ---------------------------------------------------------------------------
# Test 32: Attributed claims preservation
# ---------------------------------------------------------------------------
def test_attributed_claims_preservation() -> None:
    text = "The minister said unemployment fell to 4%."
    provider = MockLLMProvider()
    provider.queue_response(
        ClaimExtractionResult(
            claims=[
                Claim(
                    text="The minister said unemployment fell to 4%.",
                    checkable=True,
                    claim_type=ClaimType.attribution,
                )
            ]
        )
    )
    extractor = ClaimExtractor(llm_client=LLMClient(provider=provider))
    result = extractor.extract_claims(text)

    # Must preserve attribution to minister, NOT unqualified "Unemployment fell to 4%"
    assert len(result.claims) == 1
    assert "minister said" in result.claims[0].text.lower()


# ---------------------------------------------------------------------------
# Test 33: Prompt version tracked
# ---------------------------------------------------------------------------
def test_prompt_version_tracked() -> None:
    provider = MockLLMProvider()
    extractor = ClaimExtractor(llm_client=LLMClient(provider=provider), prompt_version="claim_extraction_v1")
    assert extractor.prompt_version == "claim_extraction_v1"

    result = extractor.extract_claims("Sample factual claim.")
    assert result.prompt_version == "claim_extraction_v1"

