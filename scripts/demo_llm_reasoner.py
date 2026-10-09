"""Demonstration script for the Evidence-Aware LLM Reasoner Service (Step 12).

Demonstrates structured reasoning against supplied evidence using the fixed rubric.
Operates in deterministic MOCK / OFFLINE mode by default without consuming API credits.
Live execution requires explicit '--live' flag and is strictly capped at at most 2 calls.
Never exposes secret API credentials or hidden chain-of-thought.
"""

import argparse
from pathlib import Path
import sys
import time

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.ai.llm_client import LLMClient
from backend.app.ai.providers.mock_provider import MockLLMProvider
from backend.app.config import LLMConfig
from backend.app.schemas import (
    Claim,
    ClaimType,
    EvidenceItem,
    EvidenceStance,
    EvidenceSupport,
    InternalConsistency,
    LLMAssessment,
    SourceAttribution,
    VerdictLabel,
)
from backend.app.services.llm_reasoner import LLMReasoner


def run_mock_demo() -> int:
    """Run demonstration using deterministic MockLLMProvider across 5 core reasoning archetypes."""
    print("=" * 70)
    print("LLM Reasoner Demonstration (Step 12)")
    print("MODE: MOCK / OFFLINE")
    print("Provider: mock (mock-model-v1)")
    print("=" * 70)

    examples = [
        {
            "id": "A",
            "title": "Example A: Claim with No Evidence",
            "input": "Acme Corp acquired Global Industries for $10 billion in cash.",
            "claims": [
                Claim(
                    id=1,
                    claim_index=1,
                    text="Acme Corp acquired Global Industries for $10 billion in cash.",
                    checkable=True,
                    claim_type=ClaimType.numeric,
                )
            ],
            "evidence": None,
            "mock_response": LLMAssessment(
                label=VerdictLabel.unverifiable,
                source_attribution=SourceAttribution.none,
                evidence_support=EvidenceSupport.unavailable,
                internal_consistency=InternalConsistency.consistent,
                rhetorical_cues=[],
                unsupported_claims=["Acme Corp acquired Global Industries for $10 billion in cash."],
                short_reason="No external financial or corporate documentation was supplied to substantiate the acquisition.",
            ),
        },
        {
            "id": "B",
            "title": "Example B: Claim with Supporting Evidence",
            "input": "Acme Corp announced a new factory in Texas.",
            "claims": [
                Claim(
                    id=1,
                    claim_index=1,
                    text="Acme Corp announced a new factory in Texas.",
                    checkable=True,
                    claim_type=ClaimType.event,
                )
            ],
            "evidence": [
                EvidenceItem(
                    claim_summary="Acme factory announcement",
                    evidence_text="Acme Corp announced today that it will construct a new manufacturing facility in Texas.",
                    source_name="Acme Press Release",
                    stance=EvidenceStance.supports,
                )
            ],
            "mock_response": LLMAssessment(
                label=VerdictLabel.likely_reliable,
                source_attribution=SourceAttribution.strong,
                evidence_support=EvidenceSupport.supported,
                internal_consistency=InternalConsistency.consistent,
                rhetorical_cues=[],
                unsupported_claims=[],
                short_reason="The official corporate press release directly corroborates the factory announcement.",
            ),
        },
        {
            "id": "C",
            "title": "Example C: Claim with Contradicting Evidence",
            "input": "The city announced an immediate 50% transit fare reduction.",
            "claims": [
                Claim(
                    id=1,
                    claim_index=1,
                    text="The city announced an immediate 50% transit fare reduction.",
                    checkable=True,
                    claim_type=ClaimType.numeric,
                )
            ],
            "evidence": [
                EvidenceItem(
                    claim_summary="Transit fare policy",
                    evidence_text="The Department of Transportation confirmed transit fares will remain unchanged this fiscal year.",
                    source_name="City Transit Authority",
                    stance=EvidenceStance.contradicts,
                )
            ],
            "mock_response": LLMAssessment(
                label=VerdictLabel.likely_fake,
                source_attribution=SourceAttribution.none,
                evidence_support=EvidenceSupport.contradicted,
                internal_consistency=InternalConsistency.consistent,
                rhetorical_cues=[],
                unsupported_claims=["The city announced an immediate 50% transit fare reduction."],
                short_reason="The official City Transit Authority statement directly contradicts any transit fare reduction.",
            ),
        },
        {
            "id": "D",
            "title": "Example D: Partially Supported Claim",
            "input": "The university announced a 15% tuition hike and abolished all scholarships.",
            "claims": [
                Claim(id=1, text="The university announced a 15% tuition hike.", checkable=True, claim_type=ClaimType.numeric),
                Claim(id=2, text="The university abolished all scholarships.", checkable=True, claim_type=ClaimType.event),
            ],
            "evidence": [
                EvidenceItem(
                    claim_summary="Tuition hike",
                    evidence_text="The board of regents approved a 15% tuition increase starting in the fall semester.",
                    source_name="University Gazette",
                    stance=EvidenceStance.supports,
                )
            ],
            "mock_response": LLMAssessment(
                label=VerdictLabel.misleading,
                source_attribution=SourceAttribution.partial,
                evidence_support=EvidenceSupport.partial,
                internal_consistency=InternalConsistency.consistent,
                rhetorical_cues=[],
                unsupported_claims=["The university abolished all scholarships."],
                short_reason="Supplied records confirm the tuition increase but provide no evidence for scholarship cancellation.",
            ),
        },
        {
            "id": "E",
            "title": "Example E: Sensational Text with No Evidence",
            "input": "BREAKING BOMBSHELL: Catastrophic crisis erupts as ministers resign in panic!",
            "claims": [
                Claim(id=1, text="Ministers resigned.", checkable=True, claim_type=ClaimType.event)
            ],
            "evidence": None,
            "mock_response": LLMAssessment(
                label=VerdictLabel.unverifiable,
                source_attribution=SourceAttribution.none,
                evidence_support=EvidenceSupport.unavailable,
                internal_consistency=InternalConsistency.consistent,
                rhetorical_cues=["sensational wording", "emotional framing", "urgent alarm"],
                unsupported_claims=["Ministers resigned."],
                short_reason="Sensational tone is noted; in the absence of corroborating evidence, the claim is unverifiable rather than fake.",
            ),
        },
    ]

    for item in examples:
        print(f"\n--- [{item['id']}] {item['title']} ---")
        print(f"Input text:\n  \"{item['input']}\"")
        print(f"Claims ({len(item['claims'])}):")
        for c in item["claims"]:
            print(f"  - [{c.id or c.claim_index}] \"{c.text}\" (checkable={c.checkable})")
        print("Evidence supplied:")
        if not item["evidence"]:
            print("  (None — no external evidence provided)")
        else:
            for ev in item["evidence"]:
                print(f"  - [{ev.source_name}] \"{ev.evidence_text}\" (stance={ev.stance})")

        provider = MockLLMProvider()
        provider.queue_response(item["mock_response"])
        reasoner = LLMReasoner(llm_client=LLMClient(provider=provider))

        result = reasoner.assess(item["input"], claims=item["claims"], evidence=item["evidence"])

        print("\nAssessment Output:")
        print(f"  Verdict Label:         {result.label.value}")
        print(f"  Source Attribution:    {result.source_attribution.value}")
        print(f"  Evidence Support:      {result.evidence_support.value}")
        print(f"  Internal Consistency:  {result.internal_consistency.value}")
        print(f"  Rhetorical Cues:       {result.rhetorical_cues}")
        print(f"  Unsupported Claims:    {result.unsupported_claims}")
        print(f"  Short Reason:          \"{result.short_reason}\"")

    print("\n" + "=" * 70)
    print("Mock demonstration completed successfully. Zero external network calls made.")
    print("=" * 70)
    return 0


def run_live_demo() -> int:
    """Run live smoke test against configured provider capped at at most 2 calls."""
    print("=" * 70)
    print("LLM Reasoner Demonstration (Step 12)")
    print("MODE: LIVE PROVIDER (Explicit Request — Capped at 2 Calls)")
    print("=" * 70)

    config = LLMConfig.from_env()
    if not config.provider or not config.provider.strip():
        print("ERROR: --live was specified, but no LLM_PROVIDER is configured in .env.")
        return 1

    provider_name = config.provider.strip().lower()
    if provider_name == "mock":
        print("ERROR: --live was specified, but configured provider is 'mock'.")
        print("To run live mode, configure a live provider (e.g. LLM_PROVIDER=gemini) in .env.")
        return 1

    if not config.api_key or not config.api_key.strip():
        print(f"ERROR: Provider '{provider_name}' is configured, but LLM_API_KEY is missing.")
        return 1

    live_cases = [
        {
            "id": 1,
            "title": "Live Call 1: Claim with NO Supplied Evidence",
            "input": "The Moon is made entirely of cheese.",
            "claims": [
                Claim(id=1, text="The Moon is made entirely of cheese.", checkable=True, claim_type=ClaimType.existence)
            ],
            "evidence": None,
            "expected_verdict": "unverifiable (Must not use model memory as verification)",
        },
        {
            "id": 2,
            "title": "Live Call 2: Synthetic Supporting Supplied Evidence",
            "input": "Acme Corp announced a new factory in Texas.",
            "claims": [
                Claim(id=1, text="Acme Corp announced a new factory in Texas.", checkable=True, claim_type=ClaimType.event)
            ],
            "evidence": [
                EvidenceItem(
                    claim_summary="Acme factory announcement",
                    evidence_text="Acme Corp announced today that it will construct a new manufacturing facility in Texas.",
                    source_name="Acme Press Release",
                    stance=EvidenceStance.supports,
                )
            ],
            "expected_verdict": "likely_reliable (Direct corroboration from supplied evidence)",
        },
    ]

    reasoner = LLMReasoner()
    total_calls = len(live_cases)
    successful_calls = 0

    for idx, item in enumerate(live_cases):
        if idx > 0:
            print("Pacing live calls (15s cooldown)...")
            time.sleep(15)

        print(f"\n--- [Live Call {item['id']}/{total_calls}] {item['title']} ---")
        print(f"Input text: \"{item['input']}\"")
        print(f"Claims ({len(item['claims'])}): {[c.text for c in item['claims']]}")
        if item["evidence"]:
            print(f"Evidence: [{item['evidence'][0].source_name}] \"{item['evidence'][0].evidence_text}\"")
        else:
            print("Evidence: (None supplied)")
        print(f"Expected behavior: {item['expected_verdict']}")

        start_time = time.perf_counter()
        try:
            result = reasoner.assess(item["input"], claims=item["claims"], evidence=item["evidence"])
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            print("Status: SUCCESS")
            print(f"Provider: {reasoner.llm_client.provider.provider_name}")
            print(f"Model:    {reasoner.llm_client.provider.model_name}")

            latency_str = (
                f"{reasoner.last_metadata.latency_ms:.2f} ms"
                if reasoner.last_metadata and reasoner.last_metadata.latency_ms is not None
                else f"{elapsed_ms:.2f} ms"
            )
            print(f"Latency:  {latency_str}")

            if reasoner.last_metadata and reasoner.last_metadata.total_tokens is not None:
                print(
                    f"Token Usage: input={reasoner.last_metadata.input_tokens}, "
                    f"output={reasoner.last_metadata.output_tokens}, "
                    f"total={reasoner.last_metadata.total_tokens}"
                )

            print("\nAssessment Output:")
            print(f"  Verdict Label:         {result.label.value}")
            print(f"  Source Attribution:    {result.source_attribution.value}")
            print(f"  Evidence Support:      {result.evidence_support.value}")
            print(f"  Internal Consistency:  {result.internal_consistency.value}")
            print(f"  Rhetorical Cues:       {result.rhetorical_cues}")
            print(f"  Unsupported Claims:    {result.unsupported_claims}")
            print(f"  Short Reason:          \"{result.short_reason}\"")

            successful_calls += 1

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            print(f"Status: FAILED ({type(exc).__name__}): {exc}")
            print(f"Latency: {elapsed_ms:.2f} ms")

    print("\n" + "=" * 70)
    print(f"Live Demonstration Completed: {successful_calls}/{total_calls} successful calls.")
    print("=" * 70)
    return 0 if successful_calls == total_calls else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate LLM Reasoner Service.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Execute live LLM invocation (strictly capped at 2 calls, requires valid .env credentials)",
    )
    args = parser.parse_args()

    if args.live:
        sys.exit(run_live_demo())
    else:
        sys.exit(run_mock_demo())


if __name__ == "__main__":
    main()
