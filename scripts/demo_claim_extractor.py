"""Demonstration script for the LLM-based Claim Extraction Service (Step 11).

Demonstrates factual proposition extraction across various text patterns.
Operates in deterministic MOCK / OFFLINE mode by default without consuming API credits.
Live execution requires explicit '--live' flag and configured credentials.
Never displays truth verdicts.
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
from backend.app.schemas import Claim, ClaimExtractionResult, ClaimType
from backend.app.services.claim_extractor import ClaimExtractor


def run_mock_demo() -> int:
    """Run demonstration using deterministic MockLLMProvider across 5 core text archetypes."""
    print("=" * 70)
    print("Claim Extraction Demonstration (Step 11)")
    print("MODE: MOCK / OFFLINE")
    print("Provider: mock (mock-model-v1)")
    print("=" * 70)

    examples = [
        {
            "title": "Example 1: Single Direct Claim",
            "input": "India's population exceeded 1.4 billion.",
            "response": ClaimExtractionResult(
                claims=[
                    Claim(
                        id=1,
                        claim_index=1,
                        text="India's population exceeded 1.4 billion.",
                        checkable=True,
                        claim_type=ClaimType.numeric,
                        source_span="India's population exceeded 1.4 billion.",
                    )
                ]
            ),
        },
        {
            "title": "Example 2: Multi-Claim Article Paragraph",
            "input": (
                "Acme Corp announced on Monday that it will invest $2 billion in a new factory in Texas. "
                "The company says the facility will employ 3,000 workers. "
                "Critics called the announcement unrealistic."
            ),
            "response": ClaimExtractionResult(
                claims=[
                    Claim(
                        id=1,
                        claim_index=1,
                        text="Acme Corp announced on Monday that it will invest $2 billion in a new factory in Texas.",
                        checkable=True,
                        claim_type=ClaimType.attribution,
                    ),
                    Claim(
                        id=2,
                        claim_index=2,
                        text="The company says the facility will employ 3,000 workers.",
                        checkable=True,
                        claim_type=ClaimType.attribution,
                    ),
                ]
            ),
        },
        {
            "title": "Example 3: Opinion-Only Text",
            "input": "I think this new policy is completely terrible and disastrous.",
            "response": ClaimExtractionResult(claims=[]),
        },
        {
            "title": "Example 4: Attributed Statement",
            "input": "The health minister announced that the country achieved an 85% vaccination rate.",
            "response": ClaimExtractionResult(
                claims=[
                    Claim(
                        id=1,
                        claim_index=1,
                        text="The health minister announced that the country achieved an 85% vaccination rate.",
                        checkable=True,
                        claim_type=ClaimType.attribution,
                    )
                ]
            ),
        },
        {
            "title": "Example 5: Numerical / Statistical Statement",
            "input": "Quarterly revenue grew by 14.8% to $12.3 billion in Q4 2025.",
            "response": ClaimExtractionResult(
                claims=[
                    Claim(
                        id=1,
                        claim_index=1,
                        text="Quarterly revenue grew by 14.8% to $12.3 billion in Q4 2025.",
                        checkable=True,
                        claim_type=ClaimType.numeric,
                    )
                ]
            ),
        },
    ]

    for item in examples:
        print(f"\n--- {item['title']} ---")
        print(f"Input text:\n  \"{item['input']}\"")

        provider = MockLLMProvider()
        provider.queue_response(item["response"])
        extractor = ClaimExtractor(llm_client=LLMClient(provider=provider))

        result = extractor.extract_claims(item["input"])
        print(f"Extracted claims count: {len(result.claims)}")

        if not result.claims:
            print("  (No checkable factual claims extracted — input classified as opinion/non-factual)")
        else:
            for c in result.claims:
                ctype = c.claim_type.value if c.claim_type else "unspecified"
                print(f"  [{c.id or c.claim_index}] \"{c.text}\"")
                print(f"      Checkable: {c.checkable} | Type: {ctype}")

    print("\n" + "=" * 70)
    print("Mock demonstration completed successfully. Zero external network calls made.")
    print("=" * 70)
    return 0


def run_live_demo() -> int:
    """Run live smoke test against configured provider if explicit credentials exist."""
    print("=" * 70)
    print("Claim Extraction Demonstration (Step 11)")
    print("MODE: LIVE PROVIDER (Explicit Request)")
    print("=" * 70)

    config = LLMConfig.from_env()
    if not config.provider or not config.provider.strip():
        print("ERROR: --live was specified, but no LLM_PROVIDER is configured.")
        print("Please configure LLM_PROVIDER and LLM_API_KEY in your .env file.")
        return 1

    provider_name = config.provider.strip().lower()
    if provider_name == "mock":
        print("ERROR: --live was specified, but configured provider is 'mock'.")
        print("To run live mode, configure a live provider (e.g. LLM_PROVIDER=gemini) in .env.")
        return 1

    if not config.api_key or not config.api_key.strip():
        print(f"ERROR: Provider '{provider_name}' is configured, but LLM_API_KEY is missing.")
        return 1

    live_examples = [
        {
            "id": 1,
            "title": "Example 1: Single Fact",
            "input": "Acme Corp announced that it plans to open a new office in Jaipur in 2027.",
            "expected_concept": "1 factual claim.",
        },
        {
            "id": 2,
            "title": "Example 2: Multiple Facts",
            "input": (
                "Acme Corp announced that it will invest $2 billion in a new factory in Texas. "
                "The company said the facility will employ 3,000 workers."
            ),
            "expected_concept": "2 distinct factual/attributed claims.",
        },
        {
            "id": 3,
            "title": "Example 3: Opinion",
            "input": "I think this movie is terrible and completely boring.",
            "expected_concept": "0 checkable factual claims.",
        },
        {
            "id": 4,
            "title": "Example 4: Attribution",
            "input": "The finance minister said inflation fell to 4.2% in August.",
            "expected_concept": "Attribution preserved; not rewritten as unqualified fact.",
        },
        {
            "id": 5,
            "title": "Example 5: False-Looking Assertion",
            "input": "The Moon is made entirely of cheese.",
            "expected_concept": "Claim extracted without verification or correction.",
        },
    ]

    extractor = ClaimExtractor()
    total_calls = len(live_examples)
    success_count = 0

    for idx, item in enumerate(live_examples):
        if idx > 0:
            # Respect Gemini Free Tier rate limit (5 requests per minute -> 15s pacing)
            print("Pacing request to respect Gemini Free Tier rate limit (15s)...")
            time.sleep(15)

        print(f"\n--- [Live Call {item['id']}/{total_calls}] {item['title']} ---")
        print(f"Input text:\n  \"{item['input']}\"")
        print(f"Expected concept: {item['expected_concept']}")

        for attempt in range(3):
            start_time = time.perf_counter()
            try:
                result = extractor.extract_claims(item["input"])
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                print("Status: SUCCESS")
                print(f"Provider: {extractor.llm_client.provider.provider_name}")
                print(f"Model:    {extractor.llm_client.provider.model_name}")

                latency_display = (
                    f"{extractor.last_metadata.latency_ms:.2f} ms"
                    if extractor.last_metadata and extractor.last_metadata.latency_ms is not None
                    else f"{elapsed_ms:.2f} ms"
                )
                print(f"Latency:  {latency_display}")

                if extractor.last_metadata and extractor.last_metadata.total_tokens is not None:
                    print(
                        f"Token Usage: input={extractor.last_metadata.input_tokens}, "
                        f"output={extractor.last_metadata.output_tokens}, "
                        f"total={extractor.last_metadata.total_tokens}"
                    )

                print(f"Extracted Claims ({len(result.claims)}):")
                if not result.claims:
                    print("  (Empty claims list — input identified as non-factual/opinion)")
                else:
                    for c in result.claims:
                        ctype = c.claim_type.value if c.claim_type else "unspecified"
                        print(f"  - Claim: \"{c.text}\"")
                        print(f"    Checkable: {c.checkable} | Type: {ctype}")

                success_count += 1
                break

            except Exception as exc:
                err_msg = str(exc)
                if ("retry in" in err_msg.lower() or "503" in err_msg or "resource_exhausted" in err_msg.lower()) and attempt < 2:
                    import re
                    match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_msg, re.IGNORECASE)
                    wait_sec = float(match.group(1)) + 2.0 if match else 15.0
                    print(f"Wait required by quota/server ({wait_sec:.1f}s). Sleeping...")
                    time.sleep(wait_sec)
                    continue

                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                print(f"Status: FAILED ({type(exc).__name__}): {exc}")
                print(f"Latency: {elapsed_ms:.2f} ms")
                break


    print("\n" + "=" * 70)
    print(f"Live Demonstration Completed: {success_count}/{total_calls} successful calls.")
    print("=" * 70)
    return 0 if success_count == total_calls else 1



def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate Claim Extraction Service.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Execute live LLM invocation (requires configured credentials)",
    )
    args = parser.parse_args()

    if args.live:
        sys.exit(run_live_demo())
    else:
        sys.exit(run_mock_demo())


if __name__ == "__main__":
    main()
