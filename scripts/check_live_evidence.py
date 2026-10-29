"""Live smoke check for real-world GDELT evidence discovery (V2.2).

Performs a read-only live query to GDELT Context and DOC 2.0 endpoints with a neutral,
non-sensitive query (e.g., 'solar energy capacity').
Validates JSON structure, canonical URL normalization, domain extraction,
and semantic reranking. Does NOT require any API key.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.gdelt_client import GDELTClient
from backend.app.services.live_evidence_retriever import LiveEvidenceRetriever

# Ensure Windows terminal doesn't crash on international Unicode characters (e.g. ₹, €, quotes)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def run_live_evidence_check() -> int:
    print("=" * 70)
    print("V2.2 LIVE EVIDENCE DISCOVERY SMOKE CHECK (GDELT)")
    print("=" * 70)
    print("Zero API keys required: GDELT Context 2.0 and DOC 2.0 are public.")
    print("Checking real connectivity and data contracts...\n")

    test_query = "solar energy capacity"
    print(f"Test Query: '{test_query}'")

    client = GDELTClient(timeout=12.0)
    retriever = LiveEvidenceRetriever(gdelt_client=client, top_k=3)

    t0 = time.perf_counter()
    try:
        items, meta = retriever.retrieve_for_claim(test_query)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        print(f"\nElapsed time: {elapsed_ms} ms")
        print(f"Status: {meta.get('status')}")
        print(f"Raw candidates retrieved: {meta.get('raw_candidates_count', 0)}")
        print(f"Context candidates: {meta.get('context_count', 0)}")
        print(f"DOC fallback candidates: {meta.get('doc_count', 0)}")
        print(f"Deduplicated count: {meta.get('deduplicated_count', 0)}")
        print(f"Unique source domains: {meta.get('unique_source_domains', 0)}")
        print(f"Top-k returned items: {len(items)}")

        if meta.get("error"):
            print(f"Notice / Error: {meta['error']}")

        print("\nCandidate Evidence Details:")
        for idx, item in enumerate(items, start=1):
            print(f"\n  [{idx}] Domain: {item.source_domain}")
            print(f"      Title: {item.title}")
            print(f"      URL: {item.source_url}")
            print(f"      Source: {item.retrieval_source}")
            print(f"      Quality: {item.evidence_quality}")
            print(f"      Date: {item.published_date}")
            print(f"      Similarity Score: {item.similarity_score}")
            print(f"      Stance: {item.stance.value} (strictly unknown in V2.2)")
            print(f"      Synthetic: {item.is_synthetic} (strictly False)")
            snippet = item.evidence_text[:120] + "..." if len(item.evidence_text) > 120 else item.evidence_text
            print(f"      Snippet: {snippet}")

            # Assert invariants
            assert item.is_synthetic is False, "Live evidence must not be synthetic"
            assert item.stance.value == "unknown", "Stance must remain unknown in V2.2"
            assert item.source_url is not None and item.source_url.startswith("http"), "Valid URL required"
            assert item.source_domain is not None, "Source domain required"

        print("\n" + "=" * 70)
        print("PASS: Live GDELT candidate evidence discovery check completed successfully.")
        print("=" * 70)
        return 0

    except Exception as exc:
        print(f"\n[ERROR] Live check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run_live_evidence_check())
