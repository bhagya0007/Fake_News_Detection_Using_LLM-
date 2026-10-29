"""Live demo script showing real-world evidence discovery across diverse news claims (V2.2).

Demonstrates:
- Clean query construction without operator injection.
- Provenance tracking (domain, URL, date, quality).
- Source domain diversity count.
- Deterministic MiniLM semantic reranking.
- Non-claim of truth: candidates are evidence candidates with stance=unknown.
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


DEMO_CLAIMS = [
    {
        "topic": "Clean Energy Expansion",
        "claim": "Solar energy capacity is expanding globally with new utility-scale solar projects.",
    },
    {
        "topic": "Artificial Intelligence Hardware",
        "claim": "Artificial intelligence demand is driving semiconductor chip manufacturing.",
    },
    {
        "topic": "Space Exploration Programs",
        "claim": "NASA lunar exploration programs develop rockets for the Artemis mission.",
    },
    {
        "topic": "Automotive & Electric Vehicles",
        "claim": "Electric vehicle battery manufacturers announced investments into production capacity.",
    },
]


def run_demo() -> int:
    print("=" * 76)
    print("V2.2 DEMO: REAL-WORLD PROVENANCE-BEARING NEWS EVIDENCE DISCOVERY")
    print("=" * 76)
    print("Public Discovery Sources: GDELT Context 2.0 API + GDELT DOC 2.0 Fallback")
    print("Additional API Keys Required: ZERO (GROQ_API_KEY remains the only secret)")
    print("Epistemic Guard: Retrieved candidates are EVIDENCE CANDIDATES, not truth.")
    print("Stance Policy: All live items enter with stance='unknown' pending V2.3.")
    print("=" * 76)

    client = GDELTClient(timeout=12.0)
    retriever = LiveEvidenceRetriever(gdelt_client=client, top_k=3)

    for i, test_case in enumerate(DEMO_CLAIMS, start=1):
        print(f"\n[{i}/{len(DEMO_CLAIMS)}] TOPIC: {test_case['topic']}")
        print(f"Claim: \"{test_case['claim']}\"")

        t0 = time.perf_counter()
        try:
            items, meta = retriever.retrieve_for_claim(test_case["claim"])
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

            print(f"Discovery Time: {elapsed_ms} ms | Status: {meta.get('status')}")
            print(f"Sanitized Query: '{meta.get('query')}'")
            print(f"Raw Candidates: {meta.get('raw_candidates_count', 0)} "
                  f"(Context: {meta.get('context_count', 0)}, DOC: {meta.get('doc_count', 0)})")
            print(f"Deduplicated Count: {meta.get('deduplicated_count', 0)}")
            print(f"Unique Domains: {meta.get('unique_source_domains', 0)}")

            if not items:
                print("  -> No matching live news evidence candidates found within timespan.")
                continue

            print("\n  Top Provenance-Bearing Evidence Candidates (MiniLM Reranked):")
            for rank, item in enumerate(items, start=1):
                print(f"  ({rank}) [{item.source_domain or 'unknown'}] {item.title}")
                print(f"      URL: {item.source_url}")
                print(f"      Published: {item.published_date or 'N/A'} | Quality: {item.evidence_quality}")
                print(f"      Similarity Score: {item.similarity_score} | Stance: {item.stance.value}")
                snippet = item.evidence_text[:140].strip().replace("\n", " ")
                print(f"      Snippet: {snippet}...")
                print()

        except Exception as exc:
            print(f"  [Error processing claim]: {exc}")

        # Sleep briefly between queries to respect GDELT public endpoint pacing
        if i < len(DEMO_CLAIMS):
            time.sleep(3.5)

    print("=" * 76)
    print("Demo complete. All candidates carry verifiable URLs and publisher domains.")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(run_demo())
