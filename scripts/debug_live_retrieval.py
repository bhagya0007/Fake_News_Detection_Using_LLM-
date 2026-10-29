"""Diagnostic utility for debugging live GDELT evidence retrieval (Post-V2.6 Patch).

Usage:
    python scripts/debug_live_retrieval.py "claim text"

Prints:
- Claim text and plan strategies
- Sequential attempts (endpoint, strategy, query, status, result count, elapsed ms)
- Merged raw count & deduplicated count
- Top reranked candidate details (title, domain, quality, similarity, URL)

Zero stance reasoning or LLM calls are performed by this script.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.evidence_query_builder import EvidenceQueryBuilder
from backend.app.services.gdelt_client import GDELTClient
from backend.app.services.live_evidence_retriever import LiveEvidenceRetriever

# Unicode safety on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def debug_retrieval(claim_text: str, top_k: int = 5) -> int:
    print("=" * 76)
    print("LIVE GDELT EVIDENCE RETRIEVAL DIAGNOSTIC")
    print("=" * 76)
    print(f"CLAIM: \"{claim_text}\"\n")

    builder = EvidenceQueryBuilder()
    plan = builder.build_plan(claim_text)

    print("QUERY PLAN:")
    print(f"  Detected Year: {plan.detected_year} | Is Historical: {plan.is_historical}")
    for i, q in enumerate(plan.queries, start=1):
        print(f"  Attempt {i}: [{q.strategy.upper():13}] endpoint={q.endpoint_preference:7} query='{q.query}' timespan={q.timespan or 'default'}")
    print()

    client = GDELTClient(timeout=12.0)
    retriever = LiveEvidenceRetriever(gdelt_client=client, query_builder=builder, top_k=top_k)

    t0 = time.perf_counter()
    items, meta = retriever.retrieve_for_claim(claim_text)
    total_elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    print("SEARCH EXECUTION TRACE:")
    search_trace = meta.get("search_trace", [])
    if search_trace:
        for idx, att in enumerate(search_trace, start=1):
            print(f"  Attempt {idx}:")
            print(f"    Endpoint:     {att.get('endpoint')}")
            print(f"    Strategy:     {att.get('strategy')}")
            print(f"    Query:        '{att.get('query')}'")
            print(f"    HTTP/Status:  {att.get('status')}")
            print(f"    Raw Count:    {att.get('result_count')}")
            print(f"    Elapsed:      {att.get('elapsed_ms')} ms")
    else:
        print("  No search attempts recorded in trace.")

    print(f"\nStop Reason: {meta.get('stop_reason')} | Overall Status: {meta.get('status')}")
    print(f"Total Execution Time: {total_elapsed_ms} ms")
    print(f"Merged Raw Count:     {meta.get('raw_candidates_count', 0)} (Context: {meta.get('context_count', 0)}, DOC: {meta.get('doc_count', 0)})")
    print(f"Deduplicated Count:   {meta.get('deduplicated_count', 0)}")
    print(f"Unique Source Domains: {meta.get('unique_source_domains', 0)}")

    if meta.get("error"):
        print(f"\nNotice / Error: {meta['error']}")

    print("\n" + "-" * 76)
    print(f"TOP RERANKED CANDIDATES (Top-{len(items)}):")
    print("-" * 76)

    if not items:
        print("  -> No candidate evidence items discovered within search parameters.")
    else:
        for rank, item in enumerate(items, start=1):
            print(f"\n  ({rank}) [{item.source_domain or 'unknown'}] {item.title}")
            print(f"      URL:        {item.source_url}")
            print(f"      Source:     {item.retrieval_source} | Quality: {item.evidence_quality}")
            print(f"      Date:       {item.published_date or 'N/A'} | Stance: {item.stance.value}")
            print(f"      Similarity: {item.similarity_score}")
            snippet = item.evidence_text[:140].strip().replace("\n", " ")
            print(f"      Snippet:    {snippet}...")

    print("\n" + "=" * 76)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug live GDELT retrieval for a claim.")
    parser.add_argument("claim", type=str, nargs="?", help="Claim text to investigate")
    parser.add_argument("--top-k", type=int, default=5, help="Top k items to retrieve (default: 5)")
    args = parser.parse_args()

    claim = args.claim
    if not claim:
        claim = (
            "The 2026 Nobel Prize in Physiology or Medicine was awarded to three scientists "
            "for work that led to using light to study brain activity."
        )

    return debug_retrieval(claim, top_k=args.top_k)


if __name__ == "__main__":
    sys.exit(main())
