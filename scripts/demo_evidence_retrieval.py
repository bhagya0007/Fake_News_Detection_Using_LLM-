"""Demonstration script for local evidence retrieval (Step 13).

Evaluates the local evidence retrieval system across four representative query scenarios:
Scenario A: Query directly matching Acme factory evidence (Expected: retrieved item)
Scenario B: Query matching Acme worker hiring evidence (Expected: retrieved item)
Scenario C: Query about the Moon made of cheese (Expected: NO SUFFICIENT EVIDENCE)
Scenario D: Unrelated query (Expected: NO SUFFICIENT EVIDENCE)

STRICT SAFETY & PROVENANCE GUARANTEES:
- Uses ONLY clearly marked synthetic demo fixtures (is_synthetic=True).
- Evaluates similarity scores without conflating similarity with truth or support.
- Does NOT print a final fake/real verdict.
- Zero external API/LLM calls.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.app.services.evidence_retriever import (
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K,
    EvidenceRetriever,
)


def run_demo(
    index_dir: Path,
    kb_path: Path,
    backend: str = "auto",
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
) -> None:
    """Run evidence retrieval demo across the required test scenarios."""
    print("======================================================================")
    print("Local Evidence Retrieval Demonstration (Step 13)")
    print(f"Target Index Dir : {index_dir}")
    print(f"Fallback KB Path : {kb_path}")
    print(f"Default Threshold: {threshold}")
    print(f"Default Top-k    : {top_k}")
    print("======================================================================\n")

    start_time = time.perf_counter()

    # Initialize retriever
    retriever = EvidenceRetriever(
        index_dir=index_dir if index_dir.exists() else None,
        kb_path=kb_path if kb_path.exists() else None,
        backend=backend,
        threshold=threshold,
        top_k=top_k,
    )

    print(f"Active Retrieval Backend: {retriever.backend_name}")
    print(f"Total Indexed Evidence Records: {len(retriever.records)}")
    print(f"Configured Similarity Threshold: {retriever.threshold}\n")

    scenarios = [
        (
            "Scenario A (Direct Factory Match)",
            "Acme Corp announced a new factory in Texas.",
            "Expected: Retrieve Acme Texas manufacturing evidence item.",
        ),
        (
            "Scenario B (Hiring Plans Match)",
            "Acme plans to hire 3,000 workers.",
            "Expected: Retrieve Acme 3,000 worker recruitment evidence item.",
        ),
        (
            "Scenario C (Unsubstantiated Proposition)",
            "The Moon is made of cheese.",
            "Expected: NO SUFFICIENT EVIDENCE (no matching evidence in KB).",
        ),
        (
            "Scenario D (Completely Unrelated Query)",
            "Deep ocean hydrothermal vents host unique chemotrophic ecosystems.",
            "Expected: NO SUFFICIENT EVIDENCE (similarity scores fall below threshold).",
        ),
    ]

    for title, query, expectation in scenarios:
        print("----------------------------------------------------------------------")
        print(f"{title}")
        print(f"Query      : \"{query}\"")
        print(f"Expectation: {expectation}")

        result = retriever.retrieve(query)

        print(f"Backend    : {result.retrieval_backend}")
        print(f"Threshold  : {result.threshold}")
        print(f"No Sufficient Evidence: {result.no_sufficient_evidence}")
        print(f"Retrieved Count       : {len(result.items)}")

        if result.no_sufficient_evidence or not result.items:
            print("Status     : [NO SUFFICIENT EVIDENCE RETURNED]")
        else:
            print("Retrieved Evidence Items:")
            for idx, item in enumerate(result.items, start=1):
                print(f"  [{idx}] ID: {item.id}")
                print(f"      Source Name     : {item.source_name} (Synthetic: {item.is_synthetic})")
                print(f"      Similarity Score: {item.similarity_score} (NOTE: Similarity != Stance)")
                print(f"      Recorded Stance : {item.stance}")
                print(f"      Claim Summary   : {item.claim_summary}")
                print(f"      Evidence Text   : {item.evidence_text[:120]}...")
                if item.source_url:
                    print(f"      Source URL      : {item.source_url}")

        print()

    total_duration = time.perf_counter() - start_time
    print("======================================================================")
    print(f"Demonstration Completed Successfully in {total_duration:.4f} seconds.")
    print("Zero external API calls made. All retrieval executed locally.")
    print("======================================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate local evidence retrieval.")
    parser.add_argument(
        "--index-dir",
        type=str,
        default="models/evidence",
        help="Path to prebuilt index directory.",
    )
    parser.add_argument(
        "--kb",
        type=str,
        default="data/sample/evidence_demo.jsonl",
        help="Path to fallback raw evidence KB.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "sentence_transformer", "tfidf_lexical"],
        help="Retrieval backend to use.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Similarity threshold.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Top-k items to retrieve.",
    )

    args = parser.parse_args()
    run_demo(
        index_dir=Path(args.index_dir),
        kb_path=Path(args.kb),
        backend=args.backend,
        threshold=args.threshold,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
