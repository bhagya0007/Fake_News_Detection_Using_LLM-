"""End-to-End Live Demonstration of V2.3 Claim-Evidence Stance Reasoning Pipeline.

Demonstrates the complete multi-stage pipeline:
1. Input Statement / News Text
2. Groq-Powered Atomic Claim Extraction
3. GDELT Real-World Candidate News Discovery
4. Sentence-Transformers MiniLM Semantic Reranking
5. Groq Evidence Stance Assessment (Single structured batch call)
6. Deterministic Evidence Consensus & Near-Duplicate Clustering
7. Deterministic Decision Fusion & Abstention

Run:
    python scripts/demo_v2_real_news.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
import time

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Windows terminal doesn't crash on international Unicode characters
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from backend.app.schemas import VerdictLabel
from backend.app.services.hybrid_engine import HybridAnalysisEngine

# 3 real-world news claims for demonstration
DEMO_STATEMENTS = [
    {
        "topic": "Clean Energy Infrastructure",
        "text": "Solar power generation and renewable energy projects expanded globally during 2024.",
    },
    {
        "topic": "Global Space Missions",
        "text": "NASA Artemis lunar mission prepares rockets and spacecraft for deep space exploration.",
    },
    {
        "topic": "Semiconductor & AI Industry",
        "text": "Artificial intelligence computing demand prompted major expansions in semiconductor chip fabrication facilities.",
    },
]


def run_demo() -> None:
    print("=" * 78)
    print("V2.3 DEMO: END-TO-END CLAIM-EVIDENCE STANCE REASONING & CONSENSUS")
    print("=" * 78)
    print("Single LLM Provider: Groq (openai/gpt-oss-20b)")
    print("Evidence Discovery: GDELT Context 2.0 + DOC 2.0 (Public HTTP/HTTPS, No API Key)")
    print("Semantic Reranking: sentence-transformers/all-MiniLM-L6-v2 (Local Offline)")
    print("Stance Reasoning: Structured Groq Prompt with Prompt-Injection Defense")
    print("Final Authority: Purely Deterministic Evidence Consensus + Decision Fusion")
    print("=" * 78)

    engine = HybridAnalysisEngine()

    for idx, sample in enumerate(DEMO_STATEMENTS, start=1):
        print(f"\n[{idx}/3] TOPIC: {sample['topic']}")
        print(f"Input Statement: \"{sample['text']}\"")
        print("-" * 78)

        t_start = time.perf_counter()
        try:
            response = engine.analyze_text(sample["text"])
            total_elapsed = round((time.perf_counter() - t_start) * 1000, 1)

            # 1. Claims
            print(f"1. Extracted Checkable Claims ({len(response.claims)}):")
            for c_idx, cr in enumerate(response.claims):
                print(f"   [{c_idx}] {cr.claim.text} (Checkable: {cr.claim.checkable})")

            # 2. Retrieved Candidates
            print(f"\n2. Retrieved Candidate Evidence ({len(response.evidence)} items):")
            for e_idx, ev in enumerate(response.evidence[:3], start=1):
                domain = ev.source_domain or "Unknown Publisher"
                url = ev.source_url or "N/A"
                print(f"   ({e_idx}) [{domain}] {ev.title}")
                print(f"       URL: {url}")
                print(f"       Similarity: {ev.similarity_score} | Quality: {ev.evidence_quality}")
                print(f"       Assessed Stance: {ev.stance.value} | Attribution: {ev.attribution_type or 'unclear'}")
                if ev.stance_reason:
                    print(f"       Stance Reason: {ev.stance_reason}")

            # 3. Deterministic Consensus
            print("\n3. Deterministic Evidence Consensus:")
            if response.consensus:
                print(f"   Overall Stance: {response.consensus.overall_stance}")
                print(f"   Sufficient Corroboration: {response.consensus.overall_sufficient}")
                print(f"   Eligible Clusters: {response.consensus.total_eligible_clusters}")
                print(f"   Conflict Detected: {response.consensus.conflict_detected}")
                print(f"   Justification: {response.consensus.reason}")
            else:
                print("   Consensus: Not evaluated or skipped.")

            # 4. Final Verdict
            print("\n4. Final Hybrid Verdict (Governed by Fusion Engine):")
            print(f"   Verdict:       {response.verdict.value.upper()}")
            print(f"   Support Score: {response.support_score:.4f}")
            print(f"   Abstained:     {response.verdict == VerdictLabel.unverifiable}")
            print(f"   Reason:        {response.reason}")

            # 5. Timings Breakdown
            if response.timings:
                t = response.timings
                print("\n5. Latency Breakdown:")
                print(f"   Claim Extraction: {t.claim_extraction_ms:.1f} ms | Retrieval: {t.retrieval_ms:.1f} ms")
                print(f"   Stance Reasoning: {t.stance_reasoning_ms:.1f} ms | Consensus: {t.consensus_ms:.1f} ms")
                print(f"   Decision Fusion:  {t.fusion_ms:.1f} ms | Total Pipeline: {total_elapsed:.1f} ms")

        except Exception as exc:
            print(f"   [Error executing demo]: {exc}")

        print("=" * 78)
        if idx < len(DEMO_STATEMENTS):
            time.sleep(3.0)


if __name__ == "__main__":
    run_demo()
