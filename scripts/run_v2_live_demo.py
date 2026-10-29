"""Track E: Real-World Live Pipeline Demonstration Runner.

Evaluates 3-4 diverse real-world statements using live Groq LLM and live GDELT retrieval:
- Tests real-world evidence discovery from GDELT Context & DOC APIs
- Tests Groq structured claim extraction and stance reasoning
- Tests deterministic near-duplicate clustering and corroboration gating
- Tests deterministic hybrid fusion and safe abstention
- Emits structured telemetry report to reports/v2_final/live_demo_results.json
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Fix Windows stdout encoding for international Unicode news titles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from backend.app.schemas import VerdictLabel
from backend.app.services.hybrid_engine import HybridAnalysisEngine

OUTPUT_PATH = PROJECT_ROOT / "reports" / "v2_final" / "live_demo_results.json"

DEMO_INPUTS = [
    {
        "id": "live_demo_01",
        "category": "factual_supported_news",
        "text": "Solar energy generation and renewable electricity capacity expanded globally during 2024.",
    },
    {
        "id": "live_demo_02",
        "category": "fabricated_hoax_unsupported",
        "text": "NASA astronomers confirmed a massive 5-mile asteroid will strike the Pacific Ocean next Tuesday.",
    },
    {
        "id": "live_demo_03",
        "category": "scientific_space_discovery",
        "text": "The James Webb Space Telescope observed atmospheric molecules in distant exoplanet systems.",
    },
    {
        "id": "live_demo_04",
        "category": "subjective_opinion_non_checkable",
        "text": "Renewable solar energy is aesthetically far more beautiful than coal-fired power stations.",
    },
]


def run_live_pipeline_demo() -> Dict[str, Any]:
    print("=" * 75)
    print("TRACK E: REAL-WORLD LIVE PIPELINE DEMONSTRATION")
    print("=" * 75)
    print("Live LLM Provider: Groq (openai/gpt-oss-20b)")
    print("Live Evidence Discovery: GDELT Context 2.0 + DOC 2.0 (Public HTTP/HTTPS, No API Key)")
    print("Local Reranker: sentence-transformers/all-MiniLM-L6-v2 (Offline)")
    print("Deterministic Authority: Near-Duplicate Clustering + Consensus Gating + Fusion")
    print("=" * 75)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    engine = HybridAnalysisEngine()

    results: List[Dict[str, Any]] = []

    for idx, sample in enumerate(DEMO_INPUTS, start=1):
        sample_id = sample["id"]
        category = sample["category"]
        text = sample["text"]

        print(f"\n[{idx}/{len(DEMO_INPUTS)}] ID: {sample_id} ({category})")
        print(f"Input Statement: \"{text}\"")
        print("-" * 75)

        t_start = time.perf_counter()
        try:
            response = engine.analyze_text(text)
            elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

            # Extract structured claims telemetry
            claims_data = [
                {
                    "id": cr.claim.id,
                    "text": cr.claim.text,
                    "checkable": cr.claim.checkable,
                    "claim_type": cr.claim.claim_type.value if cr.claim.claim_type else "event",
                    "evidence_count": len(cr.evidence),
                }
                for cr in response.claims
            ]

            # Extract candidate evidence telemetry
            evidence_data = [
                {
                    "id": ev.id,
                    "title": ev.title,
                    "domain": ev.source_domain,
                    "url": ev.source_url,
                    "similarity": ev.similarity_score,
                    "quality": ev.evidence_quality,
                    "stance": ev.stance.value if ev.stance else "unknown",
                    "attribution_type": ev.attribution_type.value if ev.attribution_type else None,
                    "stance_reason": ev.stance_reason,
                    "is_synthetic": ev.is_synthetic,
                }
                for ev in response.evidence[:5]
            ]

            # Consensus data
            consensus_data = None
            if response.consensus:
                consensus_data = {
                    "overall_stance": response.consensus.overall_stance,
                    "overall_sufficient": response.consensus.overall_sufficient,
                    "total_eligible_clusters": response.consensus.total_eligible_clusters,
                    "total_unique_domains": response.consensus.total_unique_domains,
                    "conflict_detected": response.consensus.conflict_detected,
                    "reason": response.consensus.reason,
                }

            # Classical ML assessment
            ml_data = None
            if response.ml:
                ml_data = {
                    "model_name": response.ml.model_name,
                    "model_version": response.ml.model_version,
                    "predicted_label": response.ml.label.value,
                    "probabilities": {
                        "likely_fake": response.ml.probabilities.likely_fake,
                        "likely_reliable": response.ml.probabilities.likely_reliable,
                        "misleading": response.ml.probabilities.misleading,
                    },
                }

            # Red flags
            rf_data = None
            if response.red_flags:
                rf_data = {
                    "total_flags": response.red_flags.summary.total_flags,
                    "high_severity_flags": response.red_flags.summary.high_severity_flags,
                    "medium_severity_flags": response.red_flags.summary.medium_severity_flags,
                    "low_severity_flags": response.red_flags.summary.low_severity_flags,
                }

            # Latency breakdown
            timings_data = None
            if response.timings:
                timings_data = {
                    "cleaning_ms": response.timings.cleaning_ms,
                    "claim_extraction_ms": response.timings.claim_extraction_ms,
                    "ml_ms": response.timings.ml_ms,
                    "red_flags_ms": response.timings.red_flags_ms,
                    "retrieval_ms": response.timings.retrieval_ms,
                    "live_retrieval_ms": getattr(response.timings, "live_retrieval_ms", 0.0),
                    "stance_reasoning_ms": getattr(response.timings, "stance_reasoning_ms", 0.0),
                    "consensus_ms": getattr(response.timings, "consensus_ms", 0.0),
                    "fusion_ms": response.timings.fusion_ms,
                    "total_ms": elapsed_ms,
                }

            case_result = {
                "id": sample_id,
                "category": category,
                "input_text": text,
                "claims": claims_data,
                "candidate_evidence_count": len(response.evidence),
                "candidate_evidence": evidence_data,
                "evidence_consensus": consensus_data,
                "classical_ml": ml_data,
                "red_flags": rf_data,
                "final_verdict": response.verdict.value,
                "support_score": response.support_score,
                "abstained": response.verdict == VerdictLabel.unverifiable,
                "fusion_reason": response.reason,
                "component_statuses": [
                    {"component": cs.component, "status": cs.status.value, "message": cs.message}
                    for cs in response.component_statuses
                ],
                "timings": timings_data,
                "status": "success",
            }

            print(f"Verdict: {response.verdict.value.upper()} | Abstained: {case_result['abstained']}")
            print(f"Evidence candidates found: {len(response.evidence)}")
            if consensus_data:
                print(f"Consensus: stance={consensus_data['overall_stance']}, sufficient={consensus_data['overall_sufficient']}, clusters={consensus_data['total_eligible_clusters']}")
            print(f"Total latency: {elapsed_ms} ms")

        except Exception as exc:
            print(f"Execution failed: {exc}")
            case_result = {
                "id": sample_id,
                "category": category,
                "input_text": text,
                "status": "error",
                "error": str(exc),
            }

        results.append(case_result)
        # Polite spacing between external live network calls
        if idx < len(DEMO_INPUTS):
            time.sleep(2.0)

    summary_payload = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "total_inputs": len(DEMO_INPUTS),
        "llm_provider": "groq (openai/gpt-oss-20b)",
        "evidence_discovery": "GDELT Context 2.0 + DOC 2.0",
        "semantic_reranker": "sentence-transformers/all-MiniLM-L6-v2",
        "results": results,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("\n" + "=" * 75)
    print(f"Track E Live Demo complete. Telemetry saved to: {OUTPUT_PATH}")
    print("=" * 75)

    return summary_payload


if __name__ == "__main__":
    run_live_pipeline_demo()
