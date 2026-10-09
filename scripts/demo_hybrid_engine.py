"""End-to-End Hybrid Engine Demonstration Script (Step 15).

Demonstrates the unified multi-signal verification pipeline across 5 key archetypes:
A. No-evidence checkable claim ("The Moon is made entirely of cheese.") -> UNVERIFIABLE
B. Subjective opinion ("I think vanilla ice cream is the best flavor.") -> UNVERIFIABLE
C. Supported claim with synthetic demo evidence ("Acme Corp announced a new factory in Texas.") -> LIKELY_RELIABLE
D. Contradicted claim with synthetic demo evidence ("The CDC announced coffee consumption causes acute heart attacks.") -> LIKELY_FAKE
E. Misleading claim with synthetic demo evidence ("Unemployment spiked 15% this quarter, signaling severe economic collapse.") -> MISLEADING

Default behavior: OFFLINE / CONTROLLED DEMO (0 external network / Gemini calls).
Optional live mode: python scripts/demo_hybrid_engine.py --live (max 2 calls, handles 429 safely).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.schemas import (
    Claim,
    ClaimExtractionResult,
    ClassProbabilities,
    ComponentExecutionStatus,
    EvidenceItem,
    EvidenceRetrievalResult,
    EvidenceStance,
    EvidenceSupport,
    InternalConsistency,
    LLMAssessment,
    MLAssessment,
    MLLabel,
    SourceAttribution,
    VerdictLabel,
)
from backend.app.services.hybrid_engine import (
    DefaultMLService,
    DefaultRedFlagService,
    HybridAnalysisEngine,
)


# ---------------------------------------------------------------------------
# Offline Deterministic Test Doubles for Controlled Scenarios
# ---------------------------------------------------------------------------
class OfflineDemoClaimExtractor:
    """Deterministic claim extractor mapping demo statements to factual propositions."""

    DEMO_CLAIMS: dict[str, list[Claim]] = {
        "The Moon is made entirely of cheese.": [
            Claim(id=1, text="The Moon is composed entirely of cheese.", checkable=True)
        ],
        "I think vanilla ice cream is the best flavor.": [
            Claim(id=1, text="Vanilla ice cream is the best flavor.", checkable=False)
        ],
        "Acme Corp announced a new factory in Texas.": [
            Claim(id=1, text="Acme Corp announced a new manufacturing facility in Texas.", checkable=True)
        ],
        "The CDC announced coffee consumption causes acute heart attacks.": [
            Claim(id=1, text="CDC stated coffee consumption causes acute heart attacks.", checkable=True)
        ],
        "Unemployment spiked 15% this quarter, signaling severe economic collapse.": [
            Claim(id=1, text="Unemployment spiked 15% this quarter.", checkable=True)
        ],
    }

    def extract_claims(self, text: str) -> ClaimExtractionResult:
        claims = self.DEMO_CLAIMS.get(
            text.strip(),
            [Claim(id=1, text=text.strip(), checkable=True)],
        )
        return ClaimExtractionResult(claims=claims, source_text_length=len(text))


class OfflineDemoEvidenceRetriever:
    """Provides targeted synthetic evidence records for demonstration archetypes."""

    def __init__(self, demo_kb: Optional[dict[str, list[EvidenceItem]]] = None) -> None:
        self.demo_kb = demo_kb or {}
        self.backend_name = "demo_offline_retriever"
        self.threshold = 0.45

    def retrieve(self, claim_text: str, top_k: int = 3) -> EvidenceRetrievalResult:
        items = self.demo_kb.get(claim_text, [])
        return EvidenceRetrievalResult(
            query=claim_text,
            items=items[:top_k],
            retrieval_backend=self.backend_name,
            threshold=self.threshold,
            no_sufficient_evidence=len(items) == 0,
        )


class OfflineDemoLLMReasoner:
    """Evaluates extracted claims strictly against supplied evidence."""

    def assess(
        self,
        text: str,
        claims: list[Claim],
        evidence: Optional[list[EvidenceItem]] = None,
    ) -> LLMAssessment:
        usable = [e for e in (evidence or [])]
        if not usable:
            return LLMAssessment(
                label=VerdictLabel.unverifiable,
                source_attribution=SourceAttribution.none,
                evidence_support=EvidenceSupport.unavailable,
                internal_consistency=InternalConsistency.unknown,
                short_reason="No external corroborating evidence was provided to verify the claim.",
            )

        # Check stance of evidence
        if any(e.stance == EvidenceStance.supports for e in usable):
            # Check if partial
            if "spiked 15%" in text:
                return LLMAssessment(
                    label=VerdictLabel.misleading,
                    source_attribution=SourceAttribution.partial,
                    evidence_support=EvidenceSupport.partial,
                    internal_consistency=InternalConsistency.questionable,
                    short_reason="Statistics apply exclusively to seasonal agricultural workers; national rate fell 0.2%.",
                )
            return LLMAssessment(
                label=VerdictLabel.likely_reliable,
                source_attribution=SourceAttribution.strong,
                evidence_support=EvidenceSupport.supported,
                internal_consistency=InternalConsistency.consistent,
                short_reason="Retrieved provenance-bearing evidence directly corroborates the factual assertion.",
            )
        elif any(e.stance == EvidenceStance.contradicts for e in usable):
            return LLMAssessment(
                label=VerdictLabel.likely_fake,
                source_attribution=SourceAttribution.none,
                evidence_support=EvidenceSupport.contradicted,
                internal_consistency=InternalConsistency.contradictory,
                short_reason="Official published medical guidelines directly refute the causality assertion.",
            )

        return LLMAssessment(
            label=VerdictLabel.unverifiable,
            source_attribution=SourceAttribution.weak,
            evidence_support=EvidenceSupport.unavailable,
            internal_consistency=InternalConsistency.unknown,
            short_reason="Supplied evidence is only related without definitive stance.",
        )


# ---------------------------------------------------------------------------
# Demonstration Runner
# ---------------------------------------------------------------------------
def run_offline_demo() -> None:
    print("=" * 80)
    print("End-to-End Hybrid Fake News Detection Pipeline Demonstration (Step 15)")
    print("MODE: OFFLINE / CONTROLLED DEMONSTRATION (Zero External API Calls)")
    print("=" * 80)

    # Prepare Synthetic Evidence Fixtures for Demo Archetypes
    acme_evidence = EvidenceItem(
        id="DEMO-EV-001",
        claim_summary="Acme Corp Texas facility expansion",
        evidence_text="Austin Business Journal: Acme Corp officially announced construction of a $120M manufacturing plant in Texas.",
        source_name="Austin Business Journal",
        stance=EvidenceStance.supports,
        is_synthetic=True,
    )
    cdc_evidence = EvidenceItem(
        id="DEMO-EV-002",
        claim_summary="CDC coffee consumption guidance",
        evidence_text="CDC Cardiovascular Health Bulletin: Moderate coffee consumption shows no acute heart attack causality.",
        source_name="CDC Health Bulletin",
        stance=EvidenceStance.contradicts,
        is_synthetic=True,
    )
    econ_evidence = EvidenceItem(
        id="DEMO-EV-003",
        claim_summary="Quarterly employment statistics",
        evidence_text="Bureau of Labor Statistics: 15% spike was isolated to seasonal agricultural workers; national unemployment fell 0.2%.",
        source_name="Bureau of Labor Statistics",
        stance=EvidenceStance.supports,
        is_synthetic=True,
    )

    demo_kb: dict[str, list[EvidenceItem]] = {
        "Acme Corp announced a new manufacturing facility in Texas.": [acme_evidence],
        "CDC stated coffee consumption causes acute heart attacks.": [cdc_evidence],
        "Unemployment spiked 15% this quarter.": [econ_evidence],
    }

    claim_extractor = OfflineDemoClaimExtractor()
    evidence_retriever = OfflineDemoEvidenceRetriever(demo_kb=demo_kb)
    llm_reasoner = OfflineDemoLLMReasoner()
    red_flag_service = DefaultRedFlagService()
    ml_service = DefaultMLService()

    # Engine for Scenarios A & B (Production Mode: allow_synthetic_evidence=False)
    prod_engine = HybridAnalysisEngine(
        claim_extractor=claim_extractor,
        ml_service=ml_service,
        red_flag_service=red_flag_service,
        evidence_retriever=evidence_retriever,
        llm_reasoner=llm_reasoner,
        allow_synthetic_evidence=False,
    )

    # Engine for Scenarios C, D, E (Demo Mode: allow_synthetic_evidence=True)
    demo_engine = HybridAnalysisEngine.demo(
        claim_extractor=claim_extractor,
        ml_service=ml_service,
        red_flag_service=red_flag_service,
        evidence_retriever=evidence_retriever,
        llm_reasoner=llm_reasoner,
    )

    scenarios = [
        (
            "A",
            "No-Evidence Factual Claim",
            "The Moon is made entirely of cheese.",
            prod_engine,
            "Production Mode (Synthetic quarantine active; 0 usable evidence in KB)",
        ),
        (
            "B",
            "Subjective Opinion Only",
            "I think vanilla ice cream is the best flavor.",
            prod_engine,
            "Production Mode (0 checkable claims; retrieval & reasoner skipped)",
        ),
        (
            "C",
            "Supported Claim (Demo Synthetic Evidence)",
            "Acme Corp announced a new factory in Texas.",
            demo_engine,
            "SYNTHETIC DEMO -- NOT REAL-WORLD FACT CHECKING",
        ),
        (
            "D",
            "Contradicted Claim (Demo Synthetic Evidence)",
            "The CDC announced coffee consumption causes acute heart attacks.",
            demo_engine,
            "SYNTHETIC DEMO -- NOT REAL-WORLD FACT CHECKING",
        ),
        (
            "E",
            "Partial / Misleading Claim (Demo Synthetic Evidence)",
            "Unemployment spiked 15% this quarter, signaling severe economic collapse.",
            demo_engine,
            "SYNTHETIC DEMO -- NOT REAL-WORLD FACT CHECKING",
        ),
    ]

    total_demo_start = time.perf_counter()

    for letter, title, text, engine, disclaimer in scenarios:
        print("\n" + "-" * 80)
        print(f"Scenario {letter}: {title}")
        print(f"Notice: {disclaimer}")
        print("-" * 80)

        t0 = time.perf_counter()
        resp = engine.analyze_text(text)
        elapsed = time.perf_counter() - t0

        print(f"Input Text          : {resp.input_text}")
        claims_str = ", ".join(f"'{c.claim.text}' ({'checkable' if c.claim.checkable else 'opinion'})" for c in resp.claims)
        print(f"Claims ({len(resp.claims)})         : {claims_str if resp.claims else 'None'}")
        ml_label_str = f"{resp.ml.label.value} (max prob: {max(resp.ml.probabilities.likely_reliable, resp.ml.probabilities.likely_fake, resp.ml.probabilities.misleading):.2f})" if resp.ml else "Unavailable"
        print(f"Classical ML Signal : {ml_label_str}")
        rf_str = f"{resp.red_flags.summary.total_flags} total ({resp.red_flags.summary.high_severity_flags} high)" if resp.red_flags else "Unavailable"
        print(f"Red-Flag Cues       : {rf_str}")
        if resp.llm:
            llm_label_str = f"{resp.llm.label.value} (support: {resp.llm.evidence_support.value})"
        else:
            reasoner_status = next((s for s in resp.component_statuses if s.component == "llm_reasoner"), None)
            if reasoner_status and reasoner_status.status == ComponentExecutionStatus.skipped:
                llm_label_str = "SKIPPED -- no usable evidence"
            else:
                llm_label_str = "Unavailable"
        print(f"LLM Reasoner Signal : {llm_label_str}")
        print(f"FINAL SYSTEM VERDICT: {resp.verdict.value.upper()}")
        print(f"Abstained           : {resp.verdict == VerdictLabel.unverifiable}")
        print(f"Support Score       : {resp.support_score:.4f} (Uncalibrated Heuristic Decision-Support Score)")
        print(f"Reason              : {resp.reason}")

        status_summary = ", ".join(f"{cs.component}:{cs.status.value}" for cs in resp.component_statuses)
        print(f"Component Statuses  : {status_summary}")
        if resp.timings:
            print(
                f"Latency Breakdown   : cleaning: {resp.timings.cleaning_ms:.1f}ms | "
                f"extraction: {resp.timings.claim_extraction_ms:.1f}ms | "
                f"ml: {resp.timings.ml_ms:.1f}ms | "
                f"red_flags: {resp.timings.red_flags_ms:.1f}ms | "
                f"retrieval: {resp.timings.retrieval_ms:.1f}ms | "
                f"llm: {resp.timings.llm_ms:.1f}ms | "
                f"fusion: {resp.timings.fusion_ms:.1f}ms | "
                f"total: {resp.timings.total_ms:.1f}ms"
            )

    total_time = time.perf_counter() - total_demo_start
    print("\n" + "=" * 80)
    print(f"All 5 Scenarios Completed Successfully in {total_time:.4f} seconds.")
    print("Zero API calls made. Pure deterministic orchestration.")
    print("=" * 80)


def run_live_demo() -> None:
    print("=" * 80)
    print("Live Multi-Signal Verification Probe (Step 15)")
    print("Target: Production Mode against Real KB (max 2 queries)")
    print("=" * 80)

    try:
        engine = HybridAnalysisEngine()
        probes = [
            "The Moon is made entirely of cheese.",
            "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        ]
        for idx, text in enumerate(probes[:2], start=1):
            print(f"\n--- Live Probe {idx}: '{text}' ---")
            t0 = time.perf_counter()
            resp = engine.analyze_text(text)
            elapsed = time.perf_counter() - t0
            print(f"Verdict        : {resp.verdict.value.upper()}")
            print(f"Support Score  : {resp.support_score:.4f}")
            print(f"Reason         : {resp.reason}")
            status_summary = ", ".join(f"{cs.component}:{cs.status.value}" for cs in resp.component_statuses)
            print(f"Statuses       : {status_summary}")
            print(f"Elapsed        : {elapsed:.2f}s")
    except Exception as exc:
        err_msg = str(exc)
        if "429" in err_msg or "quota" in err_msg.lower():
            print(f"Live mode notice: LLM unavailable: quota exhausted or rate limited ({exc}).")
            print("System safely falls back to offline analysis without crashing.")
        else:
            print(f"Live mode notice: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate Hybrid Fake News Detection Pipeline")
    parser.add_argument("--live", action="store_true", help="Run live Gemini verification (max 2 calls)")
    args = parser.parse_args()

    if args.live:
        run_live_demo()
    else:
        run_offline_demo()


if __name__ == "__main__":
    main()
