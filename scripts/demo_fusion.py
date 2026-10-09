"""Demonstration script for the Deterministic Hybrid Decision Fusion & Abstention Engine (Step 14).

Demonstrates the 7 core multi-signal decision archetypes:
1. No evidence available -> Unverifiable (safe abstention)
2. Supported evidence + reliable LLM -> Likely Reliable
3. Contradicted evidence + fake LLM -> Likely Fake
4. Partial evidence + misleading LLM -> Misleading
5. ML / LLM disagreement (factual evidence outranks text-pattern classifier)
6. High red flags with supported evidence (sensational style does not negate verified truth)
7. Synthetic evidence quarantined in production mode

STRICT SAFETY & PROVENANCE GUARANTEES:
- Uses completely synthetic, strongly typed component outputs.
- Zero live LLM calls, zero model inference, zero web requests.
- CLEARLY MARKED AS A SYNTHETIC DEMONSTRATION.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.app.schemas import (
    ClassProbabilities,
    EvidenceItem,
    EvidenceRetrievalResult,
    EvidenceStance,
    EvidenceSupport,
    InternalConsistency,
    LLMAssessment,
    MLAssessment,
    MLLabel,
    RedFlag,
    RedFlagFeatures,
    RedFlagResult,
    RedFlagSummary,
    Severity,
    SourceAttribution,
    VerdictLabel,
)
from backend.app.services.fusion import fuse


def create_sample_red_flags(total: int = 0, high: int = 0) -> RedFlagResult:
    features = RedFlagFeatures(
        character_count=60,
        word_count=10,
        uppercase_letter_ratio=0.3 if high > 0 else 0.0,
        exclamation_count=high * 2,
        question_mark_count=0,
        repeated_punctuation_count=high,
        percentage_count=0,
        statistical_expression_count=0,
        url_count=0,
        sensational_phrase_count=high,
        certainty_phrase_count=0,
        urgency_phrase_count=high,
        attribution_phrase_count=0,
        quote_count=0,
        excessive_punctuation_ratio=0.05 if high > 0 else 0.0,
    )
    flags = [
        RedFlag(type="sensationalism", severity=Severity.high, message="Sensational vocabulary"),
        RedFlag(type="urgency_phrase", severity=Severity.high, message="Urgency call"),
    ][:high]
    summary = RedFlagSummary(
        total_flags=total,
        low_severity_flags=0,
        medium_severity_flags=total - high,
        high_severity_flags=high,
    )
    return RedFlagResult(flags=flags, features=features, summary=summary)


def run_demo() -> None:
    print("======================================================================")
    print("Hybrid Decision Fusion & Abstention Demonstration (Step 14)")
    print("NOTICE: SYNTHETIC DEMONSTRATION ONLY. NOT REAL-WORLD FACT CHECKING.")
    print("======================================================================\n")

    start_time = time.perf_counter()

    # Pre-configure component signals for 7 representative scenarios
    scenarios = [
        # 1. No evidence available -> Unverifiable
        (
            "Scenario 1: Externally Checkable Claim With NO Evidence",
            "A secret facility beneath Denver Airport generates synthetic earthquakes.",
            MLAssessment(
                label=MLLabel.likely_fake,
                probabilities=ClassProbabilities(likely_reliable=0.10, likely_fake=0.80, misleading=0.10),
                model_name="Tuned Classical TF-IDF",
            ),
            create_sample_red_flags(total=1, high=1),
            LLMAssessment(
                label=VerdictLabel.unverifiable,
                source_attribution=SourceAttribution.none,
                evidence_support=EvidenceSupport.unavailable,
                internal_consistency=InternalConsistency.consistent,
                short_reason="No external evidence provided to verify this checkable assertion.",
            ),
            None,
            False,
        ),
        # 2. Supported evidence + reliable LLM -> Likely Reliable
        (
            "Scenario 2: Verified Supported Evidence + Reliable LLM",
            "NASA confirmed James Webb Telescope detected water vapor on WASP-18b.",
            MLAssessment(
                label=MLLabel.likely_reliable,
                probabilities=ClassProbabilities(likely_reliable=0.85, likely_fake=0.05, misleading=0.10),
                model_name="Tuned Classical TF-IDF",
            ),
            create_sample_red_flags(total=0, high=0),
            LLMAssessment(
                label=VerdictLabel.likely_reliable,
                source_attribution=SourceAttribution.strong,
                evidence_support=EvidenceSupport.supported,
                internal_consistency=InternalConsistency.consistent,
                short_reason="Directly substantiated by NASA official announcement.",
            ),
            EvidenceRetrievalResult(
                query="NASA JWST WASP-18b water vapor",
                items=[
                    EvidenceItem(
                        id="real_nasa_001",
                        claim_summary="JWST WASP-18b water vapor discovery",
                        evidence_text="NASA press release confirms spectroscopic signatures of atmospheric water vapor on WASP-18b.",
                        source_name="NASA Goddard",
                        source_url="https://nasa.gov/jwst-wasp18b",
                        similarity_score=0.94,
                        stance=EvidenceStance.supports,
                        is_synthetic=False,
                    )
                ],
                retrieval_backend="sentence_transformer",
                threshold=0.45,
                no_sufficient_evidence=False,
            ),
            False,
        ),
        # 3. Contradicted evidence + fake LLM -> Likely Fake
        (
            "Scenario 3: Verified Contradicted Evidence + Fake LLM",
            "The CDC announced coffee consumption causes acute heart attacks.",
            MLAssessment(
                label=MLLabel.likely_fake,
                probabilities=ClassProbabilities(likely_reliable=0.08, likely_fake=0.82, misleading=0.10),
                model_name="Tuned Classical TF-IDF",
            ),
            create_sample_red_flags(total=1, high=1),
            LLMAssessment(
                label=VerdictLabel.likely_fake,
                source_attribution=SourceAttribution.weak,
                evidence_support=EvidenceSupport.contradicted,
                internal_consistency=InternalConsistency.consistent,
                short_reason="Directly contradicted by CDC clinical health bulletin.",
            ),
            EvidenceRetrievalResult(
                query="CDC coffee acute heart attack",
                items=[
                    EvidenceItem(
                        id="real_cdc_001",
                        claim_summary="Coffee consumption and cardiac health",
                        evidence_text="CDC health bulletin states extensive studies show moderate coffee intake does not cause acute heart attacks.",
                        source_name="CDC Bulletin",
                        source_url="https://cdc.gov/dietary-health-coffee",
                        similarity_score=0.89,
                        stance=EvidenceStance.contradicts,
                        is_synthetic=False,
                    )
                ],
                retrieval_backend="sentence_transformer",
                threshold=0.45,
                no_sufficient_evidence=False,
            ),
            False,
        ),
        # 4. Partial evidence + misleading LLM -> Misleading
        (
            "Scenario 4: Partial Evidence + Cherry-Picked Context",
            "Unemployment spiked 15% this quarter, signaling severe economic collapse.",
            MLAssessment(
                label=MLLabel.misleading,
                probabilities=ClassProbabilities(likely_reliable=0.20, likely_fake=0.20, misleading=0.60),
                model_name="Tuned Classical TF-IDF",
            ),
            create_sample_red_flags(total=2, high=1),
            LLMAssessment(
                label=VerdictLabel.misleading,
                source_attribution=SourceAttribution.partial,
                evidence_support=EvidenceSupport.partial,
                internal_consistency=InternalConsistency.questionable,
                short_reason="Statistics apply exclusively to seasonal agricultural workers; national rate fell 0.2%.",
            ),
            EvidenceRetrievalResult(
                query="Unemployment spiked 15%",
                items=[
                    EvidenceItem(
                        id="real_bls_001",
                        claim_summary="Quarterly seasonal labor unemployment",
                        evidence_text="BLS report notes a 15% rise in seasonal farm worker unemployment, whereas national unemployment decreased 0.2%.",
                        source_name="Bureau of Labor Statistics",
                        source_url="https://bls.gov/labor-summary",
                        similarity_score=0.86,
                        stance=EvidenceStance.supports,
                        is_synthetic=False,
                    )
                ],
                retrieval_backend="sentence_transformer",
                threshold=0.45,
                no_sufficient_evidence=False,
            ),
            False,
        ),
        # 5. ML / LLM Disagreement (Evidence Takes Precedence)
        (
            "Scenario 5: Signal Disagreement (Classical ML Disagrees With Real Evidence)",
            "Federal Reserve kept benchmark interest rates unchanged at 5.25%.",
            MLAssessment(
                label=MLLabel.likely_fake,  # Text-pattern classifier error
                probabilities=ClassProbabilities(likely_reliable=0.25, likely_fake=0.55, misleading=0.20),
                model_name="Tuned Classical TF-IDF",
            ),
            create_sample_red_flags(total=0, high=0),
            LLMAssessment(
                label=VerdictLabel.likely_reliable,
                source_attribution=SourceAttribution.strong,
                evidence_support=EvidenceSupport.supported,
                internal_consistency=InternalConsistency.consistent,
                short_reason="Official Federal Reserve statement confirms rates were held steady.",
            ),
            EvidenceRetrievalResult(
                query="Federal Reserve benchmark interest rates 5.25%",
                items=[
                    EvidenceItem(
                        id="real_fed_001",
                        claim_summary="FOMC interest rate decision",
                        evidence_text="The Federal Open Market Committee decided today to maintain the target range for the federal funds rate at 5.25 to 5.50 percent.",
                        source_name="Federal Reserve Board",
                        source_url="https://federalreserve.gov/fomc-release",
                        similarity_score=0.92,
                        stance=EvidenceStance.supports,
                        is_synthetic=False,
                    )
                ],
                retrieval_backend="sentence_transformer",
                threshold=0.45,
                no_sufficient_evidence=False,
            ),
            False,
        ),
        # 6. High Red Flags with Supported Evidence
        (
            "Scenario 6: High Stylometric Red Flags With Supported Factual Evidence",
            "BREAKING NEWS!!! YOU WON'T BELIEVE THIS: NASA rover lands successfully on Mars crater!",
            MLAssessment(
                label=MLLabel.likely_reliable,
                probabilities=ClassProbabilities(likely_reliable=0.75, likely_fake=0.15, misleading=0.10),
                model_name="Tuned Classical TF-IDF",
            ),
            create_sample_red_flags(total=3, high=2),  # Sensationalism & excessive caps
            LLMAssessment(
                label=VerdictLabel.likely_reliable,
                source_attribution=SourceAttribution.strong,
                evidence_support=EvidenceSupport.supported,
                internal_consistency=InternalConsistency.consistent,
                short_reason="Landing confirmed by NASA mission control telemetry.",
            ),
            EvidenceRetrievalResult(
                query="NASA rover landing Mars crater",
                items=[
                    EvidenceItem(
                        id="real_nasa_002",
                        claim_summary="NASA rover touch-down on Mars",
                        evidence_text="NASA confirms successful touchdown of the Perseverance rover in Jezero Crater.",
                        source_name="NASA Jet Propulsion Laboratory",
                        source_url="https://jpl.nasa.gov/news/mars-landing",
                        similarity_score=0.90,
                        stance=EvidenceStance.supports,
                        is_synthetic=False,
                    )
                ],
                retrieval_backend="sentence_transformer",
                threshold=0.45,
                no_sufficient_evidence=False,
            ),
            False,
        ),
        # 7. Synthetic Evidence Quarantined in Production Mode
        (
            "Scenario 7: Synthetic Demo Evidence Blocked by Production Gate",
            "Acme Corp announced a new factory in Texas.",
            MLAssessment(
                label=MLLabel.likely_reliable,
                probabilities=ClassProbabilities(likely_reliable=0.88, likely_fake=0.06, misleading=0.06),
                model_name="Tuned Classical TF-IDF",
            ),
            create_sample_red_flags(total=0, high=0),
            LLMAssessment(
                label=VerdictLabel.likely_reliable,
                source_attribution=SourceAttribution.strong,
                evidence_support=EvidenceSupport.supported,
                internal_consistency=InternalConsistency.consistent,
                short_reason="Supported by synthetic demo fixture.",
            ),
            EvidenceRetrievalResult(
                query="Acme Corp announced a new factory in Texas",
                items=[
                    EvidenceItem(
                        id="synth_001",
                        claim_summary="Acme Corp announced a new factory in Texas",
                        evidence_text="Acme Corp announced today that it will construct a new manufacturing facility in Texas.",
                        source_name="Synthetic Demo Source",
                        source_url="https://example.com/synthetic/acme",
                        similarity_score=0.95,
                        stance=EvidenceStance.supports,
                        is_synthetic=True,  # Synthetic fixture!
                    )
                ],
                retrieval_backend="sentence_transformer",
                threshold=0.45,
                no_sufficient_evidence=False,
            ),
            False,  # allow_synthetic_evidence=False (Production default)
        ),
    ]

    for title, claim, ml, rf, llm, ev, allow_synth in scenarios:
        print("----------------------------------------------------------------------")
        print(f"{title}")
        print(f"Target Claim        : \"{claim}\"")
        print(f"ML Signal           : {ml.label.value} (Prob: {max(ml.probabilities.likely_reliable, ml.probabilities.likely_fake, ml.probabilities.misleading):.2f})")
        print(f"LLM Signal          : {llm.label.value if llm else 'None'} (Support: {llm.evidence_support.value if llm else 'None'})")

        ev_status = "None"
        if ev and ev.items:
            synth_tag = "Synthetic" if ev.items[0].is_synthetic else "Verified Real"
            ev_status = f"{len(ev.items)} item(s) [{synth_tag}, Stance: {ev.items[0].stance.value}]"
        print(f"Evidence Status     : {ev_status}")
        print(f"Red-Flag Summary    : {rf.summary.total_flags} total ({rf.summary.high_severity_flags} high-severity)")

        result = fuse(
            ml=ml,
            red_flags=rf,
            llm=llm,
            evidence=ev,
            has_checkable_claims=True,
            allow_synthetic_evidence=allow_synth,
        )

        print(f"FINAL SYSTEM VERDICT: {result.verdict.value.upper()}")
        print(f"Abstained           : {result.abstained}")
        print(f"Support Score       : {result.support_score} (Internal Decision Strength)")
        print(f"Signals Used        : {result.signals_used}")
        print(f"Reasoning           : {result.reason}\n")

    duration = time.perf_counter() - start_time
    print("======================================================================")
    print(f"Demonstration Completed Successfully in {duration:.4f} seconds.")
    print("Zero API calls made. Zero external model calls inside fusion.")
    print("======================================================================")


if __name__ == "__main__":
    run_demo()
