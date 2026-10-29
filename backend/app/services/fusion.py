"""Deterministic Hybrid Decision Fusion and Abstention Engine (Step 14).

This engine combines signals from:
1. Classical ML Baseline (MLAssessment)
2. Deterministic Stylometric / Red-Flag Detector (RedFlagResult)
3. Structured Evidence-Bound LLM Reasoner (LLMAssessment)
4. Local Semantic Evidence Retrieval (EvidenceRetrievalResult)

CORE PRINCIPLES & SAFETY RULES:
- NOT a majority vote.
- Evidence availability and quality act as a strict gate for factual claims.
- For checkable claims without usable non-synthetic evidence, the system abstains with 'unverifiable'.
- Synthetic demo fixtures (is_synthetic=True) are rejected by default in production mode.
- Red flags reflect presentation style and NEVER independently produce fake/misleading verdicts.
- Classical ML is a text-pattern classifier (Macro F1 = 0.4345) and cannot override factual evidence.
- Zero LLM calls, zero model inference, zero web retrieval inside fusion.
- Purely deterministic, offline, and auditable.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

from backend.app.schemas import (
    DocumentEvidenceConsensus,
    EvidenceItem,
    EvidenceRetrievalResult,
    EvidenceStance,
    EvidenceSupport,
    FinalAssessment,
    FusionInput,
    FusionSignals,
    LLMAssessment,
    MLAssessment,
    MLLabel,
    RedFlagResult,
    VerdictLabel,
)

logger = logging.getLogger(__name__)


def get_usable_real_evidence(
    evidence: Optional[EvidenceRetrievalResult],
    allow_synthetic_evidence: bool = False,
) -> List[EvidenceItem]:
    """Filter retrieved evidence records according to synthetic quarantine policy.

    Args:
        evidence: The retrieval result containing candidate items.
        allow_synthetic_evidence: Whether to permit synthetic fixtures (test/demo only).

    Returns:
        List of usable EvidenceItem objects without mutating the original container.
    """
    if evidence is None or evidence.no_sufficient_evidence or not evidence.items:
        return []

    usable: List[EvidenceItem] = []
    for item in evidence.items:
        if item.is_synthetic and not allow_synthetic_evidence:
            continue
        usable.append(item)

    return usable


def _determine_model_agreement(
    ml_label: MLLabel,
    llm_label: Optional[VerdictLabel],
) -> str:
    """Determine directional agreement between classical ML and LLM reasoner."""
    if llm_label is None or llm_label == VerdictLabel.unverifiable:
        return "not_applicable"

    # Map MLLabel to VerdictLabel
    if ml_label.value == llm_label.value:
        return "agree"
    return "disagree"


def _compute_support_score(
    verdict: VerdictLabel,
    abstained: bool,
    ml: Optional[MLAssessment],
    llm: Optional[LLMAssessment],
    red_flags: Optional[RedFlagResult],
) -> float:
    """Calculate internal decision-support strength (0.0 to 1.0).

    IMPORTANT:
    This score is an internal decision-support metric reflecting multi-signal agreement
    and evidence availability. It is NOT an objective calibrated truth probability.
    """
    if abstained or verdict == VerdictLabel.unverifiable:
        return 0.0

    # Base score when evidence gate passes and verdict is grounded
    score = 0.70

    # Classical ML agreement modifier
    if ml is not None:
        ml_max_prob = max(
            ml.probabilities.likely_reliable,
            ml.probabilities.likely_fake,
            ml.probabilities.misleading,
        )
        if ml.label.value == verdict.value:
            # Agreement bonus proportional to model confidence
            score += 0.20 * ml_max_prob
        else:
            # Disagreement penalty
            score -= 0.15

    # Red-flags modifier (bounded, capped penalty)
    if red_flags is not None:
        high_flags = red_flags.summary.high_severity_flags
        if high_flags == 1:
            score -= 0.05
        elif high_flags >= 2:
            score -= 0.10

    # Clamp strictly between 0.0 and 1.0
    return round(float(max(0.0, min(1.0, score))), 4)


def fuse(
    ml: Optional[MLAssessment] = None,
    red_flags: Optional[RedFlagResult] = None,
    llm: Optional[LLMAssessment] = None,
    evidence: Optional[EvidenceRetrievalResult] = None,
    has_checkable_claims: bool = True,
    allow_synthetic_evidence: bool = False,
    evidence_consensus: Optional[DocumentEvidenceConsensus] = None,
) -> FinalAssessment:
    """Perform deterministic hybrid decision fusion and abstention.

    Args:
        ml: Optional validated output from classical ML model.
        red_flags: Optional validated output from stylometric red-flag detector.
        llm: Optional structured assessment from LLM reasoner.
        evidence: Optional retrieval result from local evidence KB.
        has_checkable_claims: Whether the input contains checkable factual claims.
        allow_synthetic_evidence: If True, synthetic fixtures are treated as usable
                                  (STRICTLY restricted to unit tests and explicit demo mode).
        evidence_consensus: Optional deterministic consensus result across evidence clusters.

    Returns:
        FinalAssessment with strict abstention consistency and audit telemetry.
    """
    # -------------------------------------------------------------------------
    # 0. Evidence Counts & Synthetic Quarantine
    # -------------------------------------------------------------------------
    raw_evidence_items = evidence.items if (evidence and not evidence.no_sufficient_evidence) else []
    synthetic_count = sum(1 for item in raw_evidence_items if item.is_synthetic)
    real_count = sum(1 for item in raw_evidence_items if not item.is_synthetic)

    usable_evidence = get_usable_real_evidence(
        evidence=evidence,
        allow_synthetic_evidence=allow_synthetic_evidence,
    )

    has_supporting_evidence = any(item.stance == EvidenceStance.supports for item in usable_evidence)
    has_contradicting_evidence = any(item.stance == EvidenceStance.contradicts for item in usable_evidence)
    only_related_or_unknown_evidence = (
        len(usable_evidence) > 0
        and all(item.stance in {EvidenceStance.related, EvidenceStance.unknown} for item in usable_evidence)
    )

    is_demo_mode = allow_synthetic_evidence and synthetic_count > 0
    demo_prefix = "[DEMO / SYNTHETIC EVIDENCE] " if is_demo_mode else ""

    ml_max_prob = (
        max(
            ml.probabilities.likely_reliable,
            ml.probabilities.likely_fake,
            ml.probabilities.misleading,
        )
        if ml is not None
        else None
    )
    ml_label = ml.label if ml is not None else None
    llm_label = llm.label if llm else None
    llm_support = llm.evidence_support if llm else None
    model_agreement = _determine_model_agreement(ml.label, llm_label) if ml is not None else "not_applicable"
    total_rf = red_flags.summary.total_flags if red_flags is not None else 0
    high_rf = red_flags.summary.high_severity_flags if red_flags is not None else 0

    # -------------------------------------------------------------------------
    # Rule 1: No Checkable Factual Claim
    # -------------------------------------------------------------------------
    if not has_checkable_claims:
        signals_used = ["claims_filter"]
        reason = (
            "No checkable factual claims were identified; the statement consists of "
            "subjective opinion, speculative assertion, or non-verifiable content."
        )
        return FinalAssessment(
            verdict=VerdictLabel.unverifiable,
            support_score=0.0,
            reason=reason,
            abstained=True,
            signals_used=signals_used,
            fusion_signals=FusionSignals(
                ml_label=ml_label,
                ml_max_probability=round(ml_max_prob, 4) if ml_max_prob is not None else None,
                llm_label=llm_label,
                llm_evidence_support=llm_support,
                evidence_available=False,
                evidence_support=EvidenceSupport.not_applicable,
                real_evidence_count=real_count,
                synthetic_evidence_count=synthetic_count,
                red_flag_count=total_rf,
                high_severity_red_flag_count=high_rf,
                model_agreement=model_agreement,
            ),
        )

    # -------------------------------------------------------------------------
    # Rule 2: No Usable Real Evidence
    # -------------------------------------------------------------------------
    if not usable_evidence:
        signals_used = ["evidence_gate"]
        if synthetic_count > 0 and not allow_synthetic_evidence:
            signals_used.append("synthetic_quarantine")
            reason = (
                "No usable provenance-bearing non-synthetic evidence was available in the production knowledge base "
                "(synthetic demo fixtures were rejected); system abstained despite model predictions."
            )
        else:
            reason = (
                "No usable provenance-bearing evidence was available to verify or refute the claim; "
                "the system abstained from making an ungrounded factual determination."
            )

        return FinalAssessment(
            verdict=VerdictLabel.unverifiable,
            support_score=0.0,
            reason=reason,
            abstained=True,
            signals_used=signals_used,
            fusion_signals=FusionSignals(
                ml_label=ml_label,
                ml_max_probability=round(ml_max_prob, 4) if ml_max_prob is not None else None,
                llm_label=llm_label,
                llm_evidence_support=llm_support,
                evidence_available=False,
                evidence_support=EvidenceSupport.unavailable,
                real_evidence_count=real_count,
                synthetic_evidence_count=synthetic_count,
                red_flag_count=total_rf,
                high_severity_red_flag_count=high_rf,
                model_agreement=model_agreement,
            ),
        )

    # -------------------------------------------------------------------------
    # Rule 2B (V2.3): Deterministic Evidence Consensus Gating
    # -------------------------------------------------------------------------
    if evidence_consensus is not None:
        if evidence_consensus.conflict_detected or evidence_consensus.overall_stance == "conflicting":
            signals_used = ["evidence_gate", "evidence_consensus_conflict"]
            reason = (
                f"{demo_prefix}Deterministic evidence consensus detected conflicting evidence across clusters "
                f"({evidence_consensus.reason}); system safely abstained."
            )
            return FinalAssessment(
                verdict=VerdictLabel.unverifiable,
                support_score=0.0,
                reason=reason,
                abstained=True,
                signals_used=signals_used,
                fusion_signals=FusionSignals(
                    ml_label=ml_label,
                    ml_max_probability=round(ml_max_prob, 4) if ml_max_prob is not None else None,
                    llm_label=llm_label,
                    llm_evidence_support=llm_support,
                    evidence_available=True,
                    evidence_support=EvidenceSupport.unavailable,
                    real_evidence_count=real_count,
                    synthetic_evidence_count=synthetic_count,
                    red_flag_count=total_rf,
                    high_severity_red_flag_count=high_rf,
                    model_agreement=model_agreement,
                    evidence_consensus_stance=evidence_consensus.overall_stance,
                    evidence_consensus_sufficient=evidence_consensus.overall_sufficient,
                    eligible_clusters_count=evidence_consensus.total_eligible_clusters,
                ),
            )

        if not evidence_consensus.overall_sufficient:
            signals_used = ["evidence_gate", "insufficient_consensus"]
            reason = (
                f"{demo_prefix}Candidate evidence failed conservative corroboration thresholds "
                f"({evidence_consensus.reason}); system safely abstained."
            )
            return FinalAssessment(
                verdict=VerdictLabel.unverifiable,
                support_score=0.0,
                reason=reason,
                abstained=True,
                signals_used=signals_used,
                fusion_signals=FusionSignals(
                    ml_label=ml_label,
                    ml_max_probability=round(ml_max_prob, 4) if ml_max_prob is not None else None,
                    llm_label=llm_label,
                    llm_evidence_support=llm_support,
                    evidence_available=True,
                    evidence_support=EvidenceSupport.unavailable,
                    real_evidence_count=real_count,
                    synthetic_evidence_count=synthetic_count,
                    red_flag_count=total_rf,
                    high_severity_red_flag_count=high_rf,
                    model_agreement=model_agreement,
                    evidence_consensus_stance=evidence_consensus.overall_stance,
                    evidence_consensus_sufficient=evidence_consensus.overall_sufficient,
                    eligible_clusters_count=evidence_consensus.total_eligible_clusters,
                ),
            )

        # Sufficient consensus established - set evidence flags for downstream rules
        if evidence_consensus.overall_stance == "supports":
            has_supporting_evidence = True
            has_contradicting_evidence = False
            only_related_or_unknown_evidence = False
        elif evidence_consensus.overall_stance == "contradicts":
            has_contradicting_evidence = True
            has_supporting_evidence = False
            only_related_or_unknown_evidence = False
        elif evidence_consensus.overall_stance == "partial":
            has_supporting_evidence = False
            has_contradicting_evidence = False
            only_related_or_unknown_evidence = False

    if only_related_or_unknown_evidence:
        signals_used = ["evidence_gate"]
        reason = (
            f"{demo_prefix}Retrieved evidence is only topically related or has an unknown stance, "
            "providing insufficient empirical grounding to verify or refute the claim."
        )
        return FinalAssessment(
            verdict=VerdictLabel.unverifiable,
            support_score=0.0,
            reason=reason,
            abstained=True,
            signals_used=signals_used,
            fusion_signals=FusionSignals(
                ml_label=ml_label,
                ml_max_probability=round(ml_max_prob, 4) if ml_max_prob is not None else None,
                llm_label=llm_label,
                llm_evidence_support=llm_support,
                evidence_available=True,
                evidence_support=EvidenceSupport.partial,
                real_evidence_count=real_count,
                synthetic_evidence_count=synthetic_count,
                red_flag_count=total_rf,
                high_severity_red_flag_count=high_rf,
                model_agreement=model_agreement,
            ),
        )

    # -------------------------------------------------------------------------
    # Rule 3: LLM Reasoner Unavailable
    # -------------------------------------------------------------------------
    if llm is None:
        signals_used = ["evidence_gate", "llm_unavailable"]
        reason = (
            f"{demo_prefix}Verified evidence was retrieved, but the LLM reasoning component was unavailable; "
            "the system abstained rather than making an unverified determination without reasoning."
        )
        return FinalAssessment(
            verdict=VerdictLabel.unverifiable,
            support_score=0.0,
            reason=reason,
            abstained=True,
            signals_used=signals_used,
            fusion_signals=FusionSignals(
                ml_label=ml_label,
                ml_max_probability=round(ml_max_prob, 4) if ml_max_prob is not None else None,
                llm_label=None,
                llm_evidence_support=None,
                evidence_available=True,
                evidence_support=EvidenceSupport.unavailable,
                real_evidence_count=real_count,
                synthetic_evidence_count=synthetic_count,
                red_flag_count=total_rf,
                high_severity_red_flag_count=high_rf,
                model_agreement="not_applicable",
            ),
        )

    # -------------------------------------------------------------------------
    # Rule 4: LLM Concluded Unverifiable
    # -------------------------------------------------------------------------
    if llm.label == VerdictLabel.unverifiable:
        signals_used = ["evidence_gate", "llm_assessment"]
        reason = (
            f"{demo_prefix}The evidence-bound LLM reasoner assessed the claim as unverifiable "
            f"({llm.short_reason}); system abstained."
        )
        return FinalAssessment(
            verdict=VerdictLabel.unverifiable,
            support_score=0.0,
            reason=reason,
            abstained=True,
            signals_used=signals_used,
            fusion_signals=FusionSignals(
                ml_label=ml_label,
                ml_max_probability=round(ml_max_prob, 4) if ml_max_prob is not None else None,
                llm_label=llm.label,
                llm_evidence_support=llm.evidence_support,
                evidence_available=True,
                evidence_support=llm.evidence_support,
                real_evidence_count=real_count,
                synthetic_evidence_count=synthetic_count,
                red_flag_count=total_rf,
                high_severity_red_flag_count=high_rf,
                model_agreement=model_agreement,
            ),
        )

    # -------------------------------------------------------------------------
    # Rule 5: Supported Evidence + Reliable LLM
    # -------------------------------------------------------------------------
    if (
        llm.label == VerdictLabel.likely_reliable
        and (llm.evidence_support == EvidenceSupport.supported or has_supporting_evidence)
    ):
        verdict = VerdictLabel.likely_reliable
        signals_used = ["evidence_support", "llm_assessment"]

        if ml is not None and ml.label == MLLabel.likely_reliable:
            signals_used.append("ml_baseline")
            reason = (
                f"{demo_prefix}Verified evidence directly supports the claim; the evidence-bound LLM assessment "
                "and classical ML baseline both corroborate the reliable verdict."
            )
        elif ml is not None:
            signals_used.append("ml_disagreement")
            reason = (
                f"{demo_prefix}Verified evidence supports the claim and the evidence-bound LLM assessment agrees; "
                f"the classical model predicted {ml.label.value} based on language patterns, but factual evidence took precedence."
            )
        else:
            reason = (
                f"{demo_prefix}Verified evidence directly supports the claim and the evidence-bound LLM assessment corroborates the reliable verdict."
            )

        # Red-flag context
        if high_rf > 0:
            signals_used.append("red_flags")
            reason += (
                f" Note: {high_rf} high-severity stylometric red flags were detected, "
                "indicating sensational presentation."
            )

        score = _compute_support_score(verdict, False, ml, llm, red_flags)
        return FinalAssessment(
            verdict=verdict,
            support_score=score,
            reason=reason,
            abstained=False,
            signals_used=signals_used,
            fusion_signals=FusionSignals(
                ml_label=ml_label,
                ml_max_probability=round(ml_max_prob, 4) if ml_max_prob is not None else None,
                llm_label=llm.label,
                llm_evidence_support=llm.evidence_support,
                evidence_available=True,
                evidence_support=EvidenceSupport.supported,
                real_evidence_count=real_count,
                synthetic_evidence_count=synthetic_count,
                red_flag_count=total_rf,
                high_severity_red_flag_count=high_rf,
                model_agreement=model_agreement,
            ),
        )

    # -------------------------------------------------------------------------
    # Rule 6: Contradictory Evidence + Fake LLM
    # -------------------------------------------------------------------------
    if (
        llm.label == VerdictLabel.likely_fake
        and (llm.evidence_support == EvidenceSupport.contradicted or has_contradicting_evidence)
    ):
        verdict = VerdictLabel.likely_fake
        signals_used = ["evidence_contradiction", "llm_assessment"]

        if ml is not None and ml.label == MLLabel.likely_fake:
            signals_used.append("ml_baseline")
            reason = (
                f"{demo_prefix}Verified evidence directly contradicts the claim; the evidence-bound LLM assessment "
                "and classical ML baseline both indicate the claim is fabricated or false."
            )
        elif ml is not None:
            signals_used.append("ml_disagreement")
            reason = (
                f"{demo_prefix}Verified evidence directly refutes the claim and the evidence-bound LLM assessment agrees; "
                f"the classical model predicted {ml.label.value} based on language patterns, but was overridden by factual evidence."
            )
        else:
            reason = (
                f"{demo_prefix}Verified evidence directly refutes the claim and the evidence-bound LLM assessment agrees the claim is false."
            )

        if high_rf > 0:
            signals_used.append("red_flags")
            reason += (
                f" Note: {high_rf} high-severity stylometric red flags were detected, "
                "consistent with deceptive presentation."
            )

        score = _compute_support_score(verdict, False, ml, llm, red_flags)
        return FinalAssessment(
            verdict=verdict,
            support_score=score,
            reason=reason,
            abstained=False,
            signals_used=signals_used,
            fusion_signals=FusionSignals(
                ml_label=ml_label,
                ml_max_probability=round(ml_max_prob, 4) if ml_max_prob is not None else None,
                llm_label=llm.label,
                llm_evidence_support=llm.evidence_support,
                evidence_available=True,
                evidence_support=EvidenceSupport.contradicted,
                real_evidence_count=real_count,
                synthetic_evidence_count=synthetic_count,
                red_flag_count=total_rf,
                high_severity_red_flag_count=high_rf,
                model_agreement=model_agreement,
            ),
        )

    # -------------------------------------------------------------------------
    # Rule 7: Partial Evidence + Misleading LLM
    # -------------------------------------------------------------------------
    if (
        llm.label == VerdictLabel.misleading
        and llm.evidence_support == EvidenceSupport.partial
    ):
        verdict = VerdictLabel.misleading
        signals_used = ["evidence_partial", "llm_assessment"]

        if ml is not None and ml.label == MLLabel.misleading:
            signals_used.append("ml_baseline")
            reason = (
                f"{demo_prefix}Verified evidence provides only partial support or reveals significant omitted context "
                f"({llm.short_reason}); both evidence and classical model indicate misleading framing."
            )
        elif ml is not None:
            signals_used.append("ml_disagreement")
            reason = (
                f"{demo_prefix}Verified evidence reveals partial support and significant distortion ({llm.short_reason}); "
                f"the classical model predicted {ml.label.value}, but the evidence-bound assessment takes precedence."
            )
        else:
            reason = (
                f"{demo_prefix}Verified evidence reveals partial support and significant distortion ({llm.short_reason}); "
                "the evidence-bound assessment indicates misleading framing."
            )

        if high_rf > 0:
            signals_used.append("red_flags")
            reason += (
                f" Note: {high_rf} high-severity stylometric red flags were detected."
            )

        score = _compute_support_score(verdict, False, ml, llm, red_flags)
        return FinalAssessment(
            verdict=verdict,
            support_score=score,
            reason=reason,
            abstained=False,
            signals_used=signals_used,
            fusion_signals=FusionSignals(
                ml_label=ml_label,
                ml_max_probability=round(ml_max_prob, 4) if ml_max_prob is not None else None,
                llm_label=llm.label,
                llm_evidence_support=llm.evidence_support,
                evidence_available=True,
                evidence_support=EvidenceSupport.partial,
                real_evidence_count=real_count,
                synthetic_evidence_count=synthetic_count,
                red_flag_count=total_rf,
                high_severity_red_flag_count=high_rf,
                model_agreement=model_agreement,
            ),
        )

    # -------------------------------------------------------------------------
    # Rule 8: Unresolved or Conflicting Signal State (Safe Abstention Fallback)
    # -------------------------------------------------------------------------
    signals_used = ["signal_conflict"]
    reason = (
        f"{demo_prefix}Evidentiary signals and LLM assessment are inconclusive, contradictory, or inconsistent "
        f"(LLM label '{llm.label.value}', support '{llm.evidence_support.value}'); system safely abstained."
    )
    return FinalAssessment(
        verdict=VerdictLabel.unverifiable,
        support_score=0.0,
        reason=reason,
        abstained=True,
        signals_used=signals_used,
        fusion_signals=FusionSignals(
            ml_label=ml_label,
            ml_max_probability=round(ml_max_prob, 4) if ml_max_prob is not None else None,
            llm_label=llm.label,
            llm_evidence_support=llm.evidence_support,
            evidence_available=len(usable_evidence) > 0,
            evidence_support=llm.evidence_support,
            real_evidence_count=real_count,
            synthetic_evidence_count=synthetic_count,
            red_flag_count=total_rf,
            high_severity_red_flag_count=high_rf,
            model_agreement=model_agreement,
        ),
    )


def fuse_input(
    fusion_input: FusionInput,
    allow_synthetic_evidence: bool = False,
    evidence_consensus: Optional[DocumentEvidenceConsensus] = None,
) -> FinalAssessment:
    """Convenience wrapper accepting a single FusionInput container."""
    llm = fusion_input.llm if fusion_input.llm_available else None
    return fuse(
        ml=fusion_input.ml,
        red_flags=fusion_input.red_flags,
        llm=llm,
        evidence=fusion_input.evidence,
        has_checkable_claims=fusion_input.has_checkable_claims,
        allow_synthetic_evidence=allow_synthetic_evidence,
        evidence_consensus=evidence_consensus,
    )

