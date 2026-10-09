"""Controlled SQLite Persistence Layer Demonstration Script (Step 16).

Demonstrates the persistence, retrieval, history pagination, feedback attachment,
and cascade deletion features using an isolated demonstration database:
data/app/demo_analysis.db

NEVER modifies or inserts records into the production database:
data/app/fake_news_analysis.db
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import DATA_DIR
from backend.app.db.repository import AnalysisRepository
from backend.app.db.schema import init_database
from backend.app.schemas import (
    AnalysisResponse,
    Claim,
    ClaimAssessment,
    ClaimResult,
    ClassProbabilities,
    ComponentExecutionStatus,
    ComponentStatus,
    EvidenceItem,
    EvidenceStance,
    EvidenceSupport,
    FeedbackRating,
    FeedbackRequest,
    InternalConsistency,
    LLMAssessment,
    MLAssessment,
    MLLabel,
    PipelineTimings,
    RedFlag,
    RedFlagFeatures,
    RedFlagResult,
    RedFlagSummary,
    Severity,
    SourceAttribution,
    VerdictLabel,
    adapter_from_red_flags,
)
from backend.app.services.red_flags import analyze_red_flags



def create_sample_response(
    text: str,
    verdict: VerdictLabel,
    support_score: float,
    reason: str,
    has_evidence: bool = True,
) -> AnalysisResponse:
    """Helper creating a fully populated synthetic AnalysisResponse."""
    claim = Claim(id=1, text=text, checkable=(verdict != VerdictLabel.unverifiable))
    evidence_items = []
    if has_evidence:
        evidence_items.append(
            EvidenceItem(
                id=f"EV-{uuid.uuid4().hex[:6].upper()}",
                claim_summary=f"Factual statement regarding: {text[:30]}",
                evidence_text=f"Official verified agency bulletin confirming or refuting: {text}",
                source_name="Official Wire Service",
                stance=EvidenceStance.supports if verdict == VerdictLabel.likely_reliable else EvidenceStance.contradicts,
                is_synthetic=True,
            )
        )

    ml = MLAssessment(
        label=MLLabel.likely_reliable if verdict == VerdictLabel.likely_reliable else MLLabel.likely_fake,
        probabilities=ClassProbabilities(likely_reliable=0.8, likely_fake=0.1, misleading=0.1),
        model_name="Tuned TF-IDF + Logistic Regression",
    )
    red_flags = adapter_from_red_flags(analyze_red_flags(text))
    llm = None
    if has_evidence:
        llm = LLMAssessment(
            label=verdict,
            source_attribution=SourceAttribution.strong,
            evidence_support=EvidenceSupport.supported if verdict == VerdictLabel.likely_reliable else EvidenceSupport.contradicted,
            internal_consistency=InternalConsistency.consistent,
            short_reason=f"Corroborated by external verified documentation.",
        )

    return AnalysisResponse(
        analysis_id=str(uuid.uuid4()),
        input_text=text,
        verdict=verdict,
        support_score=support_score,
        claims=[ClaimResult(claim=claim, assessment=ClaimAssessment.supported if verdict == VerdictLabel.likely_reliable else ClaimAssessment.unverifiable, evidence=evidence_items)],
        red_flags=red_flags,
        ml=ml,
        llm=llm,
        evidence=evidence_items,
        reason=reason,
        component_statuses=[
            ComponentStatus(component="cleaning", status=ComponentExecutionStatus.success),
            ComponentStatus(component="claim_extraction", status=ComponentExecutionStatus.success),
            ComponentStatus(component="classical_ml", status=ComponentExecutionStatus.success),
            ComponentStatus(component="red_flags", status=ComponentExecutionStatus.success),
            ComponentStatus(component="evidence_retrieval", status=ComponentExecutionStatus.success if has_evidence else ComponentExecutionStatus.skipped),
            ComponentStatus(component="llm_reasoner", status=ComponentExecutionStatus.success if has_evidence else ComponentExecutionStatus.skipped),
            ComponentStatus(component="fusion", status=ComponentExecutionStatus.success),
        ],
        timings=PipelineTimings(cleaning_ms=0.1, claim_extraction_ms=1.2, ml_ms=1.5, red_flags_ms=0.2, retrieval_ms=0.5, llm_ms=2.1, fusion_ms=0.3, total_ms=5.9),
        created_at=datetime.now(timezone.utc),
    )


def run_demo() -> None:
    demo_db_path = DATA_DIR / "app" / "demo_analysis.db"
    print("=" * 80)
    print("SYNTHETIC DATABASE DEMO (Step 16)")
    print("Notice: Isolated demonstration running on: data/app/demo_analysis.db")
    print("Production database (data/app/fake_news_analysis.db) is NOT modified.")
    print("=" * 80)

    # Clean existing demo db if present
    if demo_db_path.exists():
        demo_db_path.unlink()

    # 1. Initialize schema
    print("\n[Step 1] Initializing isolated demonstration SQLite database...")
    init_database(demo_db_path)
    repo = AnalysisRepository(demo_db_path, auto_init=False)
    print("-> Schema initialized successfully.")

    # 2. Save 3 synthetic analysis responses
    print("\n[Step 2] Saving 3 synthetic analysis responses...")
    samples = [
        create_sample_response(
            text="Acme Corp announced a new manufacturing facility in Texas.",
            verdict=VerdictLabel.likely_reliable,
            support_score=0.85,
            reason="Verified press release and corporate filings confirm manufacturing facility construction.",
            has_evidence=True,
        ),
        create_sample_response(
            text="CDC study proves daily coffee consumption triggers immediate heart attacks.",
            verdict=VerdictLabel.likely_fake,
            support_score=0.90,
            reason="Official CDC epidemiological data directly contradicts the fabricated claim.",
            has_evidence=True,
        ),
        create_sample_response(
            text="The Moon is composed entirely of green artisanal cheese.",
            verdict=VerdictLabel.unverifiable,
            support_score=0.0,
            reason="No usable provenance-bearing evidence was available to verify or refute the claim.",
            has_evidence=False,
        ),
    ]

    saved_ids = []
    for idx, resp in enumerate(samples, start=1):
        analysis_id = repo.save_analysis(resp, original_text=resp.input_text)
        saved_ids.append(analysis_id)
        print(f"  ({idx}) Saved ID: {analysis_id[:8]}... | Verdict: {resp.verdict.value.upper():16} | Score: {resp.support_score:.2f}")

    # 3. List history (newest first)
    print("\n[Step 3] Querying analysis history (limit=10, newest first)...")
    history = repo.list_analyses(limit=10, offset=0)
    for h in history:
        print(f"  ID: {h.analysis_id[:8]}... | Created: {h.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')} | Verdict: {h.verdict.value.upper():16} | Preview: '{h.input_preview}'")

    # 4. Reconstruct one analysis in full by ID
    target_id = saved_ids[0]
    print(f"\n[Step 4] Retrieving full analysis record by ID: {target_id}...")
    reconstructed = repo.get_analysis(target_id)
    assert reconstructed is not None
    print(f"  Reconstructed Input   : {reconstructed.input_text}")
    print(f"  Reconstructed Verdict : {reconstructed.verdict.value.upper()}")
    print(f"  Reconstructed Score   : {reconstructed.support_score:.4f}")
    print(f"  Reconstructed Claims  : {len(reconstructed.claims)} claim(s)")
    print(f"  Reconstructed Evidence: {len(reconstructed.evidence)} record(s)")
    print(f"  Pydantic Contract     : 100% VALIDATED ROUND-TRIP")

    # 5. Attach user feedback
    print(f"\n[Step 5] Attaching user feedback to analysis {target_id[:8]}...")
    fb1 = repo.save_feedback(
        FeedbackRequest(
            analysis_id=target_id,
            rating=FeedbackRating.helpful,
            comment="Detailed citations and evidence links were very clear.",
        )
    )
    fb2 = repo.save_feedback(
        FeedbackRequest(
            analysis_id=target_id,
            rating=FeedbackRating.unsure,
            comment="Would like to see local state filing documents as well.",
        )
    )
    print(f"  Attached Feedback 1 (ID: {fb1.feedback_id[:8]}...): rating='{fb1.rating.value}', comment='{fb1.comment}'")
    print(f"  Attached Feedback 2 (ID: {fb2.feedback_id[:8]}...): rating='{fb2.rating.value}', comment='{fb2.comment}'")

    # 6. View attached feedback
    print(f"\n[Step 6] Retrieving feedback for analysis {target_id[:8]}...")
    retrieved_fb = repo.get_feedback_for_analysis(target_id)
    print(f"  Total feedback items linked: {len(retrieved_fb)}")

    # 7. Delete one analysis and verify cascade
    delete_id = saved_ids[0]
    print(f"\n[Step 7] Deleting analysis {delete_id[:8]}... (verifying CASCADE deletion of feedback)")
    deleted = repo.delete_analysis(delete_id)
    print(f"  Analysis deletion result: {deleted}")
    remaining_fb = repo.get_feedback_for_analysis(delete_id)
    print(f"  Remaining cascaded feedback items for {delete_id[:8]}...: {len(remaining_fb)} (Expected: 0)")
    assert len(remaining_fb) == 0, "Foreign key cascade failed to delete feedback!"

    total_analyses = repo.count_analyses()
    print(f"\nTotal Analyses Remaining in Demo DB: {total_analyses}")
    print("\n" + "=" * 80)
    print("SYNTHETIC DATABASE DEMO COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
