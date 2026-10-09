"""FastAPI Application Demonstration & Smoke Test Script (Step 17).

Demonstrates the HTTP API endpoints in an isolated, deterministic environment
using FastAPI's TestClient and a dedicated demo database:
data/app/demo_analysis.db

Notice:
- NEVER makes live Gemini calls or burns LLM quota.
- NEVER modifies the production database (data/app/fake_news_analysis.db).
- Clearly labeled as SYNTHETIC CONTROLLED ANALYSIS.
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from backend.app.api.app import create_app
from backend.app.config import DATA_DIR
from backend.app.db.repository import AnalysisRepository
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
    InternalConsistency,
    LLMAssessment,
    MLAssessment,
    MLLabel,
    PipelineTimings,
    RedFlagFeatures,
    RedFlagResult,
    RedFlagSummary,
    SourceAttribution,
    VerdictLabel,
    adapter_from_red_flags,
)
from backend.app.services.analysis_service import AnalysisService
from backend.app.services.red_flags import analyze_red_flags


class DemoFakeHybridEngine:
    """Deterministic test double for HybridAnalysisEngine ensuring zero live API calls."""

    def analyze_text(self, text: str) -> AnalysisResponse:
        claim = Claim(id=1, text=text[:200], checkable=True)
        ev = EvidenceItem(
            id="EV-DEMO-1",
            claim_summary="Corporate announcement corroboration.",
            evidence_text="Verified corporate press wire confirms new manufacturing operations in Texas.",
            source_name="Official Wire Service",
            stance=EvidenceStance.supports,
            is_synthetic=True,
        )
        ml = MLAssessment(
            label=MLLabel.likely_reliable,
            probabilities=ClassProbabilities(likely_reliable=0.88, likely_fake=0.06, misleading=0.06),
            model_name="Tuned TF-IDF + Logistic Regression",
        )
        red_flags = adapter_from_red_flags(analyze_red_flags(text))
        llm = LLMAssessment(
            label=VerdictLabel.likely_reliable,
            source_attribution=SourceAttribution.strong,
            evidence_support=EvidenceSupport.supported,
            internal_consistency=InternalConsistency.consistent,
            short_reason="Factual claim corroborated by corporate announcement records.",
        )
        statuses = [
            ComponentStatus(component="cleaning", status=ComponentExecutionStatus.success),
            ComponentStatus(component="claim_extraction", status=ComponentExecutionStatus.success),
            ComponentStatus(component="classical_ml", status=ComponentExecutionStatus.success),
            ComponentStatus(component="red_flags", status=ComponentExecutionStatus.success),
            ComponentStatus(component="evidence_retrieval", status=ComponentExecutionStatus.success),
            ComponentStatus(component="llm_reasoner", status=ComponentExecutionStatus.success),
            ComponentStatus(component="fusion", status=ComponentExecutionStatus.success),
        ]
        timings = PipelineTimings(
            cleaning_ms=0.1,
            claim_extraction_ms=1.2,
            ml_ms=1.5,
            red_flags_ms=0.2,
            retrieval_ms=0.5,
            llm_ms=2.1,
            fusion_ms=0.3,
            total_ms=5.9,
        )
        return AnalysisResponse(
            analysis_id=str(uuid.uuid4()),
            input_text=text,
            verdict=VerdictLabel.likely_reliable,
            support_score=0.88,
            claims=[ClaimResult(claim=claim, assessment=ClaimAssessment.supported, evidence=[ev])],
            red_flags=red_flags,
            ml=ml,
            llm=llm,
            evidence=[ev],
            reason="Verified press release and corporate filings confirm manufacturing facility construction.",
            component_statuses=statuses,
            timings=timings,
            created_at=datetime.now(timezone.utc),
        )


def main() -> None:
    demo_db_path = DATA_DIR / "app" / "demo_analysis.db"
    print("=" * 80)
    print("API DEMO / SYNTHETIC CONTROLLED ANALYSIS (Step 17)")
    print("Notice: Isolated demonstration running on: data/app/demo_analysis.db")
    print("Production database (data/app/fake_news_analysis.db) is NOT modified.")
    print("Zero live Gemini or external network calls are performed.")
    print("=" * 80)

    # Prepare demo repository and service
    repo = AnalysisRepository(db_path=demo_db_path)
    engine = DemoFakeHybridEngine()
    service = AnalysisService(engine=engine, repository=repo)

    # Initialize app with demo service injection
    app = create_app(analysis_service=service, repository=repo, init_db=True)

    with TestClient(app) as client:
        # 1. Health check
        print("\n[Step 1] GET /healthz")
        res = client.get("/healthz")
        print(f"  HTTP Status : {res.status_code}")
        print(f"  Payload     : {res.json()}")
        assert res.status_code == 200

        # 2. Analyze a claim
        print("\n[Step 2] POST /api/v1/analyze")
        claim_payload = {"text": "Acme Corp announced a new factory in Texas."}
        res = client.post("/api/v1/analyze", json=claim_payload)
        print(f"  HTTP Status : {res.status_code}")
        data = res.json()
        analysis_id = data["analysis_id"]
        analysis = data["analysis"]
        print(f"  Analysis ID : {analysis_id}")
        print(f"  Verdict     : {analysis['verdict'].upper()}")
        print(f"  Score       : {analysis['support_score']}")
        print(f"  Reason      : {analysis['reason']}")
        assert res.status_code == 200
        assert analysis_id is not None

        # 3. List analyses
        print("\n[Step 3] GET /api/v1/analyses (history)")
        res = client.get("/api/v1/analyses?limit=10&offset=0")
        print(f"  HTTP Status : {res.status_code}")
        history_data = res.json()
        print(f"  Total items : {history_data['total']}")
        for item in history_data["items"]:
            print(f"  - [{item['analysis_id'][:8]}...] {item['verdict'].upper():16} '{item['input_preview']}'")
        assert res.status_code == 200

        # 4. Retrieve single analysis by ID
        print(f"\n[Step 4] GET /api/v1/analyses/{analysis_id} (detail)")
        res = client.get(f"/api/v1/analyses/{analysis_id}")
        print(f"  HTTP Status : {res.status_code}")
        detail = res.json()
        print(f"  Version     : {detail['pipeline_version']}")
        print(f"  Claims      : {len(detail['analysis']['claims'])} claim(s)")
        print(f"  Evidence    : {len(detail['analysis']['evidence'])} record(s)")
        assert res.status_code == 200

        # 5. Submit feedback
        print(f"\n[Step 5] POST /api/v1/analyses/{analysis_id}/feedback")
        fb_payload = {
            "analysis_id": analysis_id,
            "rating": "helpful",
            "comment": "Accurate fact-checking and clear citations.",
        }
        res = client.post(f"/api/v1/analyses/{analysis_id}/feedback", json=fb_payload)
        print(f"  HTTP Status : {res.status_code}")
        fb_record = res.json()
        print(f"  Feedback ID : {fb_record['feedback_id']}")
        print(f"  Rating      : {fb_record['rating']}")
        assert res.status_code == 201

        # 6. Retrieve feedback
        print(f"\n[Step 6] GET /api/v1/analyses/{analysis_id}/feedback")
        res = client.get(f"/api/v1/analyses/{analysis_id}/feedback")
        print(f"  HTTP Status : {res.status_code}")
        fb_items = res.json()
        print(f"  Total feedback items: {len(fb_items)}")
        assert len(fb_items) == 1

        # 7. Delete analysis
        print(f"\n[Step 7] DELETE /api/v1/analyses/{analysis_id}")
        res = client.delete(f"/api/v1/analyses/{analysis_id}")
        print(f"  HTTP Status : {res.status_code}")
        del_data = res.json()
        print(f"  Deleted     : {del_data['deleted']}")
        assert res.status_code == 200

        # 8. Verify 404 after deletion
        res = client.get(f"/api/v1/analyses/{analysis_id}")
        print(f"  Verify 404  : HTTP {res.status_code}")
        assert res.status_code == 404

    print("\n" + "=" * 80)
    print("API DEMO COMPLETED SUCCESSFULLY WITH 100% PASSING CHECKS.")
    print("=" * 80)


if __name__ == "__main__":
    main()
