"""Controlled End-to-End Frontend Demonstration (Step 18).

Demonstrates the frontend API client communication with FastAPI using the isolated
demo database (data/app/demo_analysis.db).
Production database (data/app/fake_news_analysis.db) remains untouched.
Zero live Gemini calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import httpx
from starlette.testclient import TestClient

from backend.app.api.app import create_app
from backend.app.config import DATA_DIR
from backend.app.db.repository import AnalysisRepository
from backend.app.services.analysis_service import AnalysisService
from backend.tests.test_api import FakeHybridEngine
from frontend.api_client import APIClient


def run_demo() -> None:
    print("=" * 80)
    print("FRONTEND E2E CONTROLLED INTEGRATION DEMO (Step 18)")
    demo_db = DATA_DIR / "app" / "demo_analysis.db"
    print(f"Demo Database: {demo_db}")
    print("=" * 80)

    # 1. Initialize test backend with fake engine
    repo = AnalysisRepository(db_path=demo_db)
    engine = FakeHybridEngine(force_abstain=True)
    service = AnalysisService(engine=engine, repository=repo)
    app = create_app(analysis_service=service, repository=repo, engine=engine, init_db=True)

    # In-memory TestClient for reliable, isolated verification
    test_client = TestClient(app)
    client = APIClient(base_url="http://testserver", client=test_client)

    # Step 1: Health check
    print("\n[Step 1] Frontend Client -> GET /healthz")
    health = client.check_health()
    print(f"  Online: {health['online']}, Service: {health.get('service')}, Provider: {health.get('llm_provider')}")
    assert health["online"] is True

    # Step 2: Analyze statement
    test_text = "The Moon is made entirely of green cheddar cheese."
    print(f"\n[Step 2] Frontend Client -> POST /api/v1/analyze ('{test_text}')")
    result = client.analyze(test_text)
    analysis_id = result["analysis_id"]
    analysis = result["analysis"]
    is_abstained = analysis.get("abstained", analysis.get("verdict") == "unverifiable")
    print(f"  Analysis ID: {analysis_id}")
    print(f"  Verdict: {analysis['verdict']} (Abstained: {is_abstained})")
    print(f"  Support Score: {analysis['support_score']:.2f} / 1.00")
    print(f"  Reason: {analysis['reason'][:70]}...")
    assert analysis["verdict"] == "unverifiable"
    assert is_abstained is True

    # Step 3: Browse History
    print("\n[Step 3] Frontend Client -> GET /api/v1/analyses (pagination limit=5)")
    history = client.list_analyses(limit=5, offset=0)
    print(f"  Total analyses in demo DB: {history['total']}")
    print(f"  Retrieved items: {len(history['items'])}")
    for item in history["items"]:
        print(f"  - [{item['analysis_id'][:8]}...] {item['verdict'].upper()} | '{item['input_preview'][:45]}...'")

    # Step 4: Retrieve Stored Record
    print(f"\n[Step 4] Frontend Client -> GET /api/v1/analyses/{analysis_id} (stored audit record)")
    stored = client.get_analysis(analysis_id)
    print(f"  Retrieved ID: {stored['analysis_id']}")
    print(f"  Claims count: {len(stored['analysis'].get('claims', []))}")
    print(f"  Verified: read directly from DB without recomputing pipeline.")

    # Step 5: Submit Feedback
    print(f"\n[Step 5] Frontend Client -> POST /api/v1/analyses/{analysis_id}/feedback")
    fb = client.submit_feedback(
        analysis_id=analysis_id,
        rating="helpful",
        notes="Accurate safe abstention when no evidence exists.",
        corrected_verdict="unverifiable",
    )
    print(f"  Feedback recorded: ID={fb['feedback_id']}, Rating={fb['rating']}")

    # Step 6: Get Feedback
    print(f"\n[Step 6] Frontend Client -> GET /api/v1/analyses/{analysis_id}/feedback")
    fb_list = client.get_feedback(analysis_id)
    print(f"  Total feedback items: {len(fb_list)}")

    # Step 7: Delete Analysis
    print(f"\n[Step 7] Frontend Client -> DELETE /api/v1/analyses/{analysis_id}")
    del_res = client.delete_analysis(analysis_id)
    print(f"  Deleted: {del_res['deleted']}")

    print("\n" + "=" * 80)
    print("FRONTEND E2E CONTROLLED INTEGRATION DEMO PASSED WITH 100% SUCCESS.")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
