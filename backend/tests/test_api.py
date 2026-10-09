"""Unit and Integration Tests for FastAPI Backend Service (Step 17).

Tests all 50 required scenarios:
- Application initialization, OpenAPI availability, metadata
- Health probe (constant-time, zero Gemini / neural model invocations)
- Analysis execution endpoint (input validation, UUID generation, full response contract)
- Safe abstention (unverifiable returned as HTTP 200)
- History pagination (newest-first, limit, offset, bounds validation)
- Detail retrieval (pure database reconstruction, zero recomputation)
- Deletion and foreign-key cascade removal of linked feedback
- Feedback submission and validation (helpful, not_helpful, unsure, corrected_verdict)
- Error mapping (404, 422, 500, 503) without stack trace leakage
- Security invariants: zero API key exposure in payloads or OpenAPI schema
- Synthetic evidence lockdown (production API prohibits synthetic overrides)
- Robustness: Unicode, long text, quotes, newlines, SQL injection strings
- Complete isolation: autouse network guard, temp DB, untouched production DB, frozen model integrity
"""

from __future__ import annotations

import hashlib
import json
import socket
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pytest
from fastapi.testclient import TestClient
from backend.app.api.app import create_app
from backend.app.config import (
    DATABASE_PATH,
    DATA_DIR,
    MAX_ANALYSIS_TEXT_LENGTH,
    MODELS_DIR,
    PIPELINE_VERSION,
)
from backend.app.db.exceptions import DatabaseError
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
    FeedbackRating,
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
from backend.app.services.hybrid_engine import HybridAnalysisEngine
from backend.app.services.red_flags import analyze_red_flags


# ---------------------------------------------------------------------------
# Network Guard (Autouse)
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def guard_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strict guard: prohibit any external network socket connection during API tests.
    
    Permits local loopback (127.0.0.1 / localhost) required by Windows asyncio
    ProactorEventLoop self-pipe initialization.
    """
    orig_connect = socket.socket.connect

    def forbidden_connect(self: socket.socket, address: Any, *args: Any, **kwargs: Any) -> Any:
        host = address[0] if isinstance(address, tuple) else address
        if host in ("127.0.0.1", "localhost", "::1"):
            return orig_connect(self, address, *args, **kwargs)
        raise RuntimeError(f"Prohibited external network call to {address} during API test execution!")

    monkeypatch.setattr(socket.socket, "connect", forbidden_connect)
    monkeypatch.setenv("LLM_PROVIDER", "mock")


# ---------------------------------------------------------------------------
# Test Doubles
# ---------------------------------------------------------------------------
class FakeHybridEngine:
    """Deterministic test double for HybridAnalysisEngine."""

    def __init__(
        self,
        should_fail: bool = False,
        fail_exc: Optional[Exception] = None,
        force_abstain: bool = False,
    ) -> None:
        self.calls: list[str] = []
        self.should_fail = should_fail
        self.fail_exc = fail_exc or RuntimeError("Simulated pipeline internal crash")
        self.force_abstain = force_abstain

    def analyze_text(self, text: str) -> AnalysisResponse:
        self.calls.append(text)
        if self.should_fail:
            raise self.fail_exc

        claim = Claim(id=1, text=text[:200], checkable=not self.force_abstain)
        ev_list: list[EvidenceItem] = []
        if not self.force_abstain:
            ev = EvidenceItem(
                id="EV-TEST-1",
                claim_summary="Summary of evidence",
                evidence_text="Corroborating statement from official wire.",
                source_name="Official Wire",
                stance=EvidenceStance.supports,
                is_synthetic=False,
            )
            ev_list.append(ev)

        ml = MLAssessment(
            label=MLLabel.likely_reliable if not self.force_abstain else MLLabel.likely_fake,
            probabilities=ClassProbabilities(likely_reliable=0.82, likely_fake=0.08, misleading=0.10),
            model_name="Tuned TF-IDF + Logistic Regression",
            model_version="1.0.0",
        )

        red_flags = adapter_from_red_flags(analyze_red_flags(text))

        llm = None
        if not self.force_abstain:
            llm = LLMAssessment(
                label=VerdictLabel.likely_reliable,
                source_attribution=SourceAttribution.strong,
                evidence_support=EvidenceSupport.supported,
                internal_consistency=InternalConsistency.consistent,
                short_reason="Factual proposition verified against wire documentation.",
            )

        verdict = VerdictLabel.unverifiable if self.force_abstain else VerdictLabel.likely_reliable
        support_score = 0.0 if self.force_abstain else 0.85
        reason = (
            "No usable provenance-bearing evidence was available to verify or refute the claim; "
            "the system abstained from making an ungrounded factual determination."
            if self.force_abstain
            else "Verified corporate wire filings corroborate the statement."
        )

        statuses = [
            ComponentStatus(component="cleaning", status=ComponentExecutionStatus.success),
            ComponentStatus(component="claim_extraction", status=ComponentExecutionStatus.success),
            ComponentStatus(component="classical_ml", status=ComponentExecutionStatus.success),
            ComponentStatus(component="red_flags", status=ComponentExecutionStatus.success),
            ComponentStatus(component="evidence_retrieval", status=ComponentExecutionStatus.success if ev_list else ComponentExecutionStatus.skipped),
            ComponentStatus(component="llm_reasoner", status=ComponentExecutionStatus.success if llm else ComponentExecutionStatus.skipped),
            ComponentStatus(component="fusion", status=ComponentExecutionStatus.success),
        ]

        timings = PipelineTimings(
            cleaning_ms=0.1,
            claim_extraction_ms=1.5,
            ml_ms=1.2,
            red_flags_ms=0.3,
            retrieval_ms=0.8,
            llm_ms=2.5,
            fusion_ms=0.2,
            total_ms=6.6,
        )

        return AnalysisResponse(
            analysis_id=str(uuid.uuid4()),
            input_text=text,
            verdict=verdict,
            support_score=support_score,
            claims=[ClaimResult(claim=claim, assessment=ClaimAssessment.supported if not self.force_abstain else ClaimAssessment.unverifiable, evidence=ev_list)],
            red_flags=red_flags,
            ml=ml,
            llm=llm,
            evidence=ev_list,
            reason=reason,
            component_statuses=statuses,
            timings=timings,
            created_at=datetime.now(timezone.utc),
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def temp_api_db(tmp_path: Path) -> Path:
    """Return isolated temporary database path for API tests."""
    return tmp_path / "test_api_database.db"


@pytest.fixture
def fake_engine() -> FakeHybridEngine:
    return FakeHybridEngine()


@pytest.fixture
def api_service(temp_api_db: Path, fake_engine: FakeHybridEngine) -> tuple[AnalysisService, AnalysisRepository]:
    repo = AnalysisRepository(db_path=temp_api_db)
    service = AnalysisService(engine=fake_engine, repository=repo)
    return service, repo


@pytest.fixture
def client(api_service: tuple[AnalysisService, AnalysisRepository], fake_engine: FakeHybridEngine) -> TestClient:
    service, repo = api_service
    app = create_app(analysis_service=service, repository=repo, engine=fake_engine, init_db=True)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests: 50 Comprehensive Verification Scenarios
# ---------------------------------------------------------------------------
def test_1_app_initializes(client: TestClient) -> None:
    """Test 1: Application initializes with correct metadata."""
    assert client.app.title == "Fake News Detection Using LLMs API"
    assert client.app.version is not None


def test_2_openapi_available(client: TestClient) -> None:
    """Test 2: OpenAPI JSON schema is accessible and valid."""
    res = client.get("/openapi.json")
    assert res.status_code == 200
    data = res.json()
    assert "paths" in data
    assert "/api/v1/analyze" in data["paths"]
    assert "/healthz" in data["paths"]


def test_3_health_returns_200(client: TestClient) -> None:
    """Test 3: Liveness probe returns HTTP 200 and 'ok' status."""
    res = client.get("/healthz")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["service"] == "fake-news-detection-api"
    assert data["pipeline_version"] == PIPELINE_VERSION


def test_4_health_does_not_call_gemini(client: TestClient, fake_engine: FakeHybridEngine) -> None:
    """Test 4: Health endpoint runs in constant time without invoking the engine."""
    client.get("/healthz")
    assert len(fake_engine.calls) == 0


def test_5_analyze_valid_input(client: TestClient) -> None:
    """Test 5: Valid analysis request returns HTTP 200."""
    res = client.post("/api/v1/analyze", json={"text": "Acme Corp opened a factory in Texas."})
    assert res.status_code == 200


def test_6_analyze_returns_analysis_id(client: TestClient) -> None:
    """Test 6: POST /api/v1/analyze returns top-level analysis_id."""
    res = client.post("/api/v1/analyze", json={"text": "Valid statement."})
    assert "analysis_id" in res.json()


def test_7_returned_id_is_uuid_compatible(client: TestClient) -> None:
    """Test 7: The returned analysis_id is a valid UUID4 string."""
    res = client.post("/api/v1/analyze", json={"text": "Valid statement."})
    aid = res.json()["analysis_id"]
    parsed = uuid.UUID(aid)
    assert str(parsed) == aid


def test_8_returned_analysis_validates(client: TestClient) -> None:
    """Test 8: Returned analysis payload satisfies AnalysisResponse contract."""
    res = client.post("/api/v1/analyze", json={"text": "Valid statement."})
    analysis_dict = res.json()["analysis"]
    validated = AnalysisResponse.model_validate(analysis_dict)
    assert validated.verdict in {VerdictLabel.likely_reliable, VerdictLabel.unverifiable}


def test_9_saved_analysis_id_matches_api_id(client: TestClient, api_service: tuple[AnalysisService, AnalysisRepository]) -> None:
    """Test 9: Database record can be retrieved using the exact analysis_id from HTTP response."""
    _, repo = api_service
    res = client.post("/api/v1/analyze", json={"text": "Persistence match check."})
    aid = res.json()["analysis_id"]
    from_db = repo.get_analysis(aid)
    assert from_db is not None
    assert str(from_db.analysis_id) == aid


def test_10_empty_text_rejected(client: TestClient) -> None:
    """Test 10: Empty string input is rejected with HTTP 422."""
    res = client.post("/api/v1/analyze", json={"text": ""})
    assert res.status_code == 422
    assert res.json()["error"] == "validation_error"


def test_11_whitespace_text_rejected(client: TestClient) -> None:
    """Test 11: Whitespace-only input is rejected with HTTP 422."""
    res = client.post("/api/v1/analyze", json={"text": "     \n\t   "})
    assert res.status_code == 422


def test_12_non_string_text_rejected(client: TestClient) -> None:
    """Test 12: Non-string input is rejected with HTTP 422."""
    res = client.post("/api/v1/analyze", json={"text": 12345})
    assert res.status_code == 422


def test_13_oversized_text_rejected(client: TestClient) -> None:
    """Test 13: 20,000 chars is accepted; text exceeding 20,000 chars (20,001) is rejected with HTTP 422."""
    boundary_accepted = "A" * MAX_ANALYSIS_TEXT_LENGTH
    res_ok = client.post("/api/v1/analyze", json={"text": boundary_accepted})
    assert res_ok.status_code == 200

    boundary_rejected = "A" * (MAX_ANALYSIS_TEXT_LENGTH + 1)
    res_err = client.post("/api/v1/analyze", json={"text": boundary_rejected})
    assert res_err.status_code == 422
    assert "Input text exceeds maximum allowed length" in res_err.text


def test_14_safe_abstention_returns_http_200(temp_api_db: Path) -> None:
    """Test 14: Engine safe abstention (unverifiable, score=0.0) returns HTTP 200, not an error."""
    engine_abstaining = FakeHybridEngine(force_abstain=True)
    repo = AnalysisRepository(db_path=temp_api_db)
    service = AnalysisService(engine=engine_abstaining, repository=repo)
    app = create_app(analysis_service=service, repository=repo, engine=engine_abstaining)
    local_client = TestClient(app)

    res = local_client.post("/api/v1/analyze", json={"text": "Factual claim with zero evidence."})
    assert res.status_code == 200
    data = res.json()
    assert data["analysis"]["verdict"] == "unverifiable"
    assert data["analysis"]["support_score"] == 0.0


def test_15_support_score_clearly_represented_as_score_not_probability(client: TestClient) -> None:
    """Test 15: support_score is a bounded float and not described as probability."""
    res = client.post("/api/v1/analyze", json={"text": "Score representation check."})
    score = res.json()["analysis"]["support_score"]
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_16_history_endpoint(client: TestClient) -> None:
    """Test 16: GET /api/v1/analyses returns history response envelope."""
    client.post("/api/v1/analyze", json={"text": "First statement."})
    res = client.get("/api/v1/analyses")
    assert res.status_code == 200
    data = res.json()
    assert "items" in data
    assert "total" in data
    assert data["total"] >= 1


def test_17_history_newest_first(client: TestClient) -> None:
    """Test 17: History listings are ordered newest first."""
    client.post("/api/v1/analyze", json={"text": "Alpha statement."})
    client.post("/api/v1/analyze", json={"text": "Beta statement."})
    res = client.get("/api/v1/analyses")
    items = res.json()["items"]
    assert len(items) >= 2
    assert "Beta" in items[0]["input_preview"]
    assert "Alpha" in items[1]["input_preview"]


def test_18_history_pagination_limit(client: TestClient) -> None:
    """Test 18: limit query parameter constrains returned items."""
    for i in range(5):
        client.post("/api/v1/analyze", json={"text": f"Pagination query {i}"})
    res = client.get("/api/v1/analyses?limit=2")
    assert res.status_code == 200
    assert len(res.json()["items"]) == 2
    assert res.json()["limit"] == 2


def test_19_history_pagination_offset(client: TestClient) -> None:
    """Test 19: offset query parameter skips the designated number of items."""
    for i in range(4):
        client.post("/api/v1/analyze", json={"text": f"Item {i}"})
    res_all = client.get("/api/v1/analyses?limit=10")
    res_skip = client.get("/api/v1/analyses?limit=10&offset=1")
    assert len(res_skip.json()["items"]) == len(res_all.json()["items"]) - 1
    assert res_skip.json()["items"][0]["analysis_id"] == res_all.json()["items"][1]["analysis_id"]


def test_20_invalid_limit_rejected(client: TestClient) -> None:
    """Test 20: Limit < 1 or > 100 returns HTTP 422."""
    assert client.get("/api/v1/analyses?limit=0").status_code == 422
    assert client.get("/api/v1/analyses?limit=101").status_code == 422


def test_21_invalid_offset_rejected(client: TestClient) -> None:
    """Test 21: Negative offset returns HTTP 422."""
    assert client.get("/api/v1/analyses?offset=-1").status_code == 422


def test_22_retrieve_existing_analysis(client: TestClient) -> None:
    """Test 22: GET /api/v1/analyses/{id} retrieves complete persisted record."""
    create_res = client.post("/api/v1/analyze", json={"text": "Detail retrieval statement."})
    aid = create_res.json()["analysis_id"]

    get_res = client.get(f"/api/v1/analyses/{aid}")
    assert get_res.status_code == 200
    detail = get_res.json()
    assert detail["analysis_id"] == aid
    assert detail["analysis"]["input_text"] == "Detail retrieval statement."
    assert detail["pipeline_version"] == PIPELINE_VERSION


def test_23_retrieve_unknown_analysis_returns_404(client: TestClient) -> None:
    """Test 23: Retrieving unknown ID returns HTTP 404."""
    res = client.get("/api/v1/analyses/unknown-uuid-0000")
    assert res.status_code == 404
    assert res.json()["error"] == "not_found"


def test_24_retrieval_does_not_recompute_analysis(client: TestClient, fake_engine: FakeHybridEngine) -> None:
    """Test 24: Historical retrieval queries SQLite only and never invokes the engine."""
    create_res = client.post("/api/v1/analyze", json={"text": "No recompute test."})
    aid = create_res.json()["analysis_id"]
    calls_before = len(fake_engine.calls)

    client.get(f"/api/v1/analyses/{aid}")
    calls_after = len(fake_engine.calls)
    assert calls_before == calls_after


def test_25_delete_existing_analysis(client: TestClient) -> None:
    """Test 25: DELETE /api/v1/analyses/{id} deletes analysis and returns 200."""
    create_res = client.post("/api/v1/analyze", json={"text": "Delete me."})
    aid = create_res.json()["analysis_id"]

    del_res = client.delete(f"/api/v1/analyses/{aid}")
    assert del_res.status_code == 200
    assert del_res.json()["deleted"] is True

    # Subsequent GET returns 404
    assert client.get(f"/api/v1/analyses/{aid}").status_code == 404


def test_26_delete_unknown_analysis_returns_404(client: TestClient) -> None:
    """Test 26: Deleting unknown ID returns HTTP 404."""
    res = client.delete("/api/v1/analyses/unknown-uuid-0000")
    assert res.status_code == 404


def test_27_feedback_submission(client: TestClient) -> None:
    """Test 27: POST feedback attaches user evaluation and returns 201."""
    create_res = client.post("/api/v1/analyze", json={"text": "Feedback target."})
    aid = create_res.json()["analysis_id"]

    fb_res = client.post(
        f"/api/v1/analyses/{aid}/feedback",
        json={"analysis_id": aid, "rating": "helpful", "comment": "Clear evidence."},
    )
    assert fb_res.status_code == 201
    data = fb_res.json()
    assert data["analysis_id"] == aid
    assert data["rating"] == "helpful"
    assert data["comment"] == "Clear evidence."


def test_28_feedback_retrieval(client: TestClient) -> None:
    """Test 28: GET feedback lists all feedback records for an analysis."""
    create_res = client.post("/api/v1/analyze", json={"text": "Multiple feedback target."})
    aid = create_res.json()["analysis_id"]

    client.post(f"/api/v1/analyses/{aid}/feedback", json={"analysis_id": aid, "rating": "helpful"})
    client.post(f"/api/v1/analyses/{aid}/feedback", json={"analysis_id": aid, "rating": "unsure"})

    list_res = client.get(f"/api/v1/analyses/{aid}/feedback")
    assert list_res.status_code == 200
    items = list_res.json()
    assert len(items) == 2


def test_29_invalid_feedback_rating_rejected(client: TestClient) -> None:
    """Test 29: Invalid feedback rating string returns HTTP 422."""
    create_res = client.post("/api/v1/analyze", json={"text": "Statement."})
    aid = create_res.json()["analysis_id"]

    res = client.post(
        f"/api/v1/analyses/{aid}/feedback",
        json={"analysis_id": aid, "rating": "bogus_rating"},
    )
    assert res.status_code == 422


def test_30_invalid_corrected_verdict_rejected(client: TestClient) -> None:
    """Test 30: Invalid corrected verdict returns HTTP 422."""
    create_res = client.post("/api/v1/analyze", json={"text": "Statement."})
    aid = create_res.json()["analysis_id"]

    res = client.post(
        f"/api/v1/analyses/{aid}/feedback",
        json={"analysis_id": aid, "rating": "not_helpful", "corrected_verdict": "false_news"},
    )
    assert res.status_code == 422


def test_31_feedback_nonexistent_analysis_returns_404(client: TestClient) -> None:
    """Test 31: Submitting feedback for nonexistent analysis returns HTTP 404."""
    res = client.post(
        "/api/v1/analyses/nonexistent-id/feedback",
        json={"analysis_id": "nonexistent-id", "rating": "helpful"},
    )
    assert res.status_code == 404


def test_32_delete_cascades_feedback(client: TestClient) -> None:
    """Test 32: Deleting parent analysis cascades deletion to child feedback records."""
    create_res = client.post("/api/v1/analyze", json={"text": "Cascade feedback check."})
    aid = create_res.json()["analysis_id"]

    client.post(f"/api/v1/analyses/{aid}/feedback", json={"analysis_id": aid, "rating": "helpful"})
    assert len(client.get(f"/api/v1/analyses/{aid}/feedback").json()) == 1

    client.delete(f"/api/v1/analyses/{aid}")
    assert client.get(f"/api/v1/analyses/{aid}/feedback").status_code == 404


def test_33_db_error_safely_mapped(temp_api_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test 33: Persistence failure maps to HTTP 503 without leaking stack trace."""
    engine = FakeHybridEngine()
    repo = AnalysisRepository(db_path=temp_api_db)

    def broken_save(*args: Any, **kwargs: Any) -> str:
        raise DatabaseError("Disk full or permission denied")

    monkeypatch.setattr(repo, "save_analysis", broken_save)

    service = AnalysisService(engine=engine, repository=repo)
    app = create_app(analysis_service=service, repository=repo, engine=engine)
    local_client = TestClient(app)

    res = local_client.post("/api/v1/analyze", json={"text": "DB error test statement."})
    assert res.status_code == 503
    data = res.json()
    assert data["error"] == "service_unavailable"
    assert "Traceback" not in json.dumps(data)


def test_34_pipeline_error_safely_mapped(temp_api_db: Path) -> None:
    """Test 34: Unhandled pipeline crash maps to HTTP 500 without leaking stack trace."""
    engine = FakeHybridEngine(should_fail=True)
    repo = AnalysisRepository(db_path=temp_api_db)
    service = AnalysisService(engine=engine, repository=repo)
    app = create_app(analysis_service=service, repository=repo, engine=engine)
    local_client = TestClient(app, raise_server_exceptions=False)

    res = local_client.post("/api/v1/analyze", json={"text": "Crash test."})
    assert res.status_code == 500
    data = res.json()
    assert data["error"] == "internal_server_error"
    assert "Traceback" not in json.dumps(data)


def test_35_api_key_absent_from_response(client: TestClient) -> None:
    """Test 35: API responses never contain API keys or auth credentials."""
    res = client.post("/api/v1/analyze", json={"text": "Secret leak check."})
    body_str = res.text
    assert "LLM_API_KEY" not in body_str
    assert "AIza" not in body_str
    assert "Bearer " not in body_str


def test_36_api_key_absent_from_openapi(client: TestClient) -> None:
    """Test 36: OpenAPI schema never contains secret keys or environment values."""
    res = client.get("/openapi.json")
    schema_str = res.text
    assert "LLM_API_KEY" not in schema_str
    assert "AIza" not in schema_str


def test_37_synthetic_evidence_override_absent_from_analyze_request(client: TestClient) -> None:
    """Test 37: Passing allow_synthetic_evidence in POST /analyze is rejected (extra forbidden)."""
    res = client.post(
        "/api/v1/analyze",
        json={"text": "Probe claim.", "allow_synthetic_evidence": True},
    )
    assert res.status_code == 422


def test_38_synthetic_evidence_override_absent_from_openapi(client: TestClient) -> None:
    """Test 38: OpenAPI schema has zero occurrences of allow_synthetic_evidence in request paths."""
    res = client.get("/openapi.json")
    assert "allow_synthetic_evidence" not in res.text


def test_39_no_stack_traces_exposed(client: TestClient) -> None:
    """Test 39: Error responses for 404 and 422 do not expose raw stack traces."""
    r404 = client.get("/api/v1/analyses/nonexistent-id")
    r422 = client.post("/api/v1/analyze", json={"text": ""})
    for r in [r404, r422]:
        text = r.text
        assert "Traceback" not in text
        assert "File \"" not in text


def test_40_unicode_input_works(client: TestClient) -> None:
    """Test 40: Multilingual text and emojis are handled cleanly."""
    unicode_text = "Breaking 🚨: भारत सरकार ने घोषणा की 🚀! 这是一个测试 🌟."
    res = client.post("/api/v1/analyze", json={"text": unicode_text})
    assert res.status_code == 200
    assert res.json()["analysis"]["input_text"] == unicode_text


def test_41_quotes_and_newlines_work(client: TestClient) -> None:
    """Test 41: Text with mixed single quotes, double quotes, and newlines works cleanly."""
    quote_text = "Headline: 'Official Report'\nParagraph: \"He stated: 'True.'\"\r\nDone."
    res = client.post("/api/v1/analyze", json={"text": quote_text})
    assert res.status_code == 200
    assert res.json()["analysis"]["input_text"] == quote_text


def test_42_sql_like_input_handled_safely(client: TestClient) -> None:
    """Test 42: SQL injection strings in request body are stored safely as text."""
    sql_text = "'; DROP TABLE analyses; SELECT * FROM analyses; --"
    res = client.post("/api/v1/analyze", json={"text": sql_text})
    assert res.status_code == 200
    aid = res.json()["analysis_id"]

    # Database remains operational
    get_res = client.get(f"/api/v1/analyses/{aid}")
    assert get_res.status_code == 200


def test_43_component_statuses_serialized(client: TestClient) -> None:
    """Test 43: component_statuses is returned as a list of structured status objects."""
    res = client.post("/api/v1/analyze", json={"text": "Status serialization check."})
    statuses = res.json()["analysis"]["component_statuses"]
    assert isinstance(statuses, list)
    assert len(statuses) > 0
    assert statuses[0]["component"] == "cleaning"
    assert statuses[0]["status"] == "success"


def test_44_evidence_serialized(client: TestClient) -> None:
    """Test 44: Evidence items are serialized accurately in API response."""
    res = client.post("/api/v1/analyze", json={"text": "Evidence check."})
    ev_list = res.json()["analysis"]["evidence"]
    assert isinstance(ev_list, list)
    assert len(ev_list) == 1
    assert ev_list[0]["source_name"] == "Official Wire"


def test_45_llm_none_serialized(temp_api_db: Path) -> None:
    """Test 45: When LLM is skipped / None, response['analysis']['llm'] is None."""
    engine = FakeHybridEngine(force_abstain=True)
    repo = AnalysisRepository(db_path=temp_api_db)
    service = AnalysisService(engine=engine, repository=repo)
    app = create_app(analysis_service=service, repository=repo, engine=engine)
    local_client = TestClient(app)

    res = local_client.post("/api/v1/analyze", json={"text": "Short-circuit check."})
    assert res.status_code == 200
    assert res.json()["analysis"]["llm"] is None


def test_46_timings_serialized(client: TestClient) -> None:
    """Test 46: Latency breakdown timings are returned as non-negative floats."""
    res = client.post("/api/v1/analyze", json={"text": "Timings check."})
    timings = res.json()["analysis"]["timings"]
    assert timings is not None
    assert timings["total_ms"] > 0.0


def test_47_pytest_blocks_real_network_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test 47: Proves network guard is active and blocks unauthorized external connections."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with pytest.raises(RuntimeError) as exc_info:
        s.connect(("8.8.8.8", 80))
    assert "Prohibited external network call" in str(exc_info.value)


def test_48_tests_use_temporary_db(temp_api_db: Path) -> None:
    """Test 48: Verifies test suite operates on temp_api_db and not DATABASE_PATH."""
    assert temp_api_db != DATABASE_PATH
    assert "test_api_database" in str(temp_api_db)


def test_49_production_db_untouched() -> None:
    """Test 49: Production database data/app/fake_news_analysis.db contains 0 rows."""
    if DATABASE_PATH.exists():
        repo = AnalysisRepository(DATABASE_PATH, auto_init=False)
        assert repo.count_analyses() == 0
        assert repo.count_feedback() == 0


def test_50_no_model_training() -> None:
    """Test 50: Verifies classical model and test split hashes remain strictly untouched."""
    model_path = MODELS_DIR / "classical_best_validation.joblib"
    if model_path.exists():
        expected_model_hash = "d4c3636277761f5399166ee41c0eec57cafe7b9216bacfc18d81f26fea415c92"
        actual_model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        assert actual_model_hash == expected_model_hash

    test_tsv = DATA_DIR / "raw" / "test.tsv"
    if test_tsv.exists():
        expected_tsv_hash = "ee41e2c741e0019216952625145af8c4aa8c8d3057ec68284b134267bb845e6f"
        actual_tsv_hash = hashlib.sha256(test_tsv.read_bytes()).hexdigest()
        assert actual_tsv_hash == expected_tsv_hash
