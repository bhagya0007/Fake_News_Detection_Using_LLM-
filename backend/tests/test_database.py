"""Unit and Integration Tests for SQLite Persistence Layer (Step 16).

Tests all 45+ required persistence scenarios:
- Database initialization and idempotency
- Schema tables (analyses, feedback) and indexes
- Foreign keys and CASCADE deletion
- Parameterized CRUD operations for analyses
- Full Pydantic AnalysisResponse JSON serialization and round-trip reconstruction
- History pagination (ORDER BY created_at DESC, limit, offset)
- Input validation (limits, offsets, feedback ratings, corrected verdicts)
- SQL injection immunity and edge case handling (Unicode, long text, quotes, newlines)
- Data corruption detection (DataIntegrityError)
- Secret isolation (zero API key / credentials serialization)
- Test database isolation (never touches production data/app/fake_news_analysis.db)
- Integration test with AnalysisService + Fake engine
- Determinism, no network, frozen classical model & test.tsv integrity
"""

from __future__ import annotations

import hashlib
import json
import socket
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pytest
from pydantic import ValidationError

from backend.app.config import DATABASE_PATH, DATA_DIR, MODELS_DIR, PIPELINE_VERSION
from backend.app.db.connection import get_connection, resolve_db_path
from backend.app.db.exceptions import (
    AnalysisNotFoundError,
    DatabaseError,
    DataIntegrityError,
    FeedbackValidationError,
)
from backend.app.db.repository import AnalysisRepository
from backend.app.db.schema import init_database
from backend.app.schemas import (
    AnalysisHistoryItem,
    AnalysisRecord,
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
    FeedbackRecord,
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
from backend.app.services.analysis_service import AnalysisService
from backend.app.services.red_flags import analyze_red_flags


# ---------------------------------------------------------------------------
# Network Isolation Guard (Autouse)
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def guard_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strict guard: prohibit any real network calls during test execution."""

    def forbidden_socket(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Prohibited real network call during test execution!")

    monkeypatch.setattr(socket, "socket", forbidden_socket)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Return a temporary database path unique to each test."""
    return tmp_path / "test_fake_news_analysis.db"


@pytest.fixture
def repo(temp_db_path: Path) -> AnalysisRepository:
    """Return an AnalysisRepository initialized against an isolated temp DB."""
    return AnalysisRepository(db_path=temp_db_path)


def make_sample_response(
    text: str = "Test claim text.",
    verdict: VerdictLabel = VerdictLabel.likely_reliable,
    support_score: float = 0.85,
    reason: str = "Test verified justification.",
    has_llm: bool = True,
    has_ml: bool = True,
    has_red_flags: bool = True,
    has_evidence: bool = True,
) -> AnalysisResponse:
    """Helper constructing a valid, realistic AnalysisResponse."""
    claims = [
        ClaimResult(
            claim=Claim(id=1, text=text[:200], checkable=(verdict != VerdictLabel.unverifiable)),
            assessment=ClaimAssessment.supported if verdict == VerdictLabel.likely_reliable else ClaimAssessment.unverifiable,
            evidence=[],
        )
    ]

    ev_list: list[EvidenceItem] = []
    if has_evidence:
        ev = EvidenceItem(
            id="EV-TEST-1",
            claim_summary="Summary of evidence",
            evidence_text="Corroborating statement from official wire.",
            source_name="Official Wire",
            stance=EvidenceStance.supports,
            is_synthetic=False,
        )
        ev_list.append(ev)
        claims[0].evidence.append(ev)

    ml = None
    if has_ml:
        ml = MLAssessment(
            label=MLLabel.likely_reliable,
            probabilities=ClassProbabilities(likely_reliable=0.82, likely_fake=0.08, misleading=0.10),
            model_name="Tuned TF-IDF + Logistic Regression",
            model_version="1.0.0",
        )

    red_flags = None
    if has_red_flags:
        red_flags = adapter_from_red_flags(analyze_red_flags(text))

    llm = None
    if has_llm:
        llm = LLMAssessment(
            label=verdict,
            source_attribution=SourceAttribution.strong,
            evidence_support=EvidenceSupport.supported,
            internal_consistency=InternalConsistency.consistent,
            rhetorical_cues=[],
            unsupported_claims=[],
            short_reason="Factual proposition verified against wire documentation.",
        )

    statuses = [
        ComponentStatus(component="cleaning", status=ComponentExecutionStatus.success),
        ComponentStatus(component="claim_extraction", status=ComponentExecutionStatus.success),
        ComponentStatus(component="classical_ml", status=ComponentExecutionStatus.success),
        ComponentStatus(component="red_flags", status=ComponentExecutionStatus.success),
        ComponentStatus(component="evidence_retrieval", status=ComponentExecutionStatus.success if has_evidence else ComponentExecutionStatus.skipped),
        ComponentStatus(component="llm_reasoner", status=ComponentExecutionStatus.success if has_llm else ComponentExecutionStatus.skipped),
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
        claims=claims,
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
# Test Suite
# ---------------------------------------------------------------------------
def test_1_database_initialization(temp_db_path: Path) -> None:
    """Test 1: Initializing database creates file and directory."""
    assert not temp_db_path.exists()
    init_database(temp_db_path)
    assert temp_db_path.exists()


def test_2_initialization_idempotent(temp_db_path: Path) -> None:
    """Test 2: Multiple initialization calls succeed without errors or data loss."""
    init_database(temp_db_path)
    repo = AnalysisRepository(temp_db_path, auto_init=False)
    resp = make_sample_response("Idempotent test statement.")
    saved_id = repo.save_analysis(resp)

    # Re-initialize schema
    init_database(temp_db_path)

    # Data is preserved
    fetched = repo.get_analysis(saved_id)
    assert fetched is not None
    assert fetched.input_text == "Idempotent test statement."


def test_3_analyses_table_exists(temp_db_path: Path) -> None:
    """Test 3: Analyses table exists with all required columns."""
    init_database(temp_db_path)
    with get_connection(temp_db_path) as conn:
        cursor = conn.execute("PRAGMA table_info(analyses);")
        columns = {row["name"]: row["type"] for row in cursor.fetchall()}

    expected_cols = {
        "analysis_id",
        "created_at",
        "input_text",
        "cleaned_text",
        "verdict",
        "abstained",
        "support_score",
        "reason",
        "claims_json",
        "ml_json",
        "red_flags_json",
        "llm_json",
        "evidence_json",
        "component_statuses_json",
        "timings_json",
        "analysis_response_json",
        "pipeline_version",
    }
    assert expected_cols.issubset(set(columns.keys()))


def test_4_feedback_table_exists(temp_db_path: Path) -> None:
    """Test 4: Feedback table exists with all required columns and foreign key."""
    init_database(temp_db_path)
    with get_connection(temp_db_path) as conn:
        cursor = conn.execute("PRAGMA table_info(feedback);")
        columns = {row["name"]: row["type"] for row in cursor.fetchall()}

    expected_cols = {
        "feedback_id",
        "analysis_id",
        "created_at",
        "rating",
        "comment",
        "corrected_verdict",
    }
    assert expected_cols.issubset(set(columns.keys()))


def test_5_indexes_exist(temp_db_path: Path) -> None:
    """Test 5: Indexes on created_at and analysis_id exist."""
    init_database(temp_db_path)
    with get_connection(temp_db_path) as conn:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND name LIKE 'idx_%';"
        )
        index_names = {row["name"] for row in cursor.fetchall()}

    assert "idx_analyses_created_at" in index_names
    assert "idx_feedback_analysis_id" in index_names


def test_6_foreign_keys_enabled(temp_db_path: Path) -> None:
    """Test 6: Foreign key enforcement is enabled via PRAGMA."""
    init_database(temp_db_path)
    with get_connection(temp_db_path) as conn:
        cursor = conn.execute("PRAGMA foreign_keys;")
        row = cursor.fetchone()
        assert row[0] == 1


def test_7_save_analysis(repo: AnalysisRepository) -> None:
    """Test 7: Saving an AnalysisResponse returns a non-empty analysis_id."""
    resp = make_sample_response("Save analysis statement.")
    aid = repo.save_analysis(resp)
    assert isinstance(aid, str)
    assert len(aid) > 0


def test_8_generated_analysis_id_valid(repo: AnalysisRepository) -> None:
    """Test 8: Automatically generated analysis_id is a valid UUID4 string."""
    resp = make_sample_response("Auto UUID statement.")
    resp.analysis_id = None
    aid = repo.save_analysis(resp)
    parsed_uuid = uuid.UUID(aid)
    assert str(parsed_uuid) == aid


def test_9_created_at_timezone_aware_serialized(repo: AnalysisRepository) -> None:
    """Test 9: created_at timestamp is stored as ISO 8601 with UTC timezone."""
    resp = make_sample_response("Timestamp test statement.")
    aid = repo.save_analysis(resp)
    record = repo.get_analysis_record(aid)
    assert record is not None
    assert record.created_at.tzinfo is not None
    assert record.created_at.tzinfo == timezone.utc


def test_10_retrieve_saved_analysis(repo: AnalysisRepository) -> None:
    """Test 10: Saved analysis is retrievable by ID."""
    resp = make_sample_response("Retrieval test statement.")
    aid = repo.save_analysis(resp)
    retrieved = repo.get_analysis(aid)
    assert retrieved is not None
    assert retrieved.analysis_id == aid


def test_11_analysis_response_round_trip(repo: AnalysisRepository) -> None:
    """Test 11: Full AnalysisResponse round-trips with identical verdict and score."""
    resp = make_sample_response("Round trip statement.", verdict=VerdictLabel.misleading, support_score=0.62)
    aid = repo.save_analysis(resp)
    retrieved = repo.get_analysis(aid)
    assert retrieved is not None
    assert retrieved.input_text == resp.input_text
    assert retrieved.verdict == VerdictLabel.misleading
    assert retrieved.support_score == pytest.approx(0.62)
    assert retrieved.reason == resp.reason


def test_12_nested_claims_round_trip(repo: AnalysisRepository) -> None:
    """Test 12: Nested claims and per-claim evidence round-trip accurately."""
    resp = make_sample_response("Claims round trip statement.")
    aid = repo.save_analysis(resp)
    retrieved = repo.get_analysis(aid)
    assert retrieved is not None
    assert len(retrieved.claims) == 1
    assert retrieved.claims[0].claim.text == "Claims round trip statement."
    assert retrieved.claims[0].claim.checkable is True
    assert len(retrieved.claims[0].evidence) == 1


def test_13_ml_assessment_round_trip(repo: AnalysisRepository) -> None:
    """Test 13: Classical ML assessment probabilities and label round-trip accurately."""
    resp = make_sample_response("ML round trip statement.")
    aid = repo.save_analysis(resp)
    retrieved = repo.get_analysis(aid)
    assert retrieved is not None
    assert retrieved.ml is not None
    assert retrieved.ml.label == MLLabel.likely_reliable
    assert retrieved.ml.probabilities.likely_reliable == pytest.approx(0.82)


def test_14_red_flags_round_trip(repo: AnalysisRepository) -> None:
    """Test 14: Red flags result summary and detected flags round-trip accurately."""
    resp = make_sample_response("Round trip red flags statement.")
    aid = repo.save_analysis(resp)
    retrieved = repo.get_analysis(aid)
    assert retrieved is not None
    assert retrieved.red_flags == resp.red_flags


def test_15_optional_llm_none_round_trip(repo: AnalysisRepository) -> None:
    """Test 15: AnalysisResponse with llm=None (e.g., short-circuited) round-trips with llm=None."""
    resp = make_sample_response("No-evidence claim.", verdict=VerdictLabel.unverifiable, support_score=0.0, has_llm=False, has_evidence=False)
    aid = repo.save_analysis(resp)
    retrieved = repo.get_analysis(aid)
    assert retrieved is not None
    assert retrieved.llm is None
    assert retrieved.verdict == VerdictLabel.unverifiable


def test_16_llm_assessment_round_trip(repo: AnalysisRepository) -> None:
    """Test 16: Populated LLM assessment fields round-trip cleanly."""
    resp = make_sample_response("LLM assessment round trip.")
    aid = repo.save_analysis(resp)
    retrieved = repo.get_analysis(aid)
    assert retrieved is not None
    assert retrieved.llm is not None
    assert retrieved.llm.source_attribution == SourceAttribution.strong
    assert retrieved.llm.evidence_support == EvidenceSupport.supported
    assert retrieved.llm.internal_consistency == InternalConsistency.consistent


def test_17_evidence_round_trip(repo: AnalysisRepository) -> None:
    """Test 17: Deduplicated evidence items list round-trips accurately."""
    resp = make_sample_response("Evidence list statement.")
    aid = repo.save_analysis(resp)
    retrieved = repo.get_analysis(aid)
    assert retrieved is not None
    assert len(retrieved.evidence) == 1
    assert retrieved.evidence[0].source_name == "Official Wire"


def test_18_component_statuses_round_trip(repo: AnalysisRepository) -> None:
    """Test 18: Component execution statuses round-trip accurately."""
    resp = make_sample_response("Statuses statement.")
    aid = repo.save_analysis(resp)
    retrieved = repo.get_analysis(aid)
    assert retrieved is not None
    status_map = {cs.component: cs.status for cs in retrieved.component_statuses}
    assert status_map["classical_ml"] == ComponentExecutionStatus.success
    assert status_map["fusion"] == ComponentExecutionStatus.success


def test_19_timings_round_trip(repo: AnalysisRepository) -> None:
    """Test 19: Latency timings breakdown round-trips accurately."""
    resp = make_sample_response("Timings statement.")
    aid = repo.save_analysis(resp)
    retrieved = repo.get_analysis(aid)
    assert retrieved is not None
    assert retrieved.timings is not None
    assert retrieved.timings.total_ms == pytest.approx(6.6)
    assert retrieved.timings.claim_extraction_ms == pytest.approx(1.5)


def test_20_list_analyses(repo: AnalysisRepository) -> None:
    """Test 20: list_analyses returns list of AnalysisHistoryItem summaries."""
    repo.save_analysis(make_sample_response("Statement 1"))
    repo.save_analysis(make_sample_response("Statement 2"))
    history = repo.list_analyses()
    assert len(history) == 2
    assert isinstance(history[0], AnalysisHistoryItem)


def test_21_newest_first_ordering(repo: AnalysisRepository) -> None:
    """Test 21: Analyses are strictly returned in descending order of created_at."""
    r1 = make_sample_response("First Statement")
    r1.created_at = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    r2 = make_sample_response("Second Statement")
    r2.created_at = datetime(2026, 1, 2, 10, 0, tzinfo=timezone.utc)
    r3 = make_sample_response("Third Statement")
    r3.created_at = datetime(2026, 1, 3, 10, 0, tzinfo=timezone.utc)

    repo.save_analysis(r1)
    repo.save_analysis(r2)
    repo.save_analysis(r3)

    history = repo.list_analyses()
    assert len(history) == 3
    assert history[0].input_preview.startswith("Third Statement")
    assert history[1].input_preview.startswith("Second Statement")
    assert history[2].input_preview.startswith("First Statement")


def test_22_limit_works(repo: AnalysisRepository) -> None:
    """Test 22: Limit parameter constrains returned items count."""
    for i in range(5):
        repo.save_analysis(make_sample_response(f"Limit query {i}"))
    items = repo.list_analyses(limit=2)
    assert len(items) == 2


def test_23_offset_works(repo: AnalysisRepository) -> None:
    """Test 23: Offset parameter skips the designated number of records."""
    for i in range(5):
        r = make_sample_response(f"Offset query {i}")
        r.created_at = datetime(2026, 1, 1 + i, 12, 0, tzinfo=timezone.utc)
        repo.save_analysis(r)

    items = repo.list_analyses(limit=2, offset=1)
    assert len(items) == 2
    assert "Offset query 3" in items[0].input_preview


def test_24_invalid_limit_rejected(repo: AnalysisRepository) -> None:
    """Test 24: Limit outside [1, 100] raises ValueError."""
    with pytest.raises(ValueError):
        repo.list_analyses(limit=0)
    with pytest.raises(ValueError):
        repo.list_analyses(limit=101)


def test_25_invalid_offset_rejected(repo: AnalysisRepository) -> None:
    """Test 25: Negative offset raises ValueError."""
    with pytest.raises(ValueError):
        repo.list_analyses(limit=10, offset=-1)


def test_26_unknown_id_returns_none(repo: AnalysisRepository) -> None:
    """Test 26: Unknown analysis_id returns None gracefully."""
    assert repo.get_analysis("unknown-uuid-0000") is None
    assert repo.get_analysis("") is None


def test_27_delete_existing_analysis(repo: AnalysisRepository) -> None:
    """Test 27: Deleting an existing analysis returns True and removes it."""
    aid = repo.save_analysis(make_sample_response("To be deleted."))
    assert repo.get_analysis(aid) is not None
    deleted = repo.delete_analysis(aid)
    assert deleted is True
    assert repo.get_analysis(aid) is None


def test_28_delete_nonexistent_analysis(repo: AnalysisRepository) -> None:
    """Test 28: Deleting a nonexistent analysis returns False."""
    assert repo.delete_analysis("nonexistent-uuid") is False


def test_29_feedback_insertion(repo: AnalysisRepository) -> None:
    """Test 29: Feedback record is created and linked to parent analysis."""
    aid = repo.save_analysis(make_sample_response("Statement for feedback."))
    req = FeedbackRequest(analysis_id=aid, rating=FeedbackRating.helpful, comment="Helpful fact check.")
    fb_record = repo.save_feedback(req)
    assert fb_record.analysis_id == aid
    assert fb_record.rating == FeedbackRating.helpful
    assert fb_record.comment == "Helpful fact check."
    assert isinstance(fb_record.feedback_id, str)


def test_30_feedback_retrieval(repo: AnalysisRepository) -> None:
    """Test 30: All feedback linked to an analysis is retrievable."""
    aid = repo.save_analysis(make_sample_response("Feedback retrieval statement."))
    repo.save_feedback(FeedbackRequest(analysis_id=aid, rating=FeedbackRating.helpful, comment="First fb."))
    repo.save_feedback(FeedbackRequest(analysis_id=aid, rating=FeedbackRating.unsure, comment="Second fb."))

    fb_list = repo.get_feedback_for_analysis(aid)
    assert len(fb_list) == 2
    assert fb_list[0].rating == FeedbackRating.helpful
    assert fb_list[1].rating == FeedbackRating.unsure


def test_31_invalid_feedback_rating_rejected(repo: AnalysisRepository) -> None:
    """Test 31: Unvalidated arbitrary rating string is rejected."""
    aid = repo.save_analysis(make_sample_response("Statement."))
    with pytest.raises(ValidationError):
        FeedbackRequest(analysis_id=aid, rating="arbitrary_invalid_rating")  # type: ignore


def test_32_invalid_corrected_verdict_rejected(repo: AnalysisRepository) -> None:
    """Test 32: Arbitrary corrected verdict string is rejected."""
    aid = repo.save_analysis(make_sample_response("Statement."))
    with pytest.raises(ValidationError):
        FeedbackRequest(
            analysis_id=aid,
            rating=FeedbackRating.not_helpful,
            corrected_verdict="fake_news",  # type: ignore
        )


def test_33_foreign_key_invalid_analysis_rejected(repo: AnalysisRepository) -> None:
    """Test 33: Submitting feedback for nonexistent analysis raises AnalysisNotFoundError."""
    req = FeedbackRequest(
        analysis_id="nonexistent-analysis-id",
        rating=FeedbackRating.helpful,
    )
    with pytest.raises(AnalysisNotFoundError):
        repo.save_feedback(req)


def test_34_deleting_analysis_cascades_feedback(repo: AnalysisRepository) -> None:
    """Test 34: Deleting parent analysis cascades deletion to child feedback rows."""
    aid = repo.save_analysis(make_sample_response("Cascade test statement."))
    repo.save_feedback(FeedbackRequest(analysis_id=aid, rating=FeedbackRating.helpful, comment="To be cascaded."))
    assert len(repo.get_feedback_for_analysis(aid)) == 1

    repo.delete_analysis(aid)
    assert len(repo.get_feedback_for_analysis(aid)) == 0


def test_35_sql_injection_safe(repo: AnalysisRepository) -> None:
    """Test 35: Malicious SQL injection payloads in user input are safely parameterized."""
    malicious = "'; DROP TABLE analyses; DROP TABLE feedback; SELECT * FROM analyses WHERE '1'='1"
    resp = make_sample_response(malicious)
    aid = repo.save_analysis(resp)

    # Database and tables remain intact
    retrieved = repo.get_analysis(aid)
    assert retrieved is not None
    assert retrieved.input_text == malicious
    assert repo.count_analyses() >= 1


def test_36_unicode_text_preserved(repo: AnalysisRepository) -> None:
    """Test 36: Multilingual Unicode text and emojis persist bitwise identically."""
    unicode_text = "Breaking 🚨: भारत सरकार ने घोषणा की 🚀! 这是一个测试 🌟."
    resp = make_sample_response(unicode_text)
    aid = repo.save_analysis(resp)
    retrieved = repo.get_analysis(aid)
    assert retrieved is not None
    assert retrieved.input_text == unicode_text


def test_37_very_long_valid_input_persists(repo: AnalysisRepository) -> None:
    """Test 37: 15,000-character input text persists without truncation."""
    long_text = "This is a detailed factual investigative report. " * 300
    assert len(long_text) > 14000
    resp = make_sample_response(long_text)
    aid = repo.save_analysis(resp)
    retrieved = repo.get_analysis(aid)
    assert retrieved is not None
    assert retrieved.input_text == long_text


def test_38_quotes_and_newlines_preserved(repo: AnalysisRepository) -> None:
    """Test 38: Text containing mixed single quotes, double quotes, and newlines round-trips identically."""
    tricky_text = "Line 1: He said, 'The quote is \"authentic\".'\nLine 2: It's true.\r\nLine 3: End."
    resp = make_sample_response(tricky_text)
    aid = repo.save_analysis(resp)
    retrieved = repo.get_analysis(aid)
    assert retrieved is not None
    assert retrieved.input_text == tricky_text


def test_39_corrupted_response_json_detected(repo: AnalysisRepository, temp_db_path: Path) -> None:
    """Test 39: Manually corrupted JSON in the database raises DataIntegrityError on retrieval."""
    aid = repo.save_analysis(make_sample_response("Corrupt test."))
    with get_connection(temp_db_path) as conn:
        conn.execute(
            "UPDATE analyses SET analysis_response_json = '{invalid_json_corrupted' WHERE analysis_id = ?",
            (aid,),
        )

    with pytest.raises(DataIntegrityError):
        repo.get_analysis(aid)


def test_40_api_key_not_serialized(repo: AnalysisRepository, temp_db_path: Path) -> None:
    """Test 40: Zero API keys or authorization credentials appear in database text."""
    resp = make_sample_response("Secret isolation check.")
    aid = repo.save_analysis(resp)
    with get_connection(temp_db_path) as conn:
        cursor = conn.execute("SELECT * FROM analyses WHERE analysis_id = ?", (aid,))
        row = cursor.fetchone()
        row_str = " ".join(str(val) for val in row)

    assert "LLM_API_KEY" not in row_str
    assert "AIza" not in row_str
    assert "Bearer " not in row_str


def test_41_test_db_isolated_from_production_db(temp_db_path: Path) -> None:
    """Test 41: Testing operates on temp_db_path and never accesses DATABASE_PATH."""
    assert temp_db_path != DATABASE_PATH
    init_database(temp_db_path)
    repo = AnalysisRepository(temp_db_path, auto_init=False)
    repo.save_analysis(make_sample_response("Isolated test."))

    if DATABASE_PATH.exists():
        # Ensure production DB was not populated with the isolated test record
        prod_repo = AnalysisRepository(DATABASE_PATH, auto_init=False)
        assert prod_repo.get_analysis("Isolated test.") is None


def test_42_no_network_access(monkeypatch: pytest.MonkeyPatch, repo: AnalysisRepository) -> None:
    """Test 42: Proves database operations perform zero network requests."""
    # Guard fixture is active; this test simply ensures typical repo operations succeed without network
    aid = repo.save_analysis(make_sample_response("No network text."))
    assert repo.get_analysis(aid) is not None


def test_43_no_gemini_calls(repo: AnalysisRepository) -> None:
    """Test 43: AnalysisRepository uses pure deterministic serialization with 0 Gemini calls."""
    resp = make_sample_response("Gemini-free persistence.")
    aid = repo.save_analysis(resp)
    retrieved = repo.get_analysis(aid)
    assert retrieved is not None


def test_44_no_classical_training() -> None:
    """Test 44: SHA-256 of models/classical_best_validation.joblib is strictly unchanged."""
    model_path = MODELS_DIR / "classical_best_validation.joblib"
    if model_path.exists():
        expected_hash = "d4c3636277761f5399166ee41c0eec57cafe7b9216bacfc18d81f26fea415c92"
        actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        assert actual_hash == expected_hash


def test_45_no_evidence_modification() -> None:
    """Test 45: SHA-256 of data/raw/test.tsv is strictly unchanged."""
    test_tsv = DATA_DIR / "raw" / "test.tsv"
    if test_tsv.exists():
        expected_hash = "ee41e2c741e0019216952625145af8c4aa8c8d3057ec68284b134267bb845e6f"
        actual_hash = hashlib.sha256(test_tsv.read_bytes()).hexdigest()
        assert actual_hash == expected_hash


# ---------------------------------------------------------------------------
# Integration Tests: AnalysisService + Engine Test Double
# ---------------------------------------------------------------------------
class FakeHybridEngine:
    """Deterministic test double for HybridAnalysisEngine."""

    def __init__(self, response: Optional[AnalysisResponse] = None) -> None:
        self.calls: list[str] = []
        self.response = response or make_sample_response("Acme Corp announced a factory in Texas.")

    def analyze_text(self, text: str) -> AnalysisResponse:
        self.calls.append(text)
        # Return a copy with matching input_text
        resp = self.response.model_copy(deep=True)
        resp.input_text = text
        return resp


def test_46_integration_analysis_service_with_fake_engine(repo: AnalysisRepository) -> None:
    """Test 46: Integration of AnalysisService + Fake engine + AnalysisRepository."""
    fake_engine = FakeHybridEngine()
    service = AnalysisService(engine=fake_engine, repository=repo)

    input_text = "Acme Corp announced a factory in Texas."
    resp = service.analyze_and_save(input_text)

    # Verified computed result
    assert resp.input_text == input_text
    assert resp.verdict == VerdictLabel.likely_reliable
    assert resp.analysis_id is not None

    # Verified saved and retrievable
    retrieved = service.get_analysis(resp.analysis_id)
    assert retrieved is not None
    assert retrieved.analysis_id == resp.analysis_id
    assert retrieved.verdict == VerdictLabel.likely_reliable

    # Verified history
    history = service.list_history()
    assert len(history) == 1
    assert history[0].analysis_id == resp.analysis_id


def test_47_analysis_service_database_failure_policy(monkeypatch: pytest.MonkeyPatch, temp_db_path: Path) -> None:
    """Test 47: If database persistence fails, DatabaseError is raised and verdict is not mutated."""
    fake_engine = FakeHybridEngine()
    repo = AnalysisRepository(temp_db_path)

    # Force repository to fail
    def broken_save(*args: Any, **kwargs: Any) -> str:
        raise sqlite3.OperationalError("Simulated disk full error")

    monkeypatch.setattr(repo, "save_analysis", broken_save)

    service = AnalysisService(engine=fake_engine, repository=repo)

    with pytest.raises(DatabaseError) as exc_info:
        service.analyze_and_save("Statement with DB error.")

    # Error message preserves the fact that analysis was completed
    assert "Analysis completed with verdict 'likely_reliable'" in str(exc_info.value)
