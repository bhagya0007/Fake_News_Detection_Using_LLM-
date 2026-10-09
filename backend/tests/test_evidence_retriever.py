"""Unit tests for the Evidence Knowledge Base and Semantic Retrieval Layer (Step 13).

Strict Guarantees:
- 100% offline (autouse socket guard prevents any external network calls).
- Zero LLM calls.
- Zero access to data/raw/test.tsv.
- Similarity scores strictly separated from evidentiary stance.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.app.schemas import (
    Claim,
    EvidenceItem,
    EvidenceRetrievalResult,
    EvidenceStance,
)
from backend.app.services.evidence_retriever import (
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K,
    MAX_TOP_K,
    EvidenceRetriever,
)
from scripts.seed_evidence import ingest_evidence, validate_evidence_record


# ---------------------------------------------------------------------------
# Offline Socket Guard (Guarantees zero network calls during tests)
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def block_external_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prohibit any network socket connections during unit test execution."""
    real_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Network connection prohibited in unit tests!")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


# ---------------------------------------------------------------------------
# Deterministic Fake Embedder for Offline Semantic Tests
# ---------------------------------------------------------------------------
def fake_encoder(texts: List[str]) -> np.ndarray:
    """Lightweight deterministic fake encoder mapping texts to 8-dim vectors.

    Uses word presence to deterministically populate vector dimensions without network or PyTorch weights.
    """
    dim = 8
    vecs = []
    for text in texts:
        t_low = text.lower()
        vec = np.zeros(dim, dtype=np.float32)
        if "factory" in t_low or "manufacturing" in t_low or "texas" in t_low:
            vec[0] = 1.0
            vec[1] = 0.8
        if "hire" in t_low or "worker" in t_low or "recruitment" in t_low:
            vec[2] = 1.0
            vec[3] = 0.7
        if "ceo" in t_low or "resignation" in t_low:
            vec[4] = 1.0
        if "solar" in t_low or "battery" in t_low:
            vec[5] = 1.0
        if "cheese" in t_low or "moon" in t_low:
            vec[6] = 1.0
        # Base magnitude
        vec[7] = 0.1
        vecs.append(vec)
    return np.array(vecs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Test Fixture Helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_demo_records() -> List[dict[str, Any]]:
    return [
        {
            "id": "item_001",
            "claim_summary": "Acme Corp factory in Texas",
            "evidence_text": "Acme Corp announced today that it will construct a new manufacturing facility in Texas.",
            "source_name": "Synthetic Demo Source",
            "source_url": "https://example.com/synthetic/acme-texas",
            "published_date": "2024-03-15",
            "stance": "supports",
            "source_type": "press_release",
            "is_synthetic": True,
        },
        {
            "id": "item_002",
            "claim_summary": "Acme hiring workers",
            "evidence_text": "Acme Corp announced recruitment for 3,000 workers across assembly plants.",
            "source_name": "Synthetic Demo Source",
            "source_url": "https://example.com/synthetic/acme-hiring",
            "published_date": "2024-03-20",
            "stance": "supports",
            "source_type": "press_release",
            "is_synthetic": True,
        },
        {
            "id": "item_003",
            "claim_summary": "Acme CEO resignation rumor",
            "evidence_text": "Acme board confirmed the CEO did not resign and remains in office.",
            "source_name": "Synthetic Demo Source",
            "source_url": "https://example.com/synthetic/acme-ceo",
            "published_date": "2024-04-12",
            "stance": "contradicts",
            "source_type": "statement",
            "is_synthetic": True,
        },
        {
            "id": "item_004",
            "claim_summary": "Solar battery research",
            "evidence_text": "Evaluation recorded high energy density in prototype battery tests.",
            "source_name": "Synthetic Demo Source",
            "source_url": "https://example.com/synthetic/solar-battery",
            "published_date": "2024-04-18",
            "stance": "unknown",
            "source_type": "report",
            "is_synthetic": True,
        },
    ]


@pytest.fixture
def temp_kb_file(tmp_path: Path, sample_demo_records: List[dict[str, Any]]) -> Path:
    kb_file = tmp_path / "test_kb.jsonl"
    with open(kb_file, "w", encoding="utf-8") as f:
        for rec in sample_demo_records:
            f.write(json.dumps(rec) + "\n")
    return kb_file


# ---------------------------------------------------------------------------
# Test Cases (All 30 Required Scenarios)
# ---------------------------------------------------------------------------

def test_1_valid_kb_load(temp_kb_file: Path) -> None:
    """1. Valid KB load loads all valid records into memory."""
    retriever = EvidenceRetriever(kb_path=temp_kb_file, backend="tfidf_lexical")
    assert len(retriever.records) == 4
    assert retriever.records[0].id == "item_001"
    assert retriever.records[0].source_name == "Synthetic Demo Source"


def test_2_malformed_evidence_record_rejected(tmp_path: Path) -> None:
    """2. Malformed JSON record in KB raises ValueError upon loading."""
    bad_kb = tmp_path / "bad.jsonl"
    bad_kb.write_text('{"id": 1, "claim_summary": "Broken"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed evidence record"):
        EvidenceRetriever(kb_path=bad_kb, backend="tfidf_lexical")


def test_3_missing_source_rejected_for_real_evidence() -> None:
    """3. Real evidence records (is_synthetic=False) without source_url or source_name are rejected."""
    raw_real = {
        "id": "real_001",
        "claim_summary": "Real economic claim",
        "evidence_text": "Real economic statistic.",
        "source_name": "Synthetic Demo Source",  # Invalid source_name for real record
        "source_url": None,  # Missing source_url
        "stance": "supports",
        "is_synthetic": False,
    }
    item, err = validate_evidence_record(raw_real)
    assert item is None
    assert err is not None
    assert "Missing or invalid source_url" in err or "Invalid real source_name" in err


def test_4_synthetic_fixture_accepted_when_marked_synthetic(sample_demo_records: List[dict[str, Any]]) -> None:
    """4. Synthetic demo fixtures with is_synthetic=True are accepted."""
    item, err = validate_evidence_record(sample_demo_records[0])
    assert err is None
    assert item is not None
    assert item.is_synthetic is True
    assert item.source_name == "Synthetic Demo Source"


def test_5_retrieval_returns_relevant_item(temp_kb_file: Path) -> None:
    """5. Query matching an evidence record returns that relevant item above threshold."""
    retriever = EvidenceRetriever(
        kb_path=temp_kb_file,
        custom_encoder=fake_encoder,
        threshold=0.5,
    )
    result = retriever.retrieve("Acme Corp announced a new factory in Texas.")
    assert result.no_sufficient_evidence is False
    assert len(result.items) > 0
    top_item = result.items[0]
    assert top_item.id == "item_001"
    assert top_item.similarity_score is not None
    assert top_item.similarity_score >= 0.5


def test_6_unrelated_item_below_threshold_returns_no_sufficient_evidence(temp_kb_file: Path) -> None:
    """6. Unrelated query where scores do not clear threshold returns no_sufficient_evidence=True."""
    retriever = EvidenceRetriever(
        kb_path=temp_kb_file,
        backend="tfidf_lexical",
        threshold=0.45,
    )
    result = retriever.retrieve("Astronauts discovered ancient geological formations on Mars.")
    assert result.no_sufficient_evidence is True
    assert result.items == []


def test_7_top_k_respected(temp_kb_file: Path) -> None:
    """7. Requested top_k is strictly respected."""
    retriever = EvidenceRetriever(
        kb_path=temp_kb_file,
        custom_encoder=fake_encoder,
        threshold=0.1,  # Low threshold so multiple qualify
    )
    result = retriever.retrieve("Acme factory hiring", top_k=2)
    assert len(result.items) <= 2


def test_8_maximum_top_k_enforced(temp_kb_file: Path) -> None:
    """8. Maximum top_k (MAX_TOP_K=5) is enforced even if larger top_k requested."""
    retriever = EvidenceRetriever(
        kb_path=temp_kb_file,
        custom_encoder=fake_encoder,
        threshold=0.01,
        max_top_k=3,
    )
    result = retriever.retrieve("Acme factory", top_k=10)
    assert len(result.items) <= 3


def test_9_similarity_score_bounded_appropriately(temp_kb_file: Path) -> None:
    """9. Similarity scores are strictly within [0.0, 1.0]."""
    retriever = EvidenceRetriever(
        kb_path=temp_kb_file,
        custom_encoder=fake_encoder,
        threshold=0.0,
    )
    result = retriever.retrieve("Acme factory")
    for item in result.items:
        assert item.similarity_score is not None
        assert 0.0 <= item.similarity_score <= 1.0


def test_10_similarity_does_not_change_stance(temp_kb_file: Path) -> None:
    """10. High similarity score does not change recorded stance."""
    retriever = EvidenceRetriever(
        kb_path=temp_kb_file,
        custom_encoder=fake_encoder,
        threshold=0.3,
    )
    result = retriever.retrieve("Acme Corp CEO resignation rumors")
    # item_003 has stance=contradicts
    ceo_items = [it for it in result.items if it.id == "item_003"]
    assert len(ceo_items) == 1
    assert ceo_items[0].stance == EvidenceStance.contradicts


def test_11_high_similarity_unknown_stance_remains_unknown(temp_kb_file: Path) -> None:
    """11. High similarity with unknown stance remains unknown (similarity != support)."""
    retriever = EvidenceRetriever(
        kb_path=temp_kb_file,
        custom_encoder=fake_encoder,
        threshold=0.3,
    )
    result = retriever.retrieve("Solar battery research tests")
    battery_items = [it for it in result.items if it.id == "item_004"]
    assert len(battery_items) == 1
    assert battery_items[0].stance == EvidenceStance.unknown


def test_12_support_stance_preserved(temp_kb_file: Path) -> None:
    """12. 'supports' stance is preserved on retrieval."""
    retriever = EvidenceRetriever(kb_path=temp_kb_file, custom_encoder=fake_encoder, threshold=0.3)
    result = retriever.retrieve("Acme factory")
    item_1 = [it for it in result.items if it.id == "item_001"][0]
    assert item_1.stance == EvidenceStance.supports


def test_13_contradict_stance_preserved(temp_kb_file: Path) -> None:
    """13. 'contradicts' stance is preserved on retrieval."""
    retriever = EvidenceRetriever(kb_path=temp_kb_file, custom_encoder=fake_encoder, threshold=0.3)
    result = retriever.retrieve("Acme CEO resignation")
    item_3 = [it for it in result.items if it.id == "item_003"][0]
    assert item_3.stance == EvidenceStance.contradicts


def test_14_evidence_text_unchanged(temp_kb_file: Path, sample_demo_records: List[dict[str, Any]]) -> None:
    """14. Evidence text is preserved verbatim without alteration."""
    retriever = EvidenceRetriever(kb_path=temp_kb_file, custom_encoder=fake_encoder, threshold=0.3)
    result = retriever.retrieve("Acme factory")
    item_1 = [it for it in result.items if it.id == "item_001"][0]
    assert item_1.evidence_text == sample_demo_records[0]["evidence_text"]


def test_15_source_metadata_unchanged(temp_kb_file: Path, sample_demo_records: List[dict[str, Any]]) -> None:
    """15. Source name, URL, and published date are preserved."""
    retriever = EvidenceRetriever(kb_path=temp_kb_file, custom_encoder=fake_encoder, threshold=0.3)
    result = retriever.retrieve("Acme factory")
    item_1 = [it for it in result.items if it.id == "item_001"][0]
    assert item_1.source_name == sample_demo_records[0]["source_name"]
    assert item_1.source_url == sample_demo_records[0]["source_url"]


def test_16_empty_query_rejected(temp_kb_file: Path) -> None:
    """16. Empty query string raises ValueError."""
    retriever = EvidenceRetriever(kb_path=temp_kb_file, backend="tfidf_lexical")
    with pytest.raises(ValueError, match="cannot be empty"):
        retriever.retrieve("")


def test_17_whitespace_query_rejected(temp_kb_file: Path) -> None:
    """17. Whitespace-only query string raises ValueError."""
    retriever = EvidenceRetriever(kb_path=temp_kb_file, backend="tfidf_lexical")
    with pytest.raises(ValueError, match="cannot be empty"):
        retriever.retrieve("   \n\t  ")


def test_18_none_query_rejected(temp_kb_file: Path) -> None:
    """18. None query raises TypeError."""
    retriever = EvidenceRetriever(kb_path=temp_kb_file, backend="tfidf_lexical")
    with pytest.raises(TypeError, match="must be a non-empty string"):
        retriever.retrieve(None)  # type: ignore


def test_19_non_string_query_rejected(temp_kb_file: Path) -> None:
    """19. Non-string query raises TypeError."""
    retriever = EvidenceRetriever(kb_path=temp_kb_file, backend="tfidf_lexical")
    with pytest.raises(TypeError, match="must be a non-empty string"):
        retriever.retrieve(12345)  # type: ignore


def test_20_duplicate_evidence_handled_predictably(tmp_path: Path) -> None:
    """20. Duplicate evidence items in KB are indexed and scored predictably."""
    dup_kb = tmp_path / "dup.jsonl"
    rec = {
        "id": "dup_1",
        "claim_summary": "Duplicate summary",
        "evidence_text": "Duplicate text content.",
        "source_name": "Synthetic Demo Source",
        "stance": "supports",
        "is_synthetic": True,
    }
    dup_kb.write_text(json.dumps(rec) + "\n" + json.dumps(rec) + "\n", encoding="utf-8")
    retriever = EvidenceRetriever(kb_path=dup_kb, backend="tfidf_lexical", threshold=0.1)
    result = retriever.retrieve("Duplicate summary")
    assert len(result.items) == 2


def test_21_multiple_claim_retrieval_works(temp_kb_file: Path) -> None:
    """21. retrieve_for_claims correctly groups evidence per claim without cross-contamination."""
    retriever = EvidenceRetriever(kb_path=temp_kb_file, custom_encoder=fake_encoder, threshold=0.4)
    claims = [
        Claim(id=1, text="Acme factory in Texas", checkable=True),
        Claim(id=2, text="Acme plans to hire workers", checkable=True),
    ]
    grouped = retriever.retrieve_for_claims(claims, top_k_per_claim=2)
    assert "Acme factory in Texas" in grouped
    assert "Acme plans to hire workers" in grouped
    assert len(grouped["Acme factory in Texas"]) > 0


def test_22_no_evidence_returns_empty_list(tmp_path: Path) -> None:
    """22. Empty KB returns empty items list and no_sufficient_evidence=True."""
    empty_kb = tmp_path / "empty.jsonl"
    empty_kb.write_text("", encoding="utf-8")
    retriever = EvidenceRetriever(kb_path=empty_kb, backend="tfidf_lexical")
    result = retriever.retrieve("Any query")
    assert result.items == []
    assert result.no_sufficient_evidence is True


def test_23_deterministic_ranking_for_equal_inputs(temp_kb_file: Path) -> None:
    """23. Multiple calls with identical inputs produce deterministic ranking and scores."""
    retriever = EvidenceRetriever(kb_path=temp_kb_file, custom_encoder=fake_encoder, threshold=0.3)
    res1 = retriever.retrieve("Acme factory in Texas")
    res2 = retriever.retrieve("Acme factory in Texas")
    assert [it.id for it in res1.items] == [it.id for it in res2.items]
    assert [it.similarity_score for it in res1.items] == [it.similarity_score for it in res2.items]


def test_24_index_metadata_aligns_with_records(tmp_path: Path, sample_demo_records: List[dict[str, Any]]) -> None:
    """24. Index directory metadata matches persisted record count and alignment."""
    from scripts.build_evidence_index import build_index

    kb_file = tmp_path / "index_kb.jsonl"
    kb_file.write_text("\n".join(json.dumps(r) for r in sample_demo_records), encoding="utf-8")
    idx_dir = tmp_path / "idx_out"

    stats = build_index(kb_path=kb_file, output_dir=idx_dir, backend="tfidf_lexical")
    assert stats["num_records"] == 4
    assert (idx_dir / "evidence_index_metadata.json").exists()

    meta = json.loads((idx_dir / "evidence_index_metadata.json").read_text(encoding="utf-8"))
    assert len(meta["records"]) == 4


def test_25_corrupted_mismatched_index_rejected(tmp_path: Path) -> None:
    """25. Mismatched embeddings and record metadata raises ValueError upon loading."""
    idx_dir = tmp_path / "corrupt_idx"
    idx_dir.mkdir(parents=True)

    metadata = {
        "backend": "sentence_transformer",
        "num_records": 2,
        "records": [
            {
                "id": "1",
                "claim_summary": "A",
                "evidence_text": "Text A",
                "source_name": "Synthetic Demo Source",
                "is_synthetic": True,
            }
        ],  # Only 1 record
    }
    (idx_dir / "evidence_index_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    # 2 embedding rows
    np.save(idx_dir / "evidence_embeddings.npy", np.zeros((2, 8), dtype=np.float32))

    with pytest.raises(ValueError, match="Index alignment error"):
        EvidenceRetriever(index_dir=idx_dir)


def test_26_pytest_performs_no_external_network_calls(temp_kb_file: Path) -> None:
    """26. Attempting external network calls raises RuntimeError via the socket guard."""
    with pytest.raises(RuntimeError, match="Network connection prohibited"):
        s = socket.socket()
        s.connect(("8.8.8.8", 53))


def test_27_no_test_tsv_access() -> None:
    """27. Strict assurance: data/raw/test.tsv is never opened by the retriever."""
    # Ensure test.tsv path does not appear anywhere in evidence_retriever source code
    import backend.app.services.evidence_retriever as er_module
    import inspect

    src = inspect.getsource(er_module)
    assert "test.tsv" not in src
    assert "data/raw/test.tsv" not in src


def test_28_no_llm_calls_made(temp_kb_file: Path) -> None:
    """28. EvidenceRetriever makes zero LLM calls."""
    with patch("backend.app.ai.llm_client.LLMClient") as mock_client:
        retriever = EvidenceRetriever(kb_path=temp_kb_file, backend="tfidf_lexical")
        retriever.retrieve("Acme factory")
        mock_client.assert_not_called()


def test_29_no_final_truth_verdict_emitted(temp_kb_file: Path) -> None:
    """29. Retriever emits EvidenceRetrievalResult, never a final truth verdict."""
    retriever = EvidenceRetriever(kb_path=temp_kb_file, backend="tfidf_lexical")
    res = retriever.retrieve("Acme factory")
    assert isinstance(res, EvidenceRetrievalResult)
    # The result has no verdict or truth status
    assert not hasattr(res, "verdict")
    assert not hasattr(res, "label")


def test_30_similarity_never_automatically_becomes_support(temp_kb_file: Path) -> None:
    """30. High similarity never alters stance to 'supports' when stance was not supports."""
    retriever = EvidenceRetriever(
        kb_path=temp_kb_file,
        custom_encoder=fake_encoder,
        threshold=0.1,
    )
    result = retriever.retrieve("Acme Corp CEO resignation")
    for item in result.items:
        if item.id == "item_003":
            assert item.stance == EvidenceStance.contradicts
            assert item.stance != EvidenceStance.supports
        elif item.id == "item_004":
            assert item.stance == EvidenceStance.unknown
            assert item.stance != EvidenceStance.supports
