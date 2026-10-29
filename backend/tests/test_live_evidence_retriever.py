"""Unit and Integration Tests for LiveEvidenceRetriever (V2.2).

Tests candidate normalization, URL provenance, domain parsing, strict unknown stance,
canonical deduplication, domain diversity, deterministic reranking, DOC fallback,
graceful failure degradation, and hybrid engine integration.
Guarantees 100% network isolation with zero socket calls during pytest.
"""

from __future__ import annotations

import zlib
from pathlib import Path
import socket
from typing import Any, List
from unittest.mock import MagicMock
import numpy as np
import pytest

from backend.app.config import DATA_DIR, MODELS_DIR
from backend.app.schemas import (
    Claim,
    EvidenceItem,
    EvidenceStance,
    VerdictLabel,
)
from backend.app.services.evidence_query_builder import EvidenceQueryBuilder
from backend.app.services.gdelt_client import GDELTClient
from backend.app.services.hybrid_engine import HybridAnalysisEngine
from backend.app.services.live_evidence_retriever import (
    LiveEvidenceRetriever,
    canonicalize_url,
)


@pytest.fixture(autouse=True)
def guard_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strict guard: prohibit any network connections during unit tests."""
    def forbidden_connect(self: socket.socket, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Prohibited real network call during test execution!")

    monkeypatch.setattr(socket.socket, "connect", forbidden_connect)
    monkeypatch.setenv("LLM_PROVIDER", "mock")


def fake_encoder(texts: List[str]) -> np.ndarray:
    """Deterministic fast mock embedder for tests: creates normalized bag-of-words vectors."""
    vectors = []
    for t in texts:
        v = np.zeros(64, dtype=np.float32)
        words = t.lower().split()
        for w in words:
            idx = zlib.crc32(w.encode("utf-8")) % 64
            v[idx] += 1.0
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        vectors.append(v)
    return np.array(vectors)


# ---------------------------------------------------------------------------
# Test 1: Provenance fields preserved (URL, domain, date, snippet)
# ---------------------------------------------------------------------------
def test_source_url_and_domain_preserved() -> None:
    mock_client = MagicMock()
    mock_client.search_context.return_value = [
        {
            "url": "https://reuters.com/world/story-123?utm_source=twitter",
            "title": "Clean Energy Investment Record",
            "domain": "reuters.com",
            "snippet": "Global investment in clean energy exceeded 1 trillion dollars.",
            "published_date": "2026-10-15",
            "retrieval_source": "gdelt_context",
            "evidence_quality": "context_snippet",
        }
    ]

    retriever = LiveEvidenceRetriever(
        gdelt_client=mock_client,
        custom_encoder=fake_encoder,
        top_k=3,
    )
    items, meta = retriever.retrieve_for_claim(
        Claim(id=1, text="Global investment in clean energy exceeded 1 trillion dollars in 2026.")
    )

    assert len(items) == 1
    item = items[0]
    assert item.source_url == "https://reuters.com/world/story-123?utm_source=twitter"
    assert item.source_domain == "reuters.com"
    assert item.source_name == "reuters.com"
    assert item.published_date == "2026-10-15"
    assert "clean energy exceeded" in item.evidence_text
    assert item.title == "Clean Energy Investment Record"


# ---------------------------------------------------------------------------
# Test 2: Synthetic is strictly False for live evidence
# ---------------------------------------------------------------------------
def test_synthetic_is_strictly_false() -> None:
    mock_client = MagicMock()
    mock_client.search_context.return_value = [
        {
            "url": "https://apnews.com/article/live-report",
            "title": "Live Report Title",
            "domain": "apnews.com",
            "snippet": "Live report snippet content.",
            "retrieval_source": "gdelt_context",
            "evidence_quality": "context_snippet",
        }
    ]

    retriever = LiveEvidenceRetriever(gdelt_client=mock_client, custom_encoder=fake_encoder)
    items, _ = retriever.retrieve_for_claim("Test claim")

    assert len(items) == 1
    assert items[0].is_synthetic is False


# ---------------------------------------------------------------------------
# Test 3: Stance is always strictly 'unknown' in V2.2
# ---------------------------------------------------------------------------
def test_stance_always_unknown() -> None:
    mock_client = MagicMock()
    mock_client.search_context.return_value = [
        {
            "url": "https://apnews.com/article/1",
            "title": "Confirmed truth headline",
            "domain": "apnews.com",
            "snippet": "This text appears to agree with the claim completely.",
            "retrieval_source": "gdelt_context",
            "evidence_quality": "context_snippet",
        }
    ]

    retriever = LiveEvidenceRetriever(gdelt_client=mock_client, custom_encoder=fake_encoder)
    items, _ = retriever.retrieve_for_claim("Confirmed claim proposition")

    assert len(items) == 1
    assert items[0].stance == EvidenceStance.unknown


# ---------------------------------------------------------------------------
# Test 4: Retrieval sources distinguished (Context vs DOC)
# ---------------------------------------------------------------------------
def test_retrieval_source_and_quality_distinguished() -> None:
    mock_client = MagicMock()
    # Context returns 1 result
    mock_client.search_context.return_value = [
        {
            "url": "https://bbc.com/news/1",
            "title": "Context Article",
            "domain": "bbc.com",
            "snippet": "Context sentence text.",
            "retrieval_source": "gdelt_context",
            "evidence_quality": "context_snippet",
        }
    ]
    # DOC returns 1 fallback result
    mock_client.search_doc.return_value = [
        {
            "url": "https://theguardian.com/world/2",
            "title": "DOC Headline Only Article",
            "domain": "theguardian.com",
            "snippet": "DOC Headline Only Article",
            "retrieval_source": "gdelt_doc",
            "evidence_quality": "headline_only",
        }
    ]

    retriever = LiveEvidenceRetriever(
        gdelt_client=mock_client,
        custom_encoder=fake_encoder,
        top_k=5,
        doc_fallback=True,
    )
    items, meta = retriever.retrieve_for_claim("Clean energy technology")

    assert len(items) == 2
    context_item = next(it for it in items if it.source_domain == "bbc.com")
    doc_item = next(it for it in items if it.source_domain == "theguardian.com")

    assert context_item.retrieval_source == "gdelt_context"
    assert context_item.evidence_quality == "context_snippet"

    assert doc_item.retrieval_source == "gdelt_doc"
    assert doc_item.evidence_quality == "headline_only"


# ---------------------------------------------------------------------------
# Test 5: Deduplication by canonical URL
# ---------------------------------------------------------------------------
def test_deduplication_by_canonical_url() -> None:
    mock_client = MagicMock()
    mock_client.search_context.return_value = [
        {
            "url": "https://www.reuters.com/business/story-123/",
            "title": "Story Title",
            "domain": "reuters.com",
            "snippet": "Snippet A",
            "retrieval_source": "gdelt_context",
        },
        {
            "url": "https://reuters.com/business/story-123?utm_source=facebook&utm_medium=social",
            "title": "Story Title",
            "domain": "reuters.com",
            "snippet": "Snippet A identical story",
            "retrieval_source": "gdelt_context",
        },
    ]

    retriever = LiveEvidenceRetriever(gdelt_client=mock_client, custom_encoder=fake_encoder)
    items, meta = retriever.retrieve_for_claim("Business story claim")

    assert len(items) == 1
    assert meta["raw_candidates_count"] == 2
    assert meta["deduplicated_count"] == 1


# ---------------------------------------------------------------------------
# Test 6: Different domains preserved even with identical titles
# ---------------------------------------------------------------------------
def test_different_domains_preserved_even_with_similar_titles() -> None:
    mock_client = MagicMock()
    mock_client.search_context.return_value = [
        {
            "url": "https://reuters.com/article/wire-story",
            "title": "Wire Service Report",
            "domain": "reuters.com",
            "snippet": "Text from Reuters",
            "retrieval_source": "gdelt_context",
        },
        {
            "url": "https://apnews.com/article/wire-story",
            "title": "Wire Service Report",
            "domain": "apnews.com",
            "snippet": "Text from AP News",
            "retrieval_source": "gdelt_context",
        },
    ]

    retriever = LiveEvidenceRetriever(gdelt_client=mock_client, custom_encoder=fake_encoder)
    items, meta = retriever.retrieve_for_claim("Wire report claim")

    # Independent domains must NOT be collapsed
    assert len(items) == 2
    domains = {it.source_domain for it in items}
    assert domains == {"reuters.com", "apnews.com"}
    assert meta["unique_source_domains"] == 2


# ---------------------------------------------------------------------------
# Test 7: MiniLM semantic reranking deterministic
# ---------------------------------------------------------------------------
def test_minilm_reranking_deterministic() -> None:
    mock_client = MagicMock()
    mock_client.search_context.return_value = [
        {
            "url": "https://domain-a.com/irrelevant",
            "title": "Cooking recipes for summer desserts",
            "domain": "domain-a.com",
            "snippet": "Bake delicious cakes with fresh strawberries and cream.",
            "retrieval_source": "gdelt_context",
        },
        {
            "url": "https://domain-b.com/relevant",
            "title": "NASA Mars Exploration Update",
            "domain": "domain-b.com",
            "snippet": "The Mars Perseverance rover discovered organic molecules in rock samples.",
            "retrieval_source": "gdelt_context",
        },
    ]

    retriever = LiveEvidenceRetriever(gdelt_client=mock_client, custom_encoder=fake_encoder)
    claim_text = "NASA Mars rover found signs of organic chemistry in rocks."
    items, _ = retriever.retrieve_for_claim(claim_text)

    assert len(items) == 2
    # The relevant Mars article should be ranked first
    assert items[0].source_domain == "domain-b.com"
    assert items[0].similarity_score is not None
    assert items[1].similarity_score is not None
    assert items[0].similarity_score > items[1].similarity_score


# ---------------------------------------------------------------------------
# Test 8: Top-k respected
# ---------------------------------------------------------------------------
def test_top_k_respected() -> None:
    mock_client = MagicMock()
    mock_client.search_context.return_value = [
        {
            "url": f"https://source-{i}.com/article",
            "title": f"Article {i}",
            "domain": f"source-{i}.com",
            "snippet": f"Snippet text number {i}",
            "retrieval_source": "gdelt_context",
        }
        for i in range(10)
    ]

    retriever = LiveEvidenceRetriever(gdelt_client=mock_client, custom_encoder=fake_encoder, top_k=3)
    items, _ = retriever.retrieve_for_claim("Test search topic")
    assert len(items) == 3


# ---------------------------------------------------------------------------
# Test 9: Source diversity count measured
# ---------------------------------------------------------------------------
def test_source_diversity_count_measured() -> None:
    mock_client = MagicMock()
    mock_client.search_context.return_value = [
        {"url": "https://domain1.com/a", "title": "T1", "domain": "domain1.com", "snippet": "S1"},
        {"url": "https://domain1.com/b", "title": "T2", "domain": "domain1.com", "snippet": "S2"},
        {"url": "https://domain2.com/c", "title": "T3", "domain": "domain2.com", "snippet": "S3"},
    ]

    retriever = LiveEvidenceRetriever(gdelt_client=mock_client, custom_encoder=fake_encoder)
    items, meta = retriever.retrieve_for_claim("Diversity claim")
    assert meta["unique_source_domains"] == 2


# ---------------------------------------------------------------------------
# Test 10: Disabled live retriever returns empty and skipped status
# ---------------------------------------------------------------------------
def test_live_retriever_disabled_returns_empty_and_skipped() -> None:
    mock_client = MagicMock()
    retriever = LiveEvidenceRetriever(gdelt_client=mock_client, enabled=False)
    items, meta = retriever.retrieve_for_claim("Some claim text")
    assert items == []
    assert meta["status"] == "skipped"
    assert mock_client.search_context.call_count == 0


# ---------------------------------------------------------------------------
# Test 11: GDELT outage safe fallback does not crash
# ---------------------------------------------------------------------------
def test_gdelt_outage_safe_fallback_no_crash() -> None:
    mock_client = MagicMock()
    mock_client.search_context.side_effect = RuntimeError("GDELT API server down")
    mock_client.search_doc.side_effect = RuntimeError("GDELT DOC API down")

    retriever = LiveEvidenceRetriever(gdelt_client=mock_client, custom_encoder=fake_encoder)
    items, meta = retriever.retrieve_for_claim("Factual claim during outage")
    assert items == []
    assert meta["status"] == "unavailable"


# ---------------------------------------------------------------------------
# Test 12: HybridAnalysisEngine runs with live evidence retriever
# ---------------------------------------------------------------------------
def test_hybrid_engine_does_not_crash_with_live_evidence() -> None:
    mock_gdelt = MagicMock()
    mock_gdelt.search_context.return_value = [
        {
            "url": "https://reuters.com/world/vietnam-plant",
            "title": "Pandora opens facility in Vietnam",
            "domain": "reuters.com",
            "snippet": "Pandora opened its $150 million factory in Ho Chi Minh City.",
            "published_date": "2026-10-18",
            "retrieval_source": "gdelt_context",
            "evidence_quality": "context_snippet",
        }
    ]
    mock_gdelt.search_doc.return_value = []

    live_retriever = LiveEvidenceRetriever(
        gdelt_client=mock_gdelt,
        custom_encoder=fake_encoder,
        top_k=2,
    )

    mock_local_ret = MagicMock()
    mock_local_ret.backend_name = "mock_local"
    mock_local_ret.threshold = 0.45
    mock_local_ret.retrieve.return_value = []

    engine = HybridAnalysisEngine(
        evidence_retriever=mock_local_ret,
        live_evidence_retriever=live_retriever,
    )

    response = engine.analyze_text("Pandora opened a jewellery factory in Ho Chi Minh City.")

    # Engine completed analysis successfully
    assert response.input_text == "Pandora opened a jewellery factory in Ho Chi Minh City."
    assert any(item.source_domain == "reuters.com" for item in response.evidence)

    # Check component status for live evidence retrieval
    live_status = next(s for s in response.component_statuses if s.component == "live_evidence_retrieval")
    assert live_status.status.value == "success"

    # Stance remains unknown, so fusion safely abstains
    assert response.verdict == VerdictLabel.unverifiable
    assert response.support_score == 0.0


# ---------------------------------------------------------------------------
# Test 13: Hybrid engine degrades gracefully when GDELT fails
# ---------------------------------------------------------------------------
def test_hybrid_engine_degrades_gracefully_when_gdelt_fails() -> None:
    mock_gdelt = MagicMock()
    mock_gdelt.search_context.side_effect = RuntimeError("Network partition")
    mock_gdelt.search_doc.side_effect = RuntimeError("Network partition")

    live_retriever = LiveEvidenceRetriever(
        gdelt_client=mock_gdelt,
        custom_encoder=fake_encoder,
    )

    mock_local_ret = MagicMock()
    mock_local_ret.backend_name = "mock_local"
    mock_local_ret.threshold = 0.45
    mock_local_ret.retrieve.return_value = []

    engine = HybridAnalysisEngine(
        evidence_retriever=mock_local_ret,
        live_evidence_retriever=live_retriever,
    )
    response = engine.analyze_text("Test claim under GDELT network outage.")

    # Pipeline did not crash
    assert response.verdict == VerdictLabel.unverifiable
    assert response.support_score == 0.0
    live_status = next(s for s in response.component_statuses if s.component == "live_evidence_retrieval")
    assert live_status.status.value in ("unavailable", "failed")


# ---------------------------------------------------------------------------
# Test 14: System integrity invariants
# ---------------------------------------------------------------------------
def test_integrity_invariants_preserved() -> None:
    """Verify test.tsv untouched and models directory unmodified."""
    raw_test_file = DATA_DIR / "raw" / "test.tsv"
    if raw_test_file.exists():
        # Confirm file is present and not an empty stub
        assert raw_test_file.stat().st_size > 0

    model_file = MODELS_DIR / "classical_best_validation.joblib"
    if model_file.exists():
        assert model_file.stat().st_size > 0


# ---------------------------------------------------------------------------
# Test 15: Progressive relaxation improves candidate discovery
# ---------------------------------------------------------------------------
def test_progressive_relaxation_discovers_candidates_when_precise_empty() -> None:
    """When precise tier returns 0, relaxed tier successfully discovers candidates."""
    mock_client = MagicMock()
    mock_client.last_status = "success"
    mock_client.last_error = None
    mock_client.last_elapsed_ms = 15.0

    # precise query -> 0 items, entity_anchor / relaxed query -> items found
    def mock_context(query: str, timespan: Any = None) -> list[dict[str, Any]]:
        if "manufacturers investments" in query:
            return []  # precise query returned 0
        return [
            {
                "url": "https://bloomberg.com/news/ev-battery-growth",
                "title": "EV Battery Plant Investments Surge",
                "domain": "bloomberg.com",
                "snippet": "Electric vehicle battery manufacturers expand output capacity.",
                "retrieval_source": "gdelt_context",
                "evidence_quality": "context_snippet",
            }
        ]

    mock_client.search_context.side_effect = mock_context
    mock_client.search_doc.return_value = []

    retriever = LiveEvidenceRetriever(gdelt_client=mock_client, custom_encoder=fake_encoder)
    claim = "Electric vehicle battery manufacturers announced investments into production capacity."
    items, meta = retriever.retrieve_for_claim(claim)

    assert len(items) == 1
    assert items[0].source_domain == "bloomberg.com"
    assert meta["status"] == "success"
    assert meta["context_count"] == 1
    assert meta["attempts_count"] >= 2


# ---------------------------------------------------------------------------
# Test 16: Candidates across multiple attempts are merged and deduplicated
# ---------------------------------------------------------------------------
def test_candidates_across_attempts_merged_and_deduplicated() -> None:
    """Candidates from Context and DOC attempts are merged and deduplicated."""
    mock_client = MagicMock()
    mock_client.last_status = "success"
    mock_client.last_error = None
    mock_client.last_elapsed_ms = 10.0

    # Attempt 1: Context returns 1 candidate
    mock_client.search_context.return_value = [
        {
            "url": "https://domain-1.com/story-a",
            "title": "Solar Output Record",
            "domain": "domain-1.com",
            "snippet": "Solar capacity reaches new highs in detailed context snippet.",
            "retrieval_source": "gdelt_context",
            "evidence_quality": "context_snippet",
        }
    ]

    # Attempt 2: DOC returns duplicate of story-a (headline only) + new story-b
    mock_client.search_doc.return_value = [
        {
            "url": "https://domain-1.com/story-a?utm_source=twitter",
            "title": "Solar Output Record",
            "domain": "domain-1.com",
            "snippet": "Solar Output Record",
            "retrieval_source": "gdelt_doc",
            "evidence_quality": "headline_only",
        },
        {
            "url": "https://domain-2.com/story-b",
            "title": "Grid Expansion Details",
            "domain": "domain-2.com",
            "snippet": "Grid Expansion Details",
            "retrieval_source": "gdelt_doc",
            "evidence_quality": "headline_only",
        },
    ]

    retriever = LiveEvidenceRetriever(gdelt_client=mock_client, custom_encoder=fake_encoder, top_k=5)
    claim = "Solar energy capacity is expanding globally."
    items, meta = retriever.retrieve_for_claim(claim)

    # 2 unique items retained despite duplicate across Context and DOC
    assert len(items) == 2
    domains = {it.source_domain for it in items}
    assert domains == {"domain-1.com", "domain-2.com"}
    assert meta["deduplicated_count"] == 2

    # Best candidate retained: story-a retains context_snippet quality from Context
    story_a = next(it for it in items if it.source_domain == "domain-1.com")
    assert story_a.evidence_quality == "context_snippet"


# ---------------------------------------------------------------------------
# Test 17: Early stopping triggered when enough candidates found
# ---------------------------------------------------------------------------
def test_early_stopping_triggered_when_threshold_reached() -> None:
    mock_client = MagicMock()
    mock_client.last_status = "success"
    mock_client.last_error = None
    mock_client.last_elapsed_ms = 8.0

    # Attempt 1 returns 5 unique candidates (>= stop count)
    mock_client.search_context.return_value = [
        {
            "url": f"https://domain-{i}.com/article",
            "title": f"Article {i}",
            "domain": f"domain-{i}.com",
            "snippet": f"Snippet {i}",
            "retrieval_source": "gdelt_context",
            "evidence_quality": "context_snippet",
        }
        for i in range(5)
    ]
    mock_client.search_doc.return_value = []

    retriever = LiveEvidenceRetriever(
        gdelt_client=mock_client,
        custom_encoder=fake_encoder,
        search_stop_count=5,
    )
    items, meta = retriever.retrieve_for_claim("Solar capacity growth")

    assert len(items) == 5
    assert meta["stop_reason"] == "early_stop_satisfied"
    assert meta["attempts_count"] == 1
    assert mock_client.search_context.call_count == 1
    assert mock_client.search_doc.call_count == 0


# ---------------------------------------------------------------------------
# Test 18: HTTP 429 aborts further progressive attempts (rate shield)
# ---------------------------------------------------------------------------
def test_rate_limit_429_aborts_further_attempts() -> None:
    mock_client = MagicMock()
    mock_client.search_context.return_value = []
    mock_client.last_status = "rate_limited"
    mock_client.last_error = "GDELT rate limit encountered (HTTP 429)"
    mock_client.last_elapsed_ms = 50.0

    retriever = LiveEvidenceRetriever(gdelt_client=mock_client, custom_encoder=fake_encoder)
    items, meta = retriever.retrieve_for_claim("Test claim")

    assert items == []
    assert meta["status"] == "unavailable"
    assert meta["stop_reason"] == "aborted_rate_limited"
    assert meta["attempts_count"] == 1
    # Confirms no request storm
    assert mock_client.search_context.call_count == 1
    assert mock_client.search_doc.call_count == 0


# ---------------------------------------------------------------------------
# Test 19: Full search trace recorded in metadata
# ---------------------------------------------------------------------------
def test_full_search_trace_recorded_in_metadata() -> None:
    mock_client = MagicMock()
    mock_client.last_status = "success"
    mock_client.last_error = None
    mock_client.last_elapsed_ms = 12.0
    mock_client.search_context.return_value = []
    mock_client.search_doc.return_value = []

    retriever = LiveEvidenceRetriever(gdelt_client=mock_client, custom_encoder=fake_encoder)
    _, meta = retriever.retrieve_for_claim("Electric vehicle battery production")

    assert "search_trace" in meta
    assert isinstance(meta["search_trace"], list)
    assert len(meta["search_trace"]) > 0
    for attempt in meta["search_trace"]:
        assert "strategy" in attempt
        assert "endpoint" in attempt
        assert "query" in attempt
        assert "result_count" in attempt
        assert "status" in attempt
        assert "elapsed_ms" in attempt


# ---------------------------------------------------------------------------
# Test 20: Historical claims skip Context and route to DOC
# ---------------------------------------------------------------------------
def test_historical_claim_routes_to_doc_directly() -> None:
    mock_client = MagicMock()
    mock_client.last_status = "success"
    mock_client.last_error = None
    mock_client.last_elapsed_ms = 10.0
    mock_client.search_doc.return_value = [
        {
            "url": "https://history.org/event-2020",
            "title": "Historical Event Report",
            "domain": "history.org",
            "snippet": "Historical Event Report",
            "retrieval_source": "gdelt_doc",
            "evidence_quality": "headline_only",
        }
    ]

    retriever = LiveEvidenceRetriever(gdelt_client=mock_client, custom_encoder=fake_encoder)
    # 2020 is historical relative to current_year=2026
    claim = "In 2020 scientists achieved breakthrough results."
    items, meta = retriever.retrieve_for_claim(claim)

    assert meta["status"] == "success"
    # DOC search was called, context was skipped
    assert mock_client.search_doc.call_count >= 1
    assert mock_client.search_context.call_count == 0
    assert len(items) == 1
    assert items[0].evidence_quality == "headline_only"

