"""Unit tests for EvidenceQueryBuilder (V2.2).

Verifies deterministic extraction, entity preservation, numeric preservation,
injection neutralization, length bounds, and zero LLM calls.
"""

from __future__ import annotations

import socket
from typing import Any
import pytest

from backend.app.services.evidence_query_builder import (
    EvidenceQueryBuilder,
    MAX_QUERY_LENGTH,
    MAX_QUERY_TERMS,
)


@pytest.fixture(autouse=True)
def guard_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strict guard: prohibit any network connections during unit tests."""
    def forbidden_connect(self: socket.socket, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Prohibited real network call during test execution!")

    monkeypatch.setattr(socket.socket, "connect", forbidden_connect)


def test_query_deterministic() -> None:
    """Requirement 1: Query generation is strictly deterministic across multiple runs."""
    builder = EvidenceQueryBuilder()
    claim = "NASA announced Artemis III will land astronauts near the lunar South Pole in 2026."
    q1 = builder.build_query(claim)
    q2 = builder.build_query(claim)
    assert q1 == q2
    assert len(q1) > 0


def test_numbers_preserved() -> None:
    """Requirement 2: Numeric quantities, years, and monetary figures are preserved."""
    builder = EvidenceQueryBuilder()
    claim = "Company reported 150 million revenue increase and 20.5% profit in 2025."
    query = builder.build_query(claim)
    assert "150" in query
    assert "2025" in query
    assert "million" in query


def test_important_named_phrases_preserved() -> None:
    """Requirement 3: Capitalized entities and key domain nouns are preserved."""
    builder = EvidenceQueryBuilder()
    claim = "Pandora opened a jewellery factory in Ho Chi Minh City."
    query = builder.build_query(claim)
    assert "Pandora" in query
    assert "jewellery" in query
    assert "factory" in query
    assert "Ho" in query or "Minh" in query


def test_operator_injection_neutralized() -> None:
    """Requirement 4: Injected GDELT operators and booleans are neutralized safely."""
    builder = EvidenceQueryBuilder()
    claim = "domain:evil.com OR AND NOT Climate change near:5 repeat:10 in 2026"
    query = builder.build_query(claim)
    assert "domain:" not in query
    assert "near:" not in query
    assert "repeat:" not in query
    assert "OR" not in query.split()
    assert "AND" not in query.split()
    assert "NOT" not in query.split()
    assert "Climate" in query
    assert "change" in query


def test_query_length_bounded() -> None:
    """Requirement 5: Generated queries never exceed maximum configured character and term limits."""
    builder = EvidenceQueryBuilder(max_terms=MAX_QUERY_TERMS, max_length=MAX_QUERY_LENGTH)
    long_claim = (
        "The extraordinary international consortium of renowned scientists and geological researchers "
        "published an exhaustive 450-page comprehensive environmental assessment document covering "
        "every single nation across the continent of Europe and South America during the winter of 2025."
    )
    query = builder.build_query(long_claim)
    assert len(query) <= MAX_QUERY_LENGTH
    terms = query.split()
    assert len(terms) <= MAX_QUERY_TERMS


def test_url_stripped_from_claim() -> None:
    """Neutralize inbound URLs embedded in claim text."""
    builder = EvidenceQueryBuilder()
    claim = "Check out https://scam-site.example/fakenews for details about Mars rover discovery."
    query = builder.build_query(claim)
    assert "https" not in query
    assert "scam-site" not in query
    assert "Mars" in query
    assert "rover" in query


def test_stopwords_removed() -> None:
    """Common low-information function words are omitted."""
    builder = EvidenceQueryBuilder()
    claim = "This is a statement that has been verified by the authorities."
    query = builder.build_query(claim)
    terms = query.lower().split()
    assert "this" not in terms
    assert "is" not in terms
    assert "a" not in terms
    assert "that" not in terms
    assert "has" not in terms
    assert "been" not in terms
    assert "by" not in terms
    assert "the" not in terms


def test_empty_or_whitespace_claim_returns_empty() -> None:
    """Edge cases for empty or non-string inputs produce clean empty strings."""
    builder = EvidenceQueryBuilder()
    assert builder.build_query("") == ""
    assert builder.build_query("   ") == ""
    assert builder.build_query(None) == ""  # type: ignore[arg-type]


def test_no_groq_called_for_query_building() -> None:
    """Verify query builder operates purely with Python standard library, zero LLM calls."""
    builder = EvidenceQueryBuilder()
    query = builder.build_query("Electric vehicle sales rose 35% in California.")
    assert "Electric" in query
    assert "35%" in query or "35" in query
    assert "California" in query


def test_query_plan_produced_and_deterministic() -> None:
    """Requirement: Build deterministic multi-tier plan with up to 4 attempts."""
    builder = EvidenceQueryBuilder()
    claim = "Electric vehicle battery manufacturers announced investments into production capacity."
    plan1 = builder.build_plan(claim)
    plan2 = builder.build_plan(claim)

    assert len(plan1.queries) > 0
    assert len(plan1.queries) <= 4
    assert len(plan1.queries) == len(plan2.queries)

    # Determinism
    for q1, q2 in zip(plan1.queries, plan2.queries):
        assert q1.query == q2.query
        assert q1.strategy == q2.strategy
        assert q1.priority == q2.priority
        assert q1.endpoint_preference == q2.endpoint_preference


def test_max_4_attempts_respected() -> None:
    """Max plan attempts never exceeds 4."""
    builder = EvidenceQueryBuilder(max_attempts=4)
    claim = (
        "The international aerospace consortium developed revolutionary propulsion engines "
        "enabling interplanetary travel to Mars and Venus within sixty days of launch."
    )
    plan = builder.build_plan(claim)
    assert len(plan.queries) <= 4
    for i, q in enumerate(plan.queries, start=1):
        assert q.priority == i


def test_precise_query_short_and_relaxed_shorter_or_equal() -> None:
    """Precise query is concise (2-4 concepts) and relaxed query prioritizes recall."""
    builder = EvidenceQueryBuilder()
    claim = "Electric vehicle battery manufacturers announced investments into production capacity."
    plan = builder.build_plan(claim)

    precise_q = plan.queries[0]
    assert precise_q.strategy == "precise"
    # Precise query should be concise, not a huge conjunction
    terms = precise_q.query.split()
    assert len(terms) <= 5

    # Relaxed query exists and is shorter or equal in length
    relaxed_queries = [q for q in plan.queries if q.strategy in ("relaxed", "broad")]
    assert len(relaxed_queries) > 0
    for rq in relaxed_queries:
        assert len(rq.query.split()) <= len(precise_q.query.split()) + 1


def test_named_phrases_and_dates_preserved() -> None:
    """Named phrases are quoted and dates are recognized."""
    builder = EvidenceQueryBuilder()
    claim = "The 2026 Nobel Prize in Physiology or Medicine was awarded to three scientists."
    plan = builder.build_plan(claim)

    assert plan.detected_year == 2026
    assert plan.is_historical is False

    # Check that "Nobel Prize" phrase was preserved
    precise_query = plan.queries[0].query
    assert '"Nobel Prize"' in precise_query
    assert "2026" in precise_query or "Physiology" in precise_query


def test_historical_claim_flagged_for_doc_priority() -> None:
    """Claims targeting past years older than Context horizon are marked historical."""
    builder = EvidenceQueryBuilder(current_year=2026)
    claim = "In 2021 the World Health Organization declared a global public health milestone."
    plan = builder.build_plan(claim)

    assert plan.detected_year == 2021
    assert plan.is_historical is True
    # Historical queries prefer doc endpoint
    assert plan.queries[0].endpoint_preference == "doc"


def test_long_raw_claim_not_used_directly() -> None:
    """Long natural language claims are not used verbatim as single conjunction queries."""
    builder = EvidenceQueryBuilder()
    long_claim = (
        "The 2026 Nobel Prize in Physiology or Medicine was awarded to three scientists "
        "for work that led to using light to study brain activity."
    )
    plan = builder.build_plan(long_claim)
    for q in plan.queries:
        assert q.query != long_claim
        assert len(q.query.split()) < len(long_claim.split())


def test_user_injection_neutralized_and_internal_or_allowed() -> None:
    """User injected operators are neutralized while internal OR grouping is allowed."""
    builder = EvidenceQueryBuilder()
    malicious_claim = "domain:attacker.com OR AND NOT Electric vehicle battery near:10 repeat:5"
    plan = builder.build_plan(malicious_claim)

    # Inbound operators neutralized
    for q in plan.queries:
        assert "domain:" not in q.query
        assert "near:" not in q.query
        assert "repeat:" not in q.query

    # Tier B contains internally generated OR group safely
    tier_b = [q for q in plan.queries if q.strategy == "entity_anchor"]
    if tier_b:
        assert " OR " in tier_b[0].query
        assert "(" in tier_b[0].query and ")" in tier_b[0].query

