"""Unit and integration tests for Deterministic Evidence Consensus and Clustering (Step V2.3).

Tests cover:
1. Direct support consensus across multiple independent domains.
2. Direct contradiction consensus across multiple independent domains.
3. Partial consensus across multiple independent domains.
4. Conflict detection: coexisting support and contradiction forces 'conflicting' and safe abstention.
5. Headline-only exclusion: headline-only evidence is excluded from decisive consensus.
6. Quoted-claim restriction: quoted allegations (quoted_claim) do not confirm propositions.
7. Reported-denial restriction: reported denials cannot count as support.
8. Near-duplicate clustering: syndicated news across multiple domains forms a single cluster.
9. Single-source insufficiency: 1 cluster or 1 domain fails the conservative 2-source threshold.
10. Multi-claim document consensus: overall document consensus combines per-claim statuses.
11. Determinism and zero network calls during evaluation.
"""

from __future__ import annotations

import socket
import pytest

from backend.app.schemas import (
    AttributionType,
    Claim,
    EvidenceConsensusResult,
    EvidenceItem,
    EvidenceItemAssessment,
    EvidenceStance,
)
from backend.app.services.evidence_consensus import (
    EvidenceClusterer,
    EvidenceConsensusEngine,
)


@pytest.fixture(autouse=True)
def block_network():
    """Ensure zero network socket calls during test execution."""
    real_socket = socket.socket

    def guard(*args, **kwargs):
        raise RuntimeError("Prohibited real network call during test execution!")

    socket.socket = guard
    try:
        yield
    finally:
        socket.socket = real_socket


class MockEncoder:
    """Deterministic local mock encoder for offline near-duplicate testing."""

    def encode(self, texts, **kwargs):
        import numpy as np

        embeddings = []
        for t in texts:
            # Deterministic bag-of-chars / hash embedding
            t_lower = t.lower()
            val = sum(ord(c) for c in t_lower[:30])
            # Return 8-dim vector
            vec = np.zeros(8, dtype=float)
            for i, c in enumerate(t_lower[:8]):
                vec[i] = ord(c) / 255.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)
        return np.array(embeddings)


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


def test_1_direct_support_consensus():
    """2 distinct domains with supporting direct reports yield decisive 'supports' consensus."""
    claim = Claim(id=1, text="The central bank reduced interest rates by 25 basis points.", checkable=True)
    items = [
        EvidenceItem(
            id="ev-1",
            claim_summary="Rate cut",
            evidence_text="The monetary committee announced a 25 basis point rate cut today.",
            source_name="Financial Gazette",
            source_domain="financialgazette.com",
            retrieval_source="live_gdelt",
        ),
        EvidenceItem(
            id="ev-2",
            claim_summary="Rate cut",
            evidence_text="Official figures confirm benchmark borrowing costs dropped a quarter point.",
            source_name="Global Wire",
            source_domain="globalwire.org",
            retrieval_source="live_gdelt",
        ),
    ]
    assessments = [
        EvidenceItemAssessment(
            claim_index=0,
            evidence_id="ev-1",
            stance=EvidenceStance.supports,
            attribution_type=AttributionType.direct_report,
            context_sufficient=True,
            short_reason="Directly confirms rate cut.",
        ),
        EvidenceItemAssessment(
            claim_index=0,
            evidence_id="ev-2",
            stance=EvidenceStance.supports,
            attribution_type=AttributionType.direct_report,
            context_sufficient=True,
            short_reason="Corroborates borrowing cost reduction.",
        ),
    ]

    engine = EvidenceConsensusEngine(min_clusters=2, min_domains=2)
    res = engine.evaluate_claim_consensus(claim, 0, items, assessments)

    assert res.consensus_stance == "supports"
    assert res.consensus_sufficient is True
    assert res.support_clusters >= 2
    assert res.unique_domains >= 2
    assert res.conflict_detected is False


def test_2_direct_contradiction_consensus():
    """2 distinct domains with contradicting direct reports yield decisive 'contradicts' consensus."""
    claim = Claim(id=1, text="The president signed the treaty yesterday.", checkable=True)
    items = [
        EvidenceItem(
            id="ev-c1",
            claim_summary="Treaty signing",
            evidence_text="The presidency officially confirmed no treaty was signed yesterday.",
            source_name="Daily News",
            source_domain="dailynews.com",
            retrieval_source="live_gdelt",
        ),
        EvidenceItem(
            id="ev-c2",
            claim_summary="Treaty signing",
            evidence_text="Negotiations stalled and no accord has been signed by the president.",
            source_name="World Times",
            source_domain="worldtimes.net",
            retrieval_source="live_gdelt",
        ),
    ]
    assessments = [
        EvidenceItemAssessment(
            claim_index=0,
            evidence_id="ev-c1",
            stance=EvidenceStance.contradicts,
            attribution_type=AttributionType.direct_report,
            context_sufficient=True,
            short_reason="Explicit denial from presidency.",
        ),
        EvidenceItemAssessment(
            claim_index=0,
            evidence_id="ev-c2",
            stance=EvidenceStance.contradicts,
            attribution_type=AttributionType.direct_report,
            context_sufficient=True,
            short_reason="Reports talks stalled without agreement.",
        ),
    ]

    engine = EvidenceConsensusEngine(min_clusters=2, min_domains=2)
    res = engine.evaluate_claim_consensus(claim, 0, items, assessments)

    assert res.consensus_stance == "contradicts"
    assert res.consensus_sufficient is True
    assert res.contradiction_clusters >= 2
    assert res.support_clusters == 0


def test_3_partial_consensus():
    """2 distinct domains with partial support yield decisive 'partial' consensus."""
    claim = Claim(id=1, text="Company X laid off 5,000 workers in London.", checkable=True)
    items = [
        EvidenceItem(
            id="ev-p1",
            claim_summary="Layoffs",
            evidence_text="Company X announced 5,000 global job cuts, but only 200 in London.",
            source_name="UK Standard",
            source_domain="standard.co.uk",
            retrieval_source="live_gdelt",
        ),
        EvidenceItem(
            id="ev-p2",
            claim_summary="Layoffs",
            evidence_text="The tech firm trimmed workforce worldwide; London office lost a few hundred staff.",
            source_name="Tech Herald",
            source_domain="techherald.io",
            retrieval_source="live_gdelt",
        ),
    ]
    assessments = [
        EvidenceItemAssessment(
            claim_index=0,
            evidence_id="ev-p1",
            stance=EvidenceStance.partial,
            attribution_type=AttributionType.direct_report,
            context_sufficient=True,
            short_reason="Count matches globally but not London.",
        ),
        EvidenceItemAssessment(
            claim_index=0,
            evidence_id="ev-p2",
            stance=EvidenceStance.partial,
            attribution_type=AttributionType.direct_report,
            context_sufficient=True,
            short_reason="Partial factual alignment on cuts.",
        ),
    ]

    engine = EvidenceConsensusEngine(min_clusters=2, min_domains=2)
    res = engine.evaluate_claim_consensus(claim, 0, items, assessments)

    assert res.consensus_stance == "partial"
    assert res.consensus_sufficient is True
    assert res.partial_clusters >= 2


def test_4_conflict_detection_forces_abstention():
    """Coexisting support and contradiction clusters result in 'conflicting' and consensus_sufficient=False."""
    claim = Claim(id=1, text="The merger has received regulatory clearance.", checkable=True)
    items = [
        EvidenceItem(
            id="ev-conf1",
            claim_summary="Merger",
            evidence_text="Regulators approved the acquisition unconditionally on Monday.",
            source_name="Reuters",
            source_domain="reuters.com",
            retrieval_source="live_gdelt",
        ),
        EvidenceItem(
            id="ev-conf2",
            claim_summary="Merger",
            evidence_text="Antitrust authority blocked the merger, citing monopolistic risks.",
            source_name="Bloomberg",
            source_domain="bloomberg.com",
            retrieval_source="live_gdelt",
        ),
    ]
    assessments = [
        EvidenceItemAssessment(
            claim_index=0,
            evidence_id="ev-conf1",
            stance=EvidenceStance.supports,
            attribution_type=AttributionType.direct_report,
            context_sufficient=True,
            short_reason="Reports approval.",
        ),
        EvidenceItemAssessment(
            claim_index=0,
            evidence_id="ev-conf2",
            stance=EvidenceStance.contradicts,
            attribution_type=AttributionType.direct_report,
            context_sufficient=True,
            short_reason="Reports block.",
        ),
    ]

    engine = EvidenceConsensusEngine(min_clusters=2, min_domains=2)
    res = engine.evaluate_claim_consensus(claim, 0, items, assessments)

    assert res.consensus_stance == "conflicting"
    assert res.consensus_sufficient is False
    assert res.conflict_detected is True


def test_5_headline_only_exclusion():
    """Headline-only evidence is excluded from decisive consensus."""
    claim = Claim(id=1, text="Company Y launches new satellite.", checkable=True)
    items = [
        EvidenceItem(
            id="ev-h1",
            claim_summary="Satellite",
            evidence_text="Company Y launches new satellite.",
            source_name="News Aggregator",
            source_domain="aggregator.net",
            evidence_quality="headline_only",
            retrieval_source="live_gdelt",
        ),
        EvidenceItem(
            id="ev-h2",
            claim_summary="Satellite",
            evidence_text="Spacecraft successfully entered low Earth orbit following morning launch.",
            source_name="Space News",
            source_domain="spacenews.com",
            evidence_quality="snippet",
            retrieval_source="live_gdelt",
        ),
    ]
    assessments = [
        EvidenceItemAssessment(
            claim_index=0,
            evidence_id="ev-h1",
            stance=EvidenceStance.supports,
            attribution_type=AttributionType.direct_report,
            context_sufficient=True,
            short_reason="Matches headline.",
        ),
        EvidenceItemAssessment(
            claim_index=0,
            evidence_id="ev-h2",
            stance=EvidenceStance.supports,
            attribution_type=AttributionType.direct_report,
            context_sufficient=True,
            short_reason="Full snippet confirms launch.",
        ),
    ]

    engine = EvidenceConsensusEngine(min_clusters=2, min_domains=2)
    res = engine.evaluate_claim_consensus(claim, 0, items, assessments)

    assert res.excluded_headline_only_count == 1
    # Only 1 snippet remains eligible, so min_clusters=2 is not satisfied
    assert res.consensus_sufficient is False
    assert res.consensus_stance == "unknown"


def test_6_quoted_claim_does_not_confirm():
    """Quoted allegations (attribution_type='quoted_claim') cannot count as decisive support."""
    claim = Claim(id=1, text="Secret base discovered under mountain.", checkable=True)
    items = [
        EvidenceItem(
            id="ev-q1",
            claim_summary="Secret base",
            evidence_text="Activist claims secret military base exists beneath the mountain ridge.",
            source_name="Local Gazette",
            source_domain="localgazette.com",
            retrieval_source="live_gdelt",
        ),
        EvidenceItem(
            id="ev-q2",
            claim_summary="Secret base",
            evidence_text="Blogger asserts underground tunnels were built in secret.",
            source_name="Blogosphere",
            source_domain="blogosphere.org",
            retrieval_source="live_gdelt",
        ),
    ]
    assessments = [
        EvidenceItemAssessment(
            claim_index=0,
            evidence_id="ev-q1",
            stance=EvidenceStance.supports,
            attribution_type=AttributionType.quoted_claim,
            context_sufficient=True,
            short_reason="Quotes activist claim.",
        ),
        EvidenceItemAssessment(
            claim_index=0,
            evidence_id="ev-q2",
            stance=EvidenceStance.supports,
            attribution_type=AttributionType.quoted_claim,
            context_sufficient=True,
            short_reason="Quotes blogger assertion.",
        ),
    ]

    engine = EvidenceConsensusEngine(min_clusters=2, min_domains=2)
    res = engine.evaluate_claim_consensus(claim, 0, items, assessments)

    assert res.excluded_unclear_count == 2
    assert res.consensus_sufficient is False
    assert res.consensus_stance == "unknown"


def test_7_reported_denial_cannot_count_as_support():
    """Reported denials cannot count as support even if model marked stance=supports."""
    claim = Claim(id=1, text="Minister resigned over scandal.", checkable=True)
    items = [
        EvidenceItem(
            id="ev-d1",
            claim_summary="Resignation",
            evidence_text="Minister strongly denies resignation rumors, stating he remains in office.",
            source_name="State Radio",
            source_domain="stateradio.com",
            retrieval_source="live_gdelt",
        ),
    ]
    assessments = [
        EvidenceItemAssessment(
            claim_index=0,
            evidence_id="ev-d1",
            stance=EvidenceStance.supports,
            attribution_type=AttributionType.reported_denial,
            context_sufficient=True,
            short_reason="Mentions resignation in denial context.",
        ),
    ]

    engine = EvidenceConsensusEngine(min_clusters=2, min_domains=2)
    res = engine.evaluate_claim_consensus(claim, 0, items, assessments)

    assert res.excluded_unclear_count == 1
    assert res.consensus_sufficient is False


def test_8_near_duplicate_clustering_across_syndicated_domains():
    """Identical wire text reprinted on 3 different domains must form 1 cluster, failing 2-cluster requirement."""
    claim = Claim(id=1, text="WHO declares health emergency update.", checkable=True)
    syndicated_text = (
        "GENEVA - The World Health Organization issued an updated advisory regarding global seasonal influenza trends."
    )
    items = [
        EvidenceItem(
            id="ev-wire-1",
            claim_summary="WHO update",
            evidence_text=syndicated_text,
            source_name="Outlet Alpha",
            source_domain="alpha-news.com",
            retrieval_source="live_gdelt",
        ),
        EvidenceItem(
            id="ev-wire-2",
            claim_summary="WHO update",
            evidence_text=syndicated_text,
            source_name="Outlet Beta",
            source_domain="beta-post.com",
            retrieval_source="live_gdelt",
        ),
        EvidenceItem(
            id="ev-wire-3",
            claim_summary="WHO update",
            evidence_text=syndicated_text,
            source_name="Outlet Gamma",
            source_domain="gamma-daily.net",
            retrieval_source="live_gdelt",
        ),
    ]
    assessments = [
        EvidenceItemAssessment(
            claim_index=0,
            evidence_id="ev-wire-1",
            stance=EvidenceStance.supports,
            attribution_type=AttributionType.direct_report,
            context_sufficient=True,
            short_reason="Matches wire report.",
        ),
        EvidenceItemAssessment(
            claim_index=0,
            evidence_id="ev-wire-2",
            stance=EvidenceStance.supports,
            attribution_type=AttributionType.direct_report,
            context_sufficient=True,
            short_reason="Matches wire report.",
        ),
        EvidenceItemAssessment(
            claim_index=0,
            evidence_id="ev-wire-3",
            stance=EvidenceStance.supports,
            attribution_type=AttributionType.direct_report,
            context_sufficient=True,
            short_reason="Matches wire report.",
        ),
    ]

    engine = EvidenceConsensusEngine(min_clusters=2, min_domains=2)
    res = engine.evaluate_claim_consensus(claim, 0, items, assessments)

    # All 3 have identical text, so Jaccard similarity is 1.0 -> 1 cluster!
    assert res.cluster_count == 1
    # Despite 3 domains, only 1 independent cluster exists -> fails min_clusters=2
    assert res.consensus_sufficient is False
    assert res.consensus_stance == "unknown"


def test_9_single_source_insufficient():
    """A single high-quality evidence item alone fails the 2-cluster conservative threshold."""
    claim = Claim(id=1, text="Country Z signs peace accord.", checkable=True)
    items = [
        EvidenceItem(
            id="ev-solo",
            claim_summary="Accord",
            evidence_text="Country Z delegates signed the comprehensive peace agreement this morning in Geneva.",
            source_name="Authoritative Press",
            source_domain="authoritativepress.com",
            retrieval_source="live_gdelt",
        ),
    ]
    assessments = [
        EvidenceItemAssessment(
            claim_index=0,
            evidence_id="ev-solo",
            stance=EvidenceStance.supports,
            attribution_type=AttributionType.direct_report,
            context_sufficient=True,
            short_reason="Detailed direct report.",
        ),
    ]

    engine = EvidenceConsensusEngine(min_clusters=2, min_domains=2)
    res = engine.evaluate_claim_consensus(claim, 0, items, assessments)

    assert res.consensus_sufficient is False
    assert res.consensus_stance == "unknown"
    assert res.support_clusters == 1
    assert res.unique_domains == 1


def test_10_multi_claim_document_consensus():
    """Multi-claim document consensus: all claims must be supported for overall support; mixed results fail overall."""
    c1 = Claim(id=1, text="Claim 1: Rate cut announced.", checkable=True)
    c2 = Claim(id=2, text="Claim 2: Inflation fell to 2 percent.", checkable=True)

    items_c1 = [
        EvidenceItem(
            id="c1-1",
            claim_summary="Rate cut",
            evidence_text="Bank cut rates today.",
            source_name="Bank Herald",
            source_domain="bankherald.com",
        ),
        EvidenceItem(
            id="c1-2",
            claim_summary="Rate cut",
            evidence_text="Benchmark rates lowered 25 bps.",
            source_name="Money Wire",
            source_domain="moneywire.net",
        ),
    ]
    items_c2 = [
        EvidenceItem(
            id="c2-1",
            claim_summary="Inflation",
            evidence_text="Inflation rose to 4 percent rather than falling.",
            source_name="Econ Stat",
            source_domain="econstat.gov",
        ),
        EvidenceItem(
            id="c2-2",
            claim_summary="Inflation",
            evidence_text="Official CPI report shows 4.1 percent annual inflation rate.",
            source_name="Price Index",
            source_domain="priceindex.org",
        ),
    ]

    assessments = [
        EvidenceItemAssessment(
            claim_index=0,
            evidence_id="c1-1",
            stance=EvidenceStance.supports,
            attribution_type=AttributionType.direct_report,
            context_sufficient=True,
            short_reason="Supports rate cut.",
        ),
        EvidenceItemAssessment(
            claim_index=0,
            evidence_id="c1-2",
            stance=EvidenceStance.supports,
            attribution_type=AttributionType.direct_report,
            context_sufficient=True,
            short_reason="Supports rate cut.",
        ),
        EvidenceItemAssessment(
            claim_index=1,
            evidence_id="c2-1",
            stance=EvidenceStance.contradicts,
            attribution_type=AttributionType.direct_report,
            context_sufficient=True,
            short_reason="Contradicts inflation claim.",
        ),
        EvidenceItemAssessment(
            claim_index=1,
            evidence_id="c2-2",
            stance=EvidenceStance.contradicts,
            attribution_type=AttributionType.direct_report,
            context_sufficient=True,
            short_reason="Contradicts inflation claim.",
        ),
    ]

    engine = EvidenceConsensusEngine(min_clusters=2, min_domains=2)
    doc_consensus = engine.evaluate_document_consensus(
        claims=[c1, c2],
        evidence_by_claim=[items_c1, items_c2],
        assessments=assessments,
    )

    assert len(doc_consensus.claim_consensus) == 2
    assert doc_consensus.claim_consensus[0].consensus_stance == "supports"
    assert doc_consensus.claim_consensus[1].consensus_stance == "contradicts"
    # Mixed stances across claims -> document conflict
    assert doc_consensus.conflict_detected is True
    assert doc_consensus.overall_stance == "conflicting"
    assert doc_consensus.overall_sufficient is False
