"""Deterministic Evidence Consensus and Near-Duplicate Clustering Service (V2.3).

Core Responsibilities:
1. Near-duplicate syndication clustering:
   Groups candidate news items sharing identical or near-identical text (e.g., wire service
   stories syndicated across multiple domains) so that syndicated coverage is counted as
   one corroborating evidence cluster, not multiple independent confirmations.
2. Conservative corroboration gating:
   Requires at least 2 independent evidence clusters from at least 2 distinct publisher domains
   before a claim is considered supported or contradicted.
3. Strict exclusion policies:
   - Headline-only items (evidence_quality == 'headline_only') are non-decision-bearing.
   - Quoted allegations (attribution_type == 'quoted_claim') do not confirm the proposition.
   - Reported denials (attribution_type == 'reported_denial') cannot count as support.
   - Unclear/irrelevant items are excluded from decisive counts.
4. Conflict & Multi-claim resolution:
   - Coexisting supporting and contradicting evidence causes safe abstention (conflicting).
   - If any core checkable claim is contradicted, the document cannot be deemed likely_reliable.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np

from backend.app.schemas import (
    AttributionType,
    Claim,
    DocumentEvidenceConsensus,
    EvidenceConsensusResult,
    EvidenceItem,
    EvidenceItemAssessment,
    EvidenceStance,
)

logger = logging.getLogger(__name__)

SIMILARITY_CLUSTER_THRESHOLD: float = 0.82
JACCARD_CLUSTER_THRESHOLD: float = 0.65


def _normalize_snippet(text: str) -> str:
    """Normalize text for lexical comparison."""
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute token Jaccard similarity between two texts."""
    words_a = set(_normalize_snippet(text_a).split())
    words_b = set(_normalize_snippet(text_b).split())
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    return float(intersection) / float(union) if union > 0 else 0.0


class EvidenceCluster:
    """A cluster of semantically near-identical or syndicated evidence items."""

    def __init__(self, cluster_id: int) -> None:
        self.cluster_id: int = cluster_id
        self.items: List[EvidenceItem] = []
        self.assessments: List[EvidenceItemAssessment] = []

    @property
    def domains(self) -> Set[str]:
        return {it.source_domain for it in self.items if it.source_domain}

    @property
    def representative_text(self) -> str:
        if not self.items:
            return ""
        return self.items[0].evidence_text or self.items[0].title or ""

    def get_consensus_stance(self) -> EvidenceStance:
        """Determine stance of this cluster based on member assessments."""
        if not self.assessments:
            # Fall back to item.stance if present
            stances = [it.stance for it in self.items if it.stance]
        else:
            stances = [a.stance for a in self.assessments]

        if not stances:
            return EvidenceStance.unknown

        supports_count = sum(1 for s in stances if s == EvidenceStance.supports)
        contradicts_count = sum(1 for s in stances if s == EvidenceStance.contradicts)
        partial_count = sum(1 for s in stances if s == EvidenceStance.partial)

        if supports_count > 0 and contradicts_count > 0:
            return EvidenceStance.unclear  # Internal conflict in cluster
        if supports_count > 0:
            return EvidenceStance.supports
        if contradicts_count > 0:
            return EvidenceStance.contradicts
        if partial_count > 0:
            return EvidenceStance.partial
        return EvidenceStance.unclear


class EvidenceClusterer:
    """Clusters evidence items based on near-duplicate text similarity."""

    def __init__(
        self,
        similarity_threshold: float = SIMILARITY_CLUSTER_THRESHOLD,
        custom_encoder: Optional[Callable[[List[str]], np.ndarray]] = None,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.custom_encoder = custom_encoder
        self._encoder_model: Optional[Any] = None

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        if self.custom_encoder is not None:
            return self.custom_encoder(texts)

        if self._encoder_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            except Exception as exc:
                logger.warning("Could not load SentenceTransformer for clustering: %s", exc)
                return np.zeros((len(texts), 1), dtype=np.float32)

        emb = self._encoder_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.array(emb, dtype=np.float32)

    def cluster_evidence(
        self,
        items: Sequence[EvidenceItem],
        assessments_by_id: Optional[Dict[Union[str, int], EvidenceItemAssessment]] = None,
    ) -> List[EvidenceCluster]:
        """Group evidence items into near-duplicate clusters.

        Items that represent identical wire stories or near-duplicate excerpts across
        multiple domains are grouped into the same cluster.
        """
        if not items:
            return []

        if len(items) == 1:
            c = EvidenceCluster(cluster_id=0)
            c.items.append(items[0])
            if assessments_by_id and items[0].id in assessments_by_id:
                c.assessments.append(assessments_by_id[items[0].id])
            return [c]

        texts = [it.evidence_text or it.title or "" for it in items]
        n = len(items)

        # Compute pairwise similarity
        sim_matrix = np.eye(n, dtype=np.float32)

        # Check if semantic encoder is usable
        try:
            embeddings = self._get_embeddings(texts)
            if embeddings.shape[1] > 1:
                # Cosine similarity for normalized vectors is dot product
                sim_matrix = np.dot(embeddings, embeddings.T)
        except Exception as exc:
            logger.warning("Semantic embedding clustering failed, using lexical: %s", exc)

        # Build adjacency graph
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        for i in range(n):
            for j in range(i + 1, n):
                is_near_dup = False
                # 1. Cosine similarity threshold
                if sim_matrix[i, j] >= self.similarity_threshold:
                    is_near_dup = True
                else:
                    # 2. High lexical Jaccard overlap fallback
                    jacc = _jaccard_similarity(texts[i], texts[j])
                    if jacc >= JACCARD_CLUSTER_THRESHOLD:
                        is_near_dup = True

                if is_near_dup:
                    union(i, j)

        # Group by component
        clusters_map: Dict[int, EvidenceCluster] = {}
        cluster_id_counter = 0

        for i in range(n):
            root = find(i)
            if root not in clusters_map:
                clusters_map[root] = EvidenceCluster(cluster_id=cluster_id_counter)
                cluster_id_counter += 1

            cluster = clusters_map[root]
            item = items[i]
            cluster.items.append(item)
            if assessments_by_id and item.id in assessments_by_id:
                cluster.assessments.append(assessments_by_id[item.id])

        return list(clusters_map.values())


class EvidenceConsensusEngine:
    """Orchestrates deterministic corroboration evaluation and conflict detection."""

    def __init__(
        self,
        clusterer: Optional[EvidenceClusterer] = None,
        min_clusters: int = 2,
        min_domains: int = 2,
    ) -> None:
        self.clusterer = clusterer if clusterer is not None else EvidenceClusterer()
        self.min_clusters = min_clusters
        self.min_domains = min_domains

    def evaluate_claim_consensus(
        self,
        claim: Claim,
        claim_index: int,
        evidence_items: Sequence[EvidenceItem],
        assessments: Sequence[EvidenceItemAssessment],
    ) -> EvidenceConsensusResult:
        """Compute conservative deterministic consensus for an individual claim.

        Applies:
        - Headline exclusion (headline-only items cannot be decisive).
        - Quoted-claim restriction (mere allegations are non-confirming).
        - Denial restriction (reported denials cannot count as support).
        - Near-duplicate clustering across syndicated domains.
        - Minimum 2 clusters from 2 distinct domains for decisive stance.
        """
        assessments_by_id: Dict[Union[str, int], EvidenceItemAssessment] = {
            a.evidence_id: a for a in assessments if a.claim_index == claim_index
        }

        excluded_headline_only = 0
        excluded_unclear = 0
        eligible_items: List[EvidenceItem] = []

        for item in evidence_items:
            # 1. Headline-only check
            if item.evidence_quality == "headline_only":
                excluded_headline_only += 1
                continue

            # 2. Assessment check
            ass = assessments_by_id.get(item.id)
            if ass is None:
                # No Groq assessment available; cannot count as decisive
                excluded_unclear += 1
                continue

            # 3. Context sufficiency
            if not ass.context_sufficient:
                excluded_unclear += 1
                continue

            # 4. Quoted claim / Denial restrictions
            if ass.attribution_type == AttributionType.quoted_claim and ass.stance == EvidenceStance.supports:
                # Source merely reports that someone asserted the claim; does not prove proposition
                excluded_unclear += 1
                continue

            if ass.attribution_type == AttributionType.reported_denial and ass.stance == EvidenceStance.supports:
                # Denial cannot count as supporting evidence
                excluded_unclear += 1
                continue

            # 5. Stance validity
            if ass.stance in (EvidenceStance.irrelevant, EvidenceStance.unclear, EvidenceStance.unknown):
                excluded_unclear += 1
                continue

            eligible_items.append(item)

        # Cluster eligible items
        clusters = self.clusterer.cluster_evidence(eligible_items, assessments_by_id)

        support_clusters = 0
        contradiction_clusters = 0
        partial_clusters = 0
        eligible_domains: Set[str] = set()

        for c in clusters:
            c_stance = c.get_consensus_stance()
            if c_stance == EvidenceStance.supports:
                support_clusters += 1
                eligible_domains.update(c.domains)
            elif c_stance == EvidenceStance.contradicts:
                contradiction_clusters += 1
                eligible_domains.update(c.domains)
            elif c_stance == EvidenceStance.partial:
                partial_clusters += 1
                eligible_domains.update(c.domains)

        unique_domains_count = len(eligible_domains)
        cluster_count = len(clusters)

        # Deterministic consensus rules
        conflict_detected = support_clusters > 0 and contradiction_clusters > 0

        consensus_stance = "unknown"
        consensus_sufficient = False

        if conflict_detected:
            consensus_stance = "conflicting"
            consensus_sufficient = False
        elif (
            support_clusters >= self.min_clusters
            and unique_domains_count >= self.min_domains
            and contradiction_clusters == 0
        ):
            consensus_stance = "supports"
            consensus_sufficient = True
        elif (
            contradiction_clusters >= self.min_clusters
            and unique_domains_count >= self.min_domains
            and support_clusters == 0
        ):
            consensus_stance = "contradicts"
            consensus_sufficient = True
        elif (
            partial_clusters >= self.min_clusters
            and unique_domains_count >= self.min_domains
            and support_clusters == 0
            and contradiction_clusters == 0
        ):
            consensus_stance = "partial"
            consensus_sufficient = True
        else:
            consensus_stance = "unknown"
            consensus_sufficient = False

        return EvidenceConsensusResult(
            claim_index=claim_index,
            claim_text=claim.text,
            consensus_stance=consensus_stance,
            consensus_sufficient=consensus_sufficient,
            eligible_evidence_count=len(eligible_items),
            support_clusters=support_clusters,
            contradiction_clusters=contradiction_clusters,
            partial_clusters=partial_clusters,
            unique_domains=unique_domains_count,
            cluster_count=cluster_count,
            excluded_headline_only_count=excluded_headline_only,
            excluded_unclear_count=excluded_unclear,
            conflict_detected=conflict_detected,
            evidence_assessments=list(assessments_by_id.values()),
        )

    def evaluate_document_consensus(
        self,
        claims: Sequence[Claim],
        evidence_by_claim: Sequence[Sequence[EvidenceItem]],
        assessments: Sequence[EvidenceItemAssessment],
    ) -> DocumentEvidenceConsensus:
        """Aggregate per-claim consensus into document-level consensus.

        Multi-claim conservative policy:
        - If any checkable claim has decisive contradictory consensus, the document
          cannot be likely_reliable.
        - If coexisting supported and contradicted claims occur, document is conflicting.
        - If all checkable claims with consensus are supported (and at least one is sufficient),
          document is supported.
        """
        claim_results: List[EvidenceConsensusResult] = []

        for idx, claim in enumerate(claims):
            ev_list = evidence_by_claim[idx] if idx < len(evidence_by_claim) else []
            cr = self.evaluate_claim_consensus(
                claim=claim,
                claim_index=idx,
                evidence_items=ev_list,
                assessments=assessments,
            )
            claim_results.append(cr)

        if not claim_results:
            return DocumentEvidenceConsensus(
                overall_stance="unknown",
                overall_sufficient=False,
                claim_consensus=[],
                total_eligible_clusters=0,
                total_unique_domains=0,
                conflict_detected=False,
                reason="No checkable claims were available for consensus evaluation.",
            )

        total_clusters = sum(c.cluster_count for c in claim_results)
        total_domains = len({
            d for c in claim_results for a in c.evidence_assessments
            for d in [getattr(a, "source_domain", None)] if d
        })
        any_conflict = any(c.conflict_detected for c in claim_results)

        supported_claims = sum(1 for c in claim_results if c.consensus_stance == "supports" and c.consensus_sufficient)
        contradicted_claims = sum(1 for c in claim_results if c.consensus_stance == "contradicts" and c.consensus_sufficient)
        partial_claims = sum(1 for c in claim_results if c.consensus_stance == "partial" and c.consensus_sufficient)

        if supported_claims > 0 and contradicted_claims > 0:
            overall_stance = "conflicting"
            overall_sufficient = False
            any_conflict = True
            reason = (
                f"Multi-claim contradiction conflict detected: {supported_claims} claim(s) corroborated "
                f"while {contradicted_claims} claim(s) directly contradicted by evidence."
            )
        elif contradicted_claims > 0:
            overall_stance = "contradicts"
            overall_sufficient = True
            reason = (
                f"Decisive contradictory evidence consensus established across {contradicted_claims} checkable claim(s)."
            )
        elif supported_claims > 0 and contradicted_claims == 0:
            overall_stance = "supports"
            overall_sufficient = True
            reason = (
                f"Decisive supporting evidence consensus established across {supported_claims} checkable claim(s) "
                f"with no contradicting evidence."
            )
        elif partial_claims > 0 and supported_claims == 0 and contradicted_claims == 0:
            overall_stance = "partial"
            overall_sufficient = True
            reason = (
                f"Partial evidence consensus established across {partial_claims} claim(s)."
            )
        elif any_conflict:
            overall_stance = "conflicting"
            overall_sufficient = False
            reason = "Internal evidence conflict detected within claim evaluation."
        else:
            overall_stance = "unknown"
            overall_sufficient = False
            reason = "Insufficient independent evidence clusters to establish decisive corroboration."

        return DocumentEvidenceConsensus(
            overall_stance=overall_stance,
            overall_sufficient=overall_sufficient,
            claim_consensus=claim_results,
            total_eligible_clusters=total_clusters,
            total_unique_domains=total_domains,
            conflict_detected=any_conflict,
            reason=reason,
        )
