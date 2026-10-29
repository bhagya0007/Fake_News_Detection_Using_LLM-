"""Real-World Web-Based Live Evidence Retriever using GDELT 2.0 (Step 21 / V2.2+).

Discovers current news coverage candidates from GDELT Context 2.0 and DOC 2.0 fallback
using deterministic progressive query relaxation (4-tier query plan).
Reranks candidates via semantic similarity using sentence-transformers/all-MiniLM-L6-v2.
Preserves article provenance (URL, domain, title, snippet, publication date) while
explicitly setting stance to 'unknown' (GDELT discovers candidates; it is not a truth oracle).
Safe degradation and rate shielding ensure pipeline resilience during network disruptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

import numpy as np

from backend.app.config import (
    DEFAULT_GROQ_MODEL,
    GDELT_DOC_FALLBACK,
    GDELT_DOC_TIMESPAN,
    LIVE_EVIDENCE_ENABLED,
    LIVE_EVIDENCE_MAX_ATTEMPTS,
    LIVE_EVIDENCE_SEARCH_STOP_COUNT,
    LIVE_EVIDENCE_TOP_K,
)
from backend.app.schemas import (
    Claim,
    EvidenceItem,
    EvidenceRetrievalResult,
    EvidenceStance,
)
from backend.app.services.evidence_query_builder import (
    EvidenceQueryBuilder,
    EvidenceQueryPlan,
    EvidenceSearchQuery,
)
from backend.app.services.gdelt_client import GDELTClient

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class SearchAttempt:
    """Telemetry record for a single search attempt in progressive relaxation."""

    strategy: str
    endpoint: str
    query: str
    result_count: int
    status: str
    elapsed_ms: float


@dataclass
class LiveEvidenceSearchTrace:
    """Full execution trace of progressive live evidence retrieval for a claim."""

    claim_text: str
    attempts: List[SearchAttempt] = field(default_factory=list)
    total_raw_count: int = 0
    total_deduped_count: int = 0
    stop_reason: str = "completed"


def canonicalize_url(url: str) -> str:
    """Normalize a URL to detect syndicated duplicates across queries."""
    if not url or not isinstance(url, str):
        return ""
    try:
        parsed = urlparse(url.strip())
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = parsed.path.rstrip("/")
        # Filter out common tracking query parameters
        query_parts: list[str] = []
        if parsed.query:
            for part in parsed.query.split("&"):
                k = part.split("=")[0].lower()
                if not k.startswith("utm_") and k not in ("fbclid", "gclid", "ref", "source"):
                    query_parts.append(part)
        clean_query = "&".join(sorted(query_parts))
        return f"{netloc}{path}?{clean_query}" if clean_query else f"{netloc}{path}"
    except Exception:
        return url.strip().lower()


class LiveEvidenceRetriever:
    """Discovers and reranks real-world news evidence candidates for checkable claims."""

    def __init__(
        self,
        gdelt_client: Optional[GDELTClient] = None,
        query_builder: Optional[EvidenceQueryBuilder] = None,
        top_k: int = LIVE_EVIDENCE_TOP_K,
        enabled: bool = LIVE_EVIDENCE_ENABLED,
        doc_fallback: bool = GDELT_DOC_FALLBACK,
        custom_encoder: Optional[Callable[[List[str]], np.ndarray]] = None,
        search_stop_count: int = LIVE_EVIDENCE_SEARCH_STOP_COUNT,
        max_attempts: int = LIVE_EVIDENCE_MAX_ATTEMPTS,
        doc_timespan: str = GDELT_DOC_TIMESPAN,
    ) -> None:
        self.gdelt_client = gdelt_client if gdelt_client is not None else GDELTClient()
        self.query_builder = query_builder if query_builder is not None else EvidenceQueryBuilder()
        self.top_k = int(top_k)
        self.enabled = bool(enabled)
        self.doc_fallback = bool(doc_fallback)
        self.custom_encoder = custom_encoder
        self.search_stop_count = int(search_stop_count)
        self.max_attempts = int(max_attempts)
        self.doc_timespan = doc_timespan
        self._embedder: Any = None

    def retrieve_for_claim(
        self,
        claim: Union[Claim, str],
        claim_index: Optional[int] = None,
    ) -> Tuple[List[EvidenceItem], Dict[str, Any]]:
        """Discover, deduplicate, and rerank candidate news evidence for a single claim.

        Executes progressive deterministic query relaxation up to max_attempts:
          Tier A: Precise (Context)
          Tier B: Entity + OR concepts (Context)
          Tier C: Relaxed (DOC)
          Tier D: Broad (DOC with wide timespan)

        Stops early once enough unique candidates are found (search_stop_count).
        Aborts gracefully if GDELT rate limits (429), timeouts, or server errors occur.

        Args:
            claim: Target Claim object or claim text string.
            claim_index: Optional sequential claim index.

        Returns:
            Tuple of (top_k_evidence_items, metadata_dict).
        """
        claim_text = claim.text if isinstance(claim, Claim) else str(claim)
        idx = claim.claim_index or claim.id if isinstance(claim, Claim) else claim_index

        metadata: Dict[str, Any] = {
            "query": "",
            "plan_queries": [],
            "status": "success",
            "context_count": 0,
            "doc_count": 0,
            "raw_candidates_count": 0,
            "deduplicated_count": 0,
            "unique_source_domains": 0,
            "attempts_count": 0,
            "stop_reason": "completed",
            "search_trace": [],
            "error": None,
        }

        if not self.enabled:
            metadata["status"] = "skipped"
            metadata["reason"] = "Live evidence discovery is disabled by configuration."
            return [], metadata

        # 1. Deterministic progressive query plan generation
        plan = self.query_builder.build_plan(claim_text)
        metadata["query"] = plan.queries[0].query if plan.queries else ""
        metadata["plan_queries"] = [sq.query for sq in plan.queries]

        if not plan.queries:
            metadata["status"] = "skipped"
            metadata["reason"] = "Claim produced empty search query."
            return [], metadata

        raw_candidates: List[Dict[str, Any]] = []
        search_trace = LiveEvidenceSearchTrace(claim_text=claim_text)
        aborted = False

        # 2. Sequential Progressive Query Execution
        context_count = 0
        doc_count = 0

        for sq in plan.queries[: self.max_attempts]:
            # Early stop check before launching further queries
            if len(self._deduplicate_candidates(raw_candidates)) >= self.search_stop_count:
                search_trace.stop_reason = "early_stop_satisfied"
                break

            # Relaxation rule: only relax context if precise context returned no candidates
            if sq.endpoint_preference == "context" and sq.strategy == "entity_anchor" and context_count > 0:
                continue

            # DOC fallback rule: only execute DOC fallback if candidates so far are insufficient (< top_k)
            if sq.endpoint_preference == "doc" and len(raw_candidates) >= self.top_k:
                break

            endpoint_to_use = sq.endpoint_preference
            if endpoint_to_use == "context" and plan.is_historical:
                endpoint_to_use = "doc"

            t0_attempt = time.perf_counter()
            items_found: List[Dict[str, Any]] = []
            status = "success"
            error_msg: Optional[str] = None

            try:
                if endpoint_to_use == "doc":
                    if not self.doc_fallback and not plan.is_historical:
                        continue
                    items_found = self.gdelt_client.search_doc(
                        sq.query,
                        timespan=sq.timespan or self.doc_timespan,
                    )
                    endpoint_label = "gdelt_doc"
                else:
                    items_found = self.gdelt_client.search_context(
                        sq.query,
                        timespan=sq.timespan,
                    )
                    endpoint_label = "gdelt_context"

                status = getattr(self.gdelt_client, "last_status", "success")
                error_msg = getattr(self.gdelt_client, "last_error", None)

            except Exception as exc:
                logger.warning("GDELT search error for query '%s' on %s: %s", sq.query, endpoint_to_use, exc)
                status = "error"
                error_msg = str(exc)
                metadata["error"] = str(exc)
                endpoint_label = f"gdelt_{endpoint_to_use}"

            elapsed_ms = round((time.perf_counter() - t0_attempt) * 1000, 1)

            # Record attempt telemetry
            attempt = SearchAttempt(
                strategy=sq.strategy,
                endpoint=endpoint_label,
                query=sq.query,
                result_count=len(items_found),
                status=status,
                elapsed_ms=elapsed_ms,
            )
            search_trace.attempts.append(attempt)

            # Safe diagnostic logging (no secrets, no API keys)
            logger.debug(
                "LiveSearchAttempt [claim=%s, strat=%s, ep=%s]: query='%s', count=%d, status=%s in %.1fms",
                idx, sq.strategy, endpoint_label, sq.query, len(items_found), status, elapsed_ms
            )

            if items_found:
                raw_candidates.extend(items_found)
                if endpoint_label == "gdelt_context":
                    context_count += len(items_found)
                else:
                    doc_count += len(items_found)

            # Check rate limiting / server errors: DO NOT fire remaining attempts
            if status in ("rate_limited", "timeout", "server_error", "error"):
                search_trace.stop_reason = f"aborted_{status}"
                metadata["error"] = error_msg or f"GDELT {status}"
                logger.warning("Aborting live evidence progressive attempts due to %s", status)
                aborted = True
                break

            # Early stopping check: stop once enough unique raw candidates exist
            unique_candidates_count = len(self._deduplicate_candidates(raw_candidates))
            if unique_candidates_count >= self.search_stop_count:
                search_trace.stop_reason = "early_stop_satisfied"
                logger.debug(
                    "Early stop satisfied (%d candidates >= threshold %d) after query '%s'",
                    unique_candidates_count, self.search_stop_count, sq.query
                )
                break

        # Tally metrics
        context_count = sum(1 for c in raw_candidates if c.get("retrieval_source") == "gdelt_context")
        doc_count = sum(1 for c in raw_candidates if c.get("retrieval_source") == "gdelt_doc")

        metadata["context_count"] = context_count
        metadata["doc_count"] = doc_count
        metadata["raw_candidates_count"] = len(raw_candidates)
        metadata["attempts_count"] = len(search_trace.attempts)
        metadata["stop_reason"] = search_trace.stop_reason
        metadata["search_trace"] = [
            {
                "strategy": a.strategy,
                "endpoint": a.endpoint,
                "query": a.query,
                "result_count": a.result_count,
                "status": a.status,
                "elapsed_ms": a.elapsed_ms,
            }
            for a in search_trace.attempts
        ]

        if not raw_candidates:
            if metadata.get("error"):
                metadata["status"] = "unavailable"
            else:
                metadata["status"] = "success"  # Zero matches is valid epistemic outcome
            return [], metadata

        # 3. Canonical Deduplication
        deduped = self._deduplicate_candidates(raw_candidates)
        metadata["deduplicated_count"] = len(deduped)

        # 4. Measure source domain diversity
        domains = {c["domain"] for c in deduped if c.get("domain")}
        metadata["unique_source_domains"] = len(domains)

        # 5. Semantic Reranking with MiniLM
        reranked = self._rerank_candidates(claim_text, deduped)

        # 6. Convert to standardized EvidenceItem contracts
        evidence_items: List[EvidenceItem] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for item_idx, c in enumerate(reranked[: self.top_k], start=1):
            claim_id_str = f"claim_{idx}" if idx is not None else "claim"
            evidence_id = f"live_gdelt_{claim_id_str}_{item_idx}"

            item = EvidenceItem(
                id=evidence_id,
                claim_summary=claim_text,
                evidence_text=c.get("snippet") if c.get("snippet") else c.get("title", ""),
                source_name=c.get("domain") or "External News Publisher",
                source_url=c.get("url"),
                source_domain=c.get("domain"),
                title=c.get("title"),
                published_date=c.get("published_date"),
                similarity_score=c.get("similarity_score"),
                stance=EvidenceStance.unknown,  # Strictly unknown in V2.2/V2.3 before LLM reasoning
                source_type=c.get("source_type") or "news",
                is_synthetic=False,
                retrieval_source=c.get("retrieval_source") or "gdelt_context",
                evidence_quality=c.get("evidence_quality") or "context_snippet",
                retrieved_at=now_iso,
            )
            evidence_items.append(item)

        # Filter audit logging
        logger.debug(
            "LiveEvidenceFilterAudit [claim=%s]: raw=%d -> deduped=%d -> reranked=%d -> top_k=%d",
            idx, len(raw_candidates), len(deduped), len(reranked), len(evidence_items)
        )

        metadata["status"] = "success"
        return evidence_items, metadata

    def _deduplicate_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate records by canonical URL, then by (domain, title)."""
        seen_urls: Set[str] = set()
        seen_domain_titles: Set[Tuple[str, str]] = set()
        deduped: List[Dict[str, Any]] = []

        for c in candidates:
            url = c.get("url", "")
            canon_url = canonicalize_url(url)
            domain = str(c.get("domain", "")).lower()
            title = str(c.get("title", "")).lower()

            if canon_url and canon_url in seen_urls:
                continue

            if domain and title and (domain, title) in seen_domain_titles:
                continue

            if canon_url:
                seen_urls.add(canon_url)
            if domain and title:
                seen_domain_titles.add((domain, title))

            deduped.append(c)

        return deduped

    def _rerank_candidates(
        self,
        claim_text: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Compute cosine similarity between claim and candidate snippets/titles."""
        if not candidates:
            return []

        doc_texts = [
            c.get("snippet") if c.get("snippet") else c.get("title", "")
            for c in candidates
        ]

        try:
            # 1. Custom encoder callable if supplied (e.g., during fast unit testing)
            if self.custom_encoder is not None:
                all_embeddings = self.custom_encoder([claim_text] + doc_texts)
                q_vec = all_embeddings[0]
                d_vecs = all_embeddings[1:]
            else:
                # 2. SentenceTransformer all-MiniLM-L6-v2 local model
                self._ensure_embedder()
                all_embeddings = self._embedder.encode(
                    [claim_text] + doc_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                q_vec = all_embeddings[0]
                d_vecs = all_embeddings[1:]

            # Cosine similarity (already normalized if from SentenceTransformer)
            q_norm = np.linalg.norm(q_vec)
            d_norms = np.linalg.norm(d_vecs, axis=1)
            d_norms[d_norms == 0] = 1e-10

            if q_norm == 0:
                sims = np.zeros(len(doc_texts))
            else:
                sims = np.dot(d_vecs, q_vec) / (d_norms * q_norm)

            for idx, c in enumerate(candidates):
                score = float(sims[idx])
                # Clip to [0.0, 1.0] and round for telemetry
                c["similarity_score"] = round(max(0.0, min(1.0, score)), 4)

            # Sort descending by similarity
            return sorted(candidates, key=lambda x: x.get("similarity_score", 0.0), reverse=True)

        except Exception as exc:
            logger.warning("MiniLM semantic reranking failed; using discovery order: %s", exc)
            for c in candidates:
                if "similarity_score" not in c:
                    c["similarity_score"] = None
            return candidates

    def _ensure_embedder(self) -> None:
        """Lazy load sentence-transformers/all-MiniLM-L6-v2 embedder."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(DEFAULT_MODEL_NAME)
