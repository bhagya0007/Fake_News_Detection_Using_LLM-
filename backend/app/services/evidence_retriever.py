"""Local Evidence Knowledge Base and Semantic Retrieval Service (Step 13).

This module provides a local, trustworthy evidence retrieval layer for checkable claims.
It supports:
1. Loading and validating local evidence records (with provenance).
2. Semantic similarity retrieval using sentence-transformers (all-MiniLM-L6-v2) or
   lexical fallback (TF-IDF + cosine similarity).
3. Enforcing strict engineering similarity thresholds.
4. Returning 'no sufficient evidence' when candidates fall below threshold.
5. Strict separation between semantic similarity and evidentiary stance.
6. Zero runtime browsing, scraping, or hallucinated evidence.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from backend.app.schemas import (
    Claim,
    EvidenceItem,
    EvidenceRetrievalResult,
    EvidenceStance,
)

logger = logging.getLogger(__name__)

# Default engineering configurations (not calibrated on LIAR test set)
DEFAULT_THRESHOLD: float = 0.45
DEFAULT_TOP_K: int = 3
MAX_TOP_K: int = 5
DEFAULT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"


class EvidenceRetriever:
    """Local evidence retrieval engine supporting semantic and lexical backends."""

    def __init__(
        self,
        kb_path: Optional[Union[str, Path]] = None,
        index_dir: Optional[Union[str, Path]] = None,
        backend: str = "auto",
        threshold: float = DEFAULT_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
        max_top_k: int = MAX_TOP_K,
        custom_encoder: Optional[Callable[[List[str]], np.ndarray]] = None,
    ) -> None:
        """Initialize the EvidenceRetriever.

        Args:
            kb_path: Path to the raw evidence JSONL file (e.g. data/evidence/evidence_kb.jsonl).
            index_dir: Directory containing precomputed index artifacts (e.g. models/evidence/).
            backend: Desired backend: 'sentence_transformer', 'tfidf_lexical', or 'auto'.
            threshold: Minimum similarity score required to retain an evidence item [0.0, 1.0].
            top_k: Default number of top results to return.
            max_top_k: Hard ceiling on returned results.
            custom_encoder: Optional custom encoding callable for testing/mocking.
        """
        self.kb_path = Path(kb_path) if kb_path else None
        self.index_dir = Path(index_dir) if index_dir else None
        self.threshold = float(threshold)
        self.top_k = int(top_k)
        self.max_top_k = int(max_top_k)
        self.custom_encoder = custom_encoder

        self.records: List[EvidenceItem] = []
        self.backend_name: str = "uninitialized"
        self._sentence_transformer_model: Any = None
        self._tfidf_vectorizer: Any = None
        self._tfidf_matrix: Any = None
        self._embeddings: Optional[np.ndarray] = None

        # Resolve and initialize backend
        self._initialize(requested_backend=backend)

    def _initialize(self, requested_backend: str) -> None:
        """Load records and initialize the retrieval backend."""
        # 1. Load records from index_dir metadata or raw kb_path
        loaded_from_index = False
        if self.index_dir and self.index_dir.exists():
            meta_path = self.index_dir / "evidence_index_metadata.json"
            if meta_path.exists():
                self._load_from_index_dir()
                loaded_from_index = True

        if not loaded_from_index and self.kb_path:
            self._load_from_kb_path()

        # 2. Determine backend
        if self.custom_encoder is not None:
            self.backend_name = "custom_encoder"
            self._build_custom_embeddings()
        elif loaded_from_index:
            # Backend already set by _load_from_index_dir
            pass
        elif requested_backend in {"sentence_transformer", "auto"}:
            try:
                from sentence_transformers import SentenceTransformer

                self._sentence_transformer_model = SentenceTransformer(DEFAULT_MODEL_NAME)
                self.backend_name = "sentence_transformer"
                self._build_sentence_transformer_embeddings()
            except Exception as exc:
                if requested_backend == "sentence_transformer":
                    logger.warning(
                        f"SentenceTransformer requested but unavailable: {exc}. Falling back to lexical TF-IDF."
                    )
                self._init_tfidf_backend()
        else:
            self._init_tfidf_backend()

    def _load_from_kb_path(self) -> None:
        """Load and validate records from raw JSONL file."""
        if not self.kb_path or not self.kb_path.exists():
            self.records = []
            return

        records: List[EvidenceItem] = []
        text = self.kb_path.read_text(encoding="utf-8").strip()
        if not text:
            self.records = []
            return

        for line_num, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                data = json.loads(line)
                item = EvidenceItem.model_validate(data)
                records.append(item)
            except Exception as exc:
                raise ValueError(f"Malformed evidence record on line {line_num} of {self.kb_path}: {exc}") from exc

        self.records = records

    def _load_from_index_dir(self) -> None:
        """Load metadata and representations from prebuilt index directory."""
        meta_path = self.index_dir / "evidence_index_metadata.json"  # type: ignore[operator]
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.backend_name = metadata.get("backend", "unknown")
        raw_records = metadata.get("records", [])
        self.records = [EvidenceItem.model_validate(r) for r in raw_records]

        if self.backend_name == "sentence_transformer":
            emb_path = self.index_dir / "evidence_embeddings.npy"  # type: ignore[operator]
            if not emb_path.exists():
                raise FileNotFoundError(f"Missing embeddings file: {emb_path}")
            self._embeddings = np.load(emb_path)
            if len(self._embeddings) != len(self.records):
                raise ValueError(
                    f"Index alignment error: {len(self._embeddings)} embeddings != {len(self.records)} records."
                )
            # Load sentence transformer model for query encoding
            try:
                from sentence_transformers import SentenceTransformer

                self._sentence_transformer_model = SentenceTransformer(
                    metadata.get("model_name", DEFAULT_MODEL_NAME)
                )
            except Exception as exc:
                logger.warning(f"Could not load SentenceTransformer for inference: {exc}. Reverting to lexical.")
                self._init_tfidf_backend()

        elif self.backend_name == "tfidf_lexical":
            import joblib

            vec_path = self.index_dir / "tfidf_vectorizer.joblib"  # type: ignore[operator]
            mat_path = self.index_dir / "tfidf_matrix.joblib"  # type: ignore[operator]
            if not vec_path.exists() or not mat_path.exists():
                raise FileNotFoundError("Missing TF-IDF index artifacts in index directory.")
            self._tfidf_vectorizer = joblib.load(vec_path)
            self._tfidf_matrix = joblib.load(mat_path)
            if self._tfidf_matrix.shape[0] != len(self.records):
                raise ValueError(
                    f"Index alignment error: matrix rows {self._tfidf_matrix.shape[0]} != {len(self.records)} records."
                )

    def _build_custom_embeddings(self) -> None:
        """Build representations using injected custom encoder."""
        if not self.records or self.custom_encoder is None:
            self._embeddings = None
            return
        texts = [f"{r.claim_summary}\n{r.evidence_text}" for r in self.records]
        embeddings = self.custom_encoder(texts)
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._embeddings = embeddings / norms

    def _build_sentence_transformer_embeddings(self) -> None:
        """Compute embeddings for all KB records using SentenceTransformer."""
        if not self.records or self._sentence_transformer_model is None:
            self._embeddings = None
            return
        texts = [f"{r.claim_summary}\n{r.evidence_text}" for r in self.records]
        embeddings = self._sentence_transformer_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self._embeddings = embeddings

    def _init_tfidf_backend(self) -> None:
        """Initialize lexical TF-IDF fallback backend using scikit-learn."""
        self.backend_name = "tfidf_lexical"
        if not self.records:
            self._tfidf_vectorizer = None
            self._tfidf_matrix = None
            return

        from sklearn.feature_extraction.text import TfidfVectorizer

        self._tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        corpus = [f"{r.claim_summary} {r.evidence_text}" for r in self.records]
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(corpus)

    def retrieve(
        self,
        claim_text: str,
        top_k: Optional[int] = None,
    ) -> EvidenceRetrievalResult:
        """Retrieve top-k relevant evidence items for an individual claim.

        STRICT SAFETY & PROVENANCE GUARANTEES:
        - Validates query: rejects empty, whitespace-only, None, or non-string input.
        - High similarity score does NOT alter evidence stance (similarity != stance).
        - If no records meet the threshold, returns empty items with no_sufficient_evidence=True.
        - Caps returned results at max_top_k.

        Args:
            claim_text: The atomic checkable claim string.
            top_k: Number of results requested (capped by max_top_k).

        Returns:
            EvidenceRetrievalResult containing retrieved items and audit metadata.
        """
        # Validate query
        if claim_text is None or not isinstance(claim_text, str):
            raise TypeError(f"claim_text must be a non-empty string (received {type(claim_text).__name__}).")

        query = claim_text.strip()
        if not query:
            raise ValueError("claim_text cannot be empty or whitespace-only.")

        effective_top_k = min(top_k if top_k is not None else self.top_k, self.max_top_k)
        if effective_top_k <= 0:
            effective_top_k = 1

        # Handle empty KB
        if not self.records:
            return EvidenceRetrievalResult(
                query=query,
                items=[],
                retrieval_backend=self.backend_name,
                threshold=self.threshold,
                no_sufficient_evidence=True,
            )

        # Compute similarity scores
        scores: np.ndarray
        if self.backend_name in {"sentence_transformer", "custom_encoder"}:
            if self.backend_name == "custom_encoder" and self.custom_encoder:
                q_vec = self.custom_encoder([query])
                q_norm = np.linalg.norm(q_vec, axis=1, keepdims=True)
                q_norm[q_norm == 0] = 1.0
                q_vec = q_vec / q_norm
            else:
                q_vec = self._sentence_transformer_model.encode(
                    [query], convert_to_numpy=True, normalize_embeddings=True
                )
            # Cosine similarity between normalized vectors is dot product
            scores = np.dot(self._embeddings, q_vec.T).flatten()  # type: ignore[arg-type]

        elif self.backend_name == "tfidf_lexical":
            from sklearn.metrics.pairwise import cosine_similarity

            q_vec = self._tfidf_vectorizer.transform([query])
            scores = cosine_similarity(self._tfidf_matrix, q_vec).flatten()
        else:
            raise RuntimeError(f"Unrecognized retrieval backend: {self.backend_name}")

        # Pair scores with records and apply threshold
        candidates: List[Tuple[float, int, EvidenceItem]] = []
        for idx, (score, rec) in enumerate(zip(scores, self.records)):
            # Clamp numerical precision to [0.0, 1.0]
            bounded_score = float(max(0.0, min(1.0, float(score))))
            if bounded_score >= self.threshold:
                # Create a fresh EvidenceItem with the computed similarity score.
                # PRESERVE STANCE EXACTLY - NEVER CHANGE STANCE TO SUPPORTS/CONTRADICTS
                item_copy = EvidenceItem(
                    id=rec.id,
                    claim_summary=rec.claim_summary,
                    evidence_text=rec.evidence_text,
                    source_name=rec.source_name,
                    source_url=rec.source_url,
                    published_date=rec.published_date,
                    similarity_score=round(bounded_score, 4),
                    stance=rec.stance,  # Unaltered stance
                    source_type=rec.source_type,
                    is_synthetic=rec.is_synthetic,
                )
                candidates.append((bounded_score, idx, item_copy))

        # Sort descending by score, tie-break by index for determinism
        candidates.sort(key=lambda x: (-x[0], x[1]))

        # Select top-k
        top_candidates = [item for _, _, item in candidates[:effective_top_k]]

        if not top_candidates:
            return EvidenceRetrievalResult(
                query=query,
                items=[],
                retrieval_backend=self.backend_name,
                threshold=self.threshold,
                no_sufficient_evidence=True,
            )

        return EvidenceRetrievalResult(
            query=query,
            items=top_candidates,
            retrieval_backend=self.backend_name,
            threshold=self.threshold,
            no_sufficient_evidence=False,
        )

    def retrieve_for_claims(
        self,
        claims: Union[List[Claim], List[str]],
        top_k_per_claim: int = DEFAULT_TOP_K,
    ) -> Dict[str, List[EvidenceItem]]:
        """Retrieve relevant evidence items grouped per claim, avoiding duplicates per claim.

        Args:
            claims: List of Claim objects or raw claim strings.
            top_k_per_claim: Number of top evidence items to retrieve per claim.

        Returns:
            Dictionary mapping claim query string to list of retrieved EvidenceItems.
        """
        results: Dict[str, List[EvidenceItem]] = {}

        for claim in claims:
            if isinstance(claim, Claim):
                query = claim.text
            elif isinstance(claim, str):
                query = claim
            else:
                continue

            stripped = query.strip()
            if not stripped:
                continue

            retrieval = self.retrieve(stripped, top_k=top_k_per_claim)
            # Deduplicate by item id or content within this claim's results
            seen_ids = set()
            deduped_items = []
            for item in retrieval.items:
                identifier = item.id if item.id is not None else item.evidence_text
                if identifier not in seen_ids:
                    seen_ids.add(identifier)
                    deduped_items.append(item)

            results[stripped] = deduped_items

        return results
