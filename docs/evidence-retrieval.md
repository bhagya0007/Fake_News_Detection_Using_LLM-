# Evidence Knowledge Base & Semantic Retrieval Layer

This document details the design, storage schema, retrieval mechanics, and provenance rules for the local evidence retrieval subsystem implemented in Step 13.

---

## 1. Purpose & Architectural Role

The Evidence Retrieval Layer (`backend/app/services/evidence_retriever.py`) provides a local, traceable, and trustworthy source of reference documents to ground downstream LLM reasoning.

```text
Extracted Claim
      ↓
EvidenceRetriever
      ↓
Local Evidence KB (JSONL + Precomputed Representations)
      ↓
Top-k Relevant Evidence (Filtered by Threshold)
      ↓
LLM Reasoner
```

### Critical Epistemic Separation: Semantic Similarity ≠ Factual Support
A core principle of this architecture is that **semantic similarity is completely orthogonal to evidentiary stance**:
- **High Semantic Similarity** indicates only that an evidence item discusses the same entity, topic, or subject matter as the claim.
- **Factual Stance** (`supports`, `contradicts`, `related`, `unknown`) indicates whether the text empirically substantiates or refutes the proposition.
- Under no circumstances does the retriever alter recorded stance or infer truth from similarity scores alone.

---

## 2. Knowledge Base Storage & Format

The evidence knowledge base is maintained locally in human-readable JSON Lines format:
- **Production Store:** `data/evidence/evidence_kb.jsonl`
- **Synthetic Test / Demo Fixtures:** `data/sample/evidence_demo.jsonl`

### Schema Structure (`EvidenceItem`)
Each line represents an independent, validated JSON record conforming strictly to `EvidenceItem`:

```json
{
  "id": "synth_001",
  "claim_summary": "Acme Corp announced a new factory in Texas",
  "evidence_text": "Acme Corp announced today that it will construct a new manufacturing facility in Texas to expand clean energy production.",
  "source_name": "Synthetic Demo Source",
  "source_url": "https://example.com/synthetic/acme-texas",
  "published_date": "2024-03-15",
  "stance": "supports",
  "source_type": "press_release",
  "is_synthetic": true
}
```

### Artifact Storage
High-dimensional dense vectors and index matrices are saved separately in `models/evidence/` to keep the source JSONL human-readable:
- `models/evidence/evidence_embeddings.npy`: Precomputed dense embedding vectors (shape: $N \times D$, $D=384$ for `all-MiniLM-L6-v2`).
- `models/evidence/evidence_index_metadata.json`: Serialized metadata ensuring exact row-to-record alignment.
- Optional lexical fallback artifacts: `tfidf_vectorizer.joblib` and `tfidf_matrix.joblib`.

---

## 3. Strict Provenance Rules

To maintain high evidentiary integrity and prevent the hallucination of truth:
1. **Traceable Real Sources:** Any record marked real (`is_synthetic=False`) must possess a valid, verifiable `source_name` and a functioning `http://` or `https://` `source_url`.
2. **Zero Hallucinated Attribution:** Creating fictitious citations (such as attributing arbitrary facts to *"NASA"* or *"Reuters"* without verified underlying documents) is strictly prohibited.
3. **Explicit Synthetic Marking:** Functional tests and demonstrations utilize fixtures explicitly tagged with `source_name="Synthetic Demo Source"` and `is_synthetic=True`. Synthetic records are never mixed into research benchmark evaluations.

---

## 4. Ingestion & Validation Pipeline (`scripts/seed_evidence.py`)

Ingestion of external evidence is handled by `scripts/seed_evidence.py`:
- Validates records against the strict Pydantic contract before disk persistence.
- Enforces non-empty `evidence_text` and `source_name`.
- Rejects malformed stances and records with missing provenance.
- Never scrapes websites or queries live search engines at runtime.

---

## 5. Retrieval Backends

### Primary: Semantic Embeddings (`sentence-transformers`)
- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
- **Environment:** Tested and verified compatible with Python 3.14.7 (`torch 2.14.1+cpu`, `sentence-transformers 6.1.0`).
- **Metric:** Cosine similarity via normalized dot product.
- **Resource Footprint:** Small CPU-friendly model (~80 MB) easily runnable on an 8 GB laptop.

### Fallback: Lexical TF-IDF (`scikit-learn`)
- **Vectorizer:** Word sublinear TF-IDF (1-2 ngrams).
- **Metric:** Sparse cosine similarity.
- **Role:** Fallback if PyTorch/HuggingFace libraries are unavailable in constrained execution environments.

---

## 6. Threshold & Top-K Retrieval Policies

### Configurable Engineering Threshold
- **Default Threshold:** `EVIDENCE_SIMILARITY_THRESHOLD = 0.45`.
- *Note:* This is an engineering heuristic chosen to separate topical matches from background noise on demo fixtures. It has not yet been scientifically calibrated and is never tuned on official benchmark test data (`data/raw/test.tsv`).

### No-Sufficient-Evidence Guarantee
If no records in the knowledge base clear the configured similarity threshold:
- `items = []`
- `no_sufficient_evidence = True`
The retriever never returns the nearest poor match simply because a vector dot product exists mathematically.

### Top-K Bounds
- **Default:** `top_k = 3`.
- **Maximum Ceiling:** `max_top_k = 5`.
Restricts context size passed to downstream LLM reasoning components.

---

## 7. Operational Limitations

1. **No Runtime Web Browsing:** The retriever searches only within its local validated KB. It does not perform live Google queries, web scraping, or RSS ingestion.
2. **Local Coverage Bound:** Real-world claim verification is strictly limited by the coverage of the seeded local knowledge base.
3. **Non-Decisional:** The retriever does not judge whether a claim is true or fake. It solely delivers relevant evidence candidates with similarity scores.
