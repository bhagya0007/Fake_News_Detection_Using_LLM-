"""Build precomputed retrieval index and representations for the local evidence KB.

Responsibilities:
1. Load validated evidence records from input KB JSONL.
2. Initialize retrieval backend (sentence_transformer or tfidf_lexical).
3. Generate representations (dense embeddings or sparse TF-IDF matrices).
4. Persist index artifacts to models/evidence/.
5. Verify strict row-to-record alignment.
6. Report accurate statistics: record count, backend, vector dim, artifact size, real build time.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, List

import numpy as np

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.app.schemas import EvidenceItem

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_index(
    kb_path: Path,
    output_dir: Path,
    backend: str = "sentence_transformer",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict[str, Any]:
    """Build and persist index artifacts from a validated KB file.

    Args:
        kb_path: Path to input evidence JSONL.
        output_dir: Destination directory for artifacts (e.g. models/evidence/).
        backend: 'sentence_transformer' or 'tfidf_lexical'.
        model_name: Pretrained SentenceTransformer model identifier.

    Returns:
        Dictionary containing build metadata and statistics.
    """
    if not kb_path.exists():
        raise FileNotFoundError(f"Evidence KB file does not exist: {kb_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and validate records
    start_time = time.perf_counter()
    records: List[EvidenceItem] = []
    text = kb_path.read_text(encoding="utf-8").strip()

    if text:
        for idx, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            data = json.loads(line)
            item = EvidenceItem.model_validate(data)
            records.append(item)

    if not records:
        logger.warning(f"No records found in KB: {kb_path}. Index will be empty.")

    # 2. Build representations based on backend
    vector_dim: int | None = None
    active_backend = backend

    if backend == "sentence_transformer" and records:
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading SentenceTransformer: {model_name}...")
            model = SentenceTransformer(model_name)
            corpus_texts = [f"{r.claim_summary}\n{r.evidence_text}" for r in records]
            logger.info(f"Encoding {len(corpus_texts)} evidence representations...")
            embeddings = model.encode(
                corpus_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            # Alignment check
            assert len(embeddings) == len(records), "Embedding count must match record count exactly!"
            vector_dim = int(embeddings.shape[1])

            # Save embeddings
            emb_file = output_dir / "evidence_embeddings.npy"
            np.save(emb_file, embeddings)
            logger.info(f"Saved embeddings to {emb_file} (shape: {embeddings.shape})")

        except Exception as exc:
            logger.warning(f"SentenceTransformer build failed ({exc}). Falling back to tfidf_lexical.")
            active_backend = "tfidf_lexical"

    if active_backend == "tfidf_lexical" and records:
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer

        logger.info("Building lexical TF-IDF index...")
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        corpus_texts = [f"{r.claim_summary} {r.evidence_text}" for r in records]
        matrix = vectorizer.fit_transform(corpus_texts)
        assert matrix.shape[0] == len(records), "Matrix rows must match record count exactly!"
        vector_dim = int(matrix.shape[1])

        # Save TF-IDF artifacts
        vec_file = output_dir / "tfidf_vectorizer.joblib"
        mat_file = output_dir / "tfidf_matrix.joblib"
        joblib.dump(vectorizer, vec_file)
        joblib.dump(matrix, mat_file)
        logger.info(f"Saved TF-IDF artifacts to {vec_file} and {mat_file}")

    elif not records:
        active_backend = backend

    # 3. Save index metadata
    metadata = {
        "backend": active_backend,
        "model_name": model_name if active_backend == "sentence_transformer" else None,
        "num_records": len(records),
        "vector_dim": vector_dim,
        "source_kb": str(kb_path.resolve()),
        "records": [r.model_dump(mode="json") for r in records],
    }
    meta_file = output_dir / "evidence_index_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    build_duration = time.perf_counter() - start_time

    # Calculate total artifact size
    total_bytes = sum(f.stat().st_size for f in output_dir.iterdir() if f.is_file())

    summary = {
        "num_records": len(records),
        "backend": active_backend,
        "vector_dim": vector_dim,
        "artifact_size_bytes": total_bytes,
        "artifact_size_kb": round(total_bytes / 1024, 2),
        "build_duration_seconds": round(build_duration, 4),
        "output_dir": str(output_dir.resolve()),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build precomputed retrieval index for evidence KB.")
    parser.add_argument(
        "--kb",
        type=str,
        default="data/sample/evidence_demo.jsonl",
        help="Path to input evidence KB JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/evidence",
        help="Directory to persist index artifacts.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["sentence_transformer", "tfidf_lexical"],
        default="sentence_transformer",
        help="Retrieval representation backend.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Pretrained model name for sentence transformers.",
    )

    args = parser.parse_args()
    kb_path = Path(args.kb)
    output_dir = Path(args.output_dir)

    print("======================================================================")
    print("Building Local Evidence Retrieval Index (Step 13)")
    print(f"Source KB   : {kb_path}")
    print(f"Output Dir  : {output_dir}")
    print(f"Backend     : {args.backend}")
    print("======================================================================")

    stats = build_index(
        kb_path=kb_path,
        output_dir=output_dir,
        backend=args.backend,
        model_name=args.model_name,
    )

    print("\n----------------------------------------------------------------------")
    print("Index Build Summary:")
    print(f"  Evidence Records   : {stats['num_records']}")
    print(f"  Retrieval Backend  : {stats['backend']}")
    print(f"  Vector Dimension   : {stats['vector_dim']}")
    print(f"  Total Artifact Size: {stats['artifact_size_kb']} KB ({stats['artifact_size_bytes']} bytes)")
    print(f"  Build Time         : {stats['build_duration_seconds']:.4f} s")
    print("----------------------------------------------------------------------")


if __name__ == "__main__":
    main()
