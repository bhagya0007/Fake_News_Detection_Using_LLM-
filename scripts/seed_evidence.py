"""Evidence seeding and validation script for local evidence knowledge base.

This script ingests verified evidence records from a controlled local JSON/JSONL input file,
validates each record against the EvidenceItem Pydantic contract and strict provenance rules,
and stores valid records into the target evidence store (e.g. data/evidence/evidence_kb.jsonl).

PROVENANCE POLICY:
- Never scrapes websites automatically at runtime.
- For records marked real (is_synthetic=False), enforces valid source_name, source_url (http/https),
  and non-empty evidence_text.
- Rejects missing or invented provenance.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, List, Tuple

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.app.schemas import EvidenceItem, EvidenceStance

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def validate_evidence_record(record_data: dict[str, Any]) -> Tuple[EvidenceItem | None, str | None]:
    """Validate a single raw record against EvidenceItem and strict provenance rules.

    Args:
        record_data: Raw dictionary representation of an evidence candidate.

    Returns:
        A tuple of (validated_item, error_message). If invalid, (None, error_str).
    """
    try:
        # Pydantic schema validation
        item = EvidenceItem.model_validate(record_data)
    except Exception as exc:
        return None, f"Schema validation error: {exc}"

    # Strict text and provenance checks
    if not item.evidence_text.strip():
        return None, "Rejected: evidence_text cannot be empty or whitespace-only."

    if not item.source_name.strip():
        return None, "Rejected: source_name cannot be empty or whitespace-only."

    # If marked as real evidence (not synthetic), enforce strict verifiable provenance
    if not item.is_synthetic:
        if not item.source_url or not (
            item.source_url.startswith("http://") or item.source_url.startswith("https://")
        ):
            return (
                None,
                f"Rejected real evidence (id={item.id}): Missing or invalid source_url. "
                "Real evidence records require a valid http/https provenance URL.",
            )
        if item.source_name.strip().lower() in {"synthetic demo source", "demo", "placeholder", "unknown"}:
            return (
                None,
                f"Rejected real evidence (id={item.id}): Invalid real source_name '{item.source_name}'.",
            )

    return item, None


def ingest_evidence(
    input_file: Path,
    output_file: Path,
    overwrite: bool = False,
) -> Tuple[int, int, List[str]]:
    """Ingest, validate, and write evidence records.

    Args:
        input_file: Path to source JSON or JSONL file.
        output_file: Destination JSONL file.
        overwrite: Whether to overwrite existing destination.

    Returns:
        Tuple of (accepted_count, rejected_count, error_messages).
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    raw_records: List[dict[str, Any]] = []
    text_content = input_file.read_text(encoding="utf-8").strip()
    if not text_content:
        logger.warning(f"Input file is empty: {input_file}")
        return 0, 0, []

    # Parse JSON or JSONL
    if text_content.startswith("["):
        try:
            raw_records = json.loads(text_content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON array from {input_file}: {exc}") from exc
    else:
        for idx, line in enumerate(text_content.splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                raw_records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON on line {idx} of {input_file}: {exc}") from exc

    accepted_items: List[EvidenceItem] = []
    errors: List[str] = []

    for raw in raw_records:
        item, err = validate_evidence_record(raw)
        if item is not None:
            accepted_items.append(item)
        else:
            errors.append(err or "Unknown validation failure.")

    mode = "w" if overwrite else "a"
    with open(output_file, mode, encoding="utf-8") as out:
        for item in accepted_items:
            out.write(item.model_dump_json() + "\n")

    return len(accepted_items), len(errors), errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest and validate evidence records into local KB.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input JSON/JSONL file.")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/evidence/evidence_kb.jsonl",
        help="Path to target evidence KB JSONL file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file rather than appending.",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    logger.info(f"Ingesting evidence from: {input_path}")
    logger.info(f"Target KB: {output_path}")

    try:
        accepted, rejected, errors = ingest_evidence(input_path, output_path, overwrite=args.overwrite)
        logger.info(f"Ingestion complete: {accepted} accepted, {rejected} rejected.")
        if errors:
            logger.warning(f"Rejection reasons ({len(errors)}):")
            for err in errors[:10]:
                logger.warning(f"  - {err}")
            if len(errors) > 10:
                logger.warning(f"  ... and {len(errors) - 10} more.")
    except Exception as exc:
        logger.error(f"Ingestion failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
