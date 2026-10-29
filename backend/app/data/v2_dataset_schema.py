"""Canonical Harmonized Dataset Schemas for Project P_105 V2.

Defines standardized Pydantic models for cross-dataset ingestion, normalization,
and evaluation split protection.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class TargetLabel(str, Enum):
    """Target 4-class label space for Project P_105 V2 classifier."""
    LIKELY_RELIABLE = "likely_reliable"
    MISLEADING = "misleading"
    LIKELY_FAKE = "likely_fake"
    UNVERIFIABLE = "unverifiable"


class MappingStatus(str, Enum):
    """Audit status of a label mapping decision."""
    ACCEPTED = "accepted"
    AMBIGUOUS = "ambiguous"
    EXCLUDED = "excluded"
    REQUIRES_REVIEW = "requires_review"


class HarmonizedSample(BaseModel):
    """Canonical representation of an audited fact-checking claim across datasets."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: str = Field(description="Unique globally deterministic identifier (e.g. liar_train_00001)")
    source_dataset: str = Field(description="Provenance source dataset: liar, averitec, multifc, fever")
    source_split: str = Field(description="Original split: train, validation, dev, test")
    original_id: str = Field(description="Identifier from the raw source dataset")
    text: str = Field(description="Cleaned, normalized claim text")
    raw_text: str = Field(description="Original unedited text before normalization")
    original_label: str = Field(description="Verbatim label string from the source dataset")
    harmonized_label: Optional[str] = Field(
        default=None,
        description="Target class: likely_reliable, misleading, likely_fake, unverifiable, or None if excluded"
    )
    mapping_status: str = Field(
        default="accepted",
        description="Audit status: accepted, ambiguous, excluded, requires_review"
    )
    mapping_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in mapping validity based on verified source semantics"
    )
    mapping_rule: str = Field(description="Explicit human-audited rule or mapping version identifier")
    domain: Optional[str] = Field(default="unknown", description="Claim domain/category where available")
    speaker: Optional[str] = Field(default=None, description="Original claimant or speaker")
    claim_date: Optional[str] = Field(default=None, description="Claim origination date if documented")
    source_url: Optional[str] = Field(default=None, description="Original source or fact-check article URL")
    has_evidence: bool = Field(default=False, description="Whether dataset provides evidentiary references")
    evidence_reference: Optional[str] = Field(default=None, description="Summary or link to evidence items")
    license_tag: str = Field(description="SPDX license identifier or governance terms")
    is_protected_test: bool = Field(
        default=False,
        description="Whether this sample belongs to a protected test/dev split and must NEVER be used for training"
    )
    text_hash: str = Field(description="Deterministic SHA-256 hash of normalized text for deduplication")
    exclusion_reason: Optional[str] = Field(default=None, description="Reason if sample was excluded from training")
