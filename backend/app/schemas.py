"""Structured Data Schemas and Validated Contracts for the Hybrid AI System (Step 9).

This module defines Pydantic v2 schemas and strict enums to serve as authoritative
data contracts across all components of the multi-signal fake news detection pipeline:
- API Request & Response validation
- Red-flag stylometric signals
- Classical ML classification outputs
- Claim extraction representations
- Evidence retrieval items
- Structured LLM assessments (strictly no hidden chain-of-thought)
- Hybrid fusion and abstention outputs
- User feedback

PRINCIPLES:
1. Strict type checking and value validation using Pydantic v2.
2. Distinct label domains:
   - Classical ML (MLLabel): ['likely_reliable', 'misleading', 'likely_fake'] (strictly no 'unverifiable')
   - Hybrid / Final System (VerdictLabel): ['likely_reliable', 'misleading', 'likely_fake', 'unverifiable']
3. Consistency validation: 'unverifiable' corresponds strictly to 'abstained == True'.
4. Concise and bounded textual explanations; unrestricted reasoning dumps are prohibited.
5. Adapters provided to bridge Step 5–8 service outputs without modifying frozen code.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from backend.app.config import MAX_ANALYSIS_TEXT_LENGTH

# ---------------------------------------------------------------------------
# Global Constants & Thresholds
# ---------------------------------------------------------------------------
PROBABILITY_SUM_TOLERANCE: float = 0.01


# ---------------------------------------------------------------------------
# Strict Enums
# ---------------------------------------------------------------------------
class VerdictLabel(str, Enum):
    """Authoritative project-level verdict labels for the hybrid AI system."""

    likely_reliable = "likely_reliable"
    misleading = "misleading"
    likely_fake = "likely_fake"
    unverifiable = "unverifiable"


class MLLabel(str, Enum):
    """Labels supported by the classical ML classifier (strictly no unverifiable)."""

    likely_reliable = "likely_reliable"
    misleading = "misleading"
    likely_fake = "likely_fake"


class Severity(str, Enum):
    """Linguistic intensity severity for warning flags (not probability of falsehood)."""

    low = "low"
    medium = "medium"
    high = "high"


class SourceAttribution(str, Enum):
    """Evaluation of source attribution specificity and traceability for statements or claims."""

    strong = "strong"
    partial = "partial"
    weak = "weak"
    none = "none"
    unknown = "unknown"


class EvidenceSupport(str, Enum):
    """Categorization of external evidence support for a claim or proposition."""

    supported = "supported"
    contradicted = "contradicted"
    partial = "partial"
    unavailable = "unavailable"
    not_applicable = "not_applicable"


class InternalConsistency(str, Enum):
    """Assessment of logical coherence strictly within supplied material (not external plausibility)."""

    consistent = "consistent"
    questionable = "questionable"
    contradictory = "contradictory"
    unknown = "unknown"


class ClaimAssessment(str, Enum):
    """Per-claim verification assessment status."""

    supported = "supported"
    contradicted = "contradicted"
    partially_supported = "partially_supported"
    unverifiable = "unverifiable"
    not_checked = "not_checked"


class EvidenceStance(str, Enum):
    """Stance of retrieved evidence relative to the target claim."""

    supports = "supports"
    contradicts = "contradicts"
    partial = "partial"
    irrelevant = "irrelevant"
    unclear = "unclear"
    related = "related"
    unknown = "unknown"


class AttributionType(str, Enum):
    """Attribution relationship between retrieved source text and factual proposition."""

    direct_report = "direct_report"
    quoted_claim = "quoted_claim"
    reported_denial = "reported_denial"
    reported_correction = "reported_correction"
    secondary_reference = "secondary_reference"
    unclear = "unclear"


class ComponentExecutionStatus(str, Enum):
    """Operational status of an individual pipeline component."""

    success = "success"
    skipped = "skipped"
    unavailable = "unavailable"
    failed = "failed"


# ---------------------------------------------------------------------------
# Core Input Request Schema
# ---------------------------------------------------------------------------
class AnalysisRequest(BaseModel):
    """Inbound statement analysis request with strict sanitization and validation."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(
        ...,
        description="The statement or news claim text to evaluate.",
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Analysis text cannot be empty or whitespace-only.")
        if len(stripped) > MAX_ANALYSIS_TEXT_LENGTH:
            raise ValueError(
                f"Input text exceeds maximum allowed length of {MAX_ANALYSIS_TEXT_LENGTH:,} characters "
                f"(received {len(stripped):,} characters)."
            )
        return stripped


# ---------------------------------------------------------------------------
# Claim Extraction Schemas
# ---------------------------------------------------------------------------
class ClaimType(str, Enum):
    """Categorization of factual proposition types for checkable claims."""

    numeric = "numeric"
    event = "event"
    attribution = "attribution"
    causal = "causal"
    existence = "existence"
    other = "other"


class Claim(BaseModel):
    """Checkable atomic proposition extracted from a statement."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: Optional[int] = Field(default=None, ge=0, description="Sequential claim identifier.")
    claim_index: Optional[int] = Field(default=None, ge=0, description="Sequential claim index.")
    text: str = Field(..., min_length=1, max_length=5000, description="Atomic checkable proposition.")
    checkable: bool = Field(default=True, description="Whether the proposition is empirically checkable.")
    claim_type: Optional[ClaimType] = Field(default=None, description="Categorization of claim type.")
    source_span: Optional[str] = Field(
        default=None, max_length=500, description="Original text excerpt containing this claim."
    )

    @field_validator("text")
    @classmethod
    def validate_claim_text(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Claim text cannot be empty or whitespace-only.")
        return stripped


class ClaimExtractionResult(BaseModel):
    """Structured extraction payload containing up to 5 checkable claims.

    STRICT GUARANTEE:
    Contains NO truthfulness verdict, credibility classification, or hidden chain-of-thought.
    """

    model_config = ConfigDict(extra="forbid")

    claims: list[Claim] = Field(
        default_factory=list,
        max_length=5,
        description="Extracted checkable claims (at most 5).",
    )
    source_text_length: int = Field(
        default=0,
        ge=0,
        description="Character count of cleaned source text evaluated.",
    )
    truncated: bool = Field(
        default=False,
        description="Whether source text was truncated due to length bounds.",
    )
    prompt_version: Optional[str] = Field(
        default=None,
        description="Identifier of the system prompt version used for extraction.",
    )

    @field_validator("claims")
    @classmethod
    def validate_max_claims(cls, v: list[Claim]) -> list[Claim]:
        if len(v) > 5:
            raise ValueError(f"Extracted claims cannot exceed 5 items (received {len(v)}).")
        return v


# ---------------------------------------------------------------------------
# Red-Flag / Stylometric Schemas (Step 8 Integration)
# ---------------------------------------------------------------------------
class RedFlag(BaseModel):
    """Individual linguistic or formatting warning flag emitted by the stylometric layer."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(..., min_length=1, description="Machine-readable rule identifier.")
    severity: Severity = Field(..., description="Linguistic intensity level.")
    message: str = Field(..., min_length=1, description="Human-readable explanation of the cue.")
    snippet: Optional[str] = Field(default=None, description="Matched substring or trigger excerpt.")
    value: Optional[Union[int, float, str]] = Field(
        default=None, description="Quantitative signal value (count, ratio, or identifier)."
    )


class RedFlagFeatures(BaseModel):
    """The 15 quantitative stylometric features extracted by Step 8."""

    model_config = ConfigDict(extra="forbid")

    character_count: int = Field(..., ge=0)
    word_count: int = Field(..., ge=0)
    uppercase_letter_ratio: float = Field(..., ge=0.0, le=1.0)
    exclamation_count: int = Field(..., ge=0)
    question_mark_count: int = Field(..., ge=0)
    repeated_punctuation_count: int = Field(..., ge=0)
    percentage_count: int = Field(..., ge=0)
    statistical_expression_count: int = Field(..., ge=0)
    url_count: int = Field(..., ge=0)
    sensational_phrase_count: int = Field(..., ge=0)
    certainty_phrase_count: int = Field(..., ge=0)
    urgency_phrase_count: int = Field(..., ge=0)
    attribution_phrase_count: int = Field(..., ge=0)
    quote_count: int = Field(..., ge=0)
    excessive_punctuation_ratio: float = Field(..., ge=0.0, le=1.0)


class RedFlagSummary(BaseModel):
    """High-level summary of triggered stylometric flags."""

    model_config = ConfigDict(extra="forbid")

    total_flags: int = Field(..., ge=0)
    high_severity_flags: int = Field(..., ge=0)
    medium_severity_flags: int = Field(..., ge=0)
    low_severity_flags: int = Field(..., ge=0)
    has_attribution: bool = Field(default=False)
    has_statistical_claim: bool = Field(default=False)


class RedFlagResult(BaseModel):
    """Complete result object emitted by the red-flag detection service."""

    model_config = ConfigDict(extra="forbid")

    flags: list[RedFlag] = Field(default_factory=list)
    features: RedFlagFeatures
    summary: RedFlagSummary


# ---------------------------------------------------------------------------
# Classical ML Assessment Schemas
# ---------------------------------------------------------------------------
class ClassProbabilities(BaseModel):
    """Raw model probability estimates across the 3 classical benchmark classes."""

    model_config = ConfigDict(extra="forbid")

    likely_fake: float = Field(..., ge=0.0, le=1.0)
    likely_reliable: float = Field(..., ge=0.0, le=1.0)
    misleading: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_sum_to_one(self) -> ClassProbabilities:
        total = self.likely_fake + self.likely_reliable + self.misleading
        if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
            raise ValueError(
                f"Class probabilities must sum to approximately 1.0 (got {total:.4f}, "
                f"tolerance: +/-{PROBABILITY_SUM_TOLERANCE})."
            )
        return self


class MLAssessment(BaseModel):
    """Output contract for the frozen classical machine learning model."""

    model_config = ConfigDict(extra="forbid")

    label: MLLabel = Field(..., description="Predicted label (strictly no 'unverifiable').")
    probabilities: ClassProbabilities = Field(
        ..., description="Raw model probability estimates (not assumed to be calibrated)."
    )
    model_name: str = Field(..., min_length=1, description="Model family identifier.")
    model_version: Optional[str] = Field(default=None, description="Artifact version or identifier.")


# ---------------------------------------------------------------------------
# Evidence Retrieval Schemas
# ---------------------------------------------------------------------------
class EvidenceItem(BaseModel):
    """Structured evidence document retrieved to support or refute a claim."""

    model_config = ConfigDict(extra="forbid")

    id: Optional[Union[str, int]] = Field(default=None)
    claim_summary: str = Field(..., min_length=1, description="Summary of the target claim.")
    evidence_text: str = Field(..., min_length=1, description="Excerpt of retrieved text.")
    source_name: str = Field(..., min_length=1, description="Authoritative publisher or domain.")
    source_url: Optional[str] = Field(default=None, description="URL of reference document.")
    published_date: Optional[Union[date, str]] = Field(default=None, description="Publication date if available.")
    similarity_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Embedding/lexical similarity score (does NOT imply stance)."
    )
    stance: Optional[EvidenceStance] = Field(
        default=None, description="Evaluated stance of evidence toward claim."
    )
    source_type: Optional[str] = Field(
        default=None, description="Classification of source, e.g. news, official_report, press_release, fact_check."
    )
    is_synthetic: bool = Field(
        default=False, description="Whether this record is a synthetic demonstration fixture."
    )
    title: Optional[str] = Field(default=None, description="Title or headline of candidate article.")
    source_domain: Optional[str] = Field(default=None, description="Canonical domain extracted from source URL.")
    retrieval_source: Optional[str] = Field(
        default="local_kb", description="Discovery source channel (e.g. local_kb, gdelt_context, gdelt_doc)."
    )
    evidence_quality: Optional[str] = Field(
        default=None, description="Evidence quality indicator: context_snippet, headline_only, curated_kb."
    )
    retrieved_at: Optional[Union[datetime, str]] = Field(
        default=None, description="Timestamp when evidence candidate was discovered."
    )
    attribution_type: Optional[AttributionType] = Field(
        default=None, description="Attribution relationship classification (e.g. direct_report, quoted_claim, reported_denial)."
    )
    stance_reason: Optional[str] = Field(
        default=None, max_length=300, description="Concise justification for evaluated stance."
    )
    context_sufficient: Optional[bool] = Field(
        default=None, description="Whether excerpt provides sufficient context to establish stance."
    )


class EvidenceRetrievalResult(BaseModel):
    """Structured response from the local evidence retrieval layer."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="Target query or extracted claim text.")
    items: list[EvidenceItem] = Field(
        default_factory=list, max_length=10, description="Retrieved evidence items clearing similarity threshold."
    )
    retrieval_backend: str = Field(..., min_length=1, description="Identifier of retrieval backend used.")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Minimum similarity score required.")
    no_sufficient_evidence: bool = Field(
        default=False, description="Flag indicating no evidence cleared the similarity threshold."
    )
    unique_source_domains: Optional[int] = Field(
        default=0, ge=0, description="Count of distinct domains in candidate evidence set."
    )


# ---------------------------------------------------------------------------
# V2.3 Claim <-> Evidence Stance Reasoning & Deterministic Consensus Schemas
# ---------------------------------------------------------------------------
class EvidenceItemAssessment(BaseModel):
    """Structured assessment of a single candidate evidence snippet relative to a claim."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: Union[str, int] = Field(..., description="Target evidence candidate identifier.")
    claim_index: int = Field(..., ge=0, description="Sequential index of claim being evaluated.")
    stance: EvidenceStance = Field(..., description="Assessed relationship category.")
    context_sufficient: bool = Field(default=True, description="Whether excerpt text provides sufficient context.")
    attribution_type: AttributionType = Field(
        default=AttributionType.direct_report, description="Reporting relationship classification."
    )
    short_reason: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description="Concise, non-empty, auditable justification (max 300 characters, no hidden reasoning).",
    )


class StanceReasoningBatchResponse(BaseModel):
    """Batch structured output payload from Groq evaluating claim-evidence stances."""

    model_config = ConfigDict(extra="forbid")

    assessments: list[EvidenceItemAssessment] = Field(
        default_factory=list, description="Per-evidence stance assessments."
    )


class EvidenceConsensusResult(BaseModel):
    """Deterministic consensus evaluation across evidence clusters for a single claim."""

    model_config = ConfigDict(extra="forbid")

    claim_index: int = Field(..., ge=0, description="Index of evaluated claim.")
    claim_text: str = Field(..., description="Checkable proposition text.")
    consensus_stance: str = Field(
        ..., description="Resulting stance: supports, contradicts, partial, conflicting, or unknown."
    )
    consensus_sufficient: bool = Field(..., description="Whether conservative corroboration threshold is satisfied.")
    eligible_evidence_count: int = Field(default=0, ge=0, description="Count of eligible context evidence items.")
    support_clusters: int = Field(default=0, ge=0, description="Count of independent supporting clusters.")
    contradiction_clusters: int = Field(default=0, ge=0, description="Count of independent contradicting clusters.")
    partial_clusters: int = Field(default=0, ge=0, description="Count of independent partial clusters.")
    unique_domains: int = Field(default=0, ge=0, description="Count of distinct publisher domains among eligible items.")
    cluster_count: int = Field(default=0, ge=0, description="Total independent evidence clusters detected.")
    excluded_headline_only_count: int = Field(
        default=0, ge=0, description="Headline-only items excluded from decisive consensus."
    )
    excluded_unclear_count: int = Field(
        default=0, ge=0, description="Items excluded due to unclear, irrelevant, or quoted-claim status."
    )
    conflict_detected: bool = Field(default=False, description="Whether both supporting and contradicting clusters coexist.")
    evidence_assessments: list[EvidenceItemAssessment] = Field(
        default_factory=list, description="Associated per-evidence assessments."
    )


class DocumentEvidenceConsensus(BaseModel):
    """Deterministic document-level evidence consensus aggregated across all claims."""

    model_config = ConfigDict(extra="forbid")

    overall_stance: str = Field(
        ..., description="Document-level stance: supports, contradicts, partial, conflicting, unknown."
    )
    overall_sufficient: bool = Field(..., description="Whether document evidence satisfies multi-claim consensus.")
    claim_consensus: list[EvidenceConsensusResult] = Field(
        default_factory=list, description="Per-claim consensus results."
    )
    total_eligible_clusters: int = Field(default=0, ge=0, description="Total independent clusters across all claims.")
    total_unique_domains: int = Field(default=0, ge=0, description="Total unique publisher domains across all claims.")
    conflict_detected: bool = Field(
        default=False, description="Whether inter-claim or intra-claim contradiction conflict exists."
    )
    reason: str = Field(default="", max_length=1000, description="Summary rationale for consensus determination.")


# ---------------------------------------------------------------------------
# Structured LLM Assessment Schemas (Zero Hidden Chain-of-Thought)
# ---------------------------------------------------------------------------
class LLMAssessment(BaseModel):
    """Structured assessment contract for the upcoming LLM provider interface.

    STRICT GUARANTEE:
    This schema contains NO chain-of-thought, private_reasoning, or hidden_reasoning fields.
    All outputs are concise, structured, and auditable.
    """

    model_config = ConfigDict(extra="forbid")

    label: VerdictLabel = Field(..., description="LLM evaluated verdict.")
    source_attribution: SourceAttribution = Field(..., description="Quality of source attribution.")
    evidence_support: EvidenceSupport = Field(..., description="Extent to which external evidence supports text.")
    internal_consistency: InternalConsistency = Field(..., description="Logical consistency of propositions.")
    rhetorical_cues: list[str] = Field(
        default_factory=list, max_length=20, description="Key rhetorical patterns identified."
    )
    unsupported_claims: list[str] = Field(
        default_factory=list, max_length=20, description="Propositions lacking corroboration."
    )
    short_reason: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Concise, non-empty, auditable justification (max 1000 characters).",
    )
    prompt_version: Optional[str] = Field(
        default=None,
        description="Identifier of the reasoning prompt version used.",
    )


    @field_validator("short_reason")
    @classmethod
    def validate_short_reason(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("short_reason cannot be empty or whitespace-only.")
        return stripped


class LLMCallMetadata(BaseModel):
    """Safe metadata capturing operational telemetry of an LLM generation call.

    STRICT GUARANTEE:
    Contains NO secret API keys, authentication tokens, or raw authorization headers.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(..., min_length=1, description="LLM provider name.")
    model: str = Field(..., min_length=1, description="Model identifier.")
    latency_ms: Optional[float] = Field(default=None, ge=0.0, description="Call latency in milliseconds.")
    input_tokens: Optional[int] = Field(default=None, ge=0, description="Input / prompt token count.")
    output_tokens: Optional[int] = Field(default=None, ge=0, description="Output / completion token count.")
    total_tokens: Optional[int] = Field(default=None, ge=0, description="Total token count.")


class ConnectivityResponse(BaseModel):
    """Minimal response contract for verifying LLM adapter connectivity."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = Field(..., description="Connection health status.")


# ---------------------------------------------------------------------------
# Per-Claim Result Schema
# ---------------------------------------------------------------------------
class ClaimResult(BaseModel):
    """Verification assessment for an individual checkable claim."""

    model_config = ConfigDict(extra="forbid")

    claim: Claim
    assessment: ClaimAssessment = Field(..., description="Verification status.")
    evidence: list[EvidenceItem] = Field(default_factory=list, max_length=20)
    explanation: Optional[str] = Field(default=None, max_length=1000)
    consensus: Optional[EvidenceConsensusResult] = Field(
        default=None, description="Deterministic consensus result for this claim."
    )


# ---------------------------------------------------------------------------
# Hybrid Fusion Schemas
# ---------------------------------------------------------------------------
class FusionSignals(BaseModel):
    """Multi-signal inputs aggregated for hybrid decision fusion."""

    model_config = ConfigDict(extra="forbid")

    ml_label: Optional[MLLabel] = None
    ml_max_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    llm_label: Optional[VerdictLabel] = None
    llm_evidence_support: Optional[EvidenceSupport] = None
    evidence_available: bool = Field(default=False)
    evidence_support: EvidenceSupport = Field(default=EvidenceSupport.unavailable)
    real_evidence_count: int = Field(default=0, ge=0)
    synthetic_evidence_count: int = Field(default=0, ge=0)
    red_flag_count: int = Field(default=0, ge=0)
    high_severity_red_flag_count: int = Field(default=0, ge=0)
    model_agreement: Optional[Union[bool, str]] = None
    evidence_consensus_stance: Optional[str] = Field(
        default=None, description="Stance derived from deterministic evidence consensus."
    )
    evidence_consensus_sufficient: Optional[bool] = Field(
        default=None, description="Whether evidence consensus satisfied corroboration threshold."
    )
    eligible_clusters_count: Optional[int] = Field(
        default=None, ge=0, description="Number of eligible independent evidence clusters."
    )


class FinalAssessment(BaseModel):
    """Final system verdict resulting from hybrid fusion or abstention."""

    model_config = ConfigDict(extra="forbid")

    verdict: VerdictLabel = Field(..., description="Final assessment verdict.")
    support_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Internal decision-support score used by the hybrid fusion layer (not calibrated probability).",
    )
    reason: str = Field(..., min_length=1, max_length=2000, description="Concise justification for verdict.")
    abstained: bool = Field(..., description="Whether the system abstained from a definitive verdict.")
    signals_used: list[str] = Field(
        default_factory=list, description="Identifiers of pipeline components contributing to verdict."
    )
    fusion_signals: Optional[FusionSignals] = Field(
        default=None, description="Detailed breakdown of aggregated signals."
    )

    @model_validator(mode="after")
    def validate_abstention_consistency(self) -> FinalAssessment:
        """Enforce semantic consistency between verdict and abstention flag."""
        if self.verdict == VerdictLabel.unverifiable and not self.abstained:
            raise ValueError("When verdict is 'unverifiable', abstained must be True.")
        if self.abstained and self.verdict != VerdictLabel.unverifiable:
            raise ValueError(
                f"When abstained is True, verdict must be 'unverifiable' (got '{self.verdict.value}')."
            )
        return self


class FusionInput(BaseModel):
    """Structured input contract bundling all upstream signals for hybrid fusion."""

    model_config = ConfigDict(extra="forbid")

    ml: Optional[MLAssessment] = None
    red_flags: Optional[RedFlagResult] = None
    llm: Optional[LLMAssessment] = None
    evidence: Optional[EvidenceRetrievalResult] = None
    has_checkable_claims: bool = Field(default=True)
    llm_available: bool = Field(default=True)
    component_statuses: list[ComponentStatus] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Complete Analysis Response & User Feedback
# ---------------------------------------------------------------------------
class ComponentStatus(BaseModel):
    """Execution status for an individual pipeline subsystem."""

    model_config = ConfigDict(extra="forbid")

    component: str = Field(..., min_length=1)
    status: ComponentExecutionStatus
    message: Optional[str] = None


class PipelineTimings(BaseModel):
    """Execution latency breakdown per pipeline subsystem in milliseconds."""

    model_config = ConfigDict(extra="forbid")

    cleaning_ms: float = Field(default=0.0, ge=0.0)
    claim_extraction_ms: float = Field(default=0.0, ge=0.0)
    ml_ms: float = Field(default=0.0, ge=0.0)
    red_flags_ms: float = Field(default=0.0, ge=0.0)
    retrieval_ms: float = Field(default=0.0, ge=0.0)
    live_retrieval_ms: float = Field(default=0.0, ge=0.0)
    stance_reasoning_ms: float = Field(default=0.0, ge=0.0)
    consensus_ms: float = Field(default=0.0, ge=0.0)
    llm_ms: float = Field(default=0.0, ge=0.0)
    fusion_ms: float = Field(default=0.0, ge=0.0)
    total_ms: float = Field(default=0.0, ge=0.0)


class AnalysisResponse(BaseModel):
    """Comprehensive API response payload delivering end-to-end multi-signal analysis."""

    model_config = ConfigDict(extra="forbid")

    analysis_id: Optional[Union[int, str]] = None
    input_text: str
    verdict: VerdictLabel
    support_score: float = Field(..., ge=0.0, le=1.0)
    claims: list[ClaimResult] = Field(default_factory=list)
    red_flags: Optional[RedFlagResult] = Field(
        default=None, description="Optional RedFlagResult (supports graceful degradation)."
    )
    ml: Optional[MLAssessment] = Field(
        default=None, description="Optional MLAssessment (supports graceful degradation)."
    )
    llm: Optional[LLMAssessment] = Field(
        default=None, description="Optional LLM assessment (supports graceful degradation)."
    )
    evidence: list[EvidenceItem] = Field(default_factory=list)
    consensus: Optional[DocumentEvidenceConsensus] = Field(
        default=None, description="Deterministic document-level evidence consensus."
    )
    reason: str = Field(..., min_length=1)
    component_statuses: list[ComponentStatus] = Field(default_factory=list)
    timings: Optional[PipelineTimings] = None
    created_at: Optional[datetime] = None


class FeedbackRating(str, Enum):
    """Allowed categorical feedback rating options."""

    helpful = "helpful"
    not_helpful = "not_helpful"
    unsure = "unsure"


class FeedbackRequest(BaseModel):
    """User feedback payload capturing evaluation of an analysis."""

    model_config = ConfigDict(extra="forbid")

    analysis_id: Union[int, str]
    rating: Optional[FeedbackRating] = None
    agree: Optional[bool] = None
    comment: Optional[str] = Field(default=None, max_length=2000)
    corrected_verdict: Optional[VerdictLabel] = None

    @model_validator(mode="after")
    def validate_and_normalize_rating(self) -> FeedbackRequest:
        """Ensure either rating or agree is provided and keep them coherent."""
        if self.rating is None and self.agree is None:
            raise ValueError("Either 'rating' or 'agree' must be provided.")
        if self.rating is None and self.agree is not None:
            self.rating = FeedbackRating.helpful if self.agree else FeedbackRating.not_helpful
        elif self.agree is None and self.rating is not None:
            self.agree = (self.rating == FeedbackRating.helpful)
        return self


class FeedbackRecord(BaseModel):
    """Persisted user feedback record retrieved from database."""

    model_config = ConfigDict(extra="forbid")

    feedback_id: str
    analysis_id: str
    created_at: datetime
    rating: FeedbackRating
    comment: Optional[str] = None
    corrected_verdict: Optional[VerdictLabel] = None


class AnalysisHistoryItem(BaseModel):
    """Lightweight summary of an analysis for paginated history listings."""

    model_config = ConfigDict(extra="forbid")

    analysis_id: str
    created_at: datetime
    input_preview: str = Field(..., max_length=200)
    verdict: VerdictLabel
    abstained: bool
    support_score: float = Field(..., ge=0.0, le=1.0)
    pipeline_version: Optional[str] = None


class AnalysisRecord(BaseModel):
    """Complete persisted database record for an analysis including audit metadata."""

    model_config = ConfigDict(extra="forbid")

    analysis_id: str
    created_at: datetime
    input_text: str
    cleaned_text: Optional[str] = None
    verdict: VerdictLabel
    abstained: bool
    support_score: float = Field(..., ge=0.0, le=1.0)
    reason: str
    pipeline_version: str
    response: AnalysisResponse


# ---------------------------------------------------------------------------
# API Transport Schemas (Step 17)
# ---------------------------------------------------------------------------
class AnalysisCreateResponse(BaseModel):
    """Response envelope for a newly created and persisted analysis."""

    model_config = ConfigDict(extra="forbid")

    analysis_id: str = Field(..., description="Unique collision-resistant analysis identifier.")
    analysis: AnalysisResponse = Field(..., description="Complete multi-signal analysis payload.")


class StoredAnalysisResponse(BaseModel):
    """Auditable persisted analysis detail view."""

    model_config = ConfigDict(extra="forbid")

    analysis_id: str = Field(..., description="Unique collision-resistant analysis identifier.")
    created_at: datetime = Field(..., description="Timestamp of original analysis execution (UTC).")
    pipeline_version: str = Field(..., description="Pipeline software version identifier at analysis time.")
    analysis: AnalysisResponse = Field(..., description="Complete multi-signal analysis payload.")


class AnalysisHistoryResponse(BaseModel):
    """Paginated analysis history listing."""

    model_config = ConfigDict(extra="forbid")

    items: list[AnalysisHistoryItem] = Field(
        default_factory=list, description="List of lightweight analysis summaries."
    )
    total: int = Field(..., ge=0, description="Total count of stored analyses in database.")
    limit: int = Field(..., ge=1, le=100, description="Pagination limit applied.")
    offset: int = Field(..., ge=0, description="Pagination offset applied.")


class AnalysisDeleteResponse(BaseModel):
    """Response envelope confirming analysis deletion."""

    model_config = ConfigDict(extra="forbid")

    deleted: bool = Field(default=True, description="Whether the analysis record was deleted.")
    analysis_id: str = Field(..., description="Identifier of the deleted analysis.")


class HealthResponse(BaseModel):
    """Operational health status response."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(default="ok", description="Service health indicator.")
    service: str = Field(default="fake-news-detection-api", description="Service identifier.")
    pipeline_version: str = Field(..., description="Configured pipeline version.")
    llm_provider: Optional[str] = Field(default=None, description="Configured active LLM provider name.")


class APIErrorResponse(BaseModel):
    """Structured error response envelope."""

    model_config = ConfigDict(extra="forbid")

    error: str = Field(..., description="High-level error classification code.")
    message: str = Field(..., description="Human-readable explanation of error condition.")
    detail: Optional[str] = Field(default=None, description="Optional diagnostic details (sanitized).")


# ---------------------------------------------------------------------------
# Compatibility Adapters (Bridging Step 5–8 Outputs to Schemas)
# ---------------------------------------------------------------------------
def adapter_from_red_flags(result_dict: dict[str, Any]) -> RedFlagResult:
    """Validate and convert actual output from analyze_red_flags into RedFlagResult.

    Args:
        result_dict: Dictionary returned by backend.app.services.red_flags.analyze_red_flags.

    Returns:
        Validated RedFlagResult Pydantic model.
    """
    return RedFlagResult.model_validate(result_dict)


def adapter_from_classical_prediction(
    prediction_dict: dict[str, Any],
    model_name: str = "Tuned TF-IDF + Logistic Regression",
    model_version: Optional[str] = "1.0.0",
) -> MLAssessment:
    """Validate and convert output from predict_baseline into MLAssessment.

    Args:
        prediction_dict: Dictionary returned by predict_baseline (has 'predicted_label' and 'probabilities').
        model_name: Name of the classical model.
        model_version: Version identifier.

    Returns:
        Validated MLAssessment Pydantic model.
    """
    label_str = prediction_dict["predicted_label"]
    probs = prediction_dict["probabilities"]

    return MLAssessment(
        label=MLLabel(label_str),
        probabilities=ClassProbabilities(
            likely_fake=probs["likely_fake"],
            likely_reliable=probs["likely_reliable"],
            misleading=probs["misleading"],
        ),
        model_name=model_name,
        model_version=model_version,
    )
