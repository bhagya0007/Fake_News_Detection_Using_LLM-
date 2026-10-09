"""End-to-End Hybrid Fake News Detection Engine Orchestrator (Step 15).

Coordinates multi-signal verification across the complete analysis pipeline:
1. Input Validation & Conservative Text Cleaning
2. Claim Extraction (Prompt-based factual proposition isolation)
3. Classical ML Inference (TF-IDF + Logistic Regression baseline)
4. Deterministic Red-Flag / Stylometric Analysis
5. Local Evidence Semantic Retrieval (Top-k evidence per checkable claim)
6. Evidence Aggregation & Deduplication
7. Structured Evidence-Bound LLM Reasoning
8. Deterministic Hybrid Decision Fusion & Abstention
9. Comprehensive AnalysisResponse Construction

ARCHITECTURAL PRINCIPLES:
- Controller / Coordinator: Orchestrates existing components without duplicating logic.
- Evidence Gate: Real-world checkable claims without usable non-synthetic evidence strictly abstain.
- Quota-Aware Short-Circuit: Evidence reasoning LLM is bypassed when no usable evidence exists.
- Single Final Authority: Final verdict is determined exclusively by fusion.py.
- Graceful Degradation: Component failures (LLM quota/timeout, ML, red flags) degrade safely to abstention.
- Production Quarantine: Synthetic demo fixtures (is_synthetic=True) are rejected by default in production.
- Decision-Support Score: support_score is an uncalibrated heuristic decision-support score, never a calibrated probability.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

from backend.app.config import DATA_DIR, MODELS_DIR
from backend.app.schemas import (
    MAX_ANALYSIS_TEXT_LENGTH,
    AnalysisResponse,
    Claim,
    ClaimAssessment,
    ClaimExtractionResult,
    ClaimResult,
    ComponentExecutionStatus,
    ComponentStatus,
    EvidenceItem,
    EvidenceRetrievalResult,
    FinalAssessment,
    LLMAssessment,
    MLAssessment,
    PipelineTimings,
    RedFlagResult,
    VerdictLabel,
    adapter_from_classical_prediction,
    adapter_from_red_flags,
)
from backend.app.services.baseline_model import load_baseline_pipeline, predict_baseline
from backend.app.services.claim_extractor import ClaimExtractor
from backend.app.services.evidence_retriever import EvidenceRetriever
from backend.app.services.fusion import fuse
from backend.app.services.llm_reasoner import LLMReasoner
from backend.app.services.red_flags import analyze_red_flags
from backend.app.services.text_cleaner import clean_text

logger = logging.getLogger(__name__)


class DefaultMLService:
    """Production ML service reusing the frozen classical model."""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        if model_path is None:
            model_path = MODELS_DIR / "classical_best_validation.joblib"
        self.model_path = Path(model_path)
        self._pipeline: Any = None

    @property
    def pipeline(self) -> Any:
        if self._pipeline is None:
            self._pipeline = load_baseline_pipeline(self.model_path)
        return self._pipeline

    def predict(self, text: str) -> MLAssessment:
        pred_dict = predict_baseline(self.pipeline, text)
        return adapter_from_classical_prediction(
            pred_dict,
            model_name="Tuned TF-IDF + Logistic Regression",
            model_version="validation_frozen",
        )


class DefaultRedFlagService:
    """Production red-flag service wrapping analyze_red_flags."""

    def analyze(self, text: str) -> RedFlagResult:
        result_dict = analyze_red_flags(text)
        return adapter_from_red_flags(result_dict)


class HybridAnalysisEngine:
    """Orchestrates end-to-end multi-signal hybrid fake news detection pipeline."""

    def __init__(
        self,
        claim_extractor: Optional[Any] = None,
        ml_service: Optional[Any] = None,
        red_flag_service: Optional[Any] = None,
        evidence_retriever: Optional[Any] = None,
        llm_reasoner: Optional[Any] = None,
        fusion_engine: Optional[Any] = None,
        allow_synthetic_evidence: bool = False,
        top_k: int = 3,
    ) -> None:
        """Initialize the HybridAnalysisEngine with persistent subsystem components.

        Args:
            claim_extractor: Optional custom ClaimExtractor instance or fake.
            ml_service: Optional custom ClassicalMLService or fake.
            red_flag_service: Optional custom RedFlagService or fake.
            evidence_retriever: Optional custom EvidenceRetriever or fake.
            llm_reasoner: Optional custom LLMReasoner or fake.
            fusion_engine: Optional custom fusion engine or callable.
            allow_synthetic_evidence: Quarantine override toggle (defaults to False for production).
            top_k: Number of evidence candidates to retrieve per checkable claim.
        """
        self.claim_extractor = claim_extractor if claim_extractor is not None else ClaimExtractor()
        self.ml_service = ml_service if ml_service is not None else DefaultMLService()
        self.red_flag_service = red_flag_service if red_flag_service is not None else DefaultRedFlagService()

        self._evidence_retriever = evidence_retriever
        self.llm_reasoner = llm_reasoner if llm_reasoner is not None else LLMReasoner()
        self.fusion_engine = fusion_engine
        self.allow_synthetic_evidence = allow_synthetic_evidence
        self.top_k = top_k

    @property
    def evidence_retriever(self) -> Any:
        """Lazily initialize persistent EvidenceRetriever upon first access."""
        if self._evidence_retriever is None:
            default_kb = DATA_DIR / "evidence" / "evidence_kb.jsonl"
            default_index = MODELS_DIR / "evidence"
            if default_index.exists() and (default_index / "evidence_index_metadata.json").exists():
                self._evidence_retriever = EvidenceRetriever(index_dir=default_index)
            elif default_kb.exists():
                self._evidence_retriever = EvidenceRetriever(kb_path=default_kb)
            else:
                self._evidence_retriever = EvidenceRetriever()
        return self._evidence_retriever

    @evidence_retriever.setter
    def evidence_retriever(self, val: Any) -> None:
        self._evidence_retriever = val

    @classmethod
    def demo(
        cls,
        claim_extractor: Optional[Any] = None,
        ml_service: Optional[Any] = None,
        red_flag_service: Optional[Any] = None,
        evidence_retriever: Optional[Any] = None,
        llm_reasoner: Optional[Any] = None,
        fusion_engine: Optional[Any] = None,
        top_k: int = 3,
    ) -> "HybridAnalysisEngine":
        """Factory constructor for explicit integration tests and interactive demos with synthetic evidence enabled."""
        return cls(
            claim_extractor=claim_extractor,
            ml_service=ml_service,
            red_flag_service=red_flag_service,
            evidence_retriever=evidence_retriever,
            llm_reasoner=llm_reasoner,
            fusion_engine=fusion_engine,
            allow_synthetic_evidence=True,
            top_k=top_k,
        )

    def _validate_input(self, text: str) -> None:
        """Enforce strict inbound validation policy."""
        if text is None:
            raise ValueError("Input text cannot be None.")

        if not isinstance(text, str):
            raise TypeError(f"Input text must be a string, got {type(text).__name__}.")

        stripped = text.strip()
        if not stripped:
            raise ValueError("Input text cannot be empty or whitespace-only.")

        if len(stripped) > MAX_ANALYSIS_TEXT_LENGTH:
            raise ValueError(
                f"Input text exceeds maximum allowed length of {MAX_ANALYSIS_TEXT_LENGTH:,} characters "
                f"(received {len(stripped):,} characters)."
            )

    def analyze_text(self, text: str) -> AnalysisResponse:
        """Run the complete hybrid verification pipeline on user input text.

        Pipeline Sequence:
        1. Validate input text (strict length & character bounds)
        2. Conservative cleaning (normalize spaces while preserving signals)
        3. Claim extraction (isolated atomic checkable propositions)
        4. Classical ML baseline inference (TF-IDF + Logistic Regression)
        5. Stylometric red-flag analysis (presentation & linguistic cues)
        6. Per-claim evidence retrieval (local KB semantic search)
        7. Cross-claim evidence aggregation and deduplication
        8. Evidence-bound LLM reasoning (structured rubric assessment)
        9. Deterministic hybrid decision fusion and abstention
        10. Build comprehensive AnalysisResponse

        Args:
            text: Raw input statement, headline, or article excerpt.

        Returns:
            AnalysisResponse containing full multi-signal results and final verdict.

        Raises:
            TypeError: If input text is not a string.
            ValueError: If input text is None, empty, whitespace-only, or oversized.
        """
        # 1. Input Validation
        self._validate_input(text)

        start_total = time.perf_counter()
        component_statuses: list[ComponentStatus] = []

        # 2. Conservative Cleaning
        t0 = time.perf_counter()
        cleaned_text = clean_text(text)
        cleaning_ms = round((time.perf_counter() - t0) * 1000, 3)
        component_statuses.append(
            ComponentStatus(component="cleaning", status=ComponentExecutionStatus.success)
        )

        # 3. Claim Extraction
        t0 = time.perf_counter()
        claims: list[Claim] = []
        has_checkable_claims = False
        try:
            if hasattr(self.claim_extractor, "extract_claims"):
                extraction_res = self.claim_extractor.extract_claims(cleaned_text)
            elif callable(self.claim_extractor):
                extraction_res = self.claim_extractor(cleaned_text)
            else:
                raise TypeError("claim_extractor has no extract_claims method and is not callable")

            if isinstance(extraction_res, ClaimExtractionResult):
                claims = extraction_res.claims
            elif isinstance(extraction_res, list):
                claims = extraction_res
            has_checkable_claims = any(c.checkable for c in claims)
            component_statuses.append(
                ComponentStatus(component="claim_extraction", status=ComponentExecutionStatus.success)
            )
        except Exception as exc:
            logger.warning("Claim extraction failed: %s", exc)
            component_statuses.append(
                ComponentStatus(
                    component="claim_extraction",
                    status=ComponentExecutionStatus.failed,
                    message="Claim extraction failed; proceeding with unverified claim state.",
                )
            )
            claims = []
            has_checkable_claims = False

        claim_extraction_ms = round((time.perf_counter() - t0) * 1000, 3)

        # 4. Classical ML Inference
        t0 = time.perf_counter()
        ml_assessment: Optional[MLAssessment] = None
        try:
            if hasattr(self.ml_service, "predict"):
                ml_assessment = self.ml_service.predict(cleaned_text)
            elif callable(self.ml_service):
                ml_assessment = self.ml_service(cleaned_text)
            else:
                raise TypeError("ml_service has no predict method and is not callable")
            component_statuses.append(
                ComponentStatus(component="classical_ml", status=ComponentExecutionStatus.success)
            )
        except Exception as exc:
            logger.warning("Classical ML inference failed: %s", exc)
            component_statuses.append(
                ComponentStatus(
                    component="classical_ml",
                    status=ComponentExecutionStatus.unavailable,
                    message="Classical ML model inference unavailable; proceeding without ML signal.",
                )
            )
            ml_assessment = None

        ml_ms = round((time.perf_counter() - t0) * 1000, 3)

        # 5. Red-Flag Analysis
        t0 = time.perf_counter()
        red_flags_result: Optional[RedFlagResult] = None
        try:
            # Run on raw input to preserve original punctuation clusters, spacing, and capitalization
            if hasattr(self.red_flag_service, "analyze"):
                red_flags_result = self.red_flag_service.analyze(text)
            elif callable(self.red_flag_service):
                red_flags_result = self.red_flag_service(text)
            else:
                raise TypeError("red_flag_service has no analyze method and is not callable")
            component_statuses.append(
                ComponentStatus(component="red_flags", status=ComponentExecutionStatus.success)
            )
        except Exception as exc:
            logger.warning("Red-flag analysis failed: %s", exc)
            component_statuses.append(
                ComponentStatus(
                    component="red_flags",
                    status=ComponentExecutionStatus.unavailable,
                    message="Red-flag analysis unavailable; proceeding without stylometric signal.",
                )
            )
            red_flags_result = None

        red_flags_ms = round((time.perf_counter() - t0) * 1000, 3)

        # 6. Evidence Retrieval per Checkable Claim
        t0 = time.perf_counter()
        claim_results: list[ClaimResult] = []
        deduped_evidence: list[EvidenceItem] = []
        evidence_retrieval_result: Optional[EvidenceRetrievalResult] = None

        if not has_checkable_claims:
            component_statuses.append(
                ComponentStatus(
                    component="evidence_retrieval",
                    status=ComponentExecutionStatus.skipped,
                    message="Skipped: No checkable factual claims identified in input text.",
                )
            )
            for c in claims:
                claim_results.append(
                    ClaimResult(
                        claim=c,
                        assessment=ClaimAssessment.not_checked,
                        evidence=[],
                    )
                )
        else:
            try:
                seen_keys: set[Union[str, int, tuple[str, str]]] = set()
                backend_name = getattr(self.evidence_retriever, "backend_name", "unknown")
                threshold = getattr(self.evidence_retriever, "threshold", 0.45)

                for c in claims:
                    if c.checkable:
                        if hasattr(self.evidence_retriever, "retrieve"):
                            ret_res = self.evidence_retriever.retrieve(c.text, top_k=self.top_k)
                        elif callable(self.evidence_retriever):
                            ret_res = self.evidence_retriever(c.text, self.top_k)
                        else:
                            raise TypeError("evidence_retriever has no retrieve method and is not callable")

                        items = ret_res.items if isinstance(ret_res, EvidenceRetrievalResult) else (ret_res or [])
                        claim_results.append(
                            ClaimResult(
                                claim=c,
                                assessment=ClaimAssessment.not_checked,
                                evidence=items,
                            )
                        )
                        for item in items:
                            key = item.id if item.id is not None else (item.source_name, item.evidence_text)
                            if key not in seen_keys:
                                seen_keys.add(key)
                                deduped_evidence.append(item)
                    else:
                        claim_results.append(
                            ClaimResult(
                                claim=c,
                                assessment=ClaimAssessment.not_checked,
                                evidence=[],
                            )
                        )

                evidence_retrieval_result = EvidenceRetrievalResult(
                    query=cleaned_text[:100],
                    items=deduped_evidence,
                    retrieval_backend=backend_name,
                    threshold=threshold,
                    no_sufficient_evidence=len(deduped_evidence) == 0,
                )
                component_statuses.append(
                    ComponentStatus(component="evidence_retrieval", status=ComponentExecutionStatus.success)
                )
            except Exception as exc:
                logger.warning("Evidence retrieval failed: %s", exc)
                component_statuses.append(
                    ComponentStatus(
                        component="evidence_retrieval",
                        status=ComponentExecutionStatus.failed,
                        message="Evidence retrieval failed; proceeding with zero evidence.",
                    )
                )
                claim_results = [
                    ClaimResult(claim=c, assessment=ClaimAssessment.not_checked, evidence=[])
                    for c in claims
                ]
                deduped_evidence = []
                evidence_retrieval_result = None

        retrieval_ms = round((time.perf_counter() - t0) * 1000, 3)

        # Usable evidence check under production quarantine vs explicit demo mode
        usable_evidence = [
            item for item in deduped_evidence
            if (not item.is_synthetic) or self.allow_synthetic_evidence
        ]
        has_usable_evidence = len(usable_evidence) > 0

        # 7. LLM Evidence-Bound Reasoner (Quota-Aware Evidence Gate)
        t0 = time.perf_counter()
        llm_assessment: Optional[LLMAssessment] = None

        if not has_checkable_claims:
            component_statuses.append(
                ComponentStatus(
                    component="llm_reasoner",
                    status=ComponentExecutionStatus.skipped,
                    message="Skipped: No checkable factual claims identified in input text.",
                )
            )
        elif not has_usable_evidence:
            # Short-circuit: Do NOT call LLM Reasoner when no usable evidence exists.
            # Avoids wasting provider quota and unnecessary latency.
            component_statuses.append(
                ComponentStatus(
                    component="llm_reasoner",
                    status=ComponentExecutionStatus.skipped,
                    message="Skipped because no usable external evidence was available.",
                )
            )
        else:
            try:
                if hasattr(self.llm_reasoner, "assess"):
                    llm_assessment = self.llm_reasoner.assess(
                        text=cleaned_text,
                        claims=claims,
                        evidence=usable_evidence,
                    )
                elif callable(self.llm_reasoner):
                    llm_assessment = self.llm_reasoner(
                        cleaned_text,
                        claims,
                        usable_evidence,
                    )
                else:
                    raise TypeError("llm_reasoner has no assess method and is not callable")

                component_statuses.append(
                    ComponentStatus(component="llm_reasoner", status=ComponentExecutionStatus.success)
                )
            except Exception as exc:
                err_str = str(exc).lower()
                logger.warning("LLM reasoner call failed: %s", exc)
                if "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
                    status = ComponentExecutionStatus.unavailable
                    msg = "LLM unavailable: quota exhausted or rate limited"
                elif "timeout" in err_str or "timed out" in err_str:
                    status = ComponentExecutionStatus.unavailable
                    msg = "LLM unavailable: request timed out"
                else:
                    status = ComponentExecutionStatus.failed
                    msg = "LLM reasoner execution failed"

                component_statuses.append(
                    ComponentStatus(
                        component="llm_reasoner",
                        status=status,
                        message=msg,
                    )
                )
                llm_assessment = None

        llm_ms = round((time.perf_counter() - t0) * 1000, 3)

        # 8. Deterministic Hybrid Decision Fusion & Abstention
        t0 = time.perf_counter()
        try:
            if self.fusion_engine is not None:
                if hasattr(self.fusion_engine, "fuse"):
                    final_assessment = self.fusion_engine.fuse(
                        ml=ml_assessment,
                        red_flags=red_flags_result,
                        llm=llm_assessment,
                        evidence=evidence_retrieval_result,
                        has_checkable_claims=has_checkable_claims,
                        allow_synthetic_evidence=self.allow_synthetic_evidence,
                    )
                elif callable(self.fusion_engine):
                    final_assessment = self.fusion_engine(
                        ml=ml_assessment,
                        red_flags=red_flags_result,
                        llm=llm_assessment,
                        evidence=evidence_retrieval_result,
                        has_checkable_claims=has_checkable_claims,
                        allow_synthetic_evidence=self.allow_synthetic_evidence,
                    )
                else:
                    raise TypeError("fusion_engine is not callable")
            else:
                final_assessment = fuse(
                    ml=ml_assessment,
                    red_flags=red_flags_result,
                    llm=llm_assessment,
                    evidence=evidence_retrieval_result,
                    has_checkable_claims=has_checkable_claims,
                    allow_synthetic_evidence=self.allow_synthetic_evidence,
                )

            component_statuses.append(
                ComponentStatus(component="fusion", status=ComponentExecutionStatus.success)
            )
        except Exception as exc:
            logger.error("Fusion execution failed: %s", exc)
            component_statuses.append(
                ComponentStatus(
                    component="fusion",
                    status=ComponentExecutionStatus.failed,
                    message="Hybrid fusion encountered an internal error; system abstained safely.",
                )
            )
            final_assessment = FinalAssessment(
                verdict=VerdictLabel.unverifiable,
                support_score=0.0,
                reason="Hybrid fusion encountered an internal error; system abstained safely.",
                abstained=True,
                signals_used=["emergency_abstention"],
            )

        fusion_ms = round((time.perf_counter() - t0) * 1000, 3)

        # 9. Update Per-Claim Assessments Based on Verdict
        for cr in claim_results:
            if not cr.claim.checkable:
                cr.assessment = ClaimAssessment.not_checked
            elif final_assessment.abstained:
                cr.assessment = ClaimAssessment.unverifiable
            elif final_assessment.verdict == VerdictLabel.likely_reliable:
                cr.assessment = ClaimAssessment.supported
            elif final_assessment.verdict == VerdictLabel.likely_fake:
                cr.assessment = ClaimAssessment.contradicted
            elif final_assessment.verdict == VerdictLabel.misleading:
                cr.assessment = ClaimAssessment.partially_supported

        # 10. Assemble and Return AnalysisResponse
        total_ms = round((time.perf_counter() - start_total) * 1000, 3)
        timings = PipelineTimings(
            cleaning_ms=cleaning_ms,
            claim_extraction_ms=claim_extraction_ms,
            ml_ms=ml_ms,
            red_flags_ms=red_flags_ms,
            retrieval_ms=retrieval_ms,
            llm_ms=llm_ms,
            fusion_ms=fusion_ms,
            total_ms=total_ms,
        )

        return AnalysisResponse(
            analysis_id=str(uuid.uuid4()),
            input_text=text,
            verdict=final_assessment.verdict,
            support_score=final_assessment.support_score,
            claims=claim_results,
            red_flags=red_flags_result,
            ml=ml_assessment,
            llm=llm_assessment,
            evidence=deduped_evidence,
            reason=final_assessment.reason,
            component_statuses=component_statuses,
            timings=timings,
            created_at=datetime.now(timezone.utc),
        )
