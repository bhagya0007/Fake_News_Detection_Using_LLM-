"""Structured Evidence-Aware LLM Reasoner Service (Step 12).

Evaluates input text and extracted claims strictly against explicitly supplied
evidence using a versioned reasoning rubric.

Core Principles:
1. Zero Web Browsing / Tool Access: Evaluates exclusively based on supplied materials.
2. No Pretrained Knowledge as Evidence: Does not treat model world knowledge as verification.
3. Strict Unverifiable Policy: If external evidence is required but unavailable or insufficient,
   the verdict MUST be 'unverifiable' (with evidence_support='unavailable').
4. Independent Dimensions: Evaluates internal consistency, source attribution, and evidence
   support as orthogonal dimensions.
5. Zero Hidden Chain-of-Thought: Emits concise, structured justification conforming to LLMAssessment.
6. Vendor Neutrality: Communicates exclusively via LLMClient without third-party SDK dependencies.
"""

from pathlib import Path
from typing import Optional

from backend.app.ai.llm_client import LLMClient
from backend.app.config import APP_DIR
from backend.app.schemas import (
    MAX_ANALYSIS_TEXT_LENGTH,
    Claim,
    EvidenceItem,
    EvidenceSupport,
    InternalConsistency,
    LLMAssessment,
    LLMCallMetadata,
    SourceAttribution,
    VerdictLabel,
)
from backend.app.services.text_cleaner import clean_text

PROMPTS_DIR: Path = APP_DIR / "ai" / "prompts"
DEFAULT_PROMPT_VERSION: str = "reasoning_rubric_v1"


class LLMReasoner:
    """Orchestrates evidence-bound structured reasoning over factual claims."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
    ) -> None:
        self.llm_client: LLMClient = llm_client if llm_client is not None else LLMClient()
        self.prompt_version: str = prompt_version
        self.system_prompt: str = self._load_system_prompt(prompt_version)
        self.last_metadata: Optional[LLMCallMetadata] = None

    def _load_system_prompt(self, version: str) -> str:
        """Load the versioned reasoning prompt from storage."""
        prompt_path = PROMPTS_DIR / f"{version}.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Reasoning rubric prompt file not found for version '{version}' at {prompt_path}."
            )
        return prompt_path.read_text(encoding="utf-8").strip()

    def _format_user_prompt(
        self,
        cleaned_text: str,
        claims: list[Claim],
        evidence: Optional[list[EvidenceItem]],
    ) -> str:
        """Format source text, claims, and evidence into structured LLM input."""
        parts: list[str] = [
            "Source Text to Assess:",
            "```",
            cleaned_text,
            "```",
            "",
            "Extracted Claims:",
        ]

        if not claims:
            parts.append("No checkable factual claims were identified in the source text.")
        else:
            for idx, claim in enumerate(claims, start=1):
                type_tag = f" [Type: {claim.claim_type.value}]" if claim.claim_type else ""
                checkable_tag = "Checkable" if claim.checkable else "Non-checkable"
                parts.append(f"{idx}. {claim.text} ({checkable_tag}{type_tag})")

        parts.extend(["", "Supplied Evidence:"])
        if not evidence:
            parts.append("No external evidence has been supplied.")
        else:
            for idx, item in enumerate(evidence, start=1):
                parts.append(f"Evidence Item [{idx}]:")
                parts.append(f"- Source Name: {item.source_name}")
                if item.source_url:
                    parts.append(f"- Source URL: {item.source_url}")
                if item.published_date:
                    parts.append(f"- Published Date: {item.published_date}")
                if item.similarity_score is not None:
                    parts.append(
                        f"- Semantic Similarity: {item.similarity_score:.4f} "
                        "(Note: lexical/embedding similarity; does NOT imply factual support or stance)"
                    )
                if item.stance:
                    parts.append(
                        f"- Stance Tag: {item.stance.value} "
                        "(Note: advisory tag; verify correspondence directly against excerpt text)"
                    )
                parts.append(f"- Excerpt:\n  \"{item.evidence_text}\"")

        parts.extend([
            "",
            "Assessment Instructions:",
            "Assess the source text and claims strictly according to the rubric rules.",
            "If no external evidence is supplied for an externally checkable claim, assign verdict 'unverifiable' and evidence_support 'unavailable'.",
            "Return only the structured LLMAssessment response.",
        ])

        return "\n".join(parts)

    def assess(
        self,
        text: str,
        claims: list[Claim],
        evidence: Optional[list[EvidenceItem]] = None,
    ) -> LLMAssessment:
        """Assess text and claims against supplied evidence under the reasoning rubric.

        Args:
            text: Original statement or article text.
            claims: List of checkable claims extracted from text.
            evidence: Optional list of retrieved evidence items.

        Returns:
            Validated LLMAssessment structured object.

        Raises:
            TypeError: If input text, claims, or evidence have invalid types.
            ValueError: If input text is None, empty, or exceeds max length.
            LLMResponseValidationError: If LLM output fails schema validation.
            LLMBaseError: If LLM invocation encounters an unrecoverable error.
        """
        # 1. Input validation
        if text is None:
            raise ValueError("Input text cannot be None.")

        if not isinstance(text, str):
            raise TypeError(
                f"Input text must be a string, got {type(text).__name__}."
            )

        stripped = text.strip()
        if not stripped:
            raise ValueError("Input text cannot be empty or whitespace-only.")

        if len(stripped) > MAX_ANALYSIS_TEXT_LENGTH:
            raise ValueError(
                f"Input text exceeds maximum allowed length of {MAX_ANALYSIS_TEXT_LENGTH:,} characters "
                f"(received {len(stripped):,} characters)."
            )

        if claims is None:
            raise ValueError("Claims list cannot be None (pass an empty list if no claims exist).")

        if not isinstance(claims, list):
            raise TypeError(
                f"Claims must be a list of Claim objects, got {type(claims).__name__}."
            )

        for idx, claim in enumerate(claims):
            if not isinstance(claim, Claim):
                raise TypeError(
                    f"Item at claims[{idx}] must be a Claim instance, got {type(claim).__name__}."
                )

        if evidence is not None:
            if not isinstance(evidence, list):
                raise TypeError(
                    f"Evidence must be a list of EvidenceItem objects or None, got {type(evidence).__name__}."
                )
            for idx, item in enumerate(evidence):
                if not isinstance(item, EvidenceItem):
                    raise TypeError(
                        f"Item at evidence[{idx}] must be an EvidenceItem instance, got {type(item).__name__}."
                    )

        # 2. Conservative text cleaning
        cleaned_text = clean_text(text)

        # 3. Format user prompt
        user_prompt = self._format_user_prompt(
            cleaned_text=cleaned_text,
            claims=claims,
            evidence=evidence,
        )

        # 4. Invoke LLM client requesting LLMAssessment
        assessment, metadata = self.llm_client.generate_structured_with_metadata(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            response_model=LLMAssessment,
        )
        self.last_metadata = metadata

        # 5. Ensure prompt version is populated
        if assessment.prompt_version is None:
            assessment = assessment.model_copy(update={"prompt_version": self.prompt_version})

        return assessment
