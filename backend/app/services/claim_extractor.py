"""Claim Extraction Service (Step 11).

Extracts at most 5 atomic, checkable factual claims from arbitrary text input
(headlines, paragraphs, statements) using the provider-neutral LLM adapter.

Principles:
1. Purely descriptive: Extracts factual propositions without judging truth or falsehood.
2. Zero external knowledge: Does not correct or supplement claims with model knowledge.
3. Safe normalization: Cleans whitespace while strictly preserving numbers, entities, dates.
4. Contract enforcement: Enforces maximum 5 claims in both prompt and schema validation.
5. No vendor SDK imports: Relies exclusively on LLMClient.
"""

from pathlib import Path
from typing import Optional

from backend.app.ai.llm_client import LLMClient
from backend.app.config import APP_DIR
from backend.app.schemas import (
    MAX_ANALYSIS_TEXT_LENGTH,
    Claim,
    ClaimExtractionResult,
    LLMCallMetadata,
)
from backend.app.services.text_cleaner import clean_text

PROMPTS_DIR: Path = APP_DIR / "ai" / "prompts"
DEFAULT_PROMPT_VERSION: str = "claim_extraction_v1"


class ClaimExtractor:
    """Orchestrates structured LLM-based factual claim extraction."""

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
        """Load the versioned system prompt from storage."""
        prompt_path = PROMPTS_DIR / f"{version}.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Claim extraction prompt file not found for version '{version}' at {prompt_path}."
            )
        return prompt_path.read_text(encoding="utf-8").strip()

    def extract_claims(self, text: str) -> ClaimExtractionResult:
        """Extract up to 5 checkable factual claims from input text.

        Args:
            text: Source text (statement, headline, or article excerpt).

        Returns:
            Validated ClaimExtractionResult containing extracted claims.

        Raises:
            TypeError: If input text is not a string.
            ValueError: If input is None, empty, whitespace-only, or exceeds max length.
            LLMResponseValidationError: If LLM output fails schema validation.
            LLMBaseError: If downstream LLM client encounters an unrecoverable failure.
        """
        # 1. Strict input validation
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

        # 2. Conservative text cleaning (normalizes whitespace, preserves entities/numbers/casing)
        cleaned_text = clean_text(text)

        # 3. Construct user prompt
        user_prompt = (
            f"Source Text:\n"
            f"```\n"
            f"{cleaned_text}\n"
            f"```\n\n"
            f"Extract up to 5 atomic, checkable factual claims from the source text above."
        )

        # 4. Invoke LLM client with structured response model
        raw_result, metadata = self.llm_client.generate_structured_with_metadata(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            response_model=ClaimExtractionResult,
        )
        self.last_metadata = metadata

        # 5. Post-process extracted claims
        raw_claims = raw_result.claims

        # Explicit enforcement of maximum 5 claims
        if len(raw_claims) > 5:
            raise ValueError(
                f"Provider returned {len(raw_claims)} claims, exceeding the maximum limit of 5."
            )

        # Deterministic deduplication preserving order
        deduped_claims: list[Claim] = []
        seen_normalized: set[str] = set()

        for idx, claim in enumerate(raw_claims, start=1):
            claim_text = claim.text.strip()
            if not claim_text:
                continue

            normalized_key = claim_text.lower()
            if normalized_key in seen_normalized:
                continue
            seen_normalized.add(normalized_key)

            # Preserve attributes while ensuring claim sequence index
            claim_obj = Claim(
                id=claim.id if claim.id is not None else idx,
                claim_index=claim.claim_index if claim.claim_index is not None else idx,
                text=claim_text,
                checkable=claim.checkable,
                claim_type=claim.claim_type,
                source_span=claim.source_span,
            )
            deduped_claims.append(claim_obj)

        return ClaimExtractionResult(
            claims=deduped_claims,
            source_text_length=len(cleaned_text),
            truncated=False,
            prompt_version=self.prompt_version,
        )
