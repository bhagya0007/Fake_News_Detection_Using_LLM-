"""Project-level configuration and constants.

Safe constants using Python standard library and python-dotenv.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Project metadata
PROJECT_NAME: str = "Fake News Detection Using LLMs"
PROJECT_ID: str = "P_105"
VERSION: str = "0.1.0"

# Path definitions (relative / resolved paths)
APP_DIR: Path = Path(__file__).resolve().parent
BACKEND_DIR: Path = APP_DIR.parent
PROJECT_ROOT: Path = BACKEND_DIR.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"

# Database Configuration
_db_env = os.getenv("DATABASE_PATH")
if _db_env and _db_env.strip():
    DATABASE_PATH: Path = Path(_db_env.strip())
    if not DATABASE_PATH.is_absolute():
        DATABASE_PATH = PROJECT_ROOT / DATABASE_PATH
else:
    DATABASE_PATH: Path = DATA_DIR / "app" / "fake_news_analysis.db"

# Pipeline version for persistence and audit trail
PIPELINE_VERSION: str = "hybrid_pipeline_v1"

# Canonical Historical V1 Classical Baseline Artifact
CANONICAL_V1_BASELINE_ARTIFACT: Path = MODELS_DIR / "classical_best_validation.joblib"
CANONICAL_V1_BASELINE_SHA256: str = "d4c3636277761f5399166ee41c0eec57cafe7b9216bacfc18d81f26fea415c92"
CANONICAL_LIAR_TEST_TSV_SHA256: str = "ee41e2c741e0019216952625145af8c4aa8c8d3057ec68284b134267bb845e6f"

# Text analysis validation boundaries
MAX_ANALYSIS_TEXT_LENGTH: int = 20_000

# Load environment variables from .env if present
ENV_PATH: Path = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

# Target verdicts
VERDICTS: tuple[str, ...] = (
    "Likely Reliable",
    "Misleading",
    "Likely Fake",
    "Unverifiable",
)


def _get_float_env(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None or not val.strip():
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _get_int_env(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None or not val.strip():
        return default
    try:
        return int(val)
    except ValueError:
        return default


# Groq V2 Runtime Configuration
DEFAULT_GROQ_MODEL: str = "openai/gpt-oss-20b"
GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY")
GROQ_MODEL: str = os.getenv("GROQ_MODEL") or os.getenv("LLM_MODEL") or DEFAULT_GROQ_MODEL

# Global LLM Configuration values
LLM_PROVIDER: Optional[str] = os.getenv("LLM_PROVIDER")
LLM_API_KEY: Optional[str] = GROQ_API_KEY
LLM_MODEL: Optional[str] = GROQ_MODEL

LLM_TIMEOUT_SECONDS: float = _get_float_env("LLM_TIMEOUT_SECONDS", 30.0)
LLM_MAX_RETRIES: int = _get_int_env("LLM_MAX_RETRIES", 1)
LLM_TEMPERATURE: float = _get_float_env("LLM_TEMPERATURE", 0.0)


@dataclass(frozen=True)
class LLMConfig:
    """Immutable configuration container for LLM settings."""

    provider: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    timeout_seconds: float = 30.0
    max_retries: int = 1
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Build an LLMConfig instance from current environment variables."""
        provider = os.getenv("LLM_PROVIDER")
        # In V2, default provider to groq if not explicitly set
        if not provider or not provider.strip():
            provider = "groq"
        else:
            provider = provider.strip()

        if provider.lower() == "groq":
            api_key = os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY")
            model = os.getenv("GROQ_MODEL") or os.getenv("LLM_MODEL") or DEFAULT_GROQ_MODEL
        else:
            api_key = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")
            model = os.getenv("LLM_MODEL") or os.getenv("GROQ_MODEL")

        return cls(
            provider=provider,
            api_key=api_key,
            model=model,
            timeout_seconds=_get_float_env("LLM_TIMEOUT_SECONDS", 30.0),
            max_retries=_get_int_env("LLM_MAX_RETRIES", 1),
            temperature=_get_float_env("LLM_TEMPERATURE", 0.0),
        )

    def validate_for_runtime(self) -> None:
        """Validate configuration parameters for runtime execution."""
        from backend.app.ai.exceptions import LLMConfigurationError

        if not self.provider or not self.provider.strip():
            raise LLMConfigurationError("LLM_PROVIDER must be configured.")

        p = self.provider.strip().lower()
        if p == "groq":
            if not self.api_key or not self.api_key.strip():
                raise LLMConfigurationError(
                    "Missing GROQ_API_KEY for live Groq provider. Set GROQ_API_KEY in your environment."
                )
            if not self.model or not self.model.strip():
                raise LLMConfigurationError("Missing GROQ_MODEL for Groq provider.")
        elif p == "mock":
            pass
        elif p == "gemini":
            raise LLMConfigurationError("V2 runtime is configured for Groq-only operation.")
        else:
            raise LLMConfigurationError(f"Unsupported LLM provider '{self.provider}'. Supported: ['groq', 'mock'].")

        if self.timeout_seconds <= 0:
            raise LLMConfigurationError("LLM_TIMEOUT_SECONDS must be positive.")
        if self.max_retries < 0:
            raise LLMConfigurationError("LLM_MAX_RETRIES cannot be negative.")
        if not (0.0 <= self.temperature <= 2.0):
            raise LLMConfigurationError("LLM_TEMPERATURE must be between 0.0 and 2.0.")


# ---------------------------------------------------------------------------
# Live Evidence Retrieval Configuration (GDELT V2.2)
# ---------------------------------------------------------------------------
LIVE_EVIDENCE_ENABLED: bool = os.getenv("LIVE_EVIDENCE_ENABLED", "true").lower() in ("true", "1", "yes")
GDELT_CONTEXT_URL: str = os.getenv("GDELT_CONTEXT_URL", "https://api.gdeltproject.org/api/v2/context/context")
GDELT_DOC_URL: str = os.getenv("GDELT_DOC_URL", "https://api.gdeltproject.org/api/v2/doc/doc")
GDELT_TIMEOUT_SECONDS: float = _get_float_env("GDELT_TIMEOUT_SECONDS", 6.0)
GDELT_MAX_RECORDS: int = _get_int_env("GDELT_MAX_RECORDS", 25)
LIVE_EVIDENCE_TOP_K: int = _get_int_env("LIVE_EVIDENCE_TOP_K", 5)
GDELT_CONTEXT_TIMESPAN: str = os.getenv("GDELT_CONTEXT_TIMESPAN", "72h")
GDELT_DOC_FALLBACK: bool = os.getenv("GDELT_DOC_FALLBACK", "true").lower() in ("true", "1", "yes")
GDELT_DOC_TIMESPAN: str = os.getenv("GDELT_DOC_TIMESPAN", "3m")
LIVE_EVIDENCE_SEARCH_STOP_COUNT: int = _get_int_env("LIVE_EVIDENCE_SEARCH_STOP_COUNT", 5)
LIVE_EVIDENCE_MAX_ATTEMPTS: int = _get_int_env("LIVE_EVIDENCE_MAX_ATTEMPTS", 3)
GDELT_MIN_REQUEST_INTERVAL: float = _get_float_env("GDELT_MIN_REQUEST_INTERVAL", 1.0)
GDELT_CACHE_TTL_SECONDS: int = _get_int_env("GDELT_CACHE_TTL_SECONDS", 300)


