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


# Global LLM Configuration values
LLM_PROVIDER: Optional[str] = os.getenv("LLM_PROVIDER")
LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY")
LLM_MODEL: Optional[str] = os.getenv("LLM_MODEL")

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
        """Build an LLMConfig instance from the current environment variables."""
        return cls(
            provider=os.getenv("LLM_PROVIDER"),
            api_key=os.getenv("LLM_API_KEY"),
            model=os.getenv("LLM_MODEL"),
            timeout_seconds=_get_float_env("LLM_TIMEOUT_SECONDS", 30.0),
            max_retries=_get_int_env("LLM_MAX_RETRIES", 1),
            temperature=_get_float_env("LLM_TEMPERATURE", 0.0),
        )
