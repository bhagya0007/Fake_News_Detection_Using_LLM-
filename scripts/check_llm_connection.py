"""Connectivity verification script for the LLM adapter layer.

Performs a single minimal structured-output health check when live credentials
are configured, or reports offline status safely without failing.
Never exposes API keys or secrets.
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.ai.exceptions import (
    LLMBaseError,
    LLMConfigurationError,
)
from backend.app.ai.llm_client import LLMClient
from backend.app.ai.providers.mock_provider import MockLLMProvider
from backend.app.config import LLMConfig
from backend.app.schemas import ConnectivityResponse


def check_connection(force_mock: bool = False) -> int:
    """Check LLM provider connection or report offline mode.

    Args:
        force_mock: If True, execute the check using MockLLMProvider.

    Returns:
        0 on success or documented offline status, 1 on operational error.
    """
    print("=" * 60)
    print("Fake News Detection Project (P_105) - LLM Adapter Connectivity Check")
    print("=" * 60)

    config = LLMConfig.from_env()

    if force_mock:
        print("[Mode] Forcing Mock Provider verification (--mock specified)")
        provider = MockLLMProvider(model_name="mock-model-v1")
        client = LLMClient(config=config, provider=provider)
    else:
        # Check whether a live provider is configured
        if not config.provider or not config.provider.strip():
            print("Live LLM provider is not configured. Adapter and tests are available in offline mode.")
            print("To configure a provider, copy .env.example to .env and set LLM_PROVIDER and LLM_API_KEY.")
            print("=" * 60)
            return 0

        provider_name = config.provider.strip().lower()
        if provider_name == "mock":
            print(f"[Mode] Configured provider is '{provider_name}'. Running offline validation.")
            client = LLMClient(config=config)
        else:
            if not config.api_key or not config.api_key.strip():
                print(f"Provider '{provider_name}' is set, but LLM_API_KEY is not configured.")
                print("Live LLM provider is not configured. Adapter and tests are available in offline mode.")
                print("=" * 60)
                return 0

            # Attempt to instantiate configured live client
            try:
                client = LLMClient(config=config)
            except LLMConfigurationError as exc:
                print(f"Configuration note: {exc}")
                print("Live LLM provider is not configured. Adapter and tests are available in offline mode.")
                print("=" * 60)
                return 0

    print(f"Provider: {client.provider.provider_name}")
    print(f"Model:    {client.provider.model_name}")

    start_time = time.perf_counter()
    try:
        response = client.generate_structured(
            system_prompt="Return only the requested structured connectivity response.",
            user_prompt="Return status ok.",
            response_model=ConnectivityResponse,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        print("Connectivity: SUCCESS")
        print(f"Validated Status: {response.status}")
        if client.last_metadata and client.last_metadata.latency_ms is not None:
            print(f"Latency: {client.last_metadata.latency_ms:.2f} ms")
        else:
            print(f"Latency: {elapsed_ms:.2f} ms")

        if client.last_metadata and client.last_metadata.total_tokens is not None:
            print(
                f"Token Usage: input={client.last_metadata.input_tokens}, "
                f"output={client.last_metadata.output_tokens}, "
                f"total={client.last_metadata.total_tokens}"
            )
        print("=" * 60)
        return 0

    except LLMBaseError as err:
        print("Connectivity: FAILED")
        print(f"Error ({type(err).__name__}): {err}")
        print("=" * 60)
        return 1
    except Exception as err:
        print("Connectivity: UNEXPECTED ERROR")
        print(f"Error ({type(err).__name__}): {err}")
        print("=" * 60)
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify LLM Provider Connectivity.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Execute connectivity check using deterministic MockLLMProvider",
    )
    args = parser.parse_args()
    sys.exit(check_connection(force_mock=args.mock))


if __name__ == "__main__":
    main()
