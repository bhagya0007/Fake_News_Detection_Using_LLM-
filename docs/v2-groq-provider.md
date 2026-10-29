# Version 2.1: Single-Provider Groq Integration

## Overview

In **Version 2.1** (`v2-groq` branch), the active LLM runtime transitions from Google Gemini to **Groq** as the sole active provider. This update establishes high-throughput, low-latency LLM inference while maintaining the project's strict provider-neutral design, deterministic evidence gates, and security invariants.

---

## Key Architecture & Design Decisions

### 1. Single Active Provider Architecture
- **Sole Active Provider**: Groq (`groq==1.7.0`).
- **No Provider Fallback Chain**: There is NO runtime fallback to Gemini, Cohere, OpenRouter, Anthropic, or OpenAI.
- **Explicit Disallowal of Gemini Runtime**: If `LLM_PROVIDER=gemini` is passed to the V2 runtime, `provider_factory.get_provider()` immediately raises:
  ```text
  LLMConfigurationError("V2 runtime is configured for Groq-only operation.")
  ```
- **Single Secret**: `GROQ_API_KEY` is the single API credential used by the V2 runtime.

### 2. Active Model
- **Model Identifier**: `openai/gpt-oss-20b`
- **Context Window**: 131,072 tokens.
- **Structured Outputs**: Full native support for strict JSON Schema contracts (`response_format={"type": "json_schema", ...}`).
- **Reasoning Handling**: Strict structured JSON mode ensures deterministic output without exposing raw `<think>` or internal chain-of-thought traces to end users or audit logs.

### 3. Strict JSON Schema Normalization
Groq's structured output engine strictly validates JSON schemas according to JSON Schema 2020-12 / OpenAI strict schema rules:
1. Every object schema node must have `additionalProperties: False`.
2. All keys defined in `properties` must be explicitly listed in `required`.
3. Rules apply recursively across root objects, nested child objects, array item schemas, and definition maps (`$defs`).

The `_normalize_schema_for_strict()` utility automatically inspects Pydantic v2 schemas and applies these invariants before passing the schema contract to Groq's API.

### 4. Exception Mapping and Graceful Degradation
All downstream exceptions from the `groq` SDK are intercepted, sanitized for secrets, and mapped to the project's provider-neutral hierarchy:

| Groq SDK Exception | HTTP Code | Internal Project Exception | Pipeline Degradation Behavior |
| :--- | :---: | :--- | :--- |
| `groq.AuthenticationError` | 401 | `LLMConfigurationError` | Fails fast, informs operator without exposing key |
| `groq.RateLimitError` | 429 | `LLMProviderError` | Graceful abstention: marks LLM failed, engine abstains safely |
| `groq.APITimeoutError` | 408 | `LLMTimeoutError` | Graceful abstention: marks LLM timed out, engine abstains safely |
| `groq.NotFoundError` | 404 | `LLMConfigurationError` | Configuration error: model or endpoint not found |
| `groq.BadRequestError` | 400 | `LLMConfigurationError` / `LLMResponseValidationError` | Model error or schema mismatch |
| `groq.InternalServerError` | 5xx | `LLMProviderError` | Graceful abstention: engine continues and abstains safely |

### 5. Security & Secret Redaction
- **Never Logged**: The API key is never written to log files, stdout, SQLite, or frontend responses.
- **Redaction Utility**: All exception messages are filtered through `sanitize_sensitive_text(..., self._api_key)` replacing occurrences with `[REDACTED_API_KEY]`.

---

## Configuration

In `.env`:
```bash
# Provider identifier
LLM_PROVIDER=groq

# Secret Groq API Key
GROQ_API_KEY=gsk_...

# Model identifier
GROQ_MODEL=openai/gpt-oss-20b

# Operational parameters
LLM_TIMEOUT_SECONDS=30
LLM_MAX_RETRIES=1
LLM_TEMPERATURE=0
```

---

## Test Isolation & Zero-Network Verification

- **Automated Tests**: 100% mocked via dependency injection and socket blocking.
- **Pytest Network Guard**: `socket.socket.connect` is intercepted to prevent external network calls during `pytest`.
- **Live Calls in Tests**: Exactly **0** live Groq API calls are made during the automated test suite.
- **Offline / Mock Mode**: Passing `LLM_PROVIDER=mock` activates `MockLLMProvider` which returns deterministic mock assessments for local offline development.

---

## Connectivity Check Utility

Run the diagnostic script to verify configuration:

```bash
# Check current configuration or report safe offline mode
python scripts/check_llm_connection.py

# Force deterministic mock validation
python scripts/check_llm_connection.py --mock
```
