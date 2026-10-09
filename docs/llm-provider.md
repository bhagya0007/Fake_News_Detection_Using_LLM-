# LLM Provider Abstraction Layer (Step 10)

## Overview & Rationale

The LLM Provider Abstraction provides a unified, secure, and resilient adapter pattern for all Large Language Model interactions across the Fake News Detection pipeline. 

### Why Provider Abstraction Exists:
1. **Decoupling Business Logic from Vendor SDKs:** Services such as future claim extraction and multi-signal reasoning do not import external third-party SDKs (`openai`, `google-genai`, `anthropic`). They communicate exclusively with `LLMClient.generate_structured()`.
2. **Strict Contract Validation:** All LLM outputs are validated against Pydantic schemas before reaching the business layer, preventing runtime KeyError bugs or hallucinations from entering the pipeline.
3. **Provider Portability:** Switching models or underlying API vendors requires changing only environment variables (`LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`), with zero modifications to application logic.
4. **Deterministic Testing:** Offline development and automated CI/CD pipelines run entirely against `MockLLMProvider`, consuming zero API credits and eliminating flaky network dependencies.
5. **Centralized Operational Governance:** Timeouts, transient retries, latency instrumentation, and credential masking are handled in one place.

---

## Provider Support Matrix

| Provider | Status | Description |
| :--- | :--- | :--- |
| **`mock`** | **Implemented** | In-memory, deterministic mock provider for unit tests, offline development, and schema validation. Zero network calls. |
| **`gemini`** | **Implemented (Step 10.5)** | Official Google Gemini provider adapter using the `google-genai` SDK with native JSON schema enforcement and OpenAPI sanitization. |
| **`openai`** | *Planned (Future)* | OpenAI provider adapter using the official OpenAI Python SDK. |
| **`anthropic`** | *Planned (Future)* | Anthropic Claude provider adapter using the official Anthropic SDK. |

> [!NOTE]
> Both the **`mock`** and **`gemini`** providers are fully implemented. Unit tests run exclusively against the mock provider with network sockets disabled, while live environments can connect to Google Gemini when configured in `.env`.

---

## Architecture and Data Flow

```text
Business Service (e.g. Claim Extractor, Reasoner)
    │
    │ Calls generate_structured(system_prompt, user_prompt, response_model)
    ▼
LLMClient (backend/app/ai/llm_client.py)
    │
    │ 1. Enforces timeout via ThreadPoolExecutor
    │ 2. Handles transient retries (max_retries)
    ▼
BaseLLMProvider (backend/app/ai/provider_base.py)
    │
    │ Resolves to configured implementation via provider_factory.py
    ▼
MockLLMProvider (backend/app/ai/providers/mock_provider.py)
    │
    │ Returns raw JSON string / canned response + telemetry metadata
    ▼
LLMClient Validation Layer
    │
    │ 1. json.loads() syntax check
    │ 2. response_model.model_validate() Pydantic validation
    │ 3. Masks any sensitive credentials in exceptions
    ▼
Validated Pydantic Model Instance (e.g. LLMAssessment, ClaimResult)
```

---

## Environment Variables & Configuration

Configuration is managed via environment variables and loaded from a local `.env` file using `python-dotenv`:

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `LLM_PROVIDER` | string | `None` | Provider identifier (`mock`, `gemini`, `openai`, `anthropic`). |
| `LLM_API_KEY` | string | `None` | Secret API key for authentication. Never committed to git. |
| `LLM_MODEL` | string | `mock-model-v1` | Model family and version identifier. |
| `LLM_TIMEOUT_SECONDS` | float | `30.0` | Maximum seconds allowed per generation call before raising `LLMTimeoutError`. |
| `LLM_MAX_RETRIES` | integer | `1` | Number of retries for transient provider/timeout failures. |
| `LLM_TEMPERATURE` | float | `0.0` | Sampling temperature (0.0 for deterministic classification). |

### Template (`.env.example`):
```bash
# LLM Configuration
LLM_PROVIDER=
LLM_API_KEY=
LLM_MODEL=

LLM_TIMEOUT_SECONDS=30
LLM_MAX_RETRIES=1
LLM_TEMPERATURE=0
```

---

## Structured Output & Schema Validation

All callers must supply an explicit Pydantic model (`type[BaseModel]`). The adapter executes the following validation pipeline:

1. **Raw String / Object Check:** Verifies that a non-empty response was returned.
2. **JSON Syntax Parsing:** Parses the raw payload into a Python dictionary (`json.loads`). Malformed JSON triggers `LLMResponseValidationError`.
3. **Pydantic Model Validation:** Validates the parsed dictionary using `response_model.model_validate()`. Violations (such as missing required fields, illegal enum values, or forbidden extra fields) trigger `LLMResponseValidationError`.
4. **Zero Hidden Reasoning:** No hidden reasoning or chain-of-thought fields (`cot`, `chain_of_thought`, `thinking`) are allowed or required.

---

## Exception Hierarchy & Error Handling

All adapter exceptions inherit from `LLMBaseError`:

```text
LLMBaseError
├── LLMConfigurationError        # Missing/unsupported provider, missing API key
├── LLMResponseValidationError   # Malformed JSON or Pydantic validation mismatch
└── LLMProviderError             # Downstream provider failures, HTTP 5xx, rate limits
    └── LLMTimeoutError          # Call exceeded LLM_TIMEOUT_SECONDS
```

### Credential Protection:
Every exception raised by the adapter automatically runs through `sanitize_sensitive_text()`. If an underlying provider library echoes an API key in an error trace, the key is replaced with `[REDACTED_API_KEY]`.

---

## Timeout & Retry Policies

- **Strict Timeouts:** Every generation invocation is wrapped in a worker thread and bounded by `LLM_TIMEOUT_SECONDS` (default: 30s).
- **Safe Retries:**
  - Retried: `LLMTimeoutError`, transient `LLMProviderError` (e.g., rate limits, HTTP 500).
  - Never Retried: `LLMConfigurationError` (auth failures, invalid configuration) and `LLMResponseValidationError` (deterministic schema mismatch).
  - Maximum attempts: Exactly `1 + LLM_MAX_RETRIES`.

---

## Deterministic Mock Provider

The `MockLLMProvider` enables full testability without consuming external API credits:
- Pre-queuing responses: `provider.queue_response({"status": "ok"})` or `provider.queue_response(Exception(...))`
- Built-in defaults: Returns valid instances for standard project schemas (`ConnectivityResponse`, `LLMAssessment`).
- Call inspection: Records all queries in `provider.call_history`.

---

## Google Gemini Provider (Step 10.5)

The `GeminiProvider` (`backend/app/ai/providers/gemini_provider.py`) integrates Google's Generative AI platform via the official `google-genai` SDK:
- **SDK & Client:** Uses `google.genai.Client` with configured HTTP timeout (`LLM_TIMEOUT_SECONDS`).
- **Structured Output Generation:** Invokes `client.models.generate_content` with `GenerateContentConfig(response_mime_type="application/json", response_schema=prepared_schema)`.
- **OpenAPI Schema Sanitization:** Pydantic v2 schemas generated with `extra="forbid"` produce `additionalProperties: False`. Because the Gemini REST endpoint rejects unknown attributes in `response_schema`, the provider recursively strips `additional_properties` prior to submission.
- **Credential Protection:** API keys are never stored in exception objects, strings, or logs. Any downstream SDK error message echoing the credential is scrubbed via `sanitize_sensitive_text()`.
- **Operational Error Mapping:**
  - HTTP 401 / 403 / unauthenticated → `LLMConfigurationError`
  - HTTP 404 / model not found → `LLMConfigurationError`
  - HTTP 408 / deadlines / timeouts → `LLMTimeoutError`
  - HTTP 500 / 503 / transient server errors → `LLMProviderError`
- **Telemetry:** Measures wall-clock latency and captures token usage (`prompt_token_count`, `candidates_token_count`, `total_token_count`) in `LLMCallMetadata`.

---

## Verifying Connectivity

To verify adapter setup and connectivity:

### 1. Default Mode (Live Provider Check)
```bash
python scripts/check_llm_connection.py
```
- If configured with valid Gemini credentials (`LLM_PROVIDER=gemini`, `LLM_MODEL=gemini-3.8-flash`):
  ```text
  ============================================================
  Fake News Detection Project (P_105) - LLM Adapter Connectivity Check
  ============================================================
  Provider: gemini
  Model:    gemini-3.8-flash
  Connectivity: SUCCESS
  Validated Status: ok
  Latency: 5461.82 ms
  Token Usage: input=14, output=5, total=162
  ============================================================
  ```
- If no provider or key is set, it reports offline status and exits cleanly (`0`):
  ```text
  Live LLM provider is not configured. Adapter and tests are available in offline mode.
  To configure a provider, copy .env.example to .env and set LLM_PROVIDER and LLM_API_KEY.
  ```

### 2. Mock Verification Mode
```bash
python scripts/check_llm_connection.py --mock
```
- Executes a safe structured health check (`ConnectivityResponse`) using `MockLLMProvider`:
  ```text
  ============================================================
  Fake News Detection Project (P_105) - LLM Adapter Connectivity Check
  ============================================================
  [Mode] Forcing Mock Provider verification (--mock specified)
  Provider: mock
  Model:    mock-model-v1
  Connectivity: SUCCESS
  Validated Status: ok
  Latency: 5.00 ms
  Token Usage: input=3, output=25, total=28
  ============================================================
  ```

