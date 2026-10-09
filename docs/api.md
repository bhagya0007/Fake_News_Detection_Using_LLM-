# FastAPI REST API Specification (Step 17)

## Overview & Architecture

The FastAPI REST service provides a production-grade, asynchronous HTTP interface for the **Fake News Detection Using LLMs** system (`P_105`). It serves as a thin delivery layer wrapping existing application services:

- [`HybridAnalysisEngine`](file:///e:/p105-fake-news-detection-llm-bhagya-varshney/backend/app/services/hybrid_engine.py): Multi-signal verification pipeline (Text Cleaning → Claim Extraction → Classical ML → Red-Flag Detection → Evidence Retrieval → Evidence-Bound LLM Reasoning → Decision Fusion & Abstention).
- [`AnalysisRepository`](file:///e:/p105-fake-news-detection-llm-bhagya-varshney/backend/app/db/repository.py): Audit-grade SQLite relational persistence (`analyses` and `feedback` tables).
- [`AnalysisService`](file:///e:/p105-fake-news-detection-llm-bhagya-varshney/backend/app/services/analysis_service.py): Business orchestration coordinating analysis execution and audit storage.

```text
HTTP Client / Streamlit UI (Step 18)
                │
                ▼
      FastAPI Application (backend/app/api/app.py)
        ├── Health Router   (/healthz, /api/v1/health)
        ├── Analysis Router (/api/v1/analyze, /api/v1/analyses)
        └── Feedback Router (/api/v1/analyses/{id}/feedback)
                │
                ▼
       AnalysisService (backend/app/services/analysis_service.py)
          ┌──────┴───────────────────────────────────────┐
          ▼                                              ▼
HybridAnalysisEngine (In-Memory)        AnalysisRepository (SQLite DB)
```

---

## Local Development & Server Startup

### Running the API Server

The server can be launched using the convenience runner script:

```bash
python scripts/run_api.py
```

Or launched directly via Uvicorn CLI:

```bash
uvicorn backend.app.api.app:app --host 127.0.0.1 --port 8000 --reload
```

### Interactive Documentation & OpenAPI

Once running, interactive documentation is accessible at:
- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`
- **Raw OpenAPI 3.1.0 JSON:** `http://127.0.0.1:8000/openapi.json`

---

## Endpoints Specification

### 1. Health & Liveness

#### `GET /healthz` (Alias: `GET /api/v1/health`)
Ultra-lightweight operational liveness probe. Constant-time execution with zero LLM, neural model, or database invocations.

- **Response:** `200 OK`
```json
{
  "status": "ok",
  "service": "fake-news-detection-api",
  "pipeline_version": "hybrid_pipeline_v1",
  "llm_provider": "gemini"
}
```

---

### 2. Analysis Endpoints

#### `POST /api/v1/analyze`
Executes end-to-end multi-signal fact-checking analysis and persists the result to SQLite.

- **Request Body:** `application/json`
```json
{
  "text": "The European Central Bank announced an interest rate cut today."
}
```
- **Validation Rules:**
  - `text`: Non-empty string, min length 1 char (whitespace-trimmed), max length 20,000 chars.
  - `allow_synthetic_evidence`: Forbidden. Not accepted by the endpoint.

- **Response:** `200 OK` (Envelope containing generated `analysis_id` and complete `AnalysisResponse`)
```json
{
  "analysis_id": "863c64e0-4254-4ffa-884d-027950e04900",
  "analysis": {
    "analysis_id": "863c64e0-4254-4ffa-884d-027950e04900",
    "input_text": "The European Central Bank announced an interest rate cut today.",
    "verdict": "likely_reliable",
    "support_score": 0.88,
    "claims": [
      {
        "claim": { "id": 1, "text": "European Central Bank announced rate cut.", "checkable": true },
        "assessment": "supported",
        "evidence": [...]
      }
    ],
    "red_flags": { ... },
    "ml": {
      "label": "likely_reliable",
      "probabilities": { "likely_reliable": 0.82, "likely_fake": 0.08, "misleading": 0.10 },
      "model_name": "Tuned TF-IDF + Logistic Regression",
      "model_version": "validation_frozen"
    },
    "llm": {
      "label": "likely_reliable",
      "source_attribution": "strong",
      "evidence_support": "supported",
      "internal_consistency": "consistent",
      "short_reason": "Verified official wire press releases corroborate the statement."
    },
    "evidence": [...],
    "reason": "Verified corporate wire filings corroborate the statement.",
    "component_statuses": [...],
    "timings": {
      "cleaning_ms": 0.1,
      "claim_extraction_ms": 1.5,
      "ml_ms": 1.2,
      "red_flags_ms": 0.3,
      "retrieval_ms": 0.8,
      "llm_ms": 2.5,
      "fusion_ms": 0.2,
      "total_ms": 6.6
    },
    "created_at": "2026-10-03T21:40:00Z"
  }
}
```

> [!NOTE]
> **Safe Abstention Semantics:** If no usable provenance-bearing evidence exists, the endpoint safely returns `200 OK` with `verdict = "unverifiable"`, `abstained = True`, and `support_score = 0.0`. Safe abstention is a normal, trustworthy domain outcome, not an API error.

#### `GET /api/v1/analyses`
Retrieves a paginated list of historical analyses ordered newest-first. Lightweight summary items omit verbose evidence texts and token vectors.

- **Query Parameters:**
  - `limit`: Integer, default `20`, minimum `1`, maximum `100`.
  - `offset`: Integer, default `0`, minimum `0`.

- **Response:** `200 OK`
```json
{
  "items": [
    {
      "analysis_id": "863c64e0-4254-4ffa-884d-027950e04900",
      "input_text": "The European Central Bank announced an interest rate cut today.",
      "verdict": "likely_reliable",
      "support_score": 0.88,
      "created_at": "2026-10-03T21:40:00Z",
      "pipeline_version": "hybrid_pipeline_v1"
    }
  ],
  "total": 1,
  "limit": 20,
  "offset": 0
}
```

#### `GET /api/v1/analyses/{analysis_id}`
Retrieves the complete, validated audit record for a past analysis directly from SQLite without re-running classical ML models, vector search, or LLMs.

- **Path Parameters:**
  - `analysis_id`: UUID string.
- **Responses:**
  - `200 OK`: Stored analysis record (`StoredAnalysisResponse`).
  - `404 Not Found`: If `analysis_id` does not exist.

#### `DELETE /api/v1/analyses/{analysis_id}`
Deletes an analysis record and cascades deletion to all attached feedback items.

- **Responses:**
  - `200 OK`: `{"deleted": true, "analysis_id": "863c64e0-4254-4ffa-884d-027950e04900"}`
  - `404 Not Found`: If `analysis_id` does not exist.

---

### 3. Qualitative User Feedback Endpoints

#### `POST /api/v1/analyses/{analysis_id}/feedback`
Attaches qualitative human review to an analysis. Used exclusively for post-hoc error analysis and audit review (strictly never used for automated model retraining).

- **Request Body:** `application/json`
```json
{
  "rating": "helpful",
  "notes": "Verified against official central bank wire.",
  "corrected_verdict": "likely_reliable"
}
```
- **Validation Rules:**
  - `rating`: One of `"helpful"`, `"unhelpful"`, `"neutral"`.
  - `notes`: Optional string, max length 2,000 characters.
  - `corrected_verdict`: Optional valid `VerdictLabel` (`"likely_reliable"`, `"misleading"`, `"likely_fake"`, `"unverifiable"`).
- **Responses:**
  - `201 Created`: Returns created `FeedbackResponse` with generated `feedback_id`.
  - `404 Not Found`: If `analysis_id` does not exist.
  - `422 Unprocessable Entity`: If validation fails.

#### `GET /api/v1/analyses/{analysis_id}/feedback`
Lists all feedback items submitted for a specific analysis record.

- **Responses:**
  - `200 OK`: Array of `FeedbackResponse` objects (`list[FeedbackResponse]`).
  - `404 Not Found`: If `analysis_id` does not exist.

---

## Error Handling & Status Codes

All API errors return a standard JSON structure (`APIErrorResponse`):

```json
{
  "error": "not_found",
  "message": "Analysis record with ID 'abc' does not exist.",
  "detail": null
}
```

| HTTP Status Code | Error Code | Description / Trigger Scenario |
| :--- | :--- | :--- |
| `200 OK` | — | Successful request (including normal pipeline abstentions). |
| `201 Created` | — | Successful creation of feedback record. |
| `404 Not Found` | `not_found` | `AnalysisNotFoundError` (unknown `analysis_id`). |
| `422 Unprocessable Entity` | `validation_error` | Empty text, non-string text, oversized text (>20k chars), invalid enum values. |
| `500 Internal Server Error` | `internal_server_error` | Unhandled pipeline exception or corrupted database JSON payload. |
| `503 Service Unavailable` | `service_unavailable` | `DatabaseError` (disk full, locked SQLite database, I/O errors). |

> [!SECURITY]
> **Information Leakage Shielding:** Error handlers intentionally sanitize error output. SQL queries, internal directory paths, stack traces, and environment variables are strictly suppressed from all API responses.

---

## Security, Privacy & Design Principles

1. **Zero Credential Exposure:** Secrets (`GEMINI_API_KEY`, `LLM_API_KEY`) are never exposed in OpenAPI schemas, response bodies, or error messages.
2. **Strict Production Evidence Quarantine:** Public API requests cannot supply synthetic evidence fixtures. The engine runs strictly with `allow_synthetic_evidence=False`.
3. **Safe Abstention Returns HTTP 200:** The pipeline safely abstains (`unverifiable`) when evidence is absent, returning HTTP 200. Abstention is an expected, trustworthy domain outcome, not an API error.
4. **Read Isolation:** History and detail endpoints read directly from SQLite without invoking ML models, vector search, or external APIs.
5. **Heuristic Score Semantics:** `support_score` is explicitly defined as an uncalibrated heuristic measure (0.0 to 1.0) of decision-support strength, never an objective calibrated probability of truth.

---

## Verification & Automated Testing

### Running Tests
Execute the 50 comprehensive API tests:

```bash
pytest backend/tests/test_api.py -v
```

Execute the entire project test suite (464 passed tests):

```bash
pytest
```

### Running Demonstration Script
Run the automated end-to-end API demonstration script:

```bash
python scripts/demo_api.py
```
