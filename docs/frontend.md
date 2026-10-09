# Streamlit Frontend Specification & User Guide (Step 18)

## 1. Overview & Architectural Role

The Streamlit frontend provides an academic, accessible, and responsive user interface for **Fake News Detection Using LLMs** (`P_105`, Student: Bhagya Varshney). 

It acts strictly as an **HTTP API client** consuming the FastAPI backend. It contains zero local ML inference, zero direct database access, zero direct LLM SDK calls, and zero duplicated verdict calculation logic:

```text
Streamlit Frontend (frontend/app.py)
              │
              │ HTTP (APIClient via httpx)
              ▼
FastAPI Backend (backend/app/api/app.py)
              │
              ▼
       AnalysisService (backend/app/services/analysis_service.py)
        ┌─────┴────────────────────────────────────────┐
        ▼                                              ▼
HybridAnalysisEngine (In-Memory)        AnalysisRepository (SQLite DB)
```

### Strict Architectural Boundaries:
- **Zero Direct ML Imports:** The frontend never imports `scikit-learn`, `joblib`, or classical model artifacts.
- **Zero Direct LLM Calls:** The frontend never imports `google.genai` or commercial provider credentials.
- **Zero Direct SQLite Access:** The frontend never connects to `data/app/fake_news_analysis.db` directly; all reads, writes, and deletions flow through the REST API.
- **Zero Synthetic Mode Exposure:** The frontend never exposes `allow_synthetic_evidence` toggles or parameters.
- **Zero Client-Side Verdict Logic:** The frontend strictly presents the authoritative backend `AnalysisResponse`.

---

## 2. Configuration & Environment

The frontend is configured via the `API_BASE_URL` environment variable:

```ini
# Base URL of the FastAPI backend server
API_BASE_URL=http://127.0.0.1:8000
```

If omitted, it defaults to `http://127.0.0.1:8000`.

---

## 3. Startup Order & Instructions

To launch the full system locally, open two terminal windows:

### Terminal 1: FastAPI Backend
```bash
python scripts/run_api.py
```
*Starts Uvicorn server on `http://127.0.0.1:8000`.*

### Terminal 2: Streamlit Frontend
```bash
streamlit run frontend/app.py
```
*Starts Streamlit on `http://localhost:8501` (or next available port).*

---

## 4. User Interface Views & Features

### A. Statement Analysis View (`Analyze`)
- **Input Text Area:** Large text area accepting statements, headlines, or article excerpts up to **20,000 characters**.
- **Live Character Counter:** Displays character progress (`X / 20,000 characters`) and highlights exceeding inputs in red before submission.
- **Analyze Action:** Dispatches `POST /api/v1/analyze` inside a responsive spinner: *"Analyzing claims and available evidence..."*
- **Primary Verdict Presentation:**
  - Distinct accessible color cards for `Likely Reliable` (green), `Misleading` (amber), `Likely Fake` (red), and `Unverifiable` (slate).
  - Explicit badge: **System Abstained** when `abstained = True`.
  - Explanatory justification from the fusion engine.
- **Decision-Support Score Semantics:**
  - Labeled as **Decision-Support Strength** (`0.85 / 1.00`).
  - Tooltip: *"This score is an uncalibrated internal decision-support measure (0.00 to 1.00). It is NOT a calibrated probability that the news is true or false."*
  - Strictly never displayed as a percentage of truth.
- **Multi-Signal Subsystem Expanders:**
  - **Extracted Claims:** Numbered atomic propositions, checkability badges, and individual claim evaluations.
  - **Retrieved Evidence:** Source name, excerpt, publication date, URL link, similarity score, and stance.
    - *Disclaimer:* *"Semantic similarity measures retrieval relevance in the knowledge base. It does NOT by itself prove factual correctness or truth."*
    - *Empty State:* *"No usable provenance-bearing evidence was found in the current knowledge base."*
  - **Classical ML Signal:** TF-IDF + Logistic Regression predicted class and class probabilities.
    - *Disclaimer:* *"The classical model was trained on the LIAR political statement dataset. These scores represent statistical model output, not calibrated probabilities of factual truth."*
  - **Language / Presentation Signals:** Red-flag detector indicators (excessive capitalization, sensational vocabulary, emotional urgency).
    - *Disclaimer:* *"Stylistic indicators detect sensationalism, capitalization, and emotional urgency. They reflect writing style only and are NOT independent proof that a claim is false."*
  - **Evidence-Bound LLM Assessment:** Rubric evaluation details (source attribution, evidence support, consistency, rhetorical framing).
    - *Quota-Aware Skip Notice:* *"LLM evidence reasoning was skipped because no usable external evidence was available. This is a quota-aware design policy: the system avoids speculative or ungrounded generative inference."*
  - **Pipeline Technical Details:** Status indicators (Success, Skipped, Unavailable, Failed) and runtime latency breakdowns in milliseconds.
  - **Qualitative Feedback Form:** Captures helpfulness rating, comments, and suggested corrected verdict. Stored strictly for post-hoc error analysis (never automated retraining).

### B. History View (`Analysis History`)
- **Paginated Listing:** Paginated overview (`limit=10`, `offset=0`) ordered newest-first.
- **Direct Record Inspection:** Selecting an analysis retrieves the full stored JSON record directly from SQLite (`GET /api/v1/analyses/{id}`) without re-running vector search, classical ML, or LLM reasoning.
- **Audit Deletion:** Allows deleting an analysis record with an explicit confirmation checkbox; deletes cascaded feedback items via SQLite foreign key enforcement.

### C. System Overview View (`About & Architecture`)
- Educational breakdown of the multi-signal hybrid pipeline.
- Visual ASCII pipeline diagram.
- Transparent disclosure of research limitations (LIAR dataset domain shift, 0-record production KB, local-only search, uncalibrated score semantics, and safe abstention policies).

---

## 5. Offline & Error Handling

- **Backend Offline Shield:** If the FastAPI server is not reachable, the sidebar displays `Backend: Offline` with a red badge and instructions. The Analyze button is disabled, preventing unhandled exceptions.
- **Timeout Protection:** Client requests time out after 90 seconds, presenting an informative timeout message rather than hanging indefinitely.
- **Sanitized Status Codes:** HTTP 422, 500, and 503 errors returned by FastAPI are mapped into friendly user alerts without leaking stack traces or internal paths.

---

## 6. Testing & Demonstration

### Frontend Unit & Contract Tests
Run the 35 dedicated API client and presentation tests:
```bash
pytest backend/tests/test_frontend_client.py -v
```

### Controlled End-to-End Integration Demo
Run the automated end-to-end integration demo against an isolated demo database:
```bash
python scripts/demo_frontend.py
```
