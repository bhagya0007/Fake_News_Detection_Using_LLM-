# SQLite Persistence & Audit Layer Architecture (Step 16)

## 1. Overview and Technology Selection

The persistence layer for the Fake News Detection system provides lightweight, local, audit-grade storage for completed multi-signal analyses and user feedback.

### Why Standard-Library SQLite3?
1. **Zero External Infrastructure:** Standard Python `sqlite3` requires no running database daemon, container, or cloud dependency (e.g., PostgreSQL, MySQL, Supabase).
2. **Zero Heavy ORM Dependencies:** Avoiding SQLAlchemy, Django ORM, or Peewee preserves a lean dependency footprint, prevents impedance mismatch with Pydantic v2 schemas, and eliminates ORM session state bugs.
3. **Audit Trail Integrity:** SQLite stores data in a single portable file, making snapshot backups, reproducible evaluation benchmarks, and local inspection trivial.
4. **Local-Only Privacy:** Submitted user articles and claims are retained strictly on the local machine; zero analytics or database payloads are transmitted to external services.

---

## 2. File Location and Configuration

- **Default Location:** [`data/app/fake_news_analysis.db`](file:///e:/p105-fake-news-detection-llm-bhagya-varshney/data/app/fake_news_analysis.db)
- **Configuration Variable:** `DATABASE_PATH` in [`backend/app/config.py`](file:///e:/p105-fake-news-detection-llm-bhagya-varshney/backend/app/config.py)
- **Environment Override:** Configured via `DATABASE_PATH` in `.env` (supports relative or absolute paths).
- **Git Ignore Policy:** Runtime database files (`*.db`, `*.db-shm`, `*.db-wal`) are strictly ignored in [`.gitignore`](file:///e:/p105-fake-news-detection-llm-bhagya-varshney/.gitignore). The directory structure is preserved using `data/app/.gitkeep`.
- **Demonstration Isolation:** Demonstration scripts target [`data/app/demo_analysis.db`](file:///e:/p105-fake-news-detection-llm-bhagya-varshney/data/app/demo_analysis.db); unit tests run on isolated temporary databases via pytest `tmp_path`. Production database records remain untainted.

---

## 3. Relational Schema & Indexing

The persistence layer defines two tables in [`backend/app/db/schema.py`](file:///e:/p105-fake-news-detection-llm-bhagya-varshney/backend/app/db/schema.py):

```sql
CREATE TABLE IF NOT EXISTS analyses (
    analysis_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    input_text TEXT NOT NULL,
    cleaned_text TEXT,
    verdict TEXT NOT NULL,
    abstained INTEGER NOT NULL,
    support_score REAL NOT NULL,
    reason TEXT NOT NULL,
    claims_json TEXT NOT NULL,
    ml_json TEXT,
    red_flags_json TEXT,
    llm_json TEXT,
    evidence_json TEXT NOT NULL,
    component_statuses_json TEXT,
    timings_json TEXT,
    analysis_response_json TEXT NOT NULL,
    pipeline_version TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id TEXT PRIMARY KEY,
    analysis_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    rating TEXT NOT NULL,
    comment TEXT,
    corrected_verdict TEXT,
    FOREIGN KEY (analysis_id) REFERENCES analyses (analysis_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_analysis_id ON feedback (analysis_id);
```

### Table Breakdown

| Table | Column | Type | Description |
| :--- | :--- | :--- | :--- |
| **`analyses`** | `analysis_id` | `TEXT PRIMARY KEY` | Collision-resistant UUID4 string identifier. |
| | `created_at` | `TEXT NOT NULL` | ISO 8601 string with UTC offset (`YYYY-MM-DDTHH:MM:SS.ffffff+00:00`). |
| | `input_text` | `TEXT NOT NULL` | Verbatim original claim or article submitted by the user. |
| | `cleaned_text` | `TEXT` | Preprocessed text after punctuation and whitespace normalization. |
| | `verdict` | `TEXT NOT NULL` | Authoritative verdict (`likely_reliable`, `misleading`, `likely_fake`, `unverifiable`). |
| | `abstained` | `INTEGER NOT NULL` | Boolean flag (1 for abstained / unverifiable, 0 for definitive verdict). |
| | `support_score` | `REAL NOT NULL` | Heuristic decision-support score ($0.0 \le s \le 1.0$), strictly $0.0$ when abstained. |
| | `reason` | `TEXT NOT NULL` | Concise factual explanation from the fusion engine. |
| | `claims_json` | `TEXT NOT NULL` | Serialized list of extracted `ClaimResult` models. |
| | `ml_json` | `TEXT` | Serialized `MLAssessment` dictionary (nullable under degradation). |
| | `red_flags_json` | `TEXT` | Serialized `RedFlagResult` dictionary (nullable under degradation). |
| | `llm_json` | `TEXT` | Serialized `LLMAssessment` dictionary (nullable under short-circuit or degradation). |
| | `evidence_json` | `TEXT NOT NULL` | Serialized list of retrieved `EvidenceItem` models. |
| | `component_statuses_json`| `TEXT` | Component execution lifecycle statuses. |
| | `timings_json` | `TEXT` | Subsystem latency breakdown in milliseconds. |
| | `analysis_response_json`| `TEXT NOT NULL` | Complete validated Pydantic `AnalysisResponse` JSON payload. |
| | `pipeline_version` | `TEXT NOT NULL` | Pipeline version constant (`hybrid_pipeline_v1`). |
| **`feedback`** | `feedback_id` | `TEXT PRIMARY KEY` | UUID4 identifier for the feedback entry. |
| | `analysis_id` | `TEXT NOT NULL` | Foreign key referencing `analyses.analysis_id`. |
| | `created_at` | `TEXT NOT NULL` | ISO 8601 UTC timestamp of feedback submission. |
| | `rating` | `TEXT NOT NULL` | Categorical rating: `helpful`, `not_helpful`, or `unsure`. |
| | `comment` | `TEXT` | Optional free-form user comments (bounded to 2,000 characters). |
| | `corrected_verdict` | `TEXT` | Optional user-suggested verdict (`likely_reliable`, etc.). |

---

## 4. Structured JSON Storage & Round-Trip Reconstruction

Rather than excessively decomposing nested structures (e.g. creating separate relational tables for claim tokens, lexical cues, and embedding vectors), the persistence layer follows a **hybrid relational + document pattern**:
- High-level decision signals (`verdict`, `support_score`, `abstained`, `created_at`) are stored in indexed relational columns for fast filtering and history pagination.
- Full nested object graphs are serialized via Pydantic v2 `model_dump(mode="json")` and stored in `analysis_response_json`.

### Guaranteed Reconstruction Without Recomputation
When reading an analysis via `repo.get_analysis(id)`:
1. The raw `analysis_response_json` string is loaded from SQLite.
2. The payload is validated through `AnalysisResponse.model_validate(json_data)`.
3. If database contents have been tampered with or corrupted, a controlled [`DataIntegrityError`](file:///e:/p105-fake-news-detection-llm-bhagya-varshney/backend/app/db/exceptions.py) is raised.
4. Historical analyses represent **what the system computed at that specific point in time**; no LLMs, ML models, or evidence retrieval modules are re-invoked.

---

## 5. Architectural Separation of Concerns

```text
User Text Input
      │
      ▼
┌───────────────────────────────────────┐
│     HybridAnalysisEngine.analyze()    │  ◄── Completely independent compute engine
└───────────────────────────────────────┘      (Zero database imports or dependencies)
      │
      ▼ AnalysisResponse (In-Memory)
┌───────────────────────────────────────┐
│           AnalysisService             │  ◄── High-level coordinator
└───────────────────────────────────────┘
      │
      ├──► Returns AnalysisResponse to caller
      │
      ▼
┌───────────────────────────────────────┐
│          AnalysisRepository           │  ◄── Persistence abstraction
└───────────────────────────────────────┘
      │
      ▼
┌───────────────────────────────────────┐
│         SQLite3 (data/app/*.db)       │  ◄── Storage only (Never influences verdicts)
└───────────────────────────────────────┘
```

### Deterministic Failure Policy
- **Verdicts are Immutable:** If database writing fails (e.g., disk full, file permissions, connection lock), [`AnalysisService`](file:///e:/p105-fake-news-detection-llm-bhagya-varshney/backend/app/services/analysis_service.py) raises a controlled `DatabaseError`. It **never** alters the computed analysis result or degrades a `likely_reliable` verdict to `unverifiable`.
- **Decoupled Engine:** [`HybridAnalysisEngine`](file:///e:/p105-fake-news-detection-llm-bhagya-varshney/backend/app/services/hybrid_engine.py) can be instantiated and executed in environments without a filesystem or database.

---

## 6. User Feedback Governance

User feedback is strictly segregated from model training:
1. **Feedback is Audit Data Only:** Feedback records are collected for post-hoc evaluation, error logging, and qualitative inspection.
2. **Zero Automatic Retraining:** Submitting feedback **never** triggers fine-tuning of the Gemini LLM, retraining of the frozen classical ML model (`models/classical_best_validation.joblib`), adjustments to fusion rules, or modification of the local evidence knowledge base.
3. **Cascade Deletion:** When an analysis is deleted via `repo.delete_analysis(id)`, SQLite foreign key enforcement automatically cascades and purges all attached feedback items.

---

## 7. SQLite Engine Settings & Concurrency

Connections managed by [`get_connection()`](file:///e:/p105-fake-news-detection-llm-bhagya-varshney/backend/app/db/connection.py) enforce:
- **`PRAGMA foreign_keys = ON;`**: Enforces relational integrity and cascaded deletions.
- **`PRAGMA journal_mode = WAL;`**: Enables Write-Ahead Logging for high-concurrency read operations while writes are in progress.
- **`row_factory = sqlite3.Row`**: Enables efficient column lookup by name.
- **Parameterized SQL:** All SQL operations strictly use parameter binding (`?`), preventing SQL injection vulnerabilities.

---

## 8. Schema Evolution & Migration Limitations

For the current development milestone, schema management relies on idempotent initialization (`CREATE TABLE IF NOT EXISTS`).
- **No Heavy Migration Tooling:** Tools like Alembic or Flyway are intentionally omitted to avoid unnecessary dependency overhead.
- **Future Migration Path:** If breaking column alterations or relational reorganizations become necessary in subsequent milestones, a dedicated migration utility or lightweight script runner can be introduced without breaking repository contracts.
