# Fake News Detection Using LLMs

**Project ID:** P_105  
**Student:** Bhagya Varshney  
**Repository:** p105-fake-news-detection-llm-bhagya-varshney  
**Project Status:** Complete, Frozen Academic Research Prototype with Conservative Abstention  
**Active Development Branch:** `v2-groq` (Version 2.6 — Complete, Frozen Evidence-Assisted Research Prototype)

> [!NOTE]
> **Version 2 Release (`v2-groq` branch):** The Version 2 runtime integrates Groq as the sole active LLM provider (`openai/gpt-oss-20b`), public GDELT Context 2.0 & DOC 2.0 endpoints for real-world candidate news discovery, local `sentence-transformers/all-MiniLM-L6-v2` semantic reranking, deterministic wire syndication clustering, and conservative corroboration gating. All historical Step 20 academic submission benchmarks, classical LIAR model artifacts, and datasets remain strictly preserved and frozen.

---

## Overview

**Fake News Detection Using LLMs** is an academic research and engineering prototype that explores hybrid approaches to detecting misinformation and evaluating textual credibility. Rather than treating veracity detection as a monolithic text classification task or trusting generative models unconditionally, the system synthesizes multiple complementary analytical signals:

1. **Conservative Text Preprocessing & Cleaning:** Normalizes unicode, strips evasive zero-width artifacts, and validates input length (up to 20,000 characters).
2. **Checkable Claim Extraction:** Decomposes complex prose into discrete factual assertions and identifies checkable entities.
3. **Classical Machine Learning:** Provides a statistical text-pattern prior trained on political discourse.
4. **Stylometric Red-Flag Analysis:** Computes deterministic linguistic signals for sensationalism, emotional charge, and clickbait patterns.
5. **Semantic Evidence Retrieval:** Performs local dense vector similarity search over reference knowledge records using semantic embeddings.
6. **Evidence-Bound LLM Reasoning:** Prompts a large language model to reason strictly over retrieved reference evidence, enforcing verifiable attribution.
7. **Deterministic Fusion & Epistemic Abstention:** Integrates multi-source signals through a strict rule hierarchy, explicitly abstaining (`unverifiable`) when reference evidence is insufficient.

The system is delivered through a high-performance **FastAPI** REST backend, a relational **SQLite** persistence layer, and an interactive **Streamlit** user interface.

---

## Why This Project?

Traditional pure-learning approaches to fake news detection suffer from a fundamental limitation: **text-only machine learning classifiers learn stylistic, political, and lexical artifacts rather than factual truth**. A sensationalized truth may be falsely classified as fake, while a polished, scholarly lie easily fools linear and transformer-based classifiers.

Conversely, querying Large Language Models directly for fact-checking introduces **hallucination risks, temporal cutoff blindspots, and sycophantic reasoning**.

This project explicitly separates:
- **Pattern-based statistical classification** (stylometrics and vocabulary distributions), from
- **Evidence-grounded epistemic verification** (retrieval, citation matching, and explicit abstention).

When factual claims lack supporting or refuting evidence from a trusted knowledge base, the system chooses **epistemic humility**—refusing to guess and safely returning `unverifiable`.

---

## Architecture

The following ASCII diagram illustrates the end-to-end dataflow and signal processing pipeline:

```text
                User News Text (<= 20,000 chars)
                              |
                              v
                      Input Validation
                              |
                              v
                      Claim Extraction
                              |
                  +-----------+-----------+
                  |                       |
                  v                       v
           Classical ML              Red Flags
          (TF-IDF + LR)           (Stylometrics)
                  |                       |
                  +-----------+-----------+
                              |
                              v
                     Evidence Retrieval
                (Local Dense Embeddings)
                              |
                              v
                Evidence-Bound LLM Reasoning
                  (Gemini / Quota-Aware)
                              |
                              v
                  Deterministic Fusion
                (8-Tier Safety Hierarchy)
                              |
                              v
               Final Assessment / Abstention
                              |
                  +-----------+-----------+
                  |                       |
                  v                       v
               FastAPI                  SQLite
             REST Backend            Audit & History
                  |
                  v
             Streamlit UI
```

---

## Main Features

- **Checkable Claim Extraction:** Isolates atomic factual propositions from opinion, speculation, and hyperbole.
- **Frozen Classical ML Baseline:** Employs a scikit-learn TF-IDF vectorizer paired with a multinomial Logistic Regression classifier trained on political statements.
- **Stylometric & Linguistic Red Flags:** Detects capitalization anomalies, excessive punctuation, urgent clickbait language, and emotional manipulation.
- **Semantic Evidence Retrieval:** Implements dense cosine similarity search over local knowledge base entries using embeddings (`sentence-transformers/all-MiniLM-L6-v2`) with lexical fallback.
- **Evidence-Bound LLM Reasoning:** Evaluates claims against retrieved evidence via Google Gemini (`gemini-2.5-flash`), strictly quarantined from relying on parametric memory.
- **Epistemic Abstention Gate:** When checkable claims lack usable provenance-bearing evidence, the system abstains with `unverifiable` (`abstained=True`, `support_score=0.0`).
- **RESTful API Service:** Modern FastAPI application featuring automated Swagger documentation, input validation, structured error mapping, and audit endpoints.
- **Persistent Audit Logging:** SQLite repository recording complete analysis traces, subsystem latencies, and user feedback without modifying ML weights.
- **Interactive Streamlit Interface:** Visual dashboard presenting extracted claims, evidence cards, radar/bar score charts, history browsing, and feedback forms.

---

## Model Training & Pretrained Components

To maintain scientific integrity, the distinction between trained and pretrained components is strictly defined:

| Component | Architecture / Model | Role in Project | Training Status |
| :--- | :--- | :--- | :--- |
| **V1 Classical Baseline** | Word + Char TF-IDF + Logistic Regression (`liblinear`) | Frozen historical prior statistical text-pattern classifier | **Trained in V1** on LIAR dataset (`models/classical_best_validation.joblib`, SHA-256: `d4c36362...`) |
| **V2 Selected Classifier** | Tuned Word+Char TF-IDF + Logistic Regression | Formally selected V2 statistical text-pattern classifier | **Trained in V2.5** on LIAR + AVeriTeC (Stratified 5-Fold CV Macro F1: **0.4786**) |
| **V2 Challenger Classifier**| MiniLM Embeddings + Logistic Regression | Secondary dense-embedding text classifier | **Trained in V2.5** on LIAR + AVeriTeC (AVeriTeC Dev Macro F1: **0.4378**) |
| **Embedding Model** | `all-MiniLM-L6-v2` | Dense vector embedding generation for reranking & clustering | **Pretrained** (Sentence-Transformers / Hugging Face); used frozen for inference |
| **Large Language Model** | `openai/gpt-oss-20b` via Groq | Structured claim extraction and stance reasoning | **Pretrained external API** (Groq cloud); strictly zero fine-tuning |

> [!NOTE]
> Neither Groq nor MiniLM was trained or fine-tuned during this project. Only the classical Logistic Regression models were trained on the benchmark datasets.

---

## Evaluation

The project separates empirical classification benchmarks from controlled functional validation and operational coverage tests:

### 1. Classical ML Benchmark (Empirical Generalization)

Evaluated strictly on the frozen official **LIAR benchmark test split** (`data/raw/test.tsv`, 1,267 examples):

| Metric | Measured Value | Notes / Baseline Context |
| :--- | :---: | :--- |
| **Test Accuracy** | **43.73%** | 3-class formulation (`true`, `half-true`, `false`) |
| **Macro F1 Score** | **0.4345** | Balanced across all three classes |
| **Weighted F1 Score** | **0.4348** | Accounts for class distribution |
| Random Baseline (Uniform) | 33.33% | Chance probability across 3 balanced classes |
| Majority Class Baseline | 35.75% | Always predicting most frequent class |

*Per-Class Performance Breakdown:*
- `false` class: Precision = 0.446, Recall = 0.518, F1 = 0.479
- `half-true` class: Precision = 0.380, Recall = 0.312, F1 = 0.343
- `true` class: Precision = 0.468, Recall = 0.485, F1 = 0.476

### 2. Controlled Hybrid Functional Benchmark

Evaluated across **44 synthetic and controlled evaluation fixtures** designed to test the deterministic fusion decision tree:

| Functional Track | Total Cases | Expected Behaviors Observed | Functional Adherence |
| :--- | :---: | :---: | :---: |
| Supported Claims (Reliable Verdict) | 11 | 11 | 100.0% |
| Contradicted Claims (Fake Verdict) | 11 | 11 | 100.0% |
| Partially Supported (Misleading Verdict) | 11 | 11 | 100.0% |
| Insufficient Evidence (Abstention Gate) | 11 | 11 | 100.0% |
| **Total Controlled Adherence** | **44** | **44** | **100.0% (44/44)** |

> [!IMPORTANT]
> The 100% adherence score in Track A measures **functional rule correctness of the deterministic fusion logic**, NOT open-world fake-news detection accuracy.

### 3. Retrieval Functional Benchmark

Evaluated on the controlled synthetic/demo reference knowledge base:
- **Recall@1:** **100.0%**
- **Recall@3:** **100.0%**
- *Scope:* Evaluated strictly on controlled reference pairs; this reflects semantic similarity matching on known fixtures, not open-web search accuracy.

### 4. Production Evidence Coverage & Abstention Gate

- **Production Evidence KB:** Contains **0 provenance-bearing records** (maintained empty to prevent ungrounded, hallucinated claims).
- **Production Mode Benchmark:** 20 / 20 arbitrary real-world inputs safely abstained as `unverifiable` (`abstained=True`, `support_score=0.0`).
- *Significance:* Demonstrates that the evidence-gating mechanism functions safely in real-world deployment when evidence is absent.

### 5. Software Quality & Invariant Testing

- **Full Pytest Suite:** **514 passed**, 0 failed, 2 warnings (test runtime: ~45 seconds).
- **Fusion Safety Invariants:** 10 / 10 formal safety invariants passed (0 unsafe decisions, 0 evidence-grounding violations, 0 synthetic quarantine leaks).

---

## Limitations

1. **Domain Shift & Dataset Constraints:** The classical ML model was trained on the LIAR dataset (US political discourse 2007–2016). It struggles to generalize to modern conversational social media, consumer product claims, or geopolitical breaking news.
2. **Modest Classical Accuracy:** At 43.73% accuracy on 3-class LIAR test data, the classical model serves solely as an uncalibrated prior, not a standalone truth oracle.
3. **Absence of Live Web Retrieval:** The prototype does not include automated web scraping or live search engine integration; evidence must be curated in a structured knowledge base.
4. **Empty Production Knowledge Base:** Because the production knowledge base currently contains 0 records, real-world factual claims will abstain as `unverifiable`.
5. **LLM Quota and Latency:** The external Gemini API is rate-limited and introduces 1.5–3.5s of network latency per reasoning request.
6. **Semantic Similarity vs. Stance:** High embedding cosine similarity indicates topical relevance, not factual corroboration.
7. **Uncalibrated Heuristic Scores:** The `support_score` is a heuristic engineering metric of multi-signal alignment ($0.0$ to $1.0$), not a calibrated probability of objective truth.
8. **Academic Prototype Status:** Not intended as an operational fact-checking replacement for professional journalistic verification.

---

## Tech Stack

- **Language:** Python 3.10+ (tested on Python 3.14.7)
- **Machine Learning:** `scikit-learn==1.9.1`, `numpy==2.5.3`, `pandas==3.0.6`, `joblib==1.6.0`
- **Embeddings & NLP:** `sentence-transformers==6.1.0` (with lexical fallback)
- **LLM Provider:** Google GenAI SDK (`google-genai==2.25.0`, Gemini 2.5 Flash)
- **Data Contracts:** `pydantic==2.13.5`
- **Web Backend:** `fastapi==0.142.2`, `uvicorn==0.54.0`, `httpx==0.28.1`
- **Frontend UI:** `streamlit==1.65.0`
- **Database:** `sqlite3` (Python standard library)
- **Test Framework:** `pytest==9.1.1`

---

## Installation & Environment Setup

### 1. Clone Repository & Create Virtual Environment

```bash
git clone https://github.com/bhagya-varshney/p105-fake-news-detection-llm-bhagya-varshney.git
cd p105-fake-news-detection-llm-bhagya-varshney

python -m venv .venv
```

### 2. Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a local `.env` file from the provided template:

```bash
cp .env.example .env
```

Edit `.env` and supply your Gemini API key (optional for offline testing; required for live LLM reasoning):

```ini
GEMINI_API_KEY="your-gemini-api-key-here"
LLM_PROVIDER="gemini"
DATABASE_PATH="data/app/fake_news_analysis.db"
```

---

## Running the System

To run the complete system locally, launch both the backend and frontend in separate terminals:

### Terminal 1: FastAPI Backend Service
```bash
python scripts/run_api.py
```
- Server URL: `http://127.0.0.1:8000`
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc Documentation: `http://127.0.0.1:8000/redoc`
- Healthcheck Endpoint: `http://127.0.0.1:8000/healthz`

### Terminal 2: Streamlit Interactive Web Application
```bash
streamlit run frontend/app.py
```
- Web Application URL: `http://localhost:8501`

---

## Offline Testing & Verification

Execute the complete 514-item test suite offline:
```bash
pytest
```

Run specific subsystem test modules:
```bash
# API & route validation
pytest backend/tests/test_api.py -v

# Hybrid engine integration
pytest backend/tests/test_hybrid_engine.py -v

# Final system evaluation benchmarks
pytest backend/tests/test_final_evaluation.py -v
```

---

## Reproducible Evaluation Scripts

### Version 2 Master Scientific & Safety Evaluation Suite (Current Release)
Run the automated Version 2 evaluation suite (100% offline, 0 Groq calls, 0 GDELT calls):
```bash
python scripts/run_v2_final_evaluation.py
```
Tracks evaluated:
1. **Cryptographic Integrity:** Verifies SHA-256 hashes for all frozen models and raw test datasets
2. **Track C (Stance Reasoning):** 35 controlled pairs (100% accuracy, 1.0000 Macro F1, 0 unsafe decisions)
3. **Track D (Controlled End-to-End Hybrid):** 48 balanced test cases (100% accuracy, 1.0000 Macro F1, 100% abstention accuracy, 0 invariant violations)
4. **Track F & G (Latency & Resilience):** Latency microbenchmarking (46.7 ms p50 offline) and 9 fault-injection tests (100% pass rate)
5. **Track E (Optional Live Demo):** Append `--live` to test real-world Groq and GDELT integration

### Version 1 Historical Benchmark Script
Run the historical Step 20 multi-track evaluation script:
```bash
python scripts/evaluate_final_system.py --all-offline
```

---

## Repository Structure

```text
p105-fake-news-detection-llm-bhagya-varshney/
├── .env.example                   # Environment configuration template
├── .gitignore                      # Git exclusion rules
├── README.md                       # Academic project landing page
├── requirements.txt                # Pinned production dependencies
├── backend/
│   ├── app/
│   │   ├── ai/                     # LLM providers, prompts, and reasoners
│   │   ├── api/                    # FastAPI app, routers, and dependencies
│   │   ├── db/                     # SQLite schema, repository, and models
│   │   ├── services/               # Claim extractor, retriever, red flags, fusion, engine
│   │   ├── config.py               # Shared application configuration
│   │   └── schemas.py              # Pydantic data contracts
│   └── tests/                      # Automated test suite (514 tests)
├── data/
│   ├── app/                        # SQLite production database (0 records default)
│   ├── evaluation/                 # Controlled evaluation datasets
│   ├── evidence/                   # Local evidence knowledge base
│   ├── raw/                        # Official LIAR benchmark dataset splits
│   └── sample/                     # Inspection samples
├── docs/                           # Comprehensive technical & academic documentation
│   ├── api.md                      # REST API endpoint documentation
│   ├── architecture.md             # System architecture & design decisions
│   ├── demo-script.md              # Viva and presentation demo script
│   ├── final-project-report.md     # 26-section comprehensive academic project report
│   ├── screenshot-checklist.md     # Documentation and viva screenshot guide
│   ├── submission-checklist.md     # Pre-submission verification checklist
│   └── viva-guide.md               # 60 viva questions and concise spoken answers
├── frontend/
│   ├── app.py                      # Main Streamlit application
│   ├── client.py                   # HTTP client communicating with FastAPI
│   └── styles.css                  # UI styling
├── models/
│   └── classical_best_validation.joblib  # Frozen classical ML model
├── reports/
│   ├── evaluation/                 # Classical ML training and test reports
│   └── final_evaluation/           # Final multi-track system evaluation report
└── scripts/                        # Utility, training, and evaluation scripts
```

---

## Security & Privacy Audit

- **Zero Secrets Committed:** Environment variables and API keys are strictly excluded via `.gitignore`. Repository scans verify 0 hardcoded secrets.
- **SQL Injection Prevention:** SQLite queries utilize parameterized statements exclusively.
- **Synthetic Evidence Isolation:** Synthetic evaluation fixtures are strictly quarantined from the production knowledge base.
- **Input Bounds:** Enforces strict 20,000-character input caps to prevent memory exhaustion and resource denial.

---

## Academic Disclaimer

This project is an **academic prototype** developed for engineering evaluation and coursework demonstration. It is not an operational or commercial fact-checking service. Veracity assessments are heuristic estimates bounded by available evidence and statistical patterns. Users should consult primary news sources and professional journalists for authoritative factual determination.
