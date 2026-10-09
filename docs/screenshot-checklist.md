# Academic Project Submission: Screenshot Checklist

**Project ID:** P_105  
**Project Title:** Fake News Detection Using LLMs  
**Candidate:** Bhagya Varshney  
**Repository:** `p105-fake-news-detection-llm-bhagya-varshney`  
**Artifact Path:** `docs/screenshot-checklist.md`  

---

## Overview
This checklist itemizes the 13 recommended screenshots for inclusion in the academic project report, presentation slide deck, and oral viva demonstration.

---

## Recommended Screenshot Catalog

| # | Screenshot Name | Target Application / URL / Terminal | Description & Key Visual Elements to Highlight |
| :-: | :--- | :--- | :--- |
| **1** | `01_home_analyze_page.png` | Streamlit (`http://localhost:8501`) | Clean initial home interface showing text area, interactive character counter (0 / 20,000), sample claim buttons, and the **Analyze News / Text** submission button. |
| **2** | `02_unverifiable_verdict.png` | Streamlit (`http://localhost:8501`) | Primary verdict banner displaying **Unverifiable** with blue status card, `abstained=True`, and **Decision-Support Strength: 0.00 / 1.00**. |
| **3** | `03_extracted_claims.png` | Streamlit (Claims Expander) | Expanded **Checkable Claims** card showing atomic propositions parsed from the input, checkability badges (`Checkable: Yes`), and extracted entity tags. |
| **4** | `04_classical_ml_signal.png` | Streamlit (ML Expander) | Classical ML Baseline card displaying predicted prior class (`likely_reliable`, `misleading`, or `likely_fake`), probability distribution bar chart, and LIAR dataset disclaimer. |
| **5** | `05_evidence_retrieval.png` | Streamlit (Evidence Expander) | Retrieval section displaying knowledge base search results, empty-state notice (*"No usable provenance-bearing evidence was available"*), or retrieved evidence cards. |
| **6** | `06_llm_reasoning_section.png` | Streamlit (LLM Expander) | LLM Reasoning Inspector showing either the quota-aware bypass note (*"LLM reasoner skipped: no usable evidence to ground reasoning"*) or evidence-bound rubric output. |
| **7** | `07_pipeline_status.png` | Streamlit (Sidebar / Header) | Subsystem health indicators showing online status for classical ML model, sentence retriever, Gemini adapter, and SQLite audit repository. |
| **8** | `08_analysis_history.png` | Streamlit (History Section) | Paginated historical audit view listing previous analyses, timestamps, input summaries, and recorded verdict labels. |
| **9** | `09_feedback_submission.png` | Streamlit (Feedback Section) | Interactive feedback form showing qualitative rating buttons (`Helpful`, `Not Helpful`, `Unsure`), optional notes field, and successful submission toast (`201 Created`). |
| **10** | `10_fastapi_swagger_docs.png` | Browser (`http://127.0.0.1:8000/docs`) | Interactive Swagger UI detailing all REST endpoints: `/healthz`, `/api/v1/analyze`, `/api/v1/analyses`, and `/api/v1/analyses/{id}/feedback`. |
| **11** | `11_pytest_terminal_passing.png` | Terminal Command Window | Full test suite output executing `pytest`, showing all **514 passed tests**, 0 failures, 2 benign warnings, and total execution time (~45s). |
| **12** | `12_evaluation_terminal_output.png` | Terminal Command Window | Execution of `python scripts/evaluate_final_system.py --all-offline` displaying summary tables across all 7 evaluation tracks. |
| **13** | `13_architecture_diagram.png` | Architecture Document / Diagram | High-resolution render of the 7-stage hybrid pipeline architecture diagram depicting flow from input to fusion and persistence. |

---

## Verification & Capture Instructions

1. **Window Sizing:** Capture desktop browser windows at standard 1920x1080 resolution with 100% DPI scaling to ensure crisp font rendering.
2. **Terminal Theme:** Use a dark or high-contrast terminal background with readable monospace typography (e.g., Cascadia Code or Consolas).
3. **No Private Secrets:** Ensure no terminal windows or browser developer consoles contain active API keys (`GEMINI_API_KEY`) or system environment paths with sensitive usernames.
4. **Consistency:** Ensure verdict timestamps in screenshots match the local audit sequence.
