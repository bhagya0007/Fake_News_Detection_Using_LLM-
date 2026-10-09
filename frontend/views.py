"""Streamlit View Controllers for Fake News Detection Frontend (Step 18).

Contains page rendering logic for Analyze, History, and About views.
Communicates strictly via APIClient with zero direct ML, SQLite, or LLM access.
"""

from __future__ import annotations

from typing import Any, Optional
import streamlit as st

from frontend.api_client import (
    APIClient,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    MAX_ANALYSIS_TEXT_LENGTH,
)
from frontend.components import (
    render_claims,
    render_classical_ml,
    render_evidence,
    render_llm_assessment,
    render_pipeline_details,
    render_red_flags,
    render_verdict_card,
)


def render_sidebar_status(client: APIClient) -> bool:
    """Render backend connectivity indicator in the Streamlit sidebar."""
    st.sidebar.markdown("### Backend Status")

    # Check or reuse cached status
    if "backend_health" not in st.session_state or st.sidebar.button("🔄 Refresh Status"):
        st.session_state["backend_health"] = client.check_health()

    health = st.session_state.get("backend_health", {})
    is_online = health.get("online", False)

    if is_online:
        st.sidebar.success(
            f"**Online**\n\n"
            f"- Version: `{health.get('pipeline_version', 'v1')}`\n"
            f"- Provider: `{health.get('llm_provider', 'default')}`"
        )
    else:
        err_msg = health.get("error", "Cannot reach server.")
        st.sidebar.error(
            f"**Offline**\n\n{err_msg}\n\n"
            "Ensure the FastAPI server is running (`python scripts/run_api.py`)."
        )

    st.sidebar.markdown("---")
    st.sidebar.caption("P_105 | Bhagya Varshney\nFake News Detection Using LLMs")
    return is_online


# ---------------------------------------------------------------------------
# View 1: Analyze
# ---------------------------------------------------------------------------
def render_analyze_view(client: APIClient, is_online: bool) -> None:
    """Render the primary interactive statement analysis screen."""
    st.title("Fake News Detection Using LLMs")
    st.markdown(
        "Analyze factual claims using classical ML, semantic evidence retrieval, "
        "and evidence-bound LLM reasoning with safe abstention."
    )

    if not is_online:
        st.warning(
            "⚠️ **Backend service is currently unavailable.** "
            "Start the FastAPI server (`python scripts/run_api.py`) at "
            f"`{client.base_url}` to perform live analyses."
        )

    # Input Text Area with live character counter
    user_text = st.text_area(
        label="Enter Statement, Headline, or Article Text:",
        value=st.session_state.get("submitted_input", ""),
        height=180,
        placeholder="e.g., The European Central Bank announced an interest rate cut today amid easing inflation pressures.",
        key="analysis_input_area",
    )

    char_count = len(user_text)
    is_over_limit = char_count > MAX_ANALYSIS_TEXT_LENGTH

    # Character count indicator
    col_count, col_btn = st.columns([3, 1])
    with col_count:
        if is_over_limit:
            st.error(f"Character count: **{char_count:,}** / {MAX_ANALYSIS_TEXT_LENGTH:,} (Exceeds maximum limit)")
        else:
            st.caption(f"Character count: {char_count:,} / {MAX_ANALYSIS_TEXT_LENGTH:,}")

    with col_btn:
        analyze_clicked = st.button(
            "Analyze Statement",
            type="primary",
            use_container_width=True,
            disabled=(not is_online or char_count == 0 or is_over_limit),
        )

    # Trigger analysis on button click
    if analyze_clicked:
        stripped = user_text.strip()
        if not stripped:
            st.error("Please enter a non-empty statement to analyze.")
            return

        if len(stripped) > MAX_ANALYSIS_TEXT_LENGTH:
            st.error(f"Text exceeds the {MAX_ANALYSIS_TEXT_LENGTH:,} character limit. Please shorten your input.")
            return

        st.session_state["submitted_input"] = stripped

        with st.spinner("Analyzing claims and available evidence..."):
            try:
                result = client.analyze(stripped)
                st.session_state["current_analysis_id"] = result.get("analysis_id")
                st.session_state["current_analysis"] = result.get("analysis", {})
                st.toast("Analysis completed successfully!", icon="✅")
            except APIConnectionError as exc:
                st.error(f"Connection Error: {exc}")
            except APITimeoutError as exc:
                st.error(f"Request Timeout: The analysis took longer than {client.timeout}s to complete.")
            except APIStatusError as exc:
                st.error(f"API Error ({exc.error_code}): {exc.message}")
            except Exception as exc:
                st.error(f"An unexpected error occurred: {exc}")

    # Display Analysis Results if present in session state
    current = st.session_state.get("current_analysis")
    analysis_id = st.session_state.get("current_analysis_id")

    if current:
        st.markdown("---")
        st.subheader("Analysis Results")

        verdict = current.get("verdict", "unverifiable")
        abstained = bool(current.get("abstained", str(verdict).lower() == "unverifiable"))
        reason = current.get("reason", "No explanation provided.")
        support_score = float(current.get("support_score", 0.0))

        # 1. Primary Verdict Card
        render_verdict_card(
            verdict=verdict,
            abstained=abstained,
            reason=reason,
            support_score=support_score,
        )

        # 2. Extracted Claims
        render_claims(current.get("claims", []))

        # 3. Evidence
        render_evidence(current.get("evidence", []))

        # 4. Classical ML
        render_classical_ml(current.get("ml"))

        # 5. Red Flags
        render_red_flags(current.get("red_flags"))

        # 6. LLM Reasoner
        render_llm_assessment(current.get("llm"), current.get("component_statuses", []))

        # 7. Pipeline Technical Details
        render_pipeline_details(
            component_statuses=current.get("component_statuses", []),
            timings=current.get("timings"),
            analysis_id=analysis_id,
        )

        # 8. User Feedback Section
        render_feedback_section(client, analysis_id)


# ---------------------------------------------------------------------------
# Feedback Submission Section
# ---------------------------------------------------------------------------
def render_feedback_section(client: APIClient, analysis_id: Optional[str]) -> None:
    """Render qualitative human feedback collection interface."""
    if not analysis_id:
        return

    with st.expander("Submit Feedback on this Analysis", expanded=False):
        st.caption(
            "User feedback is collected strictly for post-hoc error analysis and qualitative evaluation. "
            "Feedback is stored for review and does not automatically retrain the model or alter fusion weights."
        )

        with st.form(key=f"feedback_form_{analysis_id}"):
            rating_choice = st.radio(
                "Was this analysis helpful and accurate?",
                options=["Helpful", "Not Helpful", "Unsure"],
                index=0,
                horizontal=True,
            )
            notes_input = st.text_area(
                "Additional Comments (optional):",
                max_chars=2000,
                placeholder="Share context, missing evidence sources, or comments...",
            )
            corrected_choice = st.selectbox(
                "Suggested Verdict Correction (optional):",
                options=["No correction", "Likely Reliable", "Misleading", "Likely Fake", "Unverifiable"],
                index=0,
            )

            submit_fb = st.form_submit_button("Submit Feedback")

        if submit_fb:
            rating_map = {
                "Helpful": "helpful",
                "Not Helpful": "unhelpful",
                "Unsure": "neutral",
            }
            verdict_map = {
                "Likely Reliable": "likely_reliable",
                "Misleading": "misleading",
                "Likely Fake": "likely_fake",
                "Unverifiable": "unverifiable",
            }

            rating = rating_map.get(rating_choice, "neutral")
            corrected = verdict_map.get(corrected_choice)

            try:
                client.submit_feedback(
                    analysis_id=analysis_id,
                    rating=rating,
                    notes=notes_input.strip() or None,
                    corrected_verdict=corrected,
                )
                st.success("Thank you! Feedback recorded for post-hoc evaluation.")
            except Exception as exc:
                st.error(f"Failed to submit feedback: {exc}")


# ---------------------------------------------------------------------------
# View 2: History
# ---------------------------------------------------------------------------
def render_history_view(client: APIClient, is_online: bool) -> None:
    """Render historical analyses listing and detailed stored record inspection."""
    st.title("Analysis History")
    st.markdown("Browse and inspect past multi-signal analyses stored in SQLite audit persistence.")

    if not is_online:
        st.error("Backend server is offline. Please start FastAPI to access stored history.")
        return

    # Pagination state
    if "history_offset" not in st.session_state:
        st.session_state["history_offset"] = 0
    limit = 10
    offset = st.session_state["history_offset"]

    try:
        data = client.list_analyses(limit=limit, offset=offset)
        items = data.get("items", [])
        total = data.get("total", 0)
    except Exception as exc:
        st.error(f"Could not load analysis history: {exc}")
        return

    st.write(f"Showing **{len(items)}** of **{total}** recorded analyses (Page {offset // limit + 1}):")

    if not items:
        st.info("No analyses have been stored in the database yet. Return to the Analyze tab to run your first check.")
        return

    # History summary table
    history_rows = []
    for item in items:
        history_rows.append(
            {
                "Analysis ID": item.get("analysis_id"),
                "Created At": item.get("created_at", "")[:19],
                "Verdict": item.get("verdict", "").replace("_", " ").title(),
                "Support Score": f"{float(item.get('support_score', 0)):.2f}",
                "Abstained": "Yes" if item.get("abstained") else "No",
                "Input Preview": item.get("input_preview", ""),
            }
        )
    st.dataframe(history_rows, use_container_width=True, hide_index=True)

    # Pagination controls
    col_prev, col_info, col_next = st.columns([1, 2, 1])
    with col_prev:
        if offset > 0:
            if st.button("⬅️ Previous Page"):
                st.session_state["history_offset"] = max(0, offset - limit)
                st.rerun()
    with col_info:
        st.caption(f"Offset: {offset} | Total Records: {total}")
    with col_next:
        if offset + limit < total:
            if st.button("Next Page ➡️"):
                st.session_state["history_offset"] = offset + limit
                st.rerun()

    st.markdown("---")
    st.subheader("Inspect Stored Analysis Record")
    st.caption("Opening an existing analysis reads stored JSON directly from SQLite without re-running models.")

    # Selection dropdown for inspection
    id_options = [item["analysis_id"] for item in items]
    preview_map = {item["analysis_id"]: f"{item['created_at'][:19]} — [{item['verdict']}] {item['input_preview'][:50]}..." for item in items}

    selected_id = st.selectbox(
        "Select an Analysis to View:",
        options=id_options,
        format_func=lambda x: preview_map.get(x, x),
        key="history_inspect_select",
    )

    if selected_id:
        try:
            record = client.get_analysis(selected_id)
            analysis = record.get("analysis", {})

            # Render complete stored view
            v_val = analysis.get("verdict", "unverifiable")
            render_verdict_card(
                verdict=v_val,
                abstained=bool(analysis.get("abstained", str(v_val).lower() == "unverifiable")),
                reason=analysis.get("reason", ""),
                support_score=float(analysis.get("support_score", 0.0)),
            )

            render_claims(analysis.get("claims", []))
            render_evidence(analysis.get("evidence", []))
            render_classical_ml(analysis.get("ml"))
            render_red_flags(analysis.get("red_flags"))
            render_llm_assessment(analysis.get("llm"), analysis.get("component_statuses", []))
            render_pipeline_details(
                component_statuses=analysis.get("component_statuses", []),
                timings=analysis.get("timings"),
                analysis_id=selected_id,
            )

            # View existing feedback for this analysis
            st.markdown("#### Attached User Feedback")
            try:
                fb_list = client.get_feedback(selected_id)
                if fb_list:
                    for fb in fb_list:
                        st.markdown(
                            f"- **Rating:** `{fb.get('rating')}` | "
                            f"**Corrected Verdict:** `{fb.get('corrected_verdict') or 'None'}` | "
                            f"**Notes:** {fb.get('notes') or '(None)'}"
                        )
                else:
                    st.caption("No feedback submitted for this record yet.")
            except Exception:
                st.caption("Feedback retrieval unavailable.")

            # Deletion interface with confirmation
            st.markdown("---")
            st.markdown("#### Danger Zone")
            confirm_delete = st.checkbox(
                f"Confirm deletion of analysis record `{selected_id}` and all attached feedback.",
                key=f"confirm_del_{selected_id}",
            )
            if st.button("Delete Analysis", type="secondary", disabled=not confirm_delete):
                try:
                    client.delete_analysis(selected_id)
                    st.success(f"Analysis `{selected_id}` successfully deleted.")
                    st.session_state["history_offset"] = 0
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to delete analysis: {exc}")

        except Exception as exc:
            st.error(f"Error retrieving analysis record: {exc}")


# ---------------------------------------------------------------------------
# View 3: About & Architecture
# ---------------------------------------------------------------------------
def render_about_view() -> None:
    """Render system overview, methodology, and explicit engineering limitations."""
    st.title("About Fake News Detection Using LLMs")
    st.markdown(
        "A modular, evidence-aware decision-support system developed for academic research "
        "(`Project P_105`, Student: Bhagya Varshney)."
    )

    st.markdown("### System Architecture & Pipeline Flow")
    st.markdown(
        """
        The hybrid pipeline combines classical machine learning, stylometric analysis,
        semantic retrieval, and evidence-bound large language model reasoning:
        """
    )

    st.code(
        """
                 User Text (Up to 20,000 characters)
                                  │
                                  ▼
                   Text Validation & Normalization
                                  │
                                  ▼
                       Claim Extraction (Gemini)
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
      Classical ML Baseline               Red-Flag Stylometrics
    (TF-IDF + Logistic Reg)               (Sensationalism Cues)
                 │                                 │
                 └────────────────┬────────────────┘
                                  │
                                  ▼
                    Local Semantic Evidence Retrieval
                      (Semantic Vector Cosine)
                                  │
                                  ▼
                     Evidence-Bound LLM Reasoner
                      (Short-circuited if no KB evidence)
                                  │
                                  ▼
                     Deterministic Hybrid Fusion
                     (Evidence Availability Gated)
                                  │
                                  ▼
                   Auditable Output & SQLite Sink
        """,
        language="text",
    )

    st.markdown("### Subsystem Responsibilities")
    st.markdown(
        """
        1. **Text Validation & Cleaning:** Normalizes erratic whitespace and invisible characters while strictly preserving punctuation, capitalization, numbers, and URLs.
        2. **Classical ML Baseline:** Fast TF-IDF (1-2 n-grams) + Logistic Regression trained on the benchmark LIAR dataset.
        3. **Deterministic Red-Flag Detector:** Rule-based stylometric heuristics measuring excessive capitalization, emotional urgency, sensationalism, and attribution patterns.
        4. **Local Evidence Knowledge Base:** Semantic cosine search cross-referencing extracted atomic propositions against verified reference corpora.
        5. **Evidence-Bound LLM Reasoner:** Structured Gemini evaluation adhering strictly to retrieved evidence without speculative hallucinations.
        6. **Hybrid Decision Fusion:** Rule-based decision hierarchy. Factual verdicts require verified external evidence; without evidence, the system safely **abstains** (`Unverifiable`).
        """
    )

    st.markdown("---")
    st.markdown("### Key Technical Limitations & Constraints")
    st.info(
        """
        - **Decision-Support Score Semantics:** The `support_score` is an **uncalibrated heuristic measure** (0.00 to 1.00) of multi-signal alignment. It is **NOT** a mathematical probability that a news story is true.
        - **Local Knowledge Base Coverage:** The production evidence knowledge base currently contains 0 records. Without verified provenance-bearing evidence, the system will abstain with `Unverifiable`.
        - **No Automatic Web Search:** All evidence retrieval operates locally over verified reference datasets. The system never executes unrestricted web search.
        - **Safe Abstention Policy:** Abstention is a core safety feature designed to prevent LLM hallucinations when evidence is absent or ambiguous.
        - **User Feedback Segregation:** Feedback collected in the UI is stored for post-hoc error inspection and qualitative study; it never automatically retrains underlying models.
        """
    )
