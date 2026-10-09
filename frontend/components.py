"""Reusable Streamlit UI Components and Presentation Formatters (Step 18).

Provides clean, accessible, academic presentation components for multi-signal
factual verification results without importing ML libraries, SQLite, or LLM providers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import streamlit as st


# ---------------------------------------------------------------------------
# Verdict Styling & Presentation
# ---------------------------------------------------------------------------
VERDICT_CONFIG = {
    "likely_reliable": {
        "label": "Likely Reliable",
        "color": "#1b5e20",  # Dark green
        "bg_color": "#e8f5e9",
        "description": "Evidence and multi-signal analysis substantiate the primary claims.",
    },
    "misleading": {
        "label": "Misleading",
        "color": "#e65100",  # Dark amber
        "bg_color": "#fff3e0",
        "description": "Claims contain factual distortions, significant omissions, or inaccurate context.",
    },
    "likely_fake": {
        "label": "Likely Fake",
        "color": "#b71c1c",  # Dark red
        "bg_color": "#ffebee",
        "description": "Primary claims contradict verified evidence or exhibit fabricated propositions.",
    },
    "unverifiable": {
        "label": "Unverifiable",
        "color": "#37474f",  # Dark slate
        "bg_color": "#eceff1",
        "description": "The system does not have sufficient usable evidence to make a grounded factual determination.",
    },
}


def render_verdict_card(verdict: str, abstained: bool, reason: str, support_score: float) -> None:
    """Render the primary verdict card with prominent, accessible visual styling."""
    norm_verdict = (verdict or "unverifiable").lower()
    cfg = VERDICT_CONFIG.get(norm_verdict, VERDICT_CONFIG["unverifiable"])

    st.markdown(
        f"""
        <div style="border-left: 6px solid {cfg['color']}; background-color: {cfg['bg_color']}; padding: 18px 24px; border-radius: 6px; margin: 12px 0 18px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <span style="font-size: 1.6rem; font-weight: 700; color: {cfg['color']};">
                    {cfg['label']}
                </span>
                {"<span style='background-color: #cfd8dc; color: #263238; font-size: 0.85rem; font-weight: 600; padding: 4px 10px; border-radius: 12px;'>System Abstained</span>" if abstained else ""}
            </div>
            <p style="font-size: 1.05rem; color: #263238; margin: 10px 0 6px 0; line-height: 1.45;">
                {reason}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Heuristic Support Score metric card
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.metric(
            label="Decision-Support Strength",
            value=f"{support_score:.2f} / 1.00",
            help="This score is an uncalibrated internal decision-support measure (0.00 to 1.00). "
            "It is NOT a calibrated probability that the news is true or false.",
        )
    with col2:
        status_text = "Abstained" if abstained else "Determined"
        st.metric(
            label="Pipeline Decision",
            value=status_text,
            help="Indicates whether the hybrid engine reached an active conclusion or safely abstained.",
        )
    with col3:
        if abstained:
            st.info(
                "**Why did the system abstain?** Factual verdicts require verifiable, provenance-bearing "
                "external evidence. Without it, the system abstains rather than generating ungrounded claims."
            )


# ---------------------------------------------------------------------------
# Claims Presentation
# ---------------------------------------------------------------------------
def render_claims(claims: list[dict[str, Any]]) -> None:
    """Render extracted claims section."""
    with st.expander("Extracted Claims", expanded=True):
        if not claims:
            st.info("No checkable factual claims were identified in the submitted text.")
            return

        for idx, item in enumerate(claims, 1):
            claim_obj = item.get("claim", {})
            c_text = claim_obj.get("text", "N/A")
            checkable = claim_obj.get("checkable", True)
            c_type = claim_obj.get("claim_type") or "Factual proposition"
            assessment = item.get("assessment", "unverifiable")

            badge_text = "Checkable" if checkable else "Non-checkable"
            st.markdown(
                f"""
                <div style="background-color: #f8f9fa; border: 1px solid #e0e0e0; padding: 12px 16px; border-radius: 6px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-weight: 600; font-size: 0.95rem; color: #424242;">Claim #{idx} &mdash; <em>{c_type}</em></span>
                        <span style="font-size: 0.8rem; background-color: #e3f2fd; color: #0d47a1; padding: 2px 8px; border-radius: 4px;">{badge_text}</span>
                    </div>
                    <div style="font-size: 1.0rem; color: #212121; margin-bottom: 6px;">{c_text}</div>
                    <div style="font-size: 0.85rem; color: #616161;">Individual Assessment: <strong>{assessment}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Evidence Presentation
# ---------------------------------------------------------------------------
def render_evidence(evidence: list[dict[str, Any]]) -> None:
    """Render retrieved evidence items with explicit semantic similarity disclaimer."""
    with st.expander("Retrieved Evidence", expanded=True):
        st.caption(
            "Note: Semantic similarity measures retrieval relevance in the knowledge base. "
            "It does NOT by itself prove factual correctness or truth."
        )

        if not evidence:
            st.info(
                "No usable provenance-bearing evidence was found in the current knowledge base. "
                "The system requires verified evidence before confirming or refuting checkable claims."
            )
            return

        for idx, ev in enumerate(evidence, 1):
            source = ev.get("source") or "Reference Document"
            snippet = ev.get("text", "")
            sim = ev.get("similarity_score")
            stance = ev.get("stance") or "unspecified"
            date = ev.get("published_date") or "Date unspecified"
            url = ev.get("url")

            sim_display = f"{sim:.3f}" if isinstance(sim, (int, float)) else "N/A"
            source_link = f"<a href='{url}' target='_blank'>{source}</a>" if url else f"<strong>{source}</strong>"

            st.markdown(
                f"""
                <div style="border: 1px solid #cfd8dc; background: #fafafa; padding: 12px 16px; border-radius: 6px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span>Source #{idx}: {source_link}</span>
                        <span style="font-size: 0.85rem; background: #e0f2f1; color: #004d40; padding: 2px 8px; border-radius: 4px;">
                            Similarity: {sim_display} | Stance: {stance}
                        </span>
                    </div>
                    <div style="font-size: 0.95rem; color: #37474f; font-style: italic; margin-bottom: 6px; line-height: 1.4;">
                        "{snippet}"
                    </div>
                    <div style="font-size: 0.8rem; color: #78909c;">Published: {date}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Classical ML Baseline Presentation
# ---------------------------------------------------------------------------
def render_classical_ml(ml: Optional[dict[str, Any]]) -> None:
    """Render the classical TF-IDF + Logistic Regression baseline signal."""
    with st.expander("Classical ML Signal (TF-IDF + Logistic Regression)", expanded=False):
        st.caption(
            "The classical model was trained on the LIAR political statement dataset. "
            "These scores represent statistical model output, not calibrated probabilities of factual truth."
        )

        if not ml:
            st.write("Classical ML signal unavailable.")
            return

        pred_label = ml.get("label", "N/A")
        probs = ml.get("probabilities", {})

        col1, col2 = st.columns([1, 2])
        with col1:
            st.write(f"**Predicted Category:** `{pred_label}`")
            model_info = f"{ml.get('model_name', 'Logistic Regression')} ({ml.get('model_version', 'v1')})"
            st.caption(f"Model: {model_info}")

        with col2:
            st.write("**Model Class Probabilities (Uncalibrated):**")
            p_rel = probs.get("likely_reliable", 0.0)
            p_fake = probs.get("likely_fake", 0.0)
            p_mis = probs.get("misleading", 0.0)

            st.progress(float(p_rel), text=f"Likely Reliable: {p_rel:.1%}")
            st.progress(float(p_mis), text=f"Misleading: {p_mis:.1%}")
            st.progress(float(p_fake), text=f"Likely Fake: {p_fake:.1%}")


# ---------------------------------------------------------------------------
# Red-Flag / Stylometric Presentation
# ---------------------------------------------------------------------------
def render_red_flags(red_flags: Optional[dict[str, Any]]) -> None:
    """Render deterministic linguistic and presentation warning signals."""
    with st.expander("Language / Presentation Signals (Red-Flag Detector)", expanded=False):
        st.caption(
            "Stylistic indicators detect sensationalism, capitalization, and emotional urgency. "
            "They reflect writing style only and are NOT independent proof that a claim is false."
        )

        if not red_flags:
            st.write("Red-flag detector output unavailable.")
            return

        summary = red_flags.get("summary", {})
        flags = red_flags.get("flags", [])

        h_count = summary.get("high_severity_count", 0)
        m_count = summary.get("medium_severity_count", 0)
        l_count = summary.get("low_severity_count", 0)

        c1, c2, c3 = st.columns(3)
        c1.metric("High Severity", h_count)
        c2.metric("Medium Severity", m_count)
        c3.metric("Low Severity", l_count)

        if not flags:
            st.success("No notable stylistic warning signals detected.")
        else:
            st.write("**Detected Indicators:**")
            for f in flags:
                rule = f.get("rule_name", "Unknown rule")
                sev = f.get("severity", "low").upper()
                msg = f.get("message", "")
                st.markdown(f"- **[{sev}] {rule}**: {msg}")


# ---------------------------------------------------------------------------
# Evidence-Bound LLM Assessment Presentation
# ---------------------------------------------------------------------------
def render_llm_assessment(
    llm: Optional[dict[str, Any]],
    component_statuses: list[dict[str, Any]],
) -> None:
    """Render the structured LLM reasoner assessment or explanation of quota-aware skipping."""
    with st.expander("Evidence-Bound LLM Assessment", expanded=False):
        if llm is not None:
            st.write(f"**LLM Subsystem Verdict:** `{llm.get('label', 'N/A')}`")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"- **Evidence Support:** {llm.get('evidence_support', 'N/A')}")
                st.write(f"- **Source Attribution:** {llm.get('source_attribution', 'N/A')}")
            with col2:
                st.write(f"- **Internal Consistency:** {llm.get('internal_consistency', 'N/A')}")

            reason = llm.get("short_reason")
            if reason:
                st.info(f"**LLM Evaluation Rationale:** {reason}")

            unsupported = llm.get("unsupported_claims", [])
            if unsupported:
                st.write("**Propositions Lacking Corroboration:**")
                for u in unsupported:
                    st.markdown(f"- {u}")

            rhetorical = llm.get("rhetorical_cues", [])
            if rhetorical:
                st.write("**Identified Rhetorical Framing:**")
                for r in rhetorical:
                    st.markdown(f"- {r}")
            return

        # LLM assessment is None: inspect execution status
        llm_status = next(
            (c.get("status") for c in component_statuses if c.get("component") == "llm_reasoner"),
            None,
        )

        if llm_status == "skipped":
            st.info(
                "**LLM evidence reasoning was skipped because no usable external evidence was available.** "
                "This is a quota-aware design policy: the system avoids speculative or ungrounded generative inference."
            )
        elif llm_status in ("unavailable", "failed"):
            st.warning(
                "**LLM component temporarily unavailable; the system safely abstained where required.** "
                "The pipeline degraded gracefully without exposing sensitive provider errors."
            )
        else:
            st.write("LLM evidence reasoning was not invoked for this request.")


# ---------------------------------------------------------------------------
# Pipeline Execution & Technical Details
# ---------------------------------------------------------------------------
def render_pipeline_details(
    component_statuses: list[dict[str, Any]],
    timings: Optional[dict[str, Any]],
    analysis_id: Optional[str],
) -> None:
    """Render engineering audit details including component statuses and latency timings."""
    with st.expander("Pipeline Technical Details", expanded=False):
        if analysis_id:
            st.markdown(f"**Analysis Identifier:** `{analysis_id}`")

        st.write("**Subsystem Execution Statuses:**")
        if component_statuses:
            status_data = []
            for item in component_statuses:
                comp = item.get("component", "Unknown").replace("_", " ").title()
                st_val = (item.get("status") or "unknown").upper()
                msg = item.get("message") or "-"
                status_data.append({"Subsystem": comp, "Execution Status": st_val, "Note": msg})
            st.table(status_data)
        else:
            st.write("Status list unavailable.")

        if timings:
            st.write("**Latency Breakdown (ms):**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Cleaning", f"{timings.get('cleaning_ms', 0):.1f} ms")
            c2.metric("Extraction", f"{timings.get('claim_extraction_ms', 0):.1f} ms")
            c3.metric("Classical ML", f"{timings.get('ml_ms', 0):.1f} ms")
            c4.metric("Red Flags", f"{timings.get('red_flags_ms', 0):.1f} ms")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Retrieval", f"{timings.get('retrieval_ms', 0):.1f} ms")
            c6.metric("LLM Reasoner", f"{timings.get('llm_ms', 0):.1f} ms")
            c7.metric("Fusion", f"{timings.get('fusion_ms', 0):.1f} ms")
            c8.metric("Total Latency", f"{timings.get('total_ms', 0):.1f} ms")
            st.caption("Latency measurements reflect system runtime overhead, not output quality or veracity.")
