"""Main Streamlit Application Entry Point (Step 18).

Provides a clean, academic, responsive interface for the Fake News Detection system.
Communicates strictly via the FastAPI backend REST API.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path for direct `streamlit run frontend/app.py`
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from frontend.api_client import APIClient
from frontend.views import (
    render_about_view,
    render_analyze_view,
    render_history_view,
    render_sidebar_status,
)


def main() -> None:
    """Streamlit application execution entry point."""
    st.set_page_config(
        page_title="Fake News Detection Using LLMs",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    client = APIClient()

    # Render backend status indicator in the sidebar
    is_online = render_sidebar_status(client)

    # Navigation selector in sidebar
    st.sidebar.markdown("### Navigation")
    nav_selection = st.sidebar.radio(
        label="Select Page:",
        options=["Analyze", "Analysis History", "About & Architecture"],
        index=0,
        label_visibility="collapsed",
    )

    # Render the active view
    if nav_selection == "Analyze":
        render_analyze_view(client=client, is_online=is_online)
    elif nav_selection == "Analysis History":
        render_history_view(client=client, is_online=is_online)
    elif nav_selection == "About & Architecture":
        render_about_view()


if __name__ == "__main__":
    main()
