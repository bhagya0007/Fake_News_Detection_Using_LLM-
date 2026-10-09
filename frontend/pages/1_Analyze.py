"""Analyze Statement Page (Step 18)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from frontend.api_client import APIClient
from frontend.views import render_analyze_view, render_sidebar_status

st.set_page_config(
    page_title="Analyze | Fake News Detection",
    page_icon="🔎",
    layout="wide",
)

client = APIClient()
is_online = render_sidebar_status(client)
render_analyze_view(client=client, is_online=is_online)
