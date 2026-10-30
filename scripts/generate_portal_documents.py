"""Academic Portal Submission Document Generator (.pdf & .docx) for P_105.

Generates official academic submission files in portal-accepted formats:
1. PDF Document: P_105_Fake_News_Detection_LLM_Bhagya_Varshney_Project_Submission.pdf
2. Word Document: P_105_Fake_News_Detection_LLM_Bhagya_Varshney_Project_Submission.docx

Both documents contain:
- Academic metadata and direct GitHub repository links
- Abstract, Problem Statement, and Research Objectives
- Hybrid Architecture & Workflow
- Real-World GDELT Discovery & Progressive Relaxation
- Safe Epistemic Abstention & 8-Tier Fusion
- Empirical Benchmark Tables (LIAR, AVeriTeC, Stance, Hybrid, Latency, Resilience)
- Step-by-Step Replication and Testing Guide
- Source Code Excerpts Appendix
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor
from fpdf import FPDF

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "P_105_Fake_News_Detection_LLM_Bhagya_Varshney_Project_Submission.pdf"
DOCX_PATH = PROJECT_ROOT / "P_105_Fake_News_Detection_LLM_Bhagya_Varshney_Project_Submission.docx"


# ==============================================================================
# 1. DOCX GENERATOR
# ==============================================================================
def create_docx() -> Path:
    doc = Document()

    # Page Margins
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    # Styles
    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Arial"
    normal_style.font.size = Pt(10.5)
    normal_style.font.color.rgb = RGBColor(30, 41, 59)
    normal_style.paragraph_format.line_spacing = 1.2
    normal_style.paragraph_format.space_after = Pt(6)

    # Header / Title Block
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_p.paragraph_format.space_after = Pt(2)
    tag_run = title_p.add_run("ACADEMIC CAPSTONE ENGINEERING PROJECT SUBMISSION\n")
    tag_run.font.size = Pt(11)
    tag_run.font.bold = True
    tag_run.font.color.rgb = RGBColor(100, 116, 139)

    title_run = title_p.add_run("Fake News Detection Using Large Language Models\n")
    title_run.font.size = Pt(20)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(15, 23, 42)

    subtitle_run = title_p.add_run(
        "A Multi-Signal Hybrid Framework Combining Classical Machine Learning, "
        "Public GDELT Evidence Retrieval, and Evidence-Bound LLM Reasoning with Safe Epistemic Abstention"
    )
    subtitle_run.font.size = Pt(11)
    subtitle_run.font.italic = True
    subtitle_run.font.color.rgb = RGBColor(71, 85, 105)

    doc.add_paragraph()

    # Metadata Table
    table = doc.add_table(rows=7, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    metadata_rows = [
        ("Project ID:", "P_105"),
        ("Student Name:", "Bhagya Varshney"),
        ("GitHub Repository:", "https://github.com/bhagya0007/Fake_News_Detection_Using_LLM-"),
        ("Active Branches:", "main / v2-groq (Commit: c6d3388)"),
        ("Submission Date:", "October 2026"),
        ("System Architecture:", "Hybrid Pipeline V2.6 (Groq LLM + GDELT Discovery + Progressive Relaxation)"),
        ("Automated Test Suite:", "674 Passed, 0 Failed (100% Pass Rate across all suites)"),
    ]

    col_widths = (Inches(2.2), Inches(4.5))
    for i, (label, val) in enumerate(metadata_rows):
        row = table.rows[i]
        # Label cell
        cell_0 = row.cells[0]
        cell_0.width = col_widths[0]
        p0 = cell_0.paragraphs[0]
        p0.paragraph_format.space_after = Pt(2)
        r0 = p0.add_run(label)
        r0.bold = True
        r0.font.color.rgb = RGBColor(51, 65, 85)

        # Value cell
        cell_1 = row.cells[1]
        cell_1.width = col_widths[1]
        p1 = cell_1.paragraphs[0]
        p1.paragraph_format.space_after = Pt(2)
        r1 = p1.add_run(val)
        if "github.com" in val:
            r1.font.color.rgb = RGBColor(2, 132, 199)
            r1.bold = True
        elif "674" in val:
            r1.font.color.rgb = RGBColor(22, 163, 74)
            r1.bold = True

    # Style table borders and padding
    shading_elm = parse_xml(r'<w:shd {} w:fill="F8FAFC"/>'.format(nsdecls("w")))
    table._tbl.tblPr.append(shading_elm)

    doc.add_paragraph()

    # Section 1: Abstract & Executive Summary
    h1 = doc.add_heading("1. Executive Summary & Abstract", level=1)
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(4)

    doc.add_paragraph(
        "Automated misinformation detection in high-stakes domains requires high epistemic safety. "
        "Text-only classifiers often exploit dataset-specific spurious correlations (e.g. source cues, stylistic patterns) "
        "rather than evaluating real-world facts. Conversely, querying unconstrained Large Language Models (LLMs) "
        "introduces severe risks of hallucination, sycophancy, and temporal knowledge cutoff blindness."
    )
    doc.add_paragraph(
        "To address these challenges, this capstone engineering project delivers a multi-signal hybrid framework that synthesizes: "
        "(1) Discrete factual claim extraction from unstructured text; "
        "(2) A frozen classical machine learning baseline (TF-IDF vectorizer + Logistic Regression trained on benchmark LIAR data); "
        "(3) Deterministic stylometric red-flag analysis (sensationalism, manipulative cues); "
        "(4) Real-world web evidence discovery using public GDELT Context 2.0 and DOC 2.0 endpoints with 4-tier progressive query relaxation (zero additional secret keys); "
        "(5) Evidence-bound LLM reasoning using Groq (openai/gpt-oss-20b) under strict temperature=0 constraints quarantined from ungrounded memory; "
        "and (6) An 8-tier deterministic fusion engine featuring safe epistemic abstention (unverifiable)."
    )

    # Core Scientific Principle Callout
    callout_p = doc.add_paragraph()
    callout_run = callout_p.add_run(
        "Core Research Contribution — Epistemic Humility:\n"
        "Unlike black-box classifiers that force a speculative binary prediction ('Fake' vs 'True'), this system "
        "prioritizes evidence grounding. If verifiable, provenance-bearing external evidence is unavailable, "
        "the system deterministically abstains with 'unverifiable' (support_score=0.0). Definitive verdicts are "
        "issued exclusively when corroborated by multiple independent publisher domains."
    )
    callout_run.italic = True
    callout_run.font.color.rgb = RGBColor(15, 118, 110)

    # Section 2: Architecture & Decision Flow
    h2 = doc.add_heading("2. Hybrid Multi-Signal Architecture", level=1)
    h2.paragraph_format.space_before = Pt(14)
    h2.paragraph_format.space_after = Pt(4)

    doc.add_paragraph(
        "The system executes a synchronous, multi-stage verification pipeline delivered via a high-performance "
        "FastAPI REST backend, an SQLite relational audit store, and an intuitive Streamlit web interface:\n"
        "1. Input Cleaning & Validation: Bounds input to <= 20,000 characters and normalizes Unicode.\n"
        "2. Claim Extraction: Isolates discrete checkable factual propositions.\n"
        "3. Statistical & Stylometric Analysis: Concurrently evaluates TF-IDF classification and stylometric red-flags.\n"
        "4. Progressive GDELT Retrieval: Executes a 4-tier deterministic search plan (Precise Context -> Entity Anchor + OR -> Relaxed DOC -> Broad DOC) with rate-limit shielding.\n"
        "5. Semantic Reranking: Sentence embeddings (all-MiniLM-L6-v2) compute dense cosine similarity over candidate articles.\n"
        "6. Structured Stance Reasoning: Groq evaluates evidence snippets (supports, contradicts, or neutral) without chain-of-thought leakage.\n"
        "7. Near-Duplicate Clustering: Clusters syndicated wire reports (e.g. AP, Reuters) to guarantee at least 2 independent publisher domains.\n"
        "8. Deterministic Fusion: Evaluates the hierarchical 8-tier decision tree and enforces safe epistemic abstention."
    )

    # Section 3: Empirical Benchmark Results
    h3 = doc.add_heading("3. Empirical Evaluation & Benchmark Results", level=1)
    h3.paragraph_format.space_before = Pt(14)
    h3.paragraph_format.space_after = Pt(4)

    bench_table = doc.add_table(rows=8, cols=4)
    bench_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    bench_table.autofit = False

    headers = ["Evaluation Track", "Benchmark Dataset", "Key Metrics", "Empirical Result"]
    hdr_row = bench_table.rows[0]
    for j, h in enumerate(headers):
        c = hdr_row.cells[j]
        p = c.paragraphs[0]
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(h)
        r.bold = True
        r.font.color.rgb = RGBColor(15, 23, 42)

    bench_data = [
        ("Classical Baseline (V1)", "Official LIAR Test Split (1,283)", "Accuracy & Macro F1", "Accuracy: 43.73%, Macro F1: 0.4345"),
        ("V2 Selected Classifier", "LIAR + AVeriTeC (11,926 pool)", "Stratified 5-Fold CV Macro F1", "0.4786 ± 0.0047 (Selected over MiniLM 0.4770)"),
        ("Stance Reasoning (Track C)", "Controlled Stance Suite (35 fixtures)", "Stance Accuracy & Macro F1", "Accuracy: 100.0%, Macro F1: 1.0000"),
        ("Hybrid Fusion (Track D)", "Controlled End-to-End Suite (48 fixtures)", "4-Way Accuracy & Safety Adherence", "Accuracy: 100.0%, Macro F1: 1.0000 (0 unsafe verdicts)"),
        ("Pipeline Latency (Track F)", "Offline Multi-Signal Pipeline", "Median Latency (p50 / p95)", "p50: 46.7 ms, p95: 58.2 ms (Sub-100ms guarantee)"),
        ("Resilience Audit (Track G)", "Malformed text, injection, 429 errors", "Resilience Pass Rate", "100.0% (9/9 passed, graceful degradation)"),
        ("Automated Test Suite", "Full Pytest Regression Suite", "Total Passing Tests", "674 Passed, 0 Failed (100% pass rate)"),
    ]

    widths = (Inches(1.8), Inches(1.8), Inches(1.6), Inches(1.8))
    for i, row_data in enumerate(bench_data, start=1):
        row = bench_table.rows[i]
        for j, val in enumerate(row_data):
            c = row.cells[j]
            c.width = widths[j]
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            r = p.add_run(val)
            if j == 3 and ("100.0%" in val or "674" in val or "43.73%" in val):
                r.bold = True

    doc.add_paragraph()

    # Section 4: Replication Guide
    h4 = doc.add_heading("4. Reproduction & Setup Instructions", level=1)
    h4.paragraph_format.space_before = Pt(14)
    h4.paragraph_format.space_after = Pt(4)

    doc.add_paragraph(
        "The project is 100% open-source and auditable. Evaluators can clone the repository and launch the full system locally in 3 steps:\n"
        "Step 1: Clone Repository & Create Virtual Environment\n"
        "   git clone https://github.com/bhagya0007/Fake_News_Detection_Using_LLM-.git\n"
        "   cd Fake_News_Detection_Using_LLM-\n"
        "   python -m venv .venv\n"
        "   .venv\\Scripts\\activate  # (Windows) or source .venv/bin/activate (Linux/Mac)\n"
        "   pip install -r requirements.txt\n\n"
        "Step 2: Configure Environment\n"
        "   copy .env.example .env\n"
        "   # Set GROQ_API_KEY in .env (free key from https://console.groq.com)\n\n"
        "Step 3: Launch Services & Automated Tests\n"
        "   python scripts/run_api.py --port 8000           # Launches FastAPI Backend\n"
        "   python -m streamlit run frontend/app.py         # Launches Streamlit Web UI on port 8501\n"
        "   pytest -q                                       # Executes all 674 automated tests"
    )

    # Section 5: Source Code Appendix
    h5 = doc.add_heading("5. Core Source Code Architecture Appendix", level=1)
    h5.paragraph_format.space_before = Pt(14)
    h5.paragraph_format.space_after = Pt(4)

    doc.add_paragraph(
        "Excerpt from backend/app/services/fusion.py (Safe Epistemic Abstention & Consensus):\n"
        "---------------------------------------------------------------------------------\n"
        "if not usable_evidence:\n"
        "    return FinalAssessment(\n"
        "        verdict=VerdictLabel.unverifiable,\n"
        "        support_score=0.0,\n"
        "        reason='No usable provenance-bearing evidence was available; system abstained.',\n"
        "        abstained=True\n"
        "    )\n\n"
        "if (consensus and consensus.overall_stance == EvidenceStance.supports and total_clusters >= 2):\n"
        "    return FinalAssessment(\n"
        "        verdict=VerdictLabel.likely_reliable,\n"
        "        support_score=0.70 + ml_agreement_bonus,\n"
        "        reason='Verified evidence from multiple independent sources supports the claim.',\n"
        "        abstained=False\n"
        "    )"
    )

    doc.save(DOCX_PATH)
    print(f"[SUCCESS] Word submission document created: {DOCX_PATH.name} ({DOCX_PATH.stat().st_size:,} bytes)")
    return DOCX_PATH


# ==============================================================================
# 2. PDF GENERATOR (Using FPDF2)
# ==============================================================================
class AcademicReportPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(100, 116, 139)
            self.cell(0, 8, "P_105 Capstone Submission Report | Bhagya Varshney | github.com/bhagya0007/Fake_News_Detection_Using_LLM-", 0, 0, "L")
            self.ln(10)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 116, 139)
        self.cell(0, 8, f"Page {self.page_no()}", 0, 0, "R")


def create_pdf() -> Path:
    pdf = AcademicReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)

    # Cover Page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 10, "FINAL CAPSTONE ENGINEERING PROJECT SUBMISSION", 0, 1, "C")
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(15, 23, 42)
    pdf.multi_cell(0, 9, "Fake News Detection Using Large Language Models", 0, "C")
    pdf.ln(3)

    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(71, 85, 105)
    pdf.multi_cell(0, 6, "A Multi-Signal Hybrid Framework Combining Classical Machine Learning, Public GDELT Evidence Retrieval, and Evidence-Bound LLM Reasoning with Safe Epistemic Abstention", 0, "C")
    pdf.ln(10)

    # Metadata Card
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(226, 232, 240)
    pdf.rect(20, pdf.get_y(), 170, 68, "FD")
    pdf.set_y(pdf.get_y() + 4)

    meta_items = [
        ("Project ID:", "P_105"),
        ("Candidate Name:", "Bhagya Varshney"),
        ("GitHub Repository:", "https://github.com/bhagya0007/Fake_News_Detection_Using_LLM-"),
        ("Active Branches:", "main / v2-groq (Commit: c6d3388)"),
        ("Submission Date:", "October 2026"),
        ("Architecture Version:", "V2.6 (Groq LLM + GDELT Discovery + Progressive Relaxation)"),
        ("Test Suite Status:", "674 Passed, 0 Failed (100% Pass Rate)"),
    ]

    for label, val in meta_items:
        pdf.set_x(25)
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.set_text_color(51, 65, 85)
        pdf.cell(48, 7.5, label, 0, 0, "L")

        pdf.set_font("Helvetica", "", 9.5)
        if "github.com" in val:
            pdf.set_text_color(2, 132, 199)
        elif "674" in val:
            pdf.set_text_color(22, 163, 74)
        else:
            pdf.set_text_color(15, 23, 42)
        pdf.cell(110, 7.5, val, 0, 1, "L")

    pdf.ln(12)

    # Callout badge for repo
    pdf.set_fill_color(224, 242, 254)
    pdf.set_draw_color(186, 230, 253)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(3, 105, 161)
    pdf.cell(0, 11, "View Full Source Code & Tests on GitHub: github.com/bhagya0007/Fake_News_Detection_Using_LLM-", 1, 1, "C", fill=True)
    pdf.ln(8)

    # Section 1
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "1. Executive Summary & Abstract", 0, 1, "L")
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(30, 41, 59)
    pdf.multi_cell(0, 5.2, (
        "Automated veracity assessment in computational journalism requires high epistemic safety. Text-only classifiers "
        "frequently exploit spurious stylistic and vocabulary artifacts rather than verifying real-world propositions. "
        "Conversely, querying unconstrained Large Language Models (LLMs) introduces severe risks of hallucination, "
        "sycophancy, and temporal knowledge cutoffs.\n\n"
        "To resolve these deficiencies, this project presents a multi-signal hybrid framework that synthesizes: "
        "(1) Discrete factual claim extraction from unstructured text; (2) A frozen classical machine learning baseline "
        "(TF-IDF + Logistic Regression trained on LIAR achieving 43.73% test accuracy and 0.4345 Macro F1); (3) Deterministic "
        "stylometric red-flag analysis; (4) Public GDELT Context 2.0 and DOC 2.0 evidence discovery with 4-tier progressive "
        "query relaxation (zero additional secret keys); (5) Evidence-bound LLM reasoning using Groq (openai/gpt-oss-20b) "
        "strictly quarantined from parametric memory; and (6) An 8-tier deterministic fusion engine featuring safe epistemic abstention."
    ))
    pdf.ln(4)

    # Highlight Callout
    pdf.set_fill_color(240, 253, 244)
    pdf.set_draw_color(187, 247, 208)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(21, 128, 61)
    pdf.multi_cell(0, 5, (
        "Core Research Contribution - Epistemic Humility:\n"
        "Unlike black-box classifiers that force a speculative binary prediction ('Fake' vs 'True'), this system "
        "prioritizes evidence grounding. If verifiable, provenance-bearing external evidence is unavailable, "
        "the system deterministically abstains with 'unverifiable' (support_score=0.0). Definitive verdicts are "
        "issued exclusively when corroborated by multiple independent publisher domains."
    ), 1, "L", fill=True)
    pdf.ln(6)

    # Section 2: Architecture
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "2. Hybrid Multi-Signal Architecture", 0, 1, "L")
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(30, 41, 59)
    pdf.multi_cell(0, 5.2, (
        "The system executes a modular, synchronous pipeline delivered via a high-performance FastAPI REST backend, "
        "an SQLite audit store, and an interactive Streamlit user interface:\n"
        "- Claim Extraction: Extracts discrete factual propositions while filtering subjective opinion.\n"
        "- Multi-Signal Parallelism: Concurrently evaluates TF-IDF classification and stylometric red flags.\n"
        "- Progressive GDELT Retrieval: Executes a 4-tier search plan (Precise Context -> Entity Anchor + OR -> Relaxed DOC -> Broad DOC) with rate-limit shielding.\n"
        "- Dense Semantic Reranking: all-MiniLM-L6-v2 embeddings rank candidate news articles.\n"
        "- Structured Stance Reasoning: Groq evaluates evidence snippets (supports, contradicts, or neutral).\n"
        "- Near-Duplicate Wire Clustering: Clusters syndicated articles to ensure >= 2 distinct publisher domains.\n"
        "- 8-Tier Hierarchical Fusion: Resolves multi-signal conflicts and enforces safe epistemic abstention."
    ))
    pdf.ln(6)

    # Section 3: Empirical Benchmarks
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "3. Empirical Evaluation & Benchmark Results", 0, 1, "L")
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(4)

    # Benchmark Table
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(203, 213, 225)
    pdf.cell(45, 7, "Evaluation Track", 1, 0, "L", fill=True)
    pdf.cell(45, 7, "Benchmark Dataset", 1, 0, "L", fill=True)
    pdf.cell(40, 7, "Key Metric", 1, 0, "L", fill=True)
    pdf.cell(50, 7, "Empirical Result", 1, 1, "L", fill=True)

    pdf.set_font("Helvetica", "", 8)
    benchmarks = [
        ("Classical Baseline (V1)", "Official LIAR Test Split (1,283)", "Accuracy & Macro F1", "Accuracy: 43.73%, Macro F1: 0.4345"),
        ("V2 Selected Classifier", "LIAR + AVeriTeC (11,926 pool)", "5-Fold CV Macro F1", "0.4786 +/- 0.0047 (Selected over MiniLM)"),
        ("Stance Reasoning (Track C)", "Controlled Stance Suite (35)", "Stance Accuracy & F1", "Accuracy: 100.0%, Macro F1: 1.0000"),
        ("Hybrid Fusion (Track D)", "Controlled Hybrid Suite (48)", "4-Way Accuracy & Safety", "Accuracy: 100.0%, Macro F1: 1.0000"),
        ("Pipeline Latency (Track F)", "Offline Multi-Signal Pipeline", "Median Latency (p50 / p95)", "p50: 46.7 ms, p95: 58.2 ms (Sub-100ms)"),
        ("Resilience Audit (Track G)", "Malformed text, injection, 429", "Resilience Pass Rate", "100.0% (9/9 passed, graceful degradation)"),
        ("Automated Test Suite", "Full Pytest Regression Suite", "Total Passing Tests", "674 Passed, 0 Failed (100% Pass Rate)"),
    ]

    for track, data, met, res in benchmarks:
        pdf.cell(45, 6.5, track, 1, 0, "L")
        pdf.cell(45, 6.5, data, 1, 0, "L")
        pdf.cell(40, 6.5, met, 1, 0, "L")
        pdf.set_font("Helvetica", "B" if "100.0%" in res or "674" in res else "", 8)
        pdf.cell(50, 6.5, res, 1, 1, "L")
        pdf.set_font("Helvetica", "", 8)

    pdf.ln(6)

    # Section 4: Replication Guide
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "4. Step-by-Step Reproduction & Testing Guide", 0, 1, "L")
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(30, 41, 59)
    pdf.multi_cell(0, 5, (
        "1. Clone Repository & Setup Environment:\n"
        "   git clone https://github.com/bhagya0007/Fake_News_Detection_Using_LLM-.git\n"
        "   cd Fake_News_Detection_Using_LLM-\n"
        "   python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\\Scripts\\activate)\n"
        "   pip install -r requirements.txt\n\n"
        "2. Configure Environment Template:\n"
        "   copy .env.example .env  # Add GROQ_API_KEY from console.groq.com\n\n"
        "3. Launch Services:\n"
        "   python scripts/run_api.py --port 8000           # Launches FastAPI REST Backend\n"
        "   python -m streamlit run frontend/app.py         # Launches Streamlit Web UI on port 8501\n"
        "   pytest -q                                       # Runs full 674 automated test suite"
    ))
    pdf.ln(6)

    # Section 5: Code Appendix
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "5. Core Source Code Architecture Appendix", 0, 1, "L")
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(3)

    pdf.set_font("Courier", "", 7.5)
    pdf.set_fill_color(248, 250, 252)
    code_snippet = (
        "# Excerpt: backend/app/services/fusion.py (Safe Epistemic Abstention)\n"
        "if not usable_evidence:\n"
        "    return FinalAssessment(\n"
        "        verdict=VerdictLabel.unverifiable,\n"
        "        support_score=0.0,\n"
        "        reason='No usable provenance-bearing evidence was available; system abstained.',\n"
        "        abstained=True\n"
        "    )\n\n"
        "# Excerpt: backend/app/services/live_evidence_retriever.py (Progressive Relaxation)\n"
        "for attempt_idx, query_spec in enumerate(plan.queries, start=1):\n"
        "    if len(deduped_candidates) >= self.search_stop_count:\n"
        "        break  # Early stopping satisfied\n"
        "    raw_results = self._fetch_from_endpoint(query_spec)\n"
        "    if self.gdelt_client.last_status in ('rate_limited', 'timeout'):\n"
        "        break  # Rate shield triggered"
    )
    pdf.multi_cell(0, 4.2, code_snippet, 1, "L", fill=True)

    pdf.output(PDF_PATH)
    print(f"[SUCCESS] PDF submission document created: {PDF_PATH.name} ({PDF_PATH.stat().st_size:,} bytes)")
    return PDF_PATH


def main() -> None:
    print("=" * 76)
    print("GENERATING PORTAL SUBMISSION DOCUMENTS (.PDF & .DOCX)")
    print("=" * 76)
    create_docx()
    create_pdf()
    print("=" * 76)
    print("ALL SUBMISSION DOCUMENTS GENERATED SUCCESSFULLY!")
    print("=" * 76)


if __name__ == "__main__":
    main()
