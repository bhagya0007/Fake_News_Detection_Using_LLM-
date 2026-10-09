"""Evaluation Dataset Generator Script (Step 19).

Generates standardized evaluation datasets in `data/evaluation/`:
1. `hybrid_controlled_eval.jsonl` (44 balanced controlled multi-signal test cases across 15 behavioral categories)
2. `production_abstention_eval.json` (20 diverse statements across 10+ domains tested against empty production KB)
3. `claim_extraction_eval.json` (15 synthetic cases testing checkability, atomic isolation, quotes, numbers)
4. `retrieval_eval.json` (16 test queries testing Recall@1, Recall@3, and no-match rejection against demo KB)

Purely synthetic/controlled fixtures. Zero modification to `data/raw/test.tsv`.
Zero external network calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT_DIR / "data" / "evaluation"


def generate_controlled_benchmark() -> List[Dict[str, Any]]:
    """Build 44 balanced controlled test cases across 15 behavioral categories."""
    cases: List[Dict[str, Any]] = [
        # =====================================================================
        # CATEGORY 1: Supported Claims (Expected: likely_reliable, abstained: False)
        # =====================================================================
        {
            "case_id": "ctrl_001",
            "category": "supported_claim",
            "description": "Directly supported by synthetic press release fixture.",
            "input_text": "Acme Corp announced today that it will construct a new manufacturing facility in Texas to expand clean energy production.",
            "expected_verdict": "likely_reliable",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Corp announced it will construct a new manufacturing facility in Texas to expand clean energy production.",
                    "checkable": True,
                    "claim_type": "event",
                    "source_span": "Acme Corp announced today that it will construct a new manufacturing facility in Texas to expand clean energy production.",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_001",
                    "claim_summary": "Acme Corp announced a new factory in Texas",
                    "evidence_text": "Acme Corp announced today that it will construct a new manufacturing facility in Texas to expand clean energy production.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-texas",
                    "published_date": "2024-03-15",
                    "stance": "supports",
                    "source_type": "press_release",
                    "is_synthetic": True,
                    "similarity_score": 0.94,
                }
            ],
            "mock_ml": {
                "label": "likely_reliable",
                "probabilities": {"likely_reliable": 0.72, "likely_fake": 0.16, "misleading": 0.12},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 0, "high_severity_flags": 0},
            "mock_llm": {
                "label": "likely_reliable",
                "evidence_support": "supported",
                "source_attribution": "strong",
                "internal_consistency": "consistent",
                "short_reason": "Verified by company announcement.",
            },
        },
        {
            "case_id": "ctrl_002",
            "category": "supported_claim",
            "description": "Supported hiring plan claim with verified company announcement.",
            "input_text": "Acme Corp plans to recruit and hire 3,000 workers across engineering and assembly lines over the next two years.",
            "expected_verdict": "likely_reliable",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Corp plans to recruit and hire 3,000 workers over the next two years.",
                    "checkable": True,
                    "claim_type": "event",
                    "source_span": "Acme Corp plans to recruit and hire 3,000 workers across engineering and assembly lines over the next two years.",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_002",
                    "claim_summary": "Acme Corp plans to hire 3,000 workers",
                    "evidence_text": "Acme Corp confirmed plans to recruit and hire 3,000 workers across engineering and assembly lines over the next two years.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-hiring",
                    "published_date": "2024-03-20",
                    "stance": "supports",
                    "source_type": "press_release",
                    "is_synthetic": True,
                    "similarity_score": 0.91,
                }
            ],
            "mock_ml": {
                "label": "likely_reliable",
                "probabilities": {"likely_reliable": 0.65, "likely_fake": 0.15, "misleading": 0.20},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 0, "high_severity_flags": 0},
            "mock_llm": {
                "label": "likely_reliable",
                "evidence_support": "supported",
                "source_attribution": "strong",
                "internal_consistency": "consistent",
                "short_reason": "Direct corporate statement confirms the hiring target.",
            },
        },
        {
            "case_id": "ctrl_003",
            "category": "supported_claim",
            "description": "Technical lab specification confirmed by laboratory evaluation record.",
            "input_text": "SolarTech prototype solid-state batteries recorded an energy density of 450 Wh/kg in official laboratory evaluations.",
            "expected_verdict": "likely_reliable",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "SolarTech prototype solid-state batteries recorded an energy density of 450 Wh/kg.",
                    "checkable": True,
                    "claim_type": "numeric",
                    "source_span": "SolarTech prototype solid-state batteries recorded an energy density of 450 Wh/kg in official laboratory evaluations.",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_005",
                    "claim_summary": "SolarTech battery efficiency testing",
                    "evidence_text": "Laboratory evaluations of SolarTech prototype solid-state batteries recorded an energy density of 450 Wh/kg under standard conditions.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/solartech-battery",
                    "published_date": "2024-04-18",
                    "stance": "supports",
                    "source_type": "technical_report",
                    "is_synthetic": True,
                    "similarity_score": 0.89,
                }
            ],
            "mock_ml": {
                "label": "likely_reliable",
                "probabilities": {"likely_reliable": 0.58, "likely_fake": 0.20, "misleading": 0.22},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 0, "high_severity_flags": 0},
            "mock_llm": {
                "label": "likely_reliable",
                "evidence_support": "supported",
                "source_attribution": "strong",
                "internal_consistency": "consistent",
                "short_reason": "Technical report corroborates the specified energy density.",
            },
        },
        {
            "case_id": "ctrl_004",
            "category": "supported_claim",
            "description": "Supported industrial manufacturing volume with independent index citation.",
            "input_text": "The industrial index confirmed that Acme Corp manufactured 250,000 components during the first quarter of 2024.",
            "expected_verdict": "likely_reliable",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Corp manufactured 250,000 components in Q1 2024.",
                    "checkable": True,
                    "claim_type": "numeric",
                    "source_span": "The industrial index confirmed that Acme Corp manufactured 250,000 components during the first quarter of 2024.",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_003",
                    "claim_summary": "Acme Corp quarterly production report",
                    "evidence_text": "The quarterly industrial index noted that Acme Corp manufactured 250,000 components in Q1 2024.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-q1",
                    "published_date": "2024-04-05",
                    "stance": "supports",
                    "source_type": "industry_report",
                    "is_synthetic": True,
                    "similarity_score": 0.92,
                }
            ],
            "mock_ml": {
                "label": "likely_reliable",
                "probabilities": {"likely_reliable": 0.62, "likely_fake": 0.18, "misleading": 0.20},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 0, "high_severity_flags": 0},
            "mock_llm": {
                "label": "likely_reliable",
                "evidence_support": "supported",
                "source_attribution": "strong",
                "internal_consistency": "consistent",
                "short_reason": "Verified by industrial index production figures.",
            },
        },

        # =====================================================================
        # CATEGORY 2: ML / LLM Disagreement (Evidence-bound LLM takes precedence)
        # =====================================================================
        {
            "case_id": "ctrl_005",
            "category": "ml_llm_disagreement",
            "description": "Classical ML predicts likely_fake due to lexical patterns, but factual evidence proves reliable.",
            "input_text": "Shocking new manufacturing plant in Texas approved by Acme Corp to produce clean energy equipment.",
            "expected_verdict": "likely_reliable",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Corp approved a new manufacturing plant in Texas for clean energy equipment.",
                    "checkable": True,
                    "claim_type": "event",
                    "source_span": "manufacturing plant in Texas approved by Acme Corp",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_001",
                    "claim_summary": "Acme Corp announced a new factory in Texas",
                    "evidence_text": "Acme Corp announced today that it will construct a new manufacturing facility in Texas to expand clean energy production.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-texas",
                    "published_date": "2024-03-15",
                    "stance": "supports",
                    "source_type": "press_release",
                    "is_synthetic": True,
                    "similarity_score": 0.85,
                }
            ],
            "mock_ml": {
                "label": "likely_fake",
                "probabilities": {"likely_reliable": 0.20, "likely_fake": 0.65, "misleading": 0.15},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 1, "high_severity_flags": 0},
            "mock_llm": {
                "label": "likely_reliable",
                "evidence_support": "supported",
                "source_attribution": "strong",
                "internal_consistency": "consistent",
                "short_reason": "Grounded evidence corroborates facility construction despite sensational word choice.",
            },
        },
        {
            "case_id": "ctrl_006",
            "category": "ml_llm_disagreement",
            "description": "Classical ML predicts misleading, but factual evidence confirms reliable claim.",
            "input_text": "Acme Corp confirmed hiring 3,000 employees over two years across its engineering divisions.",
            "expected_verdict": "likely_reliable",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Corp confirmed hiring 3,000 employees over two years.",
                    "checkable": True,
                    "claim_type": "event",
                    "source_span": "Acme Corp confirmed hiring 3,000 employees over two years",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_002",
                    "claim_summary": "Acme Corp plans to hire 3,000 workers",
                    "evidence_text": "Acme Corp confirmed plans to recruit and hire 3,000 workers across engineering and assembly lines over the next two years.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-hiring",
                    "published_date": "2024-03-20",
                    "stance": "supports",
                    "source_type": "press_release",
                    "is_synthetic": True,
                    "similarity_score": 0.88,
                }
            ],
            "mock_ml": {
                "label": "misleading",
                "probabilities": {"likely_reliable": 0.25, "likely_fake": 0.25, "misleading": 0.50},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 0, "high_severity_flags": 0},
            "mock_llm": {
                "label": "likely_reliable",
                "evidence_support": "supported",
                "source_attribution": "strong",
                "internal_consistency": "consistent",
                "short_reason": "Verified corporate announcement directly corroborates the figure.",
            },
        },
        {
            "case_id": "ctrl_007",
            "category": "ml_llm_disagreement",
            "description": "Classical ML predicts likely_reliable due to formal syntax, but factual evidence refutes CEO resignation.",
            "input_text": "The chief executive officer of Acme Corp tendered his resignation to the board this morning.",
            "expected_verdict": "likely_fake",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "The chief executive officer of Acme Corp resigned this morning.",
                    "checkable": True,
                    "claim_type": "event",
                    "source_span": "The chief executive officer of Acme Corp tendered his resignation to the board this morning.",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_004",
                    "claim_summary": "Acme Corp CEO resignation rumors",
                    "evidence_text": "Acme Corp board of directors confirmed that the chief executive officer has not resigned and remains actively leading the firm.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-ceo",
                    "published_date": "2024-04-12",
                    "stance": "contradicts",
                    "source_type": "official_statement",
                    "is_synthetic": True,
                    "similarity_score": 0.89,
                }
            ],
            "mock_ml": {
                "label": "likely_reliable",
                "probabilities": {"likely_reliable": 0.68, "likely_fake": 0.16, "misleading": 0.16},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 0, "high_severity_flags": 0},
            "mock_llm": {
                "label": "likely_fake",
                "evidence_support": "contradicted",
                "source_attribution": "weak",
                "internal_consistency": "contradictory",
                "short_reason": "Official board statement directly contradicts reported resignation.",
            },
        },
        {
            "case_id": "ctrl_008",
            "category": "ml_llm_disagreement",
            "description": "Classical ML predicts misleading, but factual evidence directly refutes the claim.",
            "input_text": "Acme Corp CEO has formally stepped down from leadership duties.",
            "expected_verdict": "likely_fake",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Corp CEO stepped down from leadership duties.",
                    "checkable": True,
                    "claim_type": "event",
                    "source_span": "Acme Corp CEO has formally stepped down from leadership duties.",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_004",
                    "claim_summary": "Acme Corp CEO resignation rumors",
                    "evidence_text": "Acme Corp board of directors confirmed that the chief executive officer has not resigned and remains actively leading the firm.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-ceo",
                    "published_date": "2024-04-12",
                    "stance": "contradicts",
                    "source_type": "official_statement",
                    "is_synthetic": True,
                    "similarity_score": 0.87,
                }
            ],
            "mock_ml": {
                "label": "misleading",
                "probabilities": {"likely_reliable": 0.20, "likely_fake": 0.35, "misleading": 0.45},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 0, "high_severity_flags": 0},
            "mock_llm": {
                "label": "likely_fake",
                "evidence_support": "contradicted",
                "source_attribution": "weak",
                "internal_consistency": "contradictory",
                "short_reason": "Corporate board explicitly denied that CEO stepped down.",
            },
        },

        # =====================================================================
        # CATEGORY 3: Red Flags Supporting Evidence (Sensational style, but true)
        # =====================================================================
        {
            "case_id": "ctrl_009",
            "category": "red_flags_supporting_evidence",
            "description": "Sensational text with excessive caps and urgency, but verified evidence confirms fact.",
            "input_text": "BREAKING NEWS ALERT!!! ACME CORP JUST ANNOUNCED A HUGE NEW FACTORY IN TEXAS TO EXPAND CLEAN ENERGY!!!",
            "expected_verdict": "likely_reliable",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Corp announced a new factory in Texas to expand clean energy.",
                    "checkable": True,
                    "claim_type": "event",
                    "source_span": "ACME CORP JUST ANNOUNCED A HUGE NEW FACTORY IN TEXAS",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_001",
                    "claim_summary": "Acme Corp announced a new factory in Texas",
                    "evidence_text": "Acme Corp announced today that it will construct a new manufacturing facility in Texas to expand clean energy production.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-texas",
                    "published_date": "2024-03-15",
                    "stance": "supports",
                    "source_type": "press_release",
                    "is_synthetic": True,
                    "similarity_score": 0.88,
                }
            ],
            "mock_ml": {
                "label": "likely_reliable",
                "probabilities": {"likely_reliable": 0.60, "likely_fake": 0.25, "misleading": 0.15},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 4, "high_severity_flags": 2},
            "mock_llm": {
                "label": "likely_reliable",
                "evidence_support": "supported",
                "source_attribution": "strong",
                "internal_consistency": "consistent",
                "short_reason": "Verified factual expansion confirmed despite sensationalized presentation.",
            },
        },
        {
            "case_id": "ctrl_010",
            "category": "red_flags_supporting_evidence",
            "description": "High urgency and exclamation points, but corroborated by verified hiring report.",
            "input_text": "MASSIVE HIRING SPREE UNLEASHED: Acme Corp is urgently hiring 3,000 workers right now across all engineering lines!!!",
            "expected_verdict": "likely_reliable",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Corp is hiring 3,000 workers across engineering lines.",
                    "checkable": True,
                    "claim_type": "event",
                    "source_span": "Acme Corp is urgently hiring 3,000 workers right now across all engineering lines",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_002",
                    "claim_summary": "Acme Corp plans to hire 3,000 workers",
                    "evidence_text": "Acme Corp confirmed plans to recruit and hire 3,000 workers across engineering and assembly lines over the next two years.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-hiring",
                    "published_date": "2024-03-20",
                    "stance": "supports",
                    "source_type": "press_release",
                    "is_synthetic": True,
                    "similarity_score": 0.87,
                }
            ],
            "mock_ml": {
                "label": "likely_reliable",
                "probabilities": {"likely_reliable": 0.55, "likely_fake": 0.30, "misleading": 0.15},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 3, "high_severity_flags": 2},
            "mock_llm": {
                "label": "likely_reliable",
                "evidence_support": "supported",
                "source_attribution": "strong",
                "internal_consistency": "consistent",
                "short_reason": "Verified recruitment target confirmed; style is hyperbolic but core fact is accurate.",
            },
        },

        # =====================================================================
        # CATEGORY 4: Attribution-Sensitive Claims
        # =====================================================================
        {
            "case_id": "ctrl_011",
            "category": "attribution_sensitive",
            "description": "Accurately attributed statement corroborated by evidence.",
            "input_text": "Acme Corp announced that its new manufacturing facility in Texas will expand clean energy production.",
            "expected_verdict": "likely_reliable",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Corp announced its Texas facility will expand clean energy production.",
                    "checkable": True,
                    "claim_type": "attribution",
                    "source_span": "Acme Corp announced that its new manufacturing facility in Texas will expand clean energy production.",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_001",
                    "claim_summary": "Acme Corp announced a new factory in Texas",
                    "evidence_text": "Acme Corp announced today that it will construct a new manufacturing facility in Texas to expand clean energy production.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-texas",
                    "published_date": "2024-03-15",
                    "stance": "supports",
                    "source_type": "press_release",
                    "is_synthetic": True,
                    "similarity_score": 0.95,
                }
            ],
            "mock_ml": {
                "label": "likely_reliable",
                "probabilities": {"likely_reliable": 0.70, "likely_fake": 0.15, "misleading": 0.15},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 0, "high_severity_flags": 0},
            "mock_llm": {
                "label": "likely_reliable",
                "evidence_support": "supported",
                "source_attribution": "strong",
                "internal_consistency": "consistent",
                "short_reason": "Attributed statement matches official announcement verbatim.",
            },
        },
        {
            "case_id": "ctrl_012",
            "category": "attribution_sensitive",
            "description": "Statement misattributes context, making it misleading.",
            "input_text": "The industrial index claimed Acme Corp's component output was strictly limited to 250,000 for the entire fiscal year.",
            "expected_verdict": "misleading",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "The industrial index claimed Acme output was limited to 250,000 for the full year.",
                    "checkable": True,
                    "claim_type": "attribution",
                    "source_span": "The industrial index claimed Acme Corp's component output was strictly limited to 250,000 for the entire fiscal year.",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_003",
                    "claim_summary": "Acme Corp quarterly production report",
                    "evidence_text": "The quarterly industrial index noted that Acme Corp manufactured 250,000 components in Q1 2024.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-q1",
                    "published_date": "2024-04-05",
                    "stance": "supports",
                    "source_type": "industry_report",
                    "is_synthetic": True,
                    "similarity_score": 0.84,
                }
            ],
            "mock_ml": {
                "label": "misleading",
                "probabilities": {"likely_reliable": 0.20, "likely_fake": 0.30, "misleading": 0.50},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 0, "high_severity_flags": 0},
            "mock_llm": {
                "label": "misleading",
                "evidence_support": "partial",
                "source_attribution": "partial",
                "internal_consistency": "questionable",
                "short_reason": "Index recorded 250,000 in Q1 alone, not the entire fiscal year.",
            },
        },
        {
            "case_id": "ctrl_013",
            "category": "attribution_sensitive",
            "description": "Fabricated quote attributed to official board members.",
            "input_text": "The Acme Corp board announced in a statement: 'Our CEO has resigned effective immediately due to financial misconduct.'",
            "expected_verdict": "likely_fake",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "The board stated the CEO resigned due to financial misconduct.",
                    "checkable": True,
                    "claim_type": "attribution",
                    "source_span": "The Acme Corp board announced in a statement: 'Our CEO has resigned effective immediately due to financial misconduct.'",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_004",
                    "claim_summary": "Acme Corp CEO resignation rumors",
                    "evidence_text": "Acme Corp board of directors confirmed that the chief executive officer has not resigned and remains actively leading the firm.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-ceo",
                    "published_date": "2024-04-12",
                    "stance": "contradicts",
                    "source_type": "official_statement",
                    "is_synthetic": True,
                    "similarity_score": 0.88,
                }
            ],
            "mock_ml": {
                "label": "likely_fake",
                "probabilities": {"likely_reliable": 0.15, "likely_fake": 0.70, "misleading": 0.15},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 1, "high_severity_flags": 0},
            "mock_llm": {
                "label": "likely_fake",
                "evidence_support": "contradicted",
                "source_attribution": "none",
                "internal_consistency": "contradictory",
                "short_reason": "Board confirmed CEO remains in place; attributed quote is fabricated.",
            },
        },

        # =====================================================================
        # CATEGORY 5: Numeric / Statistical Claims
        # =====================================================================
        {
            "case_id": "ctrl_014",
            "category": "numeric_statistical_claim",
            "description": "Exact numerical figure corroborated by verified report.",
            "input_text": "Official industrial records show Acme Corp produced 250,000 components in Q1 2024.",
            "expected_verdict": "likely_reliable",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Corp produced 250,000 components in Q1 2024.",
                    "checkable": True,
                    "claim_type": "numeric",
                    "source_span": "Acme Corp produced 250,000 components in Q1 2024.",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_003",
                    "claim_summary": "Acme Corp quarterly production report",
                    "evidence_text": "The quarterly industrial index noted that Acme Corp manufactured 250,000 components in Q1 2024.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-q1",
                    "published_date": "2024-04-05",
                    "stance": "supports",
                    "source_type": "industry_report",
                    "is_synthetic": True,
                    "similarity_score": 0.93,
                }
            ],
            "mock_ml": {
                "label": "likely_reliable",
                "probabilities": {"likely_reliable": 0.64, "likely_fake": 0.16, "misleading": 0.20},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 0, "high_severity_flags": 0},
            "mock_llm": {
                "label": "likely_reliable",
                "evidence_support": "supported",
                "source_attribution": "strong",
                "internal_consistency": "consistent",
                "short_reason": "Numerical count of 250,000 matches verified quarterly records.",
            },
        },
        {
            "case_id": "ctrl_015",
            "category": "numeric_statistical_claim",
            "description": "Completely fabricated numerical figure refuted by verified report.",
            "input_text": "Acme Corp manufactured over 25,000,000 industrial components in Q1 2024 alone.",
            "expected_verdict": "likely_fake",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Corp manufactured over 25,000,000 components in Q1 2024.",
                    "checkable": True,
                    "claim_type": "numeric",
                    "source_span": "Acme Corp manufactured over 25,000,000 industrial components in Q1 2024 alone.",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_003",
                    "claim_summary": "Acme Corp quarterly production report",
                    "evidence_text": "The quarterly industrial index noted that Acme Corp manufactured 250,000 components in Q1 2024.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-q1",
                    "published_date": "2024-04-05",
                    "stance": "contradicts",
                    "source_type": "industry_report",
                    "is_synthetic": True,
                    "similarity_score": 0.86,
                }
            ],
            "mock_ml": {
                "label": "likely_fake",
                "probabilities": {"likely_reliable": 0.15, "likely_fake": 0.72, "misleading": 0.13},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 1, "high_severity_flags": 0},
            "mock_llm": {
                "label": "likely_fake",
                "evidence_support": "contradicted",
                "source_attribution": "weak",
                "internal_consistency": "contradictory",
                "short_reason": "Output was 250,000; the claimed 25,000,000 figure is a 100x fabrication.",
            },
        },
        {
            "case_id": "ctrl_016",
            "category": "numeric_statistical_claim",
            "description": "Cherry-picked timeframe makes hiring target misleading.",
            "input_text": "Acme Corp is hiring 3,000 workers this month to address manufacturing shortages.",
            "expected_verdict": "misleading",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Corp is hiring 3,000 workers this month.",
                    "checkable": True,
                    "claim_type": "numeric",
                    "source_span": "Acme Corp is hiring 3,000 workers this month",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_002",
                    "claim_summary": "Acme Corp plans to hire 3,000 workers",
                    "evidence_text": "Acme Corp confirmed plans to recruit and hire 3,000 workers across engineering and assembly lines over the next two years.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-hiring",
                    "published_date": "2024-03-20",
                    "stance": "supports",
                    "source_type": "press_release",
                    "is_synthetic": True,
                    "similarity_score": 0.89,
                }
            ],
            "mock_ml": {
                "label": "misleading",
                "probabilities": {"likely_reliable": 0.22, "likely_fake": 0.28, "misleading": 0.50},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 0, "high_severity_flags": 0},
            "mock_llm": {
                "label": "misleading",
                "evidence_support": "partial",
                "source_attribution": "partial",
                "internal_consistency": "questionable",
                "short_reason": "Hiring target is distributed over two years, not a single month.",
            },
        },
        {
            "case_id": "ctrl_017",
            "category": "numeric_statistical_claim",
            "description": "Misleading proportion based on unverified baseline.",
            "input_text": "SolarTech batteries have tripled the energy density of all existing batteries worldwide.",
            "expected_verdict": "misleading",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "SolarTech batteries tripled energy density of all existing batteries.",
                    "checkable": True,
                    "claim_type": "numeric",
                    "source_span": "SolarTech batteries have tripled the energy density of all existing batteries worldwide.",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_005",
                    "claim_summary": "SolarTech battery efficiency testing",
                    "evidence_text": "Laboratory evaluations of SolarTech prototype solid-state batteries recorded an energy density of 450 Wh/kg under standard conditions.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/solartech-battery",
                    "published_date": "2024-04-18",
                    "stance": "supports",
                    "source_type": "technical_report",
                    "is_synthetic": True,
                    "similarity_score": 0.83,
                }
            ],
            "mock_ml": {
                "label": "misleading",
                "probabilities": {"likely_reliable": 0.25, "likely_fake": 0.30, "misleading": 0.45},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 1, "high_severity_flags": 0},
            "mock_llm": {
                "label": "misleading",
                "evidence_support": "partial",
                "source_attribution": "weak",
                "internal_consistency": "questionable",
                "short_reason": "Energy density of 450 Wh/kg is high but does not triple all worldwide battery benchmarks.",
            },
        },

        # =====================================================================
        # CATEGORY 6: Multi-Claim Statements
        # =====================================================================
        {
            "case_id": "ctrl_018",
            "category": "multi_claim_statement",
            "description": "Two checkable claims, both supported by independent evidence records.",
            "input_text": "Acme Corp announced a new factory in Texas and plans to hire 3,000 workers across engineering divisions.",
            "expected_verdict": "likely_reliable",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Corp announced a new factory in Texas.",
                    "checkable": True,
                    "claim_type": "event",
                    "source_span": "Acme Corp announced a new factory in Texas",
                },
                {
                    "id": 2,
                    "text": "Acme Corp plans to hire 3,000 workers.",
                    "checkable": True,
                    "claim_type": "event",
                    "source_span": "plans to hire 3,000 workers across engineering divisions",
                },
            ],
            "mock_evidence": [
                {
                    "id": "synth_001",
                    "claim_summary": "Acme Corp announced a new factory in Texas",
                    "evidence_text": "Acme Corp announced today that it will construct a new manufacturing facility in Texas to expand clean energy production.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-texas",
                    "published_date": "2024-03-15",
                    "stance": "supports",
                    "source_type": "press_release",
                    "is_synthetic": True,
                    "similarity_score": 0.92,
                },
                {
                    "id": "synth_002",
                    "claim_summary": "Acme Corp plans to hire 3,000 workers",
                    "evidence_text": "Acme Corp confirmed plans to recruit and hire 3,000 workers across engineering and assembly lines over the next two years.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-hiring",
                    "published_date": "2024-03-20",
                    "stance": "supports",
                    "source_type": "press_release",
                    "is_synthetic": True,
                    "similarity_score": 0.90,
                },
            ],
            "mock_ml": {
                "label": "likely_reliable",
                "probabilities": {"likely_reliable": 0.74, "likely_fake": 0.12, "misleading": 0.14},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 0, "high_severity_flags": 0},
            "mock_llm": {
                "label": "likely_reliable",
                "evidence_support": "supported",
                "source_attribution": "strong",
                "internal_consistency": "consistent",
                "short_reason": "Both propositions are corroborated by corporate press releases.",
            },
        },
        {
            "case_id": "ctrl_019",
            "category": "multi_claim_statement",
            "description": "Multi-claim statement mixing verified premise with false conclusion.",
            "input_text": "Acme Corp announced a new factory in Texas, but the entire project was cancelled when the CEO resigned.",
            "expected_verdict": "misleading",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Corp announced a new factory in Texas.",
                    "checkable": True,
                    "claim_type": "event",
                    "source_span": "Acme Corp announced a new factory in Texas",
                },
                {
                    "id": 2,
                    "text": "The project was cancelled because the CEO resigned.",
                    "checkable": True,
                    "claim_type": "event",
                    "source_span": "the entire project was cancelled when the CEO resigned",
                },
            ],
            "mock_evidence": [
                {
                    "id": "synth_001",
                    "claim_summary": "Acme Corp announced a new factory in Texas",
                    "evidence_text": "Acme Corp announced today that it will construct a new manufacturing facility in Texas to expand clean energy production.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-texas",
                    "published_date": "2024-03-15",
                    "stance": "supports",
                    "source_type": "press_release",
                    "is_synthetic": True,
                    "similarity_score": 0.88,
                },
                {
                    "id": "synth_004",
                    "claim_summary": "Acme Corp CEO resignation rumors",
                    "evidence_text": "Acme Corp board of directors confirmed that the chief executive officer has not resigned and remains actively leading the firm.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-ceo",
                    "published_date": "2024-04-12",
                    "stance": "contradicts",
                    "source_type": "official_statement",
                    "is_synthetic": True,
                    "similarity_score": 0.85,
                },
            ],
            "mock_ml": {
                "label": "misleading",
                "probabilities": {"likely_reliable": 0.20, "likely_fake": 0.35, "misleading": 0.45},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 0, "high_severity_flags": 0},
            "mock_llm": {
                "label": "misleading",
                "evidence_support": "partial",
                "source_attribution": "partial",
                "internal_consistency": "questionable",
                "short_reason": "Factory was announced, but cancellation due to resignation is contradicted.",
            },
        },
        {
            "case_id": "ctrl_020",
            "category": "multi_claim_statement",
            "description": "Multi-claim statement where primary assertion is fabricated.",
            "input_text": "Acme Corp cancelled all hiring and dissolved its corporate board after the CEO was ousted.",
            "expected_verdict": "likely_fake",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Corp cancelled all hiring.",
                    "checkable": True,
                    "claim_type": "event",
                    "source_span": "Acme Corp cancelled all hiring",
                },
                {
                    "id": 2,
                    "text": "The corporate board dissolved after CEO was ousted.",
                    "checkable": True,
                    "claim_type": "event",
                    "source_span": "dissolved its corporate board after the CEO was ousted",
                },
            ],
            "mock_evidence": [
                {
                    "id": "synth_004",
                    "claim_summary": "Acme Corp CEO resignation rumors",
                    "evidence_text": "Acme Corp board of directors confirmed that the chief executive officer has not resigned and remains actively leading the firm.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-ceo",
                    "published_date": "2024-04-12",
                    "stance": "contradicts",
                    "source_type": "official_statement",
                    "is_synthetic": True,
                    "similarity_score": 0.86,
                }
            ],
            "mock_ml": {
                "label": "likely_fake",
                "probabilities": {"likely_reliable": 0.10, "likely_fake": 0.75, "misleading": 0.15},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 1, "high_severity_flags": 0},
            "mock_llm": {
                "label": "likely_fake",
                "evidence_support": "contradicted",
                "source_attribution": "none",
                "internal_consistency": "contradictory",
                "short_reason": "Board confirmed CEO remains active and board is operating normally.",
            },
        },
        {
            "case_id": "ctrl_021",
            "category": "multi_claim_statement",
            "description": "Multi-claim statement with true production figure but false failure rate.",
            "input_text": "Acme Corp produced 250,000 components in Q1 2024, but every single unit failed standard safety testing.",
            "expected_verdict": "misleading",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Corp produced 250,000 components in Q1 2024.",
                    "checkable": True,
                    "claim_type": "numeric",
                    "source_span": "Acme Corp produced 250,000 components in Q1 2024",
                },
                {
                    "id": 2,
                    "text": "Every single unit failed standard safety testing.",
                    "checkable": True,
                    "claim_type": "event",
                    "source_span": "every single unit failed standard safety testing",
                },
            ],
            "mock_evidence": [
                {
                    "id": "synth_003",
                    "claim_summary": "Acme Corp quarterly production report",
                    "evidence_text": "The quarterly industrial index noted that Acme Corp manufactured 250,000 components in Q1 2024.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-q1",
                    "published_date": "2024-04-05",
                    "stance": "supports",
                    "source_type": "industry_report",
                    "is_synthetic": True,
                    "similarity_score": 0.88,
                }
            ],
            "mock_ml": {
                "label": "misleading",
                "probabilities": {"likely_reliable": 0.25, "likely_fake": 0.30, "misleading": 0.45},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 1, "high_severity_flags": 0},
            "mock_llm": {
                "label": "misleading",
                "evidence_support": "partial",
                "source_attribution": "partial",
                "internal_consistency": "questionable",
                "short_reason": "Production number is accurate but 100% failure rate is uncorroborated sensationalism.",
            },
        },

        # =====================================================================
        # CATEGORY 7: Contradicted Claims (Expected: likely_fake, abstained: False)
        # =====================================================================
        {
            "case_id": "ctrl_022",
            "category": "contradicted_claim",
            "description": "Directly contradicted by official corporate board statement.",
            "input_text": "Acme Corp CEO has resigned from the company following executive controversies.",
            "expected_verdict": "likely_fake",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Corp CEO resigned following executive controversies.",
                    "checkable": True,
                    "claim_type": "event",
                    "source_span": "Acme Corp CEO has resigned from the company following executive controversies.",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_004",
                    "claim_summary": "Acme Corp CEO resignation rumors",
                    "evidence_text": "Acme Corp board of directors confirmed that the chief executive officer has not resigned and remains actively leading the firm.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-ceo",
                    "published_date": "2024-04-12",
                    "stance": "contradicts",
                    "source_type": "official_statement",
                    "is_synthetic": True,
                    "similarity_score": 0.93,
                }
            ],
            "mock_ml": {
                "label": "likely_fake",
                "probabilities": {"likely_reliable": 0.15, "likely_fake": 0.72, "misleading": 0.13},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 0, "high_severity_flags": 0},
            "mock_llm": {
                "label": "likely_fake",
                "evidence_support": "contradicted",
                "source_attribution": "none",
                "internal_consistency": "contradictory",
                "short_reason": "Board of directors explicitly confirmed CEO has not resigned.",
            },
        },
        {
            "case_id": "ctrl_023",
            "category": "contradicted_claim",
            "description": "Fabricated factory cancellation contradicted by official construction announcement.",
            "input_text": "Acme Corp permanently cancelled its planned manufacturing facility in Texas.",
            "expected_verdict": "likely_fake",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Corp permanently cancelled its manufacturing facility in Texas.",
                    "checkable": True,
                    "claim_type": "event",
                    "source_span": "Acme Corp permanently cancelled its planned manufacturing facility in Texas.",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_001",
                    "claim_summary": "Acme Corp announced a new factory in Texas",
                    "evidence_text": "Acme Corp announced today that it will construct a new manufacturing facility in Texas to expand clean energy production.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-texas",
                    "published_date": "2024-03-15",
                    "stance": "contradicts",
                    "source_type": "press_release",
                    "is_synthetic": True,
                    "similarity_score": 0.89,
                }
            ],
            "mock_ml": {
                "label": "likely_fake",
                "probabilities": {"likely_reliable": 0.18, "likely_fake": 0.68, "misleading": 0.14},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 0, "high_severity_flags": 0},
            "mock_llm": {
                "label": "likely_fake",
                "evidence_support": "contradicted",
                "source_attribution": "weak",
                "internal_consistency": "contradictory",
                "short_reason": "Company confirmed construction is actively proceeding.",
            },
        },
        {
            "case_id": "ctrl_024",
            "category": "contradicted_claim",
            "description": "Fabricated layoff claim directly contradicted by recruitment announcement.",
            "input_text": "Acme Corp is terminating 3,000 workers due to financial collapse.",
            "expected_verdict": "likely_fake",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Corp is terminating 3,000 workers due to financial collapse.",
                    "checkable": True,
                    "claim_type": "event",
                    "source_span": "Acme Corp is terminating 3,000 workers due to financial collapse.",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_002",
                    "claim_summary": "Acme Corp plans to hire 3,000 workers",
                    "evidence_text": "Acme Corp confirmed plans to recruit and hire 3,000 workers across engineering and assembly lines over the next two years.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-hiring",
                    "published_date": "2024-03-20",
                    "stance": "contradicts",
                    "source_type": "press_release",
                    "is_synthetic": True,
                    "similarity_score": 0.88,
                }
            ],
            "mock_ml": {
                "label": "likely_fake",
                "probabilities": {"likely_reliable": 0.12, "likely_fake": 0.76, "misleading": 0.12},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 0, "high_severity_flags": 0},
            "mock_llm": {
                "label": "likely_fake",
                "evidence_support": "contradicted",
                "source_attribution": "weak",
                "internal_consistency": "contradictory",
                "short_reason": "Acme announced hiring 3,000 employees, directly contradicting reported mass layoffs.",
            },
        },
        {
            "case_id": "ctrl_025",
            "category": "contradicted_claim",
            "description": "Fabricated zero production claim contradicted by industrial output report.",
            "input_text": "Acme Corp produced zero industrial components during the first quarter of 2024.",
            "expected_verdict": "likely_fake",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Corp produced zero components in Q1 2024.",
                    "checkable": True,
                    "claim_type": "numeric",
                    "source_span": "Acme Corp produced zero industrial components during the first quarter of 2024.",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_003",
                    "claim_summary": "Acme Corp quarterly production report",
                    "evidence_text": "The quarterly industrial index noted that Acme Corp manufactured 250,000 components in Q1 2024.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-q1",
                    "published_date": "2024-04-05",
                    "stance": "contradicts",
                    "source_type": "industry_report",
                    "is_synthetic": True,
                    "similarity_score": 0.90,
                }
            ],
            "mock_ml": {
                "label": "likely_fake",
                "probabilities": {"likely_reliable": 0.14, "likely_fake": 0.74, "misleading": 0.12},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 0, "high_severity_flags": 0},
            "mock_llm": {
                "label": "likely_fake",
                "evidence_support": "contradicted",
                "source_attribution": "weak",
                "internal_consistency": "contradictory",
                "short_reason": "Output was verified at 250,000 components, directly refuting zero production.",
            },
        },

        # =====================================================================
        # CATEGORY 8: Red Flags Contradiction (Sensational hoax refuted by evidence)
        # =====================================================================
        {
            "case_id": "ctrl_026",
            "category": "red_flags_contradiction",
            "description": "Sensational hoax with heavy red flags and direct evidence contradiction.",
            "input_text": "ALERT: ACME CORP CEO HAS FLEED THE COUNTRY AMID CRIME INVESTIGATION AND RESIGNED IMMEDIATELY!!!",
            "expected_verdict": "likely_fake",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Corp CEO fled the country and resigned amid crime investigation.",
                    "checkable": True,
                    "claim_type": "event",
                    "source_span": "ACME CORP CEO HAS FLEED THE COUNTRY AMID CRIME INVESTIGATION AND RESIGNED",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_004",
                    "claim_summary": "Acme Corp CEO resignation rumors",
                    "evidence_text": "Acme Corp board of directors confirmed that the chief executive officer has not resigned and remains actively leading the firm.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-ceo",
                    "published_date": "2024-04-12",
                    "stance": "contradicts",
                    "source_type": "official_statement",
                    "is_synthetic": True,
                    "similarity_score": 0.88,
                }
            ],
            "mock_ml": {
                "label": "likely_fake",
                "probabilities": {"likely_reliable": 0.08, "likely_fake": 0.82, "misleading": 0.10},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 4, "high_severity_flags": 3},
            "mock_llm": {
                "label": "likely_fake",
                "evidence_support": "contradicted",
                "source_attribution": "none",
                "internal_consistency": "contradictory",
                "short_reason": "Fabricated conspiracy headline directly contradicted by board of directors.",
            },
        },
        {
            "case_id": "ctrl_027",
            "category": "red_flags_contradiction",
            "description": "Hyperbolic clickbait claiming plant shutdown contradicted by expansion record.",
            "input_text": "SHOCKING COVERUP: ACME TEXAS FACTORY SHUT DOWN AFTER EXPLOSIVE HAZARDOUS SPILL!!!",
            "expected_verdict": "likely_fake",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Texas factory shut down after a hazardous spill.",
                    "checkable": True,
                    "claim_type": "event",
                    "source_span": "ACME TEXAS FACTORY SHUT DOWN AFTER EXPLOSIVE HAZARDOUS SPILL",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_001",
                    "claim_summary": "Acme Corp announced a new factory in Texas",
                    "evidence_text": "Acme Corp announced today that it will construct a new manufacturing facility in Texas to expand clean energy production.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-texas",
                    "published_date": "2024-03-15",
                    "stance": "contradicts",
                    "source_type": "press_release",
                    "is_synthetic": True,
                    "similarity_score": 0.84,
                }
            ],
            "mock_ml": {
                "label": "likely_fake",
                "probabilities": {"likely_reliable": 0.10, "likely_fake": 0.78, "misleading": 0.12},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 3, "high_severity_flags": 2},
            "mock_llm": {
                "label": "likely_fake",
                "evidence_support": "contradicted",
                "source_attribution": "none",
                "internal_consistency": "contradictory",
                "short_reason": "Fabricated spill claim; official reports confirm plant construction is ongoing.",
            },
        },

        # =====================================================================
        # CATEGORY 9: Partial / Misleading Claims (Expected: misleading, abstained: False)
        # =====================================================================
        {
            "case_id": "ctrl_028",
            "category": "partial_misleading_claim",
            "description": "Sensational cherry-picking framing with partial evidence support.",
            "input_text": "BOMBSHELL ADMISSION: Acme Corp output capped because clean energy technology is failing!",
            "expected_verdict": "misleading",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Corp output capped because clean energy technology is failing.",
                    "checkable": True,
                    "claim_type": "causal",
                    "source_span": "Acme Corp output capped because clean energy technology is failing",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_003",
                    "claim_summary": "Acme Corp quarterly production report",
                    "evidence_text": "The quarterly industrial index noted that Acme Corp manufactured 250,000 components in Q1 2024.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-q1",
                    "published_date": "2024-04-05",
                    "stance": "supports",
                    "source_type": "industry_report",
                    "is_synthetic": True,
                    "similarity_score": 0.82,
                }
            ],
            "mock_ml": {
                "label": "misleading",
                "probabilities": {"likely_reliable": 0.20, "likely_fake": 0.35, "misleading": 0.45},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 3, "high_severity_flags": 2},
            "mock_llm": {
                "label": "misleading",
                "evidence_support": "partial",
                "source_attribution": "partial",
                "internal_consistency": "questionable",
                "short_reason": "Output was recorded in standard quarterly report; claiming technology is failing is distorted.",
            },
        },
        {
            "case_id": "ctrl_029",
            "category": "partial_misleading_claim",
            "description": "Half-truth omitting critical context about multi-year hiring timeline.",
            "input_text": "Acme Corp announced it is hiring 3,000 workers immediately, but failed to disclose heavy automation on assembly lines.",
            "expected_verdict": "misleading",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Corp is hiring 3,000 workers immediately.",
                    "checkable": True,
                    "claim_type": "event",
                    "source_span": "Acme Corp announced it is hiring 3,000 workers immediately",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_002",
                    "claim_summary": "Acme Corp plans to hire 3,000 workers",
                    "evidence_text": "Acme Corp confirmed plans to recruit and hire 3,000 workers across engineering and assembly lines over the next two years.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-hiring",
                    "published_date": "2024-03-20",
                    "stance": "supports",
                    "source_type": "press_release",
                    "is_synthetic": True,
                    "similarity_score": 0.86,
                }
            ],
            "mock_ml": {
                "label": "misleading",
                "probabilities": {"likely_reliable": 0.25, "likely_fake": 0.25, "misleading": 0.50},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 0, "high_severity_flags": 0},
            "mock_llm": {
                "label": "misleading",
                "evidence_support": "partial",
                "source_attribution": "partial",
                "internal_consistency": "questionable",
                "short_reason": "Recruitment plan is for 2 years rather than immediate onboarding.",
            },
        },
        {
            "case_id": "ctrl_030",
            "category": "partial_misleading_claim",
            "description": "Distorted claim omitting scheduled factory retooling context.",
            "input_text": "Acme Corp component manufacturing dropped severely in early 2024 compared to peak historical volumes.",
            "expected_verdict": "misleading",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Corp manufacturing dropped severely compared to peak historical volumes.",
                    "checkable": True,
                    "claim_type": "numeric",
                    "source_span": "Acme Corp component manufacturing dropped severely in early 2024 compared to peak historical volumes.",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_003",
                    "claim_summary": "Acme Corp quarterly production report",
                    "evidence_text": "The quarterly industrial index noted that Acme Corp manufactured 250,000 components in Q1 2024.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-q1",
                    "published_date": "2024-04-05",
                    "stance": "supports",
                    "source_type": "industry_report",
                    "is_synthetic": True,
                    "similarity_score": 0.84,
                }
            ],
            "mock_ml": {
                "label": "misleading",
                "probabilities": {"likely_reliable": 0.20, "likely_fake": 0.30, "misleading": 0.50},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 0, "high_severity_flags": 0},
            "mock_llm": {
                "label": "misleading",
                "evidence_support": "partial",
                "source_attribution": "partial",
                "internal_consistency": "questionable",
                "short_reason": "Omits baseline comparison figures to create misleading impression of severe decline.",
            },
        },
        {
            "case_id": "ctrl_031",
            "category": "partial_misleading_claim",
            "description": "Technically true headline with distorted causal attribution.",
            "input_text": "SolarTech battery tests only reached 450 Wh/kg because lab temperatures were artificially manipulated.",
            "expected_verdict": "misleading",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "SolarTech battery reached 450 Wh/kg due to artificial temperature manipulation.",
                    "checkable": True,
                    "claim_type": "causal",
                    "source_span": "SolarTech battery tests only reached 450 Wh/kg because lab temperatures were artificially manipulated.",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_005",
                    "claim_summary": "SolarTech battery efficiency testing",
                    "evidence_text": "Laboratory evaluations of SolarTech prototype solid-state batteries recorded an energy density of 450 Wh/kg under standard conditions.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/solartech-battery",
                    "published_date": "2024-04-18",
                    "stance": "supports",
                    "source_type": "technical_report",
                    "is_synthetic": True,
                    "similarity_score": 0.85,
                }
            ],
            "mock_ml": {
                "label": "misleading",
                "probabilities": {"likely_reliable": 0.24, "likely_fake": 0.26, "misleading": 0.50},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 0, "high_severity_flags": 0},
            "mock_llm": {
                "label": "misleading",
                "evidence_support": "partial",
                "source_attribution": "partial",
                "internal_consistency": "questionable",
                "short_reason": "450 Wh/kg was achieved under standard conditions, not artificial manipulation.",
            },
        },
        {
            "case_id": "ctrl_032",
            "category": "partial_misleading_claim",
            "description": "Selective excerpt presenting standard expansion as unpermitted construction.",
            "input_text": "Acme Corp is constructing a massive manufacturing facility in Texas without state environmental clearances.",
            "expected_verdict": "misleading",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Corp is constructing a facility in Texas without state clearances.",
                    "checkable": True,
                    "claim_type": "event",
                    "source_span": "Acme Corp is constructing a massive manufacturing facility in Texas without state environmental clearances.",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_001",
                    "claim_summary": "Acme Corp announced a new factory in Texas",
                    "evidence_text": "Acme Corp announced today that it will construct a new manufacturing facility in Texas to expand clean energy production.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-texas",
                    "published_date": "2024-03-15",
                    "stance": "supports",
                    "source_type": "press_release",
                    "is_synthetic": True,
                    "similarity_score": 0.87,
                }
            ],
            "mock_ml": {
                "label": "misleading",
                "probabilities": {"likely_reliable": 0.22, "likely_fake": 0.30, "misleading": 0.48},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 0, "high_severity_flags": 0},
            "mock_llm": {
                "label": "misleading",
                "evidence_support": "partial",
                "source_attribution": "partial",
                "internal_consistency": "questionable",
                "short_reason": "Facility construction confirmed; unpermitted status is unverified distortion.",
            },
        },
        {
            "case_id": "ctrl_033",
            "category": "red_flags_misleading",
            "description": "Sensational vocabulary distorting standard recruitment notice into panic.",
            "input_text": "PANIC IN THE WORKFORCE: Acme Corp desperately scrambling for 3,000 warm bodies to save dying factories!",
            "expected_verdict": "misleading",
            "expected_abstained": False,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Corp is scrambling for 3,000 workers to save dying factories.",
                    "checkable": True,
                    "claim_type": "event",
                    "source_span": "Acme Corp desperately scrambling for 3,000 warm bodies to save dying factories",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_002",
                    "claim_summary": "Acme Corp plans to hire 3,000 workers",
                    "evidence_text": "Acme Corp confirmed plans to recruit and hire 3,000 workers across engineering and assembly lines over the next two years.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-hiring",
                    "published_date": "2024-03-20",
                    "stance": "supports",
                    "source_type": "press_release",
                    "is_synthetic": True,
                    "similarity_score": 0.83,
                }
            ],
            "mock_ml": {
                "label": "misleading",
                "probabilities": {"likely_reliable": 0.20, "likely_fake": 0.35, "misleading": 0.45},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 3, "high_severity_flags": 2},
            "mock_llm": {
                "label": "misleading",
                "evidence_support": "partial",
                "source_attribution": "partial",
                "internal_consistency": "questionable",
                "short_reason": "Sensationalized framing of standard planned recruitment program.",
            },
        },

        # =====================================================================
        # CATEGORY 10: No Evidence Found (Expected: unverifiable, abstained: True)
        # =====================================================================
        {
            "case_id": "ctrl_034",
            "category": "no_evidence",
            "description": "Checkable factual claim with zero matching evidence in knowledge base.",
            "input_text": "The mayor of Columbus Ohio signed an executive order allocating $12 million to municipal cycling paths.",
            "expected_verdict": "unverifiable",
            "expected_abstained": True,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "The mayor of Columbus signed an executive order allocating $12 million to cycling paths.",
                    "checkable": True,
                    "claim_type": "event",
                    "source_span": "The mayor of Columbus Ohio signed an executive order allocating $12 million to municipal cycling paths.",
                }
            ],
            "mock_evidence": [],
            "mock_ml": {
                "label": "likely_reliable",
                "probabilities": {"likely_reliable": 0.60, "likely_fake": 0.20, "misleading": 0.20},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 0, "high_severity_flags": 0},
            "mock_llm": None,
        },
        {
            "case_id": "ctrl_035",
            "category": "no_evidence",
            "description": "Foreign transit investment claim with zero evidence in local repository.",
            "input_text": "The municipal council of Lyon France completed construction on twelve new rapid tramway stations in 2023.",
            "expected_verdict": "unverifiable",
            "expected_abstained": True,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Lyon municipal council completed twelve new rapid tramway stations in 2023.",
                    "checkable": True,
                    "claim_type": "event",
                    "source_span": "The municipal council of Lyon France completed construction on twelve new rapid tramway stations in 2023.",
                }
            ],
            "mock_evidence": [],
            "mock_ml": {
                "label": "likely_reliable",
                "probabilities": {"likely_reliable": 0.55, "likely_fake": 0.25, "misleading": 0.20},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 0, "high_severity_flags": 0},
            "mock_llm": None,
        },

        # =====================================================================
        # CATEGORY 11: Non-Checkable / Opinion (Expected: unverifiable, abstained: True)
        # =====================================================================
        {
            "case_id": "ctrl_036",
            "category": "non_checkable_opinion",
            "description": "Subjective aesthetic value judgment with no empirical checkability.",
            "input_text": "Modern brutalist architecture in downtown urban centers is visually unappealing and creates a gloomy atmosphere.",
            "expected_verdict": "unverifiable",
            "expected_abstained": True,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Brutalist architecture in urban centers is visually unappealing.",
                    "checkable": False,
                    "claim_type": "other",
                    "source_span": "Modern brutalist architecture in downtown urban centers is visually unappealing",
                }
            ],
            "mock_evidence": [],
            "mock_ml": {
                "label": "likely_fake",
                "probabilities": {"likely_reliable": 0.25, "likely_fake": 0.50, "misleading": 0.25},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 0, "high_severity_flags": 0},
            "mock_llm": None,
        },
        {
            "case_id": "ctrl_037",
            "category": "non_checkable_opinion",
            "description": "Pure political rhetoric and philosophical stance.",
            "input_text": "Rapid industrial development is fundamentally harmful to the spiritual well-being of modern society.",
            "expected_verdict": "unverifiable",
            "expected_abstained": True,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Industrial development is harmful to spiritual well-being.",
                    "checkable": False,
                    "claim_type": "other",
                    "source_span": "Rapid industrial development is fundamentally harmful to the spiritual well-being of modern society.",
                }
            ],
            "mock_evidence": [],
            "mock_ml": {
                "label": "misleading",
                "probabilities": {"likely_reliable": 0.30, "likely_fake": 0.30, "misleading": 0.40},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 0, "high_severity_flags": 0},
            "mock_llm": None,
        },

        # =====================================================================
        # CATEGORY 12: Conflicting Evidence (Expected: unverifiable, abstained: True)
        # =====================================================================
        {
            "case_id": "ctrl_038",
            "category": "conflicting_evidence",
            "description": "Mutually contradictory evidence documents produce safe abstention under Rule 8.",
            "input_text": "Acme Corp executive leadership status is currently disputed across regulatory filings.",
            "expected_verdict": "unverifiable",
            "expected_abstained": True,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Corp executive leadership status is disputed.",
                    "checkable": True,
                    "claim_type": "event",
                    "source_span": "Acme Corp executive leadership status is currently disputed",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_004",
                    "claim_summary": "Acme Corp CEO resignation rumors",
                    "evidence_text": "Acme Corp board of directors confirmed that the chief executive officer has not resigned and remains actively leading the firm.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-ceo",
                    "published_date": "2024-04-12",
                    "stance": "contradicts",
                    "source_type": "official_statement",
                    "is_synthetic": True,
                    "similarity_score": 0.88,
                }
            ],
            "mock_ml": {
                "label": "misleading",
                "probabilities": {"likely_reliable": 0.33, "likely_fake": 0.33, "misleading": 0.34},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 0, "high_severity_flags": 0},
            "mock_llm": {
                "label": "unverifiable",
                "evidence_support": "unavailable",
                "source_attribution": "weak",
                "internal_consistency": "contradictory",
                "short_reason": "Inconclusive evidence due to contradictory reports; system abstained safely.",
            },
        },
        {
            "case_id": "ctrl_039",
            "category": "conflicting_evidence",
            "description": "LLM reasoner assesses evidence as contradictory and concludes unverifiable under Rule 4.",
            "input_text": "SolarTech solid-state battery testing was reported as both a historic breakthrough and a total lab failure.",
            "expected_verdict": "unverifiable",
            "expected_abstained": True,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "SolarTech battery testing results are disputed.",
                    "checkable": True,
                    "claim_type": "event",
                    "source_span": "SolarTech solid-state battery testing was reported as both a historic breakthrough and a total lab failure.",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_005",
                    "claim_summary": "SolarTech battery efficiency testing",
                    "evidence_text": "Laboratory evaluations of SolarTech prototype solid-state batteries recorded an energy density of 450 Wh/kg under standard conditions.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/solartech-battery",
                    "published_date": "2024-04-18",
                    "stance": "unknown",
                    "source_type": "technical_report",
                    "is_synthetic": True,
                    "similarity_score": 0.81,
                }
            ],
            "mock_ml": {
                "label": "likely_fake",
                "probabilities": {"likely_reliable": 0.20, "likely_fake": 0.50, "misleading": 0.30},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 0, "high_severity_flags": 0},
            "mock_llm": {
                "label": "unverifiable",
                "evidence_support": "unavailable",
                "source_attribution": "unknown",
                "internal_consistency": "contradictory",
                "short_reason": "Available laboratory reports provide conflicting evaluations.",
            },
        },

        # =====================================================================
        # CATEGORY 13: Synthetic Quarantine in Production Mode
        # =====================================================================
        {
            "case_id": "ctrl_040",
            "category": "synthetic_quarantine_prod",
            "description": "Synthetic demo fixture rejected when allow_synthetic_evidence=False (Rule 2 quarantine).",
            "input_text": "Acme Corp announced today that it will construct a new manufacturing facility in Texas to expand clean energy production.",
            "expected_verdict": "unverifiable",
            "expected_abstained": True,
            "allow_synthetic_evidence": False,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Corp announced a new factory in Texas to expand clean energy.",
                    "checkable": True,
                    "claim_type": "event",
                    "source_span": "Acme Corp announced today that it will construct a new manufacturing facility in Texas to expand clean energy production.",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_001",
                    "claim_summary": "Acme Corp announced a new factory in Texas",
                    "evidence_text": "Acme Corp announced today that it will construct a new manufacturing facility in Texas to expand clean energy production.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-texas",
                    "published_date": "2024-03-15",
                    "stance": "supports",
                    "source_type": "press_release",
                    "is_synthetic": True,
                    "similarity_score": 0.94,
                }
            ],
            "mock_ml": {
                "label": "likely_reliable",
                "probabilities": {"likely_reliable": 0.72, "likely_fake": 0.16, "misleading": 0.12},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 0, "high_severity_flags": 0},
            "mock_llm": None,
        },
        {
            "case_id": "ctrl_041",
            "category": "synthetic_quarantine_prod",
            "description": "Contradicting synthetic fixture quarantined in production mode, leading to safe abstention.",
            "input_text": "Acme Corp CEO resigned from his post this morning.",
            "expected_verdict": "unverifiable",
            "expected_abstained": True,
            "allow_synthetic_evidence": False,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Corp CEO resigned this morning.",
                    "checkable": True,
                    "claim_type": "event",
                    "source_span": "Acme Corp CEO resigned from his post this morning.",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_004",
                    "claim_summary": "Acme Corp CEO resignation rumors",
                    "evidence_text": "Acme Corp board of directors confirmed that the chief executive officer has not resigned and remains actively leading the firm.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-ceo",
                    "published_date": "2024-04-12",
                    "stance": "contradicts",
                    "source_type": "official_statement",
                    "is_synthetic": True,
                    "similarity_score": 0.91,
                }
            ],
            "mock_ml": {
                "label": "likely_fake",
                "probabilities": {"likely_reliable": 0.15, "likely_fake": 0.70, "misleading": 0.15},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 0, "high_severity_flags": 0},
            "mock_llm": None,
        },

        # =====================================================================
        # CATEGORY 14: LLM Unavailable / Fallback (Rule 3)
        # =====================================================================
        {
            "case_id": "ctrl_042",
            "category": "llm_unavailable_fallback",
            "description": "Evidence exists but LLM reasoner is None (timeout/failure), causing safe abstention.",
            "input_text": "Acme Corp confirmed plans to recruit and hire 3,000 workers across engineering divisions.",
            "expected_verdict": "unverifiable",
            "expected_abstained": True,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Corp confirmed plans to recruit 3,000 workers.",
                    "checkable": True,
                    "claim_type": "event",
                    "source_span": "Acme Corp confirmed plans to recruit and hire 3,000 workers across engineering divisions.",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_002",
                    "claim_summary": "Acme Corp plans to hire 3,000 workers",
                    "evidence_text": "Acme Corp confirmed plans to recruit and hire 3,000 workers across engineering and assembly lines over the next two years.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-hiring",
                    "published_date": "2024-03-20",
                    "stance": "supports",
                    "source_type": "press_release",
                    "is_synthetic": True,
                    "similarity_score": 0.90,
                }
            ],
            "mock_ml": {
                "label": "likely_reliable",
                "probabilities": {"likely_reliable": 0.65, "likely_fake": 0.15, "misleading": 0.20},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 0, "high_severity_flags": 0},
            "mock_llm": None,
        },
        {
            "case_id": "ctrl_043",
            "category": "llm_unavailable_fallback",
            "description": "Contradicting evidence exists but LLM encounters 429 quota exhaustion (Rule 3 abstention).",
            "input_text": "Acme Corp CEO resigned amid board allegations.",
            "expected_verdict": "unverifiable",
            "expected_abstained": True,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Corp CEO resigned amid allegations.",
                    "checkable": True,
                    "claim_type": "event",
                    "source_span": "Acme Corp CEO resigned amid board allegations.",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_004",
                    "claim_summary": "Acme Corp CEO resignation rumors",
                    "evidence_text": "Acme Corp board of directors confirmed that the chief executive officer has not resigned and remains actively leading the firm.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-ceo",
                    "published_date": "2024-04-12",
                    "stance": "contradicts",
                    "source_type": "official_statement",
                    "is_synthetic": True,
                    "similarity_score": 0.90,
                }
            ],
            "mock_ml": {
                "label": "likely_fake",
                "probabilities": {"likely_reliable": 0.15, "likely_fake": 0.70, "misleading": 0.15},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 0, "high_severity_flags": 0},
            "mock_llm": None,
        },

        # =====================================================================
        # CATEGORY 15: Retrieval Rejection / Low Similarity (Rule 2)
        # =====================================================================
        {
            "case_id": "ctrl_044",
            "category": "retrieval_rejection_low_sim",
            "description": "Retrieved evidence is only topically related without decisive stance; system safely abstains.",
            "input_text": "Acme Corp quarterly production report covered general global industrial indices in 2024.",
            "expected_verdict": "unverifiable",
            "expected_abstained": True,
            "allow_synthetic_evidence": True,
            "claims": [
                {
                    "id": 1,
                    "text": "Acme Corp quarterly report covered general global industrial indices.",
                    "checkable": True,
                    "claim_type": "event",
                    "source_span": "Acme Corp quarterly production report covered general global industrial indices in 2024.",
                }
            ],
            "mock_evidence": [
                {
                    "id": "synth_003",
                    "claim_summary": "Acme Corp quarterly production report",
                    "evidence_text": "The quarterly industrial index noted that Acme Corp manufactured 250,000 components in Q1 2024.",
                    "source_name": "Synthetic Demo Source",
                    "source_url": "https://example.com/synthetic/acme-q1",
                    "published_date": "2024-04-05",
                    "stance": "related",
                    "source_type": "industry_report",
                    "is_synthetic": True,
                    "similarity_score": 0.62,
                }
            ],
            "mock_ml": {
                "label": "misleading",
                "probabilities": {"likely_reliable": 0.30, "likely_fake": 0.30, "misleading": 0.40},
                "model_name": "Tuned TF-IDF + Logistic Regression",
                "model_version": "validation_frozen",
            },
            "mock_red_flags": {"total_flags": 0, "high_severity_flags": 0},
            "mock_llm": None,
        },
    ]
    return cases


def generate_production_abstention_benchmark() -> List[Dict[str, Any]]:
    """Build 20 diverse real-world statements across domains for empty-KB production evaluation."""
    cases: List[Dict[str, Any]] = [
        {
            "statement_id": "prod_001",
            "domain": "politics",
            "claim_type": "event",
            "statement": "The United States Senate voted 52 to 48 to confirm the new federal appellate judge.",
            "has_checkable_claims": True,
        },
        {
            "statement_id": "prod_002",
            "domain": "economy",
            "claim_type": "numeric",
            "statement": "The Consumer Price Index rose by 0.3 percent in August, driven largely by energy and housing costs.",
            "has_checkable_claims": True,
        },
        {
            "statement_id": "prod_003",
            "domain": "health_medicine",
            "claim_type": "medical",
            "statement": "Drinking raw unpasteurized milk cures chronic type 2 diabetes within three weeks.",
            "has_checkable_claims": True,
        },
        {
            "statement_id": "prod_004",
            "domain": "science_climate",
            "claim_type": "scientific",
            "statement": "Global average ocean surface temperatures reached a record high of 21.1 degrees Celsius in March 2024.",
            "has_checkable_claims": True,
        },
        {
            "statement_id": "prod_005",
            "domain": "technology_ai",
            "claim_type": "event",
            "statement": "A European artificial intelligence research laboratory released a 70-billion-parameter open-weights language model under Apache 2.0.",
            "has_checkable_claims": True,
        },
        {
            "statement_id": "prod_006",
            "domain": "sports",
            "claim_type": "event",
            "statement": "The Kansas City Chiefs defeated the San Francisco 49ers in overtime to win Super Bowl LVIII.",
            "has_checkable_claims": True,
        },
        {
            "statement_id": "prod_007",
            "domain": "astronomy",
            "claim_type": "scientific",
            "statement": "NASA's James Webb Space Telescope detected carbon dioxide and methane in the atmosphere of exoplanet K2-18b.",
            "has_checkable_claims": True,
        },
        {
            "statement_id": "prod_008",
            "domain": "transportation",
            "claim_type": "numeric",
            "statement": "Japan's high-speed Shinkansen network carried over 350 million passengers in 2023 with zero passenger fatalities.",
            "has_checkable_claims": True,
        },
        {
            "statement_id": "prod_009",
            "domain": "international",
            "claim_type": "event",
            "statement": "Sweden officially deposited its instrument of accession to the North Atlantic Treaty Organization in Washington.",
            "has_checkable_claims": True,
        },
        {
            "statement_id": "prod_010",
            "domain": "entertainment",
            "claim_type": "event",
            "statement": "Christopher Nolan won the Academy Award for Best Director for the biographical drama Oppenheimer.",
            "has_checkable_claims": True,
        },
        {
            "statement_id": "prod_011",
            "domain": "sensational_hoax",
            "claim_type": "conspiracy",
            "statement": "SECRET MILITARY TUNNELS DISCOVERED UNDER DENVER AIRPORT CONTAIN ALIEN TECHNOLOGY EVIDENCE!!!",
            "has_checkable_claims": True,
        },
        {
            "statement_id": "prod_012",
            "domain": "education",
            "claim_type": "policy",
            "statement": "California public high schools will require one semester of financial literacy coursework for graduation starting in 2027.",
            "has_checkable_claims": True,
        },
        {
            "statement_id": "prod_013",
            "domain": "energy",
            "claim_type": "numeric",
            "statement": "Wind and solar power generated more electricity in the European Union than fossil fuels in May 2024.",
            "has_checkable_claims": True,
        },
        {
            "statement_id": "prod_014",
            "domain": "agriculture",
            "claim_type": "numeric",
            "statement": "Brazil harvested a record 154 million metric tons of soybeans during the 2022-2023 crop season.",
            "has_checkable_claims": True,
        },
        {
            "statement_id": "prod_015",
            "domain": "opinion",
            "claim_type": "subjective",
            "statement": "Modern electronic dance music is aesthetically inferior to mid-century jazz composition.",
            "has_checkable_claims": False,
        },
        {
            "statement_id": "prod_016",
            "domain": "opinion",
            "claim_type": "subjective",
            "statement": "Autumn is undoubtedly the most peaceful and inspiring season of the calendar year.",
            "has_checkable_claims": False,
        },
        {
            "statement_id": "prod_017",
            "domain": "philosophy",
            "claim_type": "rhetorical",
            "statement": "Why should democratic governments continue funding space exploration when poverty persists on Earth?",
            "has_checkable_claims": False,
        },
        {
            "statement_id": "prod_018",
            "domain": "business",
            "claim_type": "numeric",
            "statement": "Nvidia's market capitalization surpassed $3 trillion in June 2024, briefly making it the most valuable company in the world.",
            "has_checkable_claims": True,
        },
        {
            "statement_id": "prod_019",
            "domain": "archaeology",
            "claim_type": "historical",
            "statement": "Archaeologists in Egypt uncovered an 18-meter-long intact papyrus scroll inside a sarcophagus at Saqqara.",
            "has_checkable_claims": True,
        },
        {
            "statement_id": "prod_020",
            "domain": "telecom",
            "claim_type": "numeric",
            "statement": "Global 5G mobile network subscriptions reached 1.6 billion connections by the end of 2023.",
            "has_checkable_claims": True,
        },
    ]
    return cases


def generate_claim_extraction_benchmark() -> List[Dict[str, Any]]:
    """Build 15 synthetic cases testing claim extraction functionality."""
    cases: List[Dict[str, Any]] = [
        {
            "test_id": "claim_001",
            "text": "The Federal Reserve raised the benchmark federal funds rate by 25 basis points at its July meeting.",
            "expected_claim_count": 1,
            "expected_checkable": True,
            "claim_type": "numeric",
            "has_quote": False,
            "description": "Single atomic numeric factual proposition.",
        },
        {
            "test_id": "claim_002",
            "text": "Unemployment fell to 3.8 percent in March and total nonfarm payrolls increased by 303,000 jobs.",
            "expected_claim_count": 2,
            "expected_checkable": True,
            "claim_type": "numeric",
            "has_quote": False,
            "description": "Compound sentence with two distinct numerical factual propositions.",
        },
        {
            "test_id": "claim_003",
            "text": "Chocolate gelato with salted caramel is the most delicious dessert ever invented.",
            "expected_claim_count": 1,
            "expected_checkable": False,
            "claim_type": "other",
            "has_quote": False,
            "description": "Pure subjective taste preference / non-checkable opinion.",
        },
        {
            "test_id": "claim_004",
            "text": "Why do municipal governments spend taxpayer funds on modern decorative streetlights?",
            "expected_claim_count": 0,
            "expected_checkable": False,
            "claim_type": "other",
            "has_quote": False,
            "description": "Rhetorical question with zero checkable factual assertions.",
        },
        {
            "test_id": "claim_005",
            "text": 'Senator Miller stated: "The proposed highway widening will reduce morning traffic congestion by 35% across the northern corridor."',
            "expected_claim_count": 1,
            "expected_checkable": True,
            "claim_type": "attribution",
            "has_quote": True,
            "description": "Direct quoted statement with clear political attribution and quantitative claim.",
        },
        {
            "test_id": "claim_006",
            "text": "This newly released science fiction film is an unmitigated disaster that ruins cinema forever!",
            "expected_claim_count": 1,
            "expected_checkable": False,
            "claim_type": "other",
            "has_quote": False,
            "description": "Sensational aesthetic critique / non-checkable opinion.",
        },
        {
            "test_id": "claim_007",
            "text": "California's Central Valley produces over 80 percent of the world's commercial almond crop.",
            "expected_claim_count": 1,
            "expected_checkable": True,
            "claim_type": "numeric",
            "has_quote": False,
            "description": "Agricultural percentage claim with geographical attribution.",
        },
        {
            "test_id": "claim_008",
            "text": "The Apollo 11 lunar module Eagle touched down on the Sea of Tranquility on July 20, 1969.",
            "expected_claim_count": 1,
            "expected_checkable": True,
            "claim_type": "event",
            "has_quote": False,
            "description": "Specific historical event with precise date and location.",
        },
        {
            "test_id": "claim_009",
            "text": "Universal public healthcare is a fundamental human right that all civilized nations must guarantee.",
            "expected_claim_count": 1,
            "expected_checkable": False,
            "claim_type": "other",
            "has_quote": False,
            "description": "Ethical and moral policy assertion / non-verifiable value judgment.",
        },
        {
            "test_id": "claim_010",
            "text": "Drinking three cups of green tea daily completely prevents all known respiratory viruses.",
            "expected_claim_count": 1,
            "expected_checkable": True,
            "claim_type": "causal",
            "has_quote": False,
            "description": "Empirical medical / health proposition (false/unsupported, but checkable).",
        },
        {
            "test_id": "claim_011",
            "text": "The main branch of the public library on Oak Street will be closed for HVAC repairs from October 10 to October 15.",
            "expected_claim_count": 1,
            "expected_checkable": True,
            "claim_type": "event",
            "has_quote": False,
            "description": "Local civic logistical notice with calendar boundaries.",
        },
        {
            "test_id": "claim_012",
            "text": "Artificial general intelligence will surpass all human intellectual capability before the end of 2028.",
            "expected_claim_count": 1,
            "expected_checkable": False,
            "claim_type": "other",
            "has_quote": False,
            "description": "Unverifiable speculative future prediction.",
        },
        {
            "test_id": "claim_013",
            "text": 'Physicist Albert Einstein wrote in 1915: "Gravitation is not a force, but a curvature of spacetime caused by mass."',
            "expected_claim_count": 1,
            "expected_checkable": True,
            "claim_type": "attribution",
            "has_quote": True,
            "description": "Historical quotation attribution with speaker and year.",
        },
        {
            "test_id": "claim_014",
            "text": "TechCorp reported third-quarter revenue of $18.4 billion, exceeding Wall Street consensus estimates by 4.2 percent.",
            "expected_claim_count": 1,
            "expected_checkable": True,
            "claim_type": "numeric",
            "has_quote": False,
            "description": "Corporate quarterly earnings assertion with financial metrics.",
        },
        {
            "test_id": "claim_015",
            "text": "The incumbent governor has ruined the state budget, while state highway spending actually grew by $450 million in 2023.",
            "expected_claim_count": 2,
            "expected_checkable": True,
            "claim_type": "numeric",
            "has_quote": False,
            "description": "Mixed statement containing subjective opinion and checkable budget figure.",
        },
    ]
    return cases


def generate_retrieval_benchmark() -> List[Dict[str, Any]]:
    """Build 16 test queries against the 5 synthetic demo KB items plus out-of-domain queries."""
    cases: List[Dict[str, Any]] = [
        # Queries matching synth_001 (Acme Texas factory)
        {
            "query_id": "ret_001",
            "query": "Acme Corp manufacturing facility in Texas clean energy",
            "expected_top1_id": "synth_001",
            "relevant_ids": ["synth_001"],
            "is_out_of_domain": False,
            "target_topic": "Acme Texas factory",
        },
        {
            "query_id": "ret_002",
            "query": "Where is Acme Corp building its new clean energy production plant?",
            "expected_top1_id": "synth_001",
            "relevant_ids": ["synth_001"],
            "is_out_of_domain": False,
            "target_topic": "Acme Texas factory",
        },
        {
            "query_id": "ret_003",
            "query": "Acme Corp construction in Texas manufacturing facility announcement",
            "expected_top1_id": "synth_001",
            "relevant_ids": ["synth_001"],
            "is_out_of_domain": False,
            "target_topic": "Acme Texas factory",
        },

        # Queries matching synth_002 (Acme 3000 workers hiring)
        {
            "query_id": "ret_004",
            "query": "How many workers is Acme Corp planning to recruit and hire over two years?",
            "expected_top1_id": "synth_002",
            "relevant_ids": ["synth_002"],
            "is_out_of_domain": False,
            "target_topic": "Acme hiring",
        },
        {
            "query_id": "ret_005",
            "query": "Acme Corp hiring 3,000 employees engineering assembly lines recruitment",
            "expected_top1_id": "synth_002",
            "relevant_ids": ["synth_002"],
            "is_out_of_domain": False,
            "target_topic": "Acme hiring",
        },
        {
            "query_id": "ret_006",
            "query": "Recruitment of 3,000 engineering workers by Acme Corp",
            "expected_top1_id": "synth_002",
            "relevant_ids": ["synth_002"],
            "is_out_of_domain": False,
            "target_topic": "Acme hiring",
        },

        # Queries matching synth_003 (Acme Q1 production report)
        {
            "query_id": "ret_007",
            "query": "Acme Corp quarterly production report Q1 2024 component output",
            "expected_top1_id": "synth_003",
            "relevant_ids": ["synth_003"],
            "is_out_of_domain": False,
            "target_topic": "Acme Q1 output",
        },
        {
            "query_id": "ret_008",
            "query": "How many components did Acme Corp manufacture in Q1 according to industrial index?",
            "expected_top1_id": "synth_003",
            "relevant_ids": ["synth_003"],
            "is_out_of_domain": False,
            "target_topic": "Acme Q1 output",
        },

        # Queries matching synth_004 (Acme CEO resignation rumors)
        {
            "query_id": "ret_009",
            "query": "Acme Corp chief executive officer resignation board confirmation",
            "expected_top1_id": "synth_004",
            "relevant_ids": ["synth_004"],
            "is_out_of_domain": False,
            "target_topic": "Acme CEO rumors",
        },
        {
            "query_id": "ret_010",
            "query": "Did the CEO of Acme Corp resign from his leadership position?",
            "expected_top1_id": "synth_004",
            "relevant_ids": ["synth_004"],
            "is_out_of_domain": False,
            "target_topic": "Acme CEO rumors",
        },
        {
            "query_id": "ret_011",
            "query": "Acme Corp board statement regarding chief executive rumors",
            "expected_top1_id": "synth_004",
            "relevant_ids": ["synth_004"],
            "is_out_of_domain": False,
            "target_topic": "Acme CEO rumors",
        },

        # Queries matching synth_005 (SolarTech battery testing)
        {
            "query_id": "ret_012",
            "query": "SolarTech solid-state battery prototype energy density testing Wh/kg",
            "expected_top1_id": "synth_005",
            "relevant_ids": ["synth_005"],
            "is_out_of_domain": False,
            "target_topic": "SolarTech battery",
        },
        {
            "query_id": "ret_013",
            "query": "Laboratory evaluation of 450 Wh/kg solid-state battery SolarTech",
            "expected_top1_id": "synth_005",
            "relevant_ids": ["synth_005"],
            "is_out_of_domain": False,
            "target_topic": "SolarTech battery",
        },

        # Negative / Out-of-Domain Queries (Should NOT retrieve or should score below threshold)
        {
            "query_id": "ret_014",
            "query": "The Great Pyramid of Giza was constructed under Pharaoh Khufu during the Fourth Dynasty.",
            "expected_top1_id": None,
            "relevant_ids": [],
            "is_out_of_domain": True,
            "target_topic": "Ancient Egypt",
        },
        {
            "query_id": "ret_015",
            "query": "Bitcoin cryptocurrency proof of work SHA-256 block mining algorithm difficulty adjustment",
            "expected_top1_id": None,
            "relevant_ids": [],
            "is_out_of_domain": True,
            "target_topic": "Cryptocurrency",
        },
        {
            "query_id": "ret_016",
            "query": "Clinical trial results for pediatric antibiotic treatments in sub-Saharan Africa",
            "expected_top1_id": None,
            "relevant_ids": [],
            "is_out_of_domain": True,
            "target_topic": "Medical clinical trial",
        },
    ]
    return cases


def main() -> None:
    """Generate all evaluation datasets."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating evaluation datasets in: {EVAL_DIR}")

    # 1. Controlled benchmark (JSONL)
    ctrl_cases = generate_controlled_benchmark()
    ctrl_path = EVAL_DIR / "hybrid_controlled_eval.jsonl"
    with open(ctrl_path, "w", encoding="utf-8") as f:
        for case in ctrl_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"  [1/4] Saved {len(ctrl_cases)} controlled benchmark cases -> {ctrl_path.name}")

    # 2. Production abstention benchmark (JSON)
    prod_cases = generate_production_abstention_benchmark()
    prod_path = EVAL_DIR / "production_abstention_eval.json"
    with open(prod_path, "w", encoding="utf-8") as f:
        json.dump(prod_cases, f, indent=2, ensure_ascii=False)
    print(f"  [2/4] Saved {len(prod_cases)} production abstention cases -> {prod_path.name}")

    # 3. Claim extraction benchmark (JSON)
    claim_cases = generate_claim_extraction_benchmark()
    claim_path = EVAL_DIR / "claim_extraction_eval.json"
    with open(claim_path, "w", encoding="utf-8") as f:
        json.dump(claim_cases, f, indent=2, ensure_ascii=False)
    print(f"  [3/4] Saved {len(claim_cases)} claim extraction cases -> {claim_path.name}")

    # 4. Retrieval benchmark (JSON)
    ret_cases = generate_retrieval_benchmark()
    ret_path = EVAL_DIR / "retrieval_eval.json"
    with open(ret_path, "w", encoding="utf-8") as f:
        json.dump(ret_cases, f, indent=2, ensure_ascii=False)
    print(f"  [4/4] Saved {len(ret_cases)} retrieval queries -> {ret_path.name}")

    print("\nDataset generation complete!")


if __name__ == "__main__":
    main()
