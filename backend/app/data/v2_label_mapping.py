"""V2 Label Mapping Registry and Decision Rules.

Mapping Version: v2_label_mapping_v1

Stores audited, explicit label correspondence dictionaries across source datasets.
Unknown or unverified labels fail closed to 'ambiguous' or 'excluded' rather than
defaulting to a positive or negative verdict.
"""

from __future__ import annotations

from typing import Optional

MAPPING_VERSION = "v2_label_mapping_v1"

# ---------------------------------------------------------------------------
# 1. LIAR Mapping (Preserves exact V1 baseline semantics)
# ---------------------------------------------------------------------------
# LIAR consists strictly of verdict labels from PolitiFact journalists.
# It does NOT contain evidence-absence labels and must NEVER map to 'unverifiable'.
LIAR_LABEL_MAPPING: dict[str, tuple[str, float, str]] = {
    "true": ("likely_reliable", 1.0, "liar_v1_true_to_reliable"),
    "mostly-true": ("likely_reliable", 0.95, "liar_v1_mostly_true_to_reliable"),
    "half-true": ("misleading", 0.95, "liar_v1_half_true_to_misleading"),
    "barely-true": ("misleading", 0.90, "liar_v1_barely_true_to_misleading"),
    "false": ("likely_fake", 1.0, "liar_v1_false_to_fake"),
    "pants-fire": ("likely_fake", 1.0, "liar_v1_pants_fire_to_fake"),
}

# ---------------------------------------------------------------------------
# 2. AVeriTeC Mapping (Audited from NeurIPS 2023 official schema)
# ---------------------------------------------------------------------------
# Supported -> likely_reliable: Direct empirical support from retrieved documents.
# Refuted -> likely_fake: Direct empirical contradiction from retrieved documents.
# Not Enough Evidence -> unverifiable: Systematic web search yielded insufficient authoritative reporting.
# Conflicting Evidence/Cherrypicking -> misleading: Facts are either selectively framed out of context
# or reputable outlets report conflicting verified facts. Note: While 'cherry-picking' is rhetorically
# deceptive (misleading), genuine 'conflicting evidence' is an empirical standoff; mapping to misleading
# reflects that the overall claim cannot be accepted as clean reliable truth.
AVERITEC_LABEL_MAPPING: dict[str, tuple[str, float, str]] = {
    "supported": ("likely_reliable", 1.0, "averitec_supported_to_reliable"),
    "refuted": ("likely_fake", 1.0, "averitec_refuted_to_fake"),
    "not enough evidence": ("unverifiable", 1.0, "averitec_nei_to_unverifiable"),
    "conflicting evidence/cherrypicking": ("misleading", 0.85, "averitec_cherrypicking_to_misleading"),
    "conflicting evidence/cherry-picking": ("misleading", 0.85, "averitec_cherrypicking_to_misleading"),
}

# ---------------------------------------------------------------------------
# 3. MultiFC Audited Mapping Registry
# ---------------------------------------------------------------------------
# Only labels with unambiguous journalistic definitions are accepted.
# Non-veracity items (political pledge trackers, numeric codes) are explicitly excluded.
MULTIFC_ACCEPTED_MAPPING: dict[str, tuple[str, float, str]] = {
    # Reliable / True variants
    "true": ("likely_reliable", 1.0, "multifc_true_to_reliable"),
    "mostly true": ("likely_reliable", 0.95, "multifc_mostly_true_to_reliable"),
    "mostly-true": ("likely_reliable", 0.95, "multifc_mostly_true_to_reliable"),
    "mostly_true": ("likely_reliable", 0.95, "multifc_mostly_true_to_reliable"),
    "mostly-correct": ("likely_reliable", 0.95, "multifc_mostly_correct_to_reliable"),
    "correct": ("likely_reliable", 1.0, "multifc_correct_to_reliable"),
    "truth!": ("likely_reliable", 1.0, "multifc_truth_to_reliable"),
    "fact": ("likely_reliable", 1.0, "multifc_fact_to_reliable"),
    "accurate": ("likely_reliable", 1.0, "multifc_accurate_to_reliable"),
    "verified": ("likely_reliable", 1.0, "multifc_verified_to_reliable"),
    "verdict: true": ("likely_reliable", 1.0, "multifc_verdict_true_to_reliable"),
    "conclusion: accurate": ("likely_reliable", 1.0, "multifc_conclusion_accurate_to_reliable"),
    "factscan score: true": ("likely_reliable", 1.0, "multifc_factscan_true_to_reliable"),

    # Misleading / Mixed variants
    "misleading": ("misleading", 1.0, "multifc_misleading_to_misleading"),
    "half-true": ("misleading", 0.95, "multifc_half_true_to_misleading"),
    "barely-true": ("misleading", 0.90, "multifc_barely_true_to_misleading"),
    "partly true": ("misleading", 0.90, "multifc_partly_true_to_misleading"),
    "partially true": ("misleading", 0.90, "multifc_partially_true_to_misleading"),
    "mixture": ("misleading", 0.90, "multifc_mixture_to_misleading"),
    "cherry picks": ("misleading", 0.90, "multifc_cherry_picks_to_misleading"),
    "not the whole story": ("misleading", 0.90, "multifc_not_whole_story_to_misleading"),
    "spins the facts": ("misleading", 0.90, "multifc_spins_facts_to_misleading"),
    "exaggerates": ("misleading", 0.90, "multifc_exaggerates_to_misleading"),
    "exaggerated": ("misleading", 0.90, "multifc_exaggerated_to_misleading"),
    "distorts the facts": ("misleading", 0.90, "multifc_distorts_facts_to_misleading"),
    "in-between": ("misleading", 0.85, "multifc_in_between_to_misleading"),
    "factscan score: misleading": ("misleading", 0.90, "multifc_factscan_misleading_to_misleading"),

    # Fake / False variants
    "false": ("likely_fake", 1.0, "multifc_false_to_fake"),
    "mostly false": ("likely_fake", 0.95, "multifc_mostly_false_to_fake"),
    "mostly-false": ("likely_fake", 0.95, "multifc_mostly_false_to_fake"),
    "mostly_false": ("likely_fake", 0.95, "multifc_mostly_false_to_fake"),
    "pants on fire!": ("likely_fake", 1.0, "multifc_pants_fire_to_fake"),
    "pants-fire": ("likely_fake", 1.0, "multifc_pants_fire_to_fake"),
    "fiction!": ("likely_fake", 1.0, "multifc_fiction_to_fake"),
    "fiction": ("likely_fake", 1.0, "multifc_fiction_to_fake"),
    "fake news": ("likely_fake", 1.0, "multifc_fake_news_to_fake"),
    "incorrect": ("likely_fake", 1.0, "multifc_incorrect_to_fake"),
    "bogus warning": ("likely_fake", 0.95, "multifc_bogus_warning_to_fake"),
    "facebook scams": ("likely_fake", 0.95, "multifc_facebook_scams_to_fake"),
    "verdict: false": ("likely_fake", 1.0, "multifc_verdict_false_to_fake"),
    "conclusion: false": ("likely_fake", 1.0, "multifc_conclusion_false_to_fake"),
    "rating: false": ("likely_fake", 1.0, "multifc_rating_false_to_fake"),
    "factscan score: false": ("likely_fake", 1.0, "multifc_factscan_false_to_fake"),
    "a lot of baloney": ("likely_fake", 0.95, "multifc_baloney_to_fake"),

    # Unverifiable / Unproven variants
    "unverified": ("unverifiable", 1.0, "multifc_unverified_to_unverifiable"),
    "unproven": ("unverifiable", 1.0, "multifc_unproven_to_unverifiable"),
    "unproven!": ("unverifiable", 1.0, "multifc_unproven_to_unverifiable"),
    "no evidence": ("unverifiable", 1.0, "multifc_no_evidence_to_unverifiable"),
    "unsupported": ("unverifiable", 1.0, "multifc_unsupported_to_unverifiable"),
    "conclusion: unclear": ("unverifiable", 1.0, "multifc_conclusion_unclear_to_unverifiable"),
    "verdict: unsubstantiated": ("unverifiable", 1.0, "multifc_unsubstantiated_to_unverifiable"),
}

MULTIFC_EXCLUDED_LABELS: dict[str, str] = {
    # Non-veracity political promise tracking
    "promise kept": "political_promise_tracker_non_veracity",
    "promise broken": "political_promise_tracker_non_veracity",
    "compromise": "political_promise_tracker_non_veracity",
    "in the works": "political_promise_tracker_non_veracity",
    "stalled": "political_promise_tracker_non_veracity",
    # Political consistency tracking
    "full flop": "political_flip_tracker_non_veracity",
    "half flip": "political_flip_tracker_non_veracity",
    "no flip": "political_flip_tracker_non_veracity",
    # Empty or unannotated
    "none": "unannotated_or_missing",
    "not yet rated": "unannotated_or_missing",
    # Numeric scale where direction is site-dependent
    "0": "numeric_scale_ambiguous_direction",
    "1": "numeric_scale_ambiguous_direction",
    "2": "numeric_scale_ambiguous_direction",
    "3": "numeric_scale_ambiguous_direction",
    "4": "numeric_scale_ambiguous_direction",
    "10": "numeric_scale_ambiguous_direction",
}

# ---------------------------------------------------------------------------
# 4. FEVER Mapping (Auxiliary Only)
# ---------------------------------------------------------------------------
FEVER_LABEL_MAPPING: dict[str, tuple[str, float, str]] = {
    "supports": ("likely_reliable", 0.90, "fever_supports_auxiliary"),
    "refutes": ("likely_fake", 0.90, "fever_refutes_auxiliary"),
    "not enough info": ("unverifiable", 0.90, "fever_nei_auxiliary"),
}


def map_dataset_label(dataset: str, original_label: str) -> tuple[Optional[str], str, float, str]:
    """Map a raw dataset label to the canonical V2 4-class space.

    Returns:
        (harmonized_label, mapping_status, mapping_confidence, mapping_rule)
    """
    cleaned = (original_label or "").strip().lower()

    if dataset == "liar":
        if cleaned in LIAR_LABEL_MAPPING:
            target, conf, rule = LIAR_LABEL_MAPPING[cleaned]
            return target, "accepted", conf, rule
        return None, "excluded", 0.0, "liar_unknown_label_fail_closed"

    if dataset == "averitec":
        if cleaned in AVERITEC_LABEL_MAPPING:
            target, conf, rule = AVERITEC_LABEL_MAPPING[cleaned]
            return target, "accepted", conf, rule
        return None, "excluded", 0.0, "averitec_unknown_label_fail_closed"

    if dataset == "multifc":
        if cleaned in MULTIFC_ACCEPTED_MAPPING:
            target, conf, rule = MULTIFC_ACCEPTED_MAPPING[cleaned]
            return target, "accepted", conf, rule
        if cleaned in MULTIFC_EXCLUDED_LABELS:
            reason = MULTIFC_EXCLUDED_LABELS[cleaned]
            return None, "excluded", 0.0, f"multifc_{reason}"
        return None, "ambiguous", 0.0, "multifc_unverified_label_fail_closed"

    if dataset == "fever":
        if cleaned in FEVER_LABEL_MAPPING:
            target, conf, rule = FEVER_LABEL_MAPPING[cleaned]
            return target, "accepted", conf, rule
        return None, "excluded", 0.0, "fever_unknown_label_fail_closed"

    return None, "excluded", 0.0, f"unknown_dataset_{dataset}"
