"""Application services package."""

from backend.app.services.claim_extractor import ClaimExtractor
from backend.app.services.evidence_consensus import EvidenceClusterer, EvidenceConsensusEngine
from backend.app.services.evidence_query_builder import EvidenceQueryBuilder
from backend.app.services.gdelt_client import GDELTClient
from backend.app.services.live_evidence_retriever import LiveEvidenceRetriever
from backend.app.services.text_cleaner import clean_dataframe_text, clean_text

__all__ = [
    "ClaimExtractor",
    "EvidenceClusterer",
    "EvidenceConsensusEngine",
    "EvidenceQueryBuilder",
    "GDELTClient",
    "LiveEvidenceRetriever",
    "clean_text",
    "clean_dataframe_text",
]
