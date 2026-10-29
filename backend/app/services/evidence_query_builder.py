"""Deterministic Evidence Query Builder and Plan Generator for GDELT News Discovery (V2.2+).

Implements progressive query relaxation with a 4-tier query plan:
  Tier A (Precise): 2-4 strong concepts (named entity + key object/event phrase)
  Tier B (Entity + OR concepts): Anchor entity + claim-derived OR group
  Tier C (Relaxed): Strongest 1-3 claim terms
  Tier D (DOC Broad): Relaxed query targeted at DOC endpoint with wider timespan

Sanitizes user input against GDELT operator injection, URL fragments, control
characters, and user-supplied boolean operators. Zero LLM tokens are consumed;
all logic is strictly deterministic, bounded, and testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import List, Optional, Set

logger = logging.getLogger(__name__)

# Standard English stopwords plus reporting meta-words that dilute retrieval
STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd",
    "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "just",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than",
    "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
    "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've",
    "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while",
    "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "say", "said", "says",
    "also", "according", "reported", "reportedly", "claims", "claimed", "alleged", "allegedly",
    "announced", "announces", "stating", "stated", "confirmed", "revealed",
    "opened", "opens", "rose", "rise", "rises", "fell", "falls", "grew",
}

# GDELT special operators and user syntax to neutralize
GDELT_INJECTION_PATTERNS = [
    re.compile(r"\bdomain:\S*", re.IGNORECASE),
    re.compile(r"\bsourcelang:\S*", re.IGNORECASE),
    re.compile(r"\bsourcecountry:\S*", re.IGNORECASE),
    re.compile(r"\bnear:\S*", re.IGNORECASE),
    re.compile(r"\brepeat:\S*", re.IGNORECASE),
    re.compile(r"\btone:\S*", re.IGNORECASE),
    re.compile(r"\btheme:\S*", re.IGNORECASE),
    re.compile(r"\bOR\b"),
    re.compile(r"\bAND\b"),
    re.compile(r"\bNOT\b"),
]

# High-frequency domain phrases to recognize as compound concepts
KNOWN_COMPOUND_PHRASES: tuple[str, ...] = (
    "electric vehicle",
    "electric vehicles",
    "solar energy",
    "renewable energy",
    "artificial intelligence",
    "climate change",
    "clean energy",
    "machine learning",
    "air quality",
    "battery storage",
    "space exploration",
    "green hydrogen",
)

# URL pattern to neutralize
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

MAX_QUERY_TERMS = 8
MAX_QUERY_LENGTH = 150
MAX_PLAN_ATTEMPTS = 4
CURRENT_YEAR = 2026


@dataclass(frozen=True)
class EvidenceSearchQuery:
    """A single structured query within a progressive search plan."""

    query: str
    strategy: str  # "precise", "entity_anchor", "relaxed", "broad"
    priority: int  # 1 (highest) to 4 (lowest)
    endpoint_preference: str  # "context", "doc", or "any"
    timespan: Optional[str] = None  # None uses endpoint default


@dataclass
class EvidenceQueryPlan:
    """Ordered progressive query plan for a claim."""

    claim_text: str
    queries: List[EvidenceSearchQuery] = field(default_factory=list)
    detected_year: Optional[int] = None
    is_historical: bool = False

    def __iter__(self):
        return iter(self.queries)

    def __len__(self) -> int:
        return len(self.queries)


class EvidenceQueryBuilder:
    """Deterministic, progressive, injection-safe query builder for news discovery."""

    def __init__(
        self,
        max_terms: int = MAX_QUERY_TERMS,
        max_length: int = MAX_QUERY_LENGTH,
        max_attempts: int = MAX_PLAN_ATTEMPTS,
        current_year: int = CURRENT_YEAR,
        doc_timespan: str = "3m",
    ) -> None:
        self.max_terms = max_terms
        self.max_length = max_length
        self.max_attempts = max_attempts
        self.current_year = current_year
        self.doc_timespan = doc_timespan

    def build_query(self, claim_text: str) -> str:
        """Construct the primary search query for backward compatibility.

        Args:
            claim_text: Raw claim or proposition string.

        Returns:
            Sanitized, space-separated primary query string.
        """
        plan = self.build_plan(claim_text)
        if not plan.queries:
            return ""
        return plan.queries[0].query

    def build_plan(self, claim_text: str) -> EvidenceQueryPlan:
        """Construct a progressive multi-tier query plan from claim text.

        Args:
            claim_text: Raw claim or proposition string.

        Returns:
            EvidenceQueryPlan containing up to 4 ordered queries.
        """
        if not claim_text or not isinstance(claim_text, str) or not claim_text.strip():
            return EvidenceQueryPlan(claim_text=str(claim_text or ""))

        raw_text = claim_text.strip()

        # 1. Detect year & historical status
        year_matches = re.findall(r"\b(19\d\d|20\d\d)\b", raw_text)
        detected_year: Optional[int] = int(year_matches[0]) if year_matches else None
        is_historical: bool = detected_year is not None and detected_year < self.current_year

        # 2. Neutralize inbound URLs & GDELT operator injections
        text = URL_PATTERN.sub(" ", raw_text)
        for pattern in GDELT_INJECTION_PATTERNS:
            text = pattern.sub(" ", text)

        # 3. Extract named multi-word sequences (e.g. "Ho Chi Minh City", "Nobel Prize")
        named_phrases = self._extract_named_phrases(text)

        # 4. Extract known domain compound bigrams (e.g. "electric vehicle", "solar energy")
        compound_phrases = self._extract_compound_phrases(text)

        # Combine detected multi-word phrases (preserving original case)
        all_phrases = named_phrases + [p for p in compound_phrases if p.lower() not in {np.lower() for np in named_phrases}]

        # 5. Normalize punctuation and currencies for content word extraction
        clean_text = re.sub(r'["\'`()\[\]{}<>;:\^|\\~*?]', " ", text)
        clean_text = re.sub(r'[$€£¥]', " ", clean_text)

        # 6. Extract high-value content tokens (entities, numbers, content nouns)
        content_tokens = self._extract_content_tokens(clean_text)

        # If no tokens at all could be extracted
        if not content_tokens and not all_phrases:
            raw_fallback = [t.strip(".,!?-") for t in raw_text.split() if t.strip(".,!?-")][:3]
            fallback_q = " ".join(raw_fallback)
            if not fallback_q:
                return EvidenceQueryPlan(claim_text=raw_text)
            return EvidenceQueryPlan(
                claim_text=raw_text,
                queries=[
                    EvidenceSearchQuery(
                        query=fallback_q[:self.max_length],
                        strategy="precise",
                        priority=1,
                        endpoint_preference="doc" if is_historical else "context",
                    )
                ],
                detected_year=detected_year,
                is_historical=is_historical,
            )

        # 7. Build progressive query plan
        queries: List[EvidenceSearchQuery] = []
        seen_queries: Set[str] = set()

        def add_query(q_str: str, strategy: str, priority: int, endpoint: str, timespan: Optional[str] = None) -> None:
            q_clean = self._bound_query(q_str)
            if q_clean and q_clean not in seen_queries and len(queries) < self.max_attempts:
                seen_queries.add(q_clean)
                queries.append(
                    EvidenceSearchQuery(
                        query=q_clean,
                        strategy=strategy,
                        priority=priority,
                        endpoint_preference=endpoint,
                        timespan=timespan,
                    )
                )

        # Determine primary anchor
        anchor_phrase = all_phrases[0] if all_phrases else None
        anchor_token = None
        if not anchor_phrase:
            # First capitalized entity or first strong content token
            for t in content_tokens:
                if t[0].isupper() and len(t) > 1:
                    anchor_token = t
                    break
            if not anchor_token and content_tokens:
                anchor_token = content_tokens[0]

        # Remaining content tokens excluding tokens already inside the anchor
        anchor_words_lower = set(anchor_phrase.lower().split()) if anchor_phrase else ({anchor_token.lower()} if anchor_token else set())
        raw_remaining = [t for t in content_tokens if t.lower() not in anchor_words_lower]
        remaining_tokens = self._prioritize_tokens(raw_remaining)

        # TIER A: PRECISE (Target: 2-4 strong concepts)
        precise_endpoint = "doc" if is_historical else "context"
        if anchor_phrase:
            # Anchor phrase + prioritized remaining concepts
            selected = [f'"{anchor_phrase}"'] + remaining_tokens[:3]
            add_query(" ".join(selected), strategy="precise", priority=1, endpoint=precise_endpoint)
        elif anchor_token:
            selected = [anchor_token] + remaining_tokens[:3]
            add_query(" ".join(selected), strategy="precise", priority=1, endpoint=precise_endpoint)
        else:
            add_query(" ".join(content_tokens[:3]), strategy="precise", priority=1, endpoint=precise_endpoint)

        # TIER B: ENTITY + OR CONCEPTS (Target: 1 anchor + claim-derived OR group)
        tier_b_endpoint = "doc" if is_historical else "context"
        or_candidates = [t for t in remaining_tokens if not re.search(r"\d", t)][:3]
        if (anchor_phrase or anchor_token) and len(or_candidates) >= 2:
            anchor_str = f'"{anchor_phrase}"' if anchor_phrase else anchor_token
            or_group = f"({or_candidates[0]} OR {or_candidates[1]})"
            add_query(f"{anchor_str} {or_group}", strategy="entity_anchor", priority=2, endpoint=tier_b_endpoint)
        elif len(content_tokens) >= 3 and not re.search(r"\d", content_tokens[0]):
            or_group = f"({content_tokens[1]} OR {content_tokens[2]})"
            add_query(f"{content_tokens[0]} {or_group}", strategy="entity_anchor", priority=2, endpoint=tier_b_endpoint)
        elif remaining_tokens:
            anchor_str = f'"{anchor_phrase}"' if anchor_phrase else (anchor_token or content_tokens[0])
            add_query(f"{anchor_str} {remaining_tokens[0]}", strategy="entity_anchor", priority=2, endpoint=tier_b_endpoint)

        # TIER C: RELAXED (Target: strongest 1-2 concepts, high recall)
        tier_c_endpoint = "doc"
        if anchor_phrase and remaining_tokens:
            add_query(f'"{anchor_phrase}" {remaining_tokens[0]}', strategy="relaxed", priority=3, endpoint=tier_c_endpoint)
        elif len(content_tokens) >= 2:
            add_query(f"{content_tokens[0]} {content_tokens[1]}", strategy="relaxed", priority=3, endpoint=tier_c_endpoint)
        elif anchor_phrase:
            add_query(f'"{anchor_phrase}"', strategy="relaxed", priority=3, endpoint=tier_c_endpoint)
        elif content_tokens:
            add_query(content_tokens[0], strategy="relaxed", priority=3, endpoint=tier_c_endpoint)

        # TIER D: DOC BROAD SEARCH (Target: broad 1-2 term query, timespan=3m)
        broad_terms = [t for t in content_tokens if len(t) > 2 and not re.search(r"\d", t)]
        if len(broad_terms) >= 2:
            add_query(f"{broad_terms[0]} {broad_terms[1]}", strategy="broad", priority=4, endpoint="doc", timespan=self.doc_timespan)
        elif anchor_phrase:
            add_query(f'"{anchor_phrase}"', strategy="broad", priority=4, endpoint="doc", timespan=self.doc_timespan)
        elif broad_terms:
            add_query(broad_terms[0], strategy="broad", priority=4, endpoint="doc", timespan=self.doc_timespan)

        # If fewer than 2 queries generated, fill with safe variations
        if len(queries) < 2 and len(content_tokens) >= 2:
            add_query(f"{content_tokens[0]} {content_tokens[1]}", strategy="relaxed", priority=len(queries) + 1, endpoint="doc")

        return EvidenceQueryPlan(
            claim_text=raw_text,
            queries=queries,
            detected_year=detected_year,
            is_historical=is_historical,
        )

    def _extract_named_phrases(self, text: str) -> List[str]:
        """Extract multi-word capitalized phrases (e.g. 'Ho Chi Minh City', 'Nobel Prize')."""
        # Match sequences of 2 or more capitalized words, allowing Roman numerals (like Artemis III)
        pattern = r"\b[A-Z][a-zA-Z]*(?:\s+(?:[A-Z][a-zA-Z]*|[I|V|X]+))+\b"
        raw_phrases = re.findall(pattern, text)
        phrases: List[str] = []
        for p in raw_phrases:
            words = p.split()
            # If the entire phrase is just a capitalized stopword at start of sentence, skip
            if len(words) == 1 and words[0].lower() in STOPWORDS:
                continue
            # Filter out phrases where all words are stopwords
            if all(w.lower() in STOPWORDS for w in words):
                continue
            phrases.append(p)
        return phrases

    def _extract_compound_phrases(self, text: str) -> List[str]:
        """Extract high-frequency domain compound phrases (e.g. 'electric vehicle', 'solar energy')."""
        found: List[str] = []
        text_lower = text.lower()
        for phrase in KNOWN_COMPOUND_PHRASES:
            if re.search(r"\b" + re.escape(phrase) + r"\b", text_lower):
                # Locate in text to preserve original casing if present
                match = re.search(r"\b" + re.escape(phrase) + r"\b", text, re.IGNORECASE)
                if match:
                    found.append(match.group(0))
        return found

    def _extract_content_tokens(self, clean_text: str) -> List[str]:
        """Tokenize and filter text into high-value entities, numbers, and content words."""
        raw_tokens = [t.strip(".,!?-") for t in clean_text.split()]
        raw_tokens = [t for t in raw_tokens if t]

        selected: List[str] = []
        seen_lower: Set[str] = set()

        for token in raw_tokens:
            token_clean = token.strip()
            if not token_clean:
                continue

            lower = token_clean.lower()
            is_numeric = bool(re.search(r"\d", token_clean))
            is_capitalized = token_clean[0].isupper() and len(token_clean) > 1

            if lower in STOPWORDS and not is_numeric:
                continue

            if lower not in seen_lower:
                seen_lower.add(lower)
                selected.append(token_clean)

            if len(selected) >= self.max_terms:
                break

        return selected

    def _prioritize_tokens(self, tokens: List[str]) -> List[str]:
        """Rank tokens prioritizing years, entities, and numbers over generic terms."""
        years = [t for t in tokens if re.match(r"^(?:19|20)\d\d$", t)]
        entities = [t for t in tokens if t[0].isupper() and len(t) > 1 and t not in years]
        numbers = [t for t in tokens if (re.search(r"\d", t) or t.lower() in ("million", "billion", "trillion")) and t not in years]
        others = [t for t in tokens if t not in entities and t not in numbers and t not in years]
        return years + entities + numbers + others

    def _bound_query(self, query: str) -> str:
        """Enforce maximum length bounds and normalize inner spaces."""
        cleaned = re.sub(r"\s+", " ", query).strip()
        if len(cleaned) > self.max_length:
            cleaned = cleaned[: self.max_length].rsplit(" ", 1)[0].strip()
        return cleaned
