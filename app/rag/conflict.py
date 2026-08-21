"""Deterministic evidence analysis, conflict detection, and uncertainty evaluation."""

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.rag.models import SearchResult


@dataclass(frozen=True)
class ConflictingClaim:
    """Represents a specific contradiction detected between two source documents."""

    topic: str
    source_a: str
    heading_a: str
    claim_a: str
    source_b: str
    heading_b: str
    claim_b: str
    description: str


@dataclass(frozen=True)
class EvidenceAnalysis:
    """Structured result of analyzing retrieved evidence for consistency, conflict, or insufficiency."""

    status: str  # "consistent" | "conflict" | "insufficient"
    reason: str
    sources: List[SearchResult]
    conflicting_claims: List[ConflictingClaim] = field(default_factory=list)
    handoff_required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to a JSON-serializable dictionary for structured logging."""
        return {
            "status": self.status,
            "reason": self.reason,
            "handoff_required": self.handoff_required,
            "conflicting_claims": [asdict(c) for c in self.conflicting_claims],
            "sources": [
                {
                    "filename": s.filename,
                    "heading": s.heading,
                    "chunk_id": s.chunk_id,
                    "score": s.score,
                    "final_score": s.final_score,
                }
                for s in self.sources
            ],
        }


class EvidenceAnalyzer:
    """Deterministic analyzer detecting contradictions and gaps in retrieved customer evidence."""

    # Generic polar contradiction patterns: (positive_regex, negative_regex, topic_label)
    CONTRADICTION_PATTERNS: List[Tuple[re.Pattern, re.Pattern, str]] = [
        # Dishwasher / washing compatibility
        (
            re.compile(r"\b(all\s+components\s+are\s+dishwasher\s+safe|dishwasher\s+safe|machine\s+washable)\b", re.I),
            re.compile(r"\b(hand-washed|hand-wash|hand\s+wash|spot-clean|do\s+not\s+machine\s+wash|do\s+not\s+place\s+in\s+dishwasher)\b", re.I),
            "washing_and_care_instructions",
        ),
        # Fee vs Free
        (
            re.compile(r"\b(free\s+return\s+label|free\s+domestic\s+returns?|no\s+fee)\b", re.I),
            re.compile(r"\b(\$\d+(\.\d+)?\s+return\s+shipping\s+fee|fee\s+is\s+deducted|customer\s+is\s+responsible\s+for\s+return\s+postage)\b", re.I),
            "return_shipping_fee",
        ),
        # Absolute permission vs absolute prohibition
        (
            re.compile(r"\b(is\s+allowed|permitted|fully\s+covered|offers?\s+a\s+lifetime\s+warranty|lifetime\s+warranty)\b", re.I),
            re.compile(r"\b(not\s+allowed|prohibited|not\s+covered|does\s+not\s+offer\s+a\s+lifetime\s+warranty|no\s+lifetime\s+warranty)\b", re.I),
            "policy_permission_and_coverage",
        ),
    ]

    # Specific factual attributes that indicate ungrounded out-of-corpus queries when completely unmentioned
    UNGROUNDED_QUERY_TERMS: Set[str] = {
        "vegan",
        "cruelty-free",
        "biodegradable",
        "hypoallergenic",
        "latex",
        "adhesive",
        "organic certification",
    }

    # Known distinct product categories to avoid false conflicts between different product types
    PRODUCT_CATEGORIES: Dict[str, Set[str]] = {
        "bags": {"bag", "bags", "backpack", "backpacks", "daypack", "weekender"},
        "drinkware": {"tumbler", "tumblers", "drinkware", "breeze", "bottle", "cup"},
        "packing_cubes": {"cube", "cubes", "compression"},
        "gift_cards": {"gift card", "gift cards"},
    }

    @classmethod
    def extract_matching_sentences(cls, text: str, pattern: re.Pattern) -> List[str]:
        """Extract individual sentences matching a given regex pattern."""
        sentences = re.split(r"(?<=[.!?\n])\s+", text)
        return [s.strip() for s in sentences if pattern.search(s)]

    @classmethod
    def are_conflicting_entities(cls, src_a: SearchResult, src_b: SearchResult) -> bool:
        """Check whether two search results refer to compatible entities rather than disjoint products."""
        text_a = f"{src_a.document_title} {src_a.heading} {src_a.content}".lower()
        text_b = f"{src_b.document_title} {src_b.heading} {src_b.content}".lower()

        # If both explicitly mention different known product categories, they do not conflict
        for cat_a, words_a in cls.PRODUCT_CATEGORIES.items():
            for cat_b, words_b in cls.PRODUCT_CATEGORIES.items():
                if cat_a != cat_b:
                    a_matches_cat_a = any(re.search(rf"\b{re.escape(w)}\b", text_a) for w in words_a)
                    b_matches_cat_b = any(re.search(rf"\b{re.escape(w)}\b", text_b) for w in words_b)
                    a_matches_cat_b = any(re.search(rf"\b{re.escape(w)}\b", text_a) for w in words_b)
                    b_matches_cat_a = any(re.search(rf"\b{re.escape(w)}\b", text_b) for w in words_a)
                    if a_matches_cat_a and b_matches_cat_b and not (a_matches_cat_b or b_matches_cat_a):
                        return False
        return True

    @classmethod
    def detect_conflicts(cls, sources: List[SearchResult]) -> List[ConflictingClaim]:
        """Detect pairwise polar contradictions between active eligible customer sources."""
        conflicts: List[ConflictingClaim] = []
        if len(sources) < 2:
            return conflicts

        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                src_a = sources[i]
                src_b = sources[j]

                # Chunks from the exact same file cannot contradict each other
                if src_a.filename == src_b.filename:
                    continue

                # Ensure chunks refer to comparable items/contexts
                if not cls.are_conflicting_entities(src_a, src_b):
                    continue

                for pos_pat, neg_pat, topic in cls.CONTRADICTION_PATTERNS:
                    # Positive sentences must NOT match the negative pattern
                    a_pos = [s for s in cls.extract_matching_sentences(src_a.content, pos_pat) if not neg_pat.search(s)]
                    a_neg = cls.extract_matching_sentences(src_a.content, neg_pat)
                    b_pos = [s for s in cls.extract_matching_sentences(src_b.content, pos_pat) if not neg_pat.search(s)]
                    b_neg = cls.extract_matching_sentences(src_b.content, neg_pat)

                    # Case 1: Doc A is positive and Doc B is negative
                    if a_pos and b_neg:
                        conflicts.append(
                            ConflictingClaim(
                                topic=topic,
                                source_a=src_a.filename,
                                heading_a=src_a.heading,
                                claim_a="; ".join(a_pos),
                                source_b=src_b.filename,
                                heading_b=src_b.heading,
                                claim_b="; ".join(b_neg),
                                description=f"Contradiction in {topic}: '{src_a.filename}' asserts positive permission while '{src_b.filename}' asserts restriction/prohibition.",
                            )
                        )
                    # Case 2: Doc A is negative and Doc B is positive
                    elif a_neg and b_pos:
                        conflicts.append(
                            ConflictingClaim(
                                topic=topic,
                                source_a=src_a.filename,
                                heading_a=src_a.heading,
                                claim_a="; ".join(a_neg),
                                source_b=src_b.filename,
                                heading_b=src_b.heading,
                                claim_b="; ".join(b_pos),
                                description=f"Contradiction in {topic}: '{src_a.filename}' asserts restriction/prohibition while '{src_b.filename}' asserts positive permission.",
                            )
                        )

        return conflicts

    @classmethod
    def check_insufficient_information(
        cls, query: str, sources: List[SearchResult]
    ) -> Tuple[bool, str]:
        """Check if retrieved evidence does not establish the requested facts."""
        if not sources:
            return True, "No eligible customer evidence retrieved."

        # Check relevance floor (highest scoring chunk must meet minimum threshold)
        top_score = max((s.score for s in sources), default=0.0)
        if top_score < 0.20:
            return True, f"Top evidence score ({top_score:.3f}) is below minimum confidence threshold."

        # Check for ungrounded topic terms requested in query but unmentioned in retrieved text
        query_lower = query.lower()
        combined_text = " ".join([f"{s.document_title} {s.heading} {s.content}".lower() for s in sources])

        for term in cls.UNGROUNDED_QUERY_TERMS:
            if re.search(rf"\b{re.escape(term)}\b", query_lower):
                # If term is requested by user, it must be addressed in retrieved sources
                if not re.search(rf"\b{re.escape(term)}\b", combined_text):
                    # Special check: if query asks about lifetime warranty and doc 07 explicitly says
                    # "does not offer a lifetime warranty", it IS addressed.
                    # But for topics like "vegan", "adhesives", it is not mentioned at all.
                    return True, f"Query requests information regarding '{term}', which is not established in the supplied documents."

        return False, "Evidence provides sufficient coverage."

    @classmethod
    def analyze(cls, sources: List[SearchResult], query: str = "") -> EvidenceAnalysis:
        """Analyze customer evidence and classify as consistent, conflict, or insufficient.

        Args:
            sources: Filtered customer SearchResult objects.
            query: The user query string.

        Returns:
            Populated EvidenceAnalysis object.
        """
        # 1. Check for genuine conflicts between multiple eligible sources
        conflicts = cls.detect_conflicts(sources)
        if conflicts:
            return EvidenceAnalysis(
                status="conflict",
                reason=f"Detected {len(conflicts)} genuine contradiction(s) between active official sources.",
                sources=sources,
                conflicting_claims=conflicts,
                handoff_required=True,
            )

        # 2. Check for insufficient information
        is_insufficient, reason = cls.check_insufficient_information(query, sources)
        if is_insufficient:
            return EvidenceAnalysis(
                status="insufficient",
                reason=reason,
                sources=sources,
                conflicting_claims=[],
                handoff_required=True,
            )

        # 3. Evidence is consistent and sufficient
        # If query reports a damaged/defective item, policy (Doc 04) mandates human review before approval
        requires_human_review = False
        if any(s.filename == "04-damaged-or-wrong-items.md" for s in sources):
            if re.search(r"\b(damaged|defect|defective|broken\s*zipper|broken|arrived\s*damaged|torn|torn\s*seam|tear|rip|flaw)\b", query.lower()):
                requires_human_review = True

        reason_msg = "Retrieved evidence is consistent and authoritative."
        if requires_human_review:
            reason_msg += " Human support review required before approval of damaged item claims."

        return EvidenceAnalysis(
            status="consistent",
            reason=reason_msg,
            sources=sources,
            conflicting_claims=[],
            handoff_required=requires_human_review,
        )


def analyze_evidence(results: List[SearchResult], query: str = "") -> EvidenceAnalysis:
    """Functional convenience wrapper for EvidenceAnalyzer.analyze."""
    return EvidenceAnalyzer.analyze(sources=results, query=query)
