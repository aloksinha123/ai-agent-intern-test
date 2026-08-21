"""Metadata-aware retrieval policy, authority filtering, and precedence ranking."""

import re
from typing import Any, Dict, List, Optional, Tuple

from app.rag.models import SearchResult


class RetrievalPolicy:
    """Evaluates chunk eligibility and computes metadata-aware precedence ranking."""

    TOPIC_KEYWORD_MAP: Dict[str, str] = {
        "trailplus": "09-trailplus-membership.md",
        "trail plus": "09-trailplus-membership.md",
        "breeze": "12-breeze-tumbler-product-card.md",
        "tumbler": "12-breeze-tumbler-product-card.md",
        "warranty": "07-warranty.md",
        "cancellation": "08-order-changes-and-cancellations.md",
        "cancel": "08-order-changes-and-cancellations.md",
        "gift card": "10-gift-cards-and-price-adjustments.md",
        "price adjustment": "10-gift-cards-and-price-adjustments.md",
    }

    @classmethod
    def is_eligible(cls, metadata: Dict[str, Any], mode: str = "customer") -> Tuple[bool, str]:
        """Determine whether a chunk is eligible under the given retrieval mode.

        Args:
            metadata: The chunk's metadata dictionary.
            mode: Retrieval mode ('customer', 'internal', or 'all').

        Returns:
            Tuple of (is_eligible, policy_reason).
        """
        status = str(metadata.get("status", "")).strip().lower()
        authority = str(metadata.get("policy_authority", "")).strip().lower()
        audience = str(metadata.get("audience", "")).strip().lower()
        customer_answering = metadata.get("customer_answering", None)

        if mode == "customer":
            # 1. Status checks
            if status == "superseded":
                return False, "excluded_superseded_policy"
            if status == "draft":
                return False, "excluded_draft_content"
            if status != "active":
                return False, f"excluded_inactive_status_{status}"

            # 2. Authority checks
            if authority == "none":
                return False, "excluded_no_policy_authority"
            if authority != "official":
                return False, f"excluded_non_official_authority_{authority}"

            # 3. Audience checks
            if audience != "customer":
                return False, f"excluded_non_customer_audience_{audience}"

            # 4. Explicit customer answering flag (if explicitly False, exclude)
            if customer_answering is False:
                return False, "excluded_customer_answering_false"

            return True, "eligible_customer_evidence"

        elif mode == "internal":
            # Internal mode: allow active internal guidelines (e.g. 13-support-escalation.md)
            if status == "draft":
                return False, "excluded_draft_content"
            if authority == "none":
                return False, "excluded_no_policy_authority"
            if audience == "internal" or (status == "active" and authority == "official"):
                return True, "eligible_internal_evidence"
            return False, "excluded_internal_policy"

        elif mode == "all":
            # Audit mode: all chunks pass through for diagnostic inspection
            return True, "audit_all"

        else:
            raise ValueError(f"Unknown retrieval mode: '{mode}'. Must be 'customer', 'internal', or 'all'.")

    @classmethod
    def compute_precedence_score(cls, result: SearchResult, query: str = "") -> float:
        """Compute precedence-adjusted ranking score.

        Rules:
        - Base score is the cosine similarity score from vector search.
        - Specificity bonus (+0.15): When the user query explicitly targets a specific entity
          or policy area (e.g. 'TrailPlus') and the chunk belongs directly to the dedicated
          governing document for that entity, a modest specificity bonus is applied so primary
          authoritative sources rank above generic cross-references.

        Args:
            result: SearchResult object.
            query: The user's search query string.

        Returns:
            Precedence-adjusted float score.
        """
        score = result.score
        query_lower = query.lower()

        # Specificity adjustment for dedicated topic documents
        for keyword, target_file in cls.TOPIC_KEYWORD_MAP.items():
            if re.search(rf"\b{re.escape(keyword)}\b", query_lower):
                if result.filename == target_file:
                    score += 0.15
                    break

        return score

    @classmethod
    def diversify_sources(cls, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        """Deterministically diversify retrieved results across distinct source documents.

        Pass 1: Selects the highest-scoring chunk from each unique source document.
        Pass 2: Fills remaining top_k slots with subsequent highest-scoring chunks in score order.

        Args:
            results: Precedence-sorted list of SearchResult objects.
            top_k: Number of results to return.

        Returns:
            Diversified list of SearchResult objects up to top_k.
        """
        if len(results) <= top_k or top_k <= 1:
            return results[:top_k]

        selected: List[SearchResult] = []
        seen_filenames: set = set()
        secondary: List[SearchResult] = []

        # Pass 1: Select highest-scoring chunk per distinct document
        for res in results:
            if res.filename not in seen_filenames:
                seen_filenames.add(res.filename)
                selected.append(res)
                if len(selected) == top_k:
                    return selected
            else:
                secondary.append(res)

        # Pass 2: Fill remaining slots with next best chunks
        for res in secondary:
            selected.append(res)
            if len(selected) == top_k:
                break

        return selected

    @classmethod
    def apply(
        cls,
        candidates: List[SearchResult],
        query: str = "",
        mode: str = "customer",
        top_k: int = 5,
        diversify: bool = True,
    ) -> List[SearchResult]:
        """Filter candidates by metadata eligibility and rank by precedence with source diversification.

        Args:
            candidates: Unfiltered search results from FAISS stage 1.
            query: Query string.
            mode: Retrieval mode ('customer', 'internal', or 'all').
            top_k: Number of final results to return.
            diversify: Whether to apply source diversification across distinct documents.

        Returns:
            Filtered, ranked, and diversified List of SearchResult objects.
        """
        processed: List[SearchResult] = []

        for cand in candidates:
            eligible, reason = cls.is_eligible(cand.metadata, mode=mode)
            final_sc = cls.compute_precedence_score(cand, query=query) if eligible else 0.0

            evaluated_result = SearchResult(
                score=cand.score,
                chunk_id=cand.chunk_id,
                filename=cand.filename,
                document_title=cand.document_title,
                heading=cand.heading,
                content=cand.content,
                metadata=cand.metadata,
                policy_eligible=eligible,
                policy_reason=reason,
                final_score=final_sc,
            )
            processed.append(evaluated_result)

        if mode == "all":
            # For audit mode, return all evaluated results preserving original ranking order
            return processed[:top_k]

        # Filter to only eligible results and sort by final_score descending
        eligible_results = [r for r in processed if r.policy_eligible]
        eligible_results.sort(key=lambda r: r.final_score, reverse=True)

        if diversify:
            return cls.diversify_sources(eligible_results, top_k=top_k)

        return eligible_results[:top_k]

