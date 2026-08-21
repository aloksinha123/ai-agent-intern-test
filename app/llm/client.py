"""Gemini LLM client wrapper and response orchestration."""

from dataclasses import asdict, dataclass, field
import os
import re
import time
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from app.llm.prompts import build_orchestrator_prompt, build_order_orchestrator_prompt
from app.rag.conflict import EvidenceAnalysis
from app.rag.models import SearchResult

DEFAULT_LLM_MODEL = "gemini-3.6-flash"


def validate_citations(
    text: str,
    evidence: List[SearchResult],
    analysis: Optional[EvidenceAnalysis] = None,
) -> List[str]:
    """Extract and validate citations from LLM text strictly against supporting evidence.

    Citation rules:
    - CONFLICT: Must cite only the sources participating in the detected conflict
      (and any directly involved supporting source from analysis.conflicting_claims).
      Unrelated retrieved chunks are excluded.
    - INSUFFICIENT: Unrelated chunks are not presented as supporting evidence.
    - CONSISTENT: Citations matching an exact (filename, heading) present in the retrieved
      customer evidence pool and cited by the LLM are accepted. Fabricated citations are rejected.

    Args:
        text: Generated response text from LLM.
        evidence: Authoritative list of retrieved SearchResult objects.
        analysis: Optional EvidenceAnalysis result providing conflict provenance.

    Returns:
        List of validated citation strings formatted as 'filename.md — Heading'.
    """
    if not evidence:
        return []

    # If conflict is detected, authoritative citations are strictly the conflicting sources
    if analysis is not None and analysis.status == "conflict" and analysis.conflicting_claims:
        conflict_sources: List[str] = []
        for c in analysis.conflicting_claims:
            s_a = f"{c.source_a} — {c.heading_a}"
            s_b = f"{c.source_b} — {c.heading_b}"
            if s_a not in conflict_sources:
                conflict_sources.append(s_a)
            if s_b not in conflict_sources:
                conflict_sources.append(s_b)
        return conflict_sources

    # If evidence is insufficient, do not cite unrelated chunks
    if analysis is not None and analysis.status == "insufficient":
        return []

    valid_map = {
        (s.filename.lower().strip(), s.heading.lower().strip()): f"{s.filename} — {s.heading}"
        for s in evidence
    }

    # Find citation lines formatted like: "- 01-returns-policy-current.md — Standard return window"
    citation_pattern = re.compile(
        r"^[ \t]*[-*]\s*([0-9a-zA-Z_.-]+\.md)\s*[—–-]\s*(.+?)[ \t]*$",
        re.MULTILINE,
    )
    matches = citation_pattern.findall(text)

    validated: List[str] = []
    for fname, heading in matches:
        key = (fname.lower().strip(), heading.lower().strip())
        if key in valid_map:
            canonical = valid_map[key]
            if canonical not in validated:
                validated.append(canonical)

    return validated


@dataclass(frozen=True)
class LLMResponse:
    """Structured response container from the LLM orchestrator."""

    text: str
    model_name: str
    analysis_status: str
    handoff_required: bool
    evidence_count: int
    evidence_chunk_ids: List[str]
    citations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return safe observability dictionary without sensitive data or API keys."""
        return {
            "text": self.text,
            "model_name": self.model_name,
            "analysis_status": self.analysis_status,
            "handoff_required": self.handoff_required,
            "evidence_count": self.evidence_count,
            "evidence_chunk_ids": self.evidence_chunk_ids,
            "citations": self.citations,
        }


class GeminiClient:
    """Orchestrator client interacting with Google GenAI SDK for grounded response generation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        client: Optional[Any] = None,
    ) -> None:
        """Initialize the Gemini client.

        Args:
            api_key: Optional Gemini API key. Defaults to GEMINI_API_KEY environment variable.
            model_name: Optional model identifier. Defaults to LLM_MODEL env var or gemini-2.5-flash.
            client: Optional pre-configured genai.Client instance (useful for mocking/testing).

        Raises:
            ValueError: If api_key is not provided and not found in environment.
        """
        self.model_name = model_name or os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if client is not None:
            self.client = client
        else:
            if not self.api_key:
                raise ValueError(
                    "Missing GEMINI_API_KEY. Please set the GEMINI_API_KEY environment variable "
                    "or pass api_key directly to GeminiClient."
                )
            self.client = genai.Client(api_key=self.api_key)

    def _call_generate_with_retry(
        self,
        contents: str,
        system_instruction: str,
        max_retries: int = 5,
    ) -> str:
        """Call Gemini API with exponential backoff on transient errors."""
        delay = 5.0
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.0,
                    ),
                )
                return response.text if hasattr(response, "text") and response.text else ""
            except Exception as e:
                err_str = str(e).lower()
                if ("503" in err_str or "429" in err_str or "unavailable" in err_str or "quota" in err_str or "resource_exhausted" in err_str) and attempt < max_retries - 1:
                    sleep_time = 15.0 if "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str else delay
                    time.sleep(sleep_time)
                    delay *= 2
                else:
                    raise

        return ""

    def generate_response(
        self,
        query: str,
        evidence: List[SearchResult],
        analysis: EvidenceAnalysis,
    ) -> LLMResponse:
        """Generate a grounded customer-facing response given the query, evidence, and analysis.

        Deterministic authority rules:
        1. Handoff authority belongs strictly to EvidenceAnalysis (CONSISTENT: False, CONFLICT/INSUFFICIENT: True).
        2. Citations must be strictly validated against retrieved evidence; fabricated citations are dropped.

        Args:
            query: The customer's inquiry.
            evidence: List of retrieved SearchResult objects.
            analysis: EvidenceAnalysis result from Milestone 3C.

        Returns:
            LLMResponse object.
        """
        system_instruction, user_content = build_orchestrator_prompt(
            query=query,
            evidence=evidence,
            analysis=analysis,
        )

        response_text = self._call_generate_with_retry(
            contents=user_content,
            system_instruction=system_instruction,
        )

        # Validate citations from LLM output strictly against retrieved evidence and analysis
        validated_citations = validate_citations(response_text, evidence, analysis=analysis)

        # Application owns handoff determination derived directly from analysis
        handoff_decision = bool(analysis.handoff_required)

        return LLMResponse(
            text=response_text,
            model_name=self.model_name,
            analysis_status=analysis.status,
            handoff_required=handoff_decision,
            evidence_count=len(evidence),
            evidence_chunk_ids=[s.chunk_id for s in evidence],
            citations=validated_citations,
        )

    def generate_order_response(
        self,
        query: str,
        order_data: Dict[str, Any],
        handoff_required: bool = False,
    ) -> LLMResponse:
        """Generate a customer-facing order response given query and sanitized tool data.

        Args:
            query: Customer's inquiry.
            order_data: Sanitized customer-safe order dictionary.
            handoff_required: Deterministic handoff flag from order service.

        Returns:
            LLMResponse object.
        """
        system_instruction, user_content = build_order_orchestrator_prompt(
            query=query,
            order_data=order_data,
            handoff_required=handoff_required,
        )

        response_text = self._call_generate_with_retry(
            contents=user_content,
            system_instruction=system_instruction,
        )

        return LLMResponse(
            text=response_text,
            model_name=self.model_name,
            analysis_status="order_lookup",
            handoff_required=bool(handoff_required),
            evidence_count=0,
            evidence_chunk_ids=[],
            citations=[],
        )

