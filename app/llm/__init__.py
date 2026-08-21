"""LLM orchestration package for Aster & Row support agent."""

from app.llm.client import DEFAULT_LLM_MODEL, GeminiClient, LLMResponse, validate_citations
from app.llm.prompts import (
    SYSTEM_INSTRUCTION,
    build_orchestrator_prompt,
    build_order_orchestrator_prompt,
    format_analysis_block,
    format_evidence_block,
    format_order_tool_block,
)

__all__ = [
    "DEFAULT_LLM_MODEL",
    "GeminiClient",
    "LLMResponse",
    "validate_citations",
    "SYSTEM_INSTRUCTION",
    "build_orchestrator_prompt",
    "build_order_orchestrator_prompt",
    "format_evidence_block",
    "format_analysis_block",
    "format_order_tool_block",
]
