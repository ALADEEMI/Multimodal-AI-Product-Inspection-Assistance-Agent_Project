"""
Agent State Schema for LangGraph Multimodal Inspection Agent.
Follows ADR Phase 7 specifications.
"""

from typing import Dict, Any, Optional, List, TypedDict
from dataclasses import dataclass, field
import uuid


class InspectionState(TypedDict, total=False):
    """
    Serializable state schema for LangGraph workflow.
    Tracks user inputs, model outputs, confidences, extracted fields,
    fusion results, policy lookups, and final responses.
    """
    # Trace & Metadata
    trace_id: str
    timestamp: str

    # User Inputs
    raw_text: Optional[str]
    image_path: Optional[str]

    # Modality Routing Flags
    has_text: bool
    has_image: bool
    is_valid_input: bool

    # Model Outputs
    nlp_results: Optional[Dict[str, Any]]
    vision_results: Optional[Dict[str, Any]]
    fusion_results: Optional[Dict[str, Any]]

    # Information Extraction
    extracted_fields: Optional[Dict[str, Any]]

    # Local Policy Tool Results
    policy_lookup_required: bool
    policy_results: Optional[Dict[str, Any]]

    # Context for LLM
    evidence_context: Optional[str]

    # Final Responses
    llm_response: Optional[str]
    final_report: Optional[Dict[str, Any]]
    uncertainty_notes: Optional[List[str]]

    # Error Tracking
    errors: List[str]
