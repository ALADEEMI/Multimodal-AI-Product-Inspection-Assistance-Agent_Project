"""Information Extraction Module - Rule-Based Extraction of Structured Fields.

Extracts: Product, Problem, Severity, Requested_Action from NLP outputs.
Follows ADR Phase 5 specifications.
"""

import re
from typing import Dict, Any, Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import REQUESTED_ACTIONS


# Action keywords mapping (Arabic and English)
ACTION_KEYWORDS = {
    "return": [
        "return", "refund", "give back", "send back",
        "ارجاع", "إرجاع", "استرجاع", "رد", "ارد"
    ],
    "replacement": [
        "replace", "replacement", "exchange", "swap", "new one",
        "استبدال", "بديل", "تبديل", "واحد جديد"
    ],
    "repair": [
        "repair", "fix", "fixing",
        "إصلاح", "تصليح", "اصلح", "صلاح"
    ]
}


def extract_requested_action(text: str, nlp_results: Dict[str, Any]) -> str:
    """Extract requested action from text.

    Priority: return > replacement > repair > unspecified
    """
    if not text or not isinstance(text, str):
        return "unspecified"

    text_lower = text.lower()

    # Check for explicit action keywords
    for action, keywords in ACTION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                return action

    # If complaint is detected but no action specified
    if nlp_results.get("is_complaint", False):
        # Default complaint action based on severity
        severity = nlp_results.get("severity", "Low")
        if severity == "High":
            return "replacement"
        elif severity == "Medium":
            return "return"

    return "unspecified"


def extract_information(
    raw_text: Optional[str],
    nlp_results: Dict[str, Any],
    vision_results: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Extract structured information from analysis results.

    Args:
        raw_text: Original customer text (may be None for image-only)
        nlp_results: Output from NLP inference
        vision_results: Optional output from vision inference

    Returns:
        Structured extraction dict with:
          - Product, Problem, Severity, Requested_Action, evidence
    """
    # Default extraction
    extraction = {
        "Product": "bottle",  # Fixed for this project
        "Problem": nlp_results.get("problem_type", "Other"),
        "Severity": "Low",  # Default
        "Requested_Action": "unspecified",
        "evidence": {
            "text_confidence": nlp_results.get("problem_type_confidence", 0.0) if nlp_results else 0.0,
            "vision_confidence": vision_results.get("segmentation_confidence", 0.0) if vision_results else 0.0,
            "has_text": bool(raw_text and raw_text.strip()),
            "has_image": bool(vision_results)
        }
    }

    # Determine severity from NLP results or vision
    if nlp_results and nlp_results.get("is_complaint"):
        # Map problem type to typical severity
        problem_type = nlp_results.get("problem_type", "Other")
        if problem_type == "Damage":
            extraction["Severity"] = "High"
        elif problem_type == "Quality":
            extraction["Severity"] = "Medium"
        else:
            extraction["Severity"] = "Low"

    # Override with vision severity if image shows major defect
    if vision_results and vision_results.get("has_defect"):
        defect_area_ratio = vision_results.get("defect_area_ratio", 0.0)
        if defect_area_ratio > 0.1:
            extraction["Severity"] = "High"
        elif defect_area_ratio > 0.03:
            extraction["Severity"] = "Medium"

    # Extract requested action
    if raw_text:
        extraction["Requested_Action"] = extract_requested_action(raw_text, nlp_results)

    return extraction
