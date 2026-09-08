"""
OpenRouter LLM Adapter with Safe Deterministic Structured Fallback.
Adheres to ADR Phase 7:
- Reads credentials exclusively from environment (.env)
- Never logs API keys
- Formats final response based strictly on evidence in context (zero hallucination)
- Falls back gracefully to template-based deterministic report if API key is absent or call fails
"""

import os
import json
from typing import Dict, Any, Optional
import httpx

from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    OPENROUTER_BASE_URL,
)
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


class LLMAdapter:
    """Adapter for OpenRouter API with zero-hallucination prompting and safe fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 20.0
    ):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.model = model or OPENROUTER_MODEL
        self.base_url = base_url or OPENROUTER_BASE_URL
        self.timeout = timeout

    @property
    def is_available(self) -> bool:
        """Check if a non-empty API key is configured."""
        return bool(self.api_key and self.api_key.strip())

    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate response from LLM via OpenRouter, or return structured fallback.

        Args:
            prompt: User/evidence prompt
            system_prompt: System instructions

        Returns:
            Dict containing 'response_text', 'model_used', 'is_fallback', 'error'
        """
        if not self.is_available:
            logger.info("OpenRouter API key not configured. Using deterministic fallback formatter.")
            return {
                "response_text": None,
                "model_used": "deterministic_fallback",
                "is_fallback": True,
                "error": "OpenRouter API key not configured."
            }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/project-multimodal-inspection",
            "X-Title": "Multimodal Product Inspection Agent",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 600,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )

                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    logger.info("Successfully received response from OpenRouter.")
                    return {
                        "response_text": content,
                        "model_used": self.model,
                        "is_fallback": False,
                        "error": None
                    }
                else:
                    err_msg = f"OpenRouter API returned HTTP {response.status_code}: {response.text}"
                    logger.warning(err_msg)
                    return {
                        "response_text": None,
                        "model_used": "deterministic_fallback",
                        "is_fallback": True,
                        "error": err_msg
                    }

        except Exception as e:
            logger.warning(f"Failed to call OpenRouter API ({type(e).__name__}: {str(e)}). Using fallback.")
            return {
                "response_text": None,
                "model_used": "deterministic_fallback",
                "is_fallback": True,
                "error": str(e)
            }


def generate_deterministic_fallback_report(state: Dict[str, Any]) -> str:
    """
    Generate a clean, structured customer service report deterministically
    from evidence in state without calling an LLM.
    """
    extracted = state.get("extracted_fields") or {}
    product = extracted.get("Product", "Product")
    problem = extracted.get("Problem", "Inspection Required")
    severity = extracted.get("Severity", "Low")
    action = extracted.get("Requested_Action", "unspecified")

    nlp = state.get("nlp_results") or {}
    vision = state.get("vision_results") or {}
    fusion = state.get("fusion_results") or {}
    policy = state.get("policy_results") or {}
    uncertainties = state.get("uncertainty_notes") or []

    lines = [
        "## 📋 Customer Inspection & Assistance Report",
        "",
        f"**Product Category:** {product.capitalize()}",
        f"**Identified Issue:** {problem}",
        f"**Defect Severity:** {severity}",
        f"**Requested Action:** {action.capitalize()}",
        "",
        "### 🔍 Technical Inspection Summary:",
    ]

    if nlp.get("has_result"):
        is_comp = nlp.get("is_complaint", False)
        comp_str = "Complaint Detected" if is_comp else "No Complaint / Normal Inquiry"
        p_type = nlp.get("problem_type", "Other")
        p_conf = nlp.get("problem_type_confidence", 0.0)
        lines.append(f"- **Text Analysis:** {comp_str} (Problem: {p_type}, Confidence: {p_conf:.1%})")

    if vision.get("has_result"):
        has_def = vision.get("has_defect", False)
        loc = vision.get("location_summary", "None")
        v_conf = vision.get("segmentation_confidence", 0.0)
        def_str = "Defect Detected" if has_def else "No Defect Detected (Surface Normal)"
        lines.append(f"- **Visual Inspection:** {def_str} ({loc}, Confidence: {v_conf:.1%})")

    if fusion.get("has_result"):
        f_type = fusion.get("problem_type", "Other")
        f_conf = fusion.get("problem_confidence", 0.0)
        f_mod = fusion.get("modality_used", "multimodal")
        lines.append(f"- **Multimodal Fusion Decision:** {f_type} (Confidence: {f_conf:.1%}, Mode: {f_mod})")

    # Policy section
    if policy.get("policy_found"):
        window = policy.get("return_window_days", "N/A")
        warranty = policy.get("warranty_summary", "N/A")
        lines.extend([
            "",
            "### 📜 Policy & Resolution:",
            f"- **Return Window:** {window} days from delivery",
            f"- **Warranty Terms:** {warranty}",
            f"- **Recommendation:** Eligible for standard processing based on local policy.",
        ])
    elif action in ["return", "replacement"]:
        lines.extend([
            "",
            "### 📜 Policy & Resolution:",
            "- **Policy Notice:** No specific policy record found in the local catalog. Case escalated to customer support representative.",
        ])

    if uncertainties:
        lines.extend([
            "",
            "### ⚠️ Notes & Observations:",
        ])
        for note in uncertainties:
            lines.append(f"- {note}")

    lines.extend([
        "",
        "---",
        "*Report generated by Multimodal AI Inspection & Assistance Agent.*"
    ])

    return "\n".join(lines)
