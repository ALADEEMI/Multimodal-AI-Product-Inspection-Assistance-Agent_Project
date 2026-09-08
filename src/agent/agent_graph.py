"""
LangGraph Workflow Definition for Multimodal Inspection Agent.
Follows ADR Phase 7 specifications:
  START
  -> receive_input
  -> route_modalities
  -> text_analyzer (if text exists)
  -> image_analyzer (if image exists)
  -> information_extraction
  -> multimodal_fusion
  -> context_builder
  -> conditional_policy_lookup (if return/replacement)
  -> llm_response_node
  -> END
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Literal
from pathlib import Path

from langgraph.graph import StateGraph, START, END

from src.agent.agent_state import InspectionState
from src.agent.llm_adapter import LLMAdapter, generate_deterministic_fallback_report
from src.models.nlp.inference_nlp import NLPInference
from src.models.vision.inference_vision import VisionInference
from src.models.fusion.inference_fusion import FusionInference
from src.extraction.information_extraction import extract_information
from src.tools.policy_lookup import lookup_policy
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


class MultimodalInspectionAgent:
    """Orchestrates the multimodal inspection pipeline using LangGraph."""

    def __init__(self):
        # Lazy-loaded inference engines
        self._nlp_engine = None
        self._vision_engine = None
        self._fusion_engine = None
        self._llm_adapter = None

        # Build and compile graph
        self.graph = self._build_graph()

    @property
    def nlp_engine(self) -> NLPInference:
        if self._nlp_engine is None:
            self._nlp_engine = NLPInference()
        return self._nlp_engine

    @property
    def vision_engine(self) -> VisionInference:
        if self._vision_engine is None:
            self._vision_engine = VisionInference()
        return self._vision_engine

    @property
    def fusion_engine(self) -> FusionInference:
        if self._fusion_engine is None:
            self._fusion_engine = FusionInference()
        return self._fusion_engine

    @property
    def llm_adapter(self) -> LLMAdapter:
        if self._llm_adapter is None:
            self._llm_adapter = LLMAdapter()
        return self._llm_adapter

    # --- Node Implementations ---

    def receive_input(self, state: InspectionState) -> InspectionState:
        """Initialize state, trace ID, and timestamp."""
        state["trace_id"] = state.get("trace_id") or str(uuid.uuid4())[:8]
        state["timestamp"] = datetime.now().isoformat()
        state["errors"] = state.get("errors") or []
        state["uncertainty_notes"] = state.get("uncertainty_notes") or []

        raw_text = state.get("raw_text")
        image_path = state.get("image_path")

        has_text = bool(raw_text and str(raw_text).strip())
        has_image = bool(image_path and Path(str(image_path)).exists())

        state["has_text"] = has_text
        state["has_image"] = has_image
        state["is_valid_input"] = has_text or has_image

        if not state["is_valid_input"]:
            state["errors"].append("Invalid input: Neither text complaint nor valid product image was provided.")

        return state

    def text_analyzer(self, state: InspectionState) -> InspectionState:
        """Run BiLSTM text analysis if text is present."""
        if not state.get("has_text"):
            state["uncertainty_notes"].append("Text modality missing: Analysis relying on visual inspection.")
            state["nlp_results"] = {"has_result": False}
            return state

        try:
            raw_text = state["raw_text"]
            res = self.nlp_engine.predict(raw_text)
            res["has_result"] = True
            state["nlp_results"] = res
        except Exception as e:
            logger.error(f"Error in text_analyzer: {e}")
            state["errors"].append(f"Text analysis failed: {str(e)}")
            state["nlp_results"] = {"has_result": False, "error": str(e)}

        return state

    def image_analyzer(self, state: InspectionState) -> InspectionState:
        """Run U-Net vision segmentation if image is present."""
        if not state.get("has_image"):
            state["uncertainty_notes"].append("Image modality missing: Analysis relying on customer text description.")
            state["vision_results"] = {"has_result": False}
            return state

        try:
            image_path = state["image_path"]
            res = self.vision_engine.predict(image_path)
            res["has_result"] = True
            # Keep embeddings in memory, exclude heavy mask arrays from raw serialization
            state["vision_results"] = {
                "has_result": True,
                "has_defect": res["has_defect"],
                "segmentation_confidence": res["segmentation_confidence"],
                "defect_area_ratio": res["defect_area_ratio"],
                "location_summary": res["location_summary"],
                "image_embedding": res["image_embedding"],
            }
        except Exception as e:
            logger.error(f"Error in image_analyzer: {e}")
            state["errors"].append(f"Vision analysis failed: {str(e)}")
            state["vision_results"] = {"has_result": False, "error": str(e)}

        return state

    def information_extraction_node(self, state: InspectionState) -> InspectionState:
        """Rule-based extraction of Product, Problem, Severity, Requested_Action."""
        nlp_res = state.get("nlp_results") or {}
        vision_res = state.get("vision_results") or {}
        raw_text = state.get("raw_text")

        extracted = extract_information(raw_text, nlp_res, vision_res)
        state["extracted_fields"] = extracted

        # Check if policy lookup is needed (only for return or replacement)
        action = extracted.get("Requested_Action", "unspecified")
        state["policy_lookup_required"] = action in ["return", "replacement"]

        return state

    def multimodal_fusion_node(self, state: InspectionState) -> InspectionState:
        """Combine text and vision embeddings using baseline fusion model."""
        nlp_res = state.get("nlp_results") or {}
        vision_res = state.get("vision_results") or {}

        text_emb = nlp_res.get("text_embedding") if nlp_res.get("has_result") else None
        image_emb = vision_res.get("image_embedding") if vision_res.get("has_result") else None

        try:
            fusion_res = self.fusion_engine.predict(text_emb=text_emb, image_emb=image_emb)
            fusion_res["has_result"] = True
            state["fusion_results"] = fusion_res
        except Exception as e:
            logger.error(f"Error in multimodal_fusion_node: {e}")
            state["errors"].append(f"Fusion failed: {str(e)}")
            state["fusion_results"] = {"has_result": False, "error": str(e)}

        return state

    def context_builder(self, state: InspectionState) -> InspectionState:
        """Build structured evidence context for LLM prompt."""
        extracted = state.get("extracted_fields") or {}
        nlp = state.get("nlp_results") or {}
        vision = state.get("vision_results") or {}
        fusion = state.get("fusion_results") or {}

        evidence = [
            f"Product: {extracted.get('Product', 'Unknown')}",
            f"Primary Problem: {extracted.get('Problem', 'Unknown')}",
            f"Severity: {extracted.get('Severity', 'Unknown')}",
            f"Requested Action: {extracted.get('Requested_Action', 'unspecified')}",
            f"Modality Availability: Text={'Yes' if state.get('has_text') else 'No'}, Image={'Yes' if state.get('has_image') else 'No'}",
        ]

        if nlp.get("has_result"):
            evidence.append(
                f"NLP Finding: is_complaint={nlp.get('is_complaint')}, "
                f"problem={nlp.get('problem_type')}, confidence={nlp.get('problem_type_confidence', 0):.2%}"
            )

        if vision.get("has_result"):
            evidence.append(
                f"Vision Finding: has_defect={vision.get('has_defect')}, "
                f"location={vision.get('location_summary')}, confidence={vision.get('segmentation_confidence', 0):.2%}"
            )

        if fusion.get("has_result"):
            evidence.append(
                f"Fusion Decision: problem={fusion.get('problem_type')}, "
                f"confidence={fusion.get('problem_confidence', 0):.2%}"
            )

        state["evidence_context"] = "\n".join(evidence)
        return state

    def conditional_policy_lookup(self, state: InspectionState) -> InspectionState:
        """Look up local return/warranty policy from data/policies.csv if required."""
        if not state.get("policy_lookup_required"):
            state["policy_results"] = {"policy_found": False, "reason": "No return/replacement requested"}
            return state

        extracted = state.get("extracted_fields") or {}
        product_cat = extracted.get("Product", "bottle")

        policy_res = lookup_policy(product_cat)
        state["policy_results"] = policy_res
        return state

    def llm_response_node(self, state: InspectionState) -> InspectionState:
        """Generate final customer assistance report using OpenRouter LLM or deterministic fallback."""
        if not state.get("is_valid_input"):
            state["llm_response"] = (
                "⚠️ **Invalid Request:** Please provide at least a customer text complaint or upload a product inspection image."
            )
            return state

        # Build prompt strictly from structured evidence
        system_prompt = (
            "You are an AI Product Inspection & Customer Assistance Agent.\n"
            "Format a helpful, polite, structured inspection report and customer response.\n"
            "CRITICAL RULES:\n"
            "1. Rely ONLY on the provided structured inspection evidence.\n"
            "2. DO NOT hallucinate defects, warranties, or facts outside the context.\n"
            "3. If confidence is low or a modality is missing, clearly state that uncertainty.\n"
            "4. Include sections: Inspection Summary, Technical Findings, and Next Steps / Policy Resolution."
        )

        user_prompt = f"Inspection Evidence:\n{state.get('evidence_context', '')}\n\n"
        if state.get("policy_results", {}).get("policy_found"):
            p = state["policy_results"]
            user_prompt += f"Policy Details:\nReturn Window: {p.get('return_window_days')} days\nWarranty: {p.get('warranty_summary')}\n\n"

        if state.get("uncertainty_notes"):
            user_prompt += "Uncertainty / Notes:\n" + "\n".join(f"- {n}" for n in state["uncertainty_notes"]) + "\n\n"

        user_prompt += "Please generate the customer assistance report."

        # Call adapter
        adapter_output = self.llm_adapter.generate_response(user_prompt, system_prompt=system_prompt)

        if not adapter_output["is_fallback"] and adapter_output["response_text"]:
            state["llm_response"] = adapter_output["response_text"]
        else:
            # Deterministic Fallback Report
            state["llm_response"] = generate_deterministic_fallback_report(state)

        return state

    # --- Graph Assembly ---

    def _build_graph(self) -> StateGraph:
        """Construct the LangGraph workflow StateGraph."""
        builder = StateGraph(InspectionState)

        # Add Nodes
        builder.add_node("receive_input", self.receive_input)
        builder.add_node("text_analyzer", self.text_analyzer)
        builder.add_node("image_analyzer", self.image_analyzer)
        builder.add_node("information_extraction", self.information_extraction_node)
        builder.add_node("multimodal_fusion", self.multimodal_fusion_node)
        builder.add_node("context_builder", self.context_builder)
        builder.add_node("conditional_policy_lookup", self.conditional_policy_lookup)
        builder.add_node("llm_response_node", self.llm_response_node)

        # Edges
        builder.add_edge(START, "receive_input")
        builder.add_edge("receive_input", "text_analyzer")
        builder.add_edge("text_analyzer", "image_analyzer")
        builder.add_edge("image_analyzer", "information_extraction")
        builder.add_edge("information_extraction", "multimodal_fusion")
        builder.add_edge("multimodal_fusion", "context_builder")
        builder.add_edge("context_builder", "conditional_policy_lookup")
        builder.add_edge("conditional_policy_lookup", "llm_response_node")
        builder.add_edge("llm_response_node", END)

        return builder.compile()

    def run(
        self,
        raw_text: Optional[str] = None,
        image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute full end-to-end multimodal inspection graph.

        Args:
            raw_text: Customer complaint text (optional)
            image_path: Path to product image (optional)

        Returns:
            Final state dictionary containing report, predictions, and extracted fields.
        """
        initial_state: InspectionState = {
            "raw_text": raw_text,
            "image_path": image_path,
        }

        final_state = self.graph.invoke(initial_state)
        return final_state
