"""
Streamlit Demo UI for Multimodal Product Inspection & Assistance Agent.
Provides interactive interface for multimodal complaint analysis:
- Optional text complaint (Arabic / English)
- Optional image upload (MVTec bottle defect inspection)
- Visual segmentation mask & overlay
- Multimodal fusion confidence & decision
- Rule-based structured extraction
- Local policy enforcement
- AI Agent report (OpenRouter LLM or structured deterministic fallback)
Follows ADR Phase 8 specifications.
"""

import sys
import gc
import tempfile
import os
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    NLP_MODEL_PATH,
    VISION_MODEL_PATH,
    FUSION_MODEL_PATH,
)

# Set page config
st.set_page_config(
    page_title="Multimodal Inspection Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


def get_agent():
    """Load or retrieve cached agent instance."""
    if "agent" not in st.session_state:
        from src.agent.agent_graph import MultimodalInspectionAgent
        st.session_state["agent"] = MultimodalInspectionAgent()
    return st.session_state["agent"]


def create_overlay(image_pil: Image.Image, binary_mask: np.ndarray) -> Image.Image:
    """Create a semi-transparent red defect overlay on top of original image."""
    img_np = np.array(image_pil)
    h, w = img_np.shape[:2]
    mask_resized = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = img_np.copy()
    # Red highlight for defect regions
    overlay[mask_resized > 0] = [255, 50, 50]

    blended = cv2.addWeighted(img_np, 0.65, overlay, 0.35, 0)
    return Image.fromarray(blended)


def main():
    st.title("🔍 Multimodal AI Product Inspection & Assistance Agent")
    st.caption("Industrial Quality Inspection & Autonomous Customer Assistance System")

    # Sidebar Status & System Info
    with st.sidebar:
        st.header("🛠️ System Diagnostics")
        st.write(f"**NLP BiLSTM:** {'✅ Available' if NLP_MODEL_PATH.exists() else '⚠️ Untrained'}")
        st.write(f"**U-Net Vision:** {'✅ Available' if VISION_MODEL_PATH.exists() else '⚠️ Untrained'}")
        st.write(f"**Fusion Head:** {'✅ Available' if FUSION_MODEL_PATH.exists() else '⚠️ Untrained'}")

        st.markdown("---")
        st.subheader("💡 Quick Examples")
        if st.button("Example 1: Broken Bottle (Arabic)"):
            st.session_state["example_text"] = "الزجاجة مكسورة بشكل كبير، أريد استبدالها فورا"
        if st.button("Example 2: Contamination (English)"):
            st.session_state["example_text"] = "The product has visible dirt contamination inside. Please refund."
        if st.button("Example 3: Normal / Good (English)"):
            st.session_state["example_text"] = "The product looks great and is in good condition."

    # Input Section
    st.header("📥 Customer Submission")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Complaint Description (Text)")
        default_val = st.session_state.get("example_text", "")
        text_input = st.text_area(
            "Enter customer feedback or inquiry (Arabic or English):",
            value=default_val,
            placeholder="e.g. The bottle is cracked and leaking, I request a replacement.",
            height=160
        )

    with col2:
        st.subheader("🖼️ Product Image (Visual Inspection)")
        uploaded_file = st.file_uploader(
            "Upload bottle photo:",
            type=["png", "jpg", "jpeg"],
            help="Upload an industrial or customer product photo"
        )
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Product Image", use_container_width=True)

    st.markdown("")
    analyze_btn = st.button("🚀 Run Multimodal Inspection", type="primary", use_container_width=True)

    if analyze_btn:
        has_text = bool(text_input and text_input.strip())
        has_image = uploaded_file is not None

        if not has_text and not has_image:
            st.error("⚠️ **Validation Error:** Please provide at least text input or upload an image.")
            return

        temp_img_path = None
        if has_image:
            # Save uploaded image to temp file for inference
            suffix = Path(uploaded_file.name).suffix or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                temp_img_path = tmp.name

        with st.spinner("Processing multimodal inspection workflow..."):
            agent = get_agent()
            try:
                state = agent.run(
                    raw_text=text_input if has_text else None,
                    image_path=temp_img_path
                )
            finally:
                # Cleanup temp file
                if temp_img_path and os.path.exists(temp_img_path):
                    try:
                        os.remove(temp_img_path)
                    except Exception:
                        pass
                # Release memory after inference
                gc.collect()

        # Display Results
        st.markdown("---")
        st.header("📊 Multimodal Inspection & Diagnostics")

        # Top Metric Tiles
        extracted = state.get("extracted_fields") or {}

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Identified Product", extracted.get("Product", "Unknown").capitalize())
        with kpi2:
            st.metric("Primary Issue", extracted.get("Problem", "Normal"))
        with kpi3:
            st.metric("Defect Severity", extracted.get("Severity", "Low"))
        with kpi4:
            st.metric("Requested Action", extracted.get("Requested_Action", "unspecified").capitalize())

        # Vision Detailed Segmentation
        v_res = state.get("vision_results") or {}
        if state.get("has_image") and v_res.get("has_result"):
            st.subheader("🔬 Visual Segmentation & Defect Localization")

            # Use mask data stored in agent state (avoid re-running inference)
            binary_mask = v_res.get("binary_mask")

            if binary_mask is not None and uploaded_file is not None:
                v_col1, v_col2, v_col3 = st.columns(3)
                uploaded_file.seek(0)
                pil_img = Image.open(uploaded_file).convert("RGB")

                with v_col1:
                    st.image(pil_img, caption="Original Input Image", use_container_width=True)
                with v_col2:
                    st.image(binary_mask * 255, caption="U-Net Predicted Defect Mask", use_container_width=True)
                with v_col3:
                    overlay_img = create_overlay(pil_img, binary_mask)
                    st.image(overlay_img, caption="Defect Localization Overlay", use_container_width=True)

            st.caption(
                f"**Visual Findings:** Defect Detected: `{v_res.get('has_defect')}` | "
                f"Location: `{v_res.get('location_summary')}` | "
                f"Confidence: `{v_res.get('segmentation_confidence', 0):.1%}`"
            )

        # Policy & Resolution Details
        policy = state.get("policy_results") or {}
        if policy.get("policy_found"):
            st.subheader("📜 Local Policy Verification")
            st.success(
                f"**Return Window:** {policy.get('return_window_days')} days | "
                f"**Warranty:** {policy.get('warranty_summary')}"
            )

        # Final AI Report Output
        st.subheader("📄 Final Customer Assistance Report")
        report_text = state.get("llm_response", "No report generated.")
        st.markdown(report_text)


if __name__ == "__main__":
    main()
