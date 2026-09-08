"""
CLI Demonstration of Multimodal AI Product Inspection & Assistance Agent.
Executes the full LangGraph pipeline across all 4 operational scenarios.
"""

import sys
import io
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.agent_graph import MultimodalInspectionAgent
from src.config import PAIRED_DATASET_CSV_PATH
import pandas as pd


def print_separator(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_demo():
    print_separator("INITIALIZING MULTIMODAL INSPECTION AGENT")
    agent = MultimodalInspectionAgent()
    print("[OK] Agent initialized with BiLSTM, U-Net, Fusion, and Policy modules.")

    # Load samples from paired dataset
    df = pd.read_csv(PAIRED_DATASET_CSV_PATH)
    defect_sample = df[df["is_complaint"]].iloc[0]
    good_sample = df[~df["is_complaint"]].iloc[0]

    defect_img_path = str(PROJECT_ROOT / defect_sample["image_path"])
    good_img_path = str(PROJECT_ROOT / good_sample["image_path"])

    # -------------------------------------------------------------
    # Scenario 1: Multimodal (Text Complaint + Inspection Image)
    # -------------------------------------------------------------
    print_separator("SCENARIO 1: MULTIMODAL INPUT (Arabic Text + Defect Image)")
    arabic_text = "الزجاجة مكسورة وفيها شرخ كبير، أطلب استبدال المنتج فورا"
    print(f"Input Text: {arabic_text}")
    print(f"Input Image: {defect_img_path}")

    res1 = agent.run(raw_text=arabic_text, image_path=defect_img_path)

    print("\n[Pipeline Findings]")
    print(f"  - Extracted Product:  {res1['extracted_fields'].get('Product')}")
    print(f"  - Extracted Problem:  {res1['extracted_fields'].get('Problem')}")
    print(f"  - Requested Action:   {res1['extracted_fields'].get('Requested_Action')}")
    print(f"  - Vision Defect:      {res1['vision_results'].get('has_defect')}")
    print(f"  - Vision Location:    {res1['vision_results'].get('location_summary')}")
    print(f"  - Policy Found:       {res1['policy_results'].get('policy_found')}")
    if res1['policy_results'].get('policy_found'):
        print(f"  - Return Window:      {res1['policy_results'].get('return_window_days')} days")
        print(f"  - Warranty Summary:   {res1['policy_results'].get('warranty_summary')}")

    print("\n[Generated Customer Report]")
    print(res1["llm_response"])

    # -------------------------------------------------------------
    # Scenario 2: Image-Only Inspection (No Text)
    # -------------------------------------------------------------
    print_separator("SCENARIO 2: IMAGE-ONLY INPUT (No Customer Text)")
    print(f"Input Image: {defect_img_path}")

    res2 = agent.run(raw_text=None, image_path=defect_img_path)
    print("\n[Pipeline Findings]")
    print(f"  - Text Provided:      {res2['has_text']}")
    print(f"  - Image Provided:     {res2['has_image']}")
    print(f"  - Vision Defect:      {res2['vision_results'].get('has_defect')}")
    print(f"  - Uncertainty Notes:  {res2.get('uncertainty_notes')}")

    print("\n[Generated Customer Report]")
    print(res2["llm_response"])

    # -------------------------------------------------------------
    # Scenario 3: Text-Only Complaint (No Image)
    # -------------------------------------------------------------
    print_separator("SCENARIO 3: TEXT-ONLY INPUT (English Complaint, No Image)")
    english_text = "The bottle arrived completely shattered and contaminated. I need a refund or replacement."
    print(f"Input Text: {english_text}")

    res3 = agent.run(raw_text=english_text, image_path=None)
    print("\n[Pipeline Findings]")
    print(f"  - Extracted Action:   {res3['extracted_fields'].get('Requested_Action')}")
    print(f"  - Policy Checked:     {res3['policy_lookup_required']}")
    print(f"  - Policy Window:      {res3['policy_results'].get('return_window_days')} days")

    print("\n[Generated Customer Report]")
    print(res3["llm_response"])

    # -------------------------------------------------------------
    # Scenario 4: Normal / Good Product Inquiry
    # -------------------------------------------------------------
    print_separator("SCENARIO 4: NORMAL INQUIRY (Good Bottle)")
    normal_text = "The product arrived in great shape and perfect quality. Thank you!"
    print(f"Input Text: {normal_text}")
    print(f"Input Image: {good_img_path}")

    res4 = agent.run(raw_text=normal_text, image_path=good_img_path)
    print("\n[Pipeline Findings]")
    print(f"  - Complaint Detected: {res4['nlp_results'].get('is_complaint')}")
    print(f"  - Action:             {res4['extracted_fields'].get('Requested_Action')}")
    print(f"  - Policy Lookup:      {res4['policy_lookup_required']}")

    print("\n[Generated Customer Report]")
    print(res4["llm_response"])

    print_separator("DEMONSTRATION COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    run_demo()
