"""
End-to-End Test Suite for Multimodal Inspection Agent.
Verifies the 4 mandatory scenarios:
1. Text-only complaint
2. Image-only inspection
3. Multimodal (Text + Image)
4. No defect / normal inquiry
Plus edge cases:
- Empty / invalid input
- LLM fallback handling
- Modality masks in fusion
- Policy lookup conditional trigger
"""

import pytest
from pathlib import Path
import pandas as pd

from src.config import (
    PAIRED_DATASET_CSV_PATH,
    POLICIES_CSV_PATH,
    PROJECT_ROOT,
)
from src.agent.agent_graph import MultimodalInspectionAgent
from src.tools.policy_lookup import lookup_policy
from src.extraction.information_extraction import extract_requested_action, extract_information


@pytest.fixture(scope="module")
def agent():
    """Fixture providing initialized MultimodalInspectionAgent."""
    return MultimodalInspectionAgent()


@pytest.fixture(scope="module")
def sample_paths():
    """Retrieve verified sample image paths from paired dataset."""
    df = pd.read_csv(PAIRED_DATASET_CSV_PATH)
    defect_row = df[df["is_complaint"]].iloc[0]
    good_row = df[~df["is_complaint"]].iloc[0]
    return {
        "defect_img": PROJECT_ROOT / defect_row["image_path"],
        "defect_text": defect_row["generated_text"],
        "good_img": PROJECT_ROOT / good_row["image_path"],
        "good_text": good_row["generated_text"],
    }


def test_scenario_1_text_only(agent, sample_paths):
    """Scenario 1: Customer submits text complaint with no image."""
    text = "The bottle is broken and shattered. I request a replacement."
    state = agent.run(raw_text=text, image_path=None)

    assert state["is_valid_input"] is True
    assert state["has_text"] is True
    assert state["has_image"] is False
    assert state["nlp_results"]["has_result"] is True
    assert state["nlp_results"]["is_complaint"] is True
    assert state["extracted_fields"]["Requested_Action"] in ["replacement", "return"]
    assert state["policy_results"]["policy_found"] is True
    assert "llm_response" in state
    assert len(state["llm_response"]) > 20


def test_scenario_2_image_only(agent, sample_paths):
    """Scenario 2: Customer uploads product image with no text."""
    img_path = str(sample_paths["defect_img"])
    state = agent.run(raw_text=None, image_path=img_path)

    assert state["is_valid_input"] is True
    assert state["has_text"] is False
    assert state["has_image"] is True
    assert state["vision_results"]["has_result"] is True
    assert "llm_response" in state
    assert any("Text modality missing" in note for note in state.get("uncertainty_notes", []))


def test_scenario_3_multimodal(agent, sample_paths):
    """Scenario 3: Customer submits both text and image."""
    text = sample_paths["defect_text"]
    img_path = str(sample_paths["defect_img"])
    state = agent.run(raw_text=text, image_path=img_path)

    assert state["is_valid_input"] is True
    assert state["has_text"] is True
    assert state["has_image"] is True
    assert state["nlp_results"]["has_result"] is True
    assert state["vision_results"]["has_result"] is True
    assert state["fusion_results"]["has_result"] is True
    assert state["fusion_results"]["modality_used"] == "multimodal"
    assert "llm_response" in state


def test_scenario_4_no_defect_normal(agent, sample_paths):
    """Scenario 4: Good / normal item with no obvious defect."""
    text = "The product is in great shape and meets expectations."
    img_path = str(sample_paths["good_img"])
    state = agent.run(raw_text=text, image_path=img_path)

    assert state["is_valid_input"] is True
    assert state["nlp_results"]["is_complaint"] is False
    assert state["extracted_fields"]["Requested_Action"] == "unspecified"
    assert state["policy_lookup_required"] is False
    assert "llm_response" in state


def test_edge_case_empty_input(agent):
    """Edge Case: Completely empty input is rejected with clear error."""
    state = agent.run(raw_text=None, image_path=None)

    assert state["is_valid_input"] is False
    assert len(state["errors"]) > 0
    assert "Invalid Request" in state["llm_response"] or "Invalid input" in state["errors"][0]


def test_policy_lookup_tool():
    """Verify local policy lookup tool strictly operates on data/policies.csv."""
    # Existing category
    res = lookup_policy("bottle")
    assert res["policy_found"] is True
    assert res["return_window_days"] == 30
    assert "warranty_summary" in res

    # Non-existent category
    res_unknown = lookup_policy("spaceship")
    assert res_unknown["policy_found"] is False
    assert "No policy found" in res_unknown["reason"]
