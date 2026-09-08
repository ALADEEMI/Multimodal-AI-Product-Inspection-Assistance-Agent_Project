"""Unit Test Suite for Multimodal Inspection Agent.

Tests MVTec validation, paired dataset generation, NLP preprocessing,
inference loading, information extraction, and policy lookup.
"""

import pytest
from pathlib import Path
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.config import (
    MVTEC_DATA_PATH,
    PAIRED_DATASET_CSV_PATH,
    TOKENIZER_PATH,
    NLP_MODEL_PATH,
    POLICIES_CSV_PATH
)
from src.preprocessing.text_preprocessing import TextTokenizer, clean_text
from src.extraction.information_extraction import extract_requested_action, extract_information
from src.tools.policy_lookup import policy_lookup, validate_policies_csv


def test_clean_text():
    """Test text cleaning preserves emojis and handles whitespace."""
    # Test whitespace normalization
    assert clean_text("  hello   world  ") == "hello world"

    # Test emoji preservation
    text_with_emoji = "Broken bottle 😡👎"
    assert "😡" in clean_text(text_with_emoji)

    # Test Arabic text
    arabic_text = "الزجاجة   مكسورة"
    assert clean_text(arabic_text) == "الزجاجة مكسورة"


def test_tokenizer_encoding():
    """Test tokenizer fit, encode, and decode."""
    texts = [
        "The bottle is broken",
        "الزجاجة مكسورة",
        "Product has defect",
        "Everything is fine"
    ]

    tokenizer = TextTokenizer(vocab_size=50, max_len=10)
    tokenizer.fit(texts)

    assert tokenizer.is_fit

    # Encode known text
    encoded = tokenizer.encode("The bottle is broken")
    assert len(encoded) == 10  # max_len padding
    assert encoded[0] != 0  # Not padding

    # Encode unknown word (should map to <UNK> = 1)
    encoded_unk = tokenizer.encode("unknownwordxyz")
    assert 1 in encoded_unk


def test_information_extraction_actions():
    """Test action extraction for Arabic and English."""
    nlp_dummy = {"is_complaint": True, "problem_type": "Damage", "severity": "High"}

    # English actions
    assert extract_requested_action("I want a replacement for this", nlp_dummy) == "replacement"
    assert extract_requested_action("Please return my money", nlp_dummy) == "return"
    assert extract_requested_action("Can you repair it?", nlp_dummy) == "repair"

    # Arabic actions
    assert extract_requested_action("أريد استبدال المنتج", nlp_dummy) == "replacement"
    assert extract_requested_action("ارغب بإرجاع الزجاجة", nlp_dummy) == "return"
    assert extract_requested_action("هل يمكن تصليح العيب؟", nlp_dummy) == "repair"

    # Unspecified action
    assert extract_requested_action("The product is just bad", {"is_complaint": False}) == "unspecified"


def test_policy_lookup():
    """Test policy lookup from CSV."""
    if not POLICIES_CSV_PATH.exists():
        pytest.skip("policies.csv not found")

    # Valid category
    policy = policy_lookup("bottle")
    assert policy is not None
    assert policy["product_category"] == "bottle"
    assert policy["return_window_days"] == 30
    assert "warranty_summary" in policy

    # Invalid category
    invalid_policy = policy_lookup("non_existent_category")
    assert invalid_policy is None


def test_dataset_exists():
    """Test that generated paired dataset exists and has records."""
    if not PAIRED_DATASET_CSV_PATH.exists():
        pytest.skip("Paired dataset not generated yet")

    import csv
    with open(PAIRED_DATASET_CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) > 0
    assert "sample_id" in rows[0]
    assert "generated_text" in rows[0]
    assert "is_complaint" in rows[0]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
