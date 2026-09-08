"""Local Policy Lookup Tool - Reads Return/Warranty Policies from CSV.

Only invoked for return/replacement actions. No external API calls.
Follows ADR Phase 5 specifications.
"""

import csv
from pathlib import Path
from typing import Dict, Any, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import POLICIES_CSV_PATH


def validate_policies_csv(csv_path: Path) -> bool:
    """Validate policies CSV structure."""
    if not csv_path.exists():
        return False

    required_fields = ["product_category", "return_window_days", "warranty_summary"]

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            return all(field in fieldnames for field in required_fields)
    except Exception:
        return False


def lookup_policy(product_category: str, csv_path: Path = POLICIES_CSV_PATH) -> Dict[str, Any]:
    """Look up return/warranty policy for product category.

    Args:
        product_category: Product category (e.g., "bottle")
        csv_path: Path to policies CSV

    Returns:
        Structured dictionary with policy details or error reason
    """
    if not csv_path.exists():
        return {
            "policy_found": False,
            "product_category": product_category,
            "reason": f"Policies database file not found at {csv_path}"
        }

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("product_category", "").lower() == product_category.lower():
                    return {
                        "policy_found": True,
                        "product_category": row["product_category"],
                        "return_window_days": int(row["return_window_days"]),
                        "warranty_summary": row["warranty_summary"]
                    }
    except Exception as e:
        return {
            "policy_found": False,
            "product_category": product_category,
            "reason": f"Error reading policies: {e}"
        }

    return {
        "policy_found": False,
        "product_category": product_category,
        "reason": f"No policy found for product category '{product_category}'"
    }


def policy_lookup(product_category: str, csv_path: Path = POLICIES_CSV_PATH) -> Optional[Dict[str, Any]]:
    """Legacy lookup function returning dict or None."""
    res = lookup_policy(product_category, csv_path)
    if res.get("policy_found"):
        return res
    return None

