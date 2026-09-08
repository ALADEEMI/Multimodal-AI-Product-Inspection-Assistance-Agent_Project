"""NLP Metrics Calculation and Reporting for Binary and Multi-Class Evaluation."""

from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
    classification_report
)


def compute_binary_metrics(
    y_true: List[int], y_pred: List[int]
) -> Dict[str, float]:
    """Compute binary classification metrics for is_complaint."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


def compute_multiclass_metrics(
    y_true: List[int],
    y_pred: List[int],
    target_names: List[str]
) -> Dict[str, Any]:
    """Compute multi-class metrics for problem_type."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(target_names))), zero_division=0
    )

    per_class = {}
    for idx, name in enumerate(target_names):
        per_class[name] = {
            "precision": float(p[idx]),
            "recall": float(r[idx]),
            "f1": float(f[idx]),
            "support": int(support[idx])
        }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(target_names))))

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist()
    }
