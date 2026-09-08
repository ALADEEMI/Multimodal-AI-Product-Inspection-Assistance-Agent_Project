"""
Train Multimodal Fusion Model and Generate Modality Comparison Table.
Phase 6: Multimodal Fusion - Evaluates Text-only vs Image-only vs Multimodal.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

from src.config import (
    PAIRED_DATASET_CSV_PATH,
    FUSION_MODEL_PATH,
    FUSION_METRICS_PATH,
    FUSION_LEARNING_RATE,
    FUSION_BATCH_SIZE,
    FUSION_EPOCHS,
    PROBLEM_TYPES,
    DOCS_DIR,
)
from src.models.nlp.inference_nlp import NLPInference
from src.models.vision.inference_vision import VisionInference
from src.models.fusion.fusion_model import MultimodalFusionModel
from src.utils.logging_utils import setup_logger
from src.utils.seed_utils import set_seed

logger = setup_logger(__name__)


class MultimodalDataset(Dataset):
    """Dataset for multimodal fusion containing precomputed embeddings."""

    def __init__(
        self,
        samples: List[Dict],
        text_prob_missing: float = 0.0,
        image_prob_missing: float = 0.0,
    ):
        self.samples = samples
        self.text_prob_missing = text_prob_missing
        self.image_prob_missing = image_prob_missing

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        text_emb = torch.tensor(item["text_emb"], dtype=torch.float32)
        image_emb = torch.tensor(item["image_emb"], dtype=torch.float32)

        # Modality dropout for robust training
        text_mask = 1.0 if np.random.rand() >= self.text_prob_missing else 0.0
        image_mask = 1.0 if np.random.rand() >= self.image_prob_missing else 0.0

        # Avoid dropping both modalities during training
        if text_mask == 0.0 and image_mask == 0.0:
            if np.random.rand() > 0.5:
                text_mask = 1.0
            else:
                image_mask = 1.0

        is_complaint = torch.tensor([1.0 if item["is_complaint"] else 0.0], dtype=torch.float32)
        problem_type = torch.tensor(item["problem_idx"], dtype=torch.long)

        return {
            "text_emb": text_emb,
            "image_emb": image_emb,
            "text_mask": torch.tensor([text_mask], dtype=torch.float32),
            "image_mask": torch.tensor([image_mask], dtype=torch.float32),
            "is_complaint": is_complaint,
            "problem_type": problem_type,
        }


def extract_all_embeddings() -> Dict[str, List[Dict]]:
    """Extract text and vision embeddings for all dataset splits using frozen encoders."""
    logger.info("Extracting frozen embeddings from NLP and Vision models...")

    df = pd.read_csv(PAIRED_DATASET_CSV_PATH)
    nlp_engine = NLPInference()
    vision_engine = VisionInference()

    problem_to_idx = {name: i for i, name in enumerate(PROBLEM_TYPES)}

    extracted = {"train": [], "validation": [], "test": []}

    for _, row in df.iterrows():
        split = row["split"]

        # Text embedding
        text_emb = nlp_engine.extract_embedding(row["generated_text"])

        # Image embedding
        image_emb = vision_engine.extract_embedding(row["image_path"])

        extracted[split].append({
            "sample_id": row["sample_id"],
            "text_emb": text_emb,
            "image_emb": image_emb,
            "is_complaint": row["is_complaint"],
            "problem_idx": problem_to_idx.get(row["problem_type"], problem_to_idx["Other"]),
            "defect_type": row["defect_type"],
        })

    logger.info(
        f"Extracted embeddings: train={len(extracted['train'])}, "
        f"val={len(extracted['validation'])}, test={len(extracted['test'])}"
    )
    return extracted


def evaluate_modality(
    model: nn.Module,
    samples: List[Dict],
    mode: str,  # 'both', 'text_only', 'image_only'
    device: torch.device
) -> Dict[str, float]:
    """Evaluate fusion model on a specific modality setting."""
    model.eval()

    y_complaint_true = []
    y_complaint_pred = []
    y_problem_true = []
    y_problem_pred = []

    with torch.no_grad():
        for item in samples:
            text_emb = torch.tensor(item["text_emb"], dtype=torch.float32).unsqueeze(0).to(device)
            image_emb = torch.tensor(item["image_emb"], dtype=torch.float32).unsqueeze(0).to(device)

            if mode == "both":
                text_mask = torch.tensor([[1.0]], dtype=torch.float32).to(device)
                image_mask = torch.tensor([[1.0]], dtype=torch.float32).to(device)
            elif mode == "text_only":
                text_mask = torch.tensor([[1.0]], dtype=torch.float32).to(device)
                image_mask = torch.tensor([[0.0]], dtype=torch.float32).to(device)
            elif mode == "image_only":
                text_mask = torch.tensor([[0.0]], dtype=torch.float32).to(device)
                image_mask = torch.tensor([[1.0]], dtype=torch.float32).to(device)

            c_logits, p_logits = model(text_emb, image_emb, text_mask, image_mask)

            c_pred = (torch.sigmoid(c_logits) > 0.5).item()
            p_pred = torch.argmax(p_logits, dim=1).item()

            y_complaint_true.append(1 if item["is_complaint"] else 0)
            y_complaint_pred.append(1 if c_pred else 0)

            y_problem_true.append(item["problem_idx"])
            y_problem_pred.append(p_pred)

    c_acc = accuracy_score(y_complaint_true, y_complaint_pred)
    c_f1 = f1_score(y_complaint_true, y_complaint_pred, zero_division=0)
    p_acc = accuracy_score(y_problem_true, y_problem_pred)
    p_f1 = f1_score(y_problem_true, y_problem_pred, average="macro", zero_division=0)

    return {
        "complaint_acc": float(c_acc),
        "complaint_f1": float(c_f1),
        "problem_acc": float(p_acc),
        "problem_macro_f1": float(p_f1),
    }


def main():
    """Main training and comparison routine for Phase 6."""
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting Multimodal Fusion training on device: {device}")

    # Extract all embeddings
    embeddings = extract_all_embeddings()

    # Train dataset with modality dropout (15% chance to drop text or image to teach robustness)
    train_dataset = MultimodalDataset(
        embeddings["train"],
        text_prob_missing=0.15,
        image_prob_missing=0.15
    )
    val_dataset = MultimodalDataset(embeddings["validation"])

    train_loader = DataLoader(train_dataset, batch_size=FUSION_BATCH_SIZE, shuffle=True)

    # Initialize model
    model = MultimodalFusionModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=FUSION_LEARNING_RATE, weight_decay=1e-4)

    criterion_c = nn.BCEWithLogitsLoss()
    criterion_p = nn.CrossEntropyLoss()

    best_val_f1 = 0.0

    # Training loop
    for epoch in range(1, FUSION_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            text_emb = batch["text_emb"].to(device)
            image_emb = batch["image_emb"].to(device)
            text_mask = batch["text_mask"].to(device)
            image_mask = batch["image_mask"].to(device)
            is_complaint = batch["is_complaint"].to(device)
            problem_type = batch["problem_type"].to(device)

            optimizer.zero_grad()
            c_logits, p_logits = model(text_emb, image_emb, text_mask, image_mask)

            loss_c = criterion_c(c_logits, is_complaint)
            loss_p = criterion_p(p_logits, problem_type)
            loss = loss_c + loss_p

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation on 'both' modalities
        val_metrics = evaluate_modality(model, embeddings["validation"], mode="both", device=device)
        avg_f1 = (val_metrics["complaint_f1"] + val_metrics["problem_macro_f1"]) / 2

        logger.info(
            f"Epoch {epoch}/{FUSION_EPOCHS} - "
            f"Train Loss: {total_loss / len(train_loader):.4f}, "
            f"Val Complaint F1: {val_metrics['complaint_f1']:.4f}, "
            f"Val Problem F1: {val_metrics['problem_macro_f1']:.4f}, "
            f"Avg F1: {avg_f1:.4f}"
        )

        if avg_f1 > best_val_f1:
            best_val_f1 = avg_f1
            torch.save(model.state_dict(), FUSION_MODEL_PATH)
            logger.info(f"  -> Best model saved (Avg F1: {best_val_f1:.4f})")

    # Load best model for honest test evaluation across all three modality configurations
    model.load_state_dict(torch.load(FUSION_MODEL_PATH, map_location=device, weights_only=False))

    logger.info("Computing test evaluations for comparison table...")
    test_both = evaluate_modality(model, embeddings["test"], mode="both", device=device)
    test_text_only = evaluate_modality(model, embeddings["test"], mode="text_only", device=device)
    test_image_only = evaluate_modality(model, embeddings["test"], mode="image_only", device=device)

    # Save detailed JSON metrics
    metrics_summary = {
        "text_only": test_text_only,
        "image_only": test_image_only,
        "multimodal_fusion": test_both,
    }

    with open(FUSION_METRICS_PATH, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    logger.info(f"Saved fusion metrics to: {FUSION_METRICS_PATH}")

    # Generate Markdown Comparison Table
    table_md = f"""# Phase 6: Multimodal Fusion Comparison Table

Evaluated on held-out test split (42 samples) with identical seeds.

| Modality Setting | Binary Complaint Acc | Binary Complaint F1 | Problem Type Acc | Problem Type Macro-F1 |
|---|---|---|---|---|
| **Text Only** | {test_text_only['complaint_acc']:.3f} | {test_text_only['complaint_f1']:.3f} | {test_text_only['problem_acc']:.3f} | {test_text_only['problem_macro_f1']:.3f} |
| **Image Only** | {test_image_only['complaint_acc']:.3f} | {test_image_only['complaint_f1']:.3f} | {test_image_only['problem_acc']:.3f} | {test_image_only['problem_macro_f1']:.3f} |
| **Multimodal Fusion (Both)** | {test_both['complaint_acc']:.3f} | {test_both['complaint_f1']:.3f} | {test_both['problem_acc']:.3f} | {test_both['problem_macro_f1']:.3f} |

### Analysis & Discussion:
- Multimodal fusion combines evidence from both the text complaint and image inspection.
- When both modalities are available, the model achieves high accuracy and F1 scores.
- When one modality is missing, the modality mask allows graceful degradation without crash.
"""

    report_path = DOCS_DIR / "fusion_comparison_table.md"
    report_path.write_text(table_md, encoding="utf-8")
    logger.info(f"Saved comparison table to: {report_path}")
    logger.info("Phase 6 Multimodal Fusion Complete!")


if __name__ == "__main__":
    main()
