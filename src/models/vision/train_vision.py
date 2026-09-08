"""
Train U-Net Mini Segmentation Model.
Phase 4: Deep Learning - Supervised segmentation with BCE+Dice loss.
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from src.config import (
    PAIRED_DATASET_CSV_PATH,
    IMAGE_SIZE,
    VISION_BATCH_SIZE,
    VISION_LEARNING_RATE,
    VISION_EPOCHS,
    VISION_MODEL_PATH,
    VISION_METRICS_PATH,
    CHECKPOINTS_DIR,
    PROJECT_ROOT,
    VISUALS_DIR,
)
from src.models.vision.unet_model import MiniUNet
from src.preprocessing.image_preprocessing import SegmentationDataset
from src.evaluation.vision_metrics import DiceBCELoss, compute_segmentation_metrics
from src.utils.logging_utils import setup_logger
from src.utils.seed_utils import set_seed

logger = setup_logger(__name__)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        # Forward
        mask_logits, _ = model(images)
        loss = criterion(mask_logits, masks)

        # Backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    all_ious = []
    all_dices = []
    all_pixel_accs = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            mask_logits, _ = model(images)
            loss = criterion(mask_logits, masks)
            total_loss += loss.item()

            # Compute metrics
            pred_masks = torch.sigmoid(mask_logits) > 0.5
            metrics = compute_segmentation_metrics(
                pred_masks.cpu().numpy(),
                masks.cpu().numpy()
            )
            all_ious.append(metrics["iou"])
            all_dices.append(metrics["dice"])
            all_pixel_accs.append(metrics["pixel_accuracy"])

    avg_metrics = {
        "loss": total_loss / len(dataloader),
        "iou": np.mean(all_ious),
        "dice": np.mean(all_dices),
        "pixel_accuracy": np.mean(all_pixel_accs)
    }

    return avg_metrics["loss"], avg_metrics


def main():
    """Main training function for U-Net segmentation."""
    set_seed()

    logger.info("Starting U-Net segmentation training (Phase 4)")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load datasets
    train_dataset = SegmentationDataset(
        csv_path=PAIRED_DATASET_CSV_PATH,
        split="train",
        augment=True
    )
    val_dataset = SegmentationDataset(
        csv_path=PAIRED_DATASET_CSV_PATH,
        split="validation",
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=VISION_BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=VISION_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Model
    model = MiniUNet(in_channels=3, out_channels=1).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=VISION_LEARNING_RATE)

    # Training loop with early stopping
    best_val_dice = 0.0
    patience = 10
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}

    for epoch in range(1, VISION_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_metrics["dice"])
        history["val_iou"].append(val_metrics["iou"])

        logger.info(
            f"Epoch {epoch}/{VISION_EPOCHS} - "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Val Dice: {val_metrics['dice']:.4f}, Val IoU: {val_metrics['iou']:.4f}"
        )

        # Save best model based on validation Dice
        if val_metrics["dice"] > best_val_dice or epoch == 1:
            best_val_dice = val_metrics["dice"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": best_val_dice,
                "history": history
            }, VISION_MODEL_PATH)
            logger.info(f"  -> Best model saved (Dice: {best_val_dice:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    logger.info(f"Training complete. Best Val Dice: {best_val_dice:.4f}")

    # Evaluate on test set (frozen, one-time only)
    logger.info("Evaluating on test set...")
    test_dataset = SegmentationDataset(
        csv_path=PAIRED_DATASET_CSV_PATH,
        split="test",
        augment=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=VISION_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Load best model
    checkpoint = torch.load(VISION_MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, test_metrics = validate_epoch(model, test_loader, criterion, device)

    logger.info(
        f"Test Metrics - Dice: {test_metrics['dice']:.4f}, "
        f"IoU: {test_metrics['iou']:.4f}, "
        f"Pixel Accuracy: {test_metrics['pixel_accuracy']:.4f}"
    )

    # Save metrics
    metrics_report = {
        "best_val_dice": best_val_dice,
        "test_dice": test_metrics["dice"],
        "test_iou": test_metrics["iou"],
        "test_pixel_accuracy": test_metrics["pixel_accuracy"],
        "history": history
    }

    with open(VISION_METRICS_PATH, "w") as f:
        json.dump(metrics_report, f, indent=2)

    logger.info(f"Metrics saved to: {VISION_METRICS_PATH}")
    logger.info("Phase 4 Vision Training Complete!")


if __name__ == "__main__":
    main()
