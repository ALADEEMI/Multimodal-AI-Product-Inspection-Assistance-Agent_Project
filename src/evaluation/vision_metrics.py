"""Vision Loss Functions (BCE + Dice) and Segmentation Evaluation Metrics (IoU, Dice, Pixel Accuracy)."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple


class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice Loss for Segmentation."""

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1e-5):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute BCE + Dice loss.

        Args:
            logits: Predicted logits (B, 1, H, W)
            targets: Binary ground truth (B, 1, H, W)
        """
        # BCE Loss
        bce_loss = self.bce(logits, targets)

        # Dice Loss
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1.0 - dice

        total_loss = (self.bce_weight * bce_loss) + ((1.0 - self.bce_weight) * dice_loss)
        return total_loss


# Alias for compatibility
DiceBCELoss = BCEDiceLoss


def compute_segmentation_metrics(
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
    threshold: float = 0.5,
    smooth: float = 1e-5
) -> Dict[str, float]:
    """Calculate IoU, Dice coefficient, and Pixel Accuracy on binary masks.

    Args:
        pred_masks: Array of shape (N, H, W) with probabilities [0, 1]
        gt_masks: Array of shape (N, H, W) with binary {0, 1}
        threshold: Binarization threshold

    Returns:
        Dict with iou, dice, and pixel_accuracy
    """
    bin_preds = (pred_masks > threshold).astype(np.float32)
    bin_gts = (gt_masks > 0.5).astype(np.float32)

    intersection = np.sum(bin_preds * bin_gts)
    union = np.sum(bin_preds) + np.sum(bin_gts) - intersection

    # IoU (Jaccard Index)
    iou = (intersection + smooth) / (union + smooth)

    # Dice Score (F1)
    dice = (2.0 * intersection + smooth) / (np.sum(bin_preds) + np.sum(bin_gts) + smooth)

    # Pixel Accuracy
    correct_pixels = np.sum(bin_preds == bin_gts)
    total_pixels = bin_preds.size
    pixel_acc = correct_pixels / total_pixels

    return {
        "iou": float(iou),
        "dice": float(dice),
        "pixel_accuracy": float(pixel_acc)
    }
