"""
Image Preprocessing Pipeline for MVTec Bottle Defect Detection.

Handles: resize, denoise, contrast enhancement, normalization, augmentation,
and PyTorch SegmentationDataset.
Follows ADR Phase 3 specifications.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import IMAGE_SIZE, VISION_STATS_PATH, PROJECT_ROOT, PAIRED_DATASET_CSV_PATH


def load_image_and_mask(
    image_path: Path,
    mask_path: Optional[Path] = None,
    as_rgb: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load image and optional mask from paths."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    if as_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = None
    if mask_path and Path(mask_path).exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")

    return img, mask


def resize_image_and_mask(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    target_size: Tuple[int, int] = IMAGE_SIZE
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Resize image and mask with appropriate interpolation."""
    resized_img = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    resized_mask = None
    if mask is not None:
        # CRITICAL: Use INTER_NEAREST for mask to preserve binary values
        resized_mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    return resized_img, resized_mask


def denoise_image(image: np.ndarray, strength: int = 3) -> np.ndarray:
    """Apply light denoising using Non-Local Means."""
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)


def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Apply CLAHE contrast enhancement."""
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return enhanced


def normalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """Normalize image to float32 with mean/std."""
    img_float = image.astype(np.float32) / 255.0
    for i in range(3):
        img_float[:, :, i] = (img_float[:, :, i] - mean[i]) / std[i]
    return img_float


def compute_dataset_statistics(image_paths: List[Path]) -> Dict[str, Any]:
    """Compute mean and std from training images only."""
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized, _ = resize_image_and_mask(img, None)
        img_float = img_resized.astype(np.float64) / 255.0

        pixel_sum += img_float.reshape(-1, 3).sum(axis=0)
        pixel_sq_sum += (img_float ** 2).reshape(-1, 3).sum(axis=0)
        total_pixels += img_float.shape[0] * img_float.shape[1]

    mean = (pixel_sum / total_pixels).tolist()
    std = np.sqrt((pixel_sq_sum / total_pixels) - ((pixel_sum / total_pixels) ** 2)).tolist()

    stats = {
        "mean": mean,
        "std": std,
        "num_images": len(image_paths),
        "image_size": list(IMAGE_SIZE)
    }

    with open(VISION_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def preprocess_image(
    image_path: Path,
    augment: bool = False,
    stats_path: Path = VISION_STATS_PATH
) -> torch.Tensor:
    """Preprocess single image for model forward pass."""
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img, _ = resize_image_and_mask(img, None)
    img = denoise_image(img)
    img = enhance_contrast(img)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    if stats_path.exists():
        with open(stats_path, "r") as f:
            s = json.load(f)
            mean = tuple(s["mean"])
            std = tuple(s["std"])

    img_norm = normalize_image(img, mean, std)
    img_tensor = torch.tensor(np.transpose(img_norm, (2, 0, 1)), dtype=torch.float32)
    return img_tensor


class SegmentationDataset(Dataset):
    """PyTorch Dataset for U-Net Supervised Segmentation."""

    def __init__(
        self,
        csv_path: Path = PAIRED_DATASET_CSV_PATH,
        split: str = "train",
        augment: bool = False,
        stats_path: Path = VISION_STATS_PATH
    ):
        self.split = split
        self.augment = augment
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        # Load or compute stats
        if not stats_path.exists():
            train_paths = [
                PROJECT_ROOT / p
                for p in self.df[self.df["split"] == "train"]["image_path"].tolist()
            ]
            if train_paths:
                compute_dataset_statistics(train_paths)

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        if stats_path.exists():
            with open(stats_path, "r") as f:
                s = json.load(f)
                self.mean = tuple(s["mean"])
                self.std = tuple(s["std"])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = PROJECT_ROOT / row["image_path"]
        mask_path = PROJECT_ROOT / row["mask_path"] if pd.notna(row["mask_path"]) else None

        img, mask = load_image_and_mask(img_path, mask_path)
        img, mask = resize_image_and_mask(img, mask)

        img = denoise_image(img)
        img = enhance_contrast(img)

        # Ensure mask is binary (0 or 1)
        if mask is None:
            mask_arr = np.zeros(IMAGE_SIZE, dtype=np.float32)
        else:
            mask_arr = (mask > 127).astype(np.float32)

        img_norm = normalize_image(img, self.mean, self.std)

        # PyTorch Tensors
        img_tensor = torch.tensor(np.transpose(img_norm, (2, 0, 1)), dtype=torch.float32)
        mask_tensor = torch.tensor(mask_arr, dtype=torch.float32).unsqueeze(0)  # (1, H, W)

        # Augmentation on training only (applied synchronously to image and mask)
        if self.augment and self.split == "train":
            if random.random() > 0.5:
                img_tensor = TF.hflip(img_tensor)
                mask_tensor = TF.hflip(mask_tensor)
            if random.random() > 0.5:
                img_tensor = TF.vflip(img_tensor)
                mask_tensor = TF.vflip(mask_tensor)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "sample_id": row["sample_id"],
            "defect_type": row["defect_type"],
            "is_complaint": row["is_complaint"]
        }
