"""
Generate and Save Visual Preprocessing and Inspection Artifacts.
Generates Before/After Preprocessing, Segmentation Overlays, and Diagnostics under docs/visuals/.
"""

import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    PAIRED_DATASET_CSV_PATH,
    DOCS_DIR,
    PROJECT_ROOT,
)
from src.preprocessing.image_preprocessing import (
    load_image_and_mask,
    resize_image_and_mask,
    denoise_image,
    enhance_contrast,
)
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def generate_preprocessing_visuals():
    """Generate Before/After Preprocessing visuals."""
    output_dir = DOCS_DIR / "visuals" / "preprocessing"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PAIRED_DATASET_CSV_PATH)
    defect_sample = df[df["defect_type"] == "broken_large"].iloc[0]

    img_path = PROJECT_ROOT / defect_sample["image_path"]
    mask_path = PROJECT_ROOT / defect_sample["mask_path"] if pd.notna(defect_sample["mask_path"]) else None

    # Load raw
    raw_img, raw_mask = load_image_and_mask(img_path, mask_path)
    resized_img, resized_mask = resize_image_and_mask(raw_img, raw_mask)

    # Stages
    denoised = denoise_image(resized_img)
    contrast_enhanced = enhance_contrast(denoised)

    # Create 4-panel comparison figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(resized_img)
    axes[0].set_title("1. Raw Image (Resized)")
    axes[0].axis("off")

    axes[1].imshow(denoised)
    axes[1].set_title("2. Denoised (NLM)")
    axes[1].axis("off")

    axes[2].imshow(contrast_enhanced)
    axes[2].set_title("3. CLAHE Contrast Enhanced")
    axes[2].axis("off")

    if resized_mask is not None:
        axes[3].imshow(resized_mask, cmap="gray")
        axes[3].set_title("4. Ground Truth Mask")
    else:
        axes[3].imshow(np.zeros_like(resized_img)[:, :, 0], cmap="gray")
        axes[3].set_title("4. No Mask (Normal)")
    axes[3].axis("off")

    plt.tight_layout()
    save_path = output_dir / "preprocessing_stages_comparison.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved preprocessing stages visual to: {save_path}")


def main():
    generate_preprocessing_visuals()


if __name__ == "__main__":
    main()
