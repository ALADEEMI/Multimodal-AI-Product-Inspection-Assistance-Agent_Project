"""
Vision Model Inference Engine.
Phase 4: Loads trained U-Net, performs segmentation, extracts image embeddings.
"""

from typing import Dict, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2

from src.config import (
    VISION_MODEL_PATH,
    IMAGE_SIZE,
    PROJECT_ROOT,
)
from src.models.vision.unet_model import MiniUNet
from src.preprocessing.image_preprocessing import preprocess_image
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


class VisionInference:
    """Vision inference engine for segmentation and embedding extraction."""

    def __init__(self, model_path: Path = VISION_MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MiniUNet(in_channels=3, out_channels=1).to(self.device)

        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded vision model from: {model_path}")
        else:
            logger.warning(f"Vision model not found at {model_path}. Using untrained model.")

        self.model.eval()

    def predict(
        self,
        image_path: str,
        threshold: float = 0.5
    ) -> Dict:
        """
        Perform segmentation inference on an image.

        Args:
            image_path: Path to input image (relative to project root or absolute)
            threshold: Threshold for binary mask (default: 0.5)

        Returns:
            Dictionary containing:
              - mask_probability: (H, W) probability map
              - binary_mask: (H, W) binary mask
              - segmentation_confidence: float (mean probability in defect regions)
              - has_defect: bool
              - defect_area_ratio: float (% of image with defect)
              - location_summary: str
              - image_embedding: (IMAGE_EMBED_DIM,) numpy array
        """
        # Load and preprocess image
        if not Path(image_path).is_absolute():
            image_path = PROJECT_ROOT / image_path

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Preprocess
        image_tensor = preprocess_image(image_path, augment=False)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)  # (1, 3, H, W)

        # Inference
        with torch.no_grad():
            mask_logits, image_embedding = self.model(image_tensor)
            mask_prob = torch.sigmoid(mask_logits).squeeze().cpu().numpy()  # (H, W)
            embedding = image_embedding.squeeze().cpu().numpy()  # (IMAGE_EMBED_DIM,)

        # Binary mask
        binary_mask = (mask_prob > threshold).astype(np.uint8)

        # Compute metrics
        defect_area_ratio = binary_mask.sum() / binary_mask.size
        has_defect = defect_area_ratio > 0.01  # 1% threshold

        # Segmentation confidence (mean probability in defect regions)
        if has_defect:
            segmentation_confidence = mask_prob[binary_mask > 0].mean()
        else:
            segmentation_confidence = 1.0 - mask_prob.max()

        # Location summary
        location_summary = self._get_location_summary(binary_mask)

        return {
            "mask_probability": mask_prob,
            "binary_mask": binary_mask,
            "segmentation_confidence": float(segmentation_confidence),
            "has_defect": bool(has_defect),
            "defect_area_ratio": float(defect_area_ratio),
            "location_summary": location_summary,
            "image_embedding": embedding,
        }

    def _get_location_summary(self, binary_mask: np.ndarray) -> str:
        """Generate human-readable location summary from binary mask."""
        if binary_mask.sum() == 0:
            return "No defect detected"

        # Find bounding box
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)

        if not rows.any() or not cols.any():
            return "No defect detected"

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        h, w = binary_mask.shape
        center_r = (rmin + rmax) / 2
        center_c = (cmin + cmax) / 2

        # Quadrant detection
        vertical = "top" if center_r < h / 3 else "bottom" if center_r > 2 * h / 3 else "center"
        horizontal = "left" if center_c < w / 3 else "right" if center_c > 2 * w / 3 else "center"

        if vertical == "center" and horizontal == "center":
            return "Defect at center"
        elif vertical == "center":
            return f"Defect at {horizontal}"
        elif horizontal == "center":
            return f"Defect at {vertical}"
        else:
            return f"Defect at {vertical}-{horizontal}"

    def extract_embedding(self, image_path: str) -> np.ndarray:
        """Extract only the image embedding without segmentation."""
        if not Path(image_path).is_absolute():
            image_path = PROJECT_ROOT / image_path

        image_path = Path(image_path)
        image_tensor = preprocess_image(image_path, augment=False)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model.extract_image_embedding(image_tensor)
            return embedding.squeeze().cpu().numpy()


def main():
    """Test vision inference on a sample image."""
    from src.config import PAIRED_DATASET_CSV_PATH
    import pandas as pd

    df = pd.read_csv(PAIRED_DATASET_CSV_PATH)
    test_samples = df[df["split"] == "test"]
    defect_sample = test_samples[test_samples["is_complaint"]].iloc[0]

    logger.info(f"Testing on: {defect_sample['image_path']}")

    inference = VisionInference()
    results = inference.predict(defect_sample["image_path"])

    logger.info(f"Has defect: {results['has_defect']}")
    logger.info(f"Confidence: {results['segmentation_confidence']:.3f}")
    logger.info(f"Defect area: {results['defect_area_ratio']:.3%}")
    logger.info(f"Location: {results['location_summary']}")
    logger.info(f"Embedding shape: {results['image_embedding'].shape}")


if __name__ == "__main__":
    main()
