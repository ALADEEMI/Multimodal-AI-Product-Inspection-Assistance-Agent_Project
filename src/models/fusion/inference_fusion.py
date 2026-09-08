"""
Multimodal Fusion Inference Engine.
Loads trained MultimodalFusionModel and provides predictions for text-only,
image-only, or both modalities.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from src.config import (
    FUSION_MODEL_PATH,
    TEXT_EMBED_DIM,
    IMAGE_EMBED_DIM,
    PROBLEM_TYPES,
)
from src.models.fusion.fusion_model import MultimodalFusionModel
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


class FusionInference:
    """Inference engine for multimodal fusion model."""

    def __init__(self, model_path: Path = FUSION_MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MultimodalFusionModel().to(self.device)

        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
            logger.info(f"Loaded fusion model from: {model_path}")
        else:
            logger.warning(f"Fusion model not found at {model_path}. Using uninitialized model.")

        self.model.eval()

    def predict(
        self,
        text_emb: Optional[np.ndarray] = None,
        image_emb: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Run inference on provided modalities.

        Args:
            text_emb: (128,) numpy array or None
            image_emb: (128,) numpy array or None

        Returns:
            Dictionary with fusion decision, confidence, and probabilities.
        """
        has_text = text_emb is not None
        has_image = image_emb is not None

        if not has_text and not has_image:
            return {
                "is_complaint": False,
                "complaint_confidence": 0.0,
                "problem_type": "Other",
                "problem_confidence": 0.0,
                "modality_used": "none",
                "error": "No input modalities provided."
            }

        # Prepare tensors
        t_emb = torch.tensor(
            text_emb if has_text else np.zeros(TEXT_EMBED_DIM),
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        i_emb = torch.tensor(
            image_emb if has_image else np.zeros(IMAGE_EMBED_DIM),
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        t_mask = torch.tensor([[1.0 if has_text else 0.0]], dtype=torch.float32).to(self.device)
        i_mask = torch.tensor([[1.0 if has_image else 0.0]], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            c_logits, p_logits = self.model(t_emb, i_emb, t_mask, i_mask)

            c_prob = torch.sigmoid(c_logits).item()
            p_probs = F.softmax(p_logits, dim=1).squeeze().cpu().numpy()

        is_complaint = c_prob > 0.5
        c_conf = c_prob if is_complaint else (1.0 - c_prob)

        p_idx = int(np.argmax(p_probs))
        problem_type = PROBLEM_TYPES[p_idx]
        p_conf = float(p_probs[p_idx])

        if has_text and has_image:
            modality_used = "multimodal"
        elif has_text:
            modality_used = "text_only"
        else:
            modality_used = "image_only"

        return {
            "is_complaint": bool(is_complaint),
            "complaint_confidence": float(c_conf),
            "problem_type": problem_type,
            "problem_confidence": float(p_conf),
            "all_problem_probabilities": {
                PROBLEM_TYPES[i]: float(p_probs[i]) for i in range(len(PROBLEM_TYPES))
            },
            "modality_used": modality_used
        }
