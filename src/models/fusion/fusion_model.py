"""
Multimodal Fusion Model - Baseline Embedding Concatenation Head.
Combines: text_embedding (128) + image_embedding (128) + modality_mask (2)
-> Dense -> Output decision head.
Follows ADR Phase 6 specifications.
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.config import (
    TEXT_EMBED_DIM,
    IMAGE_EMBED_DIM,
    FUSION_INPUT_DIM,
    FUSION_HIDDEN_DIM,
    PROBLEM_TYPES,
)


class MultimodalFusionModel(nn.Module):
    """
    Baseline Multimodal Fusion Network.
    Concatenates text and image embeddings along with modality mask flags.
    Predicts:
      - is_complaint (Binary: Sigmoid)
      - problem_type (Multi-class: Softmax over 6 classes)
    """

    def __init__(
        self,
        text_dim: int = TEXT_EMBED_DIM,
        image_dim: int = IMAGE_EMBED_DIM,
        hidden_dim: int = FUSION_HIDDEN_DIM,
        num_classes: int = len(PROBLEM_TYPES),
        dropout: float = 0.3
    ):
        super().__init__()
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.input_dim = text_dim + image_dim + 2  # +2 for text_present and image_present masks
        self.num_classes = num_classes

        # Fusion Dense Layers
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln2 = nn.LayerNorm(hidden_dim // 2)

        # Output Prediction Heads
        # Head 1: Binary Complaint Classification
        self.head_complaint = nn.Linear(hidden_dim // 2, 1)

        # Head 2: Multi-class Problem Type Classification
        self.head_problem = nn.Linear(hidden_dim // 2, num_classes)

    def forward(
        self,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor,
        text_mask: torch.Tensor,
        image_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with modality masking.

        Args:
            text_emb: (B, text_dim) text embeddings (zeros if absent)
            image_emb: (B, image_dim) image embeddings (zeros if absent)
            text_mask: (B, 1) 1.0 if text available, 0.0 otherwise
            image_mask: (B, 1) 1.0 if image available, 0.0 otherwise

        Returns:
            complaint_logits: (B, 1)
            problem_logits: (B, num_classes)
        """
        # Apply zeroing to absent modalities
        masked_text = text_emb * text_mask
        masked_image = image_emb * image_mask

        # Concatenate features + masks
        combined = torch.cat([masked_text, masked_image, text_mask, image_mask], dim=1)

        # Dense layers
        x = self.fc1(combined)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = self.relu(x)
        x = self.dropout(x)

        complaint_logits = self.head_complaint(x)
        problem_logits = self.head_problem(x)

        return complaint_logits, problem_logits
