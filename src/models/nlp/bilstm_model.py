"""Shared BiLSTM Multi-Task Classification Model for Complaints and Problem Types.

Architecture:
  Embedding -> Shared BiLSTM -> Dropout -> Context Embedding
  -> Head 1: Binary Classification (is_complaint) -> Sigmoid
  -> Head 2: Multi-class Classification (problem_type) -> Softmax
  + Text embedding extraction method for multimodal fusion.

Follows ADR Phase 2 specifications.
"""

from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.config import (
    VOCAB_SIZE,
    EMBEDDING_DIM,
    TEXT_HIDDEN_DIM,
    TEXT_EMBED_DIM,
    NLP_DROPOUT,
    PROBLEM_TYPES
)


class BiLSTMMultiTaskModel(nn.Module):
    """Shared BiLSTM multi-task model with dual classification heads."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_dim: int = TEXT_HIDDEN_DIM,
        num_problem_types: int = len(PROBLEM_TYPES),
        dropout: float = NLP_DROPOUT,
        padding_idx: int = 0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_problem_types = num_problem_types
        self.dropout_rate = dropout

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        # Shared BiLSTM Encoder
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Text embedding dimension is hidden_dim * 2 (forward + backward)
        self.text_embed_dim = hidden_dim * 2

        # Binary Head: is_complaint (Output logit for BCEWithLogitsLoss)
        self.complaint_head = nn.Sequential(
            nn.Linear(self.text_embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        # Multi-class Head: problem_type (Output logits for CrossEntropyLoss)
        self.problem_head = nn.Sequential(
            nn.Linear(self.text_embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_problem_types)
        )

    def extract_text_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract text embedding representation (layer before heads).

        Args:
            x: Token tensor of shape (batch_size, max_len)

        Returns:
            Text embedding of shape (batch_size, text_embed_dim)
        """
        # Embed: (batch, seq_len, embed_dim)
        embeds = self.embedding(x)

        # LSTM: output shape (batch, seq_len, hidden_dim * 2)
        lstm_out, (hn, cn) = self.lstm(embeds)

        # Global max pooling across sequence dimension for robust representation
        # Shape: (batch, hidden_dim * 2)
        text_embedding, _ = torch.max(lstm_out, dim=1)

        return text_embedding

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input token indices of shape (batch_size, seq_len)

        Returns:
            complaint_logits: (batch_size, 1)
            problem_logits: (batch_size, num_problem_types)
            text_embedding: (batch_size, text_embed_dim)
        """
        # Get shared representation
        text_embedding = self.extract_text_embedding(x)
        text_features = self.dropout(text_embedding)

        # Predictions
        complaint_logits = self.complaint_head(text_features)
        problem_logits = self.problem_head(text_features)

        return complaint_logits, problem_logits, text_embedding
