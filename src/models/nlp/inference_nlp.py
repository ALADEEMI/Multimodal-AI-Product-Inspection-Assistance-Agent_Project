"""Inference and Embedding Extraction Pipeline for BiLSTM NLP Model.

Loads trained artifacts and exposes inference and text embedding extraction methods.
Follows ADR Phase 2 specifications.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch

import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.config import (
    NLP_MODEL_PATH,
    TOKENIZER_PATH,
    NLP_LABEL_MAPS_PATH,
    TEXT_EMBED_DIM,
    VOCAB_SIZE,
    MAX_TEXT_LEN
)
from src.preprocessing.text_preprocessing import TextTokenizer, clean_text
from src.models.nlp.bilstm_model import BiLSTMMultiTaskModel


class NLPInferenceEngine:
    """Inference engine for text complaint analysis and embedding extraction."""

    def __init__(
        self,
        model_path: Path = NLP_MODEL_PATH,
        tokenizer_path: Path = TOKENIZER_PATH,
        label_maps_path: Path = NLP_LABEL_MAPS_PATH,
        device: Optional[str] = None
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        # 1. Load Tokenizer
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Please train NLP model first.")
        self.tokenizer = TextTokenizer.load(tokenizer_path)

        # 2. Load Label Maps
        if not label_maps_path.exists():
            raise FileNotFoundError(f"Label maps not found at {label_maps_path}")
        with open(label_maps_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)
            self.problem_to_idx = label_data["problem_to_idx"]
            self.idx_to_problem = {int(k): v for k, v in label_data["idx_to_problem"].items()}
            self.problem_types = label_data["problem_types"]

        # 3. Load Trained Model
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config = checkpoint.get("config", {})

        self.model = BiLSTMMultiTaskModel(
            vocab_size=config.get("vocab_size", VOCAB_SIZE),
            embedding_dim=config.get("embedding_dim", 64),
            hidden_dim=config.get("hidden_dim", 64),
            num_problem_types=len(self.problem_types),
            dropout=0.0
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, raw_text: str) -> Dict[str, Any]:
        """Run inference on raw customer text.

        Returns structured prediction dictionary with:
          - is_complaint (bool)
          - complaint_confidence (float)
          - problem_type (str)
          - problem_type_confidence (float)
          - all_problem_confidences (Dict[str, float])
          - text_embedding (List[float])
        """
        if not raw_text or not raw_text.strip():
            # Return neutral fallback for empty text
            return {
                "is_complaint": False,
                "complaint_confidence": 0.0,
                "problem_type": "Other",
                "problem_type_confidence": 0.0,
                "all_problem_confidences": {p: 0.0 for p in self.problem_types},
                "text_embedding": [0.0] * TEXT_EMBED_DIM
            }

        # Preprocess and encode
        encoded = self.tokenizer.encode(raw_text)
        tensor = torch.tensor([encoded], dtype=torch.long).to(self.device)

        with torch.no_grad():
            c_logits, p_logits, embedding = self.model(tensor)

            # Complaint probability
            c_prob = torch.sigmoid(c_logits).item()
            is_complaint = c_prob > 0.5
            complaint_confidence = c_prob if is_complaint else (1.0 - c_prob)

            # Problem type probabilities
            p_probs = torch.softmax(p_logits, dim=1).cpu().numpy()[0]
            predicted_idx = int(np.argmax(p_probs))
            problem_type = self.idx_to_problem.get(predicted_idx, "Other")
            problem_confidence = float(p_probs[predicted_idx])

            # Embedding vector
            emb_vector = embedding.cpu().numpy()[0].tolist()

            confidences_dict = {
                self.idx_to_problem[i]: float(p_probs[i])
                for i in range(len(self.problem_types))
            }

        return {
            "is_complaint": bool(is_complaint),
            "complaint_confidence": float(complaint_confidence),
            "problem_type": str(problem_type),
            "problem_type_confidence": float(problem_confidence),
            "all_problem_confidences": confidences_dict,
            "text_embedding": emb_vector
        }

    def get_text_embedding(self, raw_text: str) -> np.ndarray:
        """Extract only text embedding representation for Fusion."""
        res = self.predict(raw_text)
        return np.array(res["text_embedding"], dtype=np.float32)

    def extract_embedding(self, raw_text: str) -> np.ndarray:
        """Alias for get_text_embedding."""
        return self.get_text_embedding(raw_text)


# Alias for backward compatibility
NLPInference = NLPInferenceEngine

