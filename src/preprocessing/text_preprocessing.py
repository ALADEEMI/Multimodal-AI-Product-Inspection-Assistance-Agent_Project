"""Text Preprocessing and Tokenization for Multimodal Product Inspection.

Normalizes text while preserving dirty-data signals (emojis, typos).
Follows ADR Phase 2 specifications.
"""

import re
import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import VOCAB_SIZE, MAX_TEXT_LEN


def clean_text(text: str) -> str:
    """Clean text while preserving important dirty data signals.

    - Normalizes Unicode
    - Preserves Arabic and English characters
    - Preserves emojis
    - Normalizes excessive whitespace
    """
    if not text or not isinstance(text, str):
        return ""

    # Normalize unicode (NFKC)
    text = unicodedata.normalize("NFKC", text)

    # Normalize excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


class TextTokenizer:
    """Word-level tokenizer with OOV handling and padding."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, max_len: int = MAX_TEXT_LEN):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.word2idx: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word: Dict[int, str] = {0: "<PAD>", 1: "<UNK>"}
        self.is_fit: bool = False

    def fit(self, texts: List[str]) -> None:
        """Fit vocabulary on training texts only."""
        word_counts: Dict[str, int] = {}

        for text in texts:
            cleaned = clean_text(text)
            tokens = cleaned.lower().split()
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1

        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        # Add top words up to vocab_size
        for word, _ in sorted_words[: self.vocab_size - 2]:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        self.is_fit = True

    def encode(self, text: str) -> List[int]:
        """Encode text to token ids with truncation and padding."""
        if not self.is_fit:
            raise ValueError("Tokenizer must be fit before encoding")

        cleaned = clean_text(text)
        tokens = cleaned.lower().split()

        # Map to indices
        indices = [self.word2idx.get(token, 1) for token in tokens]  # 1 is <UNK>

        # Truncate if necessary
        if len(indices) > self.max_len:
            indices = indices[: self.max_len]

        # Pad with 0 (<PAD>)
        while len(indices) < self.max_len:
            indices.append(0)

        return indices

    def decode(self, indices: List[int]) -> str:
        """Decode token ids back to text."""
        words = [self.idx2word.get(idx, "<UNK>") for idx in indices if idx != 0]
        return " ".join(words)

    def save(self, path: Path) -> None:
        """Save tokenizer vocabulary to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "max_len": self.max_len,
                "word2idx": self.word2idx,
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TextTokenizer":
        """Load tokenizer from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(vocab_size=data["vocab_size"], max_len=data["max_len"])
        tokenizer.word2idx = data["word2idx"]
        tokenizer.idx2word = {int(v): k for k, v in tokenizer.word2idx.items()}
        tokenizer.is_fit = True
        return tokenizer


class ComplaintDataset(Dataset):
    """PyTorch Dataset for complaint classification."""

    def __init__(
        self,
        texts: List[str],
        is_complaint_labels: List[int],
        problem_type_labels: List[int],
        tokenizer: TextTokenizer,
    ):
        self.texts = texts
        self.is_complaint_labels = is_complaint_labels
        self.problem_type_labels = problem_type_labels
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer.encode(self.texts[idx])
        text_tensor = torch.tensor(encoded, dtype=torch.long)
        complaint_tensor = torch.tensor(self.is_complaint_labels[idx], dtype=torch.float32)
        problem_tensor = torch.tensor(self.problem_type_labels[idx], dtype=torch.long)

        return text_tensor, complaint_tensor, problem_tensor
