"""
Central Configuration Module for Multimodal AI Product Inspection & Assistance Agent.
Follows ADR-001 and Phase 0 specifications.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# --- Repository Root & Core Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"
VISUALS_DIR = DOCS_DIR / "visuals"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure all primary output directories exist
for p in [PROCESSED_DATA_DIR, ARTIFACTS_DIR, CHECKPOINTS_DIR, LOGS_DIR, VISUALS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# --- MVTec Dataset Configuration ---
MVTEC_CATEGORY = os.getenv("MVTEC_CATEGORY", "bottle")

# Actual data path on disk (accounting for nested archive extraction)
MVTEC_DATA_PATH = RAW_DATA_DIR / "mvtec" / MVTEC_CATEGORY / MVTEC_CATEGORY
if not MVTEC_DATA_PATH.exists():
    # Fallback if un-nested
    alt_path = RAW_DATA_DIR / "mvtec" / MVTEC_CATEGORY
    if alt_path.exists():
        MVTEC_DATA_PATH = alt_path

# Local Policy Configuration
POLICIES_CSV_PATH = DATA_DIR / "policies.csv"

# Processed Dataset Paths
MANIFEST_CSV_PATH = PROCESSED_DATA_DIR / f"mvtec_{MVTEC_CATEGORY}_manifest.csv"
PAIRED_DATASET_CSV_PATH = PROCESSED_DATA_DIR / "paired_dataset.csv"

# --- Reproducibility & Random Seeds ---
PROJECT_SEED = int(os.getenv("PROJECT_SEED", "42"))
GENERATOR_VERSION = "v1.0.0"

def set_seed(seed: int = PROJECT_SEED) -> None:
    """Sets random seeds for reproducibility across random, numpy, and PyTorch."""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Train / Val / Test Split Ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# --- Defect Label to Problem Type & Severity Mapping ---
# Authoritative mapping as specified in ADR Phase 1
DEFECT_MAPPING = {
    "good": {
        "is_complaint": False,
        "problem_type": "Other",
        "severity": "Low",
    },
    "broken_large": {
        "is_complaint": True,
        "problem_type": "Damage",
        "severity": "High",
    },
    "broken_small": {
        "is_complaint": True,
        "problem_type": "Damage",
        "severity": "Medium",
    },
    "contamination": {
        "is_complaint": True,
        "problem_type": "Quality",
        "severity": "Medium",
    },
}

PROBLEM_TYPES = ["Quality", "Packaging", "Price", "Delivery", "Damage", "Other"]
SEVERITY_LEVELS = ["Low", "Medium", "High"]
REQUESTED_ACTIONS = ["return", "replacement", "repair", "unspecified"]

# --- NLP Model Configuration (BiLSTM) ---
VOCAB_SIZE = 1500
EMBEDDING_DIM = 64
TEXT_HIDDEN_DIM = 64
TEXT_EMBED_DIM = 128  # 2 * TEXT_HIDDEN_DIM (bidirectional)
MAX_TEXT_LEN = 32
NLP_LEARNING_RATE = 0.001
NLP_BATCH_SIZE = 16
NLP_EPOCHS = 40
NLP_DROPOUT = 0.3

# NLP Artifact Paths
NLP_MODEL_PATH = ARTIFACTS_DIR / "bilstm_multitask.pt"
TOKENIZER_PATH = ARTIFACTS_DIR / "tokenizer.json"
NLP_LABEL_MAPS_PATH = ARTIFACTS_DIR / "nlp_label_maps.json"
NLP_METRICS_PATH = ARTIFACTS_DIR / "nlp_metrics.json"

# --- Vision Preprocessing & U-Net Configuration ---
IMAGE_SIZE = (256, 256)
VISION_BATCH_SIZE = 16
VISION_LEARNING_RATE = 0.001
VISION_EPOCHS = 8
IMAGE_EMBED_DIM = 128

VISION_STATS_PATH = ARTIFACTS_DIR / "image_norm_stats.json"
VISION_MODEL_PATH = ARTIFACTS_DIR / "unet_segmentation.pt"
VISION_METRICS_PATH = ARTIFACTS_DIR / "vision_metrics.json"

# --- Fusion Model Configuration ---
# Fusion input dimension: text_embed (128) + image_embed (128) + modality_mask (2) = 258
FUSION_INPUT_DIM = TEXT_EMBED_DIM + IMAGE_EMBED_DIM + 2
FUSION_HIDDEN_DIM = 64
FUSION_LEARNING_RATE = 0.001
FUSION_BATCH_SIZE = 16
FUSION_EPOCHS = 15

FUSION_MODEL_PATH = ARTIFACTS_DIR / "multimodal_fusion.pt"
FUSION_METRICS_PATH = ARTIFACTS_DIR / "fusion_metrics.json"

# --- LLM / OpenRouter Configuration ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3-5-sonnet-20241022")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
