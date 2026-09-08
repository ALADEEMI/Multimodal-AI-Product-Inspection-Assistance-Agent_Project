# Multimodal AI Product Inspection & Assistance Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg)](https://streamlit.io/)

An academic, multimodal AI customer assistance and industrial product inspection system. It pairs customer complaint text with MVTec AD product inspection images, running specialized deep learning models, multimodal fusion, local policy validation, and LangGraph agent orchestration.

---

## 🏗️ System Architecture

```text
User Input (Text Complaint + Product Image)
                    │
       ┌────────────┴────────────┐
       ▼                         ▼
  Text Modality             Image Modality
(Arabic / English)        (MVTec Bottle 256x256)
       │                         │
       ▼                         ▼
Preprocessed Text         Preprocessed Image
       │                         │
       ▼                         ▼
BiLSTM Multi-Task       Mini U-Net Segmentation
(is_complaint, problem)   (BCE+Dice Mask Localization)
       │                         │
  [128-d Text Emb]          [128-d Image Emb]
       │                         │
       └────────────┬────────────┘
                    ▼
        Multimodal Fusion Model
        (Concatenation + Dense Head)
                    │
                    ▼
       Rule-Based Field Extraction
     (Product, Problem, Severity, Action)
                    │
                    ▼
       Local Policy Verification
        (data/policies.csv tool)
                    │
                    ▼
     LangGraph Agent & OpenRouter LLM
       (Zero-Hallucination Customer Report)
```

---

## 📁 Repository Structure

```text
├── adr/                         # Architecture Decision Records (Phases 0-9)
├── artifacts/                   # Saved model weights (.pt) and tokenizers
├── checkpoints/                 # Training checkpoints
├── data/
│   ├── raw/mvtec/bottle/bottle/ # Raw MVTec AD benchmark images & masks
│   ├── processed/               # Generated paired dataset and manifests
│   └── policies.csv             # Local return & warranty policy catalog
├── docs/                        # Academic reports & visual comparisons
│   ├── nlp_report.md            # NLP BiLSTM Multi-task course report
│   ├── vision_dl_report.md      # Computer Vision U-Net course report
│   ├── agent_report.md          # AI Agent & Multimodal Fusion report
│   ├── dataset_card.md          # Synthetic paired dataset documentation
│   └── visuals/                 # Before/After preprocessing and mask overlays
├── src/
│   ├── config.py                # Central configuration and hyperparameters
│   ├── data/                    # Dataset generation & noise injection
│   ├── preprocessing/           # Image and text preprocessing pipelines
│   ├── models/
│   │   ├── nlp/                 # BiLSTM model, training, and inference
│   │   ├── vision/              # Mini U-Net model, training, and inference
│   │   └── fusion/              # Multimodal fusion model and comparison
│   ├── extraction/              # Rule-based field extraction (Product, Action)
│   ├── tools/                   # Local policy lookup tool (no external APIs)
│   ├── agent/                   # LangGraph StateGraph & OpenRouter LLM adapter
│   ├── app/                     # Streamlit interactive web demo
│   └── utils/                   # Logging and seed management utilities
├── tests/                       # Complete pytest unit and E2E test suite
├── .env.example                 # Environment variable template
├── requirements.txt             # Pinned project dependencies
└── README.md                    # Project documentation & Quickstart
```

---

## 🚀 Quickstart & Installation

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/ALADEEMI/Multimodal-AI-Product-Inspection-Assistance-Agent.git
cd "Multimodal AI Product Inspection & Assistance Agent_Project"

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
*(Optional: Add your `OPENROUTER_API_KEY` to `.env` to enable live LLM generation. If omitted, the agent uses high-quality deterministic structured fallback reports).*

---

## 🔬 Running the Complete Pipeline

### Step 1: Validate Dataset & Generate Paired Dataset (Phases 0 & 1)
```bash
# Validate raw MVTec dataset structure
python -m src.data.mvtec_validator

# Generate paired dataset with dirty text simulation
python -m src.data.dataset_generator
```

### Step 2: Train NLP & Vision Models (Phases 2 & 4)
```bash
# Train BiLSTM multi-task classifier
python -m src.models.nlp.train_nlp

# Train Mini U-Net segmentation model
python -m src.models.vision.train_vision
```

### Step 3: Train Multimodal Fusion & Generate Comparison (Phase 6)
```bash
python -m src.models.fusion.train_fusion
```

### Step 4: Run Test Suite (Unit & End-to-End Scenarios)
```bash
pytest -v
```

### Step 5: Launch Interactive Web Demo (Phase 8)
```bash
streamlit run src/app/demo_ui.py
```

---

## 🧪 Evaluation & Results

### Multimodal Fusion vs Single Modality (Held-Out Test Set)

| Modality Setting | Binary Complaint Acc | Binary Complaint F1 | Problem Type Acc | Problem Type Macro-F1 |
|---|---|---|---|---|
| **Text Only** | 97.6% | 98.4% | 95.2% | 94.1% |
| **Image Only** | 92.9% | 94.7% | 88.1% | 85.3% |
| **Multimodal Fusion (Both)** | **97.6%** | **98.4%** | **97.6%** | **96.8%** |

---

## 📜 Academic Course Deliverables
- **NLP Report:** [`docs/nlp_report.md`](docs/nlp_report.md)
- **Computer Vision & Deep Learning Report:** [`docs/vision_dl_report.md`](docs/vision_dl_report.md)
- **AI Agent & Multimodal Fusion Report:** [`docs/agent_report.md`](docs/agent_report.md)
- **Dataset Card:** [`docs/dataset_card.md`](docs/dataset_card.md)

---

## ⚖️ Dataset & Licensing
- **MVTec AD Dataset:** Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
- **Academic Context:** Developed as a joint project for Natural Language Processing, Computer Vision / Image Processing, and Deep Learning / AI Agents.
