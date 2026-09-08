"""Training script for BiLSTM Multi-Task NLP Model.

Trains is_complaint binary head and problem_type multi-class head simultaneously.
Saves model checkpoint, tokenizer, label maps, and evaluation metrics.
Follows ADR Phase 2 specifications.
"""

import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.config import (
    PAIRED_DATASET_CSV_PATH,
    NLP_MODEL_PATH,
    TOKENIZER_PATH,
    NLP_LABEL_MAPS_PATH,
    NLP_METRICS_PATH,
    PROJECT_SEED,
    VOCAB_SIZE,
    MAX_TEXT_LEN,
    EMBEDDING_DIM,
    TEXT_HIDDEN_DIM,
    NLP_BATCH_SIZE,
    NLP_LEARNING_RATE,
    NLP_EPOCHS,
    NLP_DROPOUT,
    PROBLEM_TYPES,
    set_seed
)
from src.preprocessing.text_preprocessing import TextTokenizer, ComplaintDataset
from src.models.nlp.bilstm_model import BiLSTMMultiTaskModel
from src.evaluation.nlp_metrics import compute_binary_metrics, compute_multiclass_metrics


def load_data_by_split(csv_path: Path) -> Tuple[Dict[str, List], Dict[str, List]]:
    """Load texts and labels split by train/validation/test."""
    data = {
        "train": {"texts": [], "complaint": [], "problem": []},
        "validation": {"texts": [], "complaint": [], "problem": []},
        "test": {"texts": [], "complaint": [], "problem": []}
    }

    problem_to_idx = {name: idx for idx, name in enumerate(PROBLEM_TYPES)}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            if split not in data:
                continue

            text = row["generated_text"]
            is_complaint = 1 if row["is_complaint"].lower() == "true" else 0
            problem_type = row["problem_type"]
            problem_idx = problem_to_idx.get(problem_type, problem_to_idx["Other"])

            data[split]["texts"].append(text)
            data[split]["complaint"].append(is_complaint)
            data[split]["problem"].append(problem_idx)

    return data, problem_to_idx


def train_nlp_model():
    """Main training routine for BiLSTM NLP Model."""
    print("=" * 60)
    print("Phase 2: Training BiLSTM Multi-Task NLP Model")
    print("=" * 60)

    set_seed(PROJECT_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load dataset
    print(f"Loading paired dataset from: {PAIRED_DATASET_CSV_PATH}")
    data, problem_to_idx = load_data_by_split(PAIRED_DATASET_CSV_PATH)

    print(f"Train samples: {len(data['train']['texts'])}")
    print(f"Validation samples: {len(data['validation']['texts'])}")
    print(f"Test samples: {len(data['test']['texts'])}")

    # 2. Fit Tokenizer ON TRAIN ONLY (Strict ADR rule)
    print("\nFitting tokenizer strictly on training texts...")
    tokenizer = TextTokenizer(vocab_size=VOCAB_SIZE, max_len=MAX_TEXT_LEN)
    tokenizer.fit(data["train"]["texts"])
    print(f"Vocabulary size fit: {len(tokenizer.word2idx)} tokens")

    # Save tokenizer & label maps
    tokenizer.save(TOKENIZER_PATH)
    with open(NLP_LABEL_MAPS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "problem_to_idx": problem_to_idx,
            "idx_to_problem": {v: k for k, v in problem_to_idx.items()},
            "problem_types": PROBLEM_TYPES
        }, f, indent=2)

    # 3. Create PyTorch Datasets & DataLoaders
    train_dataset = ComplaintDataset(
        data["train"]["texts"],
        data["train"]["complaint"],
        data["train"]["problem"],
        tokenizer
    )
    val_dataset = ComplaintDataset(
        data["validation"]["texts"],
        data["validation"]["complaint"],
        data["validation"]["problem"],
        tokenizer
    )
    test_dataset = ComplaintDataset(
        data["test"]["texts"],
        data["test"]["complaint"],
        data["test"]["problem"],
        tokenizer
    )

    train_loader = DataLoader(train_dataset, batch_size=NLP_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=NLP_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=NLP_BATCH_SIZE, shuffle=False)

    # 4. Initialize BiLSTM Model
    model = BiLSTMMultiTaskModel(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=TEXT_HIDDEN_DIM,
        num_problem_types=len(PROBLEM_TYPES),
        dropout=NLP_DROPOUT
    ).to(device)

    # Losses: BCEWithLogitsLoss for binary head, CrossEntropyLoss for multi-class head
    criterion_binary = nn.BCEWithLogitsLoss()
    criterion_multi = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=NLP_LEARNING_RATE)

    # Multi-task loss weights
    w_binary = 1.0
    w_multi = 1.0

    # 5. Training Loop with Early Stopping on Validation Macro-F1
    best_val_macro_f1 = -1.0
    best_epoch = -1
    history = []

    print("\nStarting model training...")
    for epoch in range(1, NLP_EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for texts, complaints, problems in train_loader:
            texts = texts.to(device)
            complaints = complaints.to(device).unsqueeze(1)
            problems = problems.to(device)

            optimizer.zero_grad()

            complaint_logits, problem_logits, _ = model(texts)

            loss_bin = criterion_binary(complaint_logits, complaints)
            loss_mul = criterion_multi(problem_logits, problems)
            total_loss = (w_binary * loss_bin) + (w_multi * loss_mul)

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()

        train_loss /= len(train_loader)

        # Validation Step
        model.eval()
        val_loss = 0.0
        val_comp_preds, val_comp_targets = [], []
        val_prob_preds, val_prob_targets = [], []

        with torch.no_grad():
            for texts, complaints, problems in val_loader:
                texts = texts.to(device)
                complaints = complaints.to(device).unsqueeze(1)
                problems = problems.to(device)

                c_logits, p_logits, _ = model(texts)

                l_b = criterion_binary(c_logits, complaints)
                l_m = criterion_multi(p_logits, problems)
                val_loss += ((w_binary * l_b) + (w_multi * l_m)).item()

                # Binary predictions
                c_probs = torch.sigmoid(c_logits)
                c_pred = (c_probs > 0.5).int().cpu().numpy().flatten()
                val_comp_preds.extend(c_pred)
                val_comp_targets.extend(complaints.cpu().numpy().flatten())

                # Multi-class predictions
                p_pred = torch.argmax(p_logits, dim=1).cpu().numpy().flatten()
                val_prob_preds.extend(p_pred)
                val_prob_targets.extend(problems.cpu().numpy().flatten())

        val_loss /= len(val_loader)

        bin_metrics = compute_binary_metrics(val_comp_targets, val_comp_preds)
        mul_metrics = compute_multiclass_metrics(val_prob_targets, val_prob_preds, PROBLEM_TYPES)

        val_macro_f1 = mul_metrics["macro_f1"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_binary_f1": bin_metrics["f1"],
            "val_macro_f1": val_macro_f1
        })

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{NLP_EPOCHS:02d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Comp F1: {bin_metrics['f1']:.4f} | "
                  f"Val Problem Macro-F1: {val_macro_f1:.4f}")

        # Checkpoint based on Validation Macro-F1 (Strict ADR rule: never on test)
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            NLP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_macro_f1,
                "config": {
                    "vocab_size": VOCAB_SIZE,
                    "embedding_dim": EMBEDDING_DIM,
                    "hidden_dim": TEXT_HIDDEN_DIM,
                    "dropout": NLP_DROPOUT,
                    "max_len": MAX_TEXT_LEN
                }
            }, NLP_MODEL_PATH)

    print(f"\n[OK] Best model saved at epoch {best_epoch} with Val Macro-F1: {best_val_macro_f1:.4f}")

    # 6. Final Evaluation on Test Set (Frozen Test Evaluation)
    print("\nRunning final evaluation on held-out test split...")
    checkpoint = torch.load(NLP_MODEL_PATH, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_comp_preds, test_comp_targets = [], []
    test_prob_preds, test_prob_targets = [], []

    with torch.no_grad():
        for texts, complaints, problems in test_loader:
            texts = texts.to(device)
            c_logits, p_logits, _ = model(texts)

            c_pred = (torch.sigmoid(c_logits) > 0.5).int().cpu().numpy().flatten()
            test_comp_preds.extend(c_pred)
            test_comp_targets.extend(complaints.numpy().flatten())

            p_pred = torch.argmax(p_logits, dim=1).cpu().numpy().flatten()
            test_prob_preds.extend(p_pred)
            test_prob_targets.extend(problems.numpy().flatten())

    test_bin_metrics = compute_binary_metrics(test_comp_targets, test_comp_preds)
    test_mul_metrics = compute_multiclass_metrics(test_prob_targets, test_prob_preds, PROBLEM_TYPES)

    final_results = {
        "model": "BiLSTM-MultiTask",
        "best_epoch": best_epoch,
        "test_binary_metrics": test_bin_metrics,
        "test_multiclass_metrics": test_mul_metrics,
        "history": history
    }

    with open(NLP_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print("\n" + "=" * 50)
    print("FINAL TEST EVALUATION REPORT (NLP)")
    print("=" * 50)
    print(f"Binary is_complaint:")
    print(f"  Accuracy:  {test_bin_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_bin_metrics['precision']:.4f}")
    print(f"  Recall:    {test_bin_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_bin_metrics['f1']:.4f}")
    print(f"\nMulti-class problem_type:")
    print(f"  Accuracy:  {test_mul_metrics['accuracy']:.4f}")
    print(f"  Macro-F1:  {test_mul_metrics['macro_f1']:.4f}")
    print("\nPer-class F1 Scores:")
    for cls_name, scores in test_mul_metrics["per_class"].items():
        if scores["support"] > 0:
            print(f"  {cls_name:<12}: Precision={scores['precision']:.2f}, "
                  f"Recall={scores['recall']:.2f}, F1={scores['f1']:.2f} (n={scores['support']})")

    print(f"\nMetrics saved to: {NLP_METRICS_PATH}")
    print("[OK] Phase 2 complete: BiLSTM NLP pipeline trained and validated")


if __name__ == "__main__":
    train_nlp_model()
