# Academic Report: AI Agent Architecture & Multimodal Fusion
**Course:** AI Agents & Multimodal Machine Learning  
**System:** LangGraph Orchestration with Local Policy Enforcement & OpenRouter LLM Formatting

---

## 1. Executive Summary & Objective
Autonomous customer assistance requires synthesizing unstructured text, high-dimensional visual defect inspection, deterministic business policies, and human-like natural language generation. This system implements a state-of-the-art **LangGraph StateGraph** pipeline coordinating specialized BiLSTM, U-Net, and Fusion models without external product-catalog APIs.

---

## 2. Multimodal Fusion Baseline & Ablation Study (Phase 6)
The fusion module combines frozen embeddings:
$$\mathbf{z}_{fused} = [\mathbf{e}_{text} \cdot m_{text} \,\|\, \mathbf{e}_{image} \cdot m_{image} \,\|\, m_{text} \,\|\, m_{image}]$$
where $m_{text}, m_{image} \in \{0, 1\}$ are binary modality availability masks.

### Quantitative Comparison Table (Held-Out Test Set - 42 Samples)

| Modality Setting | Binary Complaint Acc | Binary Complaint F1 | Problem Type Acc | Problem Type Macro-F1 |
|---|---|---|---|---|
| **Text Only** ($m_{text}=1, m_{img}=0$) | 97.6% | 98.4% | 95.2% | 94.1% |
| **Image Only** ($m_{text}=0, m_{img}=1$) | 92.9% | 94.7% | 88.1% | 85.3% |
| **Multimodal Fusion (Both)** | **97.6%** | **98.4%** | **97.6%** | **96.8%** |

### Findings & Discussion:
- Joint multimodal fusion outperforms single-modality vision by **+9.5% accuracy** and improves Macro-F1 across ambiguous defect categories (e.g. distinguishing fine contamination from small glass cracks).
- Modality masking enables zero-crash graceful degradation when the customer omits either text or image.

---

## 3. LangGraph Workflow Graph Architecture (Phase 7)

```text
               ┌─────────┐
               │  START  │
               └────┬────┘
                    ▼
          ┌───────────────────┐
          │   receive_input   │
          └─────────┬─────────┘
                    ▼
          ┌───────────────────┐
          │   text_analyzer   │  (BiLSTM inference)
          └─────────┬─────────┘
                    ▼
          ┌───────────────────┐
          │  image_analyzer   │  (U-Net segmentation)
          └─────────┬─────────┘
                    ▼
       ┌─────────────────────────┐
       │  information_extraction │  (Rule-based field parser)
       └────────────┬────────────┘
                    ▼
          ┌───────────────────┐
          │ multimodal_fusion │  (Concatenation & Head)
          └─────────┬─────────┘
                    ▼
          ┌───────────────────┐
          │  context_builder  │  (Builds evidence string)
          └─────────┬─────────┘
                    ▼
    ┌───────────────────────────────┐
    │  conditional_policy_lookup    │  (Queries data/policies.csv if return/exchange)
    └───────────────┬───────────────┘
                    ▼
          ┌───────────────────┐
          │ llm_response_node │  (OpenRouter Claude / Fallback Formatter)
          └─────────┬─────────┘
                    ▼
                ┌───────┐
                │  END  │
                └───────┘
```

---

## 4. Local Policy Tool & Compliance (No External APIs)
In compliance with project constraints, the tool `src/tools/policy_lookup.py` queries only the local validated dataset `data/policies.csv`:
- Schema: `product_category, return_window_days, warranty_summary`.
- **Conditional Trigger:** Executed **only** when `Requested_Action` is `return` or `replacement`.
- **Zero Internet Leakage:** No external REST or RapidAPI calls are made.

---

## 5. Zero-Hallucination Prompting & Fallback Robustness
1. **Evidence-Bounded Context:** The LLM prompt receives strictly structured key-value findings from the specialized models and local policies.
2. **Explicit Uncertainty:** Missing modalities and low-confidence predictions are surfaced as explicit disclaimers in the generated report.
3. **Safe Deterministic Fallback:** If `OPENROUTER_API_KEY` is not present, the system automatically formats a clean, standardized report without breaking the user experience.
