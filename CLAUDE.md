# Multimodal AI Product Inspection & Assistance Agent

## Project Purpose
An academic, multimodal AI customer assistance system that processes paired customer complaint text and product inspection images. It uses:
- **BiLSTM (RNN)** for complaint classification and problem-type categorization.
- **U-Net (CNN)** for industrial surface defect segmentation on MVTec AD images.
- **Embedding Concatenation Fusion** combining text and vision embeddings for unified decision-making.
- **LangGraph AI Agent** providing structured customer support reports with local policy enforcement (no external product APIs).
- **Interactive Web Demo** (Streamlit/Gradio) for real-time inference and evaluation.

---

## Authoritative Documentation Order (Precedence)
When decisions or specifications conflict, follow this strict hierarchy:
1. `docs/ADR-001-multimodal-inspection-architecture-and-delivery-plan.md` (Canonical, supersedes all)
2. `adr/phase-0-foundation.md` through `adr/phase-9-delivery.md` (Phase-specific execution specs)
3. `docs/PRD-multimodal-inspection-agent.md` (Product requirements)
4. `docs/project-discussion-summary.md` (Background and rationale)

---

## Dataset & Paths
- **Selected Category:** `bottle` (single-category baseline; extensible design).
- **Actual MVTec Data Path:** `data/raw/mvtec/bottle/bottle/`
  - Train: `data/raw/mvtec/bottle/bottle/train/good/` (209 images)
  - Test: `data/raw/mvtec/bottle/bottle/test/{broken_large,broken_small,contamination,good}/` (83 images)
  - Masks: `data/raw/mvtec/bottle/bottle/ground_truth/{broken_large,broken_small,contamination}/` (63 masks)
- **Local Policy Store:** `data/policies.csv` (contains category return windows and warranty rules).
- **CRITICAL:** Preserve all raw data under `data/raw/mvtec/`. Never modify, move, or delete source images or masks.

---

## Architectural Constraints & Non-Negotiables
1. **No External Product-Data APIs:** RapidAPI / external catalog APIs are strictly forbidden. Use only local data and models.
2. **Local Policy Enforcement:** Policy lookups are handled exclusively via `data/policies.csv` for return/replacement requests.
3. **LLM Usage:** LLM (via OpenRouter) is strictly a final structured report formatter and customer response generator. It does not replace the specialized BiLSTM, U-Net, or Fusion models.
4. **OpenRouter Secrets:**
   - Read `OPENROUTER_API_KEY` exclusively from environment variables via `.env`.
   - Never hardcode, print, log, or commit API keys or tokens.
   - `.env` must always remain in `.gitignore`.
5. **Modality Handling:** The system must gracefully support 3 input modes:
   - Text only
   - Image only
   - Both text and image (multimodal)
6. **Deterministic Extraction:** Phase 5 information extraction is rule-based (regex/keyword), not LLM-driven.
7. **Zero Hallucination:** Prompts must explicitly instruct the LLM not to fabricate facts, policy terms, or inspection results outside provided context.

---

## Required Execution Phases (0–9)
- **Phase 0:** Project Foundation, directory layout, MVTec validation, unified random seeds (`PROJECT_SEED=42`).
- **Phase 1:** Synthetic Paired Dataset Generation (template + noise injection from MVTec labels).
- **Phase 2:** NLP Pipeline — BiLSTM multi-task classifier (`is_complaint`, `problem_type`) & embedding extractor.
- **Phase 3:** Image Preprocessing Pipeline (resize, denoise, contrast, normalize, train-only augmentation).
- **Phase 4:** Deep Learning Vision — Supervised U-Net mini segmentation & image embedding extractor.
- **Phase 5:** Rule-Based Information Extraction & Local Policy Store (`policies.csv`).
- **Phase 6:** Multimodal Fusion — Embedding concatenation baseline & ablation comparison table.
- **Phase 7:** LangGraph AI Agent — StateGraph orchestrating modalities, models, policy, and LLM fallback.
- **Phase 8:** Interactive Demo UI (Streamlit/Gradio) & End-to-End Test Suite.
- **Phase 9:** Academic Course Delivery Reports (NLP, Vision/DL, Agent) & Reproducibility Package.

---

## Testing & Reproducibility Rules
- Use fixed seed (`PROJECT_SEED=42`) across Python `random`, `numpy`, and `torch`.
- Split before any text noise injection or image augmentation (strictly zero data leakage).
- All evaluations must report test-set metrics frozen after training.
- Unit and integration tests must run without external API calls (mock LLM adapter in CI).
- Preserve existing user work and do not overwrite verified artifacts without explicit reason.
