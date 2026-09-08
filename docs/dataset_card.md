# Dataset Card: Synthetic Paired MVTec Bottle Dataset

## 1. Summary & Motivation
- **Dataset Name:** MVTec Bottle Multimodal Paired Dataset (`paired_dataset.csv`)
- **Version:** `v1.0.0`
- **Base Visual Dataset:** [MVTec Anomaly Detection (MVTec AD)](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) — Category `bottle`
- **Visual License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
- **Core Purpose:** Academic research solving the challenge of training multimodal models without requiring external product APIs. MVTec AD provides ground-truth defect masks but lacks natural language customer complaints. This dataset pairs every visual sample with a programmatically generated, semantically aligned customer complaint text.

---

## 2. Dataset Schema (11-Field Contract)
Each record in `data/processed/paired_dataset.csv` adheres to the strict contract:

| Field | Type | Description |
|---|---|---|
| `sample_id` | `string` | Deterministic unique sample identifier derived from file path |
| `image_path` | `string` | Relative path from project root to image PNG |
| `generated_text` | `string` | Synthetic complaint text (English/Arabic) with controlled noise |
| `is_complaint` | `boolean` | `true` if defective; `false` if good / normal |
| `problem_type` | `enum` | One of `{Quality, Packaging, Price, Delivery, Damage, Other}` |
| `defect_type` | `string` | MVTec label (`broken_large`, `broken_small`, `contamination`, `good`) |
| `severity` | `enum` | One of `{Low, Medium, High}` |
| `mask_path` | `string/null` | Path to ground-truth binary segmentation mask (null for good) |
| `split` | `enum` | One of `{train, validation, test}` (zero leakage) |
| `generator_version`| `string` | Generator version tag (`v1.0.0`) |
| `seed` | `integer` | Base seed used for reproducibility (`PROJECT_SEED=42`) |

---

## 3. Defect Mapping & Class Distribution

| Defect Type | Split Count (Train / Val / Test) | Target `is_complaint` | Target `problem_type` | Target `severity` |
|---|---|---|---|---|
| `good` | 160 / 34 / 35 | `False` | `Other` | `Low` |
| `broken_large` | 14 / 3 / 3 | `True` | `Damage` | `High` |
| `broken_small` | 16 / 3 / 3 | `True` | `Damage` | `Medium` |
| `contamination` | 14 / 4 / 3 | `True` | `Quality` | `Medium` |
| **Total** | **204 / 44 / 44 (292 Total)** | — | — | — |

---

## 4. Text Generation & Dirty Data Simulation
- **No LLM Generation:** Texts are generated deterministically using template banks and synonym permutations to ensure exact reproducibility.
- **Bilingual Support:** Contains both Arabic (e.g., `"الزجاجة مكسورة بشكل كبير"`) and English (e.g., `"The bottle is severely broken"`).
- **Dirty Data Noise Injection:**
  - Typographical character swaps based on keyboard proximity maps.
  - Character repetition (e.g., `"soooo damaged!!!!"`).
  - Emoji insertions (😠, 💔, ⚠️, 📦).
  - Punctuation removal or excessive punctuation.
  - Casing variations.

---

## 5. Leakage Prevention & Quality Assurance
1. **Pre-Augmentation Splitting:** Train, validation, and test sets are split before applying any image augmentation or text noise.
2. **Deterministic Hashing:** Per-sample seeds are derived from `MD5(base_seed + sample_id)` to guarantee that re-running generation produces byte-for-byte identical datasets.
3. **Manual Review:** A documented 20-sample manual review is saved in `docs/dataset_review_20_samples.md`.
