"""
Synthetic Paired Dataset Generator.
Phase 1: Paired Dataset - Combines MVTec samples with generated text,
applies controlled noise, and creates stratified train/val/test splits.
"""

import csv
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    MANIFEST_CSV_PATH,
    PAIRED_DATASET_CSV_PATH,
    DEFECT_MAPPING,
    PROJECT_SEED,
    GENERATOR_VERSION,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    DOCS_DIR,
)
from src.data.text_templates import get_all_templates, ACTION_KEYWORDS_EN, ACTION_KEYWORDS_AR
from src.data.noise_injector import NoiseInjector
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class PairedRecord:
    """11-field data contract as specified in ADR-001."""
    sample_id: str
    image_path: str
    generated_text: str
    is_complaint: bool
    problem_type: str
    defect_type: str
    severity: str
    mask_path: Optional[str]
    split: str
    generator_version: str
    seed: int


class PairedDatasetGenerator:
    """Generates synthetic paired multimodal dataset from MVTec manifest."""

    def __init__(
        self,
        manifest_path: Path = MANIFEST_CSV_PATH,
        seed: int = PROJECT_SEED,
        version: str = GENERATOR_VERSION,
        noise_rate: float = 0.35,
    ):
        self.manifest_path = manifest_path
        self.seed = seed
        self.version = version
        self.noise_rate = noise_rate
        self.templates = get_all_templates()
        self.records: List[PairedRecord] = []

    def _sample_seed(self, sample_id: str) -> int:
        """Derive a deterministic integer seed from sample_id and base seed."""
        hash_bytes = hashlib.md5(f"{self.seed}_{sample_id}".encode()).digest()
        return int.from_bytes(hash_bytes[:4], byteorder="big")

    def _generate_text_for_sample(self, defect_type: str, sample_id: str) -> str:
        """Generate text deterministically for a single sample."""
        sample_seed = self._sample_seed(sample_id)
        rng = random.Random(sample_seed)

        templates_list = self.templates.get(defect_type, self.templates["good"])
        base_template = rng.choice(templates_list)

        # 25% chance to append a requested action for complaints
        if defect_type != "good" and rng.random() < 0.25:
            # Decide English vs Arabic based on base text
            is_arabic = any('؀' <= c <= 'ۿ' for c in base_template)
            if is_arabic:
                action_type = rng.choice(list(ACTION_KEYWORDS_AR.keys()))
                action_word = rng.choice(ACTION_KEYWORDS_AR[action_type])
                base_template = f"{base_template}. أرجو {action_word}"
            else:
                action_type = rng.choice(list(ACTION_KEYWORDS_EN.keys()))
                action_word = rng.choice(ACTION_KEYWORDS_EN[action_type])
                base_template = f"{base_template}. Please {action_word} this"

        # Apply controlled noise
        injector = NoiseInjector(noise_rate=self.noise_rate, seed=sample_seed)
        return injector.inject_noise(base_template)

    def generate(self) -> List[PairedRecord]:
        """
        Generate paired dataset with train/val/test splits.

        Returns:
            List of PairedRecord objects
        """
        logger.info(f"Loading manifest from: {self.manifest_path}")

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        df_manifest = pd.read_csv(self.manifest_path)
        logger.info(f"Loaded {len(df_manifest)} samples from manifest")

        # First, generate records without split assignment
        temp_records = []
        for _, row in df_manifest.iterrows():
            defect_type = row["defect_type"]
            mapping = DEFECT_MAPPING.get(defect_type, DEFECT_MAPPING["good"])

            text = self._generate_text_for_sample(defect_type, row["sample_id"])

            temp_records.append({
                "sample_id": row["sample_id"],
                "image_path": row["image_path"],
                "generated_text": text,
                "is_complaint": mapping["is_complaint"],
                "problem_type": mapping["problem_type"],
                "defect_type": defect_type,
                "severity": mapping["severity"],
                "mask_path": row["mask_path"] if pd.notna(row["mask_path"]) else None,
                "split_source": row["split_source"],
                "generator_version": self.version,
                "seed": self.seed,
            })

        df_temp = pd.DataFrame(temp_records)

        # Stratified train/val/test splitting
        # Stratify by defect_type to ensure all defect types are represented in train/val/test
        train_df, test_val_df = train_test_split(
            df_temp,
            test_size=(VAL_RATIO + TEST_RATIO),
            random_state=self.seed,
            stratify=df_temp["defect_type"]
        )

        relative_val_size = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
        val_df, test_df = train_test_split(
            test_val_df,
            test_size=(1 - relative_val_size),
            random_state=self.seed,
            stratify=test_val_df["defect_type"]
        )

        train_df["split"] = "train"
        val_df["split"] = "validation"
        test_df["split"] = "test"

        df_final = pd.concat([train_df, val_df, test_df]).sort_index()

        self.records = [
            PairedRecord(
                sample_id=row["sample_id"],
                image_path=row["image_path"],
                generated_text=row["generated_text"],
                is_complaint=row["is_complaint"],
                problem_type=row["problem_type"],
                defect_type=row["defect_type"],
                severity=row["severity"],
                mask_path=row["mask_path"],
                split=row["split"],
                generator_version=row["generator_version"],
                seed=row["seed"],
            )
            for _, row in df_final.iterrows()
        ]

        logger.info(
            f"Generated paired dataset: {len(self.records)} records "
            f"(train={len(train_df)}, val={len(val_df)}, test={len(test_df)})"
        )
        return self.records

    def save(self, output_path: Path = PAIRED_DATASET_CSV_PATH):
        """Save paired dataset to CSV."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            if self.records:
                fieldnames = list(asdict(self.records[0]).keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for rec in self.records:
                    writer.writerow(asdict(rec))

        logger.info(f"Paired dataset saved to: {output_path}")

    def generate_review_report(self, n_samples: int = 20) -> str:
        """Generate a review of random samples for validation."""
        rng = random.Random(self.seed)
        samples = rng.sample(self.records, min(n_samples, len(self.records)))

        lines = [
            f"# Paired Dataset Review Report ({n_samples} Samples)",
            f"**Generator Version:** {self.version}",
            f"**Seed:** {self.seed}",
            f"**Total Records:** {len(self.records)}",
            "",
            "| Sample ID | Split | Defect Type | Problem Type | Severity | Generated Text |",
            "|---|---|---|---|---|---|",
        ]

        for s in samples:
            text_escaped = s.generated_text.replace("|", "/")
            lines.append(
                f"| `{s.sample_id}` | {s.split} | `{s.defect_type}` | {s.problem_type} | {s.severity} | {text_escaped} |"
            )

        return "\n".join(lines)


def main():
    """Run paired dataset generation."""
    generator = PairedDatasetGenerator()
    generator.generate()
    generator.save()

    report = generator.generate_review_report(20)
    report_path = DOCS_DIR / "dataset_review_20_samples.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"Sample review report saved to: {report_path}")

    return True


if __name__ == "__main__":
    main()
