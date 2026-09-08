"""Paired Dataset Generator for Multimodal Product Inspection.

Generates synthetic complaint text deterministically paired with MVTec images.
Follows ADR-001 and Phase 1 specifications.
"""

import csv
import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import (
    MANIFEST_CSV_PATH,
    PAIRED_DATASET_CSV_PATH,
    PROJECT_SEED,
    GENERATOR_VERSION,
    DEFECT_MAPPING,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    PROJECT_ROOT,
)
from src.data.templates import get_template
from src.data.noise_injection import inject_noise


def derive_seed_from_sample_id(sample_id: str, base_seed: int = PROJECT_SEED) -> int:
    """Generate deterministic seed from sample_id for reproducibility."""
    hash_obj = hashlib.md5(f"{sample_id}_{base_seed}".encode())
    return int(hash_obj.hexdigest()[:8], 16)


def assign_split(
    sample_idx: int,
    split_source: str,
    rng: random.Random
) -> str:
    """Assign train/val/test split.

    Train source images stay as train split.
    Test source images are further split into val/test according to ratios.
    """
    if split_source == "train":
        return "train"
    else:
        # Split test source into val/test
        r = rng.random()
        if r < (TEST_RATIO / (VAL_RATIO + TEST_RATIO)):
            return "test"
        else:
            return "validation"


def generate_paired_dataset(manifest_path: Path, output_path: Path) -> None:
    """Generate paired dataset CSV from manifest.

    Output schema:
        sample_id, image_path, generated_text, is_complaint, problem_type,
        defect_type, severity, mask_path, split, generator_version, seed
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read manifest
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        manifest_records = list(reader)

    print(f"Loaded {len(manifest_records)} records from manifest")

    # Generate paired records
    paired_records = []
    rng_split = random.Random(PROJECT_SEED)

    for idx, record in enumerate(manifest_records):
        sample_id = record["sample_id"]
        image_path = record["image_path"]
        split_source = record["split_source"]
        defect_type = record["defect_type"]
        is_defective = record["is_defective"].lower() == "true"
        mask_path = record["mask_path"]

        # Derive seed for this sample
        sample_seed = derive_seed_from_sample_id(sample_id, PROJECT_SEED)
        rng_sample = random.Random(sample_seed)

        # Get defect mapping
        if defect_type not in DEFECT_MAPPING:
            print(f"Warning: Unknown defect_type '{defect_type}', treating as 'good'")
            defect_type = "good"

        mapping = DEFECT_MAPPING[defect_type]
        is_complaint = mapping["is_complaint"]
        problem_type = mapping["problem_type"]
        severity = mapping["severity"]

        # Generate text from template
        template = get_template(defect_type, language="mixed")
        rng_sample.seed(sample_seed)  # Re-seed for template selection consistency

        # Apply noise injection
        generated_text = inject_noise(
            template,
            is_complaint=is_complaint,
            noise_rate=0.3,
            rng=rng_sample
        )

        # Assign final split (train/validation/test)
        final_split = assign_split(idx, split_source, rng_split)

        paired_records.append({
            "sample_id": sample_id,
            "image_path": image_path,
            "generated_text": generated_text,
            "is_complaint": is_complaint,
            "problem_type": problem_type,
            "defect_type": defect_type,
            "severity": severity,
            "mask_path": mask_path,
            "split": final_split,
            "generator_version": GENERATOR_VERSION,
            "seed": sample_seed,
        })

    # Write paired dataset CSV
    fieldnames = [
        "sample_id", "image_path", "generated_text", "is_complaint",
        "problem_type", "defect_type", "severity", "mask_path",
        "split", "generator_version", "seed"
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(paired_records)

    # Print statistics
    train_count = sum(1 for r in paired_records if r["split"] == "train")
    val_count = sum(1 for r in paired_records if r["split"] == "validation")
    test_count = sum(1 for r in paired_records if r["split"] == "test")
    complaint_count = sum(1 for r in paired_records if r["is_complaint"])

    print(f"\n[OK] Paired dataset created: {output_path}")
    print(f"  Total records: {len(paired_records)}")
    print(f"  Train: {train_count}")
    print(f"  Validation: {val_count}")
    print(f"  Test: {test_count}")
    print(f"  Complaints: {complaint_count}")

    # Problem type distribution
    problem_dist = {}
    for r in paired_records:
        pt = r["problem_type"]
        problem_dist[pt] = problem_dist.get(pt, 0) + 1

    print("\n  Problem type distribution:")
    for pt, count in sorted(problem_dist.items()):
        pct = (count / len(paired_records)) * 100
        print(f"    {pt}: {count} ({pct:.1f}%)")

    # Save distribution to JSON
    stats_path = output_path.parent / "paired_dataset_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({
            "total": len(paired_records),
            "train": train_count,
            "validation": val_count,
            "test": test_count,
            "complaints": complaint_count,
            "problem_distribution": problem_dist,
            "generator_version": GENERATOR_VERSION,
            "base_seed": PROJECT_SEED,
        }, f, indent=2)

    print(f"\n  Statistics saved: {stats_path}")


def validate_paired_dataset(dataset_path: Path) -> bool:
    """Validate paired dataset for correctness."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        records = list(reader)

    errors = []

    # Check for duplicate sample_ids
    sample_ids = [r["sample_id"] for r in records]
    if len(sample_ids) != len(set(sample_ids)):
        errors.append("Duplicate sample_ids found")

    # Check all required fields
    required_fields = [
        "sample_id", "image_path", "generated_text", "is_complaint",
        "problem_type", "defect_type", "severity", "split"
    ]
    for idx, record in enumerate(records):
        for field in required_fields:
            if not record.get(field):
                errors.append(f"Row {idx}: Missing field '{field}'")
                break

        # Check for mask_path consistency
        is_defective = (record["defect_type"] != "good")
        has_mask = bool(record.get("mask_path"))

        if is_defective and not has_mask:
            errors.append(f"Row {idx}: Defective image missing mask_path")
        if not is_defective and has_mask:
            errors.append(f"Row {idx}: Good image should not have mask_path")

    if errors:
        print(f"[ERROR] Validation failed with {len(errors)} errors:")
        for err in errors[:10]:
            print(f"  - {err}")
        return False
    else:
        print(f"[OK] Validation passed: {len(records)} records")
        return True


def main():
    """Generate and validate paired dataset."""
    print(f"Generating paired dataset from: {MANIFEST_CSV_PATH}")

    if not MANIFEST_CSV_PATH.exists():
        print(f"[ERROR] Manifest not found: {MANIFEST_CSV_PATH}")
        print("Please run scripts/validate_mvtec.py first")
        sys.exit(1)

    generate_paired_dataset(MANIFEST_CSV_PATH, PAIRED_DATASET_CSV_PATH)

    print(f"\nValidating paired dataset...")
    if not validate_paired_dataset(PAIRED_DATASET_CSV_PATH):
        sys.exit(1)

    print(f"\n[OK] Phase 1 complete: Paired dataset generated and validated")


if __name__ == "__main__":
    main()
