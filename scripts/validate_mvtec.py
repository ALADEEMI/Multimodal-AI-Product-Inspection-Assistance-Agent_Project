"""MVTec AD Dataset Validator and Manifest Generator.

Validates directory structure and creates manifest CSV for further processing.
Follows ADR-001 and Phase 0 specifications.
"""

import csv
import hashlib
from pathlib import Path
from typing import Dict, List
import sys
from dataclasses import dataclass

sys.path.append(str(Path(__file__).parent.parent))
from src.config import (
    MVTEC_DATA_PATH,
    MANIFEST_CSV_PATH,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    MVTEC_CATEGORY
)


@dataclass
class ValidationResult:
    """Result of MVTec validation."""
    valid: bool
    counts_by_split: Dict[str, int]
    missing_masks: List[str]
    errors: List[str]


def validate_mvtec(data_path: Path) -> ValidationResult:
    """Validate MVTec AD dataset structure.

    Args:
        data_path: Path to MVTec category directory (e.g., data/raw/mvtec/bottle/bottle/)

    Returns:
        ValidationResult with validation status and details
    """
    errors = []
    missing_masks = []
    counts = {"train": 0, "test": 0, "masks": 0}

    # Check train/good
    train_good_dir = data_path / "train" / "good"
    if not train_good_dir.exists():
        errors.append(f"Missing train/good directory: {train_good_dir}")
    else:
        train_images = list(train_good_dir.glob("*.png"))
        counts["train"] = len(train_images)
        if counts["train"] == 0:
            errors.append(f"No images found in train/good: {train_good_dir}")

    # Check test directories
    test_dir = data_path / "test"
    if not test_dir.exists():
        errors.append(f"Missing test directory: {test_dir}")
    else:
        test_subdirs = [d for d in test_dir.iterdir() if d.is_dir()]
        if not test_subdirs:
            errors.append(f"No test subdirectories found in: {test_dir}")
        else:
            for subdir in test_subdirs:
                test_images = list(subdir.glob("*.png"))
                counts["test"] += len(test_images)

    # Check ground_truth for defect types (not for 'good')
    gt_dir = data_path / "ground_truth"
    if gt_dir.exists():
        gt_subdirs = [d for d in gt_dir.iterdir() if d.is_dir()]
        for gt_subdir in gt_subdirs:
            masks = list(gt_subdir.glob("*.png"))
            counts["masks"] += len(masks)

            # Check mask naming convention
            defect_type = gt_subdir.name
            test_defect_dir = test_dir / defect_type
            if test_defect_dir.exists():
                test_images = list(test_defect_dir.glob("*.png"))
                for img in test_images:
                    expected_mask = gt_subdir / f"{img.stem}_mask.png"
                    if not expected_mask.exists():
                        missing_masks.append(str(expected_mask))

    valid = len(errors) == 0 and len(missing_masks) == 0

    return ValidationResult(
        valid=valid,
        counts_by_split=counts,
        missing_masks=missing_masks,
        errors=errors
    )


def generate_sample_id(image_path: Path, root: Path) -> str:
    """Generate unique sample_id from image path."""
    relative_path = image_path.relative_to(root)
    # Create hash from path for stable unique ID
    hash_obj = hashlib.md5(str(relative_path).encode())
    return f"{relative_path.parent.name}_{image_path.stem}_{hash_obj.hexdigest()[:8]}"


def create_manifest(data_path: Path, output_csv: Path) -> None:
    """Create manifest CSV from MVTec structure.

    Manifest schema: sample_id, image_path, split_source, defect_type, is_defective, mask_path
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    records = []

    # Process train/good
    train_good_dir = data_path / "train" / "good"
    if train_good_dir.exists():
        for img_path in sorted(train_good_dir.glob("*.png")):
            rel_path = img_path.relative_to(PROJECT_ROOT)
            sample_id = generate_sample_id(img_path, data_path)
            records.append({
                "sample_id": sample_id,
                "image_path": str(rel_path).replace("\\", "/"),
                "split_source": "train",
                "defect_type": "good",
                "is_defective": False,
                "mask_path": ""
            })

    # Process test directories
    test_dir = data_path / "test"
    gt_dir = data_path / "ground_truth"

    if test_dir.exists():
        for defect_subdir in sorted(test_dir.iterdir()):
            if not defect_subdir.is_dir():
                continue

            defect_type = defect_subdir.name
            is_defective = (defect_type != "good")

            for img_path in sorted(defect_subdir.glob("*.png")):
                rel_path = img_path.relative_to(PROJECT_ROOT)
                sample_id = generate_sample_id(img_path, data_path)

                # Find corresponding mask
                mask_path = ""
                if is_defective and gt_dir.exists():
                    expected_mask = gt_dir / defect_type / f"{img_path.stem}_mask.png"
                    if expected_mask.exists():
                        mask_path = str(expected_mask.relative_to(PROJECT_ROOT)).replace("\\", "/")

                records.append({
                    "sample_id": sample_id,
                    "image_path": str(rel_path).replace("\\", "/"),
                    "split_source": "test",
                    "defect_type": defect_type,
                    "is_defective": is_defective,
                    "mask_path": mask_path
                })

    # Write CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["sample_id", "image_path", "split_source", "defect_type",
                     "is_defective", "mask_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"[OK] Manifest created: {output_csv}")
    print(f"  Total records: {len(records)}")
    print(f"  Train source: {sum(1 for r in records if r['split_source'] == 'train')}")
    print(f"  Test source: {sum(1 for r in records if r['split_source'] == 'test')}")
    print(f"  Defective: {sum(1 for r in records if r['is_defective'])}")


def main():
    """Run validation and create manifest."""
    print(f"Validating MVTec AD dataset at: {MVTEC_DATA_PATH}")

    if not MVTEC_DATA_PATH.exists():
        print(f"[ERROR] Dataset directory not found: {MVTEC_DATA_PATH}")
        sys.exit(1)

    # Validate
    result = validate_mvtec(MVTEC_DATA_PATH)

    if not result.valid:
        print(f"[ERROR] Validation FAILED")
        if result.errors:
            print("\nErrors:")
            for err in result.errors:
                print(f"  - {err}")
        if result.missing_masks:
            print(f"\nMissing masks ({len(result.missing_masks)}):")
            for mask in result.missing_masks[:5]:
                print(f"  - {mask}")
        sys.exit(1)

    print(f"[OK] Validation PASSED for category '{MVTEC_CATEGORY}'")
    print(f"  Train images: {result.counts_by_split['train']}")
    print(f"  Test images: {result.counts_by_split['test']}")
    print(f"  Masks: {result.counts_by_split['masks']}")

    # Create manifest
    print(f"\nCreating manifest...")
    create_manifest(MVTEC_DATA_PATH, MANIFEST_CSV_PATH)

    print(f"\n[OK] Phase 0 validation complete!")


if __name__ == "__main__":
    main()
