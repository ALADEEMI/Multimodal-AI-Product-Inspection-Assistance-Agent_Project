"""
MVTec AD Dataset Validator and Manifest Generator.
Phase 0: Foundation - Validates dataset structure and generates manifest.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

from src.config import (
    MVTEC_DATA_PATH,
    MVTEC_CATEGORY,
    MANIFEST_CSV_PATH,
    PROJECT_ROOT,
)
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class MVTecSample:
    """Represents a single MVTec sample."""
    sample_id: str
    image_path: str  # Relative to project root
    split_source: str  # 'train' or 'test'
    defect_type: str
    is_defective: bool
    mask_path: Optional[str] = None  # Relative to project root, None for good samples


class MVTecValidator:
    """Validates MVTec AD dataset structure and generates manifest."""

    def __init__(self, data_path: Path = MVTEC_DATA_PATH, category: str = MVTEC_CATEGORY):
        self.data_path = data_path
        self.category = category
        self.samples: List[MVTecSample] = []

    def validate(self) -> bool:
        """
        Validate MVTec dataset structure.

        Returns:
            True if validation passes, False otherwise
        """
        logger.info(f"Validating MVTec AD dataset at: {self.data_path}")

        # Check if base path exists
        if not self.data_path.exists():
            logger.error(f"MVTec data path does not exist: {self.data_path}")
            return False

        # Check train directory
        train_dir = self.data_path / "train" / "good"
        if not train_dir.exists() or not any(train_dir.glob("*.png")):
            logger.error(f"Train directory missing or empty: {train_dir}")
            return False

        train_count = len(list(train_dir.glob("*.png")))
        logger.info(f"Found {train_count} training images in train/good/")

        # Check test directory
        test_dir = self.data_path / "test"
        if not test_dir.exists():
            logger.error(f"Test directory missing: {test_dir}")
            return False

        # Detect defect types from test subdirectories
        defect_types = [d.name for d in test_dir.iterdir() if d.is_dir() and d.name != "good"]
        logger.info(f"Detected defect types: {defect_types}")

        if not defect_types:
            logger.error("No defect types found in test directory")
            return False

        # Validate test images and masks
        ground_truth_dir = self.data_path / "ground_truth"
        test_good_dir = test_dir / "good"

        # Count test/good images
        test_good_count = len(list(test_good_dir.glob("*.png"))) if test_good_dir.exists() else 0
        logger.info(f"Found {test_good_count} test/good images")

        for defect_type in defect_types:
            defect_test_dir = test_dir / defect_type
            defect_mask_dir = ground_truth_dir / defect_type

            if not defect_test_dir.exists():
                logger.error(f"Missing test directory for defect: {defect_type}")
                return False

            if not defect_mask_dir.exists():
                logger.error(f"Missing ground_truth directory for defect: {defect_type}")
                return False

            test_images = sorted(defect_test_dir.glob("*.png"))
            mask_images = sorted(defect_mask_dir.glob("*.png"))

            logger.info(f"  {defect_type}: {len(test_images)} test images, {len(mask_images)} masks")

            if len(test_images) != len(mask_images):
                logger.warning(
                    f"Mismatch between test images and masks for {defect_type}: "
                    f"{len(test_images)} vs {len(mask_images)}"
                )

            # Verify mask naming convention
            for test_img in test_images:
                expected_mask_name = test_img.stem + "_mask.png"
                expected_mask_path = defect_mask_dir / expected_mask_name
                if not expected_mask_path.exists():
                    logger.warning(f"Expected mask not found: {expected_mask_path}")

        logger.info("MVTec dataset validation PASSED")
        return True

    def generate_manifest(self) -> List[MVTecSample]:
        """
        Generate manifest of all samples with paths and metadata.

        Returns:
            List of MVTecSample objects
        """
        logger.info("Generating MVTec manifest...")
        self.samples = []

        # Process training samples (all good)
        train_dir = self.data_path / "train" / "good"
        for img_path in sorted(train_dir.glob("*.png")):
            relative_path = str(img_path.relative_to(PROJECT_ROOT))
            sample_id = f"train_good_{img_path.stem}"

            sample = MVTecSample(
                sample_id=sample_id,
                image_path=relative_path,
                split_source="train",
                defect_type="good",
                is_defective=False,
                mask_path=None
            )
            self.samples.append(sample)

        # Process test samples (good)
        test_good_dir = self.data_path / "test" / "good"
        if test_good_dir.exists():
            for img_path in sorted(test_good_dir.glob("*.png")):
                relative_path = str(img_path.relative_to(PROJECT_ROOT))
                sample_id = f"test_good_{img_path.stem}"

                sample = MVTecSample(
                    sample_id=sample_id,
                    image_path=relative_path,
                    split_source="test",
                    defect_type="good",
                    is_defective=False,
                    mask_path=None
                )
                self.samples.append(sample)

        # Process test samples (defects)
        test_dir = self.data_path / "test"
        ground_truth_dir = self.data_path / "ground_truth"

        defect_types = [d.name for d in test_dir.iterdir() if d.is_dir() and d.name != "good"]

        for defect_type in defect_types:
            defect_test_dir = test_dir / defect_type
            defect_mask_dir = ground_truth_dir / defect_type

            for img_path in sorted(defect_test_dir.glob("*.png")):
                relative_img_path = str(img_path.relative_to(PROJECT_ROOT))
                sample_id = f"test_{defect_type}_{img_path.stem}"

                # Find corresponding mask
                mask_name = img_path.stem + "_mask.png"
                mask_path = defect_mask_dir / mask_name

                relative_mask_path = None
                if mask_path.exists():
                    relative_mask_path = str(mask_path.relative_to(PROJECT_ROOT))
                else:
                    logger.warning(f"Mask not found for {img_path.name}, expected: {mask_path}")

                sample = MVTecSample(
                    sample_id=sample_id,
                    image_path=relative_img_path,
                    split_source="test",
                    defect_type=defect_type,
                    is_defective=True,
                    mask_path=relative_mask_path
                )
                self.samples.append(sample)

        logger.info(f"Generated manifest with {len(self.samples)} samples")
        return self.samples

    def save_manifest(self, output_path: Path = MANIFEST_CSV_PATH):
        """Save manifest to CSV file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            if self.samples:
                fieldnames = list(asdict(self.samples[0]).keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for sample in self.samples:
                    writer.writerow(asdict(sample))

        logger.info(f"Manifest saved to: {output_path}")


def main():
    """Run MVTec validation and manifest generation."""
    from src.utils.seed_utils import set_seed

    set_seed()

    validator = MVTecValidator()

    if not validator.validate():
        logger.error("MVTec validation failed. Exiting.")
        return False

    validator.generate_manifest()
    validator.save_manifest()

    logger.info("Phase 0 MVTec validation and manifest generation complete.")
    return True


if __name__ == "__main__":
    main()
