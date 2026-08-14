"""
Prepare and split YOLO dataset exported from Label Studio.

Usage:
    python scripts/prepare_yolo_dataset.py --zip-path "C:/path/to/downloaded_export.zip"
    or
    python scripts/prepare_yolo_dataset.py --export-dir "data/exported_yolo"
"""

import argparse
import os
import shutil
import zipfile
import random
import yaml
from pathlib import Path
from loguru import logger

CLASSES = [
    "header",
    "vendor_block",
    "buyer_block",
    "line_items",
    "totals_block",
    "tax_block",
    "payment_terms",
    "qr_barcode",
]


def prepare_dataset(zip_path: str = None, export_dir: str = None, output_dir: str = "data/annotations", val_ratio: float = 0.2):
    output_path = Path(output_dir).resolve()
    temp_extract = output_path / "temp_raw"
    temp_extract.mkdir(parents=True, exist_ok=True)

    if zip_path and Path(zip_path).exists():
        logger.info(f"Extracting ZIP: {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(temp_extract)
    elif export_dir and Path(export_dir).exists():
        temp_extract = Path(export_dir)
    else:
        logger.error("Please provide a valid --zip-path or --export-dir")
        return

    # Look for images and labels in extracted content
    images_src = None
    labels_src = None

    for root, dirs, files in os.walk(temp_extract):
        p = Path(root)
        if p.name.lower() == "images":
            images_src = p
        elif p.name.lower() == "labels":
            labels_src = p

    # If flat directory structure, search directly
    if not images_src or not labels_src:
        all_imgs = [p for p in temp_extract.rglob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]]
        all_lbls = [p for p in temp_extract.rglob("*.txt") if p.name != "classes.txt"]
    else:
        all_imgs = [p for p in images_src.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]]
        all_lbls = [p for p in labels_src.glob("*.txt") if p.name != "classes.txt"]

    logger.info(f"Found {len(all_imgs)} images and {len(all_lbls)} label files")

    if not all_imgs:
        logger.error("No images found in export!")
        return

    # Map labels by stem
    label_map = {lbl.stem: lbl for lbl in all_lbls}

    # Match image pairs
    valid_pairs = []
    for img in all_imgs:
        if img.stem in label_map:
            valid_pairs.append((img, label_map[img.stem]))
        else:
            logger.warning(f"No label found for image: {img.name} (skipping)")

    logger.info(f"Total matched annotated samples: {len(valid_pairs)}")

    if not valid_pairs:
        logger.error("No matched image-label pairs found!")
        return

    # Clean and create directory structure
    for split in ["train", "val"]:
        (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Shuffle and split
    random.seed(42)
    random.shuffle(valid_pairs)
    val_count = max(1, int(len(valid_pairs) * val_ratio))
    val_pairs = valid_pairs[:val_count]
    train_pairs = valid_pairs[val_count:]

    logger.info(f"Split: {len(train_pairs)} training samples, {len(val_pairs)} validation samples")

    for img, lbl in train_pairs:
        shutil.copy2(img, output_path / "images" / "train" / img.name)
        shutil.copy2(lbl, output_path / "labels" / "train" / lbl.name)

    for img, lbl in val_pairs:
        shutil.copy2(img, output_path / "images" / "val" / img.name)
        shutil.copy2(lbl, output_path / "labels" / "val" / lbl.name)

    # Cleanup temp
    if (output_path / "temp_raw").exists():
        shutil.rmtree(output_path / "temp_raw", ignore_errors=True)

    # Generate dataset.yaml
    dataset_yaml = {
        "path": str(output_path).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASSES),
        "names": {i: name for i, name in enumerate(CLASSES)},
    }

    yaml_file = output_path / "dataset.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(dataset_yaml, f, sort_keys=False)

    logger.info(f"Dataset successfully prepared at: {output_path}")
    logger.info(f"dataset.yaml created at: {yaml_file}")
    print("\n" + "="*60)
    print("READY TO TRAIN! Run this command:")
    print(f"python scripts/train_yolo.py --data {yaml_file} --epochs 100")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from Label Studio export")
    parser.add_argument("--zip-path", type=str, help="Path to exported ZIP file from Label Studio")
    parser.add_argument("--export-dir", type=str, help="Path to extracted export folder")
    parser.add_argument("--output-dir", type=str, default="data/annotations", help="Destination annotations folder")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation set ratio (default: 0.2)")
    args = parser.parse_args()

    prepare_dataset(args.zip_path, args.export_dir, args.output_dir, args.val_ratio)
