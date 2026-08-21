"""
scripts/train_yolo.py

Fine-tune YOLOv8 on invoice region detection dataset.
Aligned with latest Invoice Builder layout regions & FastAPI schema.

Usage:
    python scripts/train_yolo.py \
        --data data/annotations/dataset.yaml \
        --epochs 80 \
        --imgsz 1024 \
        --batch 8 \
        --device cpu

Classes (nc: 8):
    0: header          - Invoice title, number, dates, PO, Place of Supply
    1: vendor_block    - Seller / Biller (name, address, GSTIN, PAN, contacts)
    2: buyer_block     - Buyer / Client / Beneficiary details
    3: line_items      - Main items table (HSN, quantities, units, rates, taxes, amounts)
    4: totals_block    - Subtotals, discounts, taxes, round off, grand total, amount in words
    5: tax_block       - GST / tax summary breakdown rates & amounts
    6: payment_terms   - Bank details, branch, IFSC, payment terms, remarks & declarations
    7: qr_barcode      - E-invoice QR code, UPI QR, or barcodes
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
import yaml

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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


def parse_args():
    default_base_model = (
        "data/models/doclayout_yolo_v8s/weights/best.pt"
        if Path("data/models/doclayout_yolo_v8s/weights/best.pt").exists()
        else "yolov8n.pt"
    )
    parser = argparse.ArgumentParser(description="Fine-tune YOLO / DocLayout on invoice layout dataset")
    parser.add_argument("--data", default="data/annotations/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--imgsz", "--img-size", dest="imgsz", type=int, default=1024, help="Image resolution for training")
    parser.add_argument("--batch", "--batch-size", dest="batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", default="", help="Device: '0' for CUDA GPU, 'cpu' for CPU (default: auto)")
    parser.add_argument("--model", default=default_base_model, help="Base model checkpoint (default: DocLayout-YOLO pretrained weights)")
    parser.add_argument("--output", "--output-dir", "--output_dir", dest="output", default="data/models/invoice_yolo.pt", help="Path to save best weights")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience")
    return parser.parse_args()


def ensure_dataset_yaml(data_path: Path) -> Path:
    """Ensure dataset.yaml exists and has valid formatted paths."""
    annotations_dir = data_path.parent if data_path.suffix.lower() == ".yaml" else data_path
    yaml_file = annotations_dir / "dataset.yaml" if annotations_dir.is_dir() else data_path

    if not yaml_file.exists():
        train_img_dir = annotations_dir / "images" / "train"
        val_img_dir = annotations_dir / "images" / "val"

        if not train_img_dir.exists() or not list(train_img_dir.glob("*")):
            print(f"No YOLO annotations found at {annotations_dir}. Attempting auto-annotation from pipeline...")
            import subprocess
            subprocess.run([sys.executable, "scripts/auto_annotate_from_pipeline.py"], check=False)

        # Create dataset.yaml
        yaml_content = {
            "path": str(annotations_dir.resolve()).replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "nc": len(CLASSES),
            "names": {i: name for i, name in enumerate(CLASSES)},
        }
        yaml_file.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, sort_keys=False)
        print(f"Created dataset.yaml at {yaml_file}")

    return yaml_file


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
        import torch
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    data_yaml = ensure_dataset_yaml(Path(args.data))
    if not data_yaml.exists():
        print(f"ERROR: Dataset YAML not found: {data_yaml}")
        sys.exit(1)

    # Determine device
    device = args.device
    if not device:
        device = "0" if torch.cuda.is_available() else "cpu"

    print(f"\n========================================================")
    print(f"Loading Base YOLOv8 Model: {args.model}")
    print(f"Dataset YAML: {data_yaml}")
    print(f"Target Classes ({len(CLASSES)}): {CLASSES}")
    print(f"Device: {device}")
    print(f"========================================================\n")

    model = YOLO(args.model)

    # Train with document-optimized hyper-parameters
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        patience=args.patience,
        save=True,
        project="data/runs",
        name="invoice_yolo",
        exist_ok=True,
        # Document layout augmentations
        degrees=3.0,          # slight rotation for scanned document deskew
        translate=0.08,       # slight translation
        scale=0.25,           # small scale variation
        shear=1.0,            # slight perspective distortion
        perspective=0.0001,
        flipud=0.0,           # documents are never inverted
        fliplr=0.0,           # text is never mirrored
        mosaic=0.4,           # modest mosaic augmentation
        mixup=0.0,            # don't blend multiple invoices into one
        hsv_h=0.015,          # paper background tint shift
        hsv_s=0.25,           # saturation variance
        hsv_v=0.25,           # brightness contrast (handles dark/light scans)
        half=(device != "cpu" and torch.cuda.is_available()),
    )

    # Locate best.pt weights
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path("data/runs/invoice_yolo")
    best_weights = save_dir / "weights" / "best.pt"

    if not best_weights.exists():
        # Fallback search in runs
        found = list(Path("data/runs").rglob("best.pt")) + list(Path("runs").rglob("best.pt"))
        if found:
            best_weights = found[-1]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if best_weights.exists():
        shutil.copy2(best_weights, output_path)
        print(f"\n[OK] Best YOLOv8 model weights saved to: {output_path}")
        print(f"  Set YOLO_MODEL_PATH={output_path} in your .env or API config")
    else:
        print(f"\nWARNING: best.pt weights file not located in {save_dir}")

    # Export training metadata
    metrics_dict = getattr(results, "results_dict", {}) or {}
    map50 = metrics_dict.get("metrics/mAP50(B)", 0.0)
    map50_95 = metrics_dict.get("metrics/mAP50-95(B)", 0.0)

    metadata = {
        "model_type": "yolov8",
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": args.epochs,
        "classes": CLASSES,
        "num_classes": len(CLASSES),
        "mAP50": float(map50) if isinstance(map50, (int, float)) else 0.0,
        "mAP50_95": float(map50_95) if isinstance(map50_95, (int, float)) else 0.0,
        "output_path": str(output_path),
    }

    meta_file = output_path.parent / "yolo_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nTraining Results Summary:")
    print(f"  mAP50:    {metadata['mAP50']:.4f}" if isinstance(map50, (int, float)) else f"  mAP50: {map50}")
    print(f"  mAP50-95: {metadata['mAP50_95']:.4f}" if isinstance(map50_95, (int, float)) else f"  mAP50-95: {map50_95}")
    print(f"  Metadata saved: {meta_file}\n")


if __name__ == "__main__":
    main()
