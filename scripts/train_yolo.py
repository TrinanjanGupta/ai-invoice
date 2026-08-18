"""
scripts/train_yolo.py

Fine-tune YOLOv8 on your annotated invoice dataset.

Usage:
    python scripts/train_yolo.py \
        --data data/annotations/dataset.yaml \
        --epochs 100 \
        --imgsz 1024 \
        --batch 8 \
        --device cpu        # or 0 for GPU

Prerequisites:
    1. Annotate invoices using Label Studio (bounding boxes)
    2. Export in YOLO format → data/annotations/
    3. Create dataset.yaml (see below)

dataset.yaml format:
    path: /absolute/path/to/data/annotations
    train: images/train
    val: images/val
    nc: 8
    names:
      0: header
      1: vendor_block
      2: buyer_block
      3: line_items
      4: totals_block
      5: tax_block
      6: payment_terms
      7: qr_barcode
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on invoice dataset")
    parser.add_argument("--data", required=True, help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="cpu or 0 (GPU index)")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="Base model: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium)")
    parser.add_argument("--output", default="data/models/invoice_yolo.pt")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Dataset not found: {data_path}")
        print("\nExpected structure:")
        print("  data/annotations/")
        print("  ├── dataset.yaml")
        print("  ├── images/")
        print("  │   ├── train/   ← invoice images")
        print("  │   └── val/")
        print("  └── labels/")
        print("      ├── train/   ← YOLO .txt label files")
        print("      └── val/")
        sys.exit(1)

    print(f"Loading base model: {args.model}")
    model = YOLO(args.model)

    print(f"\nStarting training:")
    print(f"  Dataset:  {args.data}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Image sz: {args.imgsz}")
    print(f"  Batch:    {args.batch}")
    print(f"  Device:   {args.device}")

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        save=True,
        project="data/runs",
        name="invoice_yolo",
        # Augmentation — important for invoice diversity
        hsv_h=0.01,       # slight hue shift (handles different paper colours)
        hsv_s=0.3,        # saturation variation
        hsv_v=0.3,        # brightness variation (scanned vs digital)
        degrees=5.0,      # small rotation (residual skew)
        translate=0.1,
        scale=0.3,
        flipud=0.0,       # invoices are always portrait — no vertical flip
        fliplr=0.0,       # no horizontal flip either
        mosaic=0.5,
        copy_paste=0.1,
    )

    # Copy best weights to configured path
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if not best_weights.exists():
        # Fallback search
        found = list(Path("runs").rglob("best.pt")) + list(Path("data/runs").rglob("best.pt"))
        if found:
            best_weights = found[-1]

    if best_weights.exists():
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(best_weights, output)
        print(f"\n[OK] Best weights saved to: {output}")
        print(f"  Update YOLO_MODEL_PATH={output} in your .env")
    else:
        print(f"\nWARNING: best.pt not found in {results.save_dir}")

    # Print results summary
    print(f"\nFinal metrics:")
    print(f"  mAP50:    {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    print("\nTarget: mAP50 > 0.85 before using in production.")
    print("If below target, add more annotated samples (aim for 500+ diverse invoices).")


if __name__ == "__main__":
    main()
