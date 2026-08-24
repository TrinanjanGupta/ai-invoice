"""
scripts/visualize_layoutlm_sample.py

Renders the token bounding boxes and BIO labels from a generated LayoutLM sample
onto the invoice image so you can visually verify alignment.

Usage:
    python scripts/visualize_layoutlm_sample.py --sample data/layoutlm_dataset/train/<job_id>.json [--output-dir output/debug_viz]
"""

import sys
import json
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LABEL_COLORS = {
    "INVOICE_NUMBER": (37, 99, 235),      # Blue
    "INVOICE_DATE": (16, 185, 129),       # Green
    "VENDOR_NAME": (217, 119, 6),        # Amber
    "BUYER_NAME": (139, 92, 246),        # Purple
    "VENDOR_GSTIN": (239, 68, 68),       # Red
    "BUYER_GSTIN": (244, 63, 94),        # Rose
    "GRAND_TOTAL": (5, 150, 105),        # Emerald
    "SUBTOTAL": (14, 165, 233),          # Sky
    "TAX_AMOUNT": (249, 115, 22),        # Orange
    "IFSC_CODE": (168, 85, 247),         # Violet
    "LINE_ITEM_DESC": (234, 179, 8),     # Yellow
    "LINE_ITEM_AMOUNT": (16, 185, 129),  # Green
}


def denormalize_box(box_1000, width, height):
    """Convert [0, 1000] normalized box to image pixel coords [x1, y1, x2, y2]."""
    x1 = int(box_1000[0] * width / 1000.0)
    y1 = int(box_1000[1] * height / 1000.0)
    x2 = int(box_1000[2] * width / 1000.0)
    y2 = int(box_1000[3] * height / 1000.0)
    return [x1, y1, x2, y2]


def visualize_sample(sample_path: Path, output_dir: Path):
    with open(sample_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_rel = Path(data["image_path"])
    img_path = sample_path.parent.parent / img_rel if not img_rel.is_absolute() else img_rel
    if not img_path.exists():
        img_path = Path("data/layoutlm_dataset") / img_rel
    if not img_path.exists():
        print(f"ERROR: Image not found at {img_path}")
        return

    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    draw = ImageDraw.Draw(im)

    words = data["words"]
    boxes = data["boxes"]
    labels = data["labels"]

    labeled_count = 0
    for word, box_norm, label in zip(words, boxes, labels):
        if label == "O":
            continue

        labeled_count += 1
        x1, y1, x2, y2 = denormalize_box(box_norm, w, h)
        tag = label.replace("B-", "").replace("I-", "")
        color = LABEL_COLORS.get(tag, (220, 38, 38))

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        # Draw label text header
        draw.rectangle([x1, max(0, y1 - 14), min(w, x1 + len(label) * 7 + 8), y1], fill=color)
        draw.text((x1 + 2, max(0, y1 - 13)), label, fill=(255, 255, 255))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"viz_{sample_path.stem}.png"
    im.save(out_file)
    print(f"Saved visualization to: {out_file} ({labeled_count} labeled tokens rendered)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize LayoutLM annotated sample")
    parser.add_argument("--sample", required=True, help="Path to sample JSON")
    parser.add_argument("--output-dir", default="output/debug_viz", help="Output directory for annotated image")
    args = parser.parse_args()

    visualize_sample(Path(args.sample), Path(args.output_dir))
