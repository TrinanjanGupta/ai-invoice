"""
scripts/export_reviewed_to_layoutlm.py

Exports reviewed invoices from the PostgreSQL database and raw images into
the LayoutLMv3 token classification dataset format (JSON with words, boxes, BIO labels).

Usage:
    python scripts/export_reviewed_to_layoutlm.py --val-ratio 0.2
"""

import argparse
import asyncio
import json
import os
import random
import re
from pathlib import Path
from PIL import Image
from loguru import logger
import pymupdf

# Add root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from storage.db import DatabaseManager, InvoiceRecord
from ocr.extractor import InvoiceOCR as OCRExtractor


def normalize_box(box, width, height):
    """Normalize bbox coords to [0, 1000]."""
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return [
        max(0, min(1000, int(1000 * x1 / width))),
        max(0, min(1000, int(1000 * y1 / height))),
        max(0, min(1000, int(1000 * x2 / width))),
        max(0, min(1000, int(1000 * y2 / height))),
    ]


def clean_str(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", str(s).lower())


def assign_bio_labels(words, boxes, invoice_data):
    """
    Match OCR words against ground truth fields in invoice_data
    and assign BIO tags.
    """
    labels = ["O"] * len(words)

    # Key ground-truth mappings
    fields = [
        ("INVOICE_NUMBER", invoice_data.get("invoice_number")),
        ("INVOICE_DATE", invoice_data.get("invoice_date")),
        ("DUE_DATE", invoice_data.get("due_date")),
        ("PO_NUMBER", invoice_data.get("po_number")),
        ("PLACE_OF_SUPPLY", invoice_data.get("place_of_supply")),
        ("VENDOR_NAME", invoice_data.get("vendor_name")),
        ("VENDOR_ADDRESS", invoice_data.get("vendor_address")),
        ("VENDOR_GSTIN", invoice_data.get("vendor_gstin")),
        ("BUYER_NAME", invoice_data.get("buyer_name")),
        ("BUYER_ADDRESS", invoice_data.get("buyer_address")),
        ("BUYER_GSTIN", invoice_data.get("buyer_gstin")),
        ("SUBTOTAL", str(invoice_data.get("subtotal") or "")),
        ("CGST", str(invoice_data.get("cgst") or "")),
        ("SGST", str(invoice_data.get("sgst") or "")),
        ("IGST", str(invoice_data.get("igst") or "")),
        ("TAX_AMOUNT", str(invoice_data.get("tax_amount") or "")),
        ("GRAND_TOTAL", str(invoice_data.get("grand_total") or "")),
        ("BANK_NAME", invoice_data.get("bank_name")),
        ("ACCOUNT_NUMBER", invoice_data.get("account_number")),
        ("IFSC_CODE", invoice_data.get("ifsc_code")),
    ]

    for label_type, val in fields:
        if not val or len(str(val).strip()) < 2:
            continue
        val_clean = clean_str(val)
        val_words = str(val).strip().split()
        val_words_clean = [clean_str(w) for w in val_words if clean_str(w)]

        if not val_words_clean:
            continue

        # Try to find matching sequence of words
        for i in range(len(words) - len(val_words_clean) + 1):
            window = [clean_str(words[i + j]) for j in range(len(val_words_clean))]
            if window == val_words_clean:
                labels[i] = f"B-{label_type}"
                for j in range(1, len(val_words_clean)):
                    labels[i + j] = f"I-{label_type}"
                break
        else:
            # Fallback: single word match
            for i, w in enumerate(words):
                if labels[i] == "O" and clean_str(w) == val_clean and len(val_clean) >= 3:
                    labels[i] = f"B-{label_type}"
                    break

    # Line items
    for item in invoice_data.get("line_items", []):
        desc = item.get("description", "")
        if desc:
            desc_words = [clean_str(w) for w in desc.split() if clean_str(w)]
            for i in range(len(words) - len(desc_words) + 1):
                window = [clean_str(words[i + j]) for j in range(len(desc_words))]
                if window == desc_words and all(labels[i + j] == "O" for j in range(len(desc_words))):
                    labels[i] = "B-LINE_ITEM_DESC"
                    for j in range(1, len(desc_words)):
                        labels[i + j] = "I-LINE_ITEM_DESC"
                    break

        # Match qty, rate, amount if present
        for col_label, col_val in [
            ("LINE_ITEM_QTY", item.get("quantity")),
            ("LINE_ITEM_RATE", item.get("rate")),
            ("LINE_ITEM_AMOUNT", item.get("amount")),
        ]:
            if col_val is not None:
                val_c = clean_str(col_val)
                if val_c:
                    for i, w in enumerate(words):
                        if labels[i] == "O" and clean_str(w) == val_c:
                            labels[i] = f"B-{col_label}"
                            break

    return labels


async def export_layoutlm_dataset(output_dir: str = "data/layoutlm_dataset", val_ratio: float = 0.2):
    settings = get_settings()
    db = DatabaseManager(settings.database_url)
    await db.init_db()

    ocr = OCRExtractor()

    output_path = Path(output_dir)
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    async with db.session() as session:
        from sqlalchemy import select
        stmt = select(InvoiceRecord).where(InvoiceRecord.output_json.isnot(None))
        result = await session.execute(stmt)
        jobs = result.scalars().all()

    logger.info(f"Found {len(jobs)} processed/reviewed invoices in database")

    if not jobs:
        logger.warning("No processed invoices found in database! Upload and review some invoices first.")
        return

    samples = []
    raw_images_dir = Path("data/images_to_annotate")

    for job in jobs:
        inv_data = job.output_json
        if not inv_data:
            continue

        # Look for matching document or pre-rendered image
        raw_images_dir.mkdir(parents=True, exist_ok=True)
        img_path = None

        # 1. Check existing images
        stem = Path(job.filename).stem
        candidates = list(raw_images_dir.glob(f"*{job.job_id}*")) or list(raw_images_dir.glob(f"*{stem}*"))
        if candidates:
            img_path = candidates[0]

        # 2. Check data/raw or data/uploads for PDF or image
        if not img_path:
            raw_candidates = [
                Path(f"data/raw/{job.job_id}_{job.filename}"),
                Path(f"data/raw/{job.filename}"),
                Path(f"data/uploads/{job.job_id}_{job.filename}"),
                *list(Path("data/raw").glob(f"*{job.job_id}*")),
                *list(Path("data/raw").glob(f"*{stem}*")),
            ]
            for c in raw_candidates:
                if c.exists() and c.is_file():
                    if c.suffix.lower() == ".pdf":
                        # Convert first page to PNG
                        try:
                            doc = pymupdf.open(str(c))
                            page = doc[0]
                            pix = page.get_pixmap(dpi=150)
                            target_png = raw_images_dir / f"{job.job_id}_p1.png"
                            pix.save(str(target_png))
                            doc.close()
                            img_path = target_png
                            break
                        except Exception as ex:
                            logger.warning(f"Could not rasterize PDF {c}: {ex}")
                    elif c.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
                        img_path = c
                        break

        if not img_path or not img_path.exists():
            logger.warning(f"Could not find image for job {job.job_id} ({job.filename})")
            continue

        try:
            with Image.open(img_path) as im:
                w, h = im.size
                im_rgb = im.convert("RGB")

            # Run OCR word-level extraction
            ocr_res = ocr.extract_from_image(im_rgb)
            words = []
            boxes = []
            for block in ocr_res.text_blocks:
                words.append(block.text)
                boxes.append(normalize_box(block.bbox, w, h))

            if not words:
                continue

            labels = assign_bio_labels(words, boxes, inv_data)

            sample_dict = {
                "image_path": str(img_path).replace("\\", "/"),
                "words": words,
                "boxes": boxes,
                "labels": labels,
                "job_id": job.job_id,
            }
            samples.append(sample_dict)
            logger.info(f"Generated LayoutLM sample for {job.filename} ({len(words)} tokens)")

        except Exception as e:
            logger.error(f"Failed to process sample {job.job_id}: {e}")

    logger.info(f"Total LayoutLM samples created: {len(samples)}")

    if not samples:
        return

    if len(samples) == 1:
        train_set = samples
        val_set = samples
    else:
        val_count = max(1, int(len(samples) * val_ratio))
        val_set = samples[:val_count]
        train_set = samples[val_count:]

    for s in train_set:
        with open(train_dir / f"{s['job_id']}.json", "w") as f:
            json.dump(s, f, indent=2)

    for s in val_set:
        with open(val_dir / f"{s['job_id']}.json", "w") as f:
            json.dump(s, f, indent=2)

    logger.info(f"Dataset exported: {len(train_set)} train samples, {len(val_set)} val samples in {output_path}")
    print("\n" + "="*60)
    print("READY TO TRAIN LayoutLMv3! Run:")
    print(f"python scripts/train_layoutlm.py --data_dir {output_path} --epochs 10")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export reviewed invoices to LayoutLMv3 dataset")
    parser.add_argument("--output-dir", default="data/layoutlm_dataset", help="Output directory")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    args = parser.parse_args()

    asyncio.run(export_layoutlm_dataset(args.output_dir, args.val_ratio))
