"""
scripts/export_reviewed_to_layoutlm.py

Exports reviewed invoices from the PostgreSQL database and raw images/PDFs into
the LayoutLMv3 token classification dataset format (JSON with words, boxes, BIO labels).
Synchronized with latest Invoice Builder and FastAPI schemas.

Usage:
    python scripts/export_reviewed_to_layoutlm.py --output-dir data/layoutlm_dataset --val-ratio 0.2
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
from validation.validator import InvoiceSchema


def normalize_box(box, width, height):
    """Normalize bbox coords to [0, 1000]."""
    if len(box) == 4 and isinstance(box[0], (int, float)):
        x1, y1, x2, y2 = box
    else:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return [
        max(0, min(1000, int(1000 * x1 / max(1, width)))),
        max(0, min(1000, int(1000 * y1 / max(1, height)))),
        max(0, min(1000, int(1000 * x2 / max(1, width)))),
        max(0, min(1000, int(1000 * y2 / max(1, height)))),
    ]


import difflib

def clean_str(s):
    if s is None:
        return ""
    return re.sub(r"[^a-zA-Z0-9]", "", str(s).lower())

def str_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

def try_float(s) -> float | None:
    if s is None:
        return None
    try:
        clean = re.sub(r"[^\d.]", "", str(s))
        if clean:
            return float(clean)
    except (ValueError, TypeError):
        pass
    return None

def assign_bio_labels(words, boxes, schema: InvoiceSchema):
    """
    Match OCR words against ground truth fields in InvoiceSchema
    using fuzzy text and numeric equivalence, then assign BIO tags.
    """
    labels = ["O"] * len(words)

    # 1. Primary Header, Party, Financial & Bank fields
    fields = [
        ("INVOICE_NUMBER", schema.invoice_number),
        ("PO_NUMBER", schema.po_number),
        ("INVOICE_DATE", schema.invoice_date),
        ("DUE_DATE", schema.due_date),
        ("PLACE_OF_SUPPLY", schema.place_of_supply),
        ("CATEGORY", schema.category),
        ("SUBCATEGORY", schema.subcategory),
        ("VENDOR_NAME", schema.vendor_name),
        ("VENDOR_ADDRESS", schema.vendor_address),
        ("VENDOR_GSTIN", schema.vendor_gstin),
        ("VENDOR_PAN", schema.vendor_pan),
        ("VENDOR_EMAIL", schema.vendor_email),
        ("VENDOR_PHONE", schema.vendor_phone),
        ("BUYER_NAME", schema.buyer_name),
        ("BUYER_ADDRESS", schema.buyer_address),
        ("BUYER_GSTIN", schema.buyer_gstin),
        ("BUYER_PHONE", schema.buyer_phone),
        ("SLS_CODE", schema.sls_code),
        ("SUBTOTAL", str(schema.subtotal) if schema.subtotal is not None else None),
        ("DISCOUNT", str(schema.discount) if schema.discount is not None else None),
        ("GLOBAL_DISCOUNT", str(schema.global_discount) if schema.global_discount is not None else None),
        ("TAX_AMOUNT", str(schema.tax_amount) if schema.tax_amount is not None else None),
        ("CGST", str(schema.cgst) if schema.cgst is not None else None),
        ("SGST", str(schema.sgst) if schema.sgst is not None else None),
        ("IGST", str(schema.igst) if schema.igst is not None else None),
        ("GLOBAL_CGST_RATE", str(schema.global_cgst_rate) if schema.global_cgst_rate is not None else None),
        ("GLOBAL_SGST_RATE", str(schema.global_sgst_rate) if schema.global_sgst_rate is not None else None),
        ("GLOBAL_IGST_RATE", str(schema.global_igst_rate) if schema.global_igst_rate is not None else None),
        ("ROUND_OFF", str(schema.round_off) if schema.round_off is not None else None),
        ("GRAND_TOTAL", str(schema.grand_total) if schema.grand_total is not None else None),
        ("AMOUNT_IN_WORDS", schema.amount_in_words),
        ("CURRENCY", schema.currency),
        ("BANK_NAME", schema.bank_name),
        ("BRANCH_NAME", schema.branch_name),
        ("ACCOUNT_NAME", schema.account_name),
        ("ACCOUNT_NUMBER", schema.account_number),
        ("IFSC_CODE", schema.ifsc_code),
        ("PAYMENT_TERMS", schema.payment_terms),
        ("REMARKS", schema.remarks),
    ]

    for label_type, val in fields:
        if not val or len(str(val).strip()) < 2:
            continue

        val_str = str(val).strip()
        val_c = clean_str(val_str)
        val_words = val_str.split()
        val_words_c = [clean_str(w) for w in val_words if clean_str(w)]
        val_num = try_float(val_str)

        if not val_words_c and val_num is None:
            continue

        matched = False

        # Numeric field matching (e.g. 6200.00 vs 6,200)
        if val_num is not None and len(val_c) >= 2:
            for i, w in enumerate(words):
                if labels[i] == "O":
                    w_num = try_float(w)
                    if w_num is not None and abs(w_num - val_num) < 0.01:
                        labels[i] = f"B-{label_type}"
                        matched = True
                        break

        if matched:
            continue

        # Exact / Fuzzy multi-word phrase matching
        n_words = len(val_words_c)
        if n_words > 0:
            for i in range(len(words) - n_words + 1):
                window_c = [clean_str(words[i + j]) for j in range(n_words)]
                if window_c == val_words_c or str_similarity(" ".join(window_c), " ".join(val_words_c)) >= 0.80:
                    if all(labels[i + j] == "O" for j in range(n_words)):
                        labels[i] = f"B-{label_type}"
                        for j in range(1, n_words):
                            labels[i + j] = f"I-{label_type}"
                        matched = True
                        break

        if matched:
            continue

        # Single word fuzzy matching (handles typos like JA-/3/2026 -> JA-13/2026)
        for i, w in enumerate(words):
            if labels[i] == "O":
                w_c = clean_str(w)
                if len(w_c) >= 3 and (w_c in val_c or val_c in w_c or str_similarity(w_c, val_c) >= 0.80):
                    labels[i] = f"B-{label_type}"
                    break

    # 2. Certified remarks / Declarations
    for remark in schema.certified_remarks or []:
        if not remark or len(remark.strip()) < 5:
            continue
        rem_words = [clean_str(w) for w in remark.strip().split() if clean_str(w)]
        if not rem_words:
            continue
        for i in range(len(words) - len(rem_words) + 1):
            window = [clean_str(words[i + j]) for j in range(len(rem_words))]
            if (window == rem_words or str_similarity(" ".join(window), " ".join(rem_words)) >= 0.80) and all(labels[i + j] == "O" for j in range(len(rem_words))):
                labels[i] = "B-CERTIFIED_REMARKS"
                for j in range(1, len(rem_words)):
                    labels[i + j] = "I-CERTIFIED_REMARKS"
                break

    # 3. Line items
    for item in schema.line_items or []:
        desc = item.description or ""
        if desc:
            desc_words = [clean_str(w) for w in desc.split() if clean_str(w)]
            for i in range(len(words) - len(desc_words) + 1):
                window = [clean_str(words[i + j]) for j in range(len(desc_words))]
                if (window == desc_words or str_similarity(" ".join(window), " ".join(desc_words)) >= 0.80) and all(labels[i + j] == "O" for j in range(len(desc_words))):
                    labels[i] = "B-LINE_ITEM_DESC"
                    for j in range(1, len(desc_words)):
                        labels[i + j] = "I-LINE_ITEM_DESC"
                    break

        for col_label, col_val in [
            ("LINE_ITEM_HSN", item.hsn_code),
            ("LINE_ITEM_QTY", item.quantity),
            ("LINE_ITEM_UNIT", item.unit),
            ("LINE_ITEM_RATE", item.rate),
            ("LINE_ITEM_DISCOUNT", item.discount),
            ("LINE_ITEM_TAXABLE_VALUE", item.taxable_value),
            ("LINE_ITEM_CGST_RATE", item.cgst_rate),
            ("LINE_ITEM_CGST_AMOUNT", item.cgst_amount),
            ("LINE_ITEM_SGST_RATE", item.sgst_rate),
            ("LINE_ITEM_SGST_AMOUNT", item.sgst_amount),
            ("LINE_ITEM_IGST_RATE", item.igst_rate),
            ("LINE_ITEM_IGST_AMOUNT", item.igst_amount),
            ("LINE_ITEM_AMOUNT", item.amount),
        ]:
            if col_val is not None:
                val_c = clean_str(col_val)
                val_num = try_float(col_val)
                for i, w in enumerate(words):
                    if labels[i] == "O":
                        w_num = try_float(w)
                        if val_num is not None and w_num is not None and abs(w_num - val_num) < 0.01:
                            labels[i] = f"B-{col_label}"
                            break
                        elif val_c and clean_str(w) == val_c:
                            labels[i] = f"B-{col_label}"
                            break

    return labels


def extract_words_from_pdf_native(pdf_path: Path):
    """Extract word tokens and bounding boxes directly from native PDF text layer."""
    doc = pymupdf.open(str(pdf_path))
    page = doc[0]
    rect = page.rect
    w, h = rect.width, rect.height
    words_data = page.get_text("words")
    doc.close()

    words = []
    boxes = []
    for item in words_data:
        x0, y0, x1, y1, word_text = item[:5]
        word_clean = word_text.strip()
        if not word_clean:
            continue
        words.append(word_clean)
        boxes.append(normalize_box([x0, y0, x1, y1], w, h))
    return words, boxes


async def export_layoutlm_dataset(
    output_dir: str = "data/layoutlm_dataset",
    val_ratio: float = 0.2,
    only_verified: bool = True,
):
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
        if only_verified:
            # Strictly export Verified Ground Truth invoices
            stmt = select(InvoiceRecord).where(
                InvoiceRecord.output_json.isnot(None),
                InvoiceRecord.status.in_(["reviewed", "partially_reviewed"]),
                InvoiceRecord.needs_review == False,
            )
            logger.info("Filtering for Verified Ground Truth invoices (status in ['reviewed', 'partially_reviewed'])...")
        else:
            stmt = select(InvoiceRecord).where(InvoiceRecord.output_json.isnot(None))
            logger.info("Exporting all invoices with output_json (including unverified)...")

        result = await session.execute(stmt)
        jobs = result.scalars().all()

    logger.info(f"Found {len(jobs)} eligible invoices for LayoutLM ground truth dataset")

    if not jobs:
        logger.warning("No processed invoices found in database! Upload and review some invoices first.")
        return

    samples = []
    raw_images_dir = Path("data/images_to_annotate")

    for job in jobs:
        raw_inv = job.output_json
        if not raw_inv:
            continue

        # Normalize to InvoiceSchema
        try:
            if any(k in raw_inv for k in ["company", "client", "meta", "items", "totals", "bankDetails"]):
                schema_obj = InvoiceSchema.from_invoice_builder_json(raw_inv)
            else:
                schema_obj = InvoiceSchema(**raw_inv)
        except Exception as ex:
            logger.warning(f"Could not parse InvoiceSchema for job {job.job_id}: {ex}")
            continue

        raw_images_dir.mkdir(parents=True, exist_ok=True)
        img_path = None
        source_pdf = None

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
                        source_pdf = c
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
            words = []
            boxes = []

            with Image.open(img_path) as im:
                w, h = im.size
                im_rgb = im.convert("RGB")

            # Try OCR extractor first
            try:
                ocr_res = ocr.extract_from_image(im_rgb)
                for block in ocr_res.text_blocks:
                    words.append(block.text)
                    boxes.append(normalize_box(block.bbox, w, h))
            except Exception as ocr_err:
                logger.debug(f"OCR extractor error on {job.filename}: {ocr_err}")

            # Fallback to PyMuPDF native words if PDF available and OCR gave few words
            if len(words) < 5 and source_pdf and source_pdf.exists():
                try:
                    words, boxes = extract_words_from_pdf_native(source_pdf)
                except Exception as pdf_err:
                    logger.debug(f"PyMuPDF word extraction error: {pdf_err}")

            if not words:
                logger.warning(f"No text extracted for job {job.job_id}")
                continue

            labels = assign_bio_labels(words, boxes, schema_obj)

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
        with open(train_dir / f"{s['job_id']}.json", "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2)

    for s in val_set:
        with open(val_dir / f"{s['job_id']}.json", "w", encoding="utf-8") as f:
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
    parser.add_argument("--include-unverified", action="store_true", help="Include unverified predictions (default: False, only verified ground truth)")
    args = parser.parse_args()

    asyncio.run(export_layoutlm_dataset(args.output_dir, args.val_ratio, only_verified=not args.include_unverified))
