"""
scripts/export_reviewed_to_layoutlm.py

Exports reviewed invoices from PostgreSQL and raw images/PDFs into a self-contained,
production-ready LayoutLMv3 token classification dataset.

Features:
- Guaranteed valid non-zero-area bounding boxes (strictly [0 <= x1 < x2 <= 1000, 0 <= y1 < y2 <= 1000]).
- Focused high-value header/financial label taxonomy (24 core fields by default; table lines optional).
- Self-contained dataset package (copies rasterized images to images/ and outputs metadata.json).
- Built-in QA dataset validator (ensures 100% token/box/label/image consistency).

Usage:
    python scripts/export_reviewed_to_layoutlm.py --output-dir data/layoutlm_dataset --val-ratio 0.2
"""

import argparse
import asyncio
import difflib
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Optional, Any, Union
from PIL import Image
from loguru import logger
import pymupdf

# Add root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from storage.db import DatabaseManager, InvoiceRecord
from ocr.extractor import InvoiceOCR as OCRExtractor
from validation.validator import InvoiceSchema


def normalize_box(box, width: int, height: int) -> list[int]:
    """
    Normalize bbox coordinates to [0, 1000].
    Strictly guarantees x2 > x1 and y2 > y1 with non-zero area.
    """
    if len(box) == 4 and isinstance(box[0], (int, float)):
        x1, y1, x2, y2 = box
    else:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

    w_safe = max(1, int(width))
    h_safe = max(1, int(height))

    nx1 = max(0, min(1000, int(1000.0 * x1 / w_safe)))
    ny1 = max(0, min(1000, int(1000.0 * y1 / h_safe)))
    nx2 = max(0, min(1000, int(1000.0 * x2 / w_safe)))
    ny2 = max(0, min(1000, int(1000.0 * y2 / h_safe)))

    # Ensure strictly positive width
    if nx2 <= nx1:
        nx2 = min(1000, nx1 + 3)
        if nx2 <= nx1:
            nx1 = max(0, nx2 - 3)

    # Ensure strictly positive height
    if ny2 <= ny1:
        ny2 = min(1000, ny1 + 3)
        if ny2 <= ny1:
            ny1 = max(0, ny2 - 3)

    return [int(nx1), int(ny1), int(nx2), int(ny2)]


def clean_str(s: Any) -> str:
    if s is None:
        return ""
    return re.sub(r"[^a-zA-Z0-9]", "", str(s).lower())


def str_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def try_float(s: Any) -> Optional[float]:
    if s is None:
        return None
    try:
        clean = re.sub(r"[^\d.]", "", str(s))
        if clean:
            return float(clean)
    except (ValueError, TypeError):
        pass
    return None


def assign_bio_labels(
    words: list[str],
    boxes: list[list[int]],
    schema: InvoiceSchema,
    include_line_items: bool = False,
) -> tuple[list[str], dict[str, bool]]:
    """
    Match OCR words against ground truth fields in InvoiceSchema
    using fuzzy text and numeric equivalence, then assign BIO tags.
    Returns (labels, matched_fields).
    """
    labels = ["O"] * len(words)
    matched_fields: dict[str, bool] = {}

    # Core 24 Focused Header, Party, Financial & Banking fields
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
        ("SUBTOTAL", str(schema.subtotal) if schema.subtotal is not None else None),
        ("DISCOUNT", str(schema.discount) if schema.discount is not None else None),
        ("TAX_AMOUNT", str(schema.tax_amount) if schema.tax_amount is not None else None),
        ("CGST", str(schema.cgst) if schema.cgst is not None else None),
        ("SGST", str(schema.sgst) if schema.sgst is not None else None),
        ("IGST", str(schema.igst) if schema.igst is not None else None),
        ("GRAND_TOTAL", str(schema.grand_total) if schema.grand_total is not None else None),
        ("BANK_NAME", schema.bank_name),
        ("ACCOUNT_NUMBER", schema.account_number),
        ("IFSC_CODE", schema.ifsc_code),
        ("ACCOUNT_NAME", schema.account_name),
        ("PAYMENT_TERMS", schema.payment_terms),
    ]

    for label_type, val in fields:
        if not val or len(str(val).strip()) < 2:
            continue

        matched_fields[label_type] = False
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
                        matched_fields[label_type] = True
                        break

        if matched:
            continue

        # Multi-word phrase matching
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
                        matched_fields[label_type] = True
                        break

        if matched:
            continue

        # Single word fuzzy matching
        for i, w in enumerate(words):
            if labels[i] == "O":
                w_c = clean_str(w)
                if len(w_c) >= 3 and (w_c in val_c or val_c in w_c or str_similarity(w_c, val_c) >= 0.80):
                    labels[i] = f"B-{label_type}"
                    matched_fields[label_type] = True
                    break

    # Optional fine-grained table line-item labels
    if include_line_items and schema.line_items:
        for item in schema.line_items:
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
                ("LINE_ITEM_RATE", item.rate),
                ("LINE_ITEM_QTY", item.quantity),
                ("LINE_ITEM_AMOUNT", item.amount),
            ]:
                if col_val is not None:
                    val_num = try_float(col_val)
                    for i, w in enumerate(words):
                        if labels[i] == "O":
                            w_num = try_float(w)
                            if val_num is not None and w_num is not None and abs(w_num - val_num) < 0.01:
                                labels[i] = f"B-{col_label}"
                                break

    return labels, matched_fields


def extract_words_from_pdf_native(pdf_path: Path) -> tuple[list[str], list[list[int]]]:
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
        boxes.append(normalize_box([x0, y0, x1, y1], int(w), int(h)))
    return words, boxes


def validate_dataset(dataset_dir: Path) -> bool:
    """
    Strict QA gate: Validates that all exported training and validation samples
    have strictly valid boxes, matching token lengths, and existing images.
    """
    logger.info("Running QA Validation on exported LayoutLM dataset...")
    total_samples = 0
    total_tokens = 0
    invalid_boxes = 0
    missing_images = 0
    mismatched_lengths = 0

    for split in ["train", "val"]:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
        for json_file in split_dir.glob("*.json"):
            total_samples += 1
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            words = data.get("words", [])
            boxes = data.get("boxes", [])
            labels = data.get("labels", [])
            img_rel = data.get("image_path", "")

            # Check lengths
            if not (len(words) == len(boxes) == len(labels) and len(words) > 0):
                logger.error(f"[{json_file.name}] Length mismatch: words={len(words)}, boxes={len(boxes)}, labels={len(labels)}")
                mismatched_lengths += 1

            # Check image existence
            img_path = dataset_dir / img_rel if not Path(img_rel).is_absolute() else Path(img_rel)
            if not img_path.exists():
                logger.error(f"[{json_file.name}] Image file not found: {img_path}")
                missing_images += 1

            # Check box geometries
            for idx, b in enumerate(boxes):
                total_tokens += 1
                if len(b) != 4 or b[0] >= b[2] or b[1] >= b[3] or b[0] < 0 or b[1] < 0 or b[2] > 1000 or b[3] > 1000:
                    logger.error(f"[{json_file.name}] Invalid box at token {idx} '{words[idx]}': {b}")
                    invalid_boxes += 1

    print("\n" + "=" * 65)
    print("   DATASET QA VALIDATION REPORT")
    print("=" * 65)
    print(f"Total Samples Checked:    {total_samples}")
    print(f"Total Tokens Checked:     {total_tokens}")
    print(f"Length Mismatches:        {mismatched_lengths}")
    print(f"Invalid Bounding Boxes:   {invalid_boxes}")
    print(f"Missing Image Files:      {missing_images}")
    print("-" * 65)

    if invalid_boxes == 0 and missing_images == 0 and mismatched_lengths == 0 and total_samples > 0:
        print("STATUS: PASSED (100% Valid & Self-Contained)")
        print("=" * 65 + "\n")
        return True
    else:
        print("STATUS: FAILED (Defects Detected)")
        print("=" * 65 + "\n")
        return False


async def export_layoutlm_dataset(
    output_dir: str = "data/layoutlm_dataset",
    val_ratio: float = 0.2,
    tier: str = "human_verified",
    max_samples: Optional[int] = None,
    include_line_items: bool = False,
    exclude_locked_test: bool = True,
):
    settings = get_settings()
    db = DatabaseManager(settings.database_url)
    await db.init_db()

    ocr = None

    output_path = Path(output_dir)
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    images_dir = output_path / "images"

    # Clean legacy JSON files from previous test runs to ensure 100% fresh dataset
    for d in [train_dir, val_dir]:
        if d.exists():
            for f in d.glob("*.json"):
                f.unlink(missing_ok=True)

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    async with db.session() as session:
        from sqlalchemy import select
        if tier == "gold":
            stmt = select(InvoiceRecord).where(
                InvoiceRecord.output_json.isnot(None),
                InvoiceRecord.ground_truth_source == "human_corrected",
            )
            logger.info("Filtering for Gold Tier: strictly 'human_corrected' ground truth...")
        elif tier == "human_verified":
            stmt = select(InvoiceRecord).where(
                InvoiceRecord.output_json.isnot(None),
                InvoiceRecord.ground_truth_source.in_(["human_corrected", "human_confirmed"]),
            )
            logger.info("Filtering for Verified Tier: strictly 'human_corrected' + 'human_confirmed' (Zero Bronze contamination)...")
        else:
            stmt = select(InvoiceRecord).where(InvoiceRecord.output_json.isnot(None))
            logger.info("Exporting all invoices (including auto-accepted predictions)...")

        result = await session.execute(stmt)
        jobs = result.scalars().all()

        # Fallback if no records have been tagged with ground_truth_source yet
        if not jobs and tier == "human_verified":
            logger.info("No records tagged with ground_truth_source yet. Falling back to reviewed records (strictly verified)...")
            stmt_fallback = select(InvoiceRecord).where(
                InvoiceRecord.output_json.isnot(None),
                InvoiceRecord.status == "reviewed",
                InvoiceRecord.needs_review == False,
                InvoiceRecord.ground_truth_source != "auto_accepted",
            )
            jobs = (await session.execute(stmt_fallback)).scalars().all()

    # Deduplicate on document_hash to prevent repeated uploads polluting training
    seen_hashes = set()
    deduped_jobs = []
    for j in jobs:
        d_hash = getattr(j, "document_hash", None)
        if d_hash:
            if d_hash in seen_hashes:
                continue
            seen_hashes.add(d_hash)
        deduped_jobs.append(j)
    jobs = deduped_jobs

    # Exclude locked holdout evaluation set if requested
    if exclude_locked_test:
        locked_file = Path("data/evaluation/locked_test/locked_job_ids.json")
        if locked_file.exists():
            try:
                with open(locked_file, "r", encoding="utf-8") as lf:
                    locked_ids = set(json.load(lf))
                prev_len = len(jobs)
                jobs = [j for j in jobs if j.job_id not in locked_ids]
                if len(jobs) < prev_len:
                    logger.info(f"Excluded {prev_len - len(jobs)} locked test samples from training dataset.")
            except Exception as e:
                logger.warning(f"Could not load locked test IDs: {e}")

    if max_samples and max_samples > 0:
        jobs = jobs[:max_samples]

    logger.info(f"Processing {len(jobs)} invoices for LayoutLM dataset...")

    if not jobs:
        logger.warning("No processed invoices found in database! Upload and review some invoices first.")
        return

    samples = []
    field_alignment_stats: dict[str, dict[str, int]] = {}
    raw_images_dir = Path("data/images_to_annotate")
    raw_images_dir.mkdir(parents=True, exist_ok=True)

    for idx, job in enumerate(jobs):
        raw_inv = job.output_json
        if not raw_inv:
            continue

        try:
            if any(k in raw_inv for k in ["company", "client", "meta", "items", "totals", "bankDetails"]):
                schema_obj = InvoiceSchema.from_invoice_builder_json(raw_inv)
            else:
                schema_obj = InvoiceSchema(**raw_inv)
        except Exception as ex:
            logger.warning(f"Could not parse InvoiceSchema for job {job.job_id}: {ex}")
            continue

        img_path = None
        source_pdf = None
        stem = Path(job.filename).stem

        # 1. Search for source PDF across data directories
        raw_candidates = [
            Path(f"data/raw/{job.job_id}_{job.filename}"),
            Path(f"data/raw/{job.filename}"),
            Path(f"data/uploads/{job.job_id}_{job.filename}"),
            *list(Path("data/raw").glob(f"*{job.job_id}*")),
            *list(Path("data/raw").glob(f"*{stem}*")),
            *list(Path("data/uploads").glob(f"*{job.job_id}*")),
            *list(Path("data/uploads").glob(f"*{stem}*")),
        ]
        for c in raw_candidates:
            if c.exists() and c.is_file():
                if c.suffix.lower() == ".pdf":
                    source_pdf = c
                    break
                elif c.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"] and not img_path:
                    img_path = c

        # 2. Check existing rasterized images in raw_images_dir
        if not img_path:
            img_candidates = list(raw_images_dir.glob(f"*{job.job_id}*")) or list(raw_images_dir.glob(f"*{stem}*"))
            if img_candidates:
                img_path = img_candidates[0]

        # 3. If image missing but PDF found, rasterize page 1
        if not img_path and source_pdf and source_pdf.exists():
            try:
                doc = pymupdf.open(str(source_pdf))
                page = doc[0]
                pix = page.get_pixmap(dpi=150)
                target_png = raw_images_dir / f"{job.job_id}_p1.png"
                pix.save(str(target_png))
                doc.close()
                img_path = target_png
            except Exception as ex:
                logger.warning(f"Could not rasterize PDF {source_pdf}: {ex}")

        if not img_path or not img_path.exists():
            logger.warning(f"[{idx+1}/{len(jobs)}] Could not find image for job {job.job_id} ({job.filename})")
            continue

        # Copy image into dataset images/ directory for 100% self-contained packaging
        dest_image_name = f"{job.job_id}_p1.png"
        dest_image_path = images_dir / dest_image_name
        if not dest_image_path.exists():
            shutil.copyfile(img_path, dest_image_path)

        logger.info(f"[{idx+1}/{len(jobs)}] Processing {job.filename}...")

        try:
            words = []
            boxes = []

            with Image.open(dest_image_path) as im:
                w, h = im.size
                im_rgb = im.convert("RGB")

            # 1. High-speed Path A: PyMuPDF Native PDF text layer (instant ~0.005s with exact word boxes)
            if source_pdf and source_pdf.exists():
                try:
                    words, boxes = extract_words_from_pdf_native(source_pdf)
                except Exception as pdf_err:
                    logger.debug(f"PyMuPDF word extraction error: {pdf_err}")

            # 2. Path B: Scanned PDF / Image OCR (only if native text was empty or scanned)
            if len(words) < 5:
                try:
                    if ocr is None:
                        ocr = OCRExtractor(engine="easyocr")
                    ocr_res = ocr.extract_from_image(im_rgb)
                    for block in ocr_res.text_blocks:
                        if block.words:
                            for w_obj in block.words:
                                words.append(w_obj.text)
                                boxes.append(normalize_box(w_obj.to_xyxy(), w, h))
                        else:
                            for w_str in block.text.split():
                                words.append(w_str)
                                boxes.append(normalize_box(block.bbox, w, h))
                except Exception as ocr_err:
                    logger.debug(f"OCR extractor error on {job.filename}: {ocr_err}")

            if not words:
                logger.warning(f"No text extracted for job {job.job_id}")
                continue

            labels, matched_fields = assign_bio_labels(
                words,
                boxes,
                schema_obj,
                include_line_items=include_line_items,
            )

            # Record stats
            for k, is_m in matched_fields.items():
                if k not in field_alignment_stats:
                    field_alignment_stats[k] = {"gt_count": 0, "matched_count": 0}
                field_alignment_stats[k]["gt_count"] += 1
                if is_m:
                    field_alignment_stats[k]["matched_count"] += 1

            sample_dict = {
                "image_path": f"images/{dest_image_name}",
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

    # Print Alignment Scorecard
    print("\n" + "=" * 65)
    print("   LAYOUTLM GROUND TRUTH ALIGNMENT STATISTICS")
    print("=" * 65)
    print(f"{'Field':<26} | {'Matched':<8} | {'Present':<8} | {'Alignment %':<10}")
    print("-" * 65)
    total_gt = 0
    total_matched = 0
    for f_name, stats in field_alignment_stats.items():
        m_cnt = stats["matched_count"]
        g_cnt = stats["gt_count"]
        pct = (m_cnt / g_cnt * 100.0) if g_cnt > 0 else 0.0
        total_matched += m_cnt
        total_gt += g_cnt
        print(f"{f_name:<26} | {m_cnt:<8} | {g_cnt:<8} | {pct:>10.1f}%")
    print("-" * 65)
    overall_pct = (total_matched / total_gt * 100.0) if total_gt > 0 else 0.0
    print(f"{'OVERALL ALIGNMENT':<26} | {total_matched:<8} | {total_gt:<8} | {overall_pct:>10.1f}%")
    print("=" * 65 + "\n")

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

    # Export dataset metadata
    metadata = {
        "dataset_name": "ai_invoice_layoutlmv3",
        "format": "layoutlmv3_token_classification",
        "total_samples": len(samples),
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "overall_alignment_rate": f"{overall_pct:.1f}%",
        "field_alignment_statistics": field_alignment_stats,
        "self_contained": True,
        "images_directory": "images/",
    }
    with open(output_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Dataset exported: {len(train_set)} train samples, {len(val_set)} val samples in {output_path}")

    # Run strict QA Validation
    is_valid = validate_dataset(output_path)
    if is_valid:
        print("=" * 65)
        print("READY TO TRAIN LayoutLMv3! Run:")
        print(f"python scripts/train_layoutlm.py --data_dir {output_path} --epochs 10")
        print("=" * 65 + "\n")
    else:
        logger.error("Dataset QA validation failed. Please check the reported defects.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export reviewed invoices to LayoutLMv3 dataset")
    parser.add_argument("--output-dir", default="data/layoutlm_dataset", help="Output directory")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--tier", choices=["gold", "human_verified", "all"], default="human_verified", help="Ground truth tier to export (default: human_verified)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples to process")
    parser.add_argument("--include-line-items", action="store_true", help="Include fine-grained line item labels (default: False, header/financial focus)")
    parser.add_argument("--include-locked-test", action="store_true", help="Include locked test samples in export (default: False, excluded)")
    args = parser.parse_args()

    asyncio.run(export_layoutlm_dataset(
        args.output_dir,
        args.val_ratio,
        tier=args.tier,
        max_samples=args.max_samples,
        include_line_items=args.include_line_items,
        exclude_locked_test=not args.include_locked_test,
    ))
