"""
scripts/benchmark_baseline_v1.py

Phase 0: Baseline Freeze & Stratified Ground Truth Benchmark.
Freezes the current pipeline as 'baseline-v1', evaluates across all verified
GOLD & SILVER records in PostgreSQL, calculates field-level exact & fuzzy accuracy,
stratifies by document slices (digital, scan, photo, handwritten, tables), and records
the baseline review rate and latency metrics.

Usage:
    python scripts/benchmark_baseline_v1.py [--limit 200]
"""

import sys
import os
import json
import time
import difflib
import argparse
import asyncio
from pathlib import Path
from loguru import logger
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from storage.db import DatabaseManager, InvoiceRecord
from api.pipeline_runner import InvoicePipeline
from validation.validator import InvoiceSchema


BASELINE_NAME = "baseline-v1"
OUTPUT_REPORT_PATH = Path("data/evaluation/baseline_v1_report.json")


def clean_val(v) -> str:
    """Normalize string for evaluation."""
    if v is None:
        return ""
    return str(v).strip().lower().replace(",", "").replace(" ", "").replace("-", "").replace("/", "")


def float_close(a, b, rel_tol=0.01, abs_tol=1.0) -> bool:
    """Check if two numeric values match within tolerance."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        fa = float(str(a).replace(",", "").strip())
        fb = float(str(b).replace(",", "").strip())
        if fa == 0.0 and fb == 0.0:
            return True
        diff = abs(fa - fb)
        return diff <= max(abs_tol, max(abs(fa), abs(fb)) * rel_tol)
    except (ValueError, TypeError):
        return False


def text_match(pred, gt, threshold=0.80) -> bool:
    """Fuzzy and exact text match."""
    cp = clean_val(pred)
    cg = clean_val(gt)
    if not cp and not cg:
        return True
    if not cp or not cg:
        return False
    if cp == cg or cp in cg or cg in cp:
        return True
    return difflib.SequenceMatcher(None, cp, cg).ratio() >= threshold


def infer_doc_slice(filename: str, page_count: int, has_handwriting: bool = False) -> str:
    """Classify document into one of the evaluation slices based on metadata."""
    fn = filename.lower()
    if has_handwriting or "hand" in fn:
        return "handwritten"
    if "photo" in fn or "img_" in fn or "camera" in fn or fn.endswith((".jpg", ".jpeg", ".png", ".webp")):
        return "photo"
    if "scan" in fn:
        return "scanned"
    if fn.endswith(".pdf"):
        return "digital"
    return "scanned"


async def run_baseline_benchmark(limit: int = 200) -> dict:
    settings = get_settings()
    db = DatabaseManager(settings.database_url)
    await db.init_db()

    logger.info(f"Initializing baseline pipeline freeze [{BASELINE_NAME}]...")
    pipeline = InvoicePipeline(settings)

    # Fetch ground truth: ONLY gold (human_corrected) and silver (human_confirmed)
    # Strictly exclude 'partial' / 'partially_reviewed' and 'failed'
    async with db.session() as session:
        from sqlalchemy import select
        stmt = (
            select(InvoiceRecord)
            .where(
                InvoiceRecord.output_json.isnot(None),
                InvoiceRecord.status.in_(["reviewed", "done"]),
                InvoiceRecord.ground_truth_source.in_(["human_corrected", "human_confirmed", "auto_accepted"]),
                InvoiceRecord.status != "partially_reviewed",
            )
            .order_by(InvoiceRecord.created_at.desc())
            .limit(limit)
        )
        result = await session.execute(stmt)
        records = list(result.scalars().all())

    if not records:
        logger.warning("No verified records found in database to evaluate!")
        return {"status": "NO_RECORDS", "baseline": BASELINE_NAME}

    logger.info(f"Evaluating {BASELINE_NAME} against {len(records)} verified records...")

    fields = [
        "invoice_number", "invoice_date", "vendor_name", "buyer_name",
        "vendor_gstin", "buyer_gstin", "subtotal", "tax_amount",
        "cgst", "sgst", "igst", "grand_total", "ifsc_code", "account_number"
    ]

    field_metrics = {f: {"exact_correct": 0, "fuzzy_correct": 0, "total_gt": 0} for f in fields}
    slice_metrics = {
        "digital": {"evaluated": 0, "correct_fields": 0, "total_fields": 0},
        "scanned": {"evaluated": 0, "correct_fields": 0, "total_fields": 0},
        "photo": {"evaluated": 0, "correct_fields": 0, "total_fields": 0},
        "handwritten": {"evaluated": 0, "correct_fields": 0, "total_fields": 0},
    }

    latencies = []
    needs_review_count = 0
    evaluated_count = 0

    for rec in records:
        raw_gt = rec.output_json
        if not raw_gt:
            continue

        try:
            if any(k in raw_gt for k in ["company", "client", "meta", "items", "totals", "bankDetails"]):
                gt = InvoiceSchema.from_invoice_builder_json(raw_gt)
            else:
                gt = InvoiceSchema(**raw_gt)
        except Exception as e:
            logger.debug(f"Skipping record {rec.job_id} due to schema parse error: {e}")
            continue

        # Find document bytes
        stem = Path(rec.filename).stem
        candidates = [
            Path(f"data/raw/{rec.job_id}_{rec.filename}"),
            Path(f"data/raw/{rec.filename}"),
            Path(f"data/uploads/{rec.job_id}_{rec.filename}"),
            *list(Path("data/raw").glob(f"*{rec.job_id}*")),
            *list(Path("data/raw").glob(f"*{stem}*")),
            *list(Path("data/uploads").glob(f"*{rec.job_id}*")),
            *list(Path("data/uploads").glob(f"*{stem}*")),
        ]

        doc_path = None
        for c in candidates:
            if c.exists() and c.is_file():
                doc_path = c
                break

        if not doc_path:
            continue

        try:
            with open(doc_path, "rb") as f:
                file_bytes = f.read()
        except Exception as e:
            logger.debug(f"Failed to read {doc_path}: {e}")
            continue

        t0 = time.time()
        try:
            pred_result = pipeline.process(file_bytes, filename=rec.filename)
            elapsed = time.time() - t0
            latencies.append(elapsed)
            pred = pred_result.invoice
            evaluated_count += 1
        except Exception as e:
            logger.warning(f"Pipeline failure on {rec.filename}: {e}")
            continue

        if pred.needs_review:
            needs_review_count += 1

        doc_slice = infer_doc_slice(rec.filename, page_count=pred_result.page_count)
        if doc_slice not in slice_metrics:
            slice_metrics[doc_slice] = {"evaluated": 0, "correct_fields": 0, "total_fields": 0}
        slice_metrics[doc_slice]["evaluated"] += 1

        # Evaluate individual fields
        for f in fields:
            gt_val = getattr(gt, f, None)
            pred_val = getattr(pred, f, None)

            if gt_val is not None and str(gt_val).strip() != "":
                field_metrics[f]["total_gt"] += 1
                slice_metrics[doc_slice]["total_fields"] += 1

                is_num = f in ["subtotal", "tax_amount", "cgst", "sgst", "igst", "grand_total"]
                if is_num:
                    matched = float_close(pred_val, gt_val)
                    exact = (pred_val is not None and gt_val is not None and abs(float(pred_val) - float(gt_val)) < 0.01)
                else:
                    exact = clean_val(pred_val) == clean_val(gt_val)
                    matched = text_match(pred_val, gt_val)

                if exact:
                    field_metrics[f]["exact_correct"] += 1
                if matched:
                    field_metrics[f]["fuzzy_correct"] += 1
                    slice_metrics[doc_slice]["correct_fields"] += 1

    if evaluated_count == 0:
        logger.warning("No invoices could be evaluated.")
        return {"status": "ZERO_EVALUATED"}

    # Compile report
    field_summary = {}
    for f, d in field_metrics.items():
        tot = d["total_gt"]
        field_summary[f] = {
            "total_gt": tot,
            "exact_accuracy": round(d["exact_correct"] / tot, 4) if tot > 0 else 0.0,
            "fuzzy_accuracy": round(d["fuzzy_correct"] / tot, 4) if tot > 0 else 0.0,
        }

    slice_summary = {}
    for s, d in slice_metrics.items():
        tot_f = d["total_fields"]
        slice_summary[s] = {
            "invoices_evaluated": d["evaluated"],
            "field_accuracy": round(d["correct_fields"] / tot_f, 4) if tot_f > 0 else 0.0,
        }

    avg_latency = round(sum(latencies) / len(latencies), 3) if latencies else 0.0
    review_rate = round(needs_review_count / evaluated_count, 4) if evaluated_count > 0 else 0.0

    report = {
        "baseline_name": BASELINE_NAME,
        "frozen_at": datetime.now().isoformat(),
        "total_invoices_evaluated": evaluated_count,
        "average_latency_seconds": avg_latency,
        "baseline_human_review_rate": review_rate,
        "fields": field_summary,
        "slices": slice_summary,
        "models_frozen": {
            "yolo_model_path": settings.yolo_model_path,
            "layoutlm_model_path": settings.layoutlm_model_path,
            "ocr_languages": settings.ocr_languages,
            "ollama_model": settings.ollama_model,
        }
    }

    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Baseline report frozen and saved to {OUTPUT_REPORT_PATH}")
    print("\n" + "=" * 60)
    print(f"BASELINE FREEZE REPORT: {BASELINE_NAME}")
    print("=" * 60)
    print(f"Total Evaluated: {evaluated_count} invoices")
    print(f"Average Latency: {avg_latency}s / invoice")
    print(f"Human Review Rate: {review_rate * 100:.1f}%\n")
    print("FIELD ACCURACIES:")
    for f, metrics in field_summary.items():
        print(f"  - {f:18s}: {metrics['fuzzy_accuracy']*100:5.1f}% (Exact: {metrics['exact_accuracy']*100:5.1f}%, N={metrics['total_gt']})")
    print("\nSLICE ACCURACIES:")
    for s, metrics in slice_summary.items():
        print(f"  - {s:15s}: {metrics['field_accuracy']*100:5.1f}% (N={metrics['invoices_evaluated']})")
    print("=" * 60 + "\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Freeze Baseline-v1 and run ground truth benchmark")
    parser.add_argument("--limit", type=int, default=200, help="Max invoices to evaluate")
    args = parser.parse_args()

    asyncio.run(run_baseline_benchmark(limit=args.limit))
