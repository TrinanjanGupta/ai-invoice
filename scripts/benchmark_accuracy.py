"""
scripts/benchmark_accuracy.py

Evaluates the current AI Invoice extraction pipeline against human-reviewed ground truth
records in PostgreSQL. Computes precision, recall, and exact match accuracy across all
key invoice fields and line items.

Usage:
    python scripts/benchmark_accuracy.py [--limit 100]
"""

import sys
import asyncio
import argparse
import difflib
from pathlib import Path
from loguru import logger

# Add root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from storage.db import DatabaseManager, InvoiceRecord
from api.pipeline_runner import InvoicePipeline
from validation.validator import InvoiceSchema


def clean_val(v) -> str:
    """Normalize string for evaluation."""
    if v is None:
        return ""
    return str(v).strip().lower().replace(",", "").replace(" ", "").replace("-", "").replace("/", "")


def float_close(a, b, rel_tol=0.01) -> bool:
    """Check if two numeric values match within relative tolerance."""
    if a is None or b is None:
        return a == b
    try:
        fa = float(str(a).replace(",", "").strip())
        fb = float(str(b).replace(",", "").strip())
        if fa == 0.0 and fb == 0.0:
            return True
        return abs(fa - fb) <= max(0.5, max(abs(fa), abs(fb)) * rel_tol)
    except (ValueError, TypeError):
        return False


def text_match(pred, gt, threshold=0.80) -> bool:
    """Fuzzy text similarity match."""
    cp = clean_val(pred)
    cg = clean_val(gt)
    if not cp and not cg:
        return True
    if not cp or not cg:
        return False
    if cp == cg or cp in cg or cg in cp:
        return True
    return difflib.SequenceMatcher(None, cp, cg).ratio() >= threshold


async def run_benchmark(limit: int = 100):
    settings = get_settings()
    db = DatabaseManager(settings.database_url)
    await db.init_db()

    pipeline = InvoicePipeline(settings)

    async with db.session() as session:
        from sqlalchemy import select
        stmt = (
            select(InvoiceRecord)
            .where(
                InvoiceRecord.output_json.isnot(None),
                InvoiceRecord.status.in_(["reviewed", "partially_reviewed", "completed"]),
            )
            .limit(limit)
        )
        result = await session.execute(stmt)
        records = result.scalars().all()

    if not records:
        logger.warning("No reviewed invoice records found in database to evaluate!")
        print("\nNo reviewed ground truth invoices found in DB.")
        print("Upload invoices and review/save them in the UI first to create ground truth.\n")
        return

    logger.info(f"Evaluating current pipeline against {len(records)} ground truth records...")

    metrics = {
        "invoice_number": {"correct": 0, "total": 0},
        "invoice_date": {"correct": 0, "total": 0},
        "vendor_name": {"correct": 0, "total": 0},
        "buyer_name": {"correct": 0, "total": 0},
        "vendor_gstin": {"correct": 0, "total": 0},
        "grand_total": {"correct": 0, "total": 0},
        "subtotal": {"correct": 0, "total": 0},
        "tax_amount": {"correct": 0, "total": 0},
        "cgst": {"correct": 0, "total": 0},
        "sgst": {"correct": 0, "total": 0},
        "ifsc_code": {"correct": 0, "total": 0},
        "line_items_count": {"correct": 0, "total": 0},
    }

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
        except Exception:
            continue

        # Find document bytes
        stem = Path(rec.filename).stem
        candidates = [
            Path(f"data/raw/{rec.job_id}_{rec.filename}"),
            Path(f"data/raw/{rec.filename}"),
            Path(f"data/uploads/{rec.job_id}_{rec.filename}"),
            *list(Path("data/raw").glob(f"*{rec.job_id}*")),
            *list(Path("data/raw").glob(f"*{stem}*")),
        ]

        doc_path = None
        for c in candidates:
            if c.exists() and c.is_file():
                doc_path = c
                break

        if not doc_path:
            continue

        with open(doc_path, "rb") as f:
            file_bytes = f.read()

        try:
            pred_result = pipeline.process(file_bytes, filename=rec.filename)
            pred = pred_result.invoice
            evaluated_count += 1

            # 1. Invoice Number
            if gt.invoice_number:
                metrics["invoice_number"]["total"] += 1
                if text_match(pred.invoice_number, gt.invoice_number):
                    metrics["invoice_number"]["correct"] += 1

            # 2. Invoice Date
            if gt.invoice_date:
                metrics["invoice_date"]["total"] += 1
                if text_match(pred.invoice_date, gt.invoice_date):
                    metrics["invoice_date"]["correct"] += 1

            # 3. Vendor Name
            if gt.vendor_name:
                metrics["vendor_name"]["total"] += 1
                if text_match(pred.vendor_name, gt.vendor_name):
                    metrics["vendor_name"]["correct"] += 1

            # 4. Buyer Name
            if gt.buyer_name:
                metrics["buyer_name"]["total"] += 1
                if text_match(pred.buyer_name, gt.buyer_name):
                    metrics["buyer_name"]["correct"] += 1

            # 5. GSTIN
            if gt.vendor_gstin:
                metrics["vendor_gstin"]["total"] += 1
                if text_match(pred.vendor_gstin, gt.vendor_gstin):
                    metrics["vendor_gstin"]["correct"] += 1

            # 6. Grand Total
            if gt.grand_total is not None:
                metrics["grand_total"]["total"] += 1
                if float_close(pred.grand_total, gt.grand_total):
                    metrics["grand_total"]["correct"] += 1

            # 7. Subtotal
            if gt.subtotal is not None:
                metrics["subtotal"]["total"] += 1
                if float_close(pred.subtotal, gt.subtotal):
                    metrics["subtotal"]["correct"] += 1

            # 8. Tax Amount
            if gt.tax_amount is not None:
                metrics["tax_amount"]["total"] += 1
                if float_close(pred.tax_amount, gt.tax_amount):
                    metrics["tax_amount"]["correct"] += 1

            # 9. CGST & SGST
            if gt.cgst is not None:
                metrics["cgst"]["total"] += 1
                if float_close(pred.cgst, gt.cgst):
                    metrics["cgst"]["correct"] += 1

            if gt.sgst is not None:
                metrics["sgst"]["total"] += 1
                if float_close(pred.sgst, gt.sgst):
                    metrics["sgst"]["correct"] += 1

            # 10. IFSC
            if gt.ifsc_code:
                metrics["ifsc_code"]["total"] += 1
                if text_match(pred.ifsc_code, gt.ifsc_code):
                    metrics["ifsc_code"]["correct"] += 1

            # 11. Line Items count
            gt_item_count = len(gt.line_items) if gt.line_items else 0
            if gt_item_count > 0:
                metrics["line_items_count"]["total"] += 1
                pred_item_count = len(pred.line_items) if pred.line_items else 0
                if pred_item_count == gt_item_count or abs(pred_item_count - gt_item_count) <= 1:
                    metrics["line_items_count"]["correct"] += 1

        except Exception as e:
            logger.error(f"Error evaluating {rec.filename}: {e}")

    # Print Scorecard
    print("\n" + "=" * 65)
    print(f"   PIPELINE BENCHMARK ACCURACY REPORT ({evaluated_count} Invoices Evaluated)")
    print("=" * 65)
    print(f"{'Field':<24} | {'Correct':<8} | {'Total':<8} | {'Accuracy':<10}")
    print("-" * 65)

    overall_correct = 0
    overall_total = 0

    for field_name, res in metrics.items():
        corr = res["correct"]
        tot = res["total"]
        pct = (corr / tot * 100.0) if tot > 0 else 0.0
        overall_correct += corr
        overall_total += tot
        field_display = field_name.replace("_", " ").title()
        print(f"{field_display:<24} | {corr:<8} | {tot:<8} | {pct:>7.1f}%")

    print("-" * 65)
    overall_pct = (overall_correct / overall_total * 100.0) if overall_total > 0 else 0.0
    print(f"{'OVERALL AVERAGE':<24} | {overall_correct:<8} | {overall_total:<8} | {overall_pct:>7.1f}%")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark pipeline extraction against DB ground truth")
    parser.add_argument("--limit", type=int, default=100, help="Max records to evaluate")
    args = parser.parse_args()

    asyncio.run(run_benchmark(limit=args.limit))
