"""
scripts/auto_accept_high_confidence.py

CLI tool to batch auto-accept all high-confidence (>= 0.85) invoices with valid arithmetic
directly into verified Ground Truth records in the database.

Usage:
    python scripts/auto_accept_high_confidence.py --min-confidence 0.85
"""

import argparse
import asyncio
import sys
from pathlib import Path
from loguru import logger
from sqlalchemy import select, update

# Add root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from storage.db import DatabaseManager, InvoiceRecord


async def run_auto_accept(min_confidence: float = 0.85):
    settings = get_settings()
    db = DatabaseManager(settings.database_url)
    await db.init_db()

    logger.info(f"Scanning for invoices with overall_confidence >= {min_confidence}...")

    async with db.session() as session:
        stmt = select(
            InvoiceRecord.job_id,
            InvoiceRecord.filename,
            InvoiceRecord.overall_confidence,
            InvoiceRecord.review_reasons,
            InvoiceRecord.output_json,
        ).where(
            InvoiceRecord.output_json.isnot(None),
            InvoiceRecord.status.in_(["done", "partially_reviewed", "pending"]),
            InvoiceRecord.overall_confidence >= min_confidence,
        )
        result = await session.execute(stmt)
        rows = result.all()

    eligible_job_ids = []
    for job_id, filename, conf, review_reasons, output_json in rows:
        inv = output_json or {}
        reasons = review_reasons or inv.get("review_reasons", [])
        # Check that no fatal arithmetic errors exist
        if not reasons or all("error" not in str(err).lower() for err in reasons):
            eligible_job_ids.append((job_id, filename, conf))

    if not eligible_job_ids:
        logger.info("No eligible high-confidence pending invoices found.")
        return

    logger.info(f"Auto-accepting {len(eligible_job_ids)} high-confidence invoices...")

    ids_to_update = [item[0] for item in eligible_job_ids]
    async with db.session() as session:
        stmt = (
            update(InvoiceRecord)
            .where(InvoiceRecord.job_id.in_(ids_to_update))
            .values(
                status="reviewed",
                review_status="auto_accepted",
                needs_review=False,
                review_reasons=[],
            )
        )
        await session.execute(stmt)
        await session.commit()

    print("\n" + "=" * 60)
    print("   ACTIVE LEARNING BATCH AUTO-ACCEPT RESULTS")
    print("=" * 60)
    print(f"Total High-Confidence Invoices Auto-Accepted: {len(eligible_job_ids)}")
    print("-" * 60)
    for job_id, filename, conf in eligible_job_ids[:15]:
        print(f"[OK] {filename:<38} | Conf: {conf:.2f}")
    if len(eligible_job_ids) > 15:
        print(f"... and {len(eligible_job_ids) - 15} more.")
    print("=" * 60 + "\n")
    print("These invoices are now marked as VERIFIED Ground Truth.")
    print("Run `python scripts/export_reviewed_to_layoutlm.py` to include them in the dataset!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-accept high-confidence invoices")
    parser.add_argument("--min-confidence", type=float, default=0.85, help="Minimum confidence threshold (default 0.85)")
    args = parser.parse_args()

    asyncio.run(run_auto_accept(args.min_confidence))
