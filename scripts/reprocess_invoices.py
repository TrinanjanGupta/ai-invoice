"""
Batch Re-scan / Reprocess Invoices script.

Re-runs the updated AI pipeline across existing invoices stored in SQLite/MinIO
or raw PDF files in data/raw.

Usage:
    python scripts/reprocess_invoices.py --unreviewed-only
    python scripts/reprocess_invoices.py --all
    python scripts/reprocess_invoices.py --job-id <job_id>
"""

import sys
import asyncio
import argparse
from pathlib import Path
from loguru import logger
from sqlalchemy import select

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from storage.db import DatabaseManager, InvoiceRecord, MinIOManager
from api.pipeline_runner import InvoicePipeline


def find_file(job_id: str, filename: str) -> Path | None:
    """Finds raw file on disk."""
    clean_stem = Path(filename).stem
    candidates = [
        Path(f"data/raw/{job_id}_{filename}"),
        Path(f"data/raw/{filename}"),
        Path(f"data/uploads/{job_id}_{filename}"),
        Path(f"data/uploads/{filename}"),
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c

    for s_dir in [Path("data/raw"), Path("data/uploads"), Path("data")]:
        if s_dir.exists():
            for p in s_dir.rglob(f"*{job_id[:8]}*"):
                if p.is_file():
                    return p
            for p in s_dir.rglob(f"*{clean_stem}*"):
                if p.is_file():
                    return p
    return None


async def reprocess_invoices(all_records: bool = False, target_job_id: str | None = None):
    settings = get_settings()
    db = DatabaseManager(settings.database_url)
    pipeline = InvoicePipeline(settings)

    try:
        minio = MinIOManager(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            bucket=settings.minio_bucket,
            secure=settings.minio_secure,
        )
    except Exception:
        minio = None

    async with db.session() as session:
        if target_job_id:
            stmt = select(InvoiceRecord).where(InvoiceRecord.job_id == target_job_id)
        elif all_records:
            stmt = select(InvoiceRecord).order_by(InvoiceRecord.created_at.desc())
        else:
            # Unreviewed only (exclude human-verified 'reviewed' status)
            stmt = select(InvoiceRecord).where(InvoiceRecord.status != "reviewed").order_by(InvoiceRecord.created_at.desc())

        records = list((await session.execute(stmt)).scalars().all())

    logger.info(f"Found {len(records)} invoice record(s) to re-process.")

    success_count = 0
    fail_count = 0

    for idx, rec in enumerate(records, start=1):
        job_id = rec.job_id
        filename = rec.filename
        print(f"\n[{idx}/{len(records)}] Reprocessing: {filename} (job={job_id[:8]} status={rec.status})")

        # 1. Load bytes
        file_bytes = None
        f_path = find_file(job_id, filename)
        if f_path and f_path.exists():
            file_bytes = f_path.read_bytes()
        elif minio and rec.storage_key:
            try:
                file_bytes = minio.download_file(rec.storage_key)
            except Exception as e:
                logger.warning(f"MinIO download failed: {e}")

        if not file_bytes:
            print(f"  ❌ Raw document bytes not found. Skipping.")
            fail_count += 1
            continue

        # 2. Run pipeline
        try:
            res = pipeline.process(file_bytes=file_bytes, filename=filename, job_id=job_id)
            inv = res.invoice
            valid = res.validation_report.is_valid

            # 3. Update DB
            async with db.session() as session:
                stmt = select(InvoiceRecord).where(InvoiceRecord.job_id == job_id)
                db_rec = (await session.execute(stmt)).scalar_one_or_none()
                if db_rec:
                    db_rec.status = "done"
                    db_rec.output_json = inv.model_dump()
                    db_rec.overall_confidence = inv.overall_confidence
                    db_rec.needs_review = inv.needs_review
                    db_rec.review_reasons = inv.review_reasons
                    await session.commit()

            print(f"  ✅ Complete! Conf: {inv.overall_confidence:.1%} | Valid: {valid} | Model: {res.model_used}")
            print(f"     Vendor: {inv.vendor_name!r} | Buyer: {inv.buyer_name!r} | Total: {inv.grand_total!r}")
            success_count += 1
        except Exception as e:
            logger.exception(f"Pipeline error on {filename}: {e}")
            fail_count += 1

    print("\n" + "=" * 60)
    print(f"Reprocessing Summary: {success_count} succeeded, {fail_count} failed out of {len(records)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Batch re-scan existing invoices with the latest pipeline.")
    parser.add_argument("--all", action="store_true", help="Reprocess ALL invoices, including reviewed ones.")
    parser.add_argument("--unreviewed-only", action="store_true", default=True, help="Reprocess only unreviewed / pending / done invoices.")
    parser.add_argument("--job-id", type=str, help="Specific job ID to reprocess.")
    args = parser.parse_args()

    asyncio.run(reprocess_invoices(all_records=args.all, target_job_id=args.job_id))


if __name__ == "__main__":
    main()
