"""
scripts/build_locked_test_set.py

Permanently locks an independent Gold Test Evaluation Set from verified ground-truth invoices.
These samples are strictly excluded from all training sets and serve as the immutable benchmark
for Champion vs. Challenger model promotion.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from loguru import logger
from sqlalchemy import select

# Add root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from storage.db import DatabaseManager, InvoiceRecord
from scripts.export_reviewed_to_layoutlm import export_layoutlm_dataset, validate_dataset


LOCKED_TEST_DIR = Path("data/evaluation/locked_test")
LOCKED_IDS_FILE = LOCKED_TEST_DIR / "locked_job_ids.json"


async def build_locked_test_set(test_size: int = 10, force: bool = False):
    """
    Samples verified human ground truth invoices to construct a permanent locked evaluation set.
    Enforces exact 100% sample identity and stratified diversity across document types.
    """
    if LOCKED_IDS_FILE.exists() and not force:
        with open(LOCKED_IDS_FILE, "r", encoding="utf-8") as f:
            ids = json.load(f)
        logger.info(f"Locked test set already exists with {len(ids)} samples at {LOCKED_TEST_DIR}. Use --force to recreate.")
        return ids

    settings = get_settings()
    db = DatabaseManager(settings.database_url)
    await db.init_db()

    async with db.session() as session:
        # Prioritize human_corrected and human_confirmed
        stmt = select(InvoiceRecord).where(
            InvoiceRecord.output_json.isnot(None),
            InvoiceRecord.ground_truth_source.in_(["human_corrected", "human_confirmed"]),
        ).order_by(InvoiceRecord.created_at.desc())

        result = await session.execute(stmt)
        records = result.scalars().all()

        # Fallback if no explicit source tagged yet
        if not records:
            stmt_fallback = select(InvoiceRecord).where(
                InvoiceRecord.output_json.isnot(None),
                InvoiceRecord.status.in_(["reviewed", "partially_reviewed"]),
                InvoiceRecord.needs_review == False,
            ).order_by(InvoiceRecord.created_at.desc())
            records = (await session.execute(stmt_fallback)).scalars().all()

    if not records:
        logger.error("No verified ground-truth invoices found in DB to build locked test set.")
        return []

    # Stratify selection across document types to ensure holdout diversity
    by_type: dict[str, list[InvoiceRecord]] = {}
    for r in records:
        dt = getattr(r, "document_type", "UNKNOWN") or "UNKNOWN"
        by_type.setdefault(dt, []).append(r)

    selected_records = []
    # Round-robin sampling across available categories
    while len(selected_records) < test_size and any(by_type.values()):
        for dt, type_recs in list(by_type.items()):
            if type_recs and len(selected_records) < test_size:
                selected_records.append(type_recs.pop(0))

    locked_ids = [r.job_id for r in selected_records]

    logger.info(f"Locking {len(locked_ids)} verified samples for independent holdout evaluation across {len(by_type)} document types...")

    LOCKED_TEST_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOCKED_IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(locked_ids, f, indent=2)

    # Export these EXACT samples into data/evaluation/locked_test
    exported_ids = await export_layoutlm_dataset(
        output_dir=str(LOCKED_TEST_DIR),
        val_ratio=0.0,  # all samples in train/ directory of locked test
        tier="human_verified",
        include_job_ids=set(locked_ids),
        exclude_locked_test=False,
    )

    # Strict Integrity Audit: verify 100% exact exported sample matching
    exported_files = list((LOCKED_TEST_DIR / "train").glob("*.json"))
    exported_file_ids = {f.stem for f in exported_files}
    target_set = set(locked_ids)

    missing = target_set - exported_file_ids
    if missing:
        logger.error(f"Locked test construction integrity failure! {len(missing)} IDs were not exported: {missing}")
        raise RuntimeError(f"Locked test construction failed: {len(missing)} samples missing from export")

    logger.info(f"[OK] Locked evaluation benchmark verified: {len(exported_file_ids)}/{len(locked_ids)} exact samples locked at {LOCKED_TEST_DIR}")
    return locked_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build permanent locked test set for model evaluation")
    parser.add_argument("--size", type=int, default=10, help="Number of test samples to lock")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing locked test set")
    args = parser.parse_args()

    asyncio.run(build_locked_test_set(test_size=args.size, force=args.force))

