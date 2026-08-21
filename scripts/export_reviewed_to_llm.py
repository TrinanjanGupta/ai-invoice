"""
Export Verified Invoices to LLM Instruction Fine-Tuning Format (Alpaca/ShareGPT JSONL)

Exports reviewed invoices from the PostgreSQL database into:
  data/llm_dataset/train.jsonl
  data/llm_dataset/val.jsonl

Only verified invoices (needs_review=False, status='reviewed') are included by default.
"""

import argparse
import asyncio
import json
import random
from pathlib import Path
from loguru import logger
from sqlalchemy import select

# Add project root to sys.path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from storage.db import DatabaseManager, InvoiceRecord
from config.settings import get_settings


INSTRUCTION_PROMPT = (
    "You are an expert AI Invoice Digitization Engine. "
    "Given the raw OCR text of an invoice, extract and return a valid JSON object containing all "
    "structured fields (invoice_number, invoice_date, vendor details, buyer details, line items, and totals) "
    "standardized according to Indian GST rules."
)


def get_invoice_image_path(job: InvoiceRecord) -> Path | None:
    raw_images_dir = Path("data/images_to_annotate")
    stem = Path(job.filename).stem
    candidates = list(raw_images_dir.glob(f"*{job.job_id}*")) or list(raw_images_dir.glob(f"*{stem}*"))
    if candidates:
        return candidates[0]

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
                try:
                    import pymupdf
                    doc = pymupdf.open(str(c))
                    page = doc[0]
                    pix = page.get_pixmap(dpi=150)
                    target_png = raw_images_dir / f"{job.job_id}_p1.png"
                    pix.save(str(target_png))
                    doc.close()
                    return target_png
                except Exception:
                    pass
            elif c.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
                return c
    return None


async def export_dataset(output_dir: Path, val_split: float = 0.2, only_verified: bool = True):
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    settings = get_settings()
    db = DatabaseManager(settings.database_url)
    output_dir.mkdir(parents=True, exist_ok=True)

    async with db.session_factory() as session:
        query = select(InvoiceRecord).where(InvoiceRecord.output_json.isnot(None))
        if only_verified:
            query = query.filter(
                InvoiceRecord.status.in_(["reviewed", "partially_reviewed", "done"]),
                InvoiceRecord.needs_review == False,
            )

        res = await session.execute(query)
        records = res.scalars().all()
        logger.info(f"Retrieved {len(records)} eligible invoice records from database.")

    samples = []
    for rec in records:
        img_p = get_invoice_image_path(rec)
        if not img_p:
            continue
        try:
            ocr_res = reader.readtext(str(img_p))
            input_text = "\n".join(b[1] for b in ocr_res if b and len(b) > 1 and b[1].strip())
            if not input_text.strip():
                continue

            clean_target = {k: v for k, v in rec.output_json.items() if k not in ["needs_review", "review_reasons", "overall_confidence"]}

            samples.append({
                "instruction": INSTRUCTION_PROMPT,
                "input": input_text.strip()[:4000],
                "output": json.dumps(clean_target, ensure_ascii=False, indent=2),
            })
        except Exception as e:
            logger.debug(f"Failed to process sample {rec.job_id}: {e}")

    if not samples:
        logger.warning("No valid samples could be exported. Verify invoices in the Review UI first.")
        return

    random.seed(42)
    random.shuffle(samples)

    split_idx = int(len(samples) * (1 - val_split))
    train_samples = samples[:split_idx] if split_idx > 0 else samples
    val_samples = samples[split_idx:] if split_idx < len(samples) else samples[:1]

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info(f"Dataset export complete!")
    logger.info(f"  Train samples ({len(train_samples)}): {train_path}")
    logger.info(f"  Val samples   ({len(val_samples)}):   {val_path}")


def main():
    parser = argparse.ArgumentParser(description="Export verified invoices to LLM fine-tuning JSONL format.")
    parser.add_argument("--output-dir", default="data/llm_dataset", help="Target output directory")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation set fraction (default: 0.2)")
    parser.add_argument("--include-unverified", action="store_true", help="Include unverified invoices")

    args = parser.parse_args()
    asyncio.run(export_dataset(
        output_dir=Path(args.output_dir),
        val_split=args.val_split,
        only_verified=not args.include_unverified,
    ))


if __name__ == "__main__":
    main()
