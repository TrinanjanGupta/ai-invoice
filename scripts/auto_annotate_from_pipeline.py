"""
scripts/auto_annotate_from_pipeline.py

Converts pipeline YOLO detections from reviewed invoices into proper
YOLO-format annotation files (.txt + image) inside data/annotations/.

This allows every reviewed invoice to contribute to YOLO training WITHOUT
having to draw boxes manually in Label Studio. Generated annotations are
pre-populated from live pipeline detections and only need a quick visual
verification in Label Studio (or Roboflow) before retraining.

WORKFLOW:
  1. Upload + process invoices in the Review UI
  2. Correct field errors → click Save (status becomes "reviewed")
  3. Run this script:
       python scripts/auto_annotate_from_pipeline.py
  4. Open Label Studio → import data/annotations/ → verify boxes (~5 sec each)
  5. Click "Retrain YOLOv8" in the Review UI

LABEL STUDIO ALIGNMENT:
  - Exports YOLOv8 format: class_id cx cy w h  (all normalised 0-1)
  - Uses the exact same 8-class list as data/annotations/dataset.yaml
  - Image files use uuid-prefix naming (same as existing annotated set)
  - Skips already-annotated pages (safe to re-run, idempotent)

Usage:
    python scripts/auto_annotate_from_pipeline.py
    python scripts/auto_annotate_from_pipeline.py --min-conf 0.40 --val-ratio 0.2
    python scripts/auto_annotate_from_pipeline.py --dry-run
    python scripts/auto_annotate_from_pipeline.py --status all

Classes (must match dataset.yaml):
    0: header        3: line_items    6: payment_terms
    1: vendor_block  4: totals_block  7: qr_barcode
    2: buyer_block   5: tax_block
"""

import argparse
import asyncio
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from PIL import Image

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import get_settings
from storage.db import DatabaseManager, InvoiceRecord

# ---------------------------------------------------------------------------
# Constants — must stay in sync with data/annotations/dataset.yaml
# ---------------------------------------------------------------------------
CLASSES = [
    "header",        # 0
    "vendor_block",  # 1
    "buyer_block",   # 2
    "line_items",    # 3
    "totals_block",  # 4
    "tax_block",     # 5
    "payment_terms", # 6
    "qr_barcode",    # 7
]
CLASS_ID = {name: idx for idx, name in enumerate(CLASSES)}

ANNOTATIONS_DIR = ROOT / "data" / "annotations"
IMAGES_TRAIN    = ANNOTATIONS_DIR / "images" / "train"
IMAGES_VAL      = ANNOTATIONS_DIR / "images" / "val"
LABELS_TRAIN    = ANNOTATIONS_DIR / "labels" / "train"
LABELS_VAL      = ANNOTATIONS_DIR / "labels" / "val"
RAW_DIR         = ROOT / "data" / "raw"

# Heuristic fallback grid (used when YOLO model is not yet trained)
HEURISTIC_REGIONS = [
    ("header",        0.0,  0.0,  1.0,  0.12),
    ("vendor_block",  0.0,  0.12, 0.5,  0.30),
    ("buyer_block",   0.5,  0.12, 1.0,  0.30),
    ("line_items",    0.0,  0.30, 1.0,  0.65),
    ("totals_block",  0.5,  0.65, 1.0,  0.80),
    ("tax_block",     0.0,  0.65, 0.5,  0.80),
    ("payment_terms", 0.0,  0.80, 1.0,  0.92),
]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """Convert absolute pixel bbox to YOLO normalised cx, cy, w, h."""
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return (
        round(max(0.0, min(1.0, cx)), 6),
        round(max(0.0, min(1.0, cy)), 6),
        round(max(0.001, min(1.0, w)), 6),
        round(max(0.001, min(1.0, h)), 6),
    )


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def get_existing_stems() -> set:
    """Return stems of all already-annotated images (train + val)."""
    stems = set()
    for d in [LABELS_TRAIN, LABELS_VAL]:
        if d.exists():
            for f in d.glob("*.txt"):
                stems.add(f.stem)
    return stems


def build_stem(job_id: str, filename: str, page_idx: int) -> str:
    """
    Build annotation filename stem.
    Format: <job_id[:8]>-<safe_filename>_p<page>
    Matches existing convention: e.g. 12db8d26-Adobe_Scan_14_Jul_2026_10_p2
    """
    short_id  = job_id[:8]
    safe_name = (
        Path(filename).stem
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )
    return f"{short_id}-{safe_name}_p{page_idx + 1}"


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def find_raw_file(job_id: str, filename: str) -> Path | None:
    """
    Locate the raw invoice file in data/raw/.
    Upload saves files as: data/raw/<job_id>_<original_filename>
    Falls back to bare <filename> for manually placed files.
    """
    # Primary: job_id-prefixed name (set by upload endpoint)
    prefixed = RAW_DIR / f"{job_id}_{filename}"
    if prefixed.exists():
        return prefixed

    # Secondary: scan by short job_id prefix
    for f in RAW_DIR.glob(f"{job_id[:8]}*"):
        return f

    # Tertiary: full job_id prefix
    for f in RAW_DIR.glob(f"{job_id}*"):
        return f

    # Fallback: bare filename (manually added to data/raw/)
    bare = RAW_DIR / filename
    if bare.exists():
        return bare

    return None


def _rasterize_pdf(pdf_path: Path, dpi: int = 150) -> list:
    """
    Rasterise all pages of a PDF directly via PyMuPDF.
    Returns list of BGR numpy arrays.
    DPI=150 is sufficient for YOLO region detection and annotation.
    """
    import pymupdf
    zoom = dpi / 72.0
    mat  = pymupdf.Matrix(zoom, zoom)
    pages = []
    doc = pymupdf.open(str(pdf_path))
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        # pix.samples is RGB bytes
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        pages.append(bgr)
    doc.close()
    return pages


def _rasterize_image(img_path: Path) -> list:
    """Load a single image file as a BGR numpy array."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"cv2 could not read: {img_path}")
    return [img]


def load_pages(job_id: str, filename: str) -> list:
    """
    Load all pages from data/raw/ as BGR numpy arrays.
    Uses direct PyMuPDF rasterisation — no binarisation, no deskew.
    This is intentional: YOLO detection works on the raw image.
    """
    raw_path = find_raw_file(job_id, filename)
    if not raw_path:
        logger.warning(f"  Raw file not found for job={job_id[:8]} filename={filename}")
        return []

    suffix = raw_path.suffix.lower()

    if suffix == ".pdf":
        try:
            logger.info(f"  Rasterising {raw_path.name} at 150 DPI")
            return _rasterize_pdf(raw_path, dpi=150)
        except Exception as e:
            logger.error(f"  PDF rasterisation error: {e}")
            return []

    elif suffix in {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}:
        try:
            return _rasterize_image(raw_path)
        except Exception as e:
            logger.error(f"  Image load error: {e}")
            return []

    else:
        logger.warning(f"  Unsupported format: {suffix}")
        return []



# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def run_detection(image: np.ndarray, conf_threshold: float) -> list:
    """
    Run YOLO detector on one page image.
    Returns list of detection dicts.
    Falls back to heuristic grid if YOLO model is not loaded.
    """
    settings = get_settings()
    from detection.detector import InvoiceDetector

    detector = InvoiceDetector(model_path=settings.yolo_model_path)
    det_result = detector.detect(image, conf_threshold=conf_threshold)

    if det_result.model_used == "yolo" and det_result.regions:
        return [
            {
                "class_id":  r.class_id,
                "label":     r.label,
                "confidence": r.confidence,
                "bbox":      r.bbox,   # (x1, y1, x2, y2) pixels
                "source":    "yolo",
            }
            for r in det_result.regions
        ]

    # Heuristic fallback
    logger.info("  No YOLO model — using heuristic layout seed (needs verify)")
    h, w = image.shape[:2]
    return [
        {
            "class_id":   CLASS_ID[label],
            "label":      label,
            "confidence": 0.0,
            "bbox":       (int(rx1*w), int(ry1*h), int(rx2*w), int(ry2*h)),
            "source":     "heuristic",
        }
        for label, rx1, ry1, rx2, ry2 in HEURISTIC_REGIONS
        if label in CLASS_ID
    ]


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def write_label(path: Path, detections: list, img_w: int, img_h: int):
    """Write YOLO .txt annotation file."""
    lines = []
    for det in detections:
        cx, cy, w, h = xyxy_to_yolo(*det["bbox"], img_w, img_h)
        lines.append(f"{det['class_id']} {cx} {cy} {w} {h}")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_image(image: np.ndarray, path: Path):
    """Save BGR ndarray as PNG."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(str(path), format="PNG")


# ---------------------------------------------------------------------------
# Per-invoice processing
# ---------------------------------------------------------------------------

def annotate_record(record, existing_stems: set, conf_threshold: float,
                    dry_run: bool, val_ratio: float) -> dict:
    job_id   = record.job_id
    filename = record.filename
    logger.info(f"Invoice: {filename}  (job={job_id[:8]})")

    pages = load_pages(job_id, filename)
    if not pages:
        return {"job_id": job_id, "filename": filename,
                "new": 0, "skipped": 0, "reason": "file not found"}

    new_pages = 0
    skipped   = 0
    page_results = []

    for p_idx, page_img in enumerate(pages):
        stem = build_stem(job_id, filename, p_idx)

        if stem in existing_stems:
            logger.debug(f"  p{p_idx+1}: already annotated — skip")
            skipped += 1
            continue

        detections = run_detection(page_img, conf_threshold)
        if not detections:
            logger.warning(f"  p{p_idx+1}: no regions found — skip")
            skipped += 1
            continue

        h, w = page_img.shape[:2]
        split   = "val" if random.random() < val_ratio else "train"
        img_dir = IMAGES_TRAIN if split == "train" else IMAGES_VAL
        lbl_dir = LABELS_TRAIN if split == "train" else LABELS_VAL

        img_dest = img_dir / f"{stem}.png"
        lbl_dest = lbl_dir / f"{stem}.txt"

        sources      = {d["source"] for d in detections}
        needs_verify = "heuristic" in sources or any(d["confidence"] < 0.50 for d in detections)

        logger.info(
            f"  p{p_idx+1}: {len(detections)} regions | split={split} | "
            f"sources={sources} | needs_verify={needs_verify}"
        )

        if not dry_run:
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            write_image(page_img, img_dest)
            write_label(lbl_dest, detections, w, h)
            existing_stems.add(stem)

        new_pages += 1
        page_results.append({
            "stem":         stem,
            "split":        split,
            "detections":   len(detections),
            "needs_verify": needs_verify,
            "sources":      list(sources),
        })

    return {
        "job_id":   job_id,
        "filename": filename,
        "new":      new_pages,
        "skipped":  skipped,
        "pages":    page_results,
    }


# ---------------------------------------------------------------------------
# DB fetch
# ---------------------------------------------------------------------------

async def fetch_records(status_filter: str) -> list:
    settings = get_settings()
    db = DatabaseManager(settings.database_url)
    from sqlalchemy import select

    async with db.session() as session:
        if status_filter == "all":
            stmt = select(InvoiceRecord).where(
                InvoiceRecord.status.notin_(["pending", "processing", "failed", "deleted"])
            )
        else:
            stmt = select(InvoiceRecord).where(InvoiceRecord.status == status_filter)
        result = await session.execute(stmt)
        return result.scalars().all()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(summaries: list, dry_run: bool):
    total_new    = sum(s.get("new", 0) for s in summaries)
    total_skip   = sum(s.get("skipped", 0) for s in summaries)
    need_verify  = sum(
        1 for s in summaries
        for p in s.get("pages", [])
        if p.get("needs_verify")
    )

    mode = "[DRY RUN] " if dry_run else ""
    print()
    print("=" * 60)
    print(f"  {mode}Auto-Annotation Report")
    print("=" * 60)
    print(f"  Invoices processed : {len(summaries)}")
    print(f"  New pages written  : {total_new}")
    print(f"  Pages skipped      : {total_skip}  (already annotated)")
    print(f"  Need visual verify : {need_verify}  (low-conf or heuristic seed)")
    if not dry_run and total_new > 0:
        train_n = len(list(IMAGES_TRAIN.glob("*.png")))
        val_n   = len(list(IMAGES_VAL.glob("*.png")))
        print()
        print(f"  data/annotations totals after run:")
        print(f"    Train images : {train_n}")
        print(f"    Val   images : {val_n}")
    print()
    print("  Next steps:")
    print("  1. (Optional) Open Label Studio -> import data/annotations/")
    print("     Project -> Settings -> Import -> YOLO format")
    print("     Verify / adjust boxes on 'needs_verify' pages (~5 sec each)")
    print("  2. Click 'Retrain YOLOv8' in the Review UI training modal")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Auto-generate YOLO annotations from reviewed pipeline detections"
    )
    p.add_argument("--min-conf", type=float, default=0.35,
                   help="Min YOLO confidence to include a region (default: 0.35)")
    p.add_argument("--val-ratio", type=float, default=0.2,
                   help="Fraction of pages assigned to val split (default: 0.2)")
    p.add_argument("--status", default="reviewed",
                   choices=["reviewed", "completed", "all"],
                   help="DB invoice status to include (default: reviewed)")
    p.add_argument("--dry-run", action="store_true",
                   help="Preview actions without writing any files")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducible train/val split")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    logger.info("Auto-Annotation from Pipeline")
    logger.info(f"  min-conf={args.min_conf}  val-ratio={args.val_ratio}  "
                f"status={args.status}  dry-run={args.dry_run}")

    # Ensure output dirs exist
    for d in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
        d.mkdir(parents=True, exist_ok=True)

    records = asyncio.run(fetch_records(args.status))
    logger.info(f"Found {len(records)} invoice record(s) with status='{args.status}'")

    if not records:
        logger.warning("No records found. Upload and review invoices first.")
        return

    existing_stems = get_existing_stems()
    logger.info(f"Already annotated: {len(existing_stems)} page(s)")

    summaries = []
    for rec in records:
        summary = annotate_record(
            record=rec,
            existing_stems=existing_stems,
            conf_threshold=args.min_conf,
            dry_run=args.dry_run,
            val_ratio=args.val_ratio,
        )
        summaries.append(summary)

    print_report(summaries, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
