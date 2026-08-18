"""
Convert PDF invoices into high-resolution JPG/PNG images for Label Studio annotation and YOLO training.

Usage:
    python scripts/convert_pdfs_for_annotation.py --input-dir data/raw --output-dir data/images_to_annotate
"""

import argparse
import pymupdf
from pathlib import Path
from loguru import logger


def convert_pdfs_to_images(input_dir: str | Path, output_dir: str | Path, dpi: int = 200, format: str = "png"):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        return

    zoom = dpi / 72.0
    mat = pymupdf.Matrix(zoom, zoom)

    # Deduplicate in case filesystem is case-insensitive
    all_pdfs = set(input_path.glob("*.pdf")).union(set(input_path.glob("*.PDF")))
    pdf_files = sorted(list(all_pdfs))
    if not pdf_files:
        logger.warning(f"No PDF files found in: {input_path}")
        return

    logger.info(f"Found {len(pdf_files)} PDF(s) to convert...")
    total_images = 0

    for pdf_file in pdf_files:
        try:
            doc = pymupdf.open(str(pdf_file))
            page_count = len(doc)
            stem = pdf_file.stem.replace(" ", "_").replace("(", "").replace(")", "")
            for page_num in range(page_count):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                out_filename = f"{stem}_p{page_num + 1}.{format}"
                out_filepath = output_path / out_filename
                pix.save(str(out_filepath))
                total_images += 1
            doc.close()
            logger.info(f"Converted: {pdf_file.name} ({page_count} page(s))")
        except Exception as e:
            logger.error(f"Failed to convert {pdf_file.name}: {e}")

    logger.info(f"Success! Converted {len(pdf_files)} PDFs into {total_images} images in: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PDF invoices to images for annotation")
    parser.add_argument("--input-dir", type=str, default="data/raw", help="Directory containing PDF invoices")
    parser.add_argument("--output-dir", type=str, default="data/images_to_annotate", help="Directory to save converted images")
    parser.add_argument("--dpi", type=int, default=200, help="Image DPI resolution (default: 200)")
    parser.add_argument("--format", type=str, default="png", choices=["png", "jpg"], help="Output format (png or jpg)")
    args = parser.parse_args()

    convert_pdfs_to_images(args.input_dir, args.output_dir, args.dpi, args.format)
