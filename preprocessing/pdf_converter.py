"""
PDF rasteriser + native text extractor using PyMuPDF.

DUAL-PATH ARCHITECTURE:
  - Digital PDFs (embedded text layer):
      → Extract text + bounding boxes directly from the PDF structure.
        No rasterization, no OCR. Confidence = 0.99.
        Returns NativePDFPage objects.
  - Scanned PDFs (image-only pages):
      → Rasterize at 300 DPI → InvoicePreprocessor (deskew/denoise).
        Returns standard PreprocessResult objects, same as before.

Each page is independently classified, so a single PDF can have a mix
of digital and scanned pages (e.g. page 1 = digital invoice, page 2 =
scanned attachment).
"""

import pymupdf
import numpy as np
from PIL import Image
from pathlib import Path
from loguru import logger
from dataclasses import dataclass, field
from typing import Union
from preprocessing.pipeline import InvoicePreprocessor, PreprocessResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NativeWord:
    """A single word extracted from a digital PDF text layer."""
    text: str
    bbox: list[float]       # [x0, y0, x1, y1] in page points
    block_no: int
    line_no: int
    word_no: int
    confidence: float = 0.99   # Native text — essentially perfect


@dataclass
class NativePDFPage:
    """
    Represents one page of a digital PDF.
    Contains the full native word list, structured line items, AND a low-DPI
    raster image for YOLO region detection and LayoutLMv3.
    """
    words: list[NativeWord]
    full_text: str
    image: np.ndarray          # BGR ndarray at 150 DPI (for YOLO)
    pil_image: Image.Image     # RGB PIL image (for LayoutLMv3)
    page_width: float          # original page width in points
    page_height: float         # original page height in points
    line_items: list[dict] = field(default_factory=list)
    dpi: int = 150
    is_digital: bool = True



# PageResult can be either a digital native page or a classic preprocessed page
PageResult = Union[NativePDFPage, PreprocessResult]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIGITAL_CHAR_THRESHOLD = 50    # pages with ≥ this many chars are considered digital


def _is_digital_page(page: pymupdf.Page) -> bool:
    """Return True if the page has an embedded text layer (not a scanned image)."""
    text = page.get_text("text")
    return len(text.strip()) >= _DIGITAL_CHAR_THRESHOLD


def _extract_native_words(page: pymupdf.Page) -> list[NativeWord]:
    """
    Extract all words with bounding boxes from the PDF text layer.
    PyMuPDF word format: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
    """
    words = []
    for w in page.get_text("words"):
        x0, y0, x1, y1, text, block_no, line_no, word_no = w
        text = text.strip()
        if not text:
            continue
        words.append(NativeWord(
            text=text,
            bbox=[x0, y0, x1, y1],
            block_no=block_no,
            line_no=line_no,
            word_no=word_no,
        ))
    return words


def _rasterize_page_low_dpi(page: pymupdf.Page, dpi: int = 150) -> tuple[np.ndarray, Image.Image]:
    """
    Rasterize a single page to a low-DPI image suitable for YOLO / LayoutLMv3.
    150 DPI is sufficient for region detection — keeps memory low.
    """
    zoom = dpi / 72.0
    mat = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    import cv2
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    pil = Image.fromarray(arr)  # RGB
    return bgr, pil


# ---------------------------------------------------------------------------
# PDFConverter
# ---------------------------------------------------------------------------

class PDFConverter:
    """
    Converts PDF pages to pre-processed invoice images or native text extracts.

    Digital pages return NativePDFPage (no OCR needed).
    Scanned pages return PreprocessResult (full OCR pipeline).
    """

    HIGH_DPI = 300          # for scanned pages → OCR quality
    LOW_DPI  = 150          # for digital pages → YOLO / LayoutLM image

    def __init__(self):
        self.preprocessor = InvoicePreprocessor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(self, pdf_path: str | Path) -> list[PageResult]:
        """Convert all pages of a PDF from a file path."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        doc = pymupdf.open(str(pdf_path))
        return self._process_doc(doc, name=pdf_path.name)

    def convert_bytes(self, pdf_bytes: bytes) -> list[PageResult]:
        """Convert all pages of a PDF from raw bytes (e.g. from an upload)."""
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        return self._process_doc(doc, name="<bytes>")

    # ------------------------------------------------------------------
    # Internal processing
    # ------------------------------------------------------------------

    def _process_doc(self, doc: pymupdf.Document, name: str) -> list[PageResult]:
        page_count = len(doc)
        logger.info(f"Processing PDF: {name} ({page_count} pages)")

        results = []
        digital_count = 0
        scanned_count = 0

        for page_num, page in enumerate(doc):
            logger.debug(f"Analysing page {page_num + 1}/{page_count}")

            if _is_digital_page(page):
                result = self._process_digital_page(page, page_num)
                digital_count += 1
            else:
                result = self._process_scanned_page(page, page_num)
                scanned_count += 1

            results.append(result)

        doc.close()
        logger.info(
            f"PDF conversion complete: {page_count} pages "
            f"({digital_count} digital, {scanned_count} scanned)"
        )
        return results

    def _process_digital_page(self, page: pymupdf.Page, page_num: int) -> NativePDFPage:
        """
        Extract native text + structured tables + low-DPI raster for a digital PDF page.
        No OCR is run. Text confidence = 0.99.
        """
        logger.debug(f"  Page {page_num + 1}: DIGITAL — extracting native text & tables")

        words = _extract_native_words(page)
        full_text = " ".join(w.text for w in words)
        bgr, pil = _rasterize_page_low_dpi(page, dpi=self.LOW_DPI)

        # Extract structured line items using TableExtractor
        from understanding.table_extractor import TableExtractor
        table_ext = TableExtractor()
        line_items = table_ext.extract_tables_from_page(page)

        logger.debug(
            f"  Page {page_num + 1}: {len(words)} native words, {len(line_items)} table items, "
            f"image={bgr.shape[1]}x{bgr.shape[0]}"
        )

        return NativePDFPage(
            words=words,
            full_text=full_text,
            image=bgr,
            pil_image=pil,
            page_width=page.rect.width,
            page_height=page.rect.height,
            line_items=line_items,
            dpi=self.LOW_DPI,
            is_digital=True,
        )


    def _process_scanned_page(self, page: pymupdf.Page, page_num: int) -> PreprocessResult:
        """
        Rasterize a scanned page at 300 DPI and run the full preprocessor.
        Identical behaviour to the old PDFConverter.
        """
        logger.debug(f"  Page {page_num + 1}: SCANNED — rasterising at {self.HIGH_DPI} DPI")
        zoom = self.HIGH_DPI / 72.0
        mat = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        result = self.preprocessor.process(img_bytes)
        return result

    # ------------------------------------------------------------------
    # Public helper: digital page detection (used by pipeline_runner.py)
    # ------------------------------------------------------------------

    @staticmethod
    def page_is_digital(page: pymupdf.Page) -> bool:
        """Public wrapper for external callers."""
        return _is_digital_page(page)
