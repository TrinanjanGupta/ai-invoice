"""
PDF rasteriser using PyMuPDF (fitz).
Converts each page of a PDF to a high-resolution image
and passes it through the pre-processing pipeline.
"""

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from pathlib import Path
from loguru import logger
from typing import Generator
from preprocessing.pipeline import InvoicePreprocessor, PreprocessResult


class PDFConverter:
    """
    Converts PDF pages to pre-processed invoice images.
    Handles multi-page PDFs, rotation detection, and embedded images.
    """

    DPI = 300
    ZOOM = DPI / 72  # PyMuPDF uses 72 DPI internally

    def __init__(self):
        self.preprocessor = InvoicePreprocessor()

    def convert(self, pdf_path: str | Path) -> list[PreprocessResult]:
        """
        Convert all pages of a PDF.
        Returns one PreprocessResult per page.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        results = []
        doc = fitz.open(str(pdf_path))
        logger.info(f"Processing PDF: {pdf_path.name} ({len(doc)} pages)")

        for page_num, page in enumerate(doc):
            logger.debug(f"Rasterising page {page_num + 1}/{len(doc)}")
            mat = fitz.Matrix(self.ZOOM, self.ZOOM)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("png")
            result = self.preprocessor.process(img_bytes)
            results.append(result)

        doc.close()
        logger.info(f"PDF conversion complete: {len(results)} pages")
        return results

    def convert_bytes(self, pdf_bytes: bytes) -> list[PreprocessResult]:
        """Convert a PDF from raw bytes (e.g. from an upload)."""
        results = []
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        logger.info(f"Processing PDF from bytes ({len(doc)} pages)")

        for page_num, page in enumerate(doc):
            mat = fitz.Matrix(self.ZOOM, self.ZOOM)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("png")
            result = self.preprocessor.process(img_bytes)
            results.append(result)

        doc.close()
        return results
