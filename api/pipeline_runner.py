"""
Core pipeline orchestrator.
Ties all stages together in the correct order.

Usage:
    pipeline = InvoicePipeline(settings)
    result = pipeline.process(file_bytes, filename="invoice.pdf")
"""

import uuid
import tempfile
from pathlib import Path
from loguru import logger
from dataclasses import dataclass
from typing import Optional

from config.settings import Settings
from preprocessing.pipeline import InvoicePreprocessor
from preprocessing.pdf_converter import PDFConverter
from detection.detector import InvoiceDetector
from ocr.extractor import InvoiceOCR
from understanding.layoutlm import LayoutLMExtractor
from llm_fallback.ollama_client import OllamaClient
from validation.validator import InvoiceValidator, InvoiceSchema, ValidationReport
from output.renderer import InvoiceRenderer


@dataclass
class PipelineResult:
    job_id: str
    invoice: InvoiceSchema
    validation_report: ValidationReport
    html_output: str
    pdf_path: Optional[Path]
    raw_ocr_texts: dict
    page_count: int
    model_used: str   # "yolo+layoutlm", "heuristic+llm", etc.


class InvoicePipeline:
    """
    Full invoice digitization pipeline.
    Instantiate once and call process() for each invoice.
    """

    SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".bmp"}
    SUPPORTED_PDF_FORMATS = {".pdf"}

    def __init__(self, settings: Settings):
        self.settings = settings
        logger.info("Initialising Invoice Pipeline...")

        self.preprocessor = InvoicePreprocessor()
        self.pdf_converter = PDFConverter()
        self.detector = InvoiceDetector(model_path=settings.yolo_model_path)
        self.ocr = InvoiceOCR(lang="en", use_gpu=False)
        self.extractor = LayoutLMExtractor(
            model_path=settings.layoutlm_model_path,
            base_model=settings.layoutlm_base_model,
        )
        self.llm = OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        )
        self.validator = InvoiceValidator()
        self.renderer = InvoiceRenderer()

        logger.info("Pipeline initialised successfully")

    def process(
        self,
        file_bytes: bytes,
        filename: str,
        job_id: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> PipelineResult:
        """
        Main pipeline entry point.
        Accepts raw bytes of a PDF or image file.
        Returns a PipelineResult with the validated invoice and rendered outputs.
        """
        job_id = job_id or str(uuid.uuid4())
        suffix = Path(filename).suffix.lower()
        logger.info(f"[{job_id}] Processing: {filename}")

        # --- Stage 2: Pre-processing ---
        if suffix in self.SUPPORTED_PDF_FORMATS:
            logger.info(f"[{job_id}] Stage 2: PDF → images")
            pages = self.pdf_converter.convert_bytes(file_bytes)
        elif suffix in self.SUPPORTED_IMAGE_FORMATS:
            logger.info(f"[{job_id}] Stage 2: Image pre-processing")
            page = self.preprocessor.process(file_bytes)
            pages = [page]
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        page_count = len(pages)
        logger.info(f"[{job_id}] {page_count} page(s) to process")

        # For multi-page PDFs, use the first page that looks like an invoice
        # (highest text density). For now, process page 1.
        # TODO: merge multi-page results for split invoices.
        primary_page = pages[0]

        # --- Stage 3a: Region detection ---
        logger.info(f"[{job_id}] Stage 3a: Region detection")
        detection_result = self.detector.detect(primary_page.image)
        logger.info(
            f"[{job_id}] Detected {len(detection_result.regions)} regions "
            f"using {detection_result.model_used}"
        )

        # --- Stage 3b: OCR ---
        logger.info(f"[{job_id}] Stage 3b: OCR extraction")
        if detection_result.regions:
            ocr_results = self.ocr.extract_all_regions(detection_result.regions)
        else:
            # No regions detected — run OCR on full page
            logger.warning(f"[{job_id}] No regions detected, running full-page OCR")
            full_result = self.ocr.extract_full_page(primary_page.image)
            ocr_results = {"full_page": full_result}

        raw_ocr_texts = {k: v.full_text for k, v in ocr_results.items()}

        # --- Stage 4a: Field extraction (LayoutLMv3 or heuristic) ---
        logger.info(f"[{job_id}] Stage 4a: Field extraction")
        extracted = self.extractor.extract(ocr_results, image=primary_page.pil_image)

        # --- Stage 4b: LLM fallback for low-confidence fields ---
        logger.info(f"[{job_id}] Stage 4b: LLM confidence check")
        self.llm.enhance_low_confidence_fields(
            extracted,
            raw_ocr_texts,
            confidence_threshold=self.settings.llm_fallback_threshold,
        )

        # --- Stage 4c: Validation ---
        logger.info(f"[{job_id}] Stage 4c: Validation")
        invoice_schema, validation_report = self.validator.validate(extracted)

        # Attach confidence
        invoice_schema.overall_confidence = extracted.overall_confidence

        # --- Stage 5: Render ---
        logger.info(f"[{job_id}] Stage 5: Rendering output")
        html_output = self.renderer.to_html(invoice_schema)

        pdf_path = None
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            pdf_path = self.renderer.to_pdf(invoice_schema, out / f"{job_id}.pdf")

        model_parts = []
        model_parts.append(detection_result.model_used)
        model_parts.append("layoutlm" if self.extractor.model else "heuristic")
        if self.llm.is_available():
            model_parts.append("ollama")
        model_used = "+".join(model_parts)

        logger.info(
            f"[{job_id}] Done — confidence={invoice_schema.overall_confidence:.2f}, "
            f"needs_review={invoice_schema.needs_review}, model={model_used}"
        )

        return PipelineResult(
            job_id=job_id,
            invoice=invoice_schema,
            validation_report=validation_report,
            html_output=html_output,
            pdf_path=pdf_path,
            raw_ocr_texts=raw_ocr_texts,
            page_count=page_count,
            model_used=model_used,
        )
