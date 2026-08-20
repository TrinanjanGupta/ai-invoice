"""
Core pipeline orchestrator — Dual-Path Architecture.

PATH A — Digital PDFs (native text layer detected):
  PDFConverter extracts words + bboxes directly from the PDF.
  No OCR is run. Text confidence = 0.99.
  A low-DPI raster is generated only for YOLO region detection.

PATH B — Scanned PDFs / Images (no text layer):
  PDFConverter rasterises at 300 DPI + InvoicePreprocessor.
  PaddleOCR runs on each YOLO-detected region crop.

Both paths feed into the same LayoutLM / Heuristic / LLM ensemble.

Usage:
    pipeline = InvoicePipeline(settings)
    result = pipeline.process(file_bytes, filename="invoice.pdf")
"""

import uuid
from pathlib import Path
from loguru import logger
from dataclasses import dataclass
from typing import Optional, Any, Callable

import pymupdf
from PIL import Image

from config.settings import Settings
from preprocessing.pipeline import InvoicePreprocessor
from preprocessing.pdf_converter import PDFConverter, NativePDFPage
from detection.detector import InvoiceDetector
from ocr.extractor import InvoiceOCR, OCRResult, TextBlock
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
    model_used: str   # e.g. "native_pdf+yolo+layoutlm"


# ---------------------------------------------------------------------------
# Native-text → OCRResult adapter
# ---------------------------------------------------------------------------

def _native_page_to_ocr_results(
    native_page: NativePDFPage,
    det_regions: list,
) -> dict[str, OCRResult]:
    """
    Convert native PDF words into OCRResult dicts keyed by region label.

    Words are grouped by their (block_no, line_no) to produce ONE TextBlock
    per PDF line — exactly matching the structure PaddleOCR produces.
    This is critical: _extract_heuristic relies on full_text.split("\\n")
    to separate vendor name from address, header fields from body, etc.

    Strategy:
    - If YOLO detected regions, assign each LINE to the region whose bbox
      contains the line's centre point.
    - If no YOLO regions, emit a single "full_page" OCRResult.
    """
    img_w = native_page.image.shape[1]
    img_h = native_page.image.shape[0]
    pg_w  = native_page.page_width
    pg_h  = native_page.page_height

    # Scale: PDF point space → image pixel space
    sx = img_w / pg_w if pg_w > 0 else 1.0
    sy = img_h / pg_h if pg_h > 0 else 1.0

    # ── Group words into lines: (block_no, line_no) → list of NativeWord ──
    from collections import defaultdict
    line_groups: dict[tuple, list] = defaultdict(list)
    for w in native_page.words:
        line_groups[(w.block_no, w.line_no)].append(w)

    # Sort lines by their vertical position (top-to-bottom, left-to-right)
    sorted_lines = sorted(
        line_groups.values(),
        key=lambda ws: (min(w.bbox[1] for w in ws), min(w.bbox[0] for w in ws))
    )

    # Build TextBlock per line
    def line_to_textblock(line_words: list, region_label: str) -> TextBlock:
        text = " ".join(w.text for w in line_words)
        x0 = min(w.bbox[0] for w in line_words) * sx
        y0 = min(w.bbox[1] for w in line_words) * sy
        x1 = max(w.bbox[2] for w in line_words) * sx
        y1 = max(w.bbox[3] for w in line_words) * sy
        return TextBlock(
            text=text,
            confidence=0.99,
            bbox=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            region_label=region_label,
        )

    if not det_regions:
        # No regions: emit one big full_page OCRResult with newline-delimited lines
        blocks = [line_to_textblock(lw, "full_page") for lw in sorted_lines]
        full_text = "\n".join(b.text for b in blocks)
        return {
            "full_page": OCRResult(
                region_label="full_page",
                text_blocks=blocks,
                full_text=full_text,
                avg_confidence=0.99,
            )
        }

    # ── Assign each line to a YOLO region by its centre point ─────────────
    region_line_groups: dict[str, list[list]] = {r.label: [] for r in det_regions}
    overflow_label = det_regions[0].label  # fallback

    for line_words in sorted_lines:
        # Centre of this line in pixel coords
        cx = (min(w.bbox[0] for w in line_words) + max(w.bbox[2] for w in line_words)) / 2 * sx
        cy = (min(w.bbox[1] for w in line_words) + max(w.bbox[3] for w in line_words)) / 2 * sy

        placed = False
        for region in det_regions:
            rx1, ry1, rx2, ry2 = region.bbox
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                region_line_groups[region.label].append(line_words)
                placed = True
                break
        if not placed:
            region_line_groups[overflow_label].append(line_words)

    results = {}
    for region in det_regions:
        lines = region_line_groups[region.label]
        blocks = [line_to_textblock(lw, region.label) for lw in lines]
        full_text = "\n".join(b.text for b in blocks)
        results[region.label] = OCRResult(
            region_label=region.label,
            text_blocks=blocks,
            full_text=full_text,
            avg_confidence=0.99 if blocks else 0.0,
        )
    return results



# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class InvoicePipeline:
    """
    Full invoice digitization pipeline.
    Instantiate once and call process() for each invoice.
    """

    SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".bmp"}
    SUPPORTED_PDF_FORMATS   = {".pdf"}

    def __init__(self, settings: Settings):
        self.settings = settings
        logger.info("Initialising Invoice Pipeline...")

        self.preprocessor = InvoicePreprocessor()
        self.pdf_converter = PDFConverter()
        self.detector      = InvoiceDetector(model_path=settings.yolo_model_path)
        self.ocr           = InvoiceOCR(lang="en", use_gpu=False)
        self.extractor     = LayoutLMExtractor(
            model_path=settings.layoutlm_model_path,
            base_model=settings.layoutlm_base_model,
        )
        self.llm      = OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        )
        self.validator = InvoiceValidator()
        self.renderer  = InvoiceRenderer()

        logger.info("Pipeline initialised successfully")

    def process(
        self,
        file_bytes: bytes,
        filename: str,
        job_id: Optional[str] = None,
        output_dir: Optional[str] = None,
        stage_callback: Optional[Any] = None,
    ) -> PipelineResult:
        """
        Main pipeline entry point.
        Accepts raw bytes of a PDF or image file.
        Returns a PipelineResult with the validated invoice and rendered outputs.
        """
        job_id = job_id or str(uuid.uuid4())
        suffix = Path(filename).suffix.lower()
        logger.info(f"[{job_id}] Processing: {filename}")

        def _notify(stage: str, stage_idx: int, progress: int, label: str):
            if stage_callback:
                try:
                    stage_callback(stage, stage_idx, progress, label)
                except Exception as ex:
                    logger.debug(f"stage_callback error: {ex}")

        # ── Stage 1: Pre-processing ──────────────────────────────────────────
        _notify("preprocessing", 1, 15, "Pre-processing: Deskew, denoise & auto-orient")
        if suffix in self.SUPPORTED_PDF_FORMATS:
            logger.info(f"[{job_id}] Stage 2: PDF → dual-path conversion")
            pages = self.pdf_converter.convert_bytes(file_bytes)
        elif suffix in self.SUPPORTED_IMAGE_FORMATS:
            logger.info(f"[{job_id}] Stage 2: Image pre-processing")
            page = self.preprocessor.process(file_bytes)
            pages = [page]
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        page_count = len(pages)
        logger.info(f"[{job_id}] {page_count} page(s) to process")

        # ── Stage 2: Per-page Detection + Text Extraction ───────────────────
        combined_ocr_results: dict[str, OCRResult] = {}
        all_raw_ocr_texts:    dict[str, str]       = {}
        detected_models:      list[str]            = []
        extracted_per_page                         = []
        path_summary:         list[str]            = []

        for p_idx, p_obj in enumerate(pages):
            logger.info(f"[{job_id}] Processing Page {p_idx + 1}/{page_count}...")

            # ── YOLO region detection ──────────────────────────────────────
            _notify("detection", 2, 35, f"Region Detection: YOLOv8 identifying zones (Page {p_idx + 1}/{page_count})")
            det_res = self.detector.detect(p_obj.image)
            detected_models.append(det_res.model_used)
            logger.info(
                f"[{job_id}] Page {p_idx+1}: {len(det_res.regions)} regions "
                f"via {det_res.model_used}"
            )

            # ── PATH A: Digital PDF page ───────────────────────────────────
            if isinstance(p_obj, NativePDFPage):
                path_summary.append("native_pdf")
                _notify("ocr", 3, 50, f"Extracting native PDF text (Page {p_idx + 1}/{page_count})")
                logger.info(
                    f"[{job_id}] Page {p_idx+1}: PATH A — native text "
                    f"({len(p_obj.words)} words, conf=0.99)"
                )
                p_ocr = _native_page_to_ocr_results(p_obj, det_res.regions)
                pil_image = p_obj.pil_image

            # ── PATH B: Scanned page / direct image ────────────────────────
            else:
                path_summary.append("ocr")
                _notify("ocr", 3, 60, f"OCR Extraction: PaddleOCR reading text (Page {p_idx + 1}/{page_count})")
                logger.info(f"[{job_id}] Page {p_idx+1}: PATH B — Scanned Image OCR")

                p_ocr = {}
                # Always extract full page so global context is never lost on diverse templates
                full_res = self.ocr.extract_full_page(p_obj.image)
                p_ocr[f"full_page_p{p_idx+1}"] = full_res

                if det_res.regions:
                    _notify("ocr", 3, 68, f"OCR Extraction: Reading {len(det_res.regions)} structured region blocks (Page {p_idx + 1}/{page_count})")
                    region_ocr = self.ocr.extract_all_regions(det_res.regions)
                    p_ocr.update(region_ocr)

                pil_image = p_obj.pil_image

            # Accumulate OCR results
            for k, v in p_ocr.items():
                unique_key = k if k not in combined_ocr_results else f"{k}_p{p_idx+1}"
                combined_ocr_results[unique_key] = v
                all_raw_ocr_texts[unique_key] = v.full_text

            # ── Extraction (LayoutLM / heuristic) ─────────────────────────
            _notify("understanding", 4, 78, f"AI Understanding: Mapping fields (Page {p_idx + 1}/{page_count})")
            p_extracted = self.extractor.extract(p_ocr, image=pil_image)
            if isinstance(p_obj, NativePDFPage) and p_obj.line_items:
                p_extracted.line_items = p_obj.line_items
            extracted_per_page.append(p_extracted)

        # ── Stage 4a: Merge across pages ────────────────────────────────────
        _notify("understanding", 4, 84, "AI Understanding: Reconciling document layout & tables")
        logger.info(f"[{job_id}] Stage 4a: Multi-page merge")
        extracted = extracted_per_page[0]
        for next_p in extracted_per_page[1:]:
            extracted = self.extractor._merge_invoices(extracted, next_p)
            if next_p.line_items:
                extracted.line_items.extend(next_p.line_items)

        # Global heuristic pass over all accumulated text
        _notify("understanding", 4, 88, "AI Understanding: Synthesizing line items & taxes")
        global_heuristic = self.extractor._extract_heuristic(combined_ocr_results)
        extracted = self.extractor._merge_invoices(extracted, global_heuristic)

        # ── Stage 4b: LLM fallback ───────────────────────────────────────────
        _notify("llm", 5, 90, "LLM Fallback: Checking confidence & completeness")
        logger.info(f"[{job_id}] Stage 4b: LLM confidence check")
        self.llm.enhance_low_confidence_fields(
            extracted,
            all_raw_ocr_texts,
            confidence_threshold=self.settings.llm_fallback_threshold,
        )

        # ── Stage 4c: Validation ─────────────────────────────────────────────
        _notify("validation", 6, 95, "Validation: Rules engine & arithmetic reconciliation")
        logger.info(f"[{job_id}] Stage 4c: Validation")
        invoice_schema, validation_report = self.validator.validate(extracted)
        invoice_schema.overall_confidence = extracted.overall_confidence

        # ── Stage 5: Render ──────────────────────────────────────────────────
        logger.info(f"[{job_id}] Stage 5: Rendering output")
        html_output = self.renderer.to_html(invoice_schema)

        pdf_path = None
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            pdf_path = self.renderer.to_pdf(invoice_schema, out / f"{job_id}.pdf")

        # Build model_used label
        primary_model = detected_models[0] if detected_models else "yolo"
        text_path     = "native_pdf" if "native_pdf" in path_summary else "ocr"
        model_parts   = [text_path, primary_model]
        model_parts.append("layoutlm" if self.extractor.model else "heuristic")
        if self.llm.is_available():
            model_parts.append("ollama")
        model_used = "+".join(model_parts)

        _notify("done", 6, 100, "Digitization Complete")

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
            raw_ocr_texts=all_raw_ocr_texts,
            page_count=page_count,
            model_used=model_used,
        )
