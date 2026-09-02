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

import numpy as np
import pymupdf
from PIL import Image

from config.settings import Settings
from preprocessing.pipeline import InvoicePreprocessor
from preprocessing.pdf_converter import PDFConverter, NativePDFPage, PreprocessResult
from preprocessing.quality_scorer import DocumentQualityScorer, QualityAssessment
from preprocessing.document_router import DocumentRouter, DocumentRoutingDecision
from preprocessing.document_profile import DocumentProfile
from detection.detector import InvoiceDetector
from ocr.extractor import InvoiceOCR, OCRResult, TextBlock, OCRWord
from ocr.handwriting_ocr import HandwritingOCR
from ocr.numeric_recognizer import NumericRecognizer
from ocr.candidate_generator import CandidateGenerator, OCRCandidate
from preprocessing.handwriting_cropper import FieldCrop, crop_from_yolo_regions, crop_from_tie_anchors
from understanding.layoutlm import LayoutLMExtractor, ExtractedInvoice
from understanding.table_extractor import TableExtractor
from understanding.template_extractor import TemplateExtractor
from understanding.template_retriever import TemplateRetriever, TemplateMatchResult
from llm_fallback.ollama_client import OllamaClient
from validation.validator import InvoiceValidator, InvoiceSchema, ValidationReport
from output.renderer import InvoiceRenderer


@dataclass
class ProcessingContext:
    doc_type: str
    handwriting_level: str
    handwriting_confidence: float
    quality_score: float
    is_phone_photo: bool
    original_images: list[np.ndarray] = None
    enhanced_images: list[np.ndarray] = None
    has_ruled_lines: bool = False


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
    doc_type: str = "UNKNOWN"
    quality_score: float = 1.0
    doc_profile: Optional[Any] = None
    processing_context: Optional[ProcessingContext] = None


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
    Each TextBlock retains individual word tokens and word-level bounding boxes.
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

    # Build TextBlock per line with word tokens
    def line_to_textblock(line_words: list, region_label: str) -> TextBlock:
        text = " ".join(w.text for w in line_words)
        x0 = min(w.bbox[0] for w in line_words) * sx
        y0 = min(w.bbox[1] for w in line_words) * sy
        x1 = max(w.bbox[2] for w in line_words) * sx
        y1 = max(w.bbox[3] for w in line_words) * sy
        words = [
            OCRWord(
                text=w.text,
                confidence=0.99,
                bbox=[[w.bbox[0] * sx, w.bbox[1] * sy], [w.bbox[2] * sx, w.bbox[1] * sy], [w.bbox[2] * sx, w.bbox[3] * sy], [w.bbox[0] * sx, w.bbox[3] * sy]],
            )
            for w in line_words
        ]
        return TextBlock(
            text=text,
            confidence=0.99,
            bbox=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            region_label=region_label,
            words=words,
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

    # ── Assign each line to a detected region ──
    region_line_groups: dict[str, list[list]] = {r.label: [] for r in det_regions}

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
            # Do not force unassigned lines into arbitrary regions by distance;
            # they are preserved cleanly in the unified full_page OCRResult.
            pass

    results = {}
    for region in det_regions:
        lines = region_line_groups[region.label]
        if not lines:
            continue
        blocks = [line_to_textblock(lw, region.label) for lw in lines]
        full_text = "\n".join(b.text for b in blocks)
        results[region.label] = OCRResult(
            region_label=region.label,
            text_blocks=blocks,
            full_text=full_text,
            avg_confidence=0.99 if blocks else 0.0,
            engine="native_pdf",
        )

    # Always provide unified full_page OCR result
    all_blocks = [line_to_textblock(lw, "full_page") for lw in sorted_lines]
    results["full_page"] = OCRResult(
        region_label="full_page",
        text_blocks=all_blocks,
        full_text="\n".join(b.text for b in all_blocks),
        avg_confidence=0.99,
        engine="native_pdf",
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

    @staticmethod
    def _assign_tokens_to_regions(full_page_ocr: OCRResult, regions: list) -> dict[str, OCRResult]:
        """Spatially intersects full-page OCR tokens into YOLO regions, eliminating duplicate OCR passes."""
        results = {}
        for r in regions:
            rx1, ry1, rx2, ry2 = getattr(r, "bbox", [0, 0, 0, 0])
            matching_blocks = []
            for b in full_page_ocr.text_blocks:
                bx1, by1, bx2, by2 = b.to_xyxy() if hasattr(b, "to_xyxy") else b.bbox
                if bx1 >= rx1 - 10 and by1 >= ry1 - 10 and bx2 <= rx2 + 10 and by2 <= ry2 + 10:
                    matching_blocks.append(b)
                elif not (bx2 < rx1 or bx1 > rx2 or by2 < ry1 or by1 > ry2):
                    matching_blocks.append(b)

            if matching_blocks:
                full_text = "\n".join(b.text for b in matching_blocks)
                avg_conf = sum(b.confidence for b in matching_blocks) / len(matching_blocks)
                label = getattr(r, "label", "region")
                results[label] = OCRResult(
                    region_label=label,
                    text_blocks=matching_blocks,
                    full_text=full_text,
                    avg_confidence=round(avg_conf, 3),
                )
        return results

    def __init__(
        self,
        settings: Settings,
        db: Optional[Any] = None,
        db_manager: Optional[Any] = None,
        minio_manager: Optional[Any] = None,
    ):
        self.settings = settings
        self.db = db_manager or db
        self.minio = minio_manager
        logger.info("Initialising Invoice Pipeline...")

        self.preprocessor = InvoicePreprocessor()
        self.quality_scorer = DocumentQualityScorer()
        self.document_router = DocumentRouter()
        self.pdf_converter = PDFConverter()
        self.detector      = InvoiceDetector(model_path=settings.yolo_model_path)
        self.ocr           = InvoiceOCR(languages=settings.ocr_languages, use_gpu=False)
        self.handwriting_ocr = HandwritingOCR(use_gpu=False, lazy_load=True)
        self.numeric_recognizer = NumericRecognizer()
        self.candidate_generator = CandidateGenerator(numeric_recognizer=self.numeric_recognizer)
        self.extractor     = LayoutLMExtractor(
            model_path=settings.layoutlm_model_path,
            base_model=settings.layoutlm_base_model,
        )
        self.table_extractor = TableExtractor()
        self.template_retriever = TemplateRetriever(db_manager=self.db)
        self.template_extractor = TemplateExtractor()
        self.llm      = OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            vision_model=getattr(settings, "ollama_vision_model", "minicpm-v:latest"),
            timeout=getattr(settings, "ollama_timeout", 60.0),
            enabled=getattr(settings, "enable_llm_fallback", True),
            num_ctx=getattr(settings, "ollama_num_ctx", 2048),
            keep_alive=getattr(settings, "ollama_keep_alive", "15m"),
            num_thread=getattr(settings, "ollama_num_thread", None),
            enable_vision=getattr(settings, "enable_vision_fallback", False),
        )
        self.validator = InvoiceValidator()
        self.renderer  = InvoiceRenderer()

        logger.info("=" * 65)
        logger.info("   INVOICE DIGITIZATION PIPELINE INITIALIZED")
        logger.info(f"   OCR Engine:      {self.ocr.engine_name}")
        logger.info(f"   TIE Engine:      Fast-Path Template Retriever & Anchor Extractor")
        logger.info(f"   Detector:        {'DocLayout-YOLO' if self.detector.is_doclaynet else ('Custom-YOLO' if self.detector.model else 'Heuristic')}")
        logger.info(f"   LayoutLM Model:  {settings.layoutlm_model_path if Path(settings.layoutlm_model_path).exists() else 'Heuristic Fallback'}")
        llm_status = f"{settings.ollama_model} ({settings.ollama_base_url}, timeout={getattr(settings, 'ollama_timeout', 60.0)}s)" if getattr(settings, 'enable_llm_fallback', True) else "Disabled"
        logger.info(f"   Ollama LLM:      {llm_status}")
        logger.info("=" * 65)

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

        # ── Stage 1: Document Routing & Pre-processing ──────────────────────────
        _notify("preprocessing", 1, 10, "Document Routing: Analyzing type & quality")
        routing = self.document_router.route(file_bytes, filename=filename)
        quality_score = 1.0
        original_images = []
        enhanced_images = []

        is_handwritten = routing.handwriting_level in ("MOSTLY_HANDWRITTEN", "FULLY_HANDWRITTEN") or routing.doc_type == DocumentRouter.HANDWRITTEN

        if suffix in self.SUPPORTED_PDF_FORMATS:
            logger.info(f"[{job_id}] Stage 2: PDF → dual-path conversion (Route: {routing.doc_type}, HW: {routing.handwriting_level})")
            pages = self.pdf_converter.convert_bytes(file_bytes)
            if pages and hasattr(pages[0], "image"):
                qa = self.quality_scorer.assess(pages[0].image)
                quality_score = qa.composite_score
                for p in pages:
                    if hasattr(p, "image") and isinstance(p.image, np.ndarray):
                        original_images.append(p.image.copy())
                        if is_handwritten:
                            hw_res = self.preprocessor.process_handwritten(
                                p.image,
                                handwriting_level=routing.handwriting_level,
                                is_phone_photo=(routing.doc_type == DocumentRouter.PHONE_PHOTO),
                            )
                            p.image = hw_res.image
                            if hasattr(p, "pil_image"):
                                p.pil_image = hw_res.pil_image
                            enhanced_images.append(hw_res.image)
                        elif routing.doc_type == DocumentRouter.PHONE_PHOTO:
                            p.image = self.preprocessor._remove_shadows(p.image)
                            enhanced_images.append(p.image)
                        else:
                            enhanced_images.append(p.image)
        elif suffix in self.SUPPORTED_IMAGE_FORMATS:
            logger.info(f"[{job_id}] Stage 2: Image pre-processing (Route: {routing.doc_type}, HW: {routing.handwriting_level})")
            qa = self.quality_scorer.assess(file_bytes)
            quality_score = qa.composite_score
            if is_handwritten:
                page = self.preprocessor.process_handwritten(
                    file_bytes,
                    handwriting_level=routing.handwriting_level,
                    is_phone_photo=(routing.doc_type == DocumentRouter.PHONE_PHOTO),
                )
                original_images.append(getattr(page, "original_image", page.image))
                enhanced_images.append(page.image)
            elif routing.doc_type == DocumentRouter.PHONE_PHOTO:
                page = self.preprocessor.process_photo(file_bytes)
                original_images.append(page.image)
                enhanced_images.append(page.image)
            else:
                page = self.preprocessor.process(file_bytes)
                original_images.append(page.image)
                enhanced_images.append(page.image)
            pages = [page]
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        processing_ctx = ProcessingContext(
            doc_type=routing.doc_type,
            handwriting_level=routing.handwriting_level,
            handwriting_confidence=routing.handwriting_confidence,
            quality_score=quality_score,
            is_phone_photo=(routing.doc_type == DocumentRouter.PHONE_PHOTO),
            original_images=original_images,
            enhanced_images=enhanced_images,
            has_ruled_lines=routing.has_ruled_lines,
        )

        page_count = len(pages)
        logger.info(f"[{job_id}] {page_count} page(s) to process (Quality: {quality_score:.2f})")

        # ── Stage 2: Per-page Detection + Text Extraction ───────────────────
        combined_ocr_results: dict[str, OCRResult] = {}
        all_raw_ocr_texts:    dict[str, str]       = {}
        detected_models:      list[str]            = []
        extracted_per_page                         = []
        path_summary:         list[str]            = []
        all_page_regions:     list                 = []
        page_dimensions:      dict[int, tuple[int, int]] = {}

        for p_idx, p_obj in enumerate(pages):
            logger.info(f"[{job_id}] Processing Page {p_idx + 1}/{page_count}...")
            p_num = p_idx + 1

            # Compute actual page image dimensions
            p_w, p_h = 1000, 1414
            if hasattr(p_obj.image, "shape"):
                p_h, p_w = p_obj.image.shape[:2]
            elif hasattr(p_obj.image, "size"):
                p_w, p_h = p_obj.image.size
            page_dimensions[p_num] = (p_w, p_h)

            # ── YOLO region detection ──────────────────────────────────────
            _notify("detection", 2, 35, f"Region Detection: YOLOv8 identifying zones (Page {p_num}/{page_count})")
            det_res = self.detector.detect(p_obj.image)
            detected_models.append(det_res.model_used)
            # Tag each region with its exact page number
            for r in det_res.regions:
                if hasattr(r, "page"):
                    r.page = p_num
            all_page_regions.append(det_res.regions)
            logger.info(
                f"[{job_id}] Page {p_num}: {len(det_res.regions)} regions "
                f"via {det_res.model_used}"
            )

            # ── PATH A: Digital PDF page ───────────────────────────────────
            if isinstance(p_obj, NativePDFPage):
                path_summary.append("native_pdf")
                _notify("ocr", 3, 50, f"Extracting native PDF text (Page {p_num}/{page_count})")
                logger.info(
                    f"[{job_id}] Page {p_num}: PATH A — native text "
                    f"({len(p_obj.words)} words, conf=0.99)"
                )
                p_ocr = _native_page_to_ocr_results(p_obj, det_res.regions)
                pil_image = p_obj.pil_image

            # ── PATH B: Scanned page / direct image ────────────────────────
            else:
                path_summary.append("ocr")
                _notify("ocr", 3, 60, f"OCR Extraction: PaddleOCR reading text (Page {p_num}/{page_count})")
                logger.info(f"[{job_id}] Page {p_num}: PATH B — Scanned Image OCR")

                p_ocr = {}
                # Extract full page once; spatially intersect tokens into YOLO regions to eliminate duplicate OCR passes
                full_res = self.ocr.extract_full_page(p_obj.image)
                p_ocr[f"full_page_p{p_num}"] = full_res

                if det_res.regions:
                    _notify("ocr", 3, 68, f"OCR Extraction: Mapping {len(det_res.regions)} structured region blocks (Page {p_num}/{page_count})")
                    region_ocr = self._assign_tokens_to_regions(full_res, det_res.regions)
                    p_ocr.update(region_ocr)

                # Generate handwriting & numeric candidates for detected regions
                is_hw_doc = routing.handwriting_level in ("MOSTLY_HANDWRITTEN", "FULLY_HANDWRITTEN", "MIXED")
                hw_regions = [r for r in det_res.regions if r.is_handwritten or is_hw_doc]
                if hw_regions:
                    enh_img = processing_ctx.enhanced_images[p_idx] if processing_ctx.enhanced_images and p_idx < len(processing_ctx.enhanced_images) else p_obj.image
                    crops = crop_from_yolo_regions(p_obj.image, hw_regions, enhanced_image=enh_img, page=p_num)
                    for crop in crops:
                        region_key = f"{crop.field_name}_p{p_num}" if f"_p{p_num}" not in crop.field_name else crop.field_name
                        raw_text = p_ocr.get(crop.field_name, full_res).full_text
                        cands = self.candidate_generator.generate_candidates(
                            field_crop=crop,
                            handwriting_ocr=self.handwriting_ocr,
                            raw_ocr_text=raw_text,
                        )
                        if region_key in p_ocr:
                            p_ocr[region_key].candidates = cands
                        elif crop.field_name in p_ocr:
                            p_ocr[crop.field_name].candidates = cands

                pil_image = p_obj.pil_image

            # Accumulate OCR results with page-specific keys
            for k, v in p_ocr.items():
                unique_key = k if f"_p{p_num}" in k else f"{k}_p{p_num}"
                combined_ocr_results[unique_key] = v
                all_raw_ocr_texts[unique_key] = v.full_text

        # ── Stage 2b: Multi-Invoice Document Segmentation ────────────────────
        from preprocessing.document_segmenter import DocumentSegmenter
        segmenter = DocumentSegmenter()
        page_texts = {}
        for p_idx in range(page_count):
            p_num = p_idx + 1
            if f"full_page_p{p_num}" in combined_ocr_results:
                page_texts[p_num] = combined_ocr_results[f"full_page_p{p_num}"].full_text
            else:
                page_texts[p_num] = "\n".join(
                    ocr_res.full_text for k, ocr_res in combined_ocr_results.items()
                    if f"_p{p_num}" in k and hasattr(ocr_res, "full_text")
                )
        segments = segmenter.segment(page_texts)
        if len(segments) > 1:
            logger.info(f"[{job_id}] Merged multi-invoice upload detected ({len(segments)} sub-invoices).")

        # ── Stage 3: Build DocumentProfile & Multi-Stage TIE Retrieval ──────
        _notify("retrieval", 4, 70, "TIE Retrieval: Checking known layout templates")
        primary_w, primary_h = page_dimensions.get(1, (1200, 1600))

        all_regions_flat = []
        for r_list in all_page_regions:
            all_regions_flat.extend(r_list)

        doc_profile = DocumentProfile.from_ocr_and_regions(
            ocr_results=combined_ocr_results,
            regions=all_regions_flat,
            width=primary_w,
            height=primary_h,
            page_count=page_count,
            is_digital_native=("native_pdf" in path_summary),
            quality_score=quality_score,
            page_dimensions=page_dimensions,
        )

        tpl_match = self.template_retriever.retrieve(doc_profile)
        used_tie_fast_path = False
        primary_engine_label = "heuristic"

        # ── Stage 4a: Extraction (TIE Fast Path vs AI Fallback) ─────────────
        if tpl_match.match_type in ("exact_version", "family_anchor"):
            _notify("understanding", 4, 78, f"TIE Extraction: Running rules ({tpl_match.match_type})")
            logger.info(f"[{job_id}] TIE Match: {tpl_match.match_type} (conf={tpl_match.match_confidence:.2f}, ver={tpl_match.matched_version_id})")
            extracted = self.template_extractor.extract(
                profile=doc_profile,
                field_rules=tpl_match.field_rules,
                template_version_id=tpl_match.matched_version_id,
                match_type=tpl_match.match_type,
            )
            # Use high-fidelity native digital PDF table items if available
            if pages and isinstance(pages[0], NativePDFPage) and pages[0].line_items:
                extracted.line_items = pages[0].line_items

            used_tie_fast_path = (tpl_match.match_type == "exact_version")
            primary_engine_label = "tie_fast_path" if tpl_match.match_type == "exact_version" else "tie_anchor_family"

            # Reconcile any unextracted optional fields with global heuristic
            full_page_ocrs = {k: v for k, v in combined_ocr_results.items() if "full_page" in k}
            heuristic_input = full_page_ocrs if full_page_ocrs else combined_ocr_results
            global_heuristic = self.extractor._extract_heuristic(heuristic_input)
            extracted = self.extractor._merge_invoices(extracted, global_heuristic)
            tie_snapshot = {
                f: getattr(extracted, f).value
                for f in ["invoice_number", "invoice_date", "vendor_name", "vendor_gstin", "subtotal", "tax_amount", "grand_total"]
                if getattr(extracted, f, None) and getattr(extracted, f).value
            }
            layoutlm_snapshot = {}
            heuristic_snapshot = {
                f: getattr(global_heuristic, f).value
                for f in ["invoice_number", "invoice_date", "vendor_name", "vendor_gstin", "subtotal", "tax_amount", "grand_total"]
                if getattr(global_heuristic, f, None) and getattr(global_heuristic, f).value
            }
        else:
            # Unknown / Novel layout -> AI Pipeline (LayoutLM / Heuristic)
            _notify("understanding", 4, 78, "AI Understanding: Mapping fields via LayoutLM/Heuristics")
            extracted_per_page = []
            for p_idx, p_obj in enumerate(pages):
                pil_image = p_obj.pil_image
                p_ocr = {k: v for k, v in combined_ocr_results.items() if f"p{p_idx+1}" in k or page_count == 1}
                p_extracted = self.extractor.extract(p_ocr, image=pil_image)
                if isinstance(p_obj, NativePDFPage) and p_obj.line_items:
                    p_extracted.line_items = p_obj.line_items
                else:
                    table_ocr_res = (
                        p_ocr.get("line_items")
                        or p_ocr.get(f"line_items_p{p_idx+1}")
                        or p_ocr.get(f"full_page_p{p_idx+1}")
                        or p_ocr.get("full_page")
                    )
                    if table_ocr_res:
                        spatial_items = self.table_extractor.extract_tables_from_spatial_ocr(table_ocr_res)
                        if spatial_items:
                            p_extracted.line_items = spatial_items
                extracted_per_page.append(p_extracted)

            extracted = extracted_per_page[0]
            for next_p in extracted_per_page[1:]:
                extracted = self.extractor._merge_invoices(extracted, next_p)
                if next_p.line_items:
                    extracted.line_items.extend(next_p.line_items)

            tie_snapshot = {}
            layoutlm_snapshot = {
                f: getattr(extracted, f).value
                for f in ["invoice_number", "invoice_date", "vendor_name", "vendor_gstin", "subtotal", "tax_amount", "grand_total"]
                if getattr(extracted, f, None) and getattr(extracted, f).value
            }

            full_page_ocrs = {k: v for k, v in combined_ocr_results.items() if "full_page" in k}
            heuristic_input = full_page_ocrs if full_page_ocrs else combined_ocr_results
            global_heuristic = self.extractor._extract_heuristic(heuristic_input)
            heuristic_snapshot = {
                f: getattr(global_heuristic, f).value
                for f in ["invoice_number", "invoice_date", "vendor_name", "vendor_gstin", "subtotal", "tax_amount", "grand_total"]
                if getattr(global_heuristic, f, None) and getattr(global_heuristic, f).value
            }
            extracted = self.extractor._merge_invoices(extracted, global_heuristic)
            primary_engine_label = "layoutlm" if self.extractor.model else "heuristic"

        # ── Stage 4b: Field-Level Routing & Selective LLM fallback ───────────
        critical_fields = ["grand_total", "invoice_number", "invoice_date", "vendor_gstin"]
        needs_ai_enhancement = False
        for cf in critical_fields:
            cf_obj = getattr(extracted, cf, None)
            if not cf_obj or not cf_obj.value or cf_obj.confidence < self.settings.llm_fallback_threshold:
                needs_ai_enhancement = True
                break

        # Risk-based LLM triggering: only trigger when genuine extraction ambiguity or missing critical fields
        if self.settings.enable_llm_fallback and self.llm.is_available() and needs_ai_enhancement:
            _notify("llm", 5, 90, "LLM Fallback: Resolving low-confidence/ambiguous fields")
            logger.info(f"[{job_id}] Stage 4b: Selective LLM field enhancement ({self.settings.ollama_model})")
            page_imgs = [p.image for p in pages if hasattr(p, "image") and p.image is not None]
            llm_preds = self.llm.enhance_low_confidence_fields(
                extracted,
                all_raw_ocr_texts,
                confidence_threshold=self.settings.llm_fallback_threshold,
                page_images=page_imgs,
            ) or {}
        else:
            _notify("llm", 5, 90, "LLM Fallback: Skipped (High-confidence or offline)")
            llm_preds = {}

        # ── Stage 4c: Validation ─────────────────────────────────────────────
        _notify("validation", 6, 95, "Validation: Rules engine & arithmetic reconciliation")
        logger.info(f"[{job_id}] Stage 4c: Validation")
        invoice_schema, validation_report = self.validator.validate(
            extracted,
            doc_type=routing.doc_type,
            handwriting_level=routing.handwriting_level,
            handwriting_penalty=getattr(self.settings, "handwriting_confidence_penalty", 0.85),
        )

        # ── Template Disagreement & Identity Analysis ────────────────────────
        has_contradiction = False
        is_novel = (tpl_match.match_type == "none")
        try:
            from active_learning.disagreement_engine import evaluate_model_disagreement

            disagreement_res = evaluate_model_disagreement(
                layoutlm_preds=layoutlm_snapshot,
                heuristic_preds=heuristic_snapshot,
                llm_preds=llm_preds,
                tie_preds=tie_snapshot,
            )

            canonical_tpl_id = tpl_match.matched_version_id or doc_profile.exact_fingerprint
            canonical_family_id = tpl_match.matched_family_id or doc_profile.family_fingerprint

            has_contradiction = bool(disagreement_res.get("has_contradiction", False))
            invoice_schema.template_id = canonical_tpl_id
            invoice_schema.template_family_id = canonical_family_id
            invoice_schema.template_version_id = tpl_match.matched_version_id
            invoice_schema.is_novel_template = is_novel
            invoice_schema.disagreement_score = disagreement_res.get("disagreement_score", 0.0)
            if is_novel:
                invoice_schema.review_reasons.append(f"Novel layout template ({canonical_family_id[:8]})")
            if has_contradiction:
                for d in disagreement_res.get("disagreements", []):
                    invoice_schema.review_reasons.append(d["reason"])
        except Exception as tpl_ex:
            logger.debug(f"Template/disagreement error: {tpl_ex}")

        # ── Stage 4d: Centralized Auto-Acceptance Gate ────────────────────────
        from validation.acceptance_gate import AutoAcceptanceGate
        is_auto_accepted, rejection_reasons = AutoAcceptanceGate.evaluate(
            invoice=invoice_schema,
            validation_report=validation_report,
            quality_score=quality_score,
            has_contradiction=has_contradiction,
            is_novel_template=is_novel,
            doc_type=routing.doc_type,
            match_type=tpl_match.match_type,
        )

        invoice_schema.needs_review = not is_auto_accepted
        if not is_auto_accepted:
            invoice_schema.review_reasons = list(dict.fromkeys(invoice_schema.review_reasons + rejection_reasons))

        # ── Stage 5: Render ──────────────────────────────────────────────────
        logger.info(f"[{job_id}] Stage 5: Rendering output (Auto-Accepted: {is_auto_accepted})")
        html_output = self.renderer.to_html(invoice_schema)

        pdf_path = None
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            pdf_path = self.renderer.to_pdf(invoice_schema, out / f"{job_id}.pdf")

        # Build model_used label
        primary_model = detected_models[0] if detected_models else "yolo"
        text_path     = "native_pdf" if "native_pdf" in path_summary else "ocr"
        model_parts   = [text_path, primary_model, primary_engine_label]
        if self.llm.is_available() and needs_ai_enhancement:
            model_parts.append("ollama")
        model_used = "+".join(model_parts)

        _notify("done", 6, 100, "Digitization Complete")

        logger.info(
            f"[{job_id}] Done — confidence={invoice_schema.overall_confidence:.2f}, "
            f"needs_review={invoice_schema.needs_review}, template={invoice_schema.template_id}, "
            f"disagreement={invoice_schema.disagreement_score}, model={model_used}"
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
            doc_type=routing.doc_type,
            quality_score=quality_score,
            doc_profile=doc_profile,
            processing_context=processing_ctx,
        )
