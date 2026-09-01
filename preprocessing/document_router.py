"""
preprocessing/document_router.py

Phase 2 & 3: Intelligent Document Type Router.
Classifies incoming documents into one of 6 structural paths:
1. DIGITAL_PDF    -> Vector native text layer with high font precision
2. PRINTED_SCAN   -> Raster scanner document with uniform orientation & clean edges
3. PHONE_PHOTO    -> Smartphone camera image with perspective distortion & uneven lighting
4. HANDWRITTEN    -> Handwritten invoice or receipt with cursive stroke density
5. MIXED          -> Printed form with handwritten value fills
6. UNKNOWN        -> Fallback path

Routes each document to its optimal preprocessing and extraction strategy.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path
from loguru import logger
import pymupdf


@dataclass
class DocumentRoutingDecision:
    doc_type: str                  # DIGITAL_PDF | PRINTED_SCAN | PHONE_PHOTO | HANDWRITTEN | MIXED | UNKNOWN
    confidence: float              # [0.0, 1.0]
    is_digital_native: bool
    requires_perspective_warp: bool
    requires_shadow_removal: bool
    requires_stroke_enhancement: bool
    reason: str


class DocumentRouter:
    """
    Analyzes document bytes, vector structures, and raster features to
    determine optimal extraction routing.
    """

    DIGITAL_PDF = "DIGITAL_PDF"
    PRINTED_SCAN = "PRINTED_SCAN"
    PHONE_PHOTO = "PHONE_PHOTO"
    HANDWRITTEN = "HANDWRITTEN"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"

    def route(
        self,
        file_bytes: bytes,
        filename: str = "",
        first_page_image: Optional[np.ndarray] = None,
    ) -> DocumentRoutingDecision:
        suffix = Path(filename).suffix.lower()

        # 1. Check for Digital Vector PDF (Native text stream) vs Scanned PDF (Adobe Scan / CamScanner)
        if suffix == ".pdf" or file_bytes.startswith(b"%PDF"):
            try:
                doc = pymupdf.open(stream=file_bytes, filetype="pdf")
                if len(doc) > 0:
                    page = doc[0]
                    words = page.get_text("words")
                    meta = doc.metadata or {}
                    creator = (meta.get("creator") or "").lower()
                    producer = (meta.get("producer") or "").lower()
                    title = (meta.get("title") or "").lower()
                    fn_lower = filename.lower()

                    is_scanner_app = any(s in creator or s in producer or s in title or s in fn_lower for s in ("adobe scan", "camscanner", "scanner", "scan ", "scan_", "genius scan", "clear scanner"))

                    images = page.get_images()
                    has_full_page_raster = False
                    if images:
                        for img_info in images:
                            try:
                                base_img = doc.extract_image(img_info[0])
                                if base_img and base_img.get("width", 0) > 400 and base_img.get("height", 0) > 400:
                                    has_full_page_raster = True
                                    break
                            except Exception:
                                pass

                    doc.close()

                    # Scanned PDF (e.g. Adobe Scan / phone capture) -> Route to PRINTED_SCAN so PaddleOCR performs high-precision OCR
                    if is_scanner_app or (has_full_page_raster and len(words) < 250):
                        logger.info(f"[Router] Detected Scanned PDF (scanner_app={is_scanner_app}, full_raster={has_full_page_raster}) -> Routing to PRINTED_SCAN")
                        return DocumentRoutingDecision(
                            doc_type=self.PRINTED_SCAN,
                            confidence=0.95,
                            is_digital_native=False,
                            requires_perspective_warp=False,
                            requires_shadow_removal=True,
                            requires_stroke_enhancement=False,
                            reason=f"Scanned document PDF with background raster image detected ({len(words)} vector words)",
                        )

                    # True Digital Vector PDF (e.g. Tally, SAP, Zoho, QuickBooks)
                    if len(words) >= 15 and not has_full_page_raster:
                        logger.info(f"[Router] Detected DIGITAL_PDF ({len(words)} vector words on page 1)")
                        return DocumentRoutingDecision(
                            doc_type=self.DIGITAL_PDF,
                            confidence=0.98,
                            is_digital_native=True,
                            requires_perspective_warp=False,
                            requires_shadow_removal=False,
                            requires_stroke_enhancement=False,
                            reason=f"Native vector PDF text layer detected with {len(words)} words",
                        )
            except Exception as e:
                logger.debug(f"[Router] PDF vector inspection skipped: {e}")

        # 2. Raster analysis for Images / Scanned PDFs
        img = first_page_image
        if img is None:
            img = self._decode_image(file_bytes)

        if img is None:
            return DocumentRoutingDecision(
                doc_type=self.UNKNOWN,
                confidence=0.50,
                is_digital_native=False,
                requires_perspective_warp=False,
                requires_shadow_removal=False,
                requires_stroke_enhancement=False,
                reason="Could not decode image raster for document type classification",
            )

        # Compute image geometry & lighting features
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # Feature A: Lighting uniformity / shadow presence
        blur_bg = cv2.GaussianBlur(gray, (51, 51), 0)
        bg_diff = cv2.absdiff(gray, blur_bg)
        lighting_var = float(blur_bg.var())

        # Feature B: Boundary contour rectangularity (photos often have dark desk margins or skewed corners)
        _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        has_photo_perspective = False
        if contours:
            largest = max(contours, key=cv2.contourArea)
            peri = cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
            area_ratio = cv2.contourArea(largest) / (w * h)
            # If large polygon has 4 distinct non-axis-aligned corners and doesn't cover 98% of the image
            if len(approx) == 4 and 0.40 < area_ratio < 0.94:
                has_photo_perspective = True

        # Feature C: Stroke curvature / handwriting density
        edges = cv2.Canny(gray, 80, 180)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=25, maxLineGap=5)
        straight_line_count = len(lines) if lines is not None else 0
        edge_pixel_count = cv2.countNonZero(edges)
        
        # Ratio of unstructured edge pixels to straight lines indicates cursive / handwriting
        stroke_complexity = edge_pixel_count / max(1, straight_line_count * 20)
        edge_density = edge_pixel_count / float(w * h)

        # Classification rules
        if has_photo_perspective or lighting_var > 1800:
            logger.info(f"[Router] Detected PHONE_PHOTO (perspective={has_photo_perspective}, lighting_var={lighting_var:.1f})")
            return DocumentRoutingDecision(
                doc_type=self.PHONE_PHOTO,
                confidence=0.88,
                is_digital_native=False,
                requires_perspective_warp=has_photo_perspective,
                requires_shadow_removal=lighting_var > 1200,
                requires_stroke_enhancement=False,
                reason="Camera capture detected with perspective angle / non-uniform background lighting",
            )

        if stroke_complexity > 8.0 and straight_line_count < 15 and edge_density > 0.015:
            logger.info(f"[Router] Detected HANDWRITTEN (stroke_complexity={stroke_complexity:.1f}, edge_density={edge_density:.4f})")
            return DocumentRoutingDecision(
                doc_type=self.HANDWRITTEN,
                confidence=0.82,
                is_digital_native=False,
                requires_perspective_warp=False,
                requires_shadow_removal=False,
                requires_stroke_enhancement=True,
                reason="High cursive stroke curvature and low straight line grid density",
            )

        # Default to clean PRINTED_SCAN
        logger.info("[Router] Detected PRINTED_SCAN (flat raster invoice)")
        return DocumentRoutingDecision(
            doc_type=self.PRINTED_SCAN,
            confidence=0.90,
            is_digital_native=False,
            requires_perspective_warp=False,
            requires_shadow_removal=False,
            requires_stroke_enhancement=False,
            reason="Standard scanned document with uniform layout geometry",
        )

    def _decode_image(self, file_bytes: bytes) -> Optional[np.ndarray]:
        try:
            arr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        except Exception:
            pass

        # Try rendering PDF first page with PyMuPDF
        try:
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
            if len(doc) > 0:
                pix = doc[0].get_pixmap(dpi=150)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                doc.close()
                if pix.n == 4:
                    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception:
            pass
        return None
