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


from enum import Enum


class HandwritingLevel(str, Enum):
    NONE = "NONE"
    FIELD_ONLY = "FIELD_ONLY"
    MIXED = "MIXED"
    MOSTLY_HANDWRITTEN = "MOSTLY_HANDWRITTEN"
    FULLY_HANDWRITTEN = "FULLY_HANDWRITTEN"


@dataclass
class DocumentRoutingDecision:
    doc_type: str                  # DIGITAL_PDF | PRINTED_SCAN | PHONE_PHOTO | HANDWRITTEN | MIXED | UNKNOWN
    confidence: float              # [0.0, 1.0] - document type classification confidence
    is_digital_native: bool
    requires_perspective_warp: bool
    requires_shadow_removal: bool
    requires_stroke_enhancement: bool
    reason: str
    handwriting_level: str = HandwritingLevel.NONE.value   # NONE | FIELD_ONLY | MIXED | MOSTLY_HANDWRITTEN | FULLY_HANDWRITTEN
    handwriting_confidence: float = 0.0                    # [0.0, 1.0] - separate from doc_type confidence
    has_ruled_lines: bool = False
    stroke_complexity: float = 0.0
    ruled_line_count: int = 0


class DocumentRouter:
    """
    Analyzes document bytes, vector structures, and raster features to
    determine optimal extraction routing and granular handwriting level.
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

                    # Scanned PDF (e.g. Adobe Scan / phone capture) -> Route to raster analysis
                    if not (is_scanner_app or (has_full_page_raster and len(words) < 250)):
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
                                handwriting_level=HandwritingLevel.NONE.value,
                                handwriting_confidence=0.0,
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
                handwriting_level=HandwritingLevel.NONE.value,
                handwriting_confidence=0.0,
            )

        # Compute image geometry & lighting features
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # Feature A: Lighting uniformity / shadow presence
        blur_bg = cv2.GaussianBlur(gray, (51, 51), 0)
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
            if len(approx) == 4 and 0.40 < area_ratio < 0.94:
                has_photo_perspective = True

        # Feature C: Granular handwriting & ruled pad analysis
        hw_level, hw_conf, stroke_complexity, has_ruled, ruled_count = self._analyze_handwriting(gray)

        # Primary doc_type classification
        doc_type = self.PRINTED_SCAN
        doc_conf = 0.90
        reason = "Standard scanned document with uniform layout geometry"
        req_warp = False
        req_shadow = False
        req_stroke = False

        if has_photo_perspective or lighting_var > 1800:
            doc_type = self.PHONE_PHOTO
            doc_conf = 0.88
            req_warp = has_photo_perspective
            req_shadow = lighting_var > 1200
            reason = "Camera capture detected with perspective angle / non-uniform background lighting"
        elif hw_level in (HandwritingLevel.MOSTLY_HANDWRITTEN.value, HandwritingLevel.FULLY_HANDWRITTEN.value):
            doc_type = self.HANDWRITTEN
            doc_conf = 0.85
            req_stroke = True
            reason = f"High cursive stroke complexity ({stroke_complexity:.1f}) and handwriting patterns"
        elif hw_level in (HandwritingLevel.MIXED.value, HandwritingLevel.FIELD_ONLY.value):
            doc_type = self.MIXED
            doc_conf = 0.82
            req_stroke = True
            reason = f"Mixed document: printed structural template with handwritten entries (level={hw_level})"

        if hw_level != HandwritingLevel.NONE.value:
            req_stroke = True

        logger.info(
            f"[Router] Classification: {doc_type} (conf={doc_conf:.2f}), "
            f"Handwriting: {hw_level} (conf={hw_conf:.2f}, ruled_lines={has_ruled}, count={ruled_count})"
        )

        return DocumentRoutingDecision(
            doc_type=doc_type,
            confidence=doc_conf,
            is_digital_native=False,
            requires_perspective_warp=req_warp,
            requires_shadow_removal=req_shadow,
            requires_stroke_enhancement=req_stroke,
            reason=reason,
            handwriting_level=hw_level,
            handwriting_confidence=hw_conf,
            has_ruled_lines=has_ruled,
            stroke_complexity=round(stroke_complexity, 2),
            ruled_line_count=ruled_count,
        )

    def _analyze_handwriting(self, gray: np.ndarray) -> tuple[str, float, float, bool, int]:
        """
        Analyzes stroke complexity, straight line grid density, and notebook rulings
        to classify handwriting level into NONE, FIELD_ONLY, MIXED, MOSTLY_HANDWRITTEN, FULLY_HANDWRITTEN.
        """
        h, w = gray.shape[:2]

        # 1. Edge & Line Analysis
        edges = cv2.Canny(gray, 40, 140)
        edge_pixel_count = cv2.countNonZero(edges)
        edge_density = edge_pixel_count / float(w * h)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=25, maxLineGap=10)
        straight_line_mask = np.zeros_like(edges)
        straight_h_lines = 0
        straight_v_lines = 0
        straight_line_count = len(lines) if lines is not None else 0

        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                cv2.line(straight_line_mask, (x1, y1), (x2, y2), 255, thickness=3)
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if angle < 15 or angle > 165:
                    straight_h_lines += 1
                elif 75 < angle < 105:
                    straight_v_lines += 1

        # Calculate true non-straight cursive edge pixels
        non_straight_edges = cv2.bitwise_and(edges, cv2.bitwise_not(straight_line_mask))
        non_straight_count = cv2.countNonZero(non_straight_edges)
        cursive_ratio = (non_straight_count / max(1, edge_pixel_count)) if edge_pixel_count > 0 else 0.0

        # 2. Detect Horizontal Ruled Notebook Lines
        kw = max(20, int(w * 0.10))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
        h_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_h)
        h_contours, _ = cv2.findContours(h_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        long_h_lines = [c for c in h_contours if cv2.boundingRect(c)[2] > (w * 0.15)]
        ruled_count = len(long_h_lines)

        # Ruled pad lines exist when horizontal lines dominate and cursive handwriting is present
        has_ruled_lines = ruled_count >= 3 and straight_h_lines >= (2 * straight_v_lines) and non_straight_count > 100

        # 3. Zoned Analysis (Header top 25%, Body middle 50%, Footer bottom 25%)
        h25 = int(h * 0.25)
        h75 = int(h * 0.75)
        ns_header = cv2.countNonZero(non_straight_edges[:h25, :]) / float(w * h25) if h25 > 0 else 0
        ns_body = cv2.countNonZero(non_straight_edges[h25:h75, :]) / float(w * (h75 - h25)) if (h75 - h25) > 0 else 0

        # Classification decision based on cursive_ratio, non_straight_count, and ruled structure
        if (cursive_ratio > 0.50 and edge_density > 0.008 and straight_v_lines < 8) or (non_straight_count > 2500 and straight_line_count < 10):
            hw_level = HandwritingLevel.FULLY_HANDWRITTEN.value
            hw_conf = min(0.95, round(0.75 + (cursive_ratio * 0.20), 2))
        elif (has_ruled_lines and (cursive_ratio > 0.08 or non_straight_count > 200)) or (cursive_ratio > 0.35 and edge_density > 0.004):
            hw_level = HandwritingLevel.MOSTLY_HANDWRITTEN.value
            hw_conf = 0.85
        elif (ns_body > ns_header * 1.5 and non_straight_count > 150) or (has_ruled_lines and non_straight_count > 80):
            hw_level = HandwritingLevel.MIXED.value
            hw_conf = 0.80
        elif (cursive_ratio > 0.15 and edge_density > 0.002) or (non_straight_count > 300 and straight_h_lines > 8):
            hw_level = HandwritingLevel.FIELD_ONLY.value
            hw_conf = 0.70
        else:
            hw_level = HandwritingLevel.NONE.value
            hw_conf = 0.0

        stroke_complexity = round(cursive_ratio * 100.0, 2)
        return hw_level, hw_conf, stroke_complexity, has_ruled_lines, ruled_count

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
