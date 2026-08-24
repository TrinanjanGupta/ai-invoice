"""
Stage 3b: PaddleOCR text extractor.

Extracts text from each detected region with bounding boxes and confidence.
Supports English, Hindi, and other Indic scripts out of the box.
"""

import numpy as np
from PIL import Image
from loguru import logger
from dataclasses import dataclass, field
from typing import Optional, Any, Union


import re

@dataclass
class OCRWord:
    text: str
    confidence: float
    bbox: list          # [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] or [x1, y1, x2, y2]

    def to_xyxy(self) -> list[float]:
        """Return [x1, y1, x2, y2] bounding box."""
        if len(self.bbox) == 4 and isinstance(self.bbox[0], (int, float)):
            return [float(self.bbox[0]), float(self.bbox[1]), float(self.bbox[2]), float(self.bbox[3])]
        xs = [p[0] for p in self.bbox]
        ys = [p[1] for p in self.bbox]
        return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]

    def to_poly(self) -> list[list[float]]:
        """Return 4-point polygon [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]."""
        if len(self.bbox) == 4 and isinstance(self.bbox[0], (int, float)):
            x1, y1, x2, y2 = self.bbox
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        return self.bbox

    def center(self) -> tuple[float, float]:
        """Return (cx, cy) center coordinate."""
        x1, y1, x2, y2 = self.to_xyxy()
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class TextBlock:
    text: str
    confidence: float
    bbox: list          # [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] (PaddleOCR format)
    region_label: str   # which invoice region this came from
    words: list[OCRWord] = field(default_factory=list)

    def to_xyxy(self) -> list[float]:
        if len(self.bbox) == 4 and isinstance(self.bbox[0], (int, float)):
            return [float(self.bbox[0]), float(self.bbox[1]), float(self.bbox[2]), float(self.bbox[3])]
        if not self.bbox:
            return [0.0, 0.0, 0.0, 0.0]
        xs = [p[0] for p in self.bbox]
        ys = [p[1] for p in self.bbox]
        return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def _char_width_weight(char: str) -> float:
    """Approximate relative typographical width for Latin, digits, and Indian currency glyphs."""
    if char in " .:,;!|'\"`iIl1()-[]{}":
        return 0.5
    elif char in "MWmw%@#":
        return 1.4
    elif char in "₹₹$€£":
        return 1.3
    elif char.isspace():
        return 0.6
    return 1.0


def decompose_line_into_words(
    text: str,
    bbox: list,
    confidence: float,
    line_image: Optional[np.ndarray] = None,
) -> list[OCRWord]:
    """
    Decomposes an OCR text line into individual word tokens.
    If line_image is provided, uses pixel-level vertical projection profiling
    to find actual glyph ink boundaries and word whitespace valleys on real pixels.
    Falls back to typographical metric projection if image is unavailable.
    """
    if not text or not text.strip():
        return []

    if not bbox:
        return [OCRWord(text=w, confidence=confidence, bbox=[0, 0, 0, 0]) for w in text.split()]

    if len(bbox) == 4 and isinstance(bbox[0], (int, float)):
        x_min, y_min, x_max, y_max = bbox
    else:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)

    words_in_text = [m.group() for m in re.finditer(r"\S+", text)]
    if not words_in_text:
        return []

    # Attempt Pixel-Grounded Image Projection Profiling
    if line_image is not None and line_image.size > 0 and len(words_in_text) > 1:
        try:
            import cv2
            gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY) if len(line_image.shape) == 3 else line_image
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            col_sums = np.sum(thresh, axis=0)

            # Find active ink column segments separated by whitespace gaps
            in_ink = False
            seg_start = 0
            segments = []
            min_ink = max(1.0, np.max(col_sums) * 0.05) if col_sums.size > 0 else 1.0

            for col_idx, val in enumerate(col_sums):
                if val >= min_ink:
                    if not in_ink:
                        in_ink = True
                        seg_start = col_idx
                else:
                    if in_ink:
                        in_ink = False
                        segments.append((seg_start, col_idx))
            if in_ink:
                segments.append((seg_start, len(col_sums) - 1))

            # If segment count matches word count, use exact pixel segments
            if len(segments) == len(words_in_text):
                words: list[OCRWord] = []
                for w_str, (s_px, e_px) in zip(words_in_text, segments):
                    w_x1 = x_min + float(s_px)
                    w_x2 = x_min + float(e_px)
                    word_bbox = [[w_x1, y_min], [w_x2, y_min], [w_x2, y_max], [w_x1, y_max]]
                    words.append(OCRWord(text=w_str, confidence=confidence, bbox=word_bbox))
                return words
        except Exception:
            pass

    # Typographical Cumulative Metric Projection
    weights = [_char_width_weight(c) for c in text]
    total_weight = sum(weights)
    if total_weight <= 0:
        total_weight = max(1.0, float(len(text)))
        weights = [1.0] * len(text)

    cum_weights = [0.0]
    for w in weights:
        cum_weights.append(cum_weights[-1] + w)

    words: list[OCRWord] = []
    line_width = max(1.0, float(x_max - x_min))

    for m in re.finditer(r"\S+", text):
        w_text = m.group()
        s_idx = m.start()
        e_idx = m.end()

        ratio_start = cum_weights[s_idx] / total_weight
        ratio_end = cum_weights[e_idx] / total_weight

        w_x1 = x_min + line_width * ratio_start
        w_x2 = x_min + line_width * ratio_end
        w_y1 = y_min
        w_y2 = y_max

        word_bbox = [[w_x1, w_y1], [w_x2, w_y1], [w_x2, w_y2], [w_x1, w_y2]]
        words.append(OCRWord(text=w_text, confidence=confidence, bbox=word_bbox))

    return words


@dataclass
class OCRResult:
    region_label: str
    text_blocks: list[TextBlock]
    full_text: str      # concatenated text of the region with preserved newlines
    avg_confidence: float
    engine: str = "paddleocr"  # "paddleocr" or "easyocr" or "native_pdf"


class InvoiceOCR:
    """
    PaddleOCR wrapper optimised for invoice text extraction.
    Supports multi-language (English + Hindi/Indic scripts configurable).
    Explicitly tracks and logs the active OCR engine.
    """

    def __init__(self, languages: Any = None, lang: Optional[str] = None, use_gpu: bool = False):
        langs_input = languages or lang or "en,hi"
        if isinstance(langs_input, str):
            self.languages = [l.strip() for l in langs_input.split(",") if l.strip()]
        elif isinstance(langs_input, (list, tuple)):
            self.languages = list(langs_input)
        else:
            self.languages = ["en"]
        
        self.primary_lang = self.languages[0] if self.languages else "en"
        self.use_gpu = use_gpu
        self._ocr = None
        self._easyocr = None
        self.engine_name = "unknown"
        self._init_ocr()

    def _init_ocr(self):
        try:
            from paddleocr import PaddleOCR
            try:
                self._ocr = PaddleOCR(use_angle_cls=True, lang=self.primary_lang, use_gpu=self.use_gpu, enable_mkldnn=False)
            except Exception:
                self._ocr = PaddleOCR(lang=self.primary_lang, enable_mkldnn=False)
            self.engine_name = "PaddleOCR"
            logger.info(f"[OCR] Initialised ENGINE: PaddleOCR (lang={self.primary_lang}, gpu={self.use_gpu})")
        except (ImportError, Exception) as e:
            logger.warning(f"[OCR WARNING] PaddleOCR initialization failed ({e}). Engaging EasyOCR fallback...")
            try:
                import easyocr
                self._easyocr = easyocr.Reader(self.languages, gpu=self.use_gpu, verbose=False)
                self.engine_name = "EasyOCR"
                logger.info(f"[OCR] Initialised ENGINE: EasyOCR fallback (langs={self.languages}, gpu={self.use_gpu})")
            except ImportError:
                logger.error("[OCR CRITICAL] Neither PaddleOCR nor EasyOCR is installed. Run: pip install easyocr")
                raise ImportError("Neither PaddleOCR nor EasyOCR is installed.") from e
            except Exception as ex:
                logger.error(f"[OCR CRITICAL] EasyOCR init failed: {ex}")
                raise ex

    def extract_region(self, crop: np.ndarray, region_label: str) -> OCRResult:
        """
        Run OCR on a single region crop.
        Returns structured OCRResult with pixel-grounded per-word confidence.
        """
        if crop is None or crop.size == 0:
            return OCRResult(
                region_label=region_label,
                text_blocks=[],
                full_text="",
                avg_confidence=0.0,
            )

        text_blocks = []

        if self._ocr is not None:
            try:
                results = self._ocr.ocr(crop)
                if results and results[0]:
                    for line in results[0]:
                        if line is None:
                            continue
                        if isinstance(line, (list, tuple)) and len(line) >= 2:
                            bbox = line[0]
                            if isinstance(line[1], (list, tuple)) and len(line[1]) >= 2:
                                text, conf = line[1][0], line[1][1]
                            else:
                                text, conf = str(line[1]), 0.9
                            text = str(text).strip()
                            if text:
                                # Extract line crop slice for pixel-grounded projection
                                line_slice = None
                                try:
                                    xs = [p[0] for p in bbox]
                                    ys = [p[1] for p in bbox]
                                    x1, y1, x2, y2 = int(max(0, min(xs))), int(max(0, min(ys))), int(min(crop.shape[1], max(xs))), int(min(crop.shape[0], max(ys)))
                                    if x2 > x1 and y2 > y1:
                                        line_slice = crop[y1:y2, x1:x2]
                                except Exception:
                                    pass

                                words = decompose_line_into_words(text, bbox, float(conf), line_image=line_slice)
                                text_blocks.append(TextBlock(
                                    text=text,
                                    confidence=float(conf),
                                    bbox=bbox,
                                    region_label=region_label,
                                    words=words,
                                ))
            except Exception as e:
                logger.warning(f"[OCR WARNING] PaddleOCR runtime extraction error: {e}. Falling back to EasyOCR...")
                self._ocr = None
                self.engine_name = "EasyOCR"

        if not text_blocks:
            if self._easyocr is None:
                try:
                    import easyocr
                    self._easyocr = easyocr.Reader(self.languages, gpu=self.use_gpu, verbose=False)
                    self.engine_name = "EasyOCR"
                except Exception as ex:
                    logger.error(f"[OCR CRITICAL] EasyOCR init error: {ex}")

            if self._easyocr is not None:
                try:
                    results = self._easyocr.readtext(crop)
                    for line in results:
                        if line is None:
                            continue
                        bbox, text, conf = line
                        text = str(text).strip()
                        if text:
                            # Extract line slice
                            line_slice = None
                            try:
                                xs = [p[0] for p in bbox]
                                ys = [p[1] for p in bbox]
                                x1, y1, x2, y2 = int(max(0, min(xs))), int(max(0, min(ys))), int(min(crop.shape[1], max(xs))), int(min(crop.shape[0], max(ys)))
                                if x2 > x1 and y2 > y1:
                                    line_slice = crop[y1:y2, x1:x2]
                            except Exception:
                                pass

                            words = decompose_line_into_words(text, bbox, float(conf), line_image=line_slice)
                            text_blocks.append(TextBlock(
                                text=text,
                                confidence=float(conf),
                                bbox=bbox,
                                region_label=region_label,
                                words=words,
                            ))
                except Exception as e:
                    logger.error(f"[OCR CRITICAL] EasyOCR extraction error: {e}")

        full_text = "\n".join(b.text for b in text_blocks)
        avg_conf = (
            sum(b.confidence for b in text_blocks) / len(text_blocks)
            if text_blocks else 0.0
        )

        return OCRResult(
            region_label=region_label,
            text_blocks=text_blocks,
            full_text=full_text,
            avg_confidence=avg_conf,
            engine=self.engine_name.lower(),
        )

    def extract_all_regions(
        self,
        regions: list,  # list of DetectedRegion
    ) -> dict[str, OCRResult]:
        """
        Run OCR on all detected regions.
        Returns dict mapping region_label → OCRResult.
        """
        results = {}
        for region in regions:
            logger.debug(f"OCR on region: {region.label}")
            result = self.extract_region(region.crop, region.label)
            results[region.label] = result
            logger.debug(
                f"  {region.label}: {len(result.text_blocks)} blocks, "
                f"avg_conf={result.avg_confidence:.3f}"
            )
        return results

    def extract_full_page(self, image: np.ndarray | Image.Image) -> OCRResult:
        """
        Run OCR on the full page (used as fallback or for simple invoices).
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        return self.extract_region(image, region_label="full_page")

    def extract_from_image(self, image: np.ndarray | Image.Image) -> OCRResult:
        """Alias for full page image extraction."""
        return self.extract_full_page(image)


# Backward compatibility alias
OCRExtractor = InvoiceOCR

