import os
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_onednn"] = "0"
os.environ["PADDLE_DISABLE_PIR"] = "1"

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
    page: int = 1
    source: str = "paddleocr"
    block_id: Optional[int] = None
    line_id: Optional[int] = None

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
    and multi-gap clustering to find actual glyph ink boundaries and word whitespace
    valleys directly on real pixels.
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

    # Attempt Pixel-Grounded Image Projection Profiling with Multi-Gap Clustering
    if line_image is not None and line_image.size > 0 and len(words_in_text) > 1:
        try:
            import cv2
            gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY) if len(line_image.shape) == 3 else line_image
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            col_sums = np.sum(thresh, axis=0)

            # Find active ink column segments separated by whitespace gaps
            in_ink = False
            seg_start = 0
            raw_segments = []
            min_ink = max(1.0, np.max(col_sums) * 0.05) if col_sums.size > 0 else 1.0

            for col_idx, val in enumerate(col_sums):
                if val >= min_ink:
                    if not in_ink:
                        in_ink = True
                        seg_start = col_idx
                else:
                    if in_ink:
                        in_ink = False
                        raw_segments.append((seg_start, col_idx))
            if in_ink:
                raw_segments.append((seg_start, len(col_sums) - 1))

            # Direct 1-to-1 match
            if len(raw_segments) == len(words_in_text):
                words: list[OCRWord] = []
                for w_str, (s_px, e_px) in zip(words_in_text, raw_segments):
                    w_x1 = x_min + float(s_px)
                    w_x2 = max(w_x1 + 2.0, x_min + float(e_px))
                    w_y2 = max(y_max, y_min + 2.0)
                    word_bbox = [[w_x1, y_min], [w_x2, y_min], [w_x2, w_y2], [w_x1, w_y2]]
                    words.append(OCRWord(text=w_str, confidence=confidence, bbox=word_bbox))
                return words

            # Multi-gap clustering: merge sub-character segments by partitioning at (K-1) widest whitespace gaps
            elif len(raw_segments) > len(words_in_text) and len(words_in_text) > 1:
                gaps = []
                for i in range(len(raw_segments) - 1):
                    gap_size = raw_segments[i + 1][0] - raw_segments[i][1]
                    gaps.append((gap_size, i))

                num_breaks = len(words_in_text) - 1
                top_gap_indices = set(idx for _, idx in sorted(gaps, key=lambda x: x[0], reverse=True)[:num_breaks])

                grouped_segments = []
                cur_start = raw_segments[0][0]
                cur_end = raw_segments[0][1]
                for i in range(len(raw_segments) - 1):
                    if i in top_gap_indices:
                        grouped_segments.append((cur_start, max(cur_end, cur_start + 2)))
                        cur_start = raw_segments[i + 1][0]
                        cur_end = raw_segments[i + 1][1]
                    else:
                        cur_end = raw_segments[i + 1][1]
                grouped_segments.append((cur_start, max(cur_end, cur_start + 2)))

                if len(grouped_segments) == len(words_in_text):
                    words = []
                    for w_str, (s_px, e_px) in zip(words_in_text, grouped_segments):
                        w_x1 = x_min + float(s_px)
                        w_x2 = max(w_x1 + 2.0, x_min + float(e_px))
                        w_y2 = max(y_max, y_min + 2.0)
                        word_bbox = [[w_x1, y_min], [w_x2, y_min], [w_x2, w_y2], [w_x1, w_y2]]
                        words.append(OCRWord(text=w_str, confidence=confidence, bbox=word_bbox))
                    return words
        except Exception:
            pass

    # Typographical Cumulative Metric Projection Fallback
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
        w_x2 = max(w_x1 + 2.0, x_min + line_width * ratio_end)
        w_y1 = y_min
        w_y2 = max(w_y1 + 2.0, y_max)

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

    def __init__(self, languages: Any = None, lang: Optional[str] = None, use_gpu: bool = False, engine: Optional[str] = None):
        langs_input = languages or lang or "en,hi"
        if isinstance(langs_input, str):
            self.languages = [l.strip() for l in langs_input.split(",") if l.strip()]
        elif isinstance(langs_input, (list, tuple)):
            self.languages = list(langs_input)
        else:
            self.languages = ["en"]
        
        self.primary_lang = self.languages[0] if self.languages else "en"
        self.use_gpu = use_gpu
        self.preferred_engine = (engine or os.getenv("OCR_ENGINE", "paddleocr")).lower()
        self._ocr = None
        self._easyocr = None
        self.engine_name = "unknown"
        self._init_ocr()

    def _init_ocr(self):
        easyocr_langs = ["bn", "en"] if "bn" in self.languages else ["hi", "en"] if "hi" in self.languages else ["en"]
        if self.preferred_engine == "easyocr":
            try:
                import easyocr
                self._easyocr = easyocr.Reader(easyocr_langs, gpu=self.use_gpu, verbose=False)
                self.engine_name = "EasyOCR"
                logger.info(f"[OCR] Initialised ENGINE: EasyOCR (langs={easyocr_langs}, gpu={self.use_gpu})")
                return
            except Exception as e:
                logger.warning(f"[OCR WARNING] EasyOCR direct init failed ({e}). Trying PaddleOCR...")

        try:
            from paddleocr import PaddleOCR
            try:
                self._ocr = PaddleOCR(use_angle_cls=False, lang=self.primary_lang, use_gpu=self.use_gpu, enable_mkldnn=False)
            except Exception:
                self._ocr = PaddleOCR(lang=self.primary_lang)
            self.engine_name = "PaddleOCR"
            logger.info(f"[OCR] Initialised ENGINE: PaddleOCR (lang={self.primary_lang}, gpu={self.use_gpu})")
        except (ImportError, Exception) as e:
            logger.warning(f"[OCR WARNING] PaddleOCR initialization failed ({e}). Engaging EasyOCR fallback...")
            try:
                import easyocr
                self._easyocr = easyocr.Reader(easyocr_langs, gpu=self.use_gpu, verbose=False)
                self.engine_name = "EasyOCR"
                logger.info(f"[OCR] Initialised ENGINE: EasyOCR fallback (langs={easyocr_langs}, gpu={self.use_gpu})")
            except ImportError:
                logger.error("[OCR CRITICAL] Neither PaddleOCR nor EasyOCR is installed. Run: pip install easyocr")
                raise ImportError("Neither PaddleOCR nor EasyOCR is installed.") from e
            except Exception as ex:
                logger.error(f"[OCR CRITICAL] EasyOCR init failed: {ex}")
                raise ex

    def extract_region(
        self,
        crop: np.ndarray,
        region_label: str,
        bbox_offset: tuple[float, float] = (0.0, 0.0),
    ) -> OCRResult:
        """
        Run OCR on a single region crop.
        Translates all detected bounding boxes by bbox_offset to ensure
        unified Global Page Coordinates.
        """
        if crop is None or crop.size == 0:
            return OCRResult(
                region_label=region_label,
                text_blocks=[],
                full_text="",
                avg_confidence=0.0,
            )

        offset_x, offset_y = float(bbox_offset[0]), float(bbox_offset[1])
        text_blocks = []

        def _apply_offset_poly(poly):
            if len(poly) == 4 and isinstance(poly[0], (int, float)):
                return [poly[0] + offset_x, poly[1] + offset_y, poly[2] + offset_x, poly[3] + offset_y]
            return [[p[0] + offset_x, p[1] + offset_y] for p in poly]

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
                                # Translate words to global page coordinates
                                if offset_x != 0.0 or offset_y != 0.0:
                                    for w in words:
                                        w.bbox = _apply_offset_poly(w.bbox)

                                global_bbox = _apply_offset_poly(bbox) if (offset_x != 0.0 or offset_y != 0.0) else bbox
                                text_blocks.append(TextBlock(
                                    text=text,
                                    confidence=float(conf),
                                    bbox=global_bbox,
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
                    easyocr_langs = ["bn", "en"] if "bn" in self.languages else ["hi", "en"] if "hi" in self.languages else ["en"]
                    self._easyocr = easyocr.Reader(easyocr_langs, gpu=self.use_gpu, verbose=False)
                    self.engine_name = "EasyOCR"
                except Exception as ex:
                    logger.error(f"[OCR CRITICAL] EasyOCR init error: {ex}")

            if self._easyocr is not None:
                try:
                    h, w = crop.shape[:2]
                    scale = 1.0
                    if max(h, w) > 1600:
                        scale = 1600.0 / max(h, w)
                        import cv2
                        proc_crop = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                    else:
                        proc_crop = crop

                    results = self._easyocr.readtext(proc_crop, batch_size=8, detail=1, paragraph=False)
                    for line in results:
                        if line is None:
                            continue
                        bbox, text, conf = line
                        text = str(text).strip()
                        if text:
                            if scale != 1.0:
                                bbox = [[float(p[0] / scale), float(p[1] / scale)] for p in bbox]
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
                            # Translate words to global page coordinates
                            if offset_x != 0.0 or offset_y != 0.0:
                                for w in words:
                                    w.bbox = _apply_offset_poly(w.bbox)

                            global_bbox = _apply_offset_poly(bbox) if (offset_x != 0.0 or offset_y != 0.0) else bbox
                            text_blocks.append(TextBlock(
                                text=text,
                                confidence=float(conf),
                                bbox=global_bbox,
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
        Run OCR on all detected regions and map bounding boxes to global coordinates.
        Returns dict mapping region_label → OCRResult.
        """
        results = {}
        for region in regions:
            logger.debug(f"OCR on region: {region.label}")
            offset = (float(region.bbox[0]), float(region.bbox[1])) if region.bbox and len(region.bbox) >= 2 else (0.0, 0.0)
            result = self.extract_region(region.crop, region.label, bbox_offset=offset)
            results[region.label] = result
            logger.debug(
                f"  {region.label}: {len(result.text_blocks)} blocks, "
                f"avg_conf={result.avg_confidence:.3f}, offset={offset}"
            )
        return results

    def extract_full_page(self, image: np.ndarray | Image.Image) -> OCRResult:
        """
        Run OCR on the full page (used as fallback or for simple invoices).
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        return self.extract_region(image, region_label="full_page", bbox_offset=(0.0, 0.0))

    def extract_from_image(self, image: np.ndarray | Image.Image) -> OCRResult:
        """Alias for full page image extraction."""
        return self.extract_full_page(image)


# Backward compatibility alias
OCRExtractor = InvoiceOCR

