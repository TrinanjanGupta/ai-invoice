"""
Stage 3b: PaddleOCR text extractor.

Extracts text from each detected region with bounding boxes and confidence.
Supports English, Hindi, and other Indic scripts out of the box.
"""

import numpy as np
from PIL import Image
from loguru import logger
from dataclasses import dataclass, field
from typing import Optional


import re

@dataclass
class OCRWord:
    text: str
    confidence: float
    bbox: list          # [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] or [x1, y1, x2, y2]


@dataclass
class TextBlock:
    text: str
    confidence: float
    bbox: list          # [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] (PaddleOCR format)
    region_label: str   # which invoice region this came from
    words: list[OCRWord] = field(default_factory=list)


def decompose_line_into_words(text: str, bbox: list, confidence: float) -> list[OCRWord]:
    """
    Decomposes an OCR text line into individual word tokens with
    character-proportional bounding boxes.
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

    total_len = max(1, len(text))
    words: list[OCRWord] = []

    for m in re.finditer(r"\S+", text):
        w_text = m.group()
        s_idx = m.start()
        e_idx = m.end()

        ratio_start = s_idx / total_len
        ratio_end = e_idx / total_len

        w_x1 = x_min + (x_max - x_min) * ratio_start
        w_x2 = x_min + (x_max - x_min) * ratio_end
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


class InvoiceOCR:
    """
    PaddleOCR wrapper optimised for invoice text extraction.
    Supports multi-language (English + Hindi by default).
    """

    def __init__(self, lang: str = "en", use_gpu: bool = False):
        self.lang = lang
        self.use_gpu = use_gpu
        self._ocr = None
        self._easyocr = None
        self._init_ocr()

    def _init_ocr(self):
        try:
            from paddleocr import PaddleOCR
            try:
                self._ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, use_gpu=self.use_gpu, enable_mkldnn=False)
            except Exception:
                self._ocr = PaddleOCR(lang=self.lang, enable_mkldnn=False)
            logger.info(f"PaddleOCR initialised (lang={self.lang}, gpu={self.use_gpu})")
        except (ImportError, Exception) as e:
            logger.warning(f"PaddleOCR initialisation failed/not installed ({e}). Trying EasyOCR fallback...")
            try:
                import easyocr
                self._easyocr = easyocr.Reader([self.lang], gpu=self.use_gpu, verbose=False)
                logger.info(f"EasyOCR initialised (lang={self.lang}, gpu={self.use_gpu}) as fallback")
            except ImportError:
                logger.error("Neither PaddleOCR nor EasyOCR is installed. Run: pip install easyocr")
                raise ImportError("Neither PaddleOCR nor EasyOCR is installed.") from e
            except Exception as ex:
                logger.error(f"EasyOCR init failed: {ex}")
                raise ex

    def extract_region(self, crop: np.ndarray, region_label: str) -> OCRResult:
        """
        Run OCR on a single region crop.
        Returns structured OCRResult with per-word confidence.
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
                                words = decompose_line_into_words(text, bbox, float(conf))
                                text_blocks.append(TextBlock(
                                    text=text,
                                    confidence=float(conf),
                                    bbox=bbox,
                                    region_label=region_label,
                                    words=words,
                                ))
            except Exception as e:
                logger.debug(f"PaddleOCR extraction fallback triggered: {e}")
                self._ocr = None

        if not text_blocks:
            if self._easyocr is None:
                try:
                    import easyocr
                    self._easyocr = easyocr.Reader([self.lang], gpu=self.use_gpu, verbose=False)
                except Exception as ex:
                    logger.debug(f"EasyOCR init error: {ex}")

            if self._easyocr is not None:
                try:
                    results = self._easyocr.readtext(crop)
                    for line in results:
                        if line is None:
                            continue
                        bbox, text, conf = line
                        text = str(text).strip()
                        if text:
                            words = decompose_line_into_words(text, bbox, float(conf))
                            text_blocks.append(TextBlock(
                                text=text,
                                confidence=float(conf),
                                bbox=bbox,
                                region_label=region_label,
                                words=words,
                            ))
                except Exception as e:
                    logger.debug(f"EasyOCR extraction error: {e}")

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

