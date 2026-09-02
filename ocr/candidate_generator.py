"""
ocr/candidate_generator.py

Multi-hypothesis candidate generation engine.
Combines recognition outputs across TrOCR, PaddleOCR, NumericRecognizer,
and Character Confusion mappings to produce ranked candidate hypotheses for each field crop.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Any
from loguru import logger

from preprocessing.handwriting_cropper import FieldCrop
from ocr.handwriting_ocr import HandwritingOCR
from ocr.numeric_recognizer import NumericRecognizer


@dataclass
class OCRCandidate:
    text: str
    confidence: float
    source: str           # "trocr" | "paddleocr" | "numeric_specializer" | "char_correction"
    field_name: str
    bbox: list[int]


class CandidateGenerator:
    """
    Generates and aggregates multiple recognition candidates per field crop.
    """

    def __init__(self, numeric_recognizer: Optional[NumericRecognizer] = None):
        self.numeric_recognizer = numeric_recognizer or NumericRecognizer()

    def generate_candidates(
        self,
        field_crop: FieldCrop,
        handwriting_ocr: Optional[HandwritingOCR] = None,
        raw_ocr_text: Optional[str] = None,
        raw_ocr_confidence: float = 0.80,
    ) -> list[OCRCandidate]:
        """
        Generates ranked candidates from all available recognition streams on a FieldCrop.
        """
        candidates: list[OCRCandidate] = []
        seen_texts: set[str] = set()

        # Stream 1: Primary PaddleOCR / spatial OCR text (if available)
        if raw_ocr_text and str(raw_ocr_text).strip():
            clean_text = str(raw_ocr_text).strip()
            candidates.append(OCRCandidate(
                text=clean_text,
                confidence=round(raw_ocr_confidence, 2),
                source="paddleocr",
                field_name=field_crop.field_name,
                bbox=field_crop.bbox_page,
            ))
            seen_texts.add(clean_text.lower())

        # Stream 2: Dedicated Handwriting OCR (TrOCR)
        if handwriting_ocr is not None:
            try:
                line_results = handwriting_ocr.recognize_field(field_crop)
                for txt, conf in line_results:
                    clean_txt = str(txt).strip()
                    if clean_txt and clean_txt.lower() not in seen_texts:
                        candidates.append(OCRCandidate(
                            text=clean_txt,
                            confidence=round(conf, 2),
                            source="trocr",
                            field_name=field_crop.field_name,
                            bbox=field_crop.bbox_page,
                        ))
                        seen_texts.add(clean_txt.lower())
            except Exception as e:
                logger.debug(f"Candidate generator TrOCR stream: {e}")

        # Stream 3: Numeric & Character Confusion Specialist
        # For each existing candidate text, generate constrained alternatives
        derived_candidates: list[OCRCandidate] = []
        for cand in candidates:
            num_cands = self.numeric_recognizer.recognize_numeric(
                cand.text,
                field_type=field_crop.field_name,
            )
            for n_txt, n_conf in num_cands:
                if n_txt.lower() not in seen_texts:
                    derived_candidates.append(OCRCandidate(
                        text=n_txt,
                        confidence=round(n_conf, 2),
                        source="char_correction" if n_txt != cand.text else "numeric_specializer",
                        field_name=field_crop.field_name,
                        bbox=field_crop.bbox_page,
                    ))
                    seen_texts.add(n_txt.lower())

        candidates.extend(derived_candidates)

        # Sort descending by confidence score
        return sorted(candidates, key=lambda c: c.confidence, reverse=True)
