"""
tests/test_candidate_generator.py

Unit tests for multi-candidate generation and ranking across OCR streams.
"""

import numpy as np
import pytest
from preprocessing.handwriting_cropper import FieldCrop
from ocr.numeric_recognizer import NumericRecognizer
from ocr.candidate_generator import CandidateGenerator, OCRCandidate


def test_candidate_generator_from_ocr_and_numeric():
    gen = CandidateGenerator()
    dummy_crop = np.ones((50, 200, 3), dtype=np.uint8) * 255
    fc = FieldCrop(
        field_name="grand_total",
        crop_image=dummy_crop,
        enhanced_crop=dummy_crop,
        bbox_page=[100, 100, 300, 150],
        label_text="Grand Total",
        crop_source="yolo_region",
        page=1,
    )

    # When raw PaddleOCR text contains optical error "1,56O.50"
    candidates = gen.generate_candidates(
        field_crop=fc,
        handwriting_ocr=None,
        raw_ocr_text="1,56O.50",
        raw_ocr_confidence=0.85,
    )

    assert len(candidates) >= 1
    texts = [c.text for c in candidates]
    # Check that corrected numeric candidate "1560.50" is generated
    assert "1560.50" in texts
    sources = [c.source for c in candidates]
    assert "paddleocr" in sources or "char_correction" in sources or "numeric_specializer" in sources


def test_candidate_generator_gstin_ranking():
    gen = CandidateGenerator()
    dummy_crop = np.ones((50, 300, 3), dtype=np.uint8) * 255
    fc = FieldCrop(
        field_name="vendor_gstin",
        crop_image=dummy_crop,
        enhanced_crop=dummy_crop,
        bbox_page=[100, 100, 400, 150],
        label_text="GSTIN",
        crop_source="yolo_region",
        page=1,
    )

    candidates = gen.generate_candidates(
        field_crop=fc,
        handwriting_ocr=None,
        raw_ocr_text="27AABCU9603RIZN",  # 'I' instead of '1' in 13th pos
        raw_ocr_confidence=0.80,
    )

    assert len(candidates) >= 1
    # Check that corrected candidate with valid checksum exists
    corrected = [c.text for c in candidates if c.text == "27AABCU9603R1ZN"]
    assert len(corrected) == 1
