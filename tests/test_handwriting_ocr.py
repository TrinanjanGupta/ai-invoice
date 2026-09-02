"""
tests/test_handwriting_ocr.py

Unit tests for lazy-loading HandwritingOCR engine and line recognition interface.
"""

import numpy as np
import pytest
from preprocessing.handwriting_cropper import FieldCrop
from ocr.handwriting_ocr import HandwritingOCR


def test_handwriting_ocr_lazy_initialization():
    ocr = HandwritingOCR(lazy_load=True)
    assert ocr._is_loaded is False
    assert ocr.processor is None


def test_handwriting_ocr_empty_crop_handling():
    ocr = HandwritingOCR(lazy_load=True)
    empty_crop = np.zeros((0, 0, 3), dtype=np.uint8)
    text, conf = ocr.recognize_line(empty_crop)
    assert text == ""
    assert conf == 0.0


def test_handwriting_ocr_field_crop_interface():
    ocr = HandwritingOCR(lazy_load=True)
    dummy_crop = np.ones((40, 200, 3), dtype=np.uint8) * 255
    fc = FieldCrop(
        field_name="remarks",
        crop_image=dummy_crop,
        enhanced_crop=dummy_crop,
        bbox_page=[10, 10, 210, 50],
        label_text="Remarks",
        crop_source="yolo_region",
        page=1,
    )
    results = ocr.recognize_field(fc)
    assert isinstance(results, list)
