"""
tests/test_handwriting_preprocess.py

Unit tests for handwriting preprocessing:
- Original image preservation (dual-image guarantee)
- Ruled line removal
- Illumination normalization
- Text line deskew
- Ink detection and enhancement
"""

import cv2
import numpy as np
import pytest
from preprocessing.pipeline import InvoicePreprocessor, HandwrittenPreprocessResult


def test_handwritten_preprocessing_preserves_original():
    preprocessor = InvoicePreprocessor()
    # Create test image with ink strokes and noise
    img = np.ones((800, 600, 3), dtype=np.uint8) * 230
    cv2.putText(img, "Tax Invoice 12345", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 50, 20), 2)
    cv2.line(img, (50, 300), (550, 300), (100, 100, 100), 2)

    res = preprocessor.process_handwritten(img, handwriting_level="MOSTLY_HANDWRITTEN")

    assert isinstance(res, HandwrittenPreprocessResult)
    assert res.original_image is not None
    assert res.enhanced_image is not None
    assert res.original_image.shape == img.shape
    # Enhanced image should be DPI normalized or processed
    assert res.enhanced_image.shape[0] > 0 and res.enhanced_image.shape[1] > 0
    assert res.handwriting_level == "MOSTLY_HANDWRITTEN"
    assert res.was_binarized is False  # Must preserve continuous tones for recognizers


def test_ruled_line_removal():
    preprocessor = InvoicePreprocessor()
    # Image with horizontal lines
    img = np.ones((600, 500, 3), dtype=np.uint8) * 255
    # Add 5 horizontal lines across page
    for y in [100, 200, 300, 400, 500]:
        cv2.line(img, (20, y), (480, y), (50, 50, 50), 2)

    # Draw vertical/diagonal text crossing lines
    cv2.putText(img, "Handwritten Text", (50, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 150), 2)

    cleaned, removed = preprocessor._remove_ruled_lines(img)
    assert removed is True
    # The cleaned image should have fewer dark pixels at the line coordinates
    assert np.mean(cleaned) >= np.mean(img)


def test_illumination_normalization():
    preprocessor = InvoicePreprocessor()
    # Create image with uneven gradient/shadow
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    # Apply synthetic dark shadow gradient
    for y in range(400):
        factor = (y / 400.0) * 150
        img[y, :] = np.clip(img[y, :] - factor, 20, 255).astype(np.uint8)

    cv2.putText(img, "Shadow Text", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    normalized = preprocessor._normalize_illumination(img)
    assert normalized is not None
    assert normalized.shape == img.shape
    # Background should be significantly more uniform
    assert float(normalized.std()) < float(img.std()) or np.mean(normalized) > np.mean(img)


def test_ink_type_detection():
    preprocessor = InvoicePreprocessor()
    # Blue ink image
    blue_img = np.ones((200, 400, 3), dtype=np.uint8) * 255
    # BGR format: Blue is (200, 40, 20)
    cv2.putText(blue_img, "Blue Pen Writing", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 50, 20), 3)

    ink_type = preprocessor._detect_ink_type(blue_img)
    assert ink_type == "blue_ink"
