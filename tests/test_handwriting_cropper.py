"""
tests/test_handwriting_cropper.py

Unit tests for FieldCrop generation, YOLO region cropping, TIE anchor cropping,
margin expansion, and multi-line text line slicing.
"""

import cv2
import numpy as np
import pytest
from detection.detector import DetectedRegion
from preprocessing.document_profile import DocumentProfile, WordToken
from preprocessing.handwriting_cropper import (
    FieldCrop,
    expand_crop_margin,
    crop_from_yolo_regions,
    crop_from_tie_anchors,
    crop_text_lines,
)


def test_expand_crop_margin():
    img_w, img_h = 1000, 1000
    bbox = [100, 100, 200, 200]
    exp = expand_crop_margin(bbox, img_w, img_h, margin_pct=0.10)
    # bw = 100, bh = 100 -> dx = 10, dy = 10 -> [90, 90, 210, 210]
    assert exp == [90, 90, 210, 210]

    # Boundary clipping
    edge_bbox = [0, 0, 50, 50]
    edge_exp = expand_crop_margin(edge_bbox, img_w, img_h, margin_pct=0.10)
    assert edge_exp[0] == 0 and edge_exp[1] == 0


def test_crop_from_yolo_regions():
    img = np.ones((800, 600, 3), dtype=np.uint8) * 255
    cv2.putText(img, "Total: 1560.00", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    region = DetectedRegion(
        label="totals_block",
        class_id=4,
        confidence=0.92,
        bbox=(80, 160, 400, 240),
        crop=img[160:240, 80:400],
        page=1,
        is_handwritten=True,
    )

    crops = crop_from_yolo_regions(img, [region])
    assert len(crops) == 1
    assert crops[0].field_name == "totals_block"
    assert crops[0].crop_image.shape[0] > 0 and crops[0].crop_image.shape[1] > 0
    assert crops[0].page == 1


def test_crop_from_tie_anchors():
    img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
    token = WordToken(
        text="Invoice No:",
        bbox_norm=[100, 100, 250, 140],  # 0-1000 norm coords
        bbox_raw=[100.0, 100.0, 250.0, 140.0],
        confidence=0.99,
        page=1,
    )
    profile = DocumentProfile(
        page_count=1,
        width=1000,
        height=1000,
        aspect_ratio=1.0,
        words=[token],
        regions=[],
    )
    field_rules = [
        {
            "target_field": "invoice_number",
            "strategy": "anchor_relative",
            "anchor_pattern": "invoice no",
            "offset_box": [160, -5, 400, 45],
        }
    ]

    crops = crop_from_tie_anchors(img, profile, field_rules, page=1)
    assert len(crops) == 1
    assert crops[0].field_name == "invoice_number"
    assert crops[0].crop_source == "tie_anchor"


def test_crop_text_lines():
    # Multi-line image: 3 distinct horizontal lines of text
    img = np.ones((300, 400, 3), dtype=np.uint8) * 255
    cv2.putText(img, "Line One ABC", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Line Two 123", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Line Three XYZ", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    lines = crop_text_lines(img)
    assert len(lines) >= 3
    for lc in lines:
        assert lc.shape[0] >= 8
