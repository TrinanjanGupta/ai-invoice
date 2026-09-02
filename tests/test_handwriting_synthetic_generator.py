"""
tests/test_handwriting_synthetic_generator.py

Unit tests for synthetic handwriting pad generation and annotation metadata.
"""

import numpy as np
import pytest
from data.handwriting.synthetic_generator import SyntheticHandwritingGenerator, SyntheticSample


def test_synthetic_pad_generation():
    gen = SyntheticHandwritingGenerator()
    sample = gen.generate_pad_sample(width=600, height=800)

    assert isinstance(sample, SyntheticSample)
    assert sample.image.shape == (800, 600, 3)
    assert sample.doc_type == "HANDWRITTEN_PAD"
    assert len(sample.yolo_boxes) >= 3

    labels = [b["label"] for b in sample.yolo_boxes]
    assert "handwriting" in labels or "totals_block" in labels
    assert len(sample.line_transcriptions) >= 1
    assert "grand_total" in [lt["field_name"] for lt in sample.line_transcriptions]
