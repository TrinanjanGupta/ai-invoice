"""
tests/test_tie_multi_page.py

Unit tests verifying TIE multi-page extraction correctness and page-specific isolation.
"""

import pytest
from preprocessing.document_profile import DocumentProfile, WordToken, RegionBlock
from understanding.template_extractor import TemplateExtractor


def test_find_words_in_box_multi_page_isolation():
    words = [
        WordToken(text="TOTAL", bbox_norm=[100, 500, 200, 530], bbox_raw=[10, 50, 20, 53], confidence=0.99, page=1),
        WordToken(text="1500.00", bbox_norm=[250, 500, 350, 530], bbox_raw=[25, 50, 35, 53], confidence=0.99, page=1),
        WordToken(text="TOTAL", bbox_norm=[100, 500, 200, 530], bbox_raw=[10, 50, 20, 53], confidence=0.99, page=2),
        WordToken(text="9999.00", bbox_norm=[250, 500, 350, 530], bbox_raw=[25, 50, 35, 53], confidence=0.99, page=2),
    ]

    profile = DocumentProfile(
        words=words,
        regions=[],
        page_count=2,
        width=1000,
        height=1400,
        aspect_ratio=1.4,
    )

    # Search on Page 1 explicitly
    p1_words = profile.find_words_in_box([100, 500, 400, 540], page=1)
    assert len(p1_words) == 2
    assert [w.text for w in p1_words] == ["TOTAL", "1500.00"]

    # Search on Page 2 explicitly
    p2_words = profile.find_words_in_box([100, 500, 400, 540], page=2)
    assert len(p2_words) == 2
    assert [w.text for w in p2_words] == ["TOTAL", "9999.00"]

    # Search across all pages (page=None)
    all_words = profile.find_words_in_box([100, 500, 400, 540], page=None)
    assert len(all_words) == 4


def test_template_extractor_page2_anchor_relative():
    # Anchor on Page 2 must extract target value from Page 2, not Page 1
    words = [
        WordToken(text="Tax", bbox_norm=[100, 100, 150, 120], bbox_raw=[10, 10, 15, 12], confidence=0.99, page=1),
        WordToken(text="Invoice", bbox_norm=[160, 100, 230, 120], bbox_raw=[16, 10, 23, 12], confidence=0.99, page=1),
        WordToken(text="Grand", bbox_norm=[100, 800, 160, 820], bbox_raw=[10, 80, 16, 82], confidence=0.99, page=2),
        WordToken(text="Total", bbox_norm=[170, 800, 230, 820], bbox_raw=[17, 80, 23, 82], confidence=0.99, page=2),
        WordToken(text="7500.00", bbox_norm=[250, 800, 350, 820], bbox_raw=[25, 80, 35, 82], confidence=0.99, page=2),
    ]

    profile = DocumentProfile(
        words=words,
        regions=[],
        page_count=2,
        width=1000,
        height=1400,
        aspect_ratio=1.4,
    )

    extractor = TemplateExtractor()
    field_rules = {
        "grand_total": {
            "strategy": "anchor_relative",
            "anchors": ["grand total"],
            "relative_box": [0, -5, 300, 10],
            "parser_spec": {},
            "confidence_score": 0.98,
        }
    }

    res = extractor.extract(profile, field_rules, template_version_id="ver_001")
    assert res.grand_total is not None
    assert res.grand_total.value == "7500.00"
    assert res.grand_total.confidence >= 0.90
