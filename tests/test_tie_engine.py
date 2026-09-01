"""
tests/test_tie_engine.py

Comprehensive unit tests for the TIE (Template Information Extraction) Engine:
- DocumentProfile creation, normalization, word token searching, signatures.
- TemplateRetriever multi-stage candidate pruning (deterministic, layout, anchor).
- TemplateExtractor strategies (anchor_relative, regex_pattern, semantic_numeric, spatial_table).
- GoldenInvoiceSuite regression runner.
"""

import pytest
from preprocessing.document_profile import DocumentProfile, WordToken, RegionBlock, normalize_box
from understanding.template_extractor import TemplateExtractor, clean_currency_str, clean_invoice_number
from understanding.template_retriever import TemplateRetriever, CachedTemplateVersion, TemplateMatchResult
from validation.golden_suite import GoldenInvoiceSuite


class TestDocumentProfile:
    def test_normalize_box(self):
        box = [100, 200, 300, 400]
        norm = normalize_box(box, width=1000, height=1000)
        assert norm == [100, 200, 300, 400]

        # Zero or inverted box correction
        bad_box = [200, 200, 200, 200]
        norm2 = normalize_box(bad_box, width=1000, height=1000)
        assert norm2[2] > norm2[0]
        assert norm2[3] > norm2[1]

    def test_profile_creation_and_search(self):
        words = [
            WordToken(text="INVOICE", bbox_norm=[100, 50, 200, 70], bbox_raw=[100, 50, 200, 70]),
            WordToken(text="NO:", bbox_norm=[210, 50, 250, 70], bbox_raw=[210, 50, 250, 70]),
            WordToken(text="INV-9901", bbox_norm=[260, 50, 360, 70], bbox_raw=[260, 50, 360, 70]),
            WordToken(text="GSTIN:", bbox_norm=[100, 80, 160, 95], bbox_raw=[100, 80, 160, 95]),
            WordToken(text="19AAAAA0000A1Z5", bbox_norm=[170, 80, 330, 95], bbox_raw=[170, 80, 330, 95]),
        ]
        profile = DocumentProfile(
            page_count=1,
            width=1000,
            height=1414,
            aspect_ratio=1.41,
            words=words,
        )

        assert len(profile.words) == 5
        assert "invoice" in profile.get_full_text().lower()

        # Find words in box
        found = profile.find_words_in_box([250, 40, 400, 80])
        assert len(found) == 1
        assert found[0].text == "INV-9901"

        # Find anchor tokens
        matches = profile.find_anchor_tokens("invoice no")
        assert len(matches) == 1
        assert len(matches[0]) == 2


class TestTemplateExtractor:
    def test_clean_currency_str(self):
        assert clean_currency_str("Rs. 1,560.50") == 1560.50
        assert clean_currency_str("₹ 420.00") == 420.00
        assert clean_currency_str("(100.00)") == -100.00
        assert clean_currency_str("invalid") is None

    def test_clean_invoice_number(self):
        assert clean_invoice_number("Invoice No: INV-4592") == "INV-4592"
        assert clean_invoice_number("Bill #: 992") == "992"

    def test_extraction_on_profile(self):
        words = [
            WordToken(text="Invoice", bbox_norm=[100, 50, 160, 70], bbox_raw=[100, 50, 160, 70]),
            WordToken(text="No:", bbox_norm=[165, 50, 195, 70], bbox_raw=[165, 50, 195, 70]),
            WordToken(text="WB-8812", bbox_norm=[205, 50, 290, 70], bbox_raw=[205, 50, 290, 70]),
            WordToken(text="Date:", bbox_norm=[100, 80, 140, 95], bbox_raw=[100, 80, 140, 95]),
            WordToken(text="14-08-2026", bbox_norm=[150, 80, 240, 95], bbox_raw=[150, 80, 240, 95]),
            WordToken(text="Grand", bbox_norm=[600, 300, 650, 320], bbox_raw=[600, 300, 650, 320]),
            WordToken(text="Total", bbox_norm=[660, 300, 710, 320], bbox_raw=[660, 300, 710, 320]),
            WordToken(text="Rs.", bbox_norm=[750, 300, 780, 320], bbox_raw=[750, 300, 780, 320]),
            WordToken(text="2,450.00", bbox_norm=[790, 300, 870, 320], bbox_raw=[790, 300, 870, 320]),
        ]
        profile = DocumentProfile(
            page_count=1,
            width=1000,
            height=1414,
            aspect_ratio=1.41,
            words=words,
        )

        extractor = TemplateExtractor()
        rules = [
            {
                "field_name": "invoice_number",
                "strategy": "anchor_relative",
                "anchors": ["invoice no"],
                "relative_box": [0, -5, 300, 10],
                "confidence_score": 0.95,
            },
            {
                "field_name": "invoice_date",
                "strategy": "anchor_relative",
                "anchors": ["date:"],
                "relative_box": [0, -5, 250, 10],
                "confidence_score": 0.95,
            },
            {
                "field_name": "grand_total",
                "strategy": "semantic_numeric",
                "anchors": ["grand total"],
                "confidence_score": 0.98,
            },
        ]

        extracted = extractor.extract(profile, rules)
        assert extracted.invoice_number is not None
        assert extracted.invoice_number.value == "WB-8812"
        assert extracted.invoice_date is not None
        assert extracted.invoice_date.value == "14-08-2026"
        assert extracted.grand_total is not None
        assert extracted.grand_total.value == "2450.00"
        assert extracted.overall_confidence >= 0.85


class TestTemplateRetriever:
    def test_multi_stage_retrieval(self):
        retriever = TemplateRetriever()
        tpl = CachedTemplateVersion(
            version_id="ver_abc_1",
            family_id="fam_abc",
            version_num=1,
            version_fingerprint="fp_exact_123",
            aspect_ratio=1.41,
            page_count=1,
            anchor_signature="invoice_no|date|grand_total",
            layout_signature="fp_exact_123",
            vendor_gstin="27AAAAA0000A1Z5",
            field_rules=[{"field_name": "invoice_number", "strategy": "anchor_relative", "anchors": ["invoice no"]}],
        )
        retriever.register_in_memory_template(tpl)

        # 1. Exact hash match
        prof_exact = DocumentProfile(
            page_count=1,
            width=1000,
            height=1414,
            aspect_ratio=1.41,
            layout_signature="fp_exact_123",
            anchor_signature="invoice_no|date|grand_total",
            vendor_gstin="27AAAAA0000A1Z5",
            words=[WordToken(text="invoice", bbox_norm=[100, 100, 150, 120], bbox_raw=[100, 100, 150, 120])],
        )
        res = retriever.retrieve(prof_exact)
        assert res.match_type == "exact_version"
        assert res.matched_version_id == "ver_abc_1"
        assert res.match_confidence >= 0.90

        # 2. Unknown layout
        prof_unknown = DocumentProfile(
            page_count=2,  # Different page count
            width=1000,
            height=1000,
            aspect_ratio=1.0,
            layout_signature="completely_different",
            words=[],
        )
        res2 = retriever.retrieve(prof_unknown)
        assert res2.match_type == "none"


class TestGoldenSuite:
    def test_golden_suite_execution(self):
        suite = GoldenInvoiceSuite()
        res = suite.run_suite()
        assert res["total_cases"] >= 2
        assert res["passed_cases"] == res["total_cases"]
        assert res["overall_accuracy"] == 1.0
        assert res["field_accuracies"]["grand_total"] == 1.0

