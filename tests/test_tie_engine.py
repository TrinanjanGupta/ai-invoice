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
    def test_anchor_jaccard_and_spatial_metrics(self):
        # 1. Jaccard metric
        set_a = {"invoice_no", "invoice_date", "subtotal", "grand_total"}
        set_b = {"invoice_no", "invoice_date", "gstin", "grand_total"}
        jaccard = TemplateRetriever.compute_anchor_jaccard(set_a, set_b)
        assert abs(jaccard - (3.0 / 5.0)) < 1e-4

        assert TemplateRetriever.compute_anchor_jaccard(set(), set()) == 1.0
        assert TemplateRetriever.compute_anchor_jaccard(set_a, set()) == 0.0

        # 2. Spatial alignment metric
        doc_pos = {"invoice_no": (100.0, 100.0), "grand_total": (800.0, 900.0)}
        tpl_pos_exact = {"invoice_no": (100.0, 100.0), "grand_total": (800.0, 900.0)}
        tpl_pos_offset = {"invoice_no": (110.0, 100.0), "grand_total": (800.0, 910.0)}

        score_exact = TemplateRetriever.compute_spatial_alignment(doc_pos, tpl_pos_exact, {"invoice_no", "grand_total"})
        score_offset = TemplateRetriever.compute_spatial_alignment(doc_pos, tpl_pos_offset, {"invoice_no", "grand_total"})
        assert score_exact == 1.0
        assert score_offset > 0.95

    def test_multi_stage_indexed_retrieval(self):
        retriever = TemplateRetriever()
        tpl = CachedTemplateVersion(
            version_id="ver_abc_1",
            family_id="fam_abc",
            version_num=1,
            exact_fingerprint="fp_exact_123",
            family_fingerprint="fam_fp_123",
            aspect_bucket=14,
            aspect_ratio=1.41,
            page_count=1,
            vendor_gstin="27AAAAA0000A1Z5",
            anchor_set={"invoice_no", "invoice_date", "grand_total", "subtotal"},
            anchor_positions={"invoice_no": (150.0, 100.0), "grand_total": (700.0, 800.0)},
            field_rules=[{"field_name": "invoice_number", "strategy": "anchor_relative", "anchors": ["invoice no"]}],
        )
        retriever.register_in_memory_template(tpl)

        # Verify inverted index has indexed the anchors
        assert "invoice_no" in retriever._anchor_inverted_index
        assert "ver_abc_1" in retriever._anchor_inverted_index["invoice_no"]
        assert "27AAAAA0000A1Z5" in retriever._gstin_index

        # 1. Exact hash match
        prof_exact = DocumentProfile(
            page_count=1,
            width=1000,
            height=1414,
            aspect_ratio=1.41,
            exact_fingerprint="fp_exact_123",
            layout_signature="fp_exact_123",
            vendor_gstin="27AAAAA0000A1Z5",
            words=[WordToken(text="invoice", bbox_norm=[100, 100, 150, 120], bbox_raw=[100, 100, 150, 120])],
        )
        res = retriever.retrieve(prof_exact)
        assert res.match_type == "exact_version"
        assert res.matched_version_id == "ver_abc_1"
        assert res.match_confidence >= 0.90

        # 2. Similarity match (Jaccard + Spatial Alignment)
        prof_sim = DocumentProfile(
            page_count=1,
            width=1000,
            height=1414,
            aspect_ratio=1.41,
            exact_fingerprint="fp_diff_999",
            vendor_gstin="27AAAAA0000A1Z5",
            anchor_set={"invoice_no", "invoice_date", "grand_total", "subtotal"},
            anchor_positions={"invoice_no": (152.0, 102.0), "grand_total": (698.0, 802.0)},
            words=[WordToken(text="invoice", bbox_norm=[100, 100, 150, 120], bbox_raw=[100, 100, 150, 120])],
        )
        res_sim = retriever.retrieve(prof_sim)
        assert res_sim.match_type == "exact_version"
        assert res_sim.matched_version_id == "ver_abc_1"
        assert res_sim.match_confidence >= 0.90

        # 3. Unknown layout
        prof_unknown = DocumentProfile(
            page_count=2,
            width=1000,
            height=1000,
            aspect_ratio=1.0,
            exact_fingerprint="completely_different",
            words=[],
        )
        res2 = retriever.retrieve(prof_unknown)
        assert res2.match_type == "none"


class TestMultiPageAndCandidateRanking:
    def test_multi_page_token_assignment(self):
        from ocr.extractor import OCRResult, TextBlock, OCRWord
        
        block_p1 = TextBlock(
            text="Invoice No: 123",
            bbox=[100, 100, 300, 130],
            confidence=0.99,
            region_label="full_page",
            words=[OCRWord(text="Invoice", bbox=[100, 100, 180, 130], confidence=0.99, page=1), OCRWord(text="123", bbox=[200, 100, 300, 130], confidence=0.99, page=1)],
        )
        block_p2 = TextBlock(
            text="Grand Total: 5000",
            bbox=[100, 800, 400, 830],
            confidence=0.99,
            region_label="full_page",
            words=[OCRWord(text="Grand", bbox=[100, 800, 180, 830], confidence=0.99, page=2), OCRWord(text="5000", bbox=[200, 800, 400, 830], confidence=0.99, page=2)],
        )

        ocr_map = {
            "full_page_p1": OCRResult(region_label="full_page", text_blocks=[block_p1], full_text="Invoice No: 123", avg_confidence=0.99),
            "full_page_p2": OCRResult(region_label="full_page", text_blocks=[block_p2], full_text="Grand Total: 5000", avg_confidence=0.99),
        }

        prof = DocumentProfile.from_ocr_and_regions(
            ocr_results=ocr_map,
            regions=[],
            width=1000,
            height=1414,
            page_count=2,
            page_dimensions={1: (1000, 1414), 2: (1200, 1600)},
        )

        p1_tokens = [w for w in prof.words if w.page == 1]
        p2_tokens = [w for w in prof.words if w.page == 2]
        assert len(p1_tokens) == 2
        assert len(p2_tokens) == 2
        assert prof.find_words_in_box([0, 0, 1000, 500], page=1)[0].text == "Invoice"
        assert len(prof.find_words_in_box([0, 0, 1000, 500], page=2)) == 0

    def test_vendor_and_buyer_gstin_candidate_ranking(self):
        words = [
            WordToken(text="Supplier", bbox_norm=[50, 100, 120, 120], bbox_raw=[50, 100, 120, 120], page=1),
            WordToken(text="GSTIN:", bbox_norm=[130, 100, 180, 120], bbox_raw=[130, 100, 180, 120], page=1),
            WordToken(text="19AAAAA0000A1Z5", bbox_norm=[190, 100, 350, 120], bbox_raw=[190, 100, 350, 120], page=1),
            WordToken(text="Bill", bbox_norm=[50, 400, 80, 420], bbox_raw=[50, 400, 80, 420], page=1),
            WordToken(text="To", bbox_norm=[85, 400, 110, 420], bbox_raw=[85, 400, 110, 420], page=1),
            WordToken(text="GSTIN:", bbox_norm=[120, 400, 170, 420], bbox_raw=[120, 400, 170, 420], page=1),
            WordToken(text="27BBBBB1111B1Z2", bbox_norm=[180, 400, 340, 420], bbox_raw=[180, 400, 340, 420], page=1),
        ]
        profile = DocumentProfile(page_count=1, width=1000, height=1414, aspect_ratio=1.41, words=words)
        extractor = TemplateExtractor()
        
        rules = [
            {"field_name": "vendor_gstin", "strategy": "regex_pattern", "parser_spec": {}},
            {"field_name": "buyer_gstin", "strategy": "regex_pattern", "parser_spec": {}},
        ]
        res = extractor.extract(profile, rules, match_type="exact_version")
        assert res.vendor_gstin is not None
        assert res.vendor_gstin.value == "19AAAAA0000A1Z5"
        assert res.buyer_gstin is not None
        assert res.buyer_gstin.value == "27BBBBB1111B1Z2"

    def test_idempotent_template_registration(self):
        retriever = TemplateRetriever()
        tpl = CachedTemplateVersion(
            version_id="ver_idempotent_1",
            family_id="fam_test",
            version_num=1,
            exact_fingerprint="fp_idem_1",
            family_fingerprint="fam_idem_1",
            aspect_bucket=14,
            aspect_ratio=1.41,
            page_count=1,
            vendor_gstin="27AAAAA0000A1Z5",
            anchor_set={"invoice_no", "date"},
            field_rules=[],
        )
        retriever.register_in_memory_template(tpl)
        retriever.register_in_memory_template(tpl)  # Register second time

        assert len([t for t in retriever._in_memory_index if t.version_id == "ver_idempotent_1"]) == 1
        assert len(retriever._gstin_index["27AAAAA0000A1Z5"]) == 1
        assert len(retriever._anchor_inverted_index["invoice_no"]) == 1

    def test_candidate_margin_ambiguity_penalty(self):
        words = [
            WordToken(text="Total:", bbox_norm=[100, 500, 150, 520], bbox_raw=[100, 500, 150, 520], page=1),
            WordToken(text="156.00", bbox_norm=[160, 500, 210, 520], bbox_raw=[160, 500, 210, 520], page=1, confidence=0.99),
            WordToken(text="165.00", bbox_norm=[162, 501, 212, 521], bbox_raw=[162, 501, 212, 521], page=1, confidence=0.98),
        ]
        profile = DocumentProfile(page_count=1, width=1000, height=1414, aspect_ratio=1.41, words=words)
        extractor = TemplateExtractor()
        rules = [{"field_name": "grand_total", "strategy": "semantic_numeric", "anchors": ["total:"]}]
        res = extractor.extract(profile, rules)
        assert res.grand_total is not None
        # Margin penalty must reduce confidence below 0.90 due to competing ambiguous numbers
        assert res.grand_total.confidence < 0.90

    def test_acceptance_gate_match_type_rigor(self):
        from validation.acceptance_gate import AutoAcceptanceGate
        from validation.validator import InvoiceSchema

        invoice = InvoiceSchema(
            invoice_number="INV-100",
            invoice_date="10/08/2026",
            grand_total=1000.0,
            overall_confidence=0.89,
            field_confidences={"grand_total": 0.91, "invoice_number": 0.91, "invoice_date": 0.91},
        )

        # 1. Exact match with 0.89 passes general gate (>= 0.88)
        accepted_exact, _ = AutoAcceptanceGate.evaluate(invoice, match_type="exact_version")
        assert accepted_exact is True

        # 2. Family anchor match with 0.89 is rejected because family requires >= 0.92
        accepted_family, reasons = AutoAcceptanceGate.evaluate(invoice, match_type="family_anchor")
        assert accepted_family is False
        assert any("Family" in r or "confidence" in r for r in reasons)

        # 3. Novel layout is rejected
        accepted_novel, _ = AutoAcceptanceGate.evaluate(invoice, match_type="none")
        assert accepted_novel is False

    def test_template_learner_anchor_value_pair_scoring(self):
        from understanding.template_learner import TemplateLearner
        learner = TemplateLearner()

        words = [
            # Anchor 1: Due Date with due date value
            WordToken(text="Due", bbox_norm=[100, 100, 130, 120], bbox_raw=[100, 100, 130, 120], page=1),
            WordToken(text="Date:", bbox_norm=[135, 100, 175, 120], bbox_raw=[135, 100, 175, 120], page=1),
            WordToken(text="30/08/2026", bbox_norm=[185, 100, 270, 120], bbox_raw=[185, 100, 270, 120], page=1),
            
            # Anchor 2: Invoice Date with invoice date value
            WordToken(text="Invoice", bbox_norm=[100, 300, 160, 320], bbox_raw=[100, 300, 160, 320], page=1),
            WordToken(text="Date:", bbox_norm=[165, 300, 210, 320], bbox_raw=[165, 300, 210, 320], page=1),
            WordToken(text="15/08/2026", bbox_norm=[220, 300, 305, 320], bbox_raw=[220, 300, 305, 320], page=1),
        ]
        profile = DocumentProfile(page_count=1, width=1000, height=1414, aspect_ratio=1.41, words=words)
        
        rules = learner.synthesize_field_rules(profile, {"invoice_date": "15/08/2026"})
        assert len(rules) == 1
        rule = rules[0]
        assert rule["field_name"] == "invoice_date"
        # Relative box must originate from Invoice Date (y ~ 300), not Due Date (y ~ 100)
        assert rule["relative_box"] is not None

    def test_template_retriever_match_margin_ambiguity(self):
        retriever = TemplateRetriever()
        tpl1 = CachedTemplateVersion(
            version_id="ver_tpl_1",
            family_id="fam_1",
            version_num=1,
            exact_fingerprint="fp_1",
            family_fingerprint="fam_fp_1",
            aspect_bucket=14,
            aspect_ratio=1.41,
            page_count=1,
            anchor_set={"invoice_no", "invoice_date", "total"},
            anchor_positions={"invoice_no": (150.0, 100.0), "total": (700.0, 800.0)},
            field_rules=[],
        )
        tpl2 = CachedTemplateVersion(
            version_id="ver_tpl_2",
            family_id="fam_2",
            version_num=1,
            exact_fingerprint="fp_2",
            family_fingerprint="fam_fp_2",
            aspect_bucket=14,
            aspect_ratio=1.41,
            page_count=1,
            anchor_set={"invoice_no", "invoice_date", "total"},
            anchor_positions={"invoice_no": (151.0, 101.0), "total": (701.0, 801.0)},
            field_rules=[],
        )
        retriever.register_in_memory_template(tpl1)
        retriever.register_in_memory_template(tpl2)

        prof = DocumentProfile(
            page_count=1,
            width=1000,
            height=1414,
            aspect_ratio=1.41,
            exact_fingerprint="fp_different_unseen",
            anchor_set={"invoice_no", "invoice_date", "total"},
            anchor_positions={"invoice_no": (150.5, 100.5), "total": (700.5, 800.5)},
            words=[],
        )
        res = retriever.retrieve(prof)
        # Margin is < 0.05, so retriever must flag ambiguity and degrade match_type to family_anchor
        assert res.match_type == "family_anchor"


class TestGoldenSuite:
    def test_golden_suite_execution(self):
        suite = GoldenInvoiceSuite()
        res = suite.run_suite()
        assert res["total_cases"] >= 2
        assert res["passed_cases"] == res["total_cases"]
        assert res["overall_accuracy"] == 1.0
        assert res["field_accuracies"]["grand_total"] == 1.0


