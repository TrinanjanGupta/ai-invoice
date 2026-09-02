"""
tests/test_handwriting_benchmark.py

Unit tests for handwriting CER/WER metrics and extraction benchmark reporting.
"""

import pytest
from benchmarks.benchmark_handwriting import (
    HandwritingBenchmark,
    compute_cer,
    compute_wer,
    HandwritingBenchmarkReport,
)


def test_cer_and_wer_computation():
    # Exact match
    assert compute_cer("1560.50", "1560.50") == 0.0
    assert compute_wer("Shree Ganesh", "Shree Ganesh") == 0.0

    # Minor optical substitution ('O' for '0')
    cer = compute_cer("1560.50", "156O.50")
    assert 0.0 < cer < 0.20

    # Line metric evaluation
    bm = HandwritingBenchmark()
    gt_lines = ["1560.50", "27AABCU9603R1ZN", "INV/2026/04"]
    pred_lines = ["1560.50", "27AABCU9603R1ZN", "INV/2026/04"]
    avg_cer, avg_wer = bm.evaluate_line_predictions(gt_lines, pred_lines)
    assert avg_cer == 0.0
    assert avg_wer == 0.0


def test_field_extraction_benchmark_evaluation():
    bm = HandwritingBenchmark()
    gt_invoices = [
        {"invoice_number": "INV-001", "grand_total": 1180.0, "subtotal": 1000.0, "tax_amount": 180.0, "vendor_gstin": "27AABCU9603R1ZN"}
    ]
    pred_invoices = [
        {"invoice_number": "INV-001", "grand_total": 1180.0, "subtotal": 1000.0, "tax_amount": 180.0, "vendor_gstin": "27AABCU9603R1ZN"}
    ]

    report = bm.evaluate_field_extractions(gt_invoices, pred_invoices)
    assert isinstance(report, HandwritingBenchmarkReport)
    assert report.total_samples == 1
    assert report.field_accuracy["invoice_number"] == 1.0
    assert report.arithmetic_valid_rate == 1.0
