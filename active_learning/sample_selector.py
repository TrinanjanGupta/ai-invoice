"""
active_learning/sample_selector.py

Active Learning sample prioritization and review queue engine.
Scores each unprocessed/pending invoice for 'Informativeness' so human reviewers
focus exclusively on high-impact uncertain cases rather than repetitive 99% confident templates.
"""

from typing import Any, Optional
from loguru import logger


def calculate_informativeness_score(record_data: dict) -> float:
    """
    Computes an informativeness score [0, 100] for an invoice.
    Higher score = more valuable for human review and active learning.
    """
    score = 0.0
    field_confidences = record_data.get("field_confidences") or {}
    fields_needing_review = record_data.get("fields_needing_review") or []
    review_reasons = record_data.get("review_reasons") or []
    overall_conf = record_data.get("overall_confidence", 0.90)

    # 1. Low confidence penalty (+30 per uncertain field)
    score += len(fields_needing_review) * 20.0

    # 2. Arithmetic mismatch (+40 points - very high learning value)
    if any("mismatch" in r.lower() or "total" in r.lower() for r in review_reasons):
        score += 40.0

    # 3. Overall uncertainty boost
    if overall_conf < 0.60:
        score += 30.0
    elif overall_conf < 0.80:
        score += 15.0

    # 4. Critical entity fields missing (Vendor GSTIN, Grand Total)
    if not record_data.get("vendor_name"):
        score += 15.0
    if not record_data.get("grand_total"):
        score += 25.0

    return min(100.0, score)


def prioritize_review_queue(invoices: list[dict]) -> list[dict]:
    """
    Ranks pending invoices by active learning informativeness.
    """
    scored = []
    for inv in invoices:
        score = calculate_informativeness_score(inv)
        scored.append({
            **inv,
            "active_learning_score": score,
            "priority": "HIGH" if score >= 50 else "MEDIUM" if score >= 20 else "LOW",
        })

    # Sort descending by active learning score
    return sorted(scored, key=lambda x: x["active_learning_score"], reverse=True)
