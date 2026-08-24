"""
active_learning/correction_tracker.py

Tracks human reviewer corrections against initial AI model predictions.
Computes field-level diffs, classifies review quality, and stores training examples.

Review Classifications:
- AUTO_ACCEPTED: Confidence >= 0.85, arithmetic passes, no human intervention needed.
- HUMAN_CONFIRMED: Human reviewer verified and accepted all AI predictions without edits.
- HUMAN_CORRECTED: Human reviewer fixed 1 or more fields (HIGHEST PRIORITY FOR ACTIVE LEARNING RETRAINING).
- HUMAN_REJECTED: Document was corrupted or not an invoice.
"""

from datetime import datetime
from typing import Any, Optional
from loguru import logger


def compute_field_corrections(ai_data: dict, human_data: dict) -> list[dict[str, Any]]:
    """
    Compares initial AI output against human-verified output to extract exact field diffs.
    """
    corrections = []
    keys = set(list(ai_data.keys()) + list(human_data.keys()))

    ignore_keys = {
        "needs_review", "review_reasons", "overall_confidence",
        "field_confidences", "fields_needing_review", "auto_accepted_fields",
        "created_at", "updated_at", "columns", "certified_remarks",
    }

    for k in keys:
        if k in ignore_keys:
            continue

        ai_val = ai_data.get(k)
        human_val = human_data.get(k)

        # Handle line items list
        if k == "line_items":
            ai_items = ai_val or []
            human_items = human_val or []
            if len(ai_items) != len(human_items) or ai_items != human_items:
                corrections.append({
                    "field": "line_items",
                    "ai_val": f"{len(ai_items)} items",
                    "human_val": f"{len(human_items)} items",
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": "table_correction",
                })
            continue

        # Clean comparison
        ai_clean = str(ai_val).strip() if ai_val is not None else ""
        human_clean = str(human_val).strip() if human_val is not None else ""

        if ai_clean != human_clean:
            corrections.append({
                "field": k,
                "ai_val": ai_val,
                "human_val": human_val,
                "timestamp": datetime.utcnow().isoformat(),
                "type": "field_correction",
            })

    return corrections


def classify_review_status(corrections: list[dict], needs_review: bool, overall_confidence: float) -> str:
    """
    Determines active learning sample category.
    """
    if len(corrections) > 0:
        return "human_corrected"
    elif not needs_review and overall_confidence >= 0.85:
        return "auto_accepted"
    else:
        return "human_confirmed"
